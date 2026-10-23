"""Wavelength-driven ESO Molecfit adapter, initial version.

Requires numpy, astropy and an EsoRex installation containing Molecfit recipes.
Does not change original extracted spectra. Inputs are assumed to be observed-frame,
VACUUM wavelengths (verify this for each instrument).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import re
import shutil
import subprocess
import numpy as np
from astropy.io import fits
from astropy.time import Time


@dataclass
class Segment:
    name: str
    wavelength: np.ndarray  # observed-frame vacuum microns
    flux: np.ndarray
    error: np.ndarray


@dataclass
class Observation:
    segments: list[Segment]
    header: fits.Header
    source: Path
    metadata: dict = field(default_factory=dict)

    def select(self, low: float, high: float) -> 'Observation':
        if not low < high:
            raise ValueError('Expected low < high')
        selected = []
        for s in self.segments:
            mask = ((s.wavelength >= low) & (s.wavelength <= high)
                    & np.isfinite(s.wavelength) & np.isfinite(s.flux)
                    & np.isfinite(s.error) & (s.error > 0))
            if mask.sum() >= 3:
                ind = np.argsort(s.wavelength[mask])
                selected.append(Segment(s.name, s.wavelength[mask][ind],
                                        s.flux[mask][ind], s.error[mask][ind]))
        if not selected:
            raise ValueError(f'No valid spectral segments in {low}–{high} um')
        return Observation(selected, self.header.copy(), self.source,
                           self.metadata.copy())


def load_crires(path: str | Path) -> Observation:
    """Read CR2RES obs_nodding extractedA/B FITS; preserve each chip/order."""
    path = Path(path).resolve()
    segments = []
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        for hdu in hdul[1:]:
            if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None:
                continue
            names = set(hdu.columns.names)
            for wname in sorted(n for n in names if n.endswith('_WL')):
                p = wname[:-3]
                if {p+'_SPEC', p+'_ERR'} - names:
                    continue
                wave = np.asarray(hdu.data[wname], dtype=float).ravel().copy()
                if np.isfinite(wave).any() and np.nanmedian(wave) > 100:
                    wave /= 1000.0
                flux = np.asarray(hdu.data[p+'_SPEC'], dtype=float).ravel().copy()
                err = np.asarray(hdu.data[p+'_ERR'], dtype=float).ravel().copy()
                segments.append(Segment(hdu.name+'_'+p, wave, flux, err))
    if not segments:
        raise ValueError(f'No extracted wavelength/flux/error columns in {path}')
    return Observation(segments, header, path)


def from_arrays(wavelength, flux, error, header, name='segment0', source='input.fits'):
    """For NIRSPEC/custom loaders. Call with OBSERVATORY-frame vacuum wavelengths in um."""
    return Observation([Segment(name, np.asarray(wavelength, float),
                                np.asarray(flux, float), np.asarray(error, float))],
                       header.copy(), Path(source))


def _keyword(hdr, *keys):
    for key in keys:
        value = hdr.get(key)
        if value is not None and str(value).strip():
            return value, key
    return None, None


def _metadata(obs: Observation, overrides: dict) -> dict:
    """Resolve required metadata, tracking origin. Missing meteo must be supplied."""
    h = obs.header
    definitions = {
        'mjd': ('MJD-OBS', 'ESO DET FRAM MJD'),
        'airmass_start': ('HIERARCH ESO TEL AIRM START', 'AIRMASS'),
        'airmass_end': ('HIERARCH ESO TEL AIRM END', 'AIRMASS'),
        'humidity': ('HIERARCH ESO TEL AMBI RHUM',),
        'pressure': ('HIERARCH ESO TEL AMBI PRES START', 'HIERARCH ESO TEL AMBI PRES END'),
        'temperature': ('HIERARCH ESO TEL AMBI TEMP',),
        'slit_width': ('HIERARCH ESO INS SLIT1 WID', 'HIERARCH ESO INS SLIT WIDTH'),
    }
    values = {}
    provenance = {}
    for key, candidates in definitions.items():
        if key in overrides:
            values[key] = float(overrides[key]); provenance[key] = 'explicit override'
        else:
            value, used = _keyword(h, *candidates)
            if value is not None:
                try:
                    values[key] = float(value); provenance[key] = used
                except (TypeError, ValueError):
                    pass
    if 'mjd' not in values:
        dt, src = _keyword(h, 'DATE-OBS')
        if dt:
            values['mjd'] = float(Time(str(dt), format='isot', scale='utc').mjd)
            provenance['mjd'] = 'DATE-OBS'
    if 'airmass_end' not in values and 'airmass_start' in values:
        values['airmass_end'] = values['airmass_start']
        provenance['airmass_end'] = 'airmass_start (fallback)'
    # CR2RES extracted headers may lack ambient meteo / slit width: require overrides.
    required = ['mjd', 'airmass_start', 'airmass_end', 'humidity',
                'pressure', 'temperature', 'slit_width']
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f'Missing metadata {missing}; supply metadata_overrides, '
                         'preferably copied from the matching raw ESO SCIENCE frame. '
                         'No atmospheric defaults are silently invented.')
    if 'observatory' in overrides:
        values['observatory'] = str(overrides['observatory'])
    else:
        values['observatory'] = 'paranal' if ('CRIRES' in str(h.get('INSTRUME', '')).upper() or 'CRIRES' in str(obs.source).upper()) else 'unknown'
    if values['observatory'].lower() != 'paranal':
        for fieldname in ('latitude', 'longitude', 'elevation'):
            if fieldname not in overrides:
                raise ValueError(f'{fieldname} required for non-Paranal observations')
    site = {'latitude': -24.627, 'longitude': -70.405, 'elevation': 2635.0}
    for key in site:
        values[key] = float(overrides.get(key, site[key]))
        provenance[key] = 'explicit override' if key in overrides else 'Paranal site constant'
    values['utc_seconds'] = float((values['mjd'] % 1) * 86400.0)
    values['provenance'] = provenance
    return values


def _set_rc(template: str, updates: dict, destination: Path) -> None:
    lines = template.splitlines(keepends=True)
    found = set()
    for i, line in enumerate(lines):
        if line.lstrip().startswith('#') or '=' not in line:
            continue
        key = line.split('=', 1)[0].strip()
        if key in updates:
            lines[i] = f'{key}={updates[key]}\n'
            found.add(key)
    absent = set(updates) - found
    if absent:
        raise ValueError(f'{destination.name}: recipe config missing keys {sorted(absent)}; '
                         'check installed Molecfit version/template')
    destination.write_text(''.join(lines))


def _recipe_defaults(esorex: str, recipe: str, output: Path):
    """Generate config for the installed recipe; output path is new per job."""
    result = subprocess.run([esorex, f'--create-config={output}', recipe],
                            capture_output=True, text=True)
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(f'Could not generate {recipe} config: '
                           f'{result.stderr}\n{result.stdout}')
    return output.read_text()


@dataclass
class MolecfitJob:
    directory: Path
    spectrum: Path
    settings: dict
    esorex: str = 'esorex'

    def _soflines(self, entries):
        return '\n'.join(f'{Path(path).resolve()} {tag}' for path,tag in entries)+'\n'

    def _run_recipe(self, recipe, config, sof, output):
        output.mkdir(parents=True, exist_ok=True)
        command = [self.esorex, f'--recipe-config={config}',
                   f'--output-dir={output}', recipe, str(sof)]
        # Stream the combined output to terminal AND a persistent log.
        import sys
        with (output / f'{recipe}.log').open('w') as log:
            with subprocess.Popen(command, cwd=output, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                status = proc.wait()
        if status != 0:
            raise RuntimeError(f'{recipe} failed (exit {status}); '
                               f'see {output / (recipe+".log")}')

    def run(self):
        model_dir = self.directory/'model'
        trans_dir = self.directory/'calctrans'
        corr_dir = self.directory/'correct'
        self._run_recipe('molecfit_model', self.directory/'model.rc',
                         self.directory/'model.sof', model_dir)
        model_tags = ['MODEL_MOLECULES','ATM_PARAMETERS','BEST_FIT_PARAMETERS']
        for tag in model_tags:
            if not (model_dir / f'{tag}.fits').exists():
                raise FileNotFoundError(f'Missing {model_dir / (tag+".fits")}')
        cal_sof = self.directory/'calctrans.sof'
        cal_sof.write_text(self._soflines(
            [(self.spectrum,'SCIENCE')]+[(model_dir/f'{tag}.fits', tag) for tag in model_tags]))
        self._run_recipe('molecfit_calctrans', self.directory/'calctrans.rc',
                         cal_sof, trans_dir)
        transmission = trans_dir/'TELLURIC_CORR.fits'
        if not transmission.exists():
            raise FileNotFoundError(f'No {transmission}; inspect calctrans products')
        corr_sof = self.directory/'correct.sof'
        corr_sof.write_text(self._soflines([(self.spectrum,'SCIENCE'),
                                           (transmission,'TELLURIC_CORR')]))
        self._run_recipe('molecfit_correct', self.directory/'correct.rc',
                         corr_sof, corr_dir)
        return corr_dir

    def inspect(self):
        return {'directory': str(self.directory), 'spectrum': str(self.spectrum),
                'settings': self.settings,
                'generated': [p.name for p in self.directory.iterdir()]}


def prepare(observation: Observation, *, wavelength_range: tuple[float,float],
            fit_regions: list[tuple[float,float]], output_dir: str | Path,
            metadata_overrides: Optional[dict] = None, molecules=('H2O',),
            model_template: Optional[str | Path] = None,
            calctrans_template: Optional[str | Path] = None,
            correct_template: Optional[str | Path] = None,
            esorex='esorex', continuum_degree=2,
            gaussian_fwhm_pix=3.5) -> MolecfitJob:
    """Create one multi-extension SCIENCE FITS and version-matched RC/SOF configs.

    Does not run Molecfit. Regions MUST lie inside a selected segment and
    contain enough good pixels. Source data are not changed or stitched.
    """
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is non-empty; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    # Preserve complete native pixel arrays for all segments intersecting the display range.
    # The fit regions, not the SCIENCE file, determine which pixels inform the model.
    low_bound, high_bound = wavelength_range
    if not low_bound < high_bound:
        raise ValueError('wavelength_range must be increasing')
    selected = [seg for seg in observation.segments if
                np.any(np.isfinite(seg.wavelength) &
                       (seg.wavelength >= low_bound) & (seg.wavelength <= high_bound))]
    if not selected:
        raise ValueError('No segment overlaps the requested wavelength range')
    obs = Observation(selected, observation.header.copy(), observation.source,
                      observation.metadata.copy())
    meta = _metadata(obs, metadata_overrides or {})
    if not fit_regions:
        raise ValueError('Set at least one validated telluric-fitting region')
    primary = fits.PrimaryHDU(header=obs.header.copy())
    primary.header.add_history('Molecfit bridge: segment selection; input source '+str(obs.source))
    hdu_list = [primary]
    wave_table = []
    segment_details = []
    for i, seg in enumerate(obs.segments, start=1):
        # Use native arrays and preserve original (possibly unnormalized) flux.
        columns = [fits.Column(name='wavelength', format='D', array=seg.wavelength),
                   fits.Column(name='spectrum', format='D', array=seg.flux),
                   fits.Column(name='error', format='D', array=seg.error)]
        hdu_list.append(fits.BinTableHDU.from_columns(columns, name=f'SEG{i:02d}'))
        segment_details.append({'chip': i, 'source_segment': seg.name,
                                'low': float(seg.wavelength.min()),
                                'high': float(seg.wavelength.max()),
                                'pixels': len(seg.wavelength)})
    for region in fit_regions:
        if len(region) == 3:
            low, high, seg_name = region
            matching = [i for i, seg in enumerate(obs.segments, 1) if seg.name == seg_name]
        elif len(region) == 2:
            low, high = region
            matching = [i for i, seg in enumerate(obs.segments, 1)
                        if np.isfinite(seg.wavelength).any()
                        and low >= np.nanmin(seg.wavelength)
                        and high <= np.nanmax(seg.wavelength)]
        else:
            raise ValueError(f'Expected (low,high[,segment_name]): {region}')
        if len(matching) != 1:
            raise ValueError(f'Region {region} maps to {matching}; provide segment name '
                             'to disambiguate overlapping CRIRES orders')
        chip = matching[0]
        _validate_region(obs.segments[chip - 1], low, high)
        wave_table.append((float(low), float(high), chip))
    science_file = out/'science.fits'
    fits.HDUList(hdu_list).writeto(science_file)
    include = fits.BinTableHDU.from_columns([
        fits.Column(name='LOWER_LIMIT', format='D', array=np.array([x[0] for x in wave_table])),
        fits.Column(name='UPPER_LIMIT', format='D', array=np.array([x[1] for x in wave_table])),
        fits.Column(name='MAPPED_TO_CHIP', format='J', array=np.array([x[2] for x in wave_table])),
        fits.Column(name='WLC_FIT_FLAG', format='J', array=np.zeros(len(wave_table),dtype=int)),
        fits.Column(name='CONT_FIT_FLAG', format='J', array=np.ones(len(wave_table),dtype=int)),
    ])
    wave_file=out/'wave_include.fits'
    fits.HDUList([fits.PrimaryHDU(),include]).writeto(wave_file)
    def template(kind, recipe):
        p={'model':model_template,'calctrans':calctrans_template,
           'correct':correct_template}[kind]
        return Path(p).read_text() if p is not None else _recipe_defaults(esorex, recipe, out/f'{kind}.rc')
    model=template('model','molecfit_model')
    calctrans=template('calctrans','molecfit_calctrans')
    correct=template('correct','molecfit_correct')
    mols=list(molecules)
    if not mols or len(set(mols)) != len(mols):
        raise ValueError('Provide unique molecule names')
    # Update only parameters known in the installed template.
    overrides = {
        'COLUMN_LAMBDA':'wavelength','COLUMN_FLUX':'spectrum','COLUMN_DFLUX':'error',
        'WLG_TO_MICRON':'1.0','WAVELENGTH_FRAME':'VAC',
        'USE_ONLY_INPUT_PRIMARY_DATA':'FALSE','CHIP_EXTENSIONS':'TRUE',
        'WAVE_INCLUDE':'NULL','WAVE_EXCLUDE':'NULL',
        'MAP_REGIONS_TO_CHIP':'NULL','FIT_WLC':'NULL',
        'FIT_CONTINUUM':'NULL', 'CONTINUUM_N':','.join([str(continuum_degree)]*len(wave_table)),
        'LIST_MOLEC':','.join(mols), 'FIT_MOLEC':','.join('1' for _ in mols),
        'REL_COL':','.join('1.0' for _ in mols),
        'FIT_TELESCOPE_BACKGROUND':'FALSE','FIT_RES_BOX':'FALSE',
        'FIT_RES_GAUSS':'TRUE','FIT_RES_LORENTZ':'FALSE',
        'RES_GAUSS':str(gaussian_fwhm_pix),
        'OBSERVING_DATE_KEYWORD':'NONE','OBSERVING_DATE_VALUE':str(meta['mjd']),
        'UTC_KEYWORD':'NONE','UTC_VALUE':str(meta['utc_seconds']),
        'TELESCOPE_ANGLE_KEYWORD':'NONE',
        'TELESCOPE_ANGLE_VALUE':str(np.degrees(np.arcsin(1/meta['airmass_start']))),
        'RELATIVE_HUMIDITY_KEYWORD':'NONE',
        'RELATIVE_HUMIDITY_VALUE':str(meta['humidity']),
        'PRESSURE_KEYWORD':'NONE','PRESSURE_VALUE':str(meta['pressure']),
        'TEMPERATURE_KEYWORD':'NONE','TEMPERATURE_VALUE':str(meta['temperature']),
        'SLIT_WIDTH_KEYWORD':'NONE','SLIT_WIDTH_VALUE':str(meta['slit_width']),
        'LATITUDE_KEYWORD':'NONE','LATITUDE_VALUE':str(meta['latitude']),
        'LONGITUDE_KEYWORD':'NONE','LONGITUDE_VALUE':str(meta['longitude']),
        'ELEVATION_KEYWORD':'NONE','ELEVATION_VALUE':str(meta['elevation']),
    }
    if meta['airmass_start'] < 1:
        raise ValueError('airmass_start must be >=1')
    _set_rc(model, overrides, out/'model.rc')
    (out/'calctrans.rc').write_text(calctrans)
    (out/'correct.rc').write_text(correct)
    (out/'model.sof').write_text(f'{science_file} SCIENCE\n{wave_file} WAVE_INCLUDE\n')
    manifest = {'source': str(obs.source), 'metadata': meta,
                'wavelength_range':list(wavelength_range),
                'fit_regions':[list(x) for x in fit_regions],
                'segments':segment_details, 'molecules':mols,
                'frame':'observatory vacuum', 'input_normalized': False,
                'note':'Fit windows are caller-supplied and must be validated.'}
    (out/'metadata.json').write_text(json.dumps(manifest, indent=2))
    return MolecfitJob(out, science_file, manifest, esorex)


# ===============================================================
# Interactive fit-region selection and command-line integration
# ===============================================================

def _valid_pixels(seg):
    return (np.isfinite(seg.wavelength) & np.isfinite(seg.flux) &
            np.isfinite(seg.error) & (seg.error > 0))


def suggested_regions(observation: Observation, edge_pixels: int = 100):
    """Conservative edge-trimmed initial guesses, NOT science-selected windows."""
    result = {}
    for i, seg in enumerate(observation.segments):
        idx = np.flatnonzero(_valid_pixels(seg))
        if len(idx) < 2 * edge_pixels + 10:
            result[i] = []
            continue
        w = seg.wavelength[idx[edge_pixels:len(idx)-edge_pixels or None]]
        result[i] = [(float(np.min(w)), float(np.max(w)))]
    return result


def _validate_region(seg, lo, hi, min_pixels=10):
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f'Invalid region: {lo}, {hi}')
    good = _valid_pixels(seg)
    wave = seg.wavelength[good]
    if not wave.size or lo < wave.min() or hi > wave.max():
        raise ValueError(f'Region {lo:.8f}-{hi:.8f} is outside valid coverage of {seg.name}')
    n = np.count_nonzero((wave >= lo) & (wave <= hi))
    if n < min_pixels:
        raise ValueError(f'Region {lo:.8f}-{hi:.8f} has only {n} valid pixels in {seg.name}')


def choose_regions(observation: Observation, *, edge_pixels=100,
                   initial=None, save_json=None):
    """GUI selector; arrows change segments, drag adds, d undo, c clear, s save/close."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector, Button
    if not observation.segments:
        raise ValueError('No segments to display')
    regions = {i: list((initial or suggested_regions(observation, edge_pixels)).get(i, []))
               for i in range(len(observation.segments))}
    state = {'i': 0, 'saved': False}
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(bottom=.20, top=.87)

    def draw():
        ax.clear()
        i = state['i']
        seg = observation.segments[i]
        good = _valid_pixels(seg)
        ax.plot(seg.wavelength[good], seg.flux[good], 'k-', lw=.65)
        ax.plot(seg.wavelength[~good & np.isfinite(seg.wavelength) & np.isfinite(seg.flux)],
                seg.flux[~good & np.isfinite(seg.wavelength) & np.isfinite(seg.flux)],
                'r.', ms=2)
        for lo, hi in regions[i]:
            ax.axvspan(lo, hi, alpha=.24, color='green')
        finite = seg.wavelength[np.isfinite(seg.wavelength)]
        if finite.size:
            ax.set_xlim(float(finite.min()), float(finite.max()))
        fg = seg.flux[good]
        if len(fg) > 4:
            q1, q2 = np.percentile(fg, [1, 99.5])
            if q2 > q1: ax.set_ylim(q1-.1*(q2-q1), q2+.1*(q2-q1))
        ax.set_xlabel('Observed-frame vacuum wavelength [micron]')
        ax.set_ylabel('Extracted flux')
        ax.set_title(f'Segment {i+1}/{len(observation.segments)}: {seg.name} | '
                     f'{len(regions[i])} fitting region(s)')
        fig.suptitle('Drag: add  |  ← / →: change segment  |  d: undo  |  '
                     'c: clear  |  s: save and close  |  q: cancel', fontsize=11)
        fig.canvas.draw_idle()

    def on_span(xmin, xmax):
        lo, hi = sorted((float(xmin), float(xmax)))
        try:
            _validate_region(observation.segments[state['i']], lo, hi)
        except ValueError as exc:
            print('Rejected selection:', exc)
            return
        regions[state['i']].append((lo, hi))
        draw()

    selector = SpanSelector(ax, on_span, 'horizontal', useblit=True,
                            props={'alpha': .2, 'facecolor': 'blue'})
    fig._region_selector = selector

    def move(delta):
        state['i'] = (state['i'] + delta) % len(observation.segments)
        draw()

    def save():
        cleaned = {}
        for i, entries in regions.items():
            cleaned[i] = []
            for lo, hi in sorted(entries):
                _validate_region(observation.segments[i], lo, hi)
                cleaned[i].append([float(lo), float(hi)])
        if not any(cleaned.values()):
            print('No regions selected. Add at least one.')
            return
        if save_json is not None:
            Path(save_json).write_text(json.dumps({'regions_by_segment': cleaned,
                'segment_names': [s.name for s in observation.segments]}, indent=2))
            print('Saved', save_json)
        state['saved'] = True
        plt.close(fig)

    def on_key(event):
        i = state['i']
        if event.key == 'right': move(1)
        elif event.key == 'left': move(-1)
        elif event.key == 'd' and regions[i]: regions[i].pop(); draw()
        elif event.key == 'c': regions[i] = []; draw()
        elif event.key == 's': save()
        elif event.key == 'q': plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    buttons = []
    for pos, title, cb in [(.23, 'Prev', lambda _: move(-1)),
                            (.39, 'Next', lambda _: move(1)),
                            (.55, 'Save regions', lambda _: save())]:
        axis = fig.add_axes([pos, .055, .14, .07])
        b = Button(axis, title)
        b.on_clicked(cb)
        buttons.append(b)
    fig._region_buttons = buttons
    draw()
    plt.show()
    if not state['saved']:
        raise RuntimeError('Region selection canceled; nothing prepared')
    return [(lo, hi) for i in range(len(observation.segments))
            for lo, hi in regions[i]]


def load_regions_json(path, observation):
    data = json.loads(Path(path).read_text())
    names = [seg.name for seg in observation.segments]
    if data.get('segment_names') != names:
        raise ValueError('Saved region segment names do not match input; '
                         'do not reuse regions across different orders blindly')
    by_segment = data['regions_by_segment']
    entries = []
    for i, seg in enumerate(observation.segments):
        for lo, hi in by_segment.get(str(i), []):
            _validate_region(seg, lo, hi)
            entries.append((float(lo), float(hi)))
    return entries


# Vacuum wavelengths (microns), curated from Sharon et al. (2010), AJ 139, 646,
# Table 4, doi:10.1088/0004-6256/139/2/646. These are illustrative labels,
# NOT an exhaustive VALD line list or a stellar template.
STELLAR_LINES = (
    ('Mn I', 0.9504665), ('Ti I', 0.950867), ('Ti I', 0.951329),
    ('Fe I', 0.951587), ('H I', 0.954859), ('Fe I', 0.957253),
    ('Cr I', 0.957692), ('Ti I', 0.960218), ('C I', 0.960567),
    ('Sr II', 1.003940), ('Sr II', 1.033010),
    ('C I', 1.054410), ('Fe I', 1.058000), ('Si I', 1.058800),
    ('Si I', 1.060630), ('Si I', 1.063060), ('Si I', 1.066390),
    ('C I', 1.068600), ('Si I', 1.069720), ('Si I', 1.073030),
    ('Si I', 1.075230), ('Fe I', 1.075595), ('Fe I', 1.078600),
    ('Si I', 1.078980), ('Mg I', 1.081410), ('Fe I', 1.082130),
    ('Si I', 1.083010), ('He I', 1.083330), ('Na I', 1.083780),
    ('Si I', 1.084680), ('Si I', 1.087250), ('Si I', 1.088578),
    ('Sr II', 1.091790), ('H I', 1.094110), ('Mg I', 1.095630),
    ('Mg I', 1.096030), ('Mg I', 1.096845), ('Si I', 1.098231),
    ('Cr I', 1.101870),
)
C_KMS = 299792.458


def plot_region(observation, wl_min, wl_max, *, rv_kms=0.,
                stellar_lines=True, normalized=True, ax=None):
    """Display overlapping segments only; original arrays remain unchanged.

    rv_kms is observed-topocentric stellar velocity, not solely systemic RV.
    """
    import matplotlib.pyplot as plt
    if not wl_min < wl_max:
        raise ValueError('wl_min must be less than wl_max')
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))
    count = 0
    for seg in observation.segments:
        w, f = np.asarray(seg.wavelength), np.asarray(seg.flux)
        good = _valid_pixels(seg) & (w >= wl_min) & (w <= wl_max)
        if np.count_nonzero(good) < 3:
            continue
        idx = np.argsort(w[good])
        ww, ff = w[good][idx], f[good][idx]
        if normalized:
            positives = ff[ff > 0]
            if positives.size == 0: continue
            scale = np.nanpercentile(positives, 95)
            if scale <= 0: continue
            ff = ff / scale
        ax.plot(ww, ff, lw=.85, label=seg.name)
        count += 1
    if stellar_lines:
        overlay_stellar_lines(ax, wl_min, wl_max, rv_kms=rv_kms)
    ax.set(xlim=(wl_min, wl_max), xlabel='Observed vacuum wavelength (µm)',
           ylabel='Approx. normalized flux' if normalized else 'Extracted flux')
    if count <= 12 and count > 0: ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=.15)
    return ax


def overlay_stellar_lines(ax, wl_min, wl_max, *, rv_kms=0.,
                          lines=STELLAR_LINES, labels=True):
    """Mark catalog positions shifted by the observed stellar radial velocity."""
    if abs(rv_kms) >= C_KMS:
        raise ValueError('Absolute RV must be below c')
    beta = rv_kms / C_KMS
    factor = np.sqrt((1 + beta) / (1 - beta))
    for species, rest_vac_um in lines:
        center = rest_vac_um * factor
        if wl_min <= center <= wl_max:
            ax.axvline(center, color='tab:orange', ls='--', lw=.7, alpha=.65)
            if labels:
                ax.text(center, .99, species, transform=ax.get_xaxis_transform(),
                        rotation=90, va='top', ha='right', fontsize=7,
                        color='tab:orange', clip_on=True)


def choose_regions(observation: Observation, *, wavelength_range=None,
                   edge_pixels=100, initial=None, save_json=None,
                   rv_kms=0., stellar_lines=True, normalize=True,
                   start_empty=True):
    """Notebook-friendly GUI restricted to relevant wavelength segments.

    Keys: left/right to navigate, drag to add, d undo, c clear, s save, q cancel.
    Returns (low, high, segment_name) tuples to avoid overlap ambiguity.
    Use %matplotlib qt (or a supported interactive backend) in Jupyter.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector, Button
    if wavelength_range is None:
        wavelength_range = (min(np.nanmin(s.wavelength) for s in observation.segments),
                            max(np.nanmax(s.wavelength) for s in observation.segments))
    left, right = wavelength_range
    indices = [i for i, s in enumerate(observation.segments)
               if np.any(np.isfinite(s.wavelength) & (s.wavelength >= left)
                         & (s.wavelength <= right))]
    if not indices:
        raise ValueError('No segments overlap wavelength_range')
    if initial is None:
        raw = {} if start_empty else suggested_regions(observation, edge_pixels)
    elif isinstance(initial, dict):
        raw = initial
    else:
        raw = {i: [] for i in indices}
        names = {seg.name: i for i, seg in enumerate(observation.segments)}
        for entry in initial:
            if len(entry) != 3 or entry[2] not in names:
                raise ValueError('Initial regions must be (low, high, segment_name)')
            raw[names[entry[2]]].append(tuple(entry[:2]))
    regions = {i: [tuple(x) for x in raw.get(i, [])] for i in indices}
    state = {'k': 0, 'saved': False}
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(bottom=.20, top=.86)

    def draw():
        ax.clear()
        i = indices[state['k']]
        seg = observation.segments[i]
        w, f = seg.wavelength, seg.flux
        good = _valid_pixels(seg) & (w >= left) & (w <= right)
        if good.any():
            idx = np.argsort(w[good])
            vals = f[good][idx]
            if normalize:
                positives = vals[vals > 0]
                scale = np.percentile(positives, 95) if positives.size else 1.
                if scale > 0: vals = vals / scale
            ax.plot(w[good][idx], vals, color='black', lw=.75)
            if vals.size >= 10:
                bottom, top = np.percentile(vals, [1, 99.5])
                if top > bottom: ax.set_ylim(bottom-.10*(top-bottom), top+.12*(top-bottom))
        if stellar_lines:
            overlay_stellar_lines(ax, left, right, rv_kms=rv_kms)
        for low, high in regions[i]:
            ax.axvspan(low, high, color='tab:green', alpha=.25)
        finite = w[np.isfinite(w)]
        ax.set_xlim(max(left, np.min(finite)), min(right, np.max(finite)))
        ax.set_xlabel('Observed vacuum wavelength (µm)')
        ax.set_ylabel('Approx. normalized flux' if normalize else 'Extracted flux')
        ax.set_title(f'{state["k"]+1}/{len(indices)}  {seg.name} | {len(regions[i])} region(s)')
        fig.suptitle('Drag: add  |  left/right: segment  |  d: undo  |  c: clear  |  s: save  |  q: cancel', fontsize=10)
        fig.canvas.draw_idle()

    def span(lo, hi):
        i = indices[state['k']]
        lo, hi = sorted((float(lo), float(hi)))
        try: _validate_region(observation.segments[i], lo, hi)
        except ValueError as exc:
            print('Invalid fitting interval:', exc)
            return
        regions[i].append((lo, hi))
        draw()

    selector = SpanSelector(ax, span, 'horizontal', useblit=True,
                            props={'facecolor': 'tab:blue', 'alpha': .2})
    fig._region_selector = selector
    def move(delta):
        state['k'] = (state['k'] + delta) % len(indices)
        draw()
    def save():
        saved = []
        for i in indices:
            for low, high in sorted(regions[i]):
                _validate_region(observation.segments[i], low, high)
                saved.append((float(low), float(high), observation.segments[i].name))
        if not saved:
            print('Select at least one region before saving')
            return
        if save_json:
            Path(save_json).write_text(json.dumps({
                'source': str(observation.source),
                'wavelength_frame': 'observatory vacuum microns',
                'regions': saved}, indent=2))
            print('Saved:', save_json)
        state['saved'] = True
        plt.close(fig)
    def key(event):
        if event.key == 'right': move(1)
        elif event.key == 'left': move(-1)
        elif event.key == 'd':
            if regions[indices[state['k']]]:
                regions[indices[state['k']]].pop()
                draw()
        elif event.key == 'c':
            regions[indices[state['k']]] = []
            draw()
        elif event.key == 's': save()
        elif event.key == 'q': plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', key)
    buttons = []
    for x, title, action in [(0.23, 'Previous', lambda _: move(-1)),
                             (0.41, 'Next', lambda _: move(1)),
                             (0.59, 'Save regions', lambda _: save())]:
        button = Button(fig.add_axes([x, .055, .15, .07]), title)
        button.on_clicked(action)
        buttons.append(button)
    fig._region_buttons = buttons
    draw()
    plt.show(block=True)
    if not state['saved']:
        raise RuntimeError('Selection canceled or window closed without saving')
    return [(float(lo), float(hi), observation.segments[i].name)
            for i in indices for lo, hi in sorted(regions[i])]


def load_regions_json(path, observation):
    """Load saved selections, preserving chip identifiers."""
    data = json.loads(Path(path).read_text())
    names = {seg.name: seg for seg in observation.segments}
    result = []
    for low, high, name in data['regions']:
        if name not in names:
            raise ValueError(f'Segment {name} not found in observation')
        _validate_region(names[name], low, high)
        result.append((float(low), float(high), name))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CRIRES+ interactive Molecfit bridge')
    parser.add_argument('spectrum', help='CR2RES extractedA/extractedB FITS file')
    parser.add_argument('--out', required=True, help='New output job directory')
    parser.add_argument('--range', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                        help='Wavelength limits in microns; default is full coverage')
    parser.add_argument('--trim', type=int, default=100)
    parser.add_argument('--regions-json', help='Load previously saved regions without GUI')
    parser.add_argument('--save-regions', help='Where to save interactive selections (JSON)')
    parser.add_argument('--metadata-json', help='Overrides for missing ESO header values')
    parser.add_argument('--molecules', nargs='+', default=['H2O'], help='Species to fit')
    parser.add_argument('--esorex', default='esorex', help='EsoRex executable path')
    parser.add_argument('--model-template')
    parser.add_argument('--calctrans-template')
    parser.add_argument('--correct-template')
    parser.add_argument('--run', action='store_true', help='Run all three recipes after preparing')
    args = parser.parse_args()
    if args.trim < 0: parser.error('--trim must be nonnegative')
    obs = load_crires(args.spectrum)
    full_low = min(float(np.nanmin(s.wavelength)) for s in obs.segments)
    full_high = max(float(np.nanmax(s.wavelength)) for s in obs.segments)
    wavelength_range = tuple(args.range) if args.range else (full_low, full_high)
    selected = obs.select(*wavelength_range)
    if args.regions_json:
        regions = load_regions_json(args.regions_json, selected)
    else:
        regions = choose_regions(selected, edge_pixels=args.trim,
                                 save_json=args.save_regions)
    overrides = json.loads(Path(args.metadata_json).read_text()) if args.metadata_json else {}
    job = prepare(obs, wavelength_range=wavelength_range, fit_regions=regions,
                  output_dir=args.out, metadata_overrides=overrides,
                  molecules=tuple(args.molecules), esorex=args.esorex,
                  model_template=args.model_template,
                  calctrans_template=args.calctrans_template,
                  correct_template=args.correct_template)
    print('Prepared job:', job.directory)
    print('Model SOF:', job.directory/'model.sof')
    if args.run:
        print('Corrected products:', job.run())
    else:
        print('Review config and SOF before running; pass --run to execute all recipes.')


if __name__ == '__main__':
    main()
