"""Wavelength-driven ESO Molecfit bridge, revised draft (v3).

Requires numpy, astropy and an EsoRex installation containing Molecfit recipes.
Does not change original extracted spectra. Inputs are assumed to be observed-frame,
VACUUM wavelengths (verify this for each instrument).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import subprocess
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import units as u


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
            if not (len(s.wavelength) == len(s.flux) == len(s.error)):
                raise ValueError(f'Array length mismatch in {s.name}')
            mask = ((s.wavelength >= low) & (s.wavelength <= high)
                    & np.isfinite(s.wavelength) & np.isfinite(s.flux)
                    & np.isfinite(s.error) & (s.error > 0))
            if mask.sum() >= 3:
                ind = np.argsort(s.wavelength[mask])
                selected.append(Segment(s.name, s.wavelength[mask][ind],
                                        s.flux[mask][ind], s.error[mask][ind]))
        for seg in selected:
            if np.any(np.diff(seg.wavelength) <= 0):
                raise ValueError(f'Duplicate/non-increasing wavelength samples in {seg.name}')
        if not selected:
            raise ValueError(f'No valid spectral segments in {low}–{high} um')
        return Observation(selected, self.header.copy(), self.source,
                           self.metadata.copy())


def load_crires(path: str | Path, *, wavelength_unit: str = 'nm') -> Observation:
    """Read extracted CRIRES+ table columns; one order/detector per segment.

    Default wavelength_unit='nm' is intended for standard CR2RES WL columns.
    Verify against your actual product; no wavelength-unit heuristics are used.
    """
    path = Path(path).resolve()
    scale = (1 * u.Unit(wavelength_unit)).to_value(u.um)
    segments = []
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        for hdu in hdul[1:]:
            if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None:
                continue
            names = set(hdu.columns.names)
            for wname in sorted(n for n in names if n.endswith('_WL')):
                prefix = wname[:-3]
                if {prefix + '_SPEC', prefix + '_ERR'} - names:
                    continue
                raw_wave = np.asarray(hdu.data[wname], dtype=float)
                raw_flux = np.asarray(hdu.data[prefix + '_SPEC'], dtype=float)
                raw_err = np.asarray(hdu.data[prefix + '_ERR'], dtype=float)
                if raw_wave.size == 0 or raw_wave.shape != raw_flux.shape or raw_wave.shape != raw_err.shape:
                    raise ValueError(f'Missing/mismatched arrays: {hdu.name}/{prefix}')
                if raw_wave.ndim > 2 or (raw_wave.ndim == 2 and 1 not in raw_wave.shape):
                    raise ValueError(f'Unexpected multi-spectrum column shape {raw_wave.shape}: {hdu.name}/{prefix}')
                segments.append(Segment(f'{hdu.name}_{prefix}', raw_wave.ravel().copy() * scale,
                                        raw_flux.ravel().copy(), raw_err.ravel().copy()))
    if not segments:
        raise ValueError(f'No extracted wavelength/flux/error columns in {path}')
    return Observation(segments, header, path, {'instrument': 'CRIRES+', 'input_wavelength_unit': wavelength_unit})


def load_nirspec(spectrum_path: str | Path, raw_header_path: str | Path, *,
                 wavelength_unit: str = 'angstrom', observed_frame: bool = False) -> Observation:
    """Load existing six-column NIRSPEC .spectrum files by order.

    Important: observed_frame must be explicitly True. A wavelength-corrected
    stellar/barycentric-frame .spectrum needs reversal upstream using the exact
    reduction convention, NOT an assumed velocity sign here.
    """
    if not observed_frame:
        raise ValueError('NIRSPEC wavelengths must first be verified/converted to observatory frame; set observed_frame=True only after that')
    spectrum_path = Path(spectrum_path).resolve()
    raw_header_path = Path(raw_header_path).resolve()
    data = np.loadtxt(spectrum_path, comments='#', ndmin=2)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError('Expected NIRSPEC: order pixel wavelength flux error continuum')
    scale = (1 * u.Unit(wavelength_unit)).to_value(u.um)
    with fits.open(raw_header_path) as hdul:
        header = hdul[0].header.copy()
    segments = []
    for order in np.unique(data[:, 0]):
        subset = data[data[:, 0] == order]
        continuum = subset[:, 5]
        valid = np.isfinite(continuum) & (continuum > 0)
        normalized_flux = np.full(len(subset), np.nan)
        normalized_error = np.full(len(subset), np.nan)
        normalized_flux[valid] = subset[valid, 3] / continuum[valid]
        normalized_error[valid] = subset[valid, 4] / continuum[valid]
        segments.append(Segment(f'order_{int(order)}', subset[:, 2] * scale,
                                normalized_flux, normalized_error))
    return Observation(segments, header, spectrum_path, {'instrument': 'NIRSPEC', 'raw_header': str(raw_header_path)})


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
        'mjd': ('MJD-OBS', 'MJD', 'ESO DET FRAM MJD'),
        'airmass_start': ('HIERARCH ESO TEL AIRM START', 'ESO TEL AIRM START', 'AIRMASS'),
        'airmass_end': ('HIERARCH ESO TEL AIRM END', 'ESO TEL AIRM END', 'AIRMASS'),
        'humidity': ('HIERARCH ESO TEL AMBI RHUM', 'WXOUTHUM'),
        'pressure': ('HIERARCH ESO TEL AMBI PRES START', 'HIERARCH ESO TEL AMBI PRES END', 'WXPRESS'),
        'temperature': ('HIERARCH ESO TEL AMBI TEMP', 'WXOUTTMP'),
        'slit_width': ('HIERARCH ESO INS SLIT1 WID', 'HIERARCH ESO INS SLIT WIDTH', 'SLITWIDTH'),
        'exposure_time': ('HIERARCH ESO DET SEQ1 DIT', 'EXPTIME', 'TRUITIME', 'ESO OBS EXECTIME'),
        'telescope_elevation': ('EL', 'HIERARCH ESO TEL ALT'),
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
    for key in required:
        if not np.isfinite(values[key]):
            raise ValueError(f'{key} must be finite')
    if min(values['airmass_start'], values['airmass_end']) < 1:
        raise ValueError('Airmass must be >= 1')
    if 'exposure_time' not in values:
        raise ValueError('Missing exposure_time; provide a verified exposure_time metadata override')
    if not np.isfinite(values['exposure_time']):
        raise ValueError('Exposure time must be finite')
    if values['exposure_time'] <= 0 or values['slit_width'] <= 0:
        raise ValueError('Exposure time and slit width must be positive')
    if 'observatory' in overrides:
        values['observatory'] = str(overrides['observatory'])
    else:
        values['observatory'] = 'paranal' if (obs.metadata.get('instrument') == 'CRIRES+' or 'CRIRES' in str(h.get('INSTRUME', '')).upper()) else 'unknown'
    if values['observatory'].lower() != 'paranal':
        for fieldname in ('latitude', 'longitude', 'elevation'):
            if fieldname not in overrides:
                raise ValueError(f'{fieldname} required for non-Paranal observations')
    site = {'latitude': -24.627, 'longitude': -70.405, 'elevation': 2635.0}
    for key in site:
        values[key] = float(overrides.get(key, site[key]))
        provenance[key] = 'explicit override' if key in overrides else 'Paranal site constant'
    if 'telescope_elevation' in values and not (0 < values['telescope_elevation'] <= 90):
        raise ValueError('Telescope elevation must be in (0,90] degrees')
    if 'telescope_elevation' not in values:
        values['telescope_elevation'] = float(np.degrees(np.arcsin(1 / values['airmass_start'])))
        provenance['telescope_elevation'] = 'airmass plane-parallel approximation'
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

        command = [
            self.esorex,
            "--msg-level=info",
            f"--recipe-config={config}",
            f"--output-dir={output}",
            recipe,
            str(sof),
        ]

        log_file = output / f"{recipe}.log"

        print(f"\n{'='*60}", flush=True)
        print(f"Running {recipe}", flush=True)
        print(f"{'='*60}", flush=True)

        with subprocess.Popen(
            command,
            cwd=output,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:

            with open(log_file, "w") as log:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()

            return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"{recipe} failed with exit code {return_code}. "
                f"See {log_file}"
            )

        print(f"\n✓ {recipe} completed successfully.", flush=True)

    def run(self):
        if (self.directory/'MULTICHIP_REVIEW_REQUIRED.txt').exists():
            raise RuntimeError('Multi-chip mappings require verification; see MULTICHIP_REVIEW_REQUIRED.txt')
        model_dir = self.directory/'model'
        trans_dir = self.directory/'calctrans'
        corr_dir = self.directory/'correct'
        self._run_recipe('molecfit_model', self.directory/'model.rc',
                         self.directory/'model.sof', model_dir)
        model_tags = ['MODEL_MOLECULES', 'ATM_PARAMETERS', 'BEST_FIT_PARAMETERS', 'BEST_FIT_MODEL']
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

    Does not run Molecfit. Regions MUST lie inside exactly one selected segment
    and contain >=10 good pixels. The caller must verify wavelength frame.
    Multi-segment execution is blocked pending extension mapping validation.
    """
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is non-empty; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    obs = observation.select(*wavelength_range)
    meta = _metadata(obs, metadata_overrides or {})
    if not fit_regions:
        raise ValueError('Set at least one validated telluric-fitting region')
    primary = fits.PrimaryHDU(header=obs.header.copy())
    primary.header['MJD-OBS'] = meta['mjd']
    primary.header['EXPTIME'] = meta['exposure_time']
    primary.header['AIRMASS'] = meta['airmass_start']
    primary.header['HIERARCH ESO OBS EXECTIME'] = meta['exposure_time']
    primary.header['HIERARCH ESO TEL AIRM START'] = meta['airmass_start']
    primary.header['HIERARCH ESO TEL AIRM END'] = meta['airmass_end']
    # Short FITS keys avoid ambiguous HIERARCH keyword handling across CPL versions.
    primary.header['AIRSTART'] = meta['airmass_start']
    primary.header['AIREND'] = meta['airmass_end']
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
    for low, high in fit_regions:
        if not low < high:
            raise ValueError(f'Bad fit region: {(low,high)}')
        matching=[]
        for i, seg in enumerate(obs.segments, start=1):
            contained=(low >= seg.wavelength.min() and high <= seg.wavelength.max())
            count=np.count_nonzero((seg.wavelength >= low)&(seg.wavelength <= high))
            if contained and count >= 10:
                matching.append(i)
        if len(matching) != 1:
            raise ValueError(f'Fit region {(low,high)} maps to {matching}; '
                             'must fall within exactly one segment with >=10 pixels')
        wave_table.append((float(low),float(high),matching[0]))
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
        'TELESCOPE_ANGLE_VALUE':str(meta['telescope_elevation']),
        'RELATIVE_HUMIDITY_KEYWORD':'NONE',
        'RELATIVE_HUMIDITY_VALUE':str(meta['humidity']),
        'PRESSURE_KEYWORD':'NONE','PRESSURE_VALUE':str(meta['pressure']),
        'TEMPERATURE_KEYWORD':'NONE','TEMPERATURE_VALUE':str(meta['temperature']),
        'SLIT_WIDTH_KEYWORD':'NONE','SLIT_WIDTH_VALUE':str(meta['slit_width']),
        'LATITUDE_KEYWORD':'NONE','LATITUDE_VALUE':str(meta['latitude']),
        'LONGITUDE_KEYWORD':'NONE','LONGITUDE_VALUE':str(meta['longitude']),
        'ELEVATION_KEYWORD':'NONE','ELEVATION_VALUE':str(meta['elevation']),
    }
    _set_rc(model, overrides, out/'model.rc')
    # The other recipes require header *KEYWORD NAMES*, not numeric values.
    # The same multi-chip interpretation must be used at every stage.
    _set_rc(calctrans, {
        'USE_ONLY_INPUT_PRIMARY_DATA': 'FALSE',
        'CHIP_EXTENSIONS': 'TRUE',
        'HDR_MJD': 'MJD-OBS',
        'HDR_EXP': 'EXPTIME',
        'HDR_AIR1': 'AIRSTART',
        'HDR_AIR2': 'AIREND',
        # Defaults in the Keck template assume only extensions 0,1.
        # Per-segment atmospheric/convolution mappings require validation
        # against actual model outputs, so reject multi-chip auto-execution.
    }, out/'calctrans.rc')
    _set_rc(correct, {
        'CHIP_EXTENSIONS': 'TRUE',
        'USE_ONLY_INPUT_PRIMARY_DATA': 'FALSE',
    }, out/'correct.rc')
    if len(obs.segments) > 1:
        # The installed template contains MAPPING_ATMOSPHERIC=0,1,
        # MAPPING_CONVOLVE=1,1 and MAPPING_CORRECT=0,1, which are explicitly
        # single-extension mappings. Replacing these without inspecting the
        # produced extensions can silently assign the wrong transmission.
        (out/'MULTICHIP_REVIEW_REQUIRED.txt').write_text(
            'Multiple extensions selected. Inspect the produced ATM_PARAMETERS,\n'
            'LBLRTM_RESULTS and TELLURIC_CORR extension numbers; update\n'
            'MAPPING_ATMOSPHERIC, MAPPING_CONVOLVE and MAPPING_CORRECT\n'
            'in the generated recipe configs before calling job.run().\n')

    (out/'model.sof').write_text(f'{science_file} SCIENCE\n{wave_file} WAVE_INCLUDE\n')
    manifest = {'source': str(obs.source), 'metadata': meta,
                'wavelength_range':list(wavelength_range),
                'fit_regions':[list(x) for x in fit_regions],
                'segments':segment_details, 'molecules':mols,
                'frame':'observatory vacuum (caller must verify)',
                'input_normalized': obs.metadata.get('instrument') == 'NIRSPEC',
                'note':'Fit windows are caller-supplied and must be validated.'}
    (out/'metadata.json').write_text(json.dumps(manifest, indent=2))
    return MolecfitJob(out, science_file, manifest, esorex)