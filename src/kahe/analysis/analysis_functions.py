
import wave

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import iirnotch, filtfilt
from scipy.stats import median_abs_deviation

from kahe.data import Spectrum1D

from kahe.lines import TRIPLET_LINES, PASCHEN_LINES

from kahe.utils.helper_functions import NestedInterpolator

def make_spectra_list(input_directory, skip=None): 
    """
    Generate a list of spectra filenames from the input directory. 

    Optionally skip certain files.

    """


def interpolate_to_common_wavelength(
    spectra: list[Spectrum1D],
    grid_res: float = 112_500,
    wavelength_min: float = 1.080,
    wavelength_max: float = 1.096,
    high_error: float = 1e10,
    plot: bool = False,
):
    """
    Interpolate calibrated 1D spectra onto a common
    logarithmic wavelength grid.

    Parameters
    ----------
    spectra : list of Spectrum1D
        List of extracted, wavelength-calibrated spectra.

    grid_res : float
        Inverse logarithmic wavelength spacing.

    wavelength_min, wavelength_max : float
        Wavelength limits in microns.

    high_error : float
        Uncertainty assigned to flagged pixels.

    plot : bool
        Whether to plot interpolated spectra.

    Returns
    -------
    lambda_grid : ndarray
        Common logarithmic wavelength grid.

    spectra_grid : ndarray
        Interpolated flux, shape (n_spectra, n_pixels).

    error_grid : ndarray
        Interpolated uncertainties.

    all_interpolators : list
        Original NestedInterpolator objects.

    original_rel_errors : list
        Input uncertainties with zero-flux pixels flagged.

    amplitudes : ndarray
        Median flux for each input spectrum.
    """

    if not spectra:
        raise ValueError("No spectra provided.")

    if grid_res <= 0:
        raise ValueError("grid_res must be positive.")

    if wavelength_min <= 0 or wavelength_max <= wavelength_min:
        raise ValueError("Invalid wavelength limits.")

    frames = {s.frame for s in spectra}

    if len(frames) != 1:
        raise ValueError(
            "All spectra must share a wavelength reference frame."
        )

    # Common logarithmic wavelength grid
    lambda_grid = np.exp(
        np.arange(
            np.log(wavelength_min),
            np.log(wavelength_max),
            1.0 / grid_res,
        )
    )

    spectra_grid = []
    error_grid = []
    all_interpolators = []
    original_rel_errors = []
    amplitudes = []

    if plot:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    for spectrum in spectra:

        wavelengths = spectrum.wavelength
        flux = spectrum.flux
        errors = spectrum.error

        # Original bad-pixel convention
        relative_errors = errors.copy()
        relative_errors[flux == 0] = high_error
        original_rel_errors.append(relative_errors)

        # Preserve original interpolator
        interpolator = NestedInterpolator(
            lambda_grid,
            wavelengths,
            flux,
        )

        all_interpolators.append(interpolator)

        amplitudes.append(np.median(flux))

        # Interpolate flux and errors
        flux_interp = np.interp(
            lambda_grid,
            wavelengths,
            flux,
            left=np.nan,
            right=np.nan,
        )

        error_interp = np.interp(
            lambda_grid,
            wavelengths,
            errors,
            left=np.nan,
            right=np.nan,
        )

        spectra_grid.append(flux_interp)
        error_grid.append(error_interp)

        if plot:
            ax.plot(
                lambda_grid,
                flux_interp,
                alpha=0.6,
                linewidth=0.8,
            )

    spectra_grid = np.asarray(spectra_grid)
    error_grid = np.asarray(error_grid)
    amplitudes = np.asarray(amplitudes)

    if plot:
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        ax.set_ylabel("Flux")
        ax.set_title("Interpolated Spectra")
        fig.tight_layout()
        plt.show()

    return (
        lambda_grid,
        spectra_grid,
        error_grid,
        all_interpolators,
        original_rel_errors,
        amplitudes,
    )

def reject_outliers(lambda_grid,
                    spectra_grid, 
                    sigma = 3.0, 
                    plot = True, 
                    save_path=None, 
                    return_mask=False, 
                    scale_mad=False):
    # Specify wthe wavelength ranges 
    helium_range = (1.0825, 1.0840)
    paschen_range = (1.0935, 1.0947)

    wavelength = np.asarray(lambda_grid)
    spectra = np.asarray(spectra_grid, dtype=float)

    if spectra.ndim !=2: 
        raise ValueError("spectra_grid must be a 2D array")

    if spectra.shape[1] != len(wavelength): 
        raise ValueError("Wavelength grid and spectra have incompatible dimensions")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Calcualte temporal median and MAD 
    with np.errstate(invalid='ignore'):
        median_spectrum = np.nanmedian(spectra, axis=0)
        mad = median_abs_deviation(spectra, axis=0, nan_policy='omit', scale = "normal" if scale_mad else 1.0)

    # Identify outliers
    valid_mad = np.isfinite(mad) & (mad > 0)
    outlier_mask = np.abs(spectra - median_spectrum) > (sigma * mad[None, :])
    outlier_mask &= valid_mad[None, :]

    filtered_spectra = spectra.copy()

    replacement = np.broadcast_to(median_spectrum, spectra.shape)
    filtered_spectra[outlier_mask] = replacement[outlier_mask] 

    # count the number of rejected points 
    num_rejected = np.count_nonzero(outlier_mask)
    print(f"Rejected {num_rejected} pixels")

    if plot or save_path is not None:

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 4),
            dpi=300,
        )

        regions = [
            (axes[0], helium_range, "Helium Line"),
            (axes[1], paschen_range, "Paschen Line"),
        ]

        for ax, region, title in regions:

            mask = (
                (wavelength >= region[0])
                & (wavelength <= region[1])
            )

            if not np.any(mask):
                ax.text(
                    0.5, 0.5,
                    "No wavelength coverage",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                for spectrum in filtered_spectra:
                    ax.plot(
                        wavelength[mask],
                        spectrum[mask],
                        alpha=0.7,
                    )

            ax.set_xlim(*region)
            ax.set_xlabel(r"Wavelength [$\mu$m]")
            ax.set_ylabel("Flux")
            ax.set_title(title)
            ax.grid(alpha=0.2)

        # Helium triplet
        for line in TRIPLET_LINES:
            axes[0].axvline(
                line,
                color="red",
                linestyle="--",
                alpha=0.7,
            )

        # Paschen-gamma
        for line in PASCHEN_LINES:
            axes[1].axvline(
                line,
                color="red",
                linestyle="--",
                alpha=0.7,
            )

        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            fig.savefig(save_path, dpi=300)

        if plot:
            plt.show()

        plt.close(fig)

    # IMPORTANT: return statements outside plotting block
    if return_mask:
        return filtered_spectra, outlier_mask

    return filtered_spectra

def assess_SNR(lambda_grid,
               spectra_grid, 
               error_grid,
               *, 
               plot=True, 
               save_path=None, 
               helium_range=(1.0825, 1.0840),
               paschen_range=(1.0935, 1.0947)): 
    """
    Calculate the SNR of the spectra.
    Computed by the taking the ratio of flux/error at the center of the He Triplet component and Paschen-gamma.

    Parameters
    ----------
    lambda_grid : array-like
        Wavelength grid of the spectra.
    spectra_grid : array-like
        Flux values of the spectra.
    error_grid : array-like
        Error values corresponding to the spectra.
    plot : bool, optional
        Whether to plot the spectra and SNR, by default True.
    save_path : str or Path, optional
        Path to save the plot, by default None.
    helium_range : tuple, optional
        Wavelength range for the Helium triplet, by default (1.0825, 1.0840).
    paschen_range : tuple, optional
        Wavelength range for the Paschen-gamma line, by default (1.0935, 1.0947).

    Returns
    -------
    snr_he : array-like
        SNR values for the Helium triplet.
    snr_paschen : array-like
        SNR values for the Paschen-gamma line.
    """

    wavelength = np.asarray(lambda_grid, dtype=float)
    flux = np.asarray(spectra_grid, dtype=float)
    error = np.asarray(error_grid, dtype=float)

    # Checks for consistency