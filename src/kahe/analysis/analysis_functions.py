
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import iirnotch, filtfilt
from scipy.stats import median_abs_deviation

from kahe.utils.helper_functions import NestedInterpolator


def interpolate_to_common_wavelength(
    list_of_spectra,
    config,
    plot=True,
    apply_notch=True,
    wavelength_min=1.081,
    wavelength_max=1.085,
):
    """
    Interpolate extracted 1D spectra onto a common logarithmic
    wavelength grid.

    Optionally apply a notch filter to remove fringing.

    Parameters
    ----------
    list_of_spectra : list of str
        Filenames of extracted, calibrated 1D spectra.

    config : configparser.ConfigParser
        Target configuration loaded from an INI file.

    plot : bool, optional
        Plot the interpolated spectra. Default is True.

    apply_notch : bool, optional
        Apply the original two-pass notch filter.
        Default is True.

    wavelength_min, wavelength_max : float, optional
        Wavelength limits in microns.

    Returns
    -------
    lambda_grid : ndarray
        Common logarithmic wavelength grid.

    spectra_grid : ndarray
        Interpolated spectra, shape (n_spectra, n_pixels).

    error_grid : ndarray
        Interpolated uncertainties.

    all_interpolators : list
        NestedInterpolator objects for each spectrum.

    original_rel_errors : list
        Original uncertainties with bad pixels flagged.

    amplitudes : ndarray
        Median flux of each original spectrum.
    """
    # Read the configuratiion File
    name = config["Target"]["name"]
    date = config["Target"]["date"]
    method = config["Extraction Method"]["Method"]

    grid_res = config.getfloat("Analysis", "GRID_RES")
    high_error = config.getfloat("Analysis", "HIGH_ERROR")

    input_directory = Path(
        f"all_spectra_{name}_{date}_{method}"
    )

    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_directory}"
        )

    # Construct common logarithmic wavelength grid
    lambda_grid = np.exp(
        np.arange(
            np.log(wavelength_min),
            np.log(wavelength_max),
            1.0 / grid_res,
        )
    )

    # --------------------------------------------------
    # 3. Initialize output containers
    # --------------------------------------------------

    spectra_grid = []
    error_grid = []
    all_interpolators = []
    original_rel_errors = []
    amplitudes = []

    if plot:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    # --------------------------------------------------
    # 4. Load and interpolate individual spectra
    # --------------------------------------------------

    for filename in list_of_spectra:

        if not filename:
            continue

        filepath = input_directory / filename

        if not filepath.is_file():
            raise FileNotFoundError(
                f"Spectrum not found: {filepath}"
            )

        (
            orders,
            _,
            wavelengths,
            spectra,
            errors,
            continuum,
        ) = np.loadtxt(
            filepath,
            unpack=True,
            skiprows=1,
        )

        # Ensure wavelengths are sorted for interpolation
        sort_idx = np.argsort(wavelengths)

        wavelengths = wavelengths[sort_idx]
        spectra = spectra[sort_idx]
        errors = errors[sort_idx]

        # Preserve original bad-pixel treatment
        relative_errors = errors.copy()
        relative_errors[spectra == 0] = high_error

        original_rel_errors.append(relative_errors)

        # Construct the original nested interpolator
        interpolator = NestedInterpolator(
            lambda_grid,
            wavelengths,
            spectra,
        )

        all_interpolators.append(interpolator)

        # Preserve original median-flux measurement
        amplitudes.append(np.median(spectra))

        # Interpolate flux and uncertainty
        flux_interp = np.interp(
            lambda_grid,
            wavelengths,
            spectra,
        )

        error_interp = np.interp(
            lambda_grid,
            wavelengths,
            errors,
        )

        # ----------------------------------------------
        # 5. Optional notch filtering
        # ----------------------------------------------

        if apply_notch:

            b, a = iirnotch(
                w0=0.052,
                Q=15,
                fs=1,
            )

            # Preserve original two-pass filtering
            flux_interp = filtfilt(b, a, flux_interp)
            flux_interp = filtfilt(b, a, flux_interp)

        spectra_grid.append(flux_interp)
        error_grid.append(error_interp)

        if plot:
            ax.plot(
                lambda_grid,
                flux_interp,
                alpha=0.6,
                linewidth=0.8,
            )

    # --------------------------------------------------
    # 6. Convert outputs to arrays
    # --------------------------------------------------

    spectra_grid = np.asarray(spectra_grid)
    error_grid = np.asarray(error_grid)
    amplitudes = np.asarray(amplitudes)

    # --------------------------------------------------
    # 7. Plot formatting
    # --------------------------------------------------

    if plot:
        ax.set_title(f"Reduced Spectra for {name}")
        ax.set_xlabel("Wavelength [microns]")
        ax.set_ylabel("Normalized Flux")

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

def reject_outliers(lambda_grid,spectra_grid): 
    """ 
    Performs outlier rejection based on median absolute deviation (MAD). 

    Parameters:
    ------------
    lambda_grid : np.ndarray
        The common logarithmic wavelength grid.
    spectra_grid : np.ndarray
        The grid of spectra corresponding to the wavelength grid.

    Returns:
    ------------
    filtered_spectra_grid : np.ndarray
        The spectra grid with outliers removed based on MAD.
    """

    # Instatntiate figure
    fig, ax = plt.subplots(figsize=(10,4))
    fig.suptitle('Spectra (Outlier rejected)')

    ax.set_ylabel('Normalized Flux')
    ax.set_xlabel('Wavelength [microns]')

    # Perform outlier rejection
    for c in range(spectra_grid.shape[1]):
        data = spectra_grid[:,c]
        mad = median_abs_deviation(data)
        sigmas = (data - np.median(data)) / mad
        outliers = np.abs(sigmas) > 5
        data[outliers] = np.median(data)
        
    for row in spectra_grid:
        ax.plot(lambda_grid, row)
    
    ax.axvline(He_line1, color='r', linestyle='--', alpha = 0.5)
    ax.axvline(He_line2, color='r', linestyle='--', alpha = 0.5)
    ax.axvline(He_line3, color='r', linestyle='--', alpha = 0.5)

    #Grey around the region 
    ax.axvspan(He_line2 - width/2, He_line2 + width/2, color='grey', alpha=0.3, label='Region of Interest')


    plt.savefig(f"{analysis_output}/{NAME}_{DATE}_rejected_outliers.png", dpi=600)
    plt.show()
    plt.close()