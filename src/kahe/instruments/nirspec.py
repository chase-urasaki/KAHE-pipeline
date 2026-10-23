from pathlib import Path
import numpy as np

from kahe.data import Spectrum1D

"""  NIRSPEC instrument-specific constants and functions
"""
N = 2048
LEFT_MARGIN = 10
RIGHT_MARGIN = 60
FLAT_MIN = 0.5
FLAT_MAX = 1.5
GAIN = 3.01
READ_NOISE = 28 #CDS, one co-add
MIN_ORDER_SEPARATION = 10
OVERSCAN_WIDTH = 4
HIGH_ERROR = 1e10
NIRSPEC_RES = 37500


#For empirically fitting the trace
SAVGOL_N = [51, 101, 101, 41, 21, 21, 21, 21]
SAVGOL_O = [3, 3, 3, 3, 3, 3, 2, 3]

# Upper and lower bounds for filters 
NIRSPEC1_70 = (1433), (1524)
NIRSPECHEI_1 = (1078), (1278)



def read_nirspec_spectrum(
    filepath,
    frame,
    time_mjd=None,
    airmass=None,
):
    """
    Read a calibrated, extracted NIRSPEC spectrum.

    Parameters
    ----------
    filepath : str or Path
        Path to the .spectrum file.
    frame : str
        Wavelength reference frame, e.g. 'topocentric'
        or 'barycentric'.
    time_mjd : float, optional
        Mid-exposure time in MJD UTC.
    airmass : float, optional
        Observation airmass.

    Returns
    -------
    Spectrum1D
        Standardized 1D spectrum.
    """

    filepath = Path(filepath)

    data = np.loadtxt(
        filepath,
        skiprows=1,
        ndmin=2,
    )

    if data.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns in {filepath}"
        )

    order = data[:, 0]
    wavelength = data[:, 2]
    flux = data[:, 3]
    error = data[:, 4]
    continuum = data[:, 5]

    idx = np.argsort(wavelength)

    return Spectrum1D(
        wavelength=wavelength[idx],
        flux=flux[idx],
        error=error[idx],
        frame=frame,
        time_mjd=time_mjd,
        airmass=airmass,
        metadata={
            "instrument": "NIRSPEC",
            "filename": str(filepath),
            "order": order[idx],
            "continuum": continuum[idx],
            "resolution": NIRSPEC_RES,
        },
    )