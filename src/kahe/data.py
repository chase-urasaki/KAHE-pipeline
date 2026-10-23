
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Spectrum1D:
    """
    Container for a single calibrated 1D spectrum.

    Wavelengths must be in vacuum microns.
    The wavelength reference frame must be identified.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array [microns].
    flux : ndarray
        Extracted or normalized flux.
    error : ndarray
        1-sigma flux uncertainties.
    frame : str
        Wavelength reference frame.
    time_mjd : float, optional
        Mid-exposure time in MJD UTC.
    airmass : float, optional
        Airmass of the observation.
    metadata : dict
        Additional instrument-specific information.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray

    frame: str
    time_mjd: float | None = None
    airmass: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.wavelength = np.asarray(
            self.wavelength, dtype=float
        )
        self.flux = np.asarray(
            self.flux, dtype=float
        )
        self.error = np.asarray(
            self.error, dtype=float
        )

        if self.wavelength.ndim != 1:
            raise ValueError("Spectrum must be 1D.")

        if not (
            self.wavelength.shape
            == self.flux.shape
            == self.error.shape
        ):
            raise ValueError(
                "Wavelength, flux, and error must have "
                "matching shapes."
            )

        if not np.all(np.isfinite(self.wavelength)):
            raise ValueError("Wavelengths must be finite.")

        if np.any(self.wavelength <= 0):
            raise ValueError("Wavelengths must be positive.")

        if not np.all(np.diff(self.wavelength) > 0):
            raise ValueError(
                "Wavelengths must be strictly increasing."
            )

        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError(
                "Specify the wavelength reference frame."
            )
