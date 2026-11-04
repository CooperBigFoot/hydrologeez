"""Potential evaporation preparation conventions."""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def oudin(temperature: Array, latitude: Array, day: NDArray[np.int64]) -> Array:
    """mm/day; FAO56 Eqs21–25 Re, Oudin2005 Eq3, lambda=2.45 MJ/kg."""
    phi = np.deg2rad(latitude)
    angle = 2 * np.pi * day / 365
    distance = 1 + 0.033 * np.cos(angle)
    declination = 0.409 * np.sin(angle - 1.39)
    sunset = np.arccos(np.clip(-np.tan(phi) * np.tan(declination), -1, 1))
    radiation = (
        1440
        / np.pi
        * 0.0820
        * distance
        * (sunset * np.sin(phi) * np.sin(declination) + np.cos(phi) * np.cos(declination) * np.sin(sunset))
    )
    return radiation / 2.45 * np.maximum(temperature + 5, 0) / 100
