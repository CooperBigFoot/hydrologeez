"""Snow partition, phase change, and solid-snow liquid retention."""

from numba import njit


@njit(cache=True)
def snow(
    precipitation: float,
    temperature: float,
    solid: float,
    liquid: float,
    tt: float,
    cfmax: float,
    cfr: float,
    cwh: float,
    scf: float,
) -> tuple[float, float, float, float, float, float]:
    """Return S, L, soil input, corrected precipitation, melt, refreeze (mm)."""
    corrected = precipitation * scf if temperature < tt else precipitation
    if temperature < tt:
        solid += corrected
    else:
        liquid += precipitation
    melt = min(solid, cfmax * max(temperature - tt, 0.0))
    freeze = min(liquid, cfr * cfmax * max(tt - temperature, 0.0))
    solid += freeze - melt
    liquid += melt - freeze
    soil_input = max(liquid - cwh * solid, 0.0)
    liquid = min(liquid, cwh * solid)
    return solid, liquid, soil_input, corrected, melt, freeze
