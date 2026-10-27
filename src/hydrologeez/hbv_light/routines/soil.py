"""Beginning-of-day recharge, capacity overflow, then soil evaporation."""

from numba import njit


@njit(cache=True)
def soil(
    soil_input: float,
    pet: float,
    moisture: float,
    fc: float,
    beta: float,
    lp: float,
) -> tuple[float, float, float, float, float]:
    """Return M, recharge, evaporation, raw recharge, overflow (mm)."""
    initial_recharge = soil_input * (moisture / fc) ** beta
    moisture += soil_input - initial_recharge
    overflow = max(moisture - fc, 0.0)
    moisture = min(moisture, fc)
    recharge = initial_recharge + overflow
    evaporation = min(moisture, pet * min(moisture / (lp * fc), 1.0))
    moisture -= evaporation
    return moisture, recharge, evaporation, initial_recharge, overflow
