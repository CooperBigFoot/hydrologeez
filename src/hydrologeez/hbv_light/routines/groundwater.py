"""Shared groundwater: percolation priority, nonlinear upper and linear lower outflow."""

from numba import njit


@njit(cache=True)
def groundwater(
    recharge: float,
    upper: float,
    lower: float,
    perc: float,
    alpha: float,
    k1: float,
    k2: float,
) -> tuple[float, float, float, float, float, float]:
    """Return U, V, generated runoff, percolation, q1, q2 (basin mm).

    Daily K1 uses mm**(-alpha)/day; storage is evaluated numerically in mm.
    Recharge then percolation then discharge is our explicit timing choice.
    """
    upper += recharge
    d = min(perc, upper)
    upper -= d
    lower += d
    q1 = min(k1 * upper ** (1.0 + alpha), upper)
    q2 = min(lower, k2 * lower)
    generated = q1 + q2
    return upper - q1, lower - q2, generated, d, q1, q2
