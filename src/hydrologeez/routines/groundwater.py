"""Shared groundwater: percolation priority and limited linear outflows."""

from numba import njit


@njit(cache=True)
def groundwater(
    recharge: float,
    upper: float,
    lower: float,
    perc: float,
    uzl: float,
    k0: float,
    k1: float,
    k2: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Return U, V, generated runoff, percolation, q0, q1, q2 (basin mm)."""
    upper += recharge
    d = min(perc, upper)
    upper -= d
    lower += d
    q0 = k0 * max(upper - uzl, 0.0)
    q1 = k1 * upper
    scale = min(1.0, upper / (q0 + q1)) if q0 + q1 > 0.0 else 1.0
    q0 *= scale
    q1 *= scale
    q2 = min(lower, k2 * lower)
    generated = q0 + q1 + q2
    # Proportional limiting can leave a negative roundoff residual at exhaustion.
    upper = max(upper - q0 - q1, 0.0)
    return upper, lower - q2, generated, d, q0, q1, q2
