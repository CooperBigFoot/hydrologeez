"""Experimental float64 daily equations; all inputs are resolved, valid arrays."""

import numpy as np
from numba import njit
from numpy.typing import NDArray

Array = NDArray[np.float64]


def weights(duration: float) -> Array:
    edges = np.minimum(np.arange(int(np.ceil(duration)) + 1), duration)
    cdf = np.where(edges <= duration / 2, 2 * (edges / duration) ** 2, 1 - 2 * ((duration - edges) / duration) ** 2)
    return np.diff(cdf)


@njit(cache=True)
def day(
    forcing: Array, area: Array, parameters: Array, response: Array, routing: Array, zone: Array, basin: Array
) -> tuple[Array, Array]:
    """Mutate end-of-day memory; report zone I,R,A,P*,m,f,r,o and basin Q,G,D,q0,q1,q2."""
    flux = np.empty((len(area), 8))
    recharge = 0.0
    for z in range(len(area)):
        p, t, e = forcing[z]
        tt, cfmax, cfr, cwh, scf, fc, beta, lp = parameters[z]
        s, liquid, moisture = zone[z]
        corrected = p * scf if t < tt else p
        if t < tt:
            s += corrected
        else:
            liquid += p
        melt = min(s, cfmax * max(t - tt, 0.0))
        freeze = min(liquid, cfr * cfmax * max(tt - t, 0.0))
        s += freeze - melt
        liquid += melt - freeze
        soil_input = max(liquid - cwh * s, 0.0)
        liquid -= soil_input
        initial_recharge = soil_input * (moisture / fc) ** beta
        moisture += soil_input - initial_recharge
        overflow = max(moisture - fc, 0.0)
        moisture -= overflow
        r = initial_recharge + overflow
        evaporation = min(moisture, e * min(moisture / (lp * fc), 1.0))
        moisture -= evaporation
        zone[z] = s, liquid, moisture
        flux[z] = soil_input, r, evaporation, corrected, melt, freeze, initial_recharge, overflow
        recharge += area[z] * r
    perc, uzl, k0, k1, k2 = response
    upper = basin[0] + recharge
    d = min(perc, upper)
    upper -= d
    lower = basin[1] + d
    q0 = k0 * max(upper - uzl, 0.0)
    q1 = k1 * upper
    scale = min(1.0, upper / (q0 + q1)) if q0 + q1 > 0.0 else 1.0
    q0 *= scale
    q1 *= scale
    q2 = min(lower, k2 * lower)
    generated = q0 + q1 + q2
    outlet = routing[0] * generated
    if len(routing) > 1:
        outlet += basin[2]
    for j in range(len(routing) - 1):
        carried = basin[j + 3] if j + 2 < len(routing) else 0.0
        basin[j + 2] = carried + routing[j + 1] * generated
    basin[0] = upper - q0 - q1
    basin[1] = lower - q2
    return flux, np.array([outlet, generated, d, q0, q1, q2])
