"""Integrated daily triangular weights and water-in-transit schedule."""

import numpy as np
from numba import njit
from numpy.typing import NDArray


def weights(duration: float) -> NDArray[np.float64]:
    """Integrate each daily bin of a triangle with the supplied base in days."""
    edges = np.minimum(np.arange(int(np.ceil(duration)) + 1), duration)
    cdf = np.where(edges <= duration / 2, 2 * (edges / duration) ** 2, 1 - 2 * ((duration - edges) / duration) ** 2)
    return np.diff(cdf)


@njit(cache=True)
def route(generated: float, weights: NDArray[np.float64], schedule: NDArray[np.float64]) -> float:
    """Return today's outlet and mutate the future-arrival schedule (basin mm)."""
    outlet = weights[0] * generated
    if len(schedule):
        outlet += schedule[0]
    for j in range(len(schedule)):
        carried = schedule[j + 1] if j + 1 < len(schedule) else 0.0
        schedule[j] = carried + weights[j + 1] * generated
    return outlet
