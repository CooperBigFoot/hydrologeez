"""Compiled sequential time, serial or parallel over independent simulations."""

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .inputs import Inputs
from .model import Array, day


@dataclass
class Result:
    prediction: Array
    zone: Array
    basin: Array
    zone_history: Array
    basin_history: Array
    zone_flux: Array
    basin_flux: Array


def _ensemble(
    forcing: Array,
    area: Array,
    parameters: Array,
    response: Array,
    routing: Array,
    index: NDArray[np.int64],
    zone: Array,
    basin: Array,
    start: int,
    stop: int,
    history: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    n, z, _ = zone.shape
    days = stop - start
    prediction = np.empty((n, days))
    recorded = days if history else 0
    zh = np.empty((n, recorded + 1 if history else 0, z, 3))
    bh = np.empty((n, recorded + 1 if history else 0, basin.shape[1]))
    zf = np.empty((n, recorded, z, 8))
    bf = np.empty((n, recorded, 6))
    for i in prange(n):
        b = index[i]
        if history:
            zh[i, 0] = zone[i]
            bh[i, 0] = basin[i]
        for t in range(days):
            local, shared = day(
                forcing[b, start + t], area[b], parameters[i], response[i], routing[i], zone[i], basin[i]
            )
            prediction[i, t] = shared[0]
            if history:
                zh[i, t + 1] = zone[i]
                bh[i, t + 1] = basin[i]
                zf[i, t] = local
                bf[i, t] = shared
    return prediction, zh, bh, zf, bf


serial = njit(cache=True)(_ensemble)
parallel_ensemble = njit(parallel=True)(_ensemble)


def simulate(
    data: Inputs,
    *,
    parallel: bool = False,
    history: bool = False,
    start: int = 0,
    stop: int | None = None,
    initial: tuple[Array, Array] | None = None,
) -> Result:
    zone, basin = (data.zone, data.basin) if initial is None else initial
    zone, basin = zone.copy(), basin.copy()
    kernel = parallel_ensemble if parallel else serial
    q, zh, bh, zf, bf = kernel(
        data.forcing,
        data.area,
        data.parameters,
        data.response,
        data.routing,
        data.basin_index,
        zone,
        basin,
        start,
        data.forcing.shape[1] if stop is None else stop,
        history,
    )
    return Result(q, zone, basin, zh, bh, zf, bf)
