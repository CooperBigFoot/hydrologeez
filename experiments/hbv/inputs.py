"""Seeded resolved inputs, not a configuration parser or calibration sampler."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .model import Array, weights


@dataclass
class Inputs:
    forcing: Array
    area: Array
    parameters: Array
    response: Array
    routing: Array
    basin_index: NDArray[np.int64]
    forcing_index: NDArray[np.int64]
    candidate: NDArray[np.int64]
    zone: Array
    basin: Array
    sharing: NDArray[np.int64]
    fixed: NDArray[np.bool_]


def synthetic(basins: int, candidates: int, days: int, seed: int = 20261027) -> Inputs:
    rng = np.random.default_rng(seed)
    zones = 4
    counts = 1 + np.arange(basins) % zones
    area = rng.uniform(0.1, 1.0, (basins, zones))
    area[np.arange(zones)[None, :] >= counts[:, None]] = 0.0
    area /= area.sum(axis=1, keepdims=True)
    forcing = np.empty((basins, days, zones, 3))
    forcing[..., 0] = rng.gamma(0.7, 6.0, (basins, days, zones)) * (rng.random((basins, days, zones)) < 0.45)
    seasonal = np.sin(2 * np.pi * np.arange(days) / 365.0)
    forcing[..., 1] = 8 * seasonal[None, :, None] + rng.normal(1.0, 3.0, (basins, days, zones)) - np.arange(zones)
    forcing[..., 2] = np.broadcast_to((1.5 + seasonal)[None, :, None], (basins, days, zones))
    index = np.repeat(np.arange(basins), candidates)
    candidate = np.tile(np.arange(candidates), basins)
    n = len(index)
    # Basin, vegetation, elevation, zone sharing can differ for each parameter.
    sharing = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 2, 3], [0, 0, 1, 1], [0, 1, 0, 1]]
    )
    low = np.array([-1.0, 1.0, 0.02, 0.02, 0.8, 40.0, 1.0, 0.3])
    high = np.array([1.0, 5.0, 0.2, 0.2, 1.2, 250.0, 3.0, 1.0])
    values = rng.uniform(low, high, (n, zones, 8))
    parameters = np.empty_like(values)
    for p in range(8):
        parameters[:, :, p] = values[:, sharing[p], p]
    response = rng.uniform([0.0, 1.0, 0.05, 0.02, 0.005], [3.0, 20.0, 0.8, 0.4, 0.1], (n, 5))
    duration = rng.uniform(1.0, 4.0, n)
    fixed = np.zeros((basins, 14), dtype=np.bool_)
    fixed[1::3, [0, 4, 12]] = True
    fixed[2::3] = True
    # Explicit resolved fixed values from each basin's first candidate, including all-fixed basins.
    for b in range(basins):
        rows = slice(b * candidates, (b + 1) * candidates)
        for p in range(8):
            if fixed[b, p]:
                parameters[rows, :, p] = parameters[b * candidates, :, p]
        for p in range(5):
            if fixed[b, p + 8]:
                response[rows, p] = response[b * candidates, p]
        if fixed[b, 13]:
            duration[rows] = duration[b * candidates]
    routing = np.zeros((n, 4))
    for i, duration_i in enumerate(duration):
        w = weights(float(duration_i))
        routing[i, : len(w)] = w
    zone = np.zeros((n, zones, 3))
    zone[:, :, 0] = 5.0
    zone[:, :, 2] = 0.4 * parameters[:, :, 5]
    basin = np.zeros((n, 5))
    basin[:, :2] = [3.0, 10.0]
    return Inputs(
        forcing, area, parameters, response, routing, index, index.copy(), candidate, zone, basin, sharing, fixed
    )


def expanded_forcing(data: Inputs) -> Inputs:
    """Materialize candidate copies to measure reuse versus contiguous simulation access."""
    return replace(
        data,
        forcing=data.forcing[data.forcing_index],
        area=data.area[data.forcing_index],
        forcing_index=np.arange(len(data.basin_index)),
    )
