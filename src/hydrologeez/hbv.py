"""Daily HBV rules from docs/hbv-light.md, in mm and degrees Celsius.

Arrays use float64. Zones are packed in basin order, with no inactive padding.
Simulation rows remain in caller order (each row identifies one basin/parameter
set); zones within a simulation follow that basin's forcing order. Sharing is
resolved per parameter by the caller, never by merging zone memory.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .routines.groundwater import groundwater
from .routines.routing import route, weights
from .routines.snow import snow
from .routines.soil import soil

Array = NDArray[np.float64]


@njit(cache=True)
def _day(
    forcing: Array, area: Array, parameters: Array, response: Array, routing: Array, zone: Array, basin: Array
) -> tuple[Array, Array]:
    """Mutate end-of-day memory; report zone I,R,A,P*,m,f,r,o and basin Q,G,D,q0,q1,q2."""
    flux = np.empty((len(area), 8))
    recharge = 0.0
    for z in range(len(area)):
        p, t, e = forcing[z]
        tt, cfmax, cfr, cwh, scf, fc, beta, lp = parameters[z]
        s, liquid, moisture = zone[z]
        s, liquid, soil_input, corrected, melt, freeze = snow(p, t, s, liquid, tt, cfmax, cfr, cwh, scf)
        moisture, r, evaporation, initial_recharge, overflow = soil(soil_input, e, moisture, fc, beta, lp)
        zone[z] = s, liquid, moisture
        flux[z] = soil_input, r, evaporation, corrected, melt, freeze, initial_recharge, overflow
        recharge += area[z] * r
    perc, uzl, k0, k1, k2 = response
    basin[0], basin[1], generated, d, q0, q1, q2 = groundwater(recharge, basin[0], basin[1], perc, uzl, k0, k1, k2)
    outlet = route(generated, routing, basin[2:])
    return flux, np.array([outlet, generated, d, q0, q1, q2])


@dataclass(frozen=True)
class Forcing:
    """Prepared values [day, packed basin zone, P/T/PET], fractions [zone].

    zone_offsets [basins+1] delimit each basin's zones. All days must represent
    matching daily intervals; date alignment and input preparation are upstream.
    """

    values: Array
    area: Array
    zone_offsets: NDArray[np.int64]


@dataclass(frozen=True)
class Parameters:
    """Resolved constant values, with no defaults or sharing inference.

    basin_index [simulation] selects shared forcing. zone [packed simulation
    zone, 8] contains TT, CFMAX, CFR, CWH, SCF, FC, BETA, LP. groundwater
    [simulation, 5] contains PERC, UZL, K_0, K_1, K_2. maxbas [simulation]
    gives routing duration in days. Simulation zone counts follow basin_index.
    """

    basin_index: NDArray[np.int64]
    zone: Array
    groundwater: Array
    maxbas: Array


@dataclass(frozen=True)
class Memory:
    """Complete memory: zone [packed simulation zone, S/L/M], groundwater
    [simulation, U/V], routing [packed simulation lag]. Each simulation has
    ceil(MAXBAS)-1 routing entries, ordered by next outlet arrival day.
    """

    zone: Array
    groundwater: Array
    routing: Array


@dataclass(frozen=True)
class History:
    """End-of-day memory and daily fluxes; initial memory is NOT a history row.

    Arrays have day first, then the same packed axes as Memory. zone_flux has
    I, R, A, corrected P, melt, refreeze, raw recharge, overflow in its final
    axis. basin_flux has Q, G, D, q0, q1, q2. All amounts are mm; zone quantities
    use zone area, groundwater/routing quantities use basin area.
    """

    zone: Array
    groundwater: Array
    routing: Array
    zone_flux: Array
    basin_flux: Array


@dataclass(frozen=True)
class Result:
    """Outlet [day, simulation], complete final memory and optional histories.

    start/stop identify forcing intervals [start, stop). basin_index and offsets
    preserve associations to input rows; state history row t is after day start+t.
    """

    outlet: Array
    final: Memory
    history: History | None
    start: int
    stop: int
    basin_index: NDArray[np.int64]
    zone_offsets: NDArray[np.int64]
    routing_offsets: NDArray[np.int64]


@njit(parallel=True, cache=True)
def _simulate(
    forcing: Array,
    area: Array,
    forcing_offsets: NDArray[np.int64],
    index: NDArray[np.int64],
    zone_offsets: NDArray[np.int64],
    routing_offsets: NDArray[np.int64],
    parameters: Array,
    response: Array,
    weights: Array,
    weight_offsets: NDArray[np.int64],
    zone: Array,
    groundwater: Array,
    routing: Array,
    start: int,
    stop: int,
    history: bool,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    days = stop - start
    recorded = days if history else 0
    outlet = np.empty((days, len(index)))
    zh = np.empty((recorded, len(zone), 3))
    gh = np.empty((recorded, len(index), 2))
    rh = np.empty((recorded, len(routing)))
    zf = np.empty((recorded, len(zone), 8))
    bf = np.empty((recorded, len(index), 6))
    for i in prange(len(index)):
        b = index[i]
        fs, fe = forcing_offsets[b], forcing_offsets[b + 1]
        zs, ze = zone_offsets[i], zone_offsets[i + 1]
        rs, re = routing_offsets[i], routing_offsets[i + 1]
        basin = np.empty(2 + re - rs)
        basin[:2] = groundwater[i]
        basin[2:] = routing[rs:re]
        for t in range(days):
            local, shared = _day(
                forcing[start + t, fs:fe],
                area[fs:fe],
                parameters[zs:ze],
                response[i],
                weights[weight_offsets[i] : weight_offsets[i + 1]],
                zone[zs:ze],
                basin,
            )
            outlet[t, i] = shared[0]
            if history:
                zh[t, zs:ze] = zone[zs:ze]
                gh[t, i] = basin[:2]
                rh[t, rs:re] = basin[2:]
                zf[t, zs:ze] = local
                bf[t, i] = shared
        groundwater[i] = basin[:2]
        routing[rs:re] = basin[2:]
    return outlet, zh, gh, rh, zf, bf


def _array(name: str, value: Array, shape: tuple[int, ...]) -> None:
    if value.dtype != np.float64 or value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite float64 array with shape {shape}")


def _validate(forcing: Forcing, parameters: Parameters, initial: Memory, start: int, stop: int) -> None:
    offsets = forcing.zone_offsets
    if (
        offsets.dtype != np.int64
        or offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("zone_offsets must be increasing int64 basin boundaries starting at zero")
    if forcing.values.ndim != 3:
        raise ValueError("forcing values must have axes day, zone, P/T/PET")
    days = forcing.values.shape[0]
    zones = int(offsets[-1])
    if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start <= stop <= days:
        raise ValueError("start/stop must delimit available daily intervals")
    if forcing.values.shape != (days, zones, 3):
        raise ValueError("forcing shape must match basin zone offsets")
    _array("forcing", forcing.values[start:stop], (stop - start, zones, 3))
    _array("area", forcing.area, (zones,))
    if (
        np.any(forcing.values[start:stop, :, 0] < 0)
        or np.any(forcing.values[start:stop, :, 2] < 0)
        or np.any(forcing.area <= 0)
    ):
        raise ValueError("P/PET must be nonnegative and zone fractions positive")
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        if not np.isclose(forcing.area[a:b].sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("zone fractions must sum to one within each basin")
    index = parameters.basin_index
    if (
        index.dtype != np.int64
        or index.ndim != 1
        or len(index) == 0
        or np.any(index < 0)
        or np.any(index >= len(offsets) - 1)
    ):
        raise ValueError("basin_index must be nonempty int64 valid basin indices")
    n = len(index)
    z = int(np.diff(offsets)[index].sum())
    _array("zone parameters", parameters.zone, (z, 8))
    _array("groundwater parameters", parameters.groundwater, (n, 5))
    _array("MAXBAS", parameters.maxbas, (n,))
    p = parameters.zone
    if (
        np.any(p[:, 1:4] < 0)
        or np.any(p[:, 2:4] > 1)
        or np.any(p[:, 4:] <= 0)
        or np.any(p[:, 7] > 1)
        or np.any(parameters.groundwater < 0)
        or np.any(parameters.maxbas < 1)
    ):
        raise ValueError("parameters violate HBV physical domains")
    routing_count = int((np.ceil(parameters.maxbas) - 1).sum())
    _array("initial zone", initial.zone, (z, 3))
    _array("initial groundwater", initial.groundwater, (n, 2))
    _array("initial routing", initial.routing, (routing_count,))
    if (
        np.any(initial.zone < 0)
        or np.any(initial.groundwater < 0)
        or np.any(initial.routing < 0)
        or np.any(initial.zone[:, 2] > p[:, 5])
        or np.any(initial.zone[:, 1] > p[:, 3] * initial.zone[:, 0])
    ):
        raise ValueError("initial memory violates storage or snow retention domains")


def simulate(
    forcing: Forcing,
    parameters: Parameters,
    initial: Memory,
    *,
    start: int = 0,
    stop: int | None = None,
    history: bool = False,
) -> Result:
    """Evolve supplied memory without mutating it, parallel across simulations.

    For bounded-memory histories, call on successive start/stop intervals, consume
    each result, and supply its final memory to the next call with the SAME
    parameters. Warm-up is this same evolution, not a separate state shortcut.
    Numba's thread controls apply; no thread count is imposed by the model.
    """
    stop = forcing.values.shape[0] if stop is None else stop
    _validate(forcing, parameters, initial, start, stop)
    zone_offsets = np.concatenate(([0], np.cumsum(np.diff(forcing.zone_offsets)[parameters.basin_index])))
    counts = np.ceil(parameters.maxbas).astype(np.int64)
    routing_offsets = np.concatenate(([0], np.cumsum(counts - 1)))
    weight_offsets = np.concatenate(([0], np.cumsum(counts)))
    routing_weights = np.concatenate([weights(float(b)) for b in parameters.maxbas])
    final = Memory(initial.zone.copy(), initial.groundwater.copy(), initial.routing.copy())
    q, zh, gh, rh, zf, bf = _simulate(
        forcing.values,
        forcing.area,
        forcing.zone_offsets,
        parameters.basin_index,
        zone_offsets,
        routing_offsets,
        parameters.zone,
        parameters.groundwater,
        routing_weights,
        weight_offsets,
        final.zone,
        final.groundwater,
        final.routing,
        start,
        stop,
        history,
    )
    recorded = History(zh, gh, rh, zf, bf) if history else None
    return Result(q, final, recorded, start, stop, parameters.basin_index.copy(), zone_offsets, routing_offsets)
