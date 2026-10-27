"""Local Metal experiment: float32 vectorized days, sequential recurrence.

Prepared forcing stays basin-shared. Candidate gathering happens each day.
The caller supplies valid resolved inputs; padding has valid parameters and zero area.
MLX cannot run float64 on Metal: this is a precision comparison, not bitwise equivalence.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from .inputs import Inputs


@dataclass
class DeviceInputs:
    forcing: mx.array
    area: mx.array
    parameters: mx.array
    response: mx.array
    routing: mx.array
    basin_index: mx.array
    zone: mx.array
    basin: mx.array


@dataclass
class Result:
    prediction: mx.array
    zone: mx.array
    basin: mx.array
    zone_history: mx.array | None
    basin_history: mx.array | None
    zone_flux: mx.array | None
    basin_flux: mx.array | None

    @property
    def outlet(self) -> mx.array:
        return self.prediction

    def arrays(self) -> list[mx.array]:
        return [
            a
            for a in (
                self.prediction,
                self.zone,
                self.basin,
                self.zone_history,
                self.basin_history,
                self.zone_flux,
                self.basin_flux,
            )
            if a is not None
        ]


def prepare(
    forcing: NDArray[np.float64],
    area: NDArray[np.float64],
    parameters: NDArray[np.float64],
    response: NDArray[np.float64],
    routing: NDArray[np.float64],
    basin_index: NDArray[np.int64],
    zone: NDArray[np.float64],
    basin: NDArray[np.float64],
) -> DeviceInputs:
    if not mx.metal.is_available():
        raise RuntimeError("This experiment requires a local Metal GPU")
    mx.set_default_device(mx.gpu)
    arrays = [mx.array(a, dtype=mx.float32) for a in (forcing, area, parameters, response, routing)]
    arrays += [
        mx.array(basin_index, dtype=mx.int32),
        mx.array(zone, dtype=mx.float32),
        mx.array(basin, dtype=mx.float32),
    ]
    mx.eval(*arrays)
    mx.synchronize()
    return DeviceInputs(*arrays)


@mx.compile
def day(
    forcing: mx.array,
    area: mx.array,
    parameters: mx.array,
    response: mx.array,
    routing: mx.array,
    zone: mx.array,
    basin: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    p, t, e = (forcing[..., i] for i in range(3))
    tt, cfmax, cfr, cwh, scf, fc, beta, lp = (parameters[..., i] for i in range(8))
    s, liquid, moisture = (zone[..., i] for i in range(3))
    cold = t < tt
    corrected = mx.where(cold, scf * p, p)
    s = s + mx.where(cold, corrected, 0)
    liquid = liquid + mx.where(cold, 0, p)
    melt = mx.minimum(s, cfmax * mx.maximum(t - tt, 0))
    freeze = mx.minimum(liquid, cfr * cfmax * mx.maximum(tt - t, 0))
    s = s + freeze - melt
    liquid = liquid + melt - freeze
    soil_input = mx.maximum(liquid - cwh * s, 0)
    liquid = liquid - soil_input
    raw = soil_input * (moisture / fc) ** beta
    moisture = moisture + soil_input - raw
    overflow = mx.maximum(moisture - fc, 0)
    moisture = moisture - overflow
    recharge = raw + overflow
    evaporation = mx.minimum(moisture, e * mx.minimum(moisture / (lp * fc), 1))
    zone = mx.stack([s, liquid, moisture - evaporation], axis=-1)
    flux = mx.stack([soil_input, recharge, evaporation, corrected, melt, freeze, raw, overflow], axis=-1)
    perc, uzl, k0, k1, k2 = (response[..., i] for i in range(5))
    upper = basin[:, 0] + mx.sum(area * recharge, axis=1)
    d = mx.minimum(perc, upper)
    upper = upper - d
    lower = basin[:, 1] + d
    q0 = k0 * mx.maximum(upper - uzl, 0)
    q1 = k1 * upper
    demand = q0 + q1
    scale = mx.minimum(1, upper / mx.where(demand > 0, demand, 1))
    q0, q1 = q0 * scale, q1 * scale
    q2 = mx.minimum(lower, k2 * lower)
    generated = q0 + q1 + q2
    outlet = routing[:, 0] * generated
    stores = mx.stack([upper - q0 - q1, lower - q2], axis=1)
    if routing.shape[1] > 1:
        outlet = outlet + basin[:, 2]
        shifted = mx.concatenate([basin[:, 3:], mx.zeros_like(basin[:, :1])], axis=1)
        stores = mx.concatenate([stores, shifted + routing[:, 1:] * generated[:, None]], axis=1)
    shared = mx.stack([outlet, generated, d, q0, q1, q2], axis=1)
    return zone, stores, flux, shared


def simulate(
    data: DeviceInputs,
    *,
    history: bool = False,
    start: int = 0,
    stop: int | None = None,
    initial: tuple[mx.array, mx.array] | None = None,
) -> Result:
    """Return evaluated device arrays; history states include x0, then each day end."""
    stop = data.forcing.shape[1] if stop is None else stop
    zone, basin = (data.zone, data.basin) if initial is None else initial
    zh, bh, zf, bf, predictions = [zone], [basin], [], [], []
    area = data.area[data.basin_index]
    for t in range(start, stop):
        zone, basin, local, shared = day(
            data.forcing[:, t][data.basin_index], area, data.parameters, data.response, data.routing, zone, basin
        )
        predictions.append(shared[:, 0])
        if history:
            zh.append(zone)
            bh.append(basin)
            zf.append(local)
            bf.append(shared)
        # Evaluate each day to bound the lazy graph; no host transfer.
        mx.eval(zone, basin, shared, *([local] if history else []))
    q = mx.stack(predictions, axis=1) if predictions else mx.zeros((zone.shape[0], 0))
    result = Result(
        q,
        zone,
        basin,
        mx.stack(zh, axis=1) if history else None,
        mx.stack(bh, axis=1) if history else None,
        mx.stack(zf, axis=1) if history and zf else None,
        mx.stack(bf, axis=1) if history and bf else None,
    )
    mx.eval(*result.arrays())
    mx.synchronize()
    return result


def benchmark(data: Inputs, *, history: bool = False, repeats: int = 3) -> dict[str, Any]:
    """JSON-ready synchronized GPU versus compiled serial CPU evidence.

    Host copies are timed separately. Apple unified memory does not imply free
    dtype conversion, allocation, GPU synchronization, or output materialization.
    CPU startup includes any Numba compilation/cache load; GPU startup includes
    first-evaluation compilation. Process/import startup is outside this helper.
    """
    from numba import get_num_threads

    from .execution import simulate as cpu_simulate

    start = perf_counter()
    device = prepare(
        data.forcing, data.area, data.parameters, data.response, data.routing, data.basin_index, data.zone, data.basin
    )
    preparation = perf_counter() - start
    start = perf_counter()
    cpu = cpu_simulate(data, history=history)
    cpu_startup = perf_counter() - start
    start = perf_counter()
    cpu_simulate(data, history=history, parallel=True)
    parallel_startup = perf_counter() - start
    metal_baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    start = perf_counter()
    result = simulate(device, history=history)
    gpu_startup = perf_counter() - start
    gpu_times: list[float] = []
    cpu_times: list[float] = []
    parallel_times: list[float] = []
    for _ in range(repeats):
        del result
        start = perf_counter()
        result = simulate(device, history=history)
        gpu_times.append(perf_counter() - start)
        start = perf_counter()
        cpu = cpu_simulate(data, history=history)
        cpu_times.append(perf_counter() - start)
        start = perf_counter()
        cpu_simulate(data, history=history, parallel=True)
        parallel_times.append(perf_counter() - start)
    start = perf_counter()
    fields = ["prediction", "zone", "basin"]
    if history:
        fields += ["zone_history", "basin_history", "zone_flux", "basin_flux"]
    host = {name: np.array(getattr(result, name)) for name in fields}
    materialization = perf_counter() - start
    errors = {}
    for name, value in host.items():
        reference = getattr(cpu, name)
        difference = np.abs(value - reference)
        errors[name] = {
            "max_absolute_mm": float(difference.max(initial=0)),
            "rmse_mm": float(np.sqrt(np.mean(difference**2))),
            "max_tolerance_ratio_atol_0.0002_rtol_0.00002": float(
                np.max(difference / (2e-4 + 2e-5 * np.abs(reference)), initial=0)
            ),
            "max_relative_above_1mm": float(np.max(difference / np.maximum(np.abs(reference), 1.0), initial=0)),
        }
    count = data.zone.shape[0] * data.forcing.shape[1]
    return {
        "backend": "MLX Metal GPU",
        "cpu_parallel_threads": get_num_threads(),
        "precision": "float32 vs CPU float64",
        "mlx_device": mx.device_info(),
        "history": history,
        "shape": {
            "basins": data.forcing.shape[0],
            "simulations": data.zone.shape[0],
            "days": data.forcing.shape[1],
            "padded_zones": data.zone.shape[1],
        },
        "prepare_convert_transfer_sync_seconds": preparation,
        "first_gpu_evaluation_seconds": gpu_startup,
        "first_cpu_evaluation_seconds": cpu_startup,
        "gpu_repeated_seconds": gpu_times,
        "cpu_serial_repeated_seconds": cpu_times,
        "cpu_parallel_repeated_seconds": parallel_times,
        "first_cpu_parallel_evaluation_seconds": parallel_startup,
        "gpu_simulation_days_per_second": count / float(np.median(gpu_times)),
        "host_output_materialization_seconds": materialization,
        "metal_peak_allocated_bytes": mx.get_peak_memory(),
        "metal_prepared_baseline_bytes": metal_baseline,
        "metal_incremental_peak_bytes": mx.get_peak_memory() - metal_baseline,
        "errors": errors,
        "notes": [
            "Per-day graph evaluation and final synchronization are included.",
            "Metal float64 unsupported; threshold-near float32 rounding can change branches.",
            "Peak is Metal allocator only, not whole-process RSS; incremental excludes prepared inputs.",
            "Previous GPU outputs released before repeated evaluation; host outputs materialized after timing.",
            "Tolerance ratio <=1 means abs(error)<=2e-4mm + 2e-5*abs(reference).",
            "Histories include initial memory, end-of-day memory, and all recorded fluxes.",
            "No disk writes, process startup, or synthetic generation timed here.",
            "Compiled elementwise graph; no custom fused Metal time loop attempted.",
        ],
    }


if __name__ == "__main__":
    import argparse
    import json

    from .inputs import synthetic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basins", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    inputs = synthetic(args.basins, args.candidates, args.days)
    generation = perf_counter() - start
    report = benchmark(inputs, history=args.history, repeats=args.repeats)
    report["synthetic_preparation_seconds"] = generation
    print(json.dumps(report, indent=2))
