"""Run one bounded workload per process so peak RSS has a meaningful scope."""

import argparse
import json
import platform
import resource
from pathlib import Path
from time import perf_counter

import numba
import numpy as np

from .execution import simulate
from .inputs import expanded_forcing, synthetic
from .recording import record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basins", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--backend", choices=["serial", "parallel", "expanded"], default="serial")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--disk", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    numba.set_num_threads(min(8, numba.get_num_threads()))
    tick = perf_counter()
    data = synthetic(args.basins, args.candidates, args.days)
    preparation = perf_counter() - tick
    tick = perf_counter()
    if args.backend == "expanded":
        data = expanded_forcing(data)
    expansion = perf_counter() - tick
    tick = perf_counter()
    simulate(synthetic(2, 2, 4), parallel=args.backend == "parallel", history=args.history)
    startup = perf_counter() - tick
    if args.disk:
        timings_disk = record(data, args.disk)
        report_disk = {
            "settings": vars(args),
            "requested_outputs": "initial and full daily states, routing, and fluxes",
            "preparation_seconds": preparation,
            "startup_seconds": startup,
            "disk_seconds": timings_disk,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report_disk, indent=2, default=str) + "\n")
        print(json.dumps(report_disk, indent=2, default=str))
        return
    timings = []
    result = None
    for _ in range(3):
        tick = perf_counter()
        result = simulate(data, parallel=args.backend == "parallel", history=args.history)
        timings.append(perf_counter() - tick)
    assert result is not None
    # Separate repeat from reference allocation; same requests on both contenders.
    reference = simulate(data, history=args.history)
    errors = {}
    for name in vars(reference):
        expected = getattr(reference, name)
        actual = getattr(result, name)
        errors[name] = float(np.max(np.abs(actual - expected))) if expected.size else 0.0
    report = {
        "settings": vars(args),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": numba.get_num_threads(),
        "preparation_seconds": preparation,
        "expansion_seconds": expansion,
        "startup_seconds": startup,
        "evaluation_seconds": timings,
        "simulation_days_per_second": args.basins * args.candidates * args.days / float(np.median(timings)),
        "max_absolute_errors": errors,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "forcing_bytes": data.forcing.nbytes,
        "disk_seconds": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
