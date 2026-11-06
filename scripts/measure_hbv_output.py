"""One production-output measurement per fresh process; see docs/hbv-output.md."""

import argparse
import json
import platform
import resource
import time
from datetime import date
from pathlib import Path

import numba
import numpy as np

from hydrologeez.hbv_light import simulate
from hydrologeez.hbv_light.output import VARIABLES, run_sweep
from scripts.hbv_output_writing import camels, parameters_for


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--days", type=int, default=1460)
    parser.add_argument("--copies", type=int, default=10000)
    args = parser.parse_args()
    numba.set_num_threads(1)
    begin = time.perf_counter()
    forcing, ids, dates = camels(args.raw, args.days)
    preparation = time.perf_counter() - begin
    p, m, _ = parameters_for(forcing, 0, 4, args.copies)
    begin = time.perf_counter()
    simulate(forcing, p, m, stop=2, history=True)
    warmup = time.perf_counter() - begin
    begin = time.perf_counter()
    parameters, initial, parameter_ids = parameters_for(forcing, 0, 4 * args.copies, args.copies)
    run_sweep(
        args.output,
        forcing,
        parameters,
        initial,
        basin_ids=ids,
        parameter_ids=[str(i) for i in parameter_ids],
        start=date.fromordinal(int(dates[0])).isoformat(),
        variables=list(VARIABLES),
    )
    elapsed = time.perf_counter() - begin
    files = [p for p in args.output.rglob("*") if p.is_file()]
    # Largest actual tile result, not an RSS bound. Measure after timed writing.
    p, m, _ = parameters_for(forcing, max(0, 4 * args.copies - 256), 4 * args.copies, args.copies)
    result = simulate(forcing, p, m, stop=min(365, args.days), history=True)
    assert result.history is not None
    payload = result.outlet.nbytes + sum(
        getattr(result.history, n).nbytes for n in ("zone", "groundwater", "routing", "zone_flux", "basin_flux")
    )
    evidence = {
        "days": args.days,
        "copies": args.copies,
        "simulations": 4 * args.copies,
        "basins": ids,
        "batch_size": 256,
        "window_size": 365,
        "threads": 1,
        "workers": 1,
        "preparation_seconds": preparation,
        "warmup_seconds": warmup,
        "parameter_hash_simulation_write_inspect_seconds": elapsed,
        "total_excluding_warmup_seconds": preparation + elapsed,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        "stored_bytes": sum(p.stat().st_size for p in files),
        "files": len(files),
        "largest_tile_history_bytes": payload,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
    }
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
