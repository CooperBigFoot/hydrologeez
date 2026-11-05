"""Local HBV output screening; not a production storage API.

Run one case per fresh process. Orchestrate repeats outside this module so RSS
high-water marks never include another case or the correctness reference.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import resource
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray

from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate
from scripts.hbv_input_path import prepare


def parameters_for(
    forcing: Forcing, start: int, stop: int, copies: int
) -> tuple[Parameters, Memory, NDArray[np.int64]]:
    """Same deterministic library per basin, generated only for the active rows."""
    rows = np.arange(start, stop, dtype=np.int64)
    basins = len(forcing.zone_offsets) - 1
    index = rows % basins
    ids = rows // basins
    x = ids / max(1, copies - 1)
    counts = np.diff(forcing.zone_offsets)[index]
    zone = np.column_stack(
        (-0.5 + x, 1.5 + x, 0.02 + 0.06 * x, 0.05 + 0.1 * x, 0.8 + 0.4 * x, 80 + 120 * x, 1 + x, 0.4 + 0.4 * x)
    )
    zone = np.repeat(zone, counts, axis=0)
    groundwater = np.column_stack((0.5 + x, 0.2 + 0.6 * x, 0.05 + 0.1 * x, 0.02 + 0.06 * x))
    maxbas = 1.0 + 3.5 * x
    parameters = Parameters(index, zone, groundwater, maxbas)
    initial_zone = np.zeros((len(zone), 3))
    initial_zone[:, 2] = zone[:, 5] / 2
    initial = Memory(initial_zone, np.zeros((len(rows), 2)), np.zeros(int((np.ceil(maxbas) - 1).sum())))
    return parameters, initial, ids


def write_run(
    directory: Path,
    forcing: Forcing,
    basin_ids: list[str],
    dates: NDArray[np.int64],
    simulations: int,
    copies: int,
    kind: str,
    history: bool,
    *,
    batch: int = 256,
    window: int = 365,
) -> dict[str, int]:
    """Retain equivalent float64 values in uncompressed or deflated NPZ shards."""
    directory.mkdir(parents=True, exist_ok=False)
    days = len(dates)
    batch = simulations if kind == "resident" else batch
    window = days if kind in ("resident", "batch") else window
    manifest: dict[str, Any] = {"basins": basin_ids, "dates": dates.tolist(), "shards": []}
    largest = 0
    for a in range(0, simulations, batch):
        b = min(a + batch, simulations)
        parameters, memory, ids = parameters_for(forcing, a, b, copies)
        for start in range(0, days, window):
            stop = min(start + window, days)
            result = simulate(forcing, parameters, memory, start=start, stop=stop, history=history)
            arrays: dict[str, NDArray[Any]] = {
                "outlet": result.outlet,
                "basin_index": result.basin_index,
                "parameter_id": ids,
                "zone_offsets": result.zone_offsets,
                "routing_offsets": result.routing_offsets,
                "parameters_zone": parameters.zone,
                "parameters_groundwater": parameters.groundwater,
                "maxbas": parameters.maxbas,
            }
            for name in ("zone", "groundwater", "routing"):
                arrays["final_" + name] = getattr(result.final, name)
            if result.history is not None:
                for name in ("zone", "groundwater", "routing", "zone_flux", "basin_flux"):
                    arrays[name] = getattr(result.history, name)
            largest = max(largest, sum(array.nbytes for array in arrays.values()))
            filename = f"rows-{a}-{b}-days-{start}-{stop}.npz"
            with (directory / filename).open("wb") as output:
                save = np.savez_compressed if kind == "compressed" else np.savez
                save(output, allow_pickle=False, **arrays)
                output.flush()
                os.fsync(output.fileno())
            manifest["shards"].append({"file": filename, "rows": [a, b], "days": [start, stop]})
            memory = result.final
            del arrays, result
    with (directory / "manifest.json").open("w") as output:
        json.dump(manifest, output, separators=(",", ":"))
        output.flush()
        os.fsync(output.fileno())
    return {"largest_array_payload_bytes": largest, "shards": len(manifest["shards"])}


def camels(raw: Path, days: int) -> tuple[Forcing, list[str], NDArray[np.int64]]:
    attributes = sorted(
        csv.DictReader((raw / "camels_topo.txt").read_text().splitlines(), delimiter=";"),
        key=lambda row: row["gauge_id"],
    )
    chosen = [attributes[i] for i in np.linspace(0, len(attributes) - 1, 4, dtype=int)]
    dates = np.arange(date(1990, 1, 1).toordinal(), date(1990, 1, 1).toordinal() + days, dtype=np.int64)
    prepared = []
    with zipfile.ZipFile(raw / "basin_timeseries_v1p2_metForcing_obsFlow.zip") as archive:
        members = {
            Path(name).name[:8]: name
            for name in archive.namelist()
            if "/basin_mean_forcing/daymet/" in name and name.endswith("_lump_cida_forcing_leap.txt")
        }
        for row in chosen:
            text = archive.read(members[row["gauge_id"]]).decode()
            source_dates, values = prepare(np.loadtxt(io.StringIO(text), skiprows=4), float(row["gauge_lat"]))
            mask = (source_dates >= dates[0]) & (source_dates <= dates[-1])
            np.testing.assert_array_equal(source_dates[mask], dates)
            prepared.append(values[mask])
    return (
        Forcing(np.stack(prepared, axis=1), np.ones(4), np.arange(5, dtype=np.int64)),
        [r["gauge_id"] for r in chosen],
        dates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--kind", choices=["resident", "batch", "tile", "compressed"], required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--copies", type=int, default=128)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    numba.set_num_threads(1)
    begin = time.perf_counter()
    forcing, ids, dates = camels(args.raw, args.days)
    preparation = time.perf_counter() - begin
    begin = time.perf_counter()
    p, m, _ = parameters_for(forcing, 0, 4, args.copies)
    simulate(forcing, p, m, stop=2, history=args.history)
    warmup = time.perf_counter() - begin
    begin = time.perf_counter()
    details = write_run(args.output, forcing, ids, dates, 4 * args.copies, args.copies, args.kind, args.history)
    elapsed = time.perf_counter() - begin
    files = list(args.output.iterdir())
    evidence = {
        "kind": args.kind,
        "days": args.days,
        "copies": args.copies,
        "simulations": 4 * args.copies,
        "history": args.history,
        "batch": 256,
        "window": 365,
        "threads": 1,
        "workers": 1,
        "basins": ids,
        "start": "1990-01-01",
        "preparation_seconds": preparation,
        "warmup_seconds": warmup,
        "simulation_write_seconds": elapsed,
        "total_excluding_warmup_seconds": preparation + elapsed,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        "stored_bytes": sum(p.stat().st_size for p in files),
        "files": len(files),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        **details,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
