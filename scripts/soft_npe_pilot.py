"""Measure full-period objective costs without launching a calibration campaign.

Run from the checkout: uv run --extra hybrid python -m scripts.soft_npe_pilot
"""

# ruff: noqa: E402
# Thread limits must precede numerical imports in spawned workers.
import os

# Set before numerical imports and before spawning workers.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numba
import numpy as np
import torch

from hydrologeez.calibration import SEARCH_BOUNDS, CandidateEvaluator, Period
from hydrologeez.camels import Camels
from hydrologeez.losses import soft_npe_loss
from hydrologeez.metrics import evaluate
from scripts.soft_npe_benchmark import SoftNPE

SOFTNESSES = (0.01, 0.03, 0.1, 0.3, 1.0)


def measure(task: tuple[str, str, int]) -> dict[str, Any]:
    store_path, basin, candidates = task
    numba.set_num_threads(1)
    torch.set_num_threads(1)
    selected = Camels(Path(store_path)).select([basin], "1987-10-01", "1999-10-01")
    evaluator = CandidateEvaluator(
        selected.forcing, selected.dates, selected.discharge_mm_day[:, 0], 0, Period("1989-10-01", "1999-10-01")
    )
    mask = evaluator.masks["training"]
    observed = evaluator.observed[mask]
    rng = np.random.default_rng(42)
    parameters = rng.uniform(SEARCH_BOUNDS[:, 0], SEARCH_BOUNDS[:, 1], (candidates, 13))
    evaluator.discharge(parameters[:1])  # Exclude HBV compilation from steady-state cost.
    start = perf_counter()
    discharge = evaluator.discharge(parameters)[mask]
    simulation_seconds = (perf_counter() - start) / candidates
    objective = SoftNPE(observed, 0.1)
    objective(discharge[:, 0].copy())  # Exclude rank compilation.
    start = perf_counter()
    exact = [evaluate(observed, predicted)["kge_np"] for predicted in discharge.T]
    exact_seconds = (perf_counter() - start) / candidates
    records = []
    for softness in SOFTNESSES:
        objective = SoftNPE(observed, softness)
        start = perf_counter()
        losses = [objective(predicted.copy()) for predicted in discharge.T]
        seconds = (perf_counter() - start) / candidates
        # Reference check and memory peak include the original quadratic tensor implementation.
        start = perf_counter()
        reference = soft_npe_loss(
            torch.tensor(discharge[:, :1]),
            torch.tensor(observed[:, None]),
            torch.ones((len(observed), 1), dtype=torch.bool),
            softness=softness,
        ).item()
        torch_seconds = perf_counter() - start
        np.testing.assert_allclose(losses[0], reference, rtol=0, atol=2e-12)
        records.append(
            {
                "softness": softness,
                "seconds_per_candidate": seconds,
                "torch_seconds_one_candidate": torch_seconds,
                "reference_absolute_error": abs(losses[0] - reference),
                "invalid_candidates": sum(not np.isfinite(loss) for loss in losses),
            }
        )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "basin": basin,
        "scored_days": len(observed),
        "simulation_days": len(selected.dates),
        "candidate_seed": 42,
        "candidates": candidates,
        "simulation_seconds_per_candidate": simulation_seconds,
        "exact_seconds_per_candidate": exact_seconds,
        "soft": records,
        "invalid_exact_candidates": sum(not np.isfinite(score) for score in exact),
        "process_peak_rss_bytes": peak if platform.system() == "Darwin" else peak * 1024,
        "numba_threads": numba.get_num_threads(),
        "torch_threads": torch.get_num_threads(),
        "input_sha256": hashlib.sha256(
            selected.forcing.values.tobytes() + selected.discharge_mm_day.tobytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/camels-us/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basins", nargs="+", default=["01013500", "09480000", "14301000"])
    parser.add_argument("--candidates", type=int, default=10)
    args = parser.parse_args()
    tasks = [(str(args.store.resolve()), basin, args.candidates) for basin in args.basins]
    # Repeating basins measures six concurrent workers; not six independent scientific observations.
    tasks = [tasks[index % len(tasks)] for index in range(6)]
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(measure, tasks))
    seconds = perf_counter() - start
    mean_simulation = float(np.mean([row["simulation_seconds_per_candidate"] for row in records]))
    mean_exact = float(np.mean([row["exact_seconds_per_candidate"] for row in records]))
    mean_soft = [float(np.mean([row["soft"][i]["seconds_per_candidate"] for row in records])) for i in range(5)]
    report = {
        "purpose": "Full-period six-worker feasibility pilot; no GA trials or campaign results",
        "workers": 6,
        "internal_threads": 1,
        "elapsed_seconds": seconds,
        "basin_selection": "Explicit contrasting pilot basins, not the random campaign sample",
        "records": records,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "numba", "scipy", "torch", "hydrologeez")
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("soft_npe_benchmark.py"), Path("uv.lock"))
        },
        "projected_six_worker_hours_excluding_ga_overhead": 1_750_000
        * (6 * mean_simulation + mean_exact + sum(mean_soft))
        / 6
        / 3600,
        "projection_assumptions": "Equal task sharing; pilot basin costs representative; excludes GA overhead, startup and retries",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
