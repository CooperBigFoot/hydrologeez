"""Bounded soft NPE upper benchmark: freeze, run, then summarize.

Run from checkout with uv run --extra hybrid python -m scripts.soft_npe_campaign.
Plotting additionally needs matplotlib and seaborn (uv run --with matplotlib --with seaborn).
"""

# ruff: noqa: E402
import os

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
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any

import numba
import numpy as np

from hydrologeez.calibration import SEARCH_BOUNDS, CandidateEvaluator, Period
from hydrologeez.calibration._evolution import evolve
from hydrologeez.camels import Camels
from hydrologeez.metrics import evaluate, objective_loss
from scripts.soft_npe_benchmark import Array, SoftNPE

SETTINGS = (None, 0.01, 0.03, 0.1, 0.3, 1.0)
SEEDS = tuple(range(42, 52))
ARCHIVE = Path("/Users/nicolaslazaro/Desktop/uzh-results/2026-11-11-soft-npe-upper-benchmark")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def provenance(store: Path) -> dict[str, Any]:
    sources = [
        *Path("src/hydrologeez").rglob("*.py"),
        Path(__file__),
        Path(__file__).with_name("soft_npe_benchmark.py"),
        Path("uv.lock"),
    ]
    return {
        "sources": {str(p.resolve()): digest(p) for p in sorted(sources)},
        "data": {str(p.relative_to(store)): digest(p) for p in sorted(store.rglob("*")) if p.is_file()},
        "python": platform.python_version(),
        "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
    }


def freeze(output: Path, store: Path, original_manifest: Path, workers: int = 16) -> None:
    path = output / "manifest.json"
    if path.exists():
        raise ValueError("Manifest already exists; preserve the frozen experiment")
    pool = sorted(json.loads(original_manifest.read_text())["design"]["ids"])
    if len(pool) != 671 or set(pool) != set(Camels(store).ids):
        raise ValueError("Expected archived 671-basin pool")
    ids = np.random.default_rng(42).choice(pool, 50, replace=False).tolist()
    output.mkdir(parents=True, exist_ok=True)
    (output / "trials").mkdir(exist_ok=True)
    record = {
        "ids": ids,
        "selection_seed": 42,
        "selection_rng": "numpy.default_rng PCG64",
        "pool": pool,
        "settings": SETTINGS,
        "seeds": SEEDS,
        "population": 50,
        "generations": 69,
        "bounds": SEARCH_BOUNDS.tolist(),
        "workers": workers,
        "internal_threads": 1,
        "simulation": ["1987-10-01", "1999-10-01"],
        "scoring": ["1989-10-01", "1999-10-01"],
        "store": str(store.resolve()),
        "archive_manifest_sha256": digest(original_manifest),
        "provenance": provenance(store),
    }
    write_json(path, record)
    for source in (Path(__file__), Path(__file__).with_name("soft_npe_benchmark.py")):
        (output / source.name).write_bytes(source.read_bytes())


def select_best(trials: list[dict[str, Any]]) -> dict[str, Any]:
    return min(trials, key=lambda record: record["loss"])


def validate_complete(trials: list[dict[str, Any]]) -> None:
    if (
        len(trials) != 10
        or {t["seed"] for t in trials} != set(SEEDS)
        or any(
            t["status"] != "success"
            or t["evaluations"] != 3500
            or not np.isfinite(t["loss"])
            or not np.isfinite(t["exact_npe"])
            for t in trials
        )
    ):
        raise ValueError("incomplete equal-budget basin/setting comparison")


def trial(task: tuple[str, str, int, int, str]) -> dict[str, Any]:
    store, basin, setting, seed, identity = task
    numba.set_num_threads(1)
    start = perf_counter()
    evaluations = 0
    invalid = 0
    record: dict[str, Any] = {
        "basin": basin,
        "setting": setting,
        "softness": SETTINGS[setting],
        "seed": seed,
        "manifest_sha256": identity,
    }
    try:
        selected = Camels(Path(store)).select([basin], "1987-10-01", "1999-10-01")
        evaluator = CandidateEvaluator(
            selected.forcing, selected.dates, selected.discharge_mm_day[:, 0], 0, Period("1989-10-01", "1999-10-01")
        )
        mask = evaluator.masks["training"]
        observed = evaluator.observed[mask]
        softness = SETTINGS[setting]
        soft = None if softness is None else SoftNPE(observed, softness)

        def objective(parameters: Array) -> Array:
            nonlocal evaluations, invalid
            predictions = evaluator.discharge(parameters)[mask]
            losses = np.array(
                [
                    objective_loss("kge_np", evaluate(observed, predicted)["kge_np"])
                    if soft is None
                    else soft(predicted.copy())
                    for predicted in predictions.T
                ]
            )
            evaluations += len(parameters)
            invalid += int((~np.isfinite(losses)).sum())
            return losses[:, None]

        population = evolve(objective, SEARCH_BOUNDS, 50, 69, seed)
        index = int(np.argmin(population.losses[:, 0]))
        loss = float(population.losses[index, 0])
        parameters = population.parameters[index]
        exact = evaluate(observed, evaluator.discharge(parameters[None])[mask, 0])["kge_np"]
        if not np.isfinite(loss) or not np.isfinite(exact):
            raise ValueError("No valid optimum")
        record.update(
            status="success", loss=loss, exact_npe=exact, parameters=parameters.tolist(), scored_days=int(mask.sum())
        )
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
    record.update(evaluations=evaluations, invalid_candidates=invalid, elapsed_seconds=perf_counter() - start)
    return record


def run(output: Path) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = digest(manifest_path)
    if provenance(Path(manifest["store"])) != manifest["provenance"]:
        raise ValueError("Source, environment or data changed since freeze")
    workers = int(manifest["workers"])
    if not 1 <= workers <= (os.cpu_count() or 1):
        raise ValueError("Worker count exceeds available CPUs")
    tasks = []
    for basin in manifest["ids"]:
        for setting in range(6):
            for seed in SEEDS:
                path = output / "trials" / f"{basin}-{setting}-{seed}.json"
                if path.exists():
                    existing = json.loads(path.read_text())
                    if existing["manifest_sha256"] != identity:
                        raise ValueError("Mismatched checkpoint provenance")
                    # Failures remain evidence, not silently retried or counted as successes.
                    continue
                tasks.append((manifest["store"], basin, setting, seed, identity))
    start = perf_counter()
    receipt = output / f"execution-{time_ns()}.json"
    write_json(receipt, {"status": "running", "pending_trials": len(tasks), "workers": workers})
    completed = failed = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        # Python 3.14 has buffersize; bounded submission here also supports project Python 3.13.
        for offset in range(0, len(tasks), workers):
            for result in pool.map(trial, tasks[offset : offset + workers]):
                path = output / "trials" / f"{result['basin']}-{result['setting']}-{result['seed']}.json"
                write_json(path, result)
                completed += 1
                failed += result["status"] != "success"
                print(f"{completed}/{len(tasks)} {path.name} {result['status']}", flush=True)
    write_json(
        receipt,
        {
            "status": "finished",
            "attempted": completed,
            "failed": failed,
            "elapsed_seconds": perf_counter() - start,
            "workers": workers,
        },
    )


def summarize(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    identity = digest(output / "manifest.json")
    selected: list[list[dict[str, Any]]] = [[] for _ in SETTINGS]
    coverage = []
    for basin in manifest["ids"]:
        for setting in range(6):
            trials = [json.loads(p.read_text()) for p in sorted((output / "trials").glob(f"{basin}-{setting}-*.json"))]
            if any(t["manifest_sha256"] != identity for t in trials):
                raise ValueError("Mismatched checkpoint provenance")
            coverage.append(
                {
                    "basin": basin,
                    "setting": setting,
                    "attempted": len(trials),
                    "successful": sum(t["status"] == "success" for t in trials),
                    "evaluations": sum(t["evaluations"] for t in trials),
                    "elapsed_seconds": sum(t["elapsed_seconds"] for t in trials),
                }
            )
            try:
                validate_complete(trials)
            except ValueError:
                continue
            selected[setting].append(select_best(trials))
    write_json(output / "coverage.json", {"basin_settings": coverage})
    if any(len(rows) != 50 for rows in selected):
        raise ValueError("incomplete comparison; see coverage.json; no completed figure generated")
    scores = np.array([[row["exact_npe"] for row in rows] for rows in selected])
    statistics = []
    for index, softness in enumerate(SETTINGS):
        differences = scores[index] - scores[0]
        statistics.append(
            {
                "softness": softness,
                "mean": float(scores[index].mean()),
                "median": float(np.median(scores[index])),
                "wins": int((differences > 0).sum()),
                "losses": int((differences < 0).sum()),
                "mean_change": float(differences.mean()),
                "median_change": float(np.median(differences)),
                "paired_changes": differences.tolist(),
            }
        )
    write_json(output / "selected.json", {"settings": selected})
    paired_trials = []
    for setting in range(1, 6):
        differences = []
        for basin in manifest["ids"]:
            for seed in SEEDS:
                baseline = json.loads((output / "trials" / f"{basin}-0-{seed}.json").read_text())
                soft = json.loads((output / "trials" / f"{basin}-{setting}-{seed}.json").read_text())
                differences.append(soft["exact_npe"] - baseline["exact_npe"])
        paired_trials.append(
            {
                "softness": SETTINGS[setting],
                "wins": sum(value > 0 for value in differences),
                "mean_change": float(np.mean(differences)),
                "median_change": float(np.median(differences)),
            }
        )
    write_json(output / "summary.json", {"statistics": statistics, "paired_trials": paired_trials})
    lines = [
        "# Soft NPE upper benchmark",
        "",
        "Completed equal-budget, in-sample comparison: 50 basins, ten paired seeds, 3500 evaluations per trial.",
        "",
        "| Softness | Mean NPE | Median NPE | Basin wins/losses | Mean change | Median change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in statistics:
        lines.append(
            f"| {row['softness']} | {row['mean']:.6f} | {row['median']:.6f} | {row['wins']}/{row['losses']} | {row['mean_change']:.6f} | {row['median_change']:.6f} |"
        )
    lines.extend(
        ["", "Paired-trial differences (descriptive only; repeated trials within a basin are not independent):", ""]
    )
    for row in paired_trials:
        lines.append(
            f"- Softness {row['softness']}: {row['wins']}/500 wins; mean change {row['mean_change']:.6f}; median {row['median_change']:.6f}."
        )
    lines.extend(
        [
            "",
            "Runtime includes simulation, scoring, GA and final diagnostics; startup overhead is recorded separately in execution receipts.",
            "",
        ]
    )
    for setting in range(6):
        rows = [row for row in coverage if row["setting"] == setting]
        lines.append(
            f"- Setting {setting}: {sum(row['elapsed_seconds'] for row in rows) / 3600:.3f} summed worker-hours, {sum(row['attempted'] for row in rows)} attempted trials, {sum(row['successful'] for row in rows)} successes."
        )
    lines.extend(
        [
            "",
            "Gains concern finite-budget search, not the theoretical maximum. Softness settings remain separate; no pooled best-soft score is reported. There is no held-out prediction evaluation. Negative findings on this grid would not rule out other surrogates. Review paired basin changes in summary.json and all trial records before judging consistency.",
            "",
            "Production adoption requires explicit human approval. No adoption is implied by these results.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _, axis = plt.subplots(figsize=(9, 6))
    labels = ["Exact NPE", *[f"Soft NPE\n{s:g}" for s in SETTINGS[1:]]]
    sns.boxplot(data=scores.T, palette=["#999999", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"], ax=axis)
    axis.set_xticks(range(6), labels)
    axis.set_ylabel("NPE")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output / "npe-boxplot.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["freeze", "trial", "run", "summarize"])
    parser.add_argument("--output", type=Path, default=ARCHIVE)
    parser.add_argument("--store", type=Path, default=Path("data/camels-us/prepared"))
    parser.add_argument("--original-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.original_manifest is None:
            parser.error("freeze needs --original-manifest")
        if not 1 <= args.workers <= (os.cpu_count() or 1):
            parser.error("workers must fit available CPUs")
        freeze(args.output, args.store, args.original_manifest, args.workers)
    elif args.action == "trial":
        args.output.mkdir(parents=True, exist_ok=True)
        for setting in (0, 3):
            result = trial((str(args.store.resolve()), "01013500", setting, 42, "pilot-only"))
            write_json(args.output / f"full-trial-pilot-{setting}.json", result)
            print(json.dumps(result), flush=True)
    elif args.action == "run":
        run(args.output)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
