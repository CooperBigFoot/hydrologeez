"""Named experiments and portable seasonal result bundles."""

import csv
import json
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .fit import Candidate, fit_candidates, per_basin_scale, pooled_scale
from .seasonal import CALENDAR_POLICY, SeasonalCycles
from .selection import ElbowSettings, Selection, select_count, validate_selection


@dataclass(frozen=True)
class Experiment:
    directory: Path
    cycles: SeasonalCycles
    candidates: dict[int, Candidate]
    selected_count: int | None
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    coordinates: NDArray[np.float64] | None
    metadata: dict[str, Any]

    @property
    def selection(self) -> str:
        return self.metadata.get("selection_method", "fixed" if self.metadata["mode"] == "fixed" else "silhouette")

    @property
    def elbow(self) -> ElbowSettings | None:
        settings = self.metadata.get("elbow")
        return ElbowSettings(**settings) if settings is not None else None

    @property
    def selection_reason(self) -> str:
        return self.metadata.get("selection_reason", self.metadata["selection"])

    def select(self, *, selection: Selection = "silhouette", elbow: ElbowSettings | None = None) -> "Experiment":
        """Return an in-memory selection view without fitting or changing the bundle."""
        selected, reason = select_count(
            {k: fit.silhouette for k, fit in self.candidates.items()},
            {k: fit.inertia for k, fit in self.candidates.items()},
            selection,
            elbow,
        )
        metadata = {
            **self.metadata,
            "selected_count": selected,
            "selection_method": selection,
            "selection_reason": reason,
            "selection": reason,
            "elbow": asdict(elbow or ElbowSettings()) if selection == "elbow" else None,
        }
        return replace(self, selected_count=selected, metadata=metadata)

    def _selected(self) -> Candidate:
        if self.selected_count is None:
            raise ValueError("No selected partition; inspect candidates or explicitly select a retained count")
        return self.candidates[self.selected_count]

    @property
    def labels(self) -> NDArray[np.int64]:
        return self._selected().labels

    @property
    def normalization(self) -> str:
        return self.metadata.get("normalization", "pooled")

    @property
    def normalized_members(self) -> NDArray[np.float64]:
        return (self.cycles.values - self.mean) / self.std

    @property
    def centroid_units(self) -> tuple[str, ...]:
        return (
            self.cycles.units
            if self.normalization == "pooled"
            else ("standardized (dimensionless)",) * len(self.cycles.variables)
        )

    @property
    def centroids(self) -> NDArray[np.float64]:
        fitted = self._selected().centroids
        return fitted * self.std + self.mean if self.normalization == "pooled" else fitted

    @property
    def clusters(self) -> dict[int, list[str]]:
        return {
            k: [g for g, label in zip(self.cycles.ids, self.labels, strict=True) if label == k]
            for k in range(self._selected().count)
        }

    def plot(self, directory: Path | None = None, *, rows: int | None = None, columns: int | None = None) -> None:
        from .plotting import render_diagnostics, render_figures

        output = directory or self.directory / "figures"
        if self.metadata["mode"] == "search" or self.selection in ("elbow", "silhouette"):
            render_diagnostics(
                output,
                {k: fit.silhouette for k, fit in self.candidates.items()},
                {k: fit.inertia for k, fit in self.candidates.items()},
                self.selected_count if self.selection == "elbow" else None,
            )
        if self.selected_count is None:
            return
        if self.coordinates is None:
            raise ValueError("Gauge coordinates are required for the map; supply basin-ordered longitude/latitude")
        render_figures(
            output,
            self.cycles.values if self.normalization == "pooled" else self.normalized_members,
            self.centroids,
            self.labels,
            self.cycles.variables,
            self.centroid_units,
            self.coordinates,
            {k: fit.silhouette for k, fit in self.candidates.items()},
            False,
            rows=rows,
            columns=columns,
        )


def run_experiment(
    cycles: SeasonalCycles,
    output_dir: Path,
    experiment: str,
    *,
    k: int | None = None,
    candidates: tuple[int, ...] | None = None,
    selection: Selection = "silhouette",
    elbow: ElbowSettings | None = None,
    seed: int = 0,
    n_init: int = 3,
    max_iter: int = 50,
    tolerance: float = 1e-6,
    coordinates: NDArray[np.float64] | None = None,
    figures: bool = True,
    normalization: Literal["pooled", "per_basin"] = "pooled",
    rows: int | None = None,
    columns: int | None = None,
) -> Experiment:
    """Fit one count or an explicit search. Coordinates never enter the distance.

    Search ties choose the smaller count. Undefined silhouettes are null and
    cannot win. Fit failure rejects the entire experiment, without exclusions.
    """
    if not experiment or Path(experiment).name != experiment or experiment in (".", ".."):
        raise ValueError("Experiment must be a single nonempty directory name")
    directory = Path(output_dir) / experiment
    if directory.exists():
        raise FileExistsError(directory)
    if (k is None) == (candidates is None):
        raise ValueError("Specify exactly one of k or candidates")
    validate_selection(selection, elbow)
    if k is not None and (selection != "silhouette" or elbow is not None):
        raise ValueError("Automatic selection settings are incompatible with fixed k")
    if normalization not in ("pooled", "per_basin"):
        raise ValueError("normalization must be pooled or per_basin")
    values = cycles.values
    if (
        values.shape != (len(cycles.ids), 52, len(cycles.variables))
        or not np.isfinite(values).all()
        or not cycles.ids
        or len(set(cycles.ids)) != len(cycles.ids)
        or any(not isinstance(g, str) or not g for g in cycles.ids)
        or not cycles.variables
        or len(set(cycles.variables)) != len(cycles.variables)
        or len(cycles.units) != len(cycles.variables)
    ):
        raise ValueError("Invalid seasonal curves or axis metadata")
    if coordinates is not None and (coordinates.shape != (len(cycles.ids), 2) or not np.isfinite(coordinates).all()):
        raise ValueError("Coordinates must be finite basin-ordered longitude/latitude pairs")
    if figures and coordinates is None and selection != "elbow":
        raise ValueError("Coordinates required for figures; set figures=False for numerical outputs only")
    scaled, mean, std = pooled_scale(values) if normalization == "pooled" else per_basin_scale(values)
    counts = (k,) if k is not None else tuple(sorted(candidates or ()))
    fitted = fit_candidates(scaled, counts, seed=seed, n_init=n_init, max_iter=max_iter, tolerance=tolerance, radius=2)
    selected, reason = (
        (k, "Explicit fixed count")
        if k is not None
        else select_count(
            {count: fit.silhouette for count, fit in fitted.items()},
            {count: fit.inertia for count, fit in fitted.items()},
            selection,
            elbow,
        )
    )
    if figures and selected is not None and coordinates is None:
        raise ValueError("Coordinates required for selected partition figures; set figures=False")
    metadata: dict[str, Any] = {
        "format": 3,
        "selection_method": "fixed" if k is not None else selection,
        "selection_reason": reason,
        "elbow": asdict(elbow or ElbowSettings()) if selection == "elbow" else None,
        "normalization": normalization,
        "stored_centroids": "standardized",
        "mode": "fixed" if k is not None else "search",
        "selected_count": selected,
        "counts": list(counts),
        "ids": list(cycles.ids),
        "variables": list(cycles.variables),
        "units": list(cycles.units),
        "axes": ["basin", "seasonal_step", "variable"],
        "cluster_labels": "zero-based; centroid axis matches label",
        "seasonal_steps": list(range(52)),
        "start": cycles.start,
        "stop": cycles.stop,
        "calendar_policy": CALENDAR_POLICY,
        "provenance": cycles.provenance,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "seed": seed,
        "n_init": n_init,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "initialization": "k-means++",
        "distance": "dependent multivariate DTW; sqrt summed squared Euclidean local costs",
        "sakoe_chiba_radius": 2,
        "boundary": "anchored endpoints; no wrapping",
        "channel_weights": "equal after scaling",
        "barycenter": {
            "method": "tslearn Petitjean DBA",
            "max_iter": 30,
            "tolerance": 1e-5,
            "convergence": "not exposed by library",
        },
        "selection": reason,
        "exclusions": [],
        "versions": {name: version(name) for name in ("hydrologeez", "numpy", "tslearn", "scikit-learn", "kneed")},
        "fits": {
            str(count): {key: value for key, value in asdict(fit).items() if key not in ("labels", "centroids")}
            for count, fit in fitted.items()
        },
    }
    result = Experiment(directory, cycles, fitted, selected, mean, std, coordinates, metadata)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {"members": values}
    if selected is not None:
        with (directory / "membership.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("basin_id", "cluster"))
            writer.writerows(zip(cycles.ids, result.labels, strict=True))
        (directory / "clusters.json").write_text(json.dumps(result.clusters, indent=2) + "\n")
        arrays.update(centroids=fitted[selected].centroids, labels=result.labels)
    if coordinates is not None:
        arrays["coordinates"] = coordinates
    for count, fit in fitted.items():
        arrays[f"centroids_{count}"] = fit.centroids
        arrays[f"labels_{count}"] = fit.labels
    np.savez_compressed(directory / "seasonal_cycles.npz", allow_pickle=False, **arrays)
    with (directory / "diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("count", "silhouette", "inertia", "converged", "iterations", "attempts", "successful_starts"))
        for count, fit in fitted.items():
            writer.writerow(
                (
                    count,
                    fit.silhouette,
                    fit.inertia,
                    fit.converged,
                    len(fit.history),
                    fit.attempts,
                    fit.successful_starts,
                )
            )
    (directory / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if figures:
        result.plot(rows=rows, columns=columns)
    return result


def load_experiment(directory: Path, *, count: int | None = None) -> Experiment:
    """Reload or select any previously evaluated count without refitting."""
    metadata: dict[str, Any] = json.loads((directory / "run.json").read_text())
    if metadata["format"] not in (1, 2, 3):
        raise ValueError("Unsupported experiment format")
    selected = metadata["selected_count"] if count is None else count
    if selected is not None and selected not in metadata["counts"]:
        raise ValueError("Requested count was not evaluated")
    if count is not None:
        metadata = {
            **metadata,
            "selected_count": count,
            "selection_method": "manual",
            "selection_reason": "Explicit retained count",
            "selection": "Explicit retained count",
            "elbow": None,
        }
    mean, std = np.asarray(metadata["mean"]), np.asarray(metadata["std"])
    with np.load(directory / "seasonal_cycles.npz", allow_pickle=False) as arrays:
        cycles = SeasonalCycles(
            tuple(metadata["ids"]),
            tuple(metadata["variables"]),
            tuple(metadata["units"]),
            arrays["members"],
            metadata["start"],
            metadata["stop"],
            metadata["provenance"],
        )
        fits = {}
        for candidate in metadata["counts"]:
            details = metadata["fits"][str(candidate)]
            fits[candidate] = Candidate(
                candidate,
                arrays[f"labels_{candidate}"],
                (arrays[f"centroids_{candidate}"] - mean) / std
                if metadata["format"] == 1
                else arrays[f"centroids_{candidate}"],
                details["silhouette"],
                details["inertia"],
                tuple(details["history"]),
                details["converged"],
                details["attempts"],
                details["successful_starts"],
                tuple(details["warnings"]),
            )
        coordinates = arrays.get("coordinates", None)
    return Experiment(directory, cycles, fits, selected, mean, std, coordinates, metadata)
