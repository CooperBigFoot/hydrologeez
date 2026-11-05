"""Euclidean K-means experiments on explicitly selected static attributes."""

import csv
import json
import warnings
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .selection import ElbowSettings, Selection, select_count, validate_selection

Array = NDArray[np.float64]


@dataclass(frozen=True)
class AttributeCandidate:
    count: int
    labels: NDArray[np.int64]
    centroids: Array
    silhouette: float | None
    inertia: float
    iterations: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AttributeExperiment:
    directory: Path
    ids: tuple[str, ...]
    fields: tuple[str, ...]
    values: Array
    mean: Array
    std: Array
    candidates: dict[int, AttributeCandidate]
    selected_count: int | None
    coordinates: Array | None
    metadata: dict[str, Any]

    @property
    def selection(self) -> str:
        return self.metadata["selection_method"]

    @property
    def selection_reason(self) -> str:
        return self.metadata["selection_reason"]

    @property
    def elbow(self) -> ElbowSettings | None:
        settings = self.metadata["elbow"]
        return ElbowSettings(**settings) if settings is not None else None

    def select(
        self, *, selection: Selection = "silhouette", elbow: ElbowSettings | None = None
    ) -> "AttributeExperiment":
        selected, reason = select_count(
            {k: fit.silhouette for k, fit in self.candidates.items()},
            {k: fit.inertia for k, fit in self.candidates.items()},
            selection,
            elbow,
            inertia_name="sum of squared Euclidean distances",
        )
        return replace(
            self,
            selected_count=selected,
            metadata={
                **self.metadata,
                "selected_count": selected,
                "selection_method": selection,
                "selection_reason": reason,
                "elbow": asdict(elbow or ElbowSettings()) if selection == "elbow" else None,
            },
        )

    def plot(self, directory: Path | None = None) -> None:
        from .plotting import render_diagnostics, render_map

        output = directory or self.directory / "figures"
        if self.metadata["mode"] == "search" or self.selection in ("silhouette", "elbow"):
            render_diagnostics(
                output,
                {k: fit.silhouette for k, fit in self.candidates.items()},
                {k: fit.inertia for k, fit in self.candidates.items()},
                self.selected_count if self.selection == "elbow" else None,
                inertia_label="Sum of squared Euclidean distances",
            )
        if self.selected_count is None:
            return
        if self.coordinates is None:
            raise ValueError("Gauge coordinates are required for the map")
        render_map(output, self.labels, self.coordinates, self.selected_count)

    def _selected(self) -> AttributeCandidate:
        if self.selected_count is None:
            raise ValueError("No selected partition; inspect candidates or select a retained count")
        return self.candidates[self.selected_count]

    @property
    def labels(self) -> NDArray[np.int64]:
        return self._selected().labels

    @property
    def normalized_members(self) -> Array:
        return (self.values - self.mean) / self.std

    @property
    def standardized_centroids(self) -> Array:
        return self._selected().centroids

    @property
    def centroids(self) -> Array:
        return self.standardized_centroids * self.std + self.mean

    @property
    def clusters(self) -> dict[int, list[str]]:
        return {
            label: [basin for basin, assigned in zip(self.ids, self.labels, strict=True) if assigned == label]
            for label in range(self._selected().count)
        }


def _attribute_values(table: pl.DataFrame, fields: tuple[str, ...]) -> tuple[tuple[str, ...], Array]:
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Select explicit unique predictor fields")
    if "gauge_id" not in table.columns or table.schema["gauge_id"] != pl.String:
        raise ValueError("gauge_id must contain string basin identities")
    ids = tuple(table["gauge_id"].to_list())
    if not ids or any(not basin for basin in ids) or len(set(ids)) != len(ids):
        raise ValueError("Basin identities must be unique nonempty strings")
    for field in fields:
        if field in ("gauge_id", "gauge_lon", "gauge_lat"):
            raise ValueError(f"Identity and coordinate fields are not predictors: {field}")
        if field not in table.columns:
            raise ValueError(f"Unknown predictor field: {field}")
        if not table.schema[field].is_numeric():
            raise ValueError(f"Selected predictors must be numeric: {field}")
    values = table.select(fields).to_numpy().astype(np.float64)
    invalid = ~np.isfinite(values)
    if invalid.any():
        details = "; ".join(
            f"{field}: {', '.join(ids[row] for row in np.flatnonzero(invalid[:, col]))}"
            for col, field in enumerate(fields)
            if invalid[:, col].any()
        )
        raise ValueError(f"Missing or nonfinite selected predictors — {details}")
    return ids, values


def run_attribute_experiment(
    table: pl.DataFrame,
    output_dir: Path,
    experiment: str,
    *,
    fields: tuple[str, ...],
    k: int | None = None,
    candidates: tuple[int, ...] | None = None,
    selection: Selection = "silhouette",
    elbow: ElbowSettings | None = None,
    seed: int = 0,
    n_init: int = 3,
    max_iter: int = 50,
    tolerance: float = 1e-6,
    coordinates: pl.DataFrame | None = None,
    figures: bool = True,
) -> AttributeExperiment:
    """Fit explicit numeric fields; coordinates are joined by gauge_id for maps only."""
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
    ids, values = _attribute_values(table, fields)
    counts = (k,) if k is not None else tuple(sorted(candidates or ()))
    if (
        not counts
        or len(set(counts)) != len(counts)
        or any(type(count) is not int or not 1 <= count <= len(ids) for count in counts)
    ):
        raise ValueError("Cluster counts must be unique integers between 1 and the basin count")
    if n_init < 1 or max_iter < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive restarts, iteration limit and tolerance required")
    aligned = _coordinates(coordinates, ids) if coordinates is not None else None
    mean, std = values.mean(axis=0), values.std(axis=0)
    if np.any(std == 0):
        raise ValueError(
            f"Zero variance selected fields: {', '.join(field for field, scale in zip(fields, std, strict=True) if scale == 0)}"
        )
    scaled = (values - mean) / std
    fitted: dict[int, AttributeCandidate] = {}
    for count in counts:
        model = KMeans(
            n_clusters=count,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            tol=tolerance,
            random_state=seed,
            algorithm="lloyd",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(scaled)
        labels = np.asarray(model.labels_, dtype=np.int64)
        if len(np.unique(labels)) != count:
            raise ValueError(f"k={count}: empty clusters; try fewer clusters")
        # At an iteration limit sklearn's final reassignment can leave its centers
        # different from member means. Export the required arithmetic representatives.
        centers = np.stack([scaled[labels == label].mean(axis=0) for label in range(count)])
        fitted[count] = AttributeCandidate(
            count,
            labels,
            centers,
            float(silhouette_score(scaled, labels, metric="euclidean")) if 1 < count < len(ids) else None,
            float(np.sum((scaled - centers[labels]) ** 2)),
            int(model.n_iter_),
            tuple(str(w.message) for w in caught),
        )
    selected, reason = (
        (k, "Explicit fixed count")
        if k is not None
        else select_count(
            {count: fit.silhouette for count, fit in fitted.items()},
            {count: fit.inertia for count, fit in fitted.items()},
            selection,
            elbow,
            inertia_name="sum of squared Euclidean distances",
        )
    )
    metadata: dict[str, Any] = {
        "format": 1,
        "kind": "static_attributes",
        "ids": list(ids),
        "fields": list(fields),
        "axes": ["basin", "field"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "normalization": "population mean and standard deviation per field; equal weights",
        "algorithm": "Euclidean K-means",
        "implementation": "sklearn KMeans",
        "solver": "lloyd",
        "initialization": "k-means++",
        "seed": seed,
        "n_init": n_init,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "inertia": "sum of squared Euclidean distances to retained member-mean centroids",
        "stored_centroids": "arithmetic member means in standardized attribute space",
        "cluster_labels": "zero-based; centroid axis matches label",
        "mode": "fixed" if k is not None else "search",
        "counts": list(counts),
        "selected_count": selected,
        "selection_method": "fixed" if k is not None else selection,
        "selection_reason": reason,
        "elbow": asdict(elbow or ElbowSettings()) if selection == "elbow" else None,
        "versions": {name: version(name) for name in ("hydrologeez", "numpy", "scikit-learn", "kneed", "polars")},
        "fits": {
            str(count): {
                key: list(value) if key == "warnings" else value
                for key, value in asdict(fit).items()
                if key not in ("labels", "centroids")
            }
            for count, fit in fitted.items()
        },
    }
    result = AttributeExperiment(directory, ids, fields, values, mean, std, fitted, selected, aligned, metadata)
    _save_experiment(result)
    if figures:
        result.plot()
    return result


def _coordinates(table: pl.DataFrame, ids: tuple[str, ...]) -> Array:
    columns = ("gauge_id", "gauge_lon", "gauge_lat")
    if not set(columns) <= set(table.columns) or table.schema["gauge_id"] != pl.String:
        raise ValueError("Coordinates require string gauge_id, gauge_lon and gauge_lat")
    gauges = table["gauge_id"].to_list()
    if len(set(gauges)) != len(gauges) or any(not gauge for gauge in gauges) or not set(ids) <= set(gauges):
        raise ValueError("Coordinates require unique basin identities covering every selected basin")
    positions = {gauge: index for index, gauge in enumerate(gauges)}
    selected = table[[positions[basin] for basin in ids]]
    if any(not selected.schema[field].is_numeric() for field in columns[1:]):
        raise ValueError("Coordinates must be numeric longitude/latitude")
    values = selected.select(columns[1:]).to_numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Coordinates must be finite longitude/latitude")
    return values


def _save_experiment(result: AttributeExperiment) -> None:
    directory = result.directory
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {"members": result.values}
    if result.coordinates is not None:
        arrays["coordinates"] = result.coordinates
    for count, fit in result.candidates.items():
        arrays[f"centroids_{count}"] = fit.centroids
        arrays[f"labels_{count}"] = fit.labels
    np.savez_compressed(directory / "attributes.npz", allow_pickle=False, **arrays)
    if result.selected_count is not None:
        with (directory / "membership.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("basin_id", "cluster"))
            writer.writerows(zip(result.ids, result.labels, strict=True))
        (directory / "clusters.json").write_text(json.dumps(result.clusters, indent=2) + "\n")
    with (directory / "diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("count", "silhouette", "inertia", "iterations"))
        writer.writerows((fit.count, fit.silhouette, fit.inertia, fit.iterations) for fit in result.candidates.values())
    (directory / "run.json").write_text(json.dumps(result.metadata, indent=2, allow_nan=False) + "\n")


def load_attribute_experiment(directory: Path, *, count: int | None = None) -> AttributeExperiment:
    """Reload every retained fit, optionally choosing a count without changing the bundle."""
    metadata: dict[str, Any] = json.loads((directory / "run.json").read_text())
    if metadata["format"] != 1 or metadata.get("kind") != "static_attributes":
        raise ValueError("Unsupported attribute experiment format")
    selected = metadata["selected_count"] if count is None else count
    if selected is not None and selected not in metadata["counts"]:
        raise ValueError("Requested count was not evaluated")
    if count is not None:
        metadata = {
            **metadata,
            "selected_count": count,
            "selection_method": "manual",
            "selection_reason": "Explicit retained count",
            "elbow": None,
        }
    with np.load(directory / "attributes.npz", allow_pickle=False) as arrays:
        fitted = {
            k: AttributeCandidate(
                k,
                arrays[f"labels_{k}"],
                arrays[f"centroids_{k}"],
                metadata["fits"][str(k)]["silhouette"],
                metadata["fits"][str(k)]["inertia"],
                metadata["fits"][str(k)]["iterations"],
                tuple(metadata["fits"][str(k)]["warnings"]),
            )
            for k in metadata["counts"]
        }
        return AttributeExperiment(
            directory,
            tuple(metadata["ids"]),
            tuple(metadata["fields"]),
            arrays["members"],
            np.asarray(metadata["mean"]),
            np.asarray(metadata["std"]),
            fitted,
            selected,
            arrays.get("coordinates"),
            metadata,
        )
