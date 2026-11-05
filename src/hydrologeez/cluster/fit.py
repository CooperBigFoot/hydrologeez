"""Constrained dependent DTW fits with observed outer-loop diagnostics."""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw

Array = NDArray[np.float64]


def pooled_scale(values: Array) -> tuple[Array, Array, Array]:
    mean = values.mean(axis=(0, 1))
    std = values.std(axis=(0, 1))
    if np.any(std == 0):
        raise ValueError("Selected variables must have nonzero pooled variance")
    return (values - mean) / std, mean, std


def distances(values: Array, centers: Array, *, radius: int = 2) -> Array:
    return np.asarray(cdist_dtw(values, centers, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius, n_jobs=1))


@dataclass(frozen=True)
class Candidate:
    count: int
    labels: NDArray[np.int64]
    centroids: Array
    silhouette: float | None
    inertia: float
    history: tuple[float, ...]
    converged: bool
    attempts: int
    successful_starts: int
    warnings: tuple[str, ...]


class _ObservedKMeans(TimeSeriesKMeans):
    attempts: int
    successful_starts: int
    best_inertia: float
    history: list[float]
    best_history: tuple[float, ...]

    # tslearn 0.10 invokes this once per measured assignment objective, before DBA.
    def _update_centroids(self, X: Array) -> None:
        self.history.append(float(self.inertia_))
        super()._update_centroids(X)

    def _fit_one_init(self, X: Array, x_squared_norms: Array | None, rs: np.random.RandomState) -> "_ObservedKMeans":
        self.attempts += 1
        self.history = []
        super()._fit_one_init(X, x_squared_norms, rs)
        self.successful_starts += 1
        if self.inertia_ < self.best_inertia:
            self.best_inertia = float(self.inertia_)
            self.best_history = tuple(self.history)
        return self


def fit_candidates(
    values: Array, counts: tuple[int, ...], *, seed: int, n_init: int, max_iter: int, tolerance: float, radius: int
) -> dict[int, Candidate]:
    if not counts or len(set(counts)) != len(counts) or any(k < 1 or k > len(values) for k in counts):
        raise ValueError("Cluster counts must be unique and between 1 and the basin count")
    if n_init < 1 or max_iter < 1 or tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("Positive restarts, iteration limit and tolerance required")
    pairwise = distances(values, values, radius=radius)
    np.fill_diagonal(pairwise, 0.0)
    results: dict[int, Candidate] = {}
    for count in counts:
        model = _ObservedKMeans(
            n_clusters=count,
            metric="dtw",
            metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": radius},
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            tol=tolerance,
            random_state=seed,
            n_jobs=1,
        )
        model.attempts = 0
        model.successful_starts = 0
        model.best_inertia = float("inf")
        model.best_history = ()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(values)
        if model.successful_starts == 0 or model.cluster_centers_ is None:
            raise ValueError(f"k={count}: no successful fit (empty clusters); try fewer clusters")
        centers = np.asarray(model.cluster_centers_, dtype=np.float64)
        d = distances(values, centers, radius=radius)
        labels = np.asarray(d.argmin(axis=1), dtype=np.int64)
        if not np.isfinite(centers).all() or len(np.unique(labels)) != count:
            raise ValueError(f"k={count}: invalid final centers or empty clusters")
        history = model.best_history
        results[count] = Candidate(
            count,
            labels,
            centers,
            float(silhouette_score(pairwise, labels, metric="precomputed")) if 1 < count < len(values) else None,
            float(np.mean(d.min(axis=1) ** 2)),
            history,
            len(history) >= 2 and abs(history[-2] - history[-1]) < tolerance,
            model.attempts,
            model.successful_starts,
            tuple(str(w.message) for w in caught),
        )
    return results
