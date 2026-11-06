"""Cluster-count selection from retained fit diagnostics."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from kneed import KneeLocator

Selection = Literal["silhouette", "elbow"]


@dataclass(frozen=True)
class ElbowSettings:
    """Kneedle controls; S is sensitivity, not a confidence level."""

    S: float = 1.0
    interp_method: Literal["interp1d", "polynomial"] = "interp1d"
    polynomial_degree: int = 3
    online: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.S) or self.S <= 0:
            raise ValueError("S must be finite and positive")
        if self.interp_method not in ("interp1d", "polynomial"):
            raise ValueError("interp_method must be interp1d or polynomial")
        if type(self.polynomial_degree) is not int or self.polynomial_degree < 1:
            raise ValueError("polynomial_degree must be a positive integer")
        if type(self.online) is not bool:
            raise ValueError("online must be a boolean")


def validate_selection(selection: Selection, elbow: ElbowSettings | None) -> None:
    if selection not in ("silhouette", "elbow"):
        raise ValueError("selection must be silhouette or elbow")
    if elbow is not None and (selection != "elbow" or not isinstance(elbow, ElbowSettings)):
        raise ValueError("elbow requires selection='elbow' and ElbowSettings")


def select_count(
    silhouettes: dict[int, float | None],
    inertias: dict[int, float],
    selection: Selection,
    elbow: ElbowSettings | None,
    *,
    inertia_name: str = "constrained-DTW inertia",
) -> tuple[int | None, str]:
    validate_selection(selection, elbow)
    if selection == "silhouette":
        eligible = {count: score for count, score in silhouettes.items() if score is not None}
        if not eligible:
            raise ValueError("Search requires at least one defined silhouette (2 <= k < basin count)")
        selected = max(eligible, key=lambda count: (eligible[count], -count))
        return selected, "Highest defined mean silhouette; ties smaller count; not a scientific optimum"
    settings = elbow or ElbowSettings()
    counts = sorted(inertias)
    values = np.array([inertias[count] for count in counts])
    reason = "No elbow detected with these settings"
    if len(counts) < 3:
        return None, f"{reason}: at least three candidate counts required"
    if not np.isfinite(values).all() or np.any(values < 0):
        return None, f"{reason}: inertia must be finite and nonnegative"
    if np.ptp(values) == 0:
        return None, f"{reason}: flat inertia curve"
    if np.any(np.diff(values) > 0):
        return None, f"{reason}: nonmonotonic inertia curve (not altered)"
    if settings.interp_method == "polynomial" and settings.polynomial_degree >= len(counts):
        return None, f"{reason}: polynomial degree must be below the number of candidates"
    detector = KneeLocator(
        counts,
        values,
        S=settings.S,
        curve="convex",
        direction="decreasing",
        interp_method=settings.interp_method,
        polynomial_degree=settings.polynomial_degree,
        online=settings.online,
    )
    if detector.elbow is None:
        return None, reason
    return int(detector.elbow), f"Kneedle elbow in {inertia_name}; not a scientific optimum"
