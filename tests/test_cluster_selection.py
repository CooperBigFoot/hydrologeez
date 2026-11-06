"""Selection expectations use specified curves, independently of clustering fits."""

import pytest

from hydrologeez.cluster import ElbowSettings
from hydrologeez.cluster.selection import select_count


def test_clear_elbow_and_straight_line() -> None:
    # Piecewise linear: steep improvements stop at count 3, then a flat tail.
    counts = [1, 2, 3, 4, 5, 6, 7]
    selected, reason = select_count(
        dict.fromkeys(counts),
        dict(zip(counts, [10.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0], strict=True)),
        "elbow",
        ElbowSettings(),
    )
    assert selected == 3
    assert reason
    selected, reason = select_count(
        dict.fromkeys(counts),
        dict(zip(counts, [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], strict=True)),
        "elbow",
        ElbowSettings(),
    )
    assert selected is None
    assert "No elbow" in reason


@pytest.mark.parametrize("inertias", [{1: 2.0, 2: 2.0, 3: 2.0}, {1: 3.0, 2: 1.0}, {1: 3.0, 2: 4.0, 3: 1.0}])
def test_unusable_curves_have_no_selection(inertias: dict[int, float]) -> None:
    selected, reason = select_count(dict.fromkeys(inertias), inertias, "elbow", ElbowSettings())
    assert selected is None
    assert "No elbow" in reason


@pytest.mark.parametrize(
    "settings",
    [
        {"S": 0},
        {"S": float("nan")},
        {"interp_method": "cubic"},
        {"polynomial_degree": 0},
        {"polynomial_degree": 2.5},
        {"online": 1},
    ],
)
def test_settings_validation(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ElbowSettings(**settings)  # ty: ignore[invalid-argument-type]


def test_silhouette_ties_and_undefined() -> None:
    assert select_count({3: 0.5, 2: 0.5, 1: None}, {1: 10.0, 2: 5.0, 3: 2.0}, "silhouette", None)[0] == 2
    with pytest.raises(ValueError, match="defined silhouette"):
        select_count({1: None}, {1: 10.0}, "silhouette", None)
