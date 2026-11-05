import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hydrologeez.cluster import SeasonalCycles, load_experiment, run_experiment


def cycles() -> SeasonalCycles:
    return SeasonalCycles(
        ("001", "002", "003", "004"),
        ("P", "T"),
        ("mm/day", "°C"),
        np.array([np.full((52, 2), x) for x in (0.0, 0.1, 5.0, 5.1)]),
        "2001-01-01",
        "2002-01-01",
        {"source": "test"},
    )


def test_fixed_bundle_identity_units_and_collision(tmp_path: Path) -> None:
    result = run_experiment(cycles(), tmp_path, "fixed", k=2, seed=4, figures=False)
    assert result.selected_count == 2
    assert set(result.candidates) == {2}
    loaded = load_experiment(tmp_path / "fixed")
    assert loaded.cycles.ids == cycles().ids
    assert_array_equal(loaded.labels, result.labels)
    assert_allclose(loaded.cycles.values, cycles().values)
    assert_allclose(loaded.centroids, result.centroids)
    assert set(sum(loaded.clusters.values(), [])) == set(cycles().ids)
    assert (tmp_path / "fixed" / "membership.csv").read_text().startswith("basin_id,cluster\n001,")
    metadata = json.loads((tmp_path / "fixed" / "run.json").read_text())
    assert metadata["mode"] == "fixed"
    assert metadata["exclusions"] == []
    with pytest.raises(FileExistsError):
        run_experiment(cycles(), tmp_path, "fixed", k=2, figures=False)


def test_search_selects_best_and_retains_other_fits(tmp_path: Path) -> None:
    result = run_experiment(cycles(), tmp_path, "search", candidates=(1, 2, 3), figures=False)
    assert result.selected_count == 2
    score2, score3 = result.candidates[2].silhouette, result.candidates[3].silhouette
    assert score2 is not None and score3 is not None
    assert score2 > score3
    other = load_experiment(tmp_path / "search", count=3)
    assert other.selected_count == 3
    assert_array_equal(other.labels, result.candidates[3].labels)
    assert result.candidates[1].silhouette is None
    with pytest.raises(ValueError, match="defined silhouette"):
        run_experiment(cycles(), tmp_path, "undefined", candidates=(1, 4), figures=False)
    assert not (tmp_path / "undefined").exists()


def test_iteration_limit_and_empty_clusters(tmp_path: Path) -> None:
    result = run_experiment(cycles(), tmp_path, "limited", k=2, max_iter=1, figures=False)
    assert not result.candidates[2].converged
    same = cycles()
    object.__setattr__(same, "values", np.zeros((4, 52, 2)))
    with pytest.raises(ValueError, match="variance"):
        run_experiment(same, tmp_path, "constant", k=2, figures=False)
