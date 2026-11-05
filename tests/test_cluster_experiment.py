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


def test_per_basin_bundle_candidates_and_replot_without_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    shape = np.sin(np.arange(52) * 2 * np.pi / 52)
    values = np.stack([np.column_stack((np.roll(shape, shift), np.roll(shape, shift + 5))) for shift in (0, 1, 20, 21)])
    seasonal = replace(cycles(), values=values * np.arange(1, 5)[:, None, None] + 10)
    result = run_experiment(
        seasonal,
        tmp_path,
        "shape",
        candidates=(1, 2),
        normalization="per_basin",
        figures=False,
        coordinates=np.array([[-120.0, 40], [-110, 35], [-90, 40], [-80, 35]]),
    )

    def no_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("Reloading and plotting must not fit")

    monkeypatch.setattr("hydrologeez.cluster.experiment.fit_candidates", no_fit)
    for count in (1, 2):
        loaded = load_experiment(result.directory, count=count)
        assert loaded.normalization == "per_basin"
        assert loaded.centroid_units == ("standardized (dimensionless)",) * 2
        assert_allclose(loaded.centroids, result.candidates[count].centroids)
        assert_allclose(loaded.mean, np.full((4, 1, 2), 10))
        assert_allclose(loaded.std[:, 0, 0], np.arange(1, 5) / np.sqrt(2))
        assert_allclose(loaded.cycles.values, seasonal.values)
        assert_allclose(loaded.normalized_members, values * np.sqrt(2), atol=1e-14)

    import cartopy.mpl.geoaxes

    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "coastlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "add_feature", lambda *args, **kwargs: None)
    loaded.plot(rows=1, columns=4)
    assert {p.stem for p in (loaded.directory / "figures").glob("*.png")} == {
        "precipitation_profiles",
        "temperature_profiles",
        "conus_cluster_map",
        "cluster_count_diagnostic",
        "inertia_diagnostic",
    }


def test_legacy_pooled_bundle_and_default_reproducibility(tmp_path: Path) -> None:
    default = run_experiment(cycles(), tmp_path, "default", k=2, figures=False)
    explicit = run_experiment(cycles(), tmp_path, "explicit", k=2, normalization="pooled", figures=False)
    assert_array_equal(default.labels, explicit.labels)
    assert_array_equal(default.centroids, explicit.centroids)
    # Format 1 stored physical-unit centers and omitted the normalization mode.
    metadata = default.metadata.copy()
    metadata["format"] = 1
    metadata.pop("normalization")
    metadata.pop("stored_centroids")
    (default.directory / "run.json").write_text(json.dumps(metadata))
    np.savez_compressed(
        default.directory / "seasonal_cycles.npz",
        members=cycles().values,
        centroids_2=np.array([np.full((52, 2), x) for x in (0.05, 5.05)]),
        labels_2=np.array([0, 0, 1, 1]),
    )
    loaded = load_experiment(default.directory)
    assert loaded.normalization == "pooled"
    assert loaded.centroid_units == cycles().units
    assert_allclose(loaded.centroids[:, 0, 0], [0.05, 5.05])


def test_elbow_no_selection_roundtrip_and_reselection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hydrologeez.cluster import ElbowSettings

    result = run_experiment(cycles(), tmp_path, "elbow", candidates=(1, 2), selection="elbow", figures=True)
    assert result.selected_count is None  # Two points cannot establish curvature.
    assert result.selection == "elbow"
    assert result.elbow == ElbowSettings()
    assert "three" in result.selection_reason
    assert not (result.directory / "membership.csv").exists()
    assert not (result.directory / "clusters.json").exists()
    before = (result.directory / "run.json").read_bytes()

    def no_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("Selection must reuse retained fits")

    monkeypatch.setattr("hydrologeez.cluster.experiment.fit_candidates", no_fit)
    loaded = load_experiment(result.directory)
    assert loaded.selected_count is None
    assert loaded.elbow == result.elbow
    for property_name in ("labels", "centroids", "clusters"):
        with pytest.raises(ValueError, match="No selected partition"):
            getattr(loaded, property_name)
    selected = loaded.select(selection="silhouette")
    assert selected.selected_count == 2
    assert selected.selection == "silhouette"
    assert selected.candidates[2] is loaded.candidates[2]
    assert loaded.selected_count is None
    assert load_experiment(result.directory, count=1).selected_count == 1
    assert (result.directory / "run.json").read_bytes() == before
    loaded.plot()
    assert {p.stem for p in (result.directory / "figures").glob("*.png")} == {
        "cluster_count_diagnostic",
        "inertia_diagnostic",
    }


def test_incompatible_selection_arguments(tmp_path: Path) -> None:
    from hydrologeez.cluster import ElbowSettings

    with pytest.raises(ValueError, match="fixed"):
        run_experiment(cycles(), tmp_path, "fixed-elbow", k=2, selection="elbow", figures=False)
    with pytest.raises(ValueError, match="elbow"):
        run_experiment(cycles(), tmp_path, "wrong-settings", candidates=(1, 2), elbow=ElbowSettings(), figures=False)


def test_selected_elbow_roundtrip_and_view_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from matplotlib.figure import Figure

    from hydrologeez.cluster import ElbowSettings
    from hydrologeez.cluster.fit import Candidate

    seasonal = replace(
        cycles(), ids=tuple(str(i) for i in range(7)), values=np.arange(7 * 52 * 2).reshape(7, 52, 2).astype(float)
    )
    # A specified steep segment ending at 3, followed by a flat tail.
    fits = {
        k: Candidate(k, np.arange(7) % k, seasonal.values[:k], 0.5 if k == 2 else 0.1, inertia, (), True, 1, 1, ())
        for k, inertia in enumerate([10.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0], start=1)
    }
    monkeypatch.setattr("hydrologeez.cluster.experiment.fit_candidates", lambda *args, **kwargs: fits)
    result = run_experiment(
        seasonal,
        tmp_path,
        "clear",
        candidates=tuple(fits),
        selection="elbow",
        elbow=ElbowSettings(S=0.5, online=True),
        figures=False,
    )
    loaded = load_experiment(result.directory)
    assert loaded.selected_count == 3
    assert loaded.elbow == ElbowSettings(S=0.5, online=True)
    assert loaded.selection_reason == result.selection_reason
    assert loaded.metadata["versions"]["kneed"]
    assert (result.directory / "membership.csv").exists()
    markers: list[list[float]] = []

    def inspect_save(figure: Figure, directory: Path, name: str) -> None:
        if name == "inertia_diagnostic":
            markers.append([float(np.asarray(line.get_xdata())[0]) for line in figure.axes[0].lines[1:]])

    monkeypatch.setattr("hydrologeez.cluster.plotting._save", inspect_save)
    monkeypatch.setattr("hydrologeez.cluster.plotting.render_figures", lambda *args, **kwargs: None)
    loaded = replace(loaded, coordinates=np.zeros((7, 2)))
    loaded.plot()
    loaded.select(selection="silhouette").plot(tmp_path / "comparison")
    loaded.select(selection="elbow", elbow=ElbowSettings(S=100)).plot(tmp_path / "none")
    assert markers == [[3.0], [], []]
