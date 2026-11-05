from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hydrologeez.cluster.attributes import run_attribute_experiment
from hydrologeez.cluster.selection import Selection


def attributes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gauge_id": ["004", "001", "003", "002"],
            "area": [0.0, 2.0, 10.0, 12.0],
            "height": [100.0, 120.0, 200.0, 220.0],
            "unused": [None, None, None, None],
        }
    )


def test_numeric_scaling_grouping_and_centroids(tmp_path: Path) -> None:
    result = run_attribute_experiment(attributes(), tmp_path, "groups", fields=("height", "area"), k=2, figures=False)
    assert result.ids == ("004", "001", "003", "002")
    assert result.fields == ("height", "area")
    assert_allclose(result.mean, [160.0, 6.0])
    assert_allclose(result.std, [np.sqrt(2600), np.sqrt(26)])
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]
    for label in range(2):
        assert_allclose(
            result.standardized_centroids[label], result.normalized_members[result.labels == label].mean(axis=0)
        )
    assert_allclose(result.centroids[result.labels[0]], [110.0, 1.0])
    assert_allclose(result.centroids[result.labels[2]], [210.0, 11.0])
    assert_allclose(result.candidates[2].inertia, 8 / 26)
    # Within-group distance is 2; mean other-group distances are 11 and 9.
    assert result.candidates[2].silhouette is not None
    assert_allclose(result.candidates[2].silhouette, ((1 - 2 / 11) + (1 - 2 / 9)) / 2)


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("area", [0.0, None, 10.0, 12.0], "area.*001"),
        ("area", [0.0, float("inf"), 10.0, 12.0], "area.*001"),
        ("area", [1.0, 1.0, 1.0, 1.0], "variance.*area"),
        ("area", ["a", "b", "c", "d"], "numeric.*area"),
    ],
)
def test_invalid_selected_fields(tmp_path: Path, column: str, values: list[object], message: str) -> None:
    table = attributes().with_columns(pl.Series(column, values))
    with pytest.raises(ValueError, match=message):
        run_attribute_experiment(table, tmp_path, "invalid", fields=(column,), k=2, figures=False)


def test_search_reload_and_selection(tmp_path: Path) -> None:
    from hydrologeez.cluster.attributes import load_attribute_experiment

    result = run_attribute_experiment(
        attributes(), tmp_path, "search", fields=("area", "height"), candidates=(1, 2, 4), seed=17, figures=False
    )
    assert result.selected_count == 2
    assert result.candidates[1].silhouette is None
    assert result.candidates[4].silhouette is None
    loaded = load_attribute_experiment(result.directory)
    assert loaded.metadata == result.metadata
    assert loaded.metadata["seed"] == 17
    assert_array_equal(loaded.labels, result.labels)
    assert_allclose(loaded.centroids, result.centroids)
    assert loaded.clusters == result.clusters
    assert loaded.ids == result.ids
    assert set(loaded.candidates) == {1, 2, 4}
    for count in result.candidates:
        assert_allclose(loaded.candidates[count].centroids, result.candidates[count].centroids)
    manual = load_attribute_experiment(result.directory, count=1)
    assert manual.selection == "manual"
    assert_allclose(manual.centroids, [[6.0, 160.0]])
    assert manual.select().selected_count == 2
    assert load_attribute_experiment(result.directory).selected_count == 2
    with pytest.raises(ValueError, match="not evaluated"):
        load_attribute_experiment(result.directory, count=3)
    with pytest.raises(FileExistsError):
        run_attribute_experiment(attributes(), tmp_path, "search", fields=("area",), k=2, figures=False)


def test_no_elbow_retains_diagnostics_without_membership(tmp_path: Path) -> None:
    from hydrologeez.cluster.attributes import load_attribute_experiment

    result = run_attribute_experiment(
        attributes(), tmp_path, "no-elbow", fields=("area",), candidates=(1, 2), selection="elbow", figures=False
    )
    assert result.selected_count is None
    assert "three candidate" in result.selection_reason
    assert not (result.directory / "membership.csv").exists()
    assert not (result.directory / "clusters.json").exists()
    result.plot()
    assert (result.directory / "figures/inertia_diagnostic.png").exists()
    assert not (result.directory / "figures/conus_cluster_map.png").exists()
    loaded = load_attribute_experiment(result.directory)
    assert loaded.selected_count is None
    with pytest.raises(ValueError, match="No selected partition"):
        _ = loaded.centroids
    manual = load_attribute_experiment(result.directory, count=2)
    assert len(manual.clusters) == 2


@pytest.mark.parametrize(
    ("k", "candidates", "selection"),
    [
        (None, None, "silhouette"),
        (2, (2, 3), "silhouette"),
        (2, None, "elbow"),
        (0, None, "silhouette"),
        (5, None, "silhouette"),
        (None, (), "silhouette"),
        (None, (2, 2), "silhouette"),
    ],
)
def test_invalid_counts(
    tmp_path: Path, k: int | None, candidates: tuple[int, ...] | None, selection: Selection
) -> None:
    with pytest.raises(ValueError):
        run_attribute_experiment(
            attributes(),
            tmp_path,
            "bad",
            fields=("area",),
            figures=False,
            k=k,
            candidates=candidates,
            selection=selection,
        )


def test_mapping_aligns_identities_and_survives_plot_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import csv

    import cartopy.mpl.geoaxes
    from matplotlib.figure import Figure

    from hydrologeez.cluster import plotting
    from hydrologeez.cluster.attributes import load_attribute_experiment

    coordinates = pl.DataFrame(
        {
            "gauge_id": ["002", "003", "001", "004"],
            "gauge_lon": [-80.0, -90.0, -100.0, -110.0],
            "gauge_lat": [32.0, 34.0, 36.0, 38.0],
        }
    )
    original = plotting.render_map

    def fail_map(*args: object, **kwargs: object) -> None:
        raise RuntimeError("plot unavailable")

    monkeypatch.setattr(plotting, "render_map", fail_map)
    with pytest.raises(RuntimeError, match="plot unavailable"):
        run_attribute_experiment(attributes(), tmp_path, "mapped", fields=("area",), k=2, coordinates=coordinates)
    loaded = load_attribute_experiment(tmp_path / "mapped")
    assert loaded.coordinates is not None
    assert_allclose(loaded.coordinates, [[-110.0, 38.0], [-100.0, 36.0], [-90.0, 34.0], [-80.0, 32.0]])
    with (loaded.directory / "membership.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["basin_id"] for row in rows] == list(loaded.ids)
    assert [int(row["cluster"]) for row in rows] == loaded.labels.tolist()
    monkeypatch.setattr(plotting, "render_map", original)
    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "coastlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "add_feature", lambda *args, **kwargs: None)
    original_save = Figure.savefig
    plotted: dict[int, np.ndarray] = {}

    def inspect_map(figure: Figure, fname: str | Path, *, transparent: bool | None = None, **kwargs: object) -> None:
        for label, collection in enumerate(figure.axes[0].collections):
            plotted[label] = np.asarray(collection.get_offsets())
        original_save(figure, fname, transparent=transparent, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_map)
    loaded.plot()
    assert loaded.coordinates is not None
    for label in range(2):
        assert_allclose(plotted[label], loaded.coordinates[loaded.labels == label])
    assert {p.name for p in (loaded.directory / "figures").iterdir()} == {
        "conus_cluster_map.png",
        "conus_cluster_map.svg",
    }


def test_iteration_limit_keeps_arithmetic_centroids(tmp_path: Path) -> None:
    table = pl.DataFrame({"gauge_id": [f"{i:03d}" for i in range(8)], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 20.0]})
    result = run_attribute_experiment(
        table, tmp_path, "limited", fields=("x",), k=3, seed=0, n_init=1, max_iter=1, figures=False
    )
    for label in range(3):
        assert_allclose(
            result.standardized_centroids[label], result.normalized_members[result.labels == label].mean(axis=0)
        )
    residuals = result.normalized_members - result.standardized_centroids[result.labels]
    assert_allclose(result.candidates[3].inertia, np.sum(residuals**2))


@pytest.mark.parametrize("fields", [(), ("area", "area"), ("unknown",), ("gauge_id",), ("gauge_lon",)])
def test_explicit_predictor_selection(tmp_path: Path, fields: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        run_attribute_experiment(attributes(), tmp_path, "bad", fields=fields, k=2, figures=False)


@pytest.mark.parametrize("ids", [["a", "a", "b", "c"], [1, 2, 3, 4], ["a", None, "b", "c"]])
def test_invalid_basin_identities(tmp_path: Path, ids: list[object]) -> None:
    table = attributes().with_columns(pl.Series("gauge_id", ids))
    with pytest.raises(ValueError, match="identit"):
        run_attribute_experiment(table, tmp_path, "bad", fields=("area",), k=2, figures=False)


def test_reload_never_refits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sklearn.cluster import KMeans

    from hydrologeez.cluster.attributes import load_attribute_experiment

    result = run_attribute_experiment(
        attributes(), tmp_path, "retained", fields=("area",), candidates=(1, 2, 3), figures=False
    )

    def reject_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("Reload and selection must not refit")

    monkeypatch.setattr(KMeans, "fit", reject_fit)
    loaded = load_attribute_experiment(result.directory, count=3)
    assert_allclose(loaded.standardized_centroids, result.candidates[3].centroids)
    assert loaded.select().selected_count == result.selected_count


@pytest.mark.parametrize("ids", [["004", "001", "003"], ["004", "001", "003", "003"]])
def test_coordinates_require_complete_unique_identities(tmp_path: Path, ids: list[str]) -> None:
    coordinates = pl.DataFrame({"gauge_id": ids, "gauge_lon": [-100.0] * len(ids), "gauge_lat": [40.0] * len(ids)})
    with pytest.raises(ValueError, match="Coordinates.*identities"):
        run_attribute_experiment(attributes(), tmp_path, "bad-map", fields=("area",), k=2, coordinates=coordinates)


def test_euclidean_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.figure import Figure

    from hydrologeez.cluster import plotting

    labels: dict[str, str] = {}

    def inspect(figure: Figure, directory: Path, name: str) -> None:
        labels[name] = figure.axes[0].get_ylabel()

    monkeypatch.setattr(plotting, "_save", inspect)
    result = run_attribute_experiment(
        attributes(), tmp_path, "diagnostics", fields=("area",), candidates=(1, 2), selection="elbow", figures=False
    )
    result.plot()
    assert labels["inertia_diagnostic"] == "Sum of squared Euclidean distances"
    assert labels["cluster_count_diagnostic"] == "Mean silhouette"
