"""Scientific figure outputs on small, independently specified seasonal curves."""

from pathlib import Path

import cartopy.mpl.geoaxes
import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.testing import assert_allclose
from numpy.typing import NDArray

matplotlib.use("Agg")

from hydrologeez.cluster.plotting import render_figures


@pytest.mark.parametrize("search", [False, True])
def test_figures_are_saved_in_both_formats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, search: bool) -> None:
    # Keep the test independent of Natural Earth downloads; the map projection and
    # gauge plotting are real. An online smoke render exercises geographic detail.
    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "coastlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "add_feature", lambda *args, **kwargs: None)
    saved_profiles: list[tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, float]]] = []
    original_savefig = Figure.savefig

    def inspect_save(figure: Figure, fname: str | Path, *, transparent: bool | None = None, **kwargs: object) -> None:
        if Path(fname).name in {"precipitation_profiles.png", "temperature_profiles.png"}:
            for axis in figure.axes:
                saved_profiles.append(
                    (
                        np.asarray(axis.lines[0].get_ydata(), dtype=float),
                        np.asarray(axis.lines[-1].get_ydata(), dtype=float),
                        axis.get_ylim(),
                    )
                )
        original_savefig(figure, fname, transparent=transparent, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_save)
    members = np.broadcast_to(np.array([[[2.0, 10.0]], [[4.0, 20.0]], [[8.0, 30.0]]]), (3, 52, 2)).copy()
    render_figures(
        tmp_path,
        members,
        members[[0, 2]],
        np.array([0, 0, 1]),
        ("P", "T"),
        ("mm/day", "°C"),
        np.array([[-120.0, 40.0], [-100.0, 35.0], [-80.0, 42.0]]),
        {2: 0.4, 3: None},
        search,
    )
    names = {"conus_cluster_map", "precipitation_profiles", "temperature_profiles"}
    if search:
        names.add("cluster_count_diagnostic")
    assert {p.name for p in tmp_path.iterdir()} == {
        f"{name}.{extension}" for name in names for extension in ("png", "svg")
    }
    assert all(p.stat().st_size > 1000 for p in tmp_path.iterdir())
    assert len(saved_profiles) == 4
    for index, value in enumerate([2.0, 8.0, 10.0, 30.0]):
        assert_allclose(saved_profiles[index][0], np.full(52, value))
        assert_allclose(saved_profiles[index][1], np.full(52, value))
    assert saved_profiles[0][2] == saved_profiles[1][2]
    assert saved_profiles[2][2] == saved_profiles[3][2]


def test_separate_variable_grids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "coastlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "add_feature", lambda *args, **kwargs: None)
    original_savefig = Figure.savefig

    def inspect_grid(figure: Figure, fname: str | Path, *, transparent: bool | None = None, **kwargs: object) -> None:
        if Path(fname).stem.endswith("_profiles"):
            assert len(figure.axes) == 4
            assert [axis.get_visible() for axis in figure.axes] == [True, False, False, False]
        original_savefig(figure, fname, transparent=transparent, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_grid)
    curves = np.full((1, 52, 2), 5.0)
    render_figures(
        tmp_path,
        curves,
        curves,
        np.array([0]),
        ("P", "T"),
        ("mm/day", "°C"),
        np.array([[-100.0, 40.0]]),
        {1: None},
        False,
        rows=2,
        columns=2,
    )
    assert {p.stem for p in tmp_path.iterdir()} == {
        "conus_cluster_map",
        "precipitation_profiles",
        "temperature_profiles",
    }


@pytest.mark.parametrize(("rows", "columns"), [(1, 1), (0, 2), (2, -1)])
def test_invalid_grids_rejected(tmp_path: Path, rows: int, columns: int) -> None:
    curves = np.ones((2, 52, 1))
    with pytest.raises(ValueError, match="Grid"):
        render_figures(
            tmp_path,
            curves,
            curves,
            np.array([0, 1]),
            ("P",),
            ("mm/day",),
            np.array([[-100.0, 40], [-90, 35]]),
            {},
            False,
            rows=rows,
            columns=columns,
        )
