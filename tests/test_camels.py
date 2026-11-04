import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hydrologeez.camels import Camels, prepare, prepare_forcing


def test_forcing_retains_leap_day_and_marks_invalid_temperature():
    rows = np.array(
        [
            [2000, 2, 28, 12, 0, 2, 0, 0, 10, 0, 0],
            [2000, 2, 29, 12, 0, 3, 0, 0, 12, 2, 0],
            [2000, 3, 1, 12, 0, 4, 0, 0, -999, -999, 0],
        ],
        dtype=float,
    )
    dates, values = prepare_forcing(rows, 45.0)
    assert dates.tolist() == [date(2000, 2, d).toordinal() for d in (28, 29)] + [date(2000, 3, 1).toordinal()]
    assert_allclose(values[:2, :2], [[2, 5], [3, 7]])
    assert np.isnan(values[2, 1:]).all()


def source_files(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("topo", "name", "clim", "hydro", "soil", "geol", "vege"):
        (raw / f"camels_{name}.txt").write_text("gauge_id;gauge_lat\n00000001;40\n00000002;45\n")
    areas = raw / "areas.json"
    areas.write_text(
        json.dumps(
            {
                "source": "test snapshot",
                "basins": {
                    "00000001": {"total_km2": 2, "contributing_km2": None},
                    "00000002": {"total_km2": 3, "contributing_km2": 1},
                },
            }
        )
    )
    with zipfile.ZipFile(raw / "basin_timeseries_v1p2_metForcing_obsFlow.zip", "w") as z:
        for i in (1, 2):
            z.writestr(
                f"x/basin_mean_forcing/daymet/01/{i:08}_lump_cida_forcing_leap.txt",
                "40\n100\n1000000\nYear Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)\n"
                f"2000 2 28 12 0 {i} 0 0 10 0 0\n2000 2 29 12 0 {i + 1} 0 0 12 2 0\n2000 3 1 12 0 {i + 2} 0 0 14 4 0\n",
            )
            z.writestr(
                f"x/usgs_streamflow/01/{i:08}_streamflow_qc.txt",
                f"{i:08} 2000 2 27 1 A\n{i:08} 2000 2 28 10 A:e\n{i:08} 2000 3 1 -999 M\n{i:08} 2000 3 2 2 A\n",
            )
    return raw, areas


def test_full_preparation_alignment_and_explicit_selections(tmp_path: Path) -> None:
    raw, areas = source_files(tmp_path)
    out = tmp_path / "prepared"
    prepare(raw, out, areas)
    store = Camels(out)
    assert store.ids == ("00000001", "00000002")
    selected = store.select(["00000002", "00000001"], "2000-02-28", "2000-03-02")
    assert_array_equal(selected.forcing.values[:, :, 0], [[2, 1], [3, 2], [4, 3]])
    assert_array_equal(selected.present[:, 0], [True, False, True])
    assert selected.qc[:, 0].tolist() == ["A:e", "", "M"]
    assert_allclose(selected.discharge_mm_day[0], 10 * 0.028316846592 * 86400 / 1000000 * 1000)
    assert np.isnan(selected.discharge_mm_day[1:]).all()
    assert selected.discharge_cfs[2, 0] == -999
    observations = store.observations("00000001")
    assert len(observations.dates) == 5
    assert observations.discharge_cfs[-1] == 2
    assert store.basins[0].usgs_contributing_km2 is None
    for ids, start, stop in [
        (["missing"], "2000-02-28", "2000-03-01"),
        (["00000001"] * 2, "2000-02-28", "2000-03-01"),
        (["00000001"], "2000-02-27", "2000-03-01"),
        (["00000001"], "2000-02-28", "2000-02-28"),
    ]:
        with pytest.raises(ValueError):
            store.select(ids, start, stop)


def test_selection_reuses_forcing_and_matches_resident_hbv(tmp_path: Path) -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    raw, areas = source_files(tmp_path)
    out = tmp_path / "prepared"
    prepare(raw, out, areas)
    selected = Camels(out).select(["00000002", "00000001"], "2000-02-28", "2000-03-02")
    parameters = Parameters(
        np.array([1, 0, 1], dtype=np.int64),
        np.tile([0, 2, 0.05, 0.1, 1, 100, 1, 0.5], (3, 1)),
        np.tile([1, 0.5, 0.1, 0.05], (3, 1)),
        np.full(3, 2.5),
    )
    initial = Memory(np.tile([0.0, 0.0, 50.0], (3, 1)), np.zeros((3, 2)), np.zeros(6))
    # Resident forcing from hand-worked P/T, with PET independently anchored below.
    resident = selected.forcing.values.copy()
    resident[:, :, :2] = [[[2, 5], [1, 5]], [[3, 7], [2, 7]], [[4, 9], [3, 9]]]
    reference = simulate(Forcing(resident, np.ones(2), np.arange(3, dtype=np.int64)), parameters, initial)
    for _ in range(2):
        result = simulate(selected.forcing, parameters, initial)
        assert_array_equal(result.outlet, reference.outlet)
        for name in ("zone", "groundwater", "routing"):
            assert_array_equal(getattr(result.final, name), getattr(reference.final, name))
    assert selected.forcing.values.shape == (3, 2, 3)


def test_invalid_forcing_and_missing_dates_are_audited_not_filled(tmp_path: Path) -> None:
    raw, areas = source_files(tmp_path)
    path = raw / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
    with zipfile.ZipFile(path) as z:
        entries = {name: z.read(name) for name in z.namelist()}
    key = next(name for name in entries if "00000001_lump" in name)
    entries[key] = entries[key].replace(b"2000 2 29 12 0 2 0 0 12 2 0\n", b"").replace(b"14 4 0", b"-999 -999 0")
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    out = tmp_path / "prepared"
    prepare(raw, out, areas)
    store = Camels(out)
    assert store.basins[0].missing_forcing_dates == 1
    assert store.basins[0].invalid_forcing_rows == 2
    with pytest.raises(ValueError, match="invalid/missing forcing"):
        store.select(["00000001"], "2000-02-28", "2000-03-02")
    assert store.select(["00000001"], "2000-02-28", "2000-02-29").forcing.values.shape == (1, 1, 3)


def test_source_pet_anchor_uses_attribute_latitude() -> None:
    rows = np.array([[1980, 7, 1, 12, 0, 3.01, 0, 0, 20, 9.11, 0]], dtype=float)
    _, values = prepare_forcing(rows, 47.23739)
    assert_allclose(values[0], [3.01, 14.555, 3.3157159628066526], rtol=1e-13)
