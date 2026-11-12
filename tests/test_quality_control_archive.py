from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hydrologeez.camels import Camels, prepare
from hydrologeez.quality_control import (
    DischargeRecord,
    DischargeSource,
    StreamflowQualityArchive,
    assess_indices,
    camels_records,
    camels_source,
    compute_indices,
    screen,
    write_archive,
)
from tests.test_camels import source_files

SOURCE = DischargeSource("synthetic test source", "m3/s", 1.0, {"A": "reliable", "P": "suspect"})
START, STOP = date(2001, 1, 1).toordinal(), date(2003, 1, 1).toordinal()


def records() -> list[DischargeRecord]:
    rng = np.random.default_rng(0)
    dates = np.arange(START, STOP, dtype=np.int64)
    automated = 2.0 + rng.random(len(dates))
    automated[40] = -5.0
    missing = np.zeros(len(dates), dtype=np.bool_)
    missing[100:110] = True
    automated[100:110] = -999.0
    flagged = 1.0 + rng.random(len(dates) - 30)
    flags = np.full(len(flagged), "A")
    flags[5] = "P"
    return [
        DischargeRecord("automated", dates, automated, missing, np.full(len(dates), "")),
        DischargeRecord("flagged", dates[30:], flagged, np.zeros(len(flagged), dtype=np.bool_), flags),
    ]


def daily_inputs(record: DischargeRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    screening = screen(record, SOURCE)
    return screening.dates, screening.discharge, screening.usable


def test_archive_preserves_observations_and_separates_reasons(tmp_path: Path) -> None:
    write_archive(tmp_path / "qc", SOURCE, records())
    archive = StreamflowQualityArchive(tmp_path / "qc")
    automated, flagged = records()
    daily = archive.daily(["flagged", "automated"], "2001-01-01", "2003-01-01")
    assert daily.routes == ("provider_flags", "automated")
    assert_array_equal(daily.discharge[:, 1], automated.discharge)
    assert_array_equal(daily.missing[:, 1], automated.missing)
    assert_array_equal(daily.discharge[30:, 0], flagged.discharge)
    assert np.isnan(daily.discharge[:30, 0]).all() and daily.missing[:30, 0].all()
    assert_array_equal(daily.flags[30:, 0], flagged.flags)
    assert_array_equal(np.flatnonzero(daily.reasons["negative"][:, 1]), [40])
    assert_array_equal(np.flatnonzero(daily.reasons["provider_suspect"][:, 0]), [35])
    assert_array_equal(np.flatnonzero(daily.suspect.any(axis=1)), [35, 40])
    assert not daily.observed[100:110, 1].any()

    monthly = archive.indices(["automated"], "monthly")
    february = monthly.period_end.tolist().index(date(2001, 2, 28).toordinal())
    feb = slice(31, 59)
    usable = np.ones(28, dtype=np.bool_)
    usable[40 - 31] = False
    assert monthly.values[february, 0, monthly.names.index("MEAN")] == pytest.approx(
        automated.discharge[feb][usable].mean()
    )
    assert monthly.counts[february, 0] == 27
    assert monthly.reliable[february, 0]
    assert monthly.units[monthly.names.index("CV")] == "-"

    homogeneity = archive.homogeneity(["automated", "flagged"], "monthly")
    direct = assess_indices(compute_indices(*daily_inputs(automated), "monthly"))
    assert homogeneity.count[0].tolist() == [h.count for h in direct] == [23] * 8  # April 2001 has 20 usable days
    assert homogeneity.classification[0].tolist() == [h.classification for h in direct]
    assert_array_equal(homogeneity.statistics[0], [h.statistics for h in direct])
    assert_array_equal(homogeneity.rejected[0], [h.rejected for h in direct])
    yearly = archive.homogeneity(["automated"], "yearly")
    assert yearly.classification[0, 0] == "not sufficient data"

    method = archive.metadata["method"]
    assert method["outlier_offset"] == {"value": 0.01, "units": "m3/s"}
    assert archive.metadata["source"]["provenance"] == "synthetic test source"
    with pytest.raises(FileExistsError):
        write_archive(tmp_path / "qc", SOURCE, records())


def test_results_do_not_depend_on_batch_boundaries(tmp_path: Path) -> None:
    write_archive(tmp_path / "together", SOURCE, records())
    write_archive(tmp_path / "alone", SOURCE, records()[:1])
    together, alone = StreamflowQualityArchive(tmp_path / "together"), StreamflowQualityArchive(tmp_path / "alone")
    full = together.daily(["automated"], "2001-01-01", "2003-01-01")
    single = alone.daily(["automated"], "2001-01-01", "2003-01-01")
    parts = [
        together.daily(["automated"], a, b) for a, b in [("2001-01-01", "2001-08-17"), ("2001-08-17", "2003-01-01")]
    ]
    for name in ("discharge", "missing", "flags", "suspect"):
        assert_array_equal(getattr(full, name), getattr(single, name))
        assert_array_equal(getattr(full, name), np.concatenate([getattr(p, name) for p in parts]))
    for resolution in ("yearly", "seasonal", "monthly"):
        assert_array_equal(
            together.indices(["automated"], resolution).values, alone.indices(["automated"], resolution).values
        )
        a, b = together.homogeneity(["automated"], resolution), alone.homogeneity(["automated"], resolution)
        assert_array_equal(a.statistics, b.statistics)
        assert_array_equal(a.classification, b.classification)


def test_camels_store_is_a_flagged_cfs_source(tmp_path: Path) -> None:
    raw, areas = source_files(tmp_path)
    prepare(raw, tmp_path / "prepared", areas)
    store = Camels(tmp_path / "prepared")
    source = camels_source(store)
    assert (source.units, source.cubic_metres_per_second_per_unit) == ("ft3/s", 0.028316846592)
    record = next(iter(camels_records(store, ["00000002"])))
    assert record.basin == "00000002"
    assert record.flags.tolist() == ["A", "A:e", "", "M", "A"]
    assert_array_equal(record.missing, [False, False, True, True, False])
    assert record.discharge[3] == -999
    write_archive(tmp_path / "qc", source, camels_records(store))
    daily = StreamflowQualityArchive(tmp_path / "qc").daily(["00000001"], "2000-02-27", "2000-03-03")
    assert daily.routes == ("provider_flags",)
    assert not daily.suspect.any()
