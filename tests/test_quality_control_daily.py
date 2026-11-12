from datetime import date, timedelta

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hydrologeez.quality_control.daily import REASONS, DischargeRecord, DischargeSource, screen

M3S = DischargeSource("test source", "m3/s", 1.0, None)
FLAGGED = DischargeSource("test source", "m3/s", 1.0, {"A": "reliable", "P": "suspect", "M": "unflagged"})


def days(start: date, stop: date) -> np.ndarray:
    return np.arange(start.toordinal(), stop.toordinal(), dtype=np.int64)


def record(
    dates: np.ndarray, discharge: np.ndarray, missing: np.ndarray | None = None, flags: np.ndarray | None = None
) -> DischargeRecord:
    return DischargeRecord(
        "basin",
        dates,
        discharge.astype(np.float64),
        np.zeros(len(dates), dtype=np.bool_) if missing is None else missing,
        np.full(len(dates), "") if flags is None else flags,
    )


def reason(screening_reasons: np.ndarray, name: str) -> np.ndarray:
    return screening_reasons[:, REASONS.index(name)]


def test_negative_values_and_runs_of_more_than_ten_equal_positive_days() -> None:
    q = np.array([1.0, -0.5] + [3.0] * 10 + [2.0] + [4.0] * 11 + [0.0] * 15 + [5.0] * 6 + [np.nan] + [5.0] * 6)
    missing = np.zeros(len(q), dtype=np.bool_)
    screening = screen(record(days(date(2000, 1, 1), date(2000, 1, 1) + timedelta(len(q))), q, missing), M3S)
    assert screening.route == "automated"
    assert_array_equal(np.flatnonzero(reason(screening.reasons, "negative")), [1])
    # Exactly ten equal values pass, eleven fail; zero runs are not rejected; a nonfinite day ends a run.
    assert_array_equal(np.flatnonzero(reason(screening.reasons, "constant_run")), np.arange(13, 24))
    assert not reason(screening.reasons, "provider_suspect").any()


def test_absent_dates_end_runs_and_are_stored_as_missing() -> None:
    dates = np.concatenate([days(date(2000, 1, 1), date(2000, 1, 7)), days(date(2000, 1, 8), date(2000, 1, 14))])
    screening = screen(record(dates, np.full(len(dates), 2.0)), M3S)
    assert_array_equal(screening.dates, days(date(2000, 1, 1), date(2000, 1, 14)))
    assert_array_equal(np.flatnonzero(screening.missing), [6])
    assert np.isnan(screening.discharge[6])
    assert not screening.reasons.any()


@pytest.mark.parametrize(
    ("dates", "error"),
    [(np.array([2, 1]), "increasing"), (np.array([1, 1]), "increasing"), (np.array([], dtype=np.int64), "empty")],
)
def test_rejects_unordered_duplicate_or_empty_dates(dates: np.ndarray, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        screen(record(dates, np.ones(len(dates))), M3S)


def spike_record(first_year: int, last_year: int, spike: date, missing_window_days: int = 0) -> DischargeRecord:
    """Background log(Q + 0.01) = 0 everywhere and one spike.

    A window of n pooled values holding one spike s and n - 1 zeros has mean s/n and SD s/sqrt(n),
    so the spike's deviation is (n - 1)/sqrt(n) SDs: flagged for n >= 38, not for n <= 37.
    """
    dates = days(date(first_year, 1, 1), date(last_year + 1, 1, 1))
    q = np.full(len(dates), 0.99)
    q[dates == spike.toordinal()] = 100.0
    missing = np.zeros(len(dates), dtype=np.bool_)
    for year in range(first_year, first_year + missing_window_days):
        missing[dates == spike.replace(year=year).toordinal()] = True
    return record(dates, q, missing)


@pytest.mark.parametrize(
    ("first_year", "last_year", "spike", "missing", "flagged"),
    [
        (2001, 2008, date(2005, 7, 15), 2, True),  # 8 years x 5 calendar days - 2 = 38
        (2001, 2008, date(2005, 7, 15), 3, False),  # 37
        (2001, 2008, date(2005, 1, 1), 0, True),  # Dec 30-31 of all years pool with Jan 1-3: 40
        (2002, 2005, date(2004, 2, 29), 0, False),  # Feb 27-Mar 2: 3 x 4 + 5 = 17
        (1993, 2000, date(1996, 2, 29), 0, False),  # 6 x 4 + 2 x 5 = 34
        (1991, 2000, date(1996, 2, 29), 0, True),  # 7 x 4 + 3 x 5 = 43
    ],
)
def test_seasonal_outlier_pools_five_calendar_days_across_years(
    first_year: int, last_year: int, spike: date, missing: int, flagged: bool
) -> None:
    screening = screen(spike_record(first_year, last_year, spike, missing), M3S)
    outliers = reason(screening.reasons, "seasonal_outlier")
    assert_array_equal(np.flatnonzero(outliers), [spike.toordinal() - screening.dates[0]] if flagged else [])


def test_later_observations_change_earlier_outlier_flags() -> None:
    spike = date(2003, 7, 15)
    early = screen(spike_record(2001, 2004, spike), M3S)  # 20 pooled values
    full = screen(spike_record(2001, 2008, spike), M3S)  # 40 pooled values
    position = spike.toordinal() - full.dates[0]
    assert not reason(early.reasons, "seasonal_outlier")[position]
    assert reason(full.reasons, "seasonal_outlier")[position]


def test_outlier_offset_is_applied_in_cubic_metres_per_second() -> None:
    # Pool of 40 for 15 July: 20 zeros, 19 values of 0.05 and one spike. In m3/s, zeros sit at log(0.01).
    dates = days(date(2001, 1, 1), date(2009, 1, 1))
    q = np.full(len(dates), np.nan)
    spike = date(2008, 7, 15).toordinal() - dates[0]
    others = [date(y, 7, d).toordinal() - dates[0] for y in range(2001, 2009) for d in range(13, 18)]
    others.remove(spike)
    q[others[:20]], q[others[20:]], q[spike] = 0.0, 0.05, 10.0
    missing = np.isnan(q)
    cfs = 0.028316846592

    def flagged(pool: np.ndarray) -> bool:
        x = np.log(pool + 0.01)
        return bool(abs(x[-1] - x.mean()) > 6 * x.std(ddof=1))

    pool = q[[*others, spike]]
    assert flagged(pool * cfs) != flagged(pool)
    for source, scale in [(DischargeSource("cfs", "ft3/s", cfs, None), cfs), (M3S, 1.0)]:
        outlier = reason(screen(record(dates, q, missing), source).reasons, "seasonal_outlier")
        assert outlier[spike] == flagged(pool * scale)


def test_reliable_provider_flags_replace_automated_checks() -> None:
    q = np.array([-1.0] + [2.0] * 12 + [3.0])
    flags = np.array(["A"] + [""] * 12 + ["P"])
    screening = screen(record(days(date(2000, 1, 1), date(2000, 1, 15)), q, flags=flags), FLAGGED)
    # Unflagged days in a partly flagged record count as reliable; negative and run checks are not applied.
    assert screening.route == "provider_flags"
    assert_array_equal(np.flatnonzero(screening.reasons.any(axis=1)), [13])
    assert_array_equal(np.flatnonzero(reason(screening.reasons, "provider_suspect")), [13])
    assert_array_equal(screening.flags, flags)


@pytest.mark.parametrize(
    ("source", "flags"), [(FLAGGED, ["M", "", "M"]), (M3S, ["A", "P", "A"])], ids=["no-verdicts", "discouraged"]
)
def test_records_without_usable_flags_use_automated_checks(source: DischargeSource, flags: list[str]) -> None:
    screening = screen(
        record(days(date(2000, 1, 1), date(2000, 1, 4)), np.array([1.0, -1, 1]), flags=np.array(flags)), source
    )
    assert screening.route == "automated"
    assert_array_equal(np.flatnonzero(screening.reasons.any(axis=1)), [1])
    assert_array_equal(screening.flags, flags)


def test_unknown_provider_flags_are_rejected() -> None:
    with pytest.raises(ValueError, match="X"):
        screen(record(np.array([1, 2]), np.ones(2), flags=np.array(["A", "X"])), FLAGGED)
