from datetime import date

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hydrologeez.quality_control.indices import Resolution, compute_indices


def days(start: date, stop: date) -> np.ndarray:
    return np.arange(start.toordinal(), stop.toordinal(), dtype=np.int64)


def test_yearly_indices_of_day_of_year_flow() -> None:
    dates = days(date(2001, 1, 1), date(2002, 1, 1))
    q = np.arange(1.0, 366)
    series = compute_indices(dates, q, np.ones(365, dtype=np.bool_), "yearly")
    n = 365
    sd = np.sqrt(n * (n + 1) / 12)
    expected = {
        "MEAN": 183,
        "SD": sd,
        "CV": sd / 183,
        "IQR": 182,
        "MIN": 1,
        "MAX": 365,
        "MIN7": 4,  # first full backward window ends on day 7
        "MAX7": 362,
        **{f"P{p}": 1 + p / 100 * 364 for p in range(10, 100, 10)},
        "CT": 258,  # 258 * 259 / 2 >= 365 * 366 / 4 > 257 * 258 / 2
        "DOYMIN": 1,
        "DOYMAX": 365,
        "DOYMIN7": 7,
        "DOYMAX7": 365,
        "GINI": (n - 1) / (3 * n),
    }
    assert series.names == tuple(expected)
    assert_array_equal(series.periods, [2001])
    assert_allclose(series.values[0], list(expected.values()))
    assert_array_equal(series.counts, [365])
    assert series.reliable.tolist() == [True]


@pytest.mark.parametrize(
    ("resolution", "removed", "period", "count", "reliable"),
    [
        ("yearly", (date(2001, 3, 1), date(2001, 3, 16)), 2001, 350, True),
        ("yearly", (date(2001, 3, 1), date(2001, 3, 17)), 2001, 349, False),
        ("seasonal", (date(2000, 12, 1), date(2000, 12, 6)), 2001 * 4, 85, True),  # DJF 2001 starts in Dec 2000
        ("seasonal", (date(2001, 2, 1), date(2001, 2, 7)), 2001 * 4, 84, False),
        ("monthly", (date(2001, 2, 1), date(2001, 2, 4)), 2001 * 12 + 1, 25, True),
        ("monthly", (date(2001, 2, 1), date(2001, 2, 5)), 2001 * 12 + 1, 24, False),
    ],
)
def test_reliability_follows_usable_day_thresholds(
    resolution: Resolution, removed: tuple[date, date], period: int, count: int, reliable: bool
) -> None:
    dates = days(date(2000, 12, 1), date(2002, 1, 1))
    usable = ~np.isin(dates, days(*removed))
    series = compute_indices(dates, np.ones(len(dates)), usable, resolution)
    position = int(np.flatnonzero(series.periods == period)[0])
    assert series.counts[position] == count
    assert series.reliable[position] == reliable


def test_excluded_values_do_not_enter_indices_and_empty_periods_are_undefined() -> None:
    dates = days(date(2001, 1, 1), date(2001, 4, 1))
    q = np.ones(len(dates))
    q[:31] = 1000.0
    usable = np.ones(len(dates), dtype=np.bool_)
    usable[:31] = False  # all of January
    series = compute_indices(dates, q, usable, "monthly")
    assert_array_equal(series.counts, [0, 28, 31])
    assert np.isnan(series.values[0]).all()
    assert_allclose(series.values[1:, series.names.index("MAX")], [1, 1])
    assert_array_equal(series.period_end, [date(2001, m, d).toordinal() for m, d in [(1, 31), (2, 28), (3, 31)]])
