"""Yearly, seasonal and monthly streamflow indices (GSIM Part 2, Sect. 3, Tables 3-4).

Indices use only usable days (observed, finite, not suspect); suspect values remain in the
archive. Seasons are DJF, MAM, JJA and SON; a DJF season belongs to the year of its January.
Each period spans the calendar, so days outside the record count as unusable. Index values are
computed whenever defined, and ``counts`` give the usable days. A period is reliable with at
least 350 (yearly), 85 (seasonal) or 25 (monthly) usable days.

Numerical conventions where the paper is silent: SD uses ddof=1; percentiles and IQR use
linear interpolation (numpy default, R type 7); MIN7/MAX7 use the backward 7-day mean over
the full record, defined only when all 7 days are usable and assigned to the window's last
day; day-of-year indices take the first day reaching the extreme; CT is the first day on which
the cumulative usable flow reaches half of the period total. CV, CT and GINI are undefined
(NaN) for zero total flow.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from hydrologeez.quality_control.daily import EPOCH_ORDINAL, calendar

type Resolution = Literal["yearly", "seasonal", "monthly"]

RESOLUTIONS: tuple[Resolution, ...] = ("yearly", "seasonal", "monthly")
PERIODS_PER_YEAR = {"yearly": 1, "seasonal": 4, "monthly": 12}
RELIABLE_DAYS = {"yearly": 350, "seasonal": 85, "monthly": 25}
SEASONS = ("DJF", "MAM", "JJA", "SON")

_COMMON = ("MEAN", "SD", "CV", "IQR", "MIN", "MAX", "MIN7", "MAX7")
_PERCENTILES = tuple(f"P{p}" for p in range(10, 100, 10))
_TIMING = ("CT", "DOYMIN", "DOYMAX", "DOYMIN7", "DOYMAX7", "GINI")
INDICES: dict[Resolution, tuple[str, ...]] = {
    "yearly": _COMMON + _PERCENTILES + _TIMING,
    "seasonal": _COMMON + _PERCENTILES,
    "monthly": _COMMON,
}
DIMENSIONLESS = ("CV", "GINI")
DAY_OF_YEAR = ("CT", "DOYMIN", "DOYMAX", "DOYMIN7", "DOYMAX7")


@dataclass(frozen=True)
class IndexSeries:
    """Index values [period, index] for consecutive period keys.

    Keys are the year (yearly), year * 4 + season (seasonal) or year * 12 + month - 1 (monthly).
    """

    resolution: Resolution
    names: tuple[str, ...]
    periods: NDArray[np.int64]
    values: NDArray[np.float64]
    counts: NDArray[np.int64]

    @property
    def reliable(self) -> NDArray[np.bool_]:
        return self.counts >= RELIABLE_DAYS[self.resolution]

    @property
    def year(self) -> NDArray[np.int64]:
        return self.periods // PERIODS_PER_YEAR[self.resolution]

    @property
    def position(self) -> NDArray[np.int64]:
        """Season or month position within the year (0 for yearly)."""
        return self.periods % PERIODS_PER_YEAR[self.resolution]

    @property
    def period_end(self) -> NDArray[np.int64]:
        return period_end(self.resolution, self.periods)


def period_keys(resolution: Resolution, dates: NDArray[np.int64]) -> NDArray[np.int64]:
    year, month, _ = calendar(dates)
    if resolution == "yearly":
        return year
    if resolution == "monthly":
        return year * 12 + month - 1
    return (year + (month == 12)) * 4 + month % 12 // 3


def period_end(resolution: Resolution, periods: NDArray[np.int64]) -> NDArray[np.int64]:
    """Gregorian ordinal of the last day of each period."""
    following = {"yearly": (periods + 1) * 12, "seasonal": periods * 3 + 2, "monthly": periods + 1}[resolution]
    month = np.datetime64("1970-01", "M") + (following - 1970 * 12)
    return (month.astype("datetime64[D]") - np.datetime64("1970-01-01", "D")).astype(np.int64) + EPOCH_ORDINAL - 1


def compute_indices(
    dates: NDArray[np.int64], discharge: NDArray[np.float64], usable: NDArray[np.bool_], resolution: Resolution
) -> IndexSeries:
    """Indices of a gap-free daily record (consecutive ordinals)."""
    q = np.where(usable, discharge, np.nan)
    q7 = np.full(len(q), np.nan)
    if len(q) >= 7:
        q7[6:] = np.lib.stride_tricks.sliding_window_view(q, 7).mean(axis=1)
    _, _, day_of_year = calendar(dates)
    keys = period_keys(resolution, dates)
    periods = np.arange(keys[0], keys[-1] + 1, dtype=np.int64)
    names = INDICES[resolution]
    values = np.full((len(periods), len(names)), np.nan)
    counts = np.zeros(len(periods), dtype=np.int64)
    bounds = np.searchsorted(keys, np.append(periods, periods[-1] + 1))
    for i in range(len(periods)):
        days = slice(bounds[i], bounds[i + 1])
        selected = usable[days]
        counts[i] = selected.sum()
        if counts[i]:
            row = _period_indices(q[days][selected], q7[days], day_of_year[days][selected], day_of_year[days])
            values[i] = row[: len(names)]
    return IndexSeries(resolution, names, periods, values, counts)


def _period_indices(
    q: NDArray[np.float64], q7: NDArray[np.float64], doy: NDArray[np.int64], doy7: NDArray[np.int64]
) -> list[float]:
    n = len(q)
    mean = q.mean()
    total = q.sum()
    sd = q.std(ddof=1) if n > 1 else np.nan
    p25, p75, *percentiles = np.percentile(q, [25, 75, *range(10, 100, 10)])
    defined7 = np.isfinite(q7)
    smooth = q7[defined7]
    ordered = np.sort(q)
    nan = float("nan")
    cv = sd / mean if mean else nan
    gini = (n + 1 - 2 * np.sum((n - np.arange(n)) * ordered) / total) / n if total else nan
    centre = doy[np.argmax(np.cumsum(q) >= total / 2)] if total else nan
    return [
        mean,
        sd,
        cv,
        p75 - p25,
        q.min(),
        q.max(),
        smooth.min() if len(smooth) else nan,
        smooth.max() if len(smooth) else nan,
        *percentiles,
        centre,
        doy[np.argmin(q)],
        doy[np.argmax(q)],
        doy7[defined7][np.argmin(smooth)] if len(smooth) else nan,
        doy7[defined7][np.argmax(smooth)] if len(smooth) else nan,
        gini,
    ]
