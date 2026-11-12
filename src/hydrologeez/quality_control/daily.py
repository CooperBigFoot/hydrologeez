"""Daily discharge screening (GSIM Part 2, Sect. 2).

Each basin record follows one route. Records whose source supplies reliable provider flags,
and that carry at least one reliable or suspect flag, use those flags (Sect. 2.2): unflagged
days count as reliable and only provider-suspect days are rejected. All other records use
the automated checks of Sect. 2.3:

- ``negative``: Q < 0.
- ``constant_run``: more than 10 consecutive calendar days with the same positive value.
  A missing, nonfinite or absent day ends a run; every day of a qualifying run is flagged.
- ``seasonal_outlier``: log(Q + 0.01), with Q in m³/s, deviates from the mean by more than
  6 standard deviations. Mean and sample SD (ddof=1) pool all years for the 5 calendar days
  centred on the day's calendar day, including the tested value. Calendar days use a
  366-day cycle in which 29 February has its own position, so non-leap years contribute 4
  days to windows covering it; windows wrap from 31 December to 1 January. Negative values
  do not enter the statistics and are not tested. Windows with fewer than 2 values flag nothing.

The offset 0.01 is defined in m³/s and is not invariant under unit conversion, so sources
declare their conversion factor; stored discharge stays in source units.

Screening is retrospective: statistics use the full basin record, so later observations can
change earlier ``seasonal_outlier`` flags. Results never depend on how records were chunked.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type FlagMeaning = Literal["reliable", "suspect", "unflagged"]
type Route = Literal["provider_flags", "automated"]

REASONS = ("provider_suspect", "negative", "constant_run", "seasonal_outlier")
RUN_LENGTH = 10
OUTLIER_SDS = 6.0
OUTLIER_OFFSET_M3S = 0.01
WINDOW_HALF_WIDTH = 2

EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


@dataclass(frozen=True)
class DischargeSource:
    """Discharge provenance, units and provider-flag interpretation.

    ``flag_meanings`` translates every nonempty provider flag (GSIM Table 2); use None when the
    source has no flags or its provider advises against using them. Empty flags are unflagged.
    """

    provenance: str
    units: str
    cubic_metres_per_second_per_unit: float
    flag_meanings: Mapping[str, FlagMeaning] | None


@dataclass(frozen=True)
class DischargeRecord:
    """Full daily record of one basin: Gregorian ordinals strictly increasing, gaps allowed."""

    basin: str
    dates: NDArray[np.int64]
    discharge: NDArray[np.float64]
    missing: NDArray[np.bool_]
    flags: NDArray[np.str_]


@dataclass(frozen=True)
class DailyScreening:
    """Record on a gap-free daily timeline; absent dates are missing with NaN discharge."""

    dates: NDArray[np.int64]
    discharge: NDArray[np.float64]
    missing: NDArray[np.bool_]
    flags: NDArray[np.str_]
    reasons: NDArray[np.bool_]  # [day, reason] in REASONS order
    route: Route

    @property
    def usable(self) -> NDArray[np.bool_]:
        """Observed, finite and not suspect: the values entering index calculations."""
        return ~self.missing & np.isfinite(self.discharge) & ~self.reasons.any(axis=1)


def calendar(dates: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Year, month and day of year of Gregorian ordinals."""
    days = (dates - EPOCH_ORDINAL).astype("datetime64[D]")
    years = days.astype("datetime64[Y]")
    months = days.astype("datetime64[M]")
    year = years.astype(np.int64) + 1970
    month = (months - years.astype("datetime64[M]")).astype(np.int64) + 1
    day_of_year = (days - years.astype("datetime64[D]")).astype(np.int64) + 1
    return year, month, day_of_year


def screen(record: DischargeRecord, source: DischargeSource) -> DailyScreening:
    """Screen one full basin record; never pass a partial chunk of a record."""
    dates = record.dates
    if not len(dates):
        raise ValueError("Discharge record is empty")
    if np.any(np.diff(dates) <= 0):
        raise ValueError("Discharge dates must be strictly increasing")
    if not len(record.discharge) == len(record.missing) == len(record.flags) == len(dates):
        raise ValueError("Discharge record arrays must align with dates")
    timeline = np.arange(dates[0], dates[-1] + 1, dtype=np.int64)
    position = dates - dates[0]
    discharge = np.full(len(timeline), np.nan)
    discharge[position] = record.discharge
    missing = np.ones(len(timeline), dtype=np.bool_)
    missing[position] = record.missing
    flags = np.full(len(timeline), "", dtype=record.flags.dtype)
    flags[position] = record.flags
    reasons = np.zeros((len(timeline), len(REASONS)), dtype=np.bool_)
    meanings = _meanings(flags, source.flag_meanings)
    if meanings is not None and np.any(meanings != "unflagged"):
        reasons[:, 0] = meanings == "suspect"
        return DailyScreening(timeline, discharge, missing, flags, reasons, "provider_flags")
    observed = ~missing & np.isfinite(discharge)
    reasons[:, 1] = observed & (discharge < 0)
    reasons[:, 2] = _constant_runs(discharge, observed)
    reasons[:, 3] = _seasonal_outliers(timeline, discharge * source.cubic_metres_per_second_per_unit, observed)
    return DailyScreening(timeline, discharge, missing, flags, reasons, "automated")


def _meanings(flags: NDArray[np.str_], meanings: Mapping[str, FlagMeaning] | None) -> NDArray[np.str_] | None:
    if meanings is None:
        return None
    unknown = set(np.unique(flags).tolist()) - set(meanings) - {""}
    if unknown:
        raise ValueError(f"Provider flags without a declared meaning: {sorted(unknown)}")
    table = {"": "unflagged", **meanings}
    return np.array([table[flag] for flag in flags.tolist()])


def _constant_runs(discharge: NDArray[np.float64], observed: NDArray[np.bool_]) -> NDArray[np.bool_]:
    positive = observed & (discharge > 0)
    continues = np.zeros(len(discharge), dtype=np.bool_)
    continues[1:] = positive[1:] & positive[:-1] & (discharge[1:] == discharge[:-1])
    run = np.cumsum(~continues)
    return positive & (np.bincount(run)[run] > RUN_LENGTH)


def _seasonal_outliers(
    dates: NDArray[np.int64], discharge_m3s: NDArray[np.float64], observed: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    tested = observed & (discharge_m3s >= 0)
    year, _, day_of_year = calendar(dates[tested])
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    slot = day_of_year - 1 + (~leap & (day_of_year > 59))
    x = np.log(discharge_m3s[tested] + OUTLIER_OFFSET_M3S)
    offsets = range(-WINDOW_HALF_WIDTH, WINDOW_HALF_WIDTH + 1)
    windows = [(slot - offset) % 366 for offset in offsets]
    count = sum(np.bincount(w, minlength=366) for w in windows)
    total = sum(np.bincount(w, weights=x, minlength=366) for w in windows)
    mean = np.divide(total, count, out=np.full(366, np.nan), where=count > 0)
    squares = sum(np.bincount(w, weights=(x - mean[w]) ** 2, minlength=366) for w in windows)
    sd = np.sqrt(np.divide(squares, count - 1, out=np.full(366, np.nan), where=count > 1))
    outlier = np.zeros(len(dates), dtype=np.bool_)
    outlier[tested] = np.abs(x - mean[slot]) > OUTLIER_SDS * sd[slot]
    return outlier
