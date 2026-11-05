"""Complete-year seasonal climatologies and a prepared CAMELS forcing adapter."""

from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from hydrologeez.camels import Camels

CALENDAR_POLICY = (
    "Gregorian complete years; Jan 1 anchored nonleap day index // 7 capped at 51; "
    "52 bins, final bin 8 days; Feb 29 shares Feb 28 bin; daily means per year/bin "
    "then equal-weight mean across years; linear Jan-Dec order, no circular alignment; stop exclusive"
)


@dataclass(frozen=True)
class SeasonalCycles:
    """Original-unit seasonal means, ordered basin × 52 steps × variable."""

    ids: tuple[str, ...]
    variables: tuple[str, ...]
    units: tuple[str, ...]
    values: NDArray[np.float64]
    start: str
    stop: str
    provenance: dict[str, str]


def seasonal_cycles(
    dates: NDArray[np.int64],
    values: NDArray[np.float64],
    ids: tuple[str, ...],
    variables: tuple[str, ...],
    units: tuple[str, ...],
    provenance: dict[str, str] | None = None,
) -> SeasonalCycles:
    """Average corresponding weekly periods across complete years.

    Every daily value must be finite. No dates, basins or variables are omitted,
    repaired or imputed. Means retain input units (precipitation is not summed).
    Leap days contribute to their year's February 28 bin; years have equal weight.
    """
    if not ids or any(not isinstance(g, str) or not g for g in ids) or len(set(ids)) != len(ids):
        raise ValueError("Basin identities must be unique nonempty strings")
    if not variables or not all(variables) or len(set(variables)) != len(variables) or len(units) != len(variables):
        raise ValueError("Select unique nonempty variables and one unit per variable")
    if dates.ndim != 1 or not len(dates) or not np.issubdtype(dates.dtype, np.integer):
        raise ValueError("Dates must be a nonempty one-dimensional integer ordinal array")
    if np.any(np.diff(dates) != 1):
        raise ValueError("Dates must be daily, increasing and complete; no gaps or duplicates")
    if values.shape != (len(ids), len(dates), len(variables)):
        raise ValueError("Values shape must be basin × day × variable, matching axis metadata")
    calendar = [date.fromordinal(int(d)) for d in dates]
    if (calendar[0].month, calendar[0].day) != (1, 1) or (calendar[-1].month, calendar[-1].day) != (12, 31):
        raise ValueError("Seasonal climatologies require complete calendar years, Jan 1 through Dec 31")
    invalid = ~np.isfinite(values)
    if invalid.any():
        basin, day, channel = np.argwhere(invalid)[0]
        raise ValueError(
            f"Invalid input: basin {ids[basin]}, variable {variables[channel]}, date {calendar[day]}; "
            f"{int(invalid.sum())} nonfinite values; no imputation or exclusions"
        )
    years = np.array([d.year for d in calendar])
    bins = np.array(
        [
            min((date(2001, d.month, min(d.day, 28) if d.month == 2 else d.day).timetuple().tm_yday - 1) // 7, 51)
            for d in calendar
        ]
    )
    annual = [
        np.stack([values[:, (years == year) & (bins == step), :].mean(axis=1) for step in range(52)], axis=1)
        for year in np.unique(years)
    ]
    return SeasonalCycles(
        ids,
        variables,
        units,
        np.mean(annual, axis=0),
        calendar[0].isoformat(),
        date.fromordinal(int(dates[-1]) + 1).isoformat(),
        {**(provenance or {}), "calendar_policy": CALENDAR_POLICY},
    )


def camels_cycles(
    store: Camels,
    ids: list[str],
    start: str,
    stop: str,
    variables: tuple[str, ...] = ("precipitation", "mean_temperature"),
) -> SeasonalCycles:
    """Read selected prepared forcing channels over [start, stop), without Q.

    Unselected channels may be missing. Selected precipitation and PET must be
    nonnegative, and every selected value must be finite. Forcing files and
    bounds follow the prepared CAMELS format; no observation files are read.
    """
    first, last = date.fromisoformat(start).toordinal(), date.fromisoformat(stop).toordinal()
    if not ids or not all(isinstance(g, str) and g for g in ids) or len(set(ids)) != len(ids):
        raise ValueError("Basin identities must be unique nonempty strings")
    if last <= first:
        raise ValueError("Select a nonempty date window")
    available: list[str] = store.metadata["forcing_variables"]
    if not variables or len(set(variables)) != len(variables) or any(v not in available for v in variables):
        raise ValueError(f"Select unique known forcing variables; requested {variables}, available {available}")
    columns = [available.index(v) for v in variables]
    units = tuple(str(store.metadata["forcing_units"][i]) for i in columns)
    basins = {b.gauge_id: b for b in store.basins}
    values = np.empty((len(ids), last - first, len(variables)), dtype=np.float64)
    for i, gauge in enumerate(ids):
        if gauge not in basins:
            raise ValueError(f"Unknown CAMELS basin: {gauge}")
        basin = basins[gauge]
        if first < basin.forcing_start or last > basin.forcing_stop:
            raise ValueError(f"Selection exceeds forcing coverage for basin {gauge}")
        forcing = np.load(store.directory / gauge / "forcing.npy", mmap_mode="r", allow_pickle=False)
        selected = forcing[first - basin.forcing_start : last - basin.forcing_start, columns]
        if selected.shape != values[i].shape:
            raise ValueError(f"Incomplete forcing file for basin {gauge}: expected shape {values[i].shape}")
        values[i] = selected
        for channel, variable in enumerate(variables):
            if variable in ("precipitation", "oudin_pet"):
                invalid = values[i, :, channel] < 0
                if invalid.any():
                    day = date.fromordinal(first + int(np.flatnonzero(invalid)[0]))
                    raise ValueError(f"Invalid input: basin {gauge}, variable {variable}, date {day}; negative forcing")
    provenance = {"source": str(store.metadata["source"]), "prepared_directory": str(store.directory.resolve())}
    if "oudin_pet" in variables and "pet" in store.metadata:
        provenance["pet"] = str(store.metadata["pet"])
    return seasonal_cycles(np.arange(first, last, dtype=np.int64), values, tuple(ids), variables, units, provenance)
