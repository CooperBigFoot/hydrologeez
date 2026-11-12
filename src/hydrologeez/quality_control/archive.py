"""Persistent retrospective streamflow QC archive and its reader.

The archive keeps original discharge, source missingness and provider flags beside separate
daily QC reasons, index values with usable-day counts and homogeneity assessments. It never
drops basins, repairs values or applies exclusions; experiments build their own masks from
the reader's aligned arrays.

Layout: ``metadata.json`` (written last), ``basins/<position>/`` with daily arrays and
per-resolution index arrays, and ``homogeneity/<resolution>_*.npy`` arrays [basin, index, ...].
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hydrologeez.quality_control.daily import (
    OUTLIER_OFFSET_M3S,
    OUTLIER_SDS,
    REASONS,
    RUN_LENGTH,
    WINDOW_HALF_WIDTH,
    DischargeRecord,
    DischargeSource,
    Route,
    screen,
)
from hydrologeez.quality_control.homogeneity import (
    CLASSIFICATIONS,
    CRITICAL_VALUES,
    MINIMUM_VALUES,
    TESTS,
    assess_indices,
)
from hydrologeez.quality_control.indices import (
    DAY_OF_YEAR,
    DIMENSIONLESS,
    INDICES,
    RELIABLE_DAYS,
    RESOLUTIONS,
    Resolution,
    compute_indices,
    period_end,
)

FORMAT = 1


def _method() -> dict[str, Any]:
    return {
        "reference": "Gudmundsson et al. (2018), GSIM Part 2, https://doi.org/10.5194/essd-10-787-2018",
        "screening": "retrospective full-record statistics; later observations can change earlier flags",
        "reasons": list(REASONS),
        "constant_run": {"longer_than_days": RUN_LENGTH, "values": "positive and equal"},
        "seasonal_outlier": {
            "transform": "log(Q + offset), Q in m3/s",
            "standard_deviations": OUTLIER_SDS,
            "window_calendar_days": 2 * WINDOW_HALF_WIDTH + 1,
            "calendar": "366-day cycle with own 29 February position; wraps at year end",
            "sd": "sample (ddof=1), includes tested value",
        },
        "outlier_offset": {"value": OUTLIER_OFFSET_M3S, "units": "m3/s"},
        "reliable_days": RELIABLE_DAYS,
        "indices": {resolution: list(INDICES[resolution]) for resolution in RESOLUTIONS},
        "homogeneity": {
            "tests": list(TESTS),
            "classifications": list(CLASSIFICATIONS),
            "minimum_values": MINIMUM_VALUES,
            "level": 0.01,
            "critical_values": CRITICAL_VALUES.tolist(),
        },
    }


def write_archive(output: Path, source: DischargeSource, records: Iterable[DischargeRecord]) -> None:
    """Screen each full basin record, compute indices and assess them; ``output`` must not exist."""
    output.mkdir(parents=True, exist_ok=False)
    basins: list[dict[str, Any]] = []
    seen: set[str] = set()
    assessments: dict[Resolution, list[list[Any]]] = {resolution: [] for resolution in RESOLUTIONS}
    for position, record in enumerate(records):
        if record.basin in seen:
            raise ValueError(f"Duplicate basin: {record.basin}")
        seen.add(record.basin)
        screening = screen(record, source)
        folder = output / "basins" / str(position)
        folder.mkdir(parents=True)
        np.save(folder / "discharge.npy", screening.discharge)
        np.save(folder / "missing.npy", screening.missing)
        np.save(folder / "flags.npy", screening.flags)
        np.save(folder / "reasons.npy", screening.reasons)
        periods: dict[str, list[int]] = {}
        for resolution in RESOLUTIONS:
            series = compute_indices(screening.dates, screening.discharge, screening.usable, resolution)
            np.save(folder / f"{resolution}_values.npy", series.values)
            np.save(folder / f"{resolution}_counts.npy", series.counts)
            periods[resolution] = [int(series.periods[0]), int(series.periods[-1] + 1)]
            assessments[resolution].append(assess_indices(series))
        basins.append(
            {
                "basin": record.basin,
                "start": int(screening.dates[0]),
                "stop": int(screening.dates[-1] + 1),
                "route": screening.route,
                "periods": periods,
            }
        )
    homogeneity = output / "homogeneity"
    homogeneity.mkdir()
    for resolution, rows in assessments.items():
        np.save(
            homogeneity / f"{resolution}_classification.npy",
            np.array([[CLASSIFICATIONS.index(h.classification) for h in row] for row in rows], dtype=np.int8),
        )
        np.save(homogeneity / f"{resolution}_count.npy", np.array([[h.count for h in row] for row in rows]))
        np.save(homogeneity / f"{resolution}_statistics.npy", np.array([[h.statistics for h in row] for row in rows]))
        np.save(homogeneity / f"{resolution}_rejected.npy", np.array([[h.rejected for h in row] for row in rows]))
    metadata = {
        "format": FORMAT,
        "source": {
            "provenance": source.provenance,
            "units": source.units,
            "cubic_metres_per_second_per_unit": source.cubic_metres_per_second_per_unit,
            "flag_meanings": None if source.flag_meanings is None else dict(source.flag_meanings),
        },
        "calendar": "Gregorian; ordinals; stop exclusive",
        "method": _method(),
        "basins": basins,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


@dataclass(frozen=True)
class DailyQuality:
    """Daily arrays [day, basin]; days outside a basin's record are missing with NaN discharge.

    ``reasons`` maps each reason to its flags. Only the basin's route is evaluated:
    ``provider_suspect`` for "provider_flags", the automated reasons for "automated".
    """

    ids: tuple[str, ...]
    dates: NDArray[np.int64]
    discharge: NDArray[np.float64]
    missing: NDArray[np.bool_]
    flags: NDArray[np.str_]
    reasons: dict[str, NDArray[np.bool_]]
    routes: tuple[Route, ...]

    @property
    def observed(self) -> NDArray[np.bool_]:
        return ~self.missing & np.isfinite(self.discharge)

    @property
    def suspect(self) -> NDArray[np.bool_]:
        """Days rejected by the basin's QC route; index calculations excluded them."""
        return np.logical_or.reduce(list(self.reasons.values()))


@dataclass(frozen=True)
class IndexQuality:
    """Index values [period, basin, index] on the union of the selected basins' periods."""

    ids: tuple[str, ...]
    resolution: Resolution
    names: tuple[str, ...]
    units: tuple[str, ...]
    period_end: NDArray[np.int64]
    values: NDArray[np.float64]
    counts: NDArray[np.int64]
    reliable: NDArray[np.bool_]


@dataclass(frozen=True)
class HomogeneityQuality:
    """Assessments [basin, index]; statistics and rejections add a trailing test axis."""

    ids: tuple[str, ...]
    resolution: Resolution
    names: tuple[str, ...]
    tests: tuple[str, ...]
    classification: NDArray[np.str_]
    count: NDArray[np.int64]
    statistics: NDArray[np.float64]
    rejected: NDArray[np.bool_]


class StreamflowQualityArchive:
    """Metadata-only open; selections load only the requested basins."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.metadata: dict[str, Any] = json.loads((directory / "metadata.json").read_text())
        if self.metadata["format"] != FORMAT:
            raise ValueError("Unsupported streamflow QC archive format")
        self.ids = tuple(b["basin"] for b in self.metadata["basins"])
        self._positions = {basin: i for i, basin in enumerate(self.ids)}

    def _select(self, ids: Sequence[str]) -> list[int]:
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Select unique nonempty basins")
        unknown = [basin for basin in ids if basin not in self._positions]
        if unknown:
            raise ValueError(f"Unknown basins: {unknown}")
        return [self._positions[basin] for basin in ids]

    def _load(self, position: int, name: str) -> NDArray[Any]:
        return np.load(self.directory / "basins" / str(position) / f"{name}.npy", mmap_mode="r")

    def daily(self, ids: Sequence[str], start: str, stop: str) -> DailyQuality:
        """Daily quality for [start, stop) ISO dates in caller basin order."""
        first, last = date.fromisoformat(start).toordinal(), date.fromisoformat(stop).toordinal()
        if last <= first:
            raise ValueError("Select a nonempty date window")
        positions = self._select(ids)
        shape = (last - first, len(ids))
        discharge = np.full(shape, np.nan)
        missing = np.ones(shape, dtype=np.bool_)
        flags = np.full(shape, "", dtype=np.result_type(*(self._load(p, "flags").dtype for p in positions)))
        reasons = np.zeros((*shape, len(REASONS)), dtype=np.bool_)
        for column, position in enumerate(positions):
            basin = self.metadata["basins"][position]
            a, z = max(first, basin["start"]), min(last, basin["stop"])
            if z <= a:
                continue
            source = slice(a - basin["start"], z - basin["start"])
            target = slice(a - first, z - first)
            discharge[target, column] = self._load(position, "discharge")[source]
            missing[target, column] = self._load(position, "missing")[source]
            flags[target, column] = self._load(position, "flags")[source]
            reasons[target, column] = self._load(position, "reasons")[source]
        return DailyQuality(
            tuple(ids),
            np.arange(first, last, dtype=np.int64),
            discharge,
            missing,
            flags,
            {name: reasons[:, :, i] for i, name in enumerate(REASONS)},
            tuple(self.metadata["basins"][p]["route"] for p in positions),
        )

    def indices(self, ids: Sequence[str], resolution: Resolution) -> IndexQuality:
        positions = self._select(ids)
        spans = [self.metadata["basins"][p]["periods"][resolution] for p in positions]
        first, last = min(s[0] for s in spans), max(s[1] for s in spans)
        names = INDICES[resolution]
        values = np.full((last - first, len(ids), len(names)), np.nan)
        counts = np.zeros((last - first, len(ids)), dtype=np.int64)
        for column, (position, (a, z)) in enumerate(zip(positions, spans, strict=True)):
            values[a - first : z - first, column] = self._load(position, f"{resolution}_values")
            counts[a - first : z - first, column] = self._load(position, f"{resolution}_counts")
        units = self.metadata["source"]["units"]
        return IndexQuality(
            tuple(ids),
            resolution,
            names,
            tuple("-" if n in DIMENSIONLESS else "day of year" if n in DAY_OF_YEAR else units for n in names),
            period_end(resolution, np.arange(first, last, dtype=np.int64)),
            values,
            counts,
            counts >= self.metadata["method"]["reliable_days"][resolution],
        )

    def homogeneity(self, ids: Sequence[str], resolution: Resolution) -> HomogeneityQuality:
        positions = self._select(ids)

        def load(name: str) -> NDArray[Any]:
            return np.load(self.directory / "homogeneity" / f"{resolution}_{name}.npy", mmap_mode="r")[positions]

        return HomogeneityQuality(
            tuple(ids),
            resolution,
            INDICES[resolution],
            TESTS,
            np.array(CLASSIFICATIONS)[load("classification")],
            load("count"),
            load("statistics"),
            load("rejected"),
        )
