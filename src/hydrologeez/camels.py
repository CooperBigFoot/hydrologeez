"""Local CAMELS-US daily inputs, static attributes, and reusable selections."""

import csv
import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hydrologeez.hbv_light import Forcing
from hydrologeez.pet import oudin

Array = NDArray[np.float64]


def prepare_attributes(raw: Path, output: Path) -> None:
    """Prepare seven CAMELS-US static tables without daily inputs or network access.

    Write typed columns and source metadata to JSON in a new directory.
    Source NA is missing; categorical whitespace and string codes are retained.
    """
    metadata = json.loads(files("hydrologeez").joinpath("camels_attributes.json").read_text())
    fields: dict[str, dict[str, str]] = metadata["fields"]
    columns: dict[str, list[str | float | None]] = {}
    ids: list[str] | None = None
    for name in ("topo", "clim", "soil", "vege", "geol", "hydro", "name"):
        with (raw / f"camels_{name}.txt").open(newline="") as source:
            reader = csv.DictReader(source, delimiter=";")
            rows = list(reader)
            names = reader.fieldnames
        if not names or "gauge_id" not in names or len(set(names)) != len(names):
            raise ValueError("Attribute tables need unique fields and gauge_id")
        current = sorted(row["gauge_id"] for row in rows)
        if not current or not all(current) or len(set(current)) != len(current) or (ids is not None and current != ids):
            raise ValueError("Attribute tables must agree on unique gauge identities")
        ids = current
        rows.sort(key=lambda row: row["gauge_id"])
        for field in names:
            if field == "gauge_id":
                continue
            if field in columns or field not in fields or fields[field]["source_file"] != f"camels_{name}.txt":
                raise ValueError(f"Unexpected or duplicate CAMELS attribute: {field}")
            columns[field] = [
                None if row[field] == "NA" else row[field] if fields[field]["dtype"] == "string" else float(row[field])
                for row in rows
            ]
    metadata["fields"] = {field: fields[field] for field in ("gauge_id", *columns)}
    payload = {"format": 1, "ids": ids, "columns": columns, "metadata": metadata}
    # Validate serializability before creating the completion marker.
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "attributes.json").write_text(text)


def read_attributes(directory: Path, ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a reusable labeled table, optionally in caller basin order.

    Rows are string gauge IDs; columns retain source names. DataFrame.attrs
    carries field descriptions, units, provenance and source missing semantics.
    This reads only attributes.json, never raw tables or daily records.
    """
    payload = json.loads((directory / "attributes.json").read_text())
    if payload["format"] != 1:
        raise ValueError("Unsupported CAMELS attributes format")
    selected = payload["ids"] if ids is None else list(ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(payload["ids"]):
        raise ValueError("Select unique nonempty known CAMELS basins")
    metadata = payload["metadata"]
    table = pd.DataFrame(
        {
            field: pd.array(values, dtype=metadata["fields"][field]["dtype"])
            for field, values in payload["columns"].items()
        },
        index=pd.Index(payload["ids"], name="gauge_id", dtype="string"),
    )
    table.attrs = metadata
    return table.loc[selected].copy()


def prepare_forcing(rows: Array, latitude: float) -> tuple[NDArray[np.int64], Array]:
    """Retain Gregorian labels, marking invalid source forcing rather than repairing it."""
    dates = np.array([date(int(y), int(m), int(d)).toordinal() for y, m, d in rows[:, :3]], dtype=np.int64)
    if not len(dates) or np.any(np.diff(dates) <= 0):
        raise ValueError("Forcing dates must be unique and increasing")
    precipitation = rows[:, 5].copy()
    temperature = (rows[:, 8] + rows[:, 9]) / 2
    precipitation[~np.isfinite(precipitation) | (precipitation < 0)] = np.nan
    invalid = ~np.isfinite(rows[:, 8:10]).all(axis=1) | (rows[:, 8:10] <= -999).any(axis=1)
    temperature[invalid] = np.nan
    day = np.array([date.fromordinal(int(d)).timetuple().tm_yday for d in dates], dtype=np.int64)
    return dates, np.column_stack((precipitation, temperature, oudin(temperature, np.asarray(latitude), day)))


@dataclass(frozen=True)
class Basin:
    gauge_id: str
    latitude: float
    header_area_m2: float
    usgs_total_km2: float | None
    usgs_contributing_km2: float | None
    forcing_start: int
    forcing_stop: int
    observation_start: int
    observation_stop: int
    forcing_member: str
    observation_member: str
    missing_forcing_dates: int
    invalid_forcing_rows: int
    published_missing_observations: int
    absent_observation_dates: int  # Missing rows within observation bounds, not forcing bounds.


@dataclass(frozen=True)
class Observations:
    dates: NDArray[np.int64]
    discharge_cfs: Array
    discharge_mm_day: Array
    qc: NDArray[np.str_]
    present: NDArray[np.bool_]


@dataclass(frozen=True)
class Selection(Observations):
    ids: tuple[str, ...]
    forcing: Forcing


def _dense(dates: NDArray[np.int64], values: Array) -> Array:
    dense = np.full((int(dates[-1] - dates[0] + 1), *values.shape[1:]), np.nan)
    dense[dates - dates[0]] = values
    return dense


def prepare(raw: Path, output: Path, usgs_areas: Path) -> None:
    """Prepare every attribute identity; write metadata last as completion marker.

    USGS snapshot JSON: source (provenance), basins keyed by gauge ID with
    total_km2 and contributing_km2 (positive numbers or null, never substitutes).
    Existing output directories are rejected; raw inputs are never changed.
    """
    tables = {
        name: list(csv.DictReader((raw / f"camels_{name}.txt").read_text().splitlines(), delimiter=";"))
        for name in ("topo", "name", "clim", "hydro", "soil", "geol", "vege")
    }
    ids = sorted(row["gauge_id"] for row in tables["topo"])
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(sorted(row["gauge_id"] for row in table) != ids for table in tables.values())
    ):
        raise ValueError("Attribute tables must agree on unique gauge identities")
    snapshot = json.loads(usgs_areas.read_text())
    areas = snapshot["basins"]
    if not snapshot["source"] or not set(ids) <= areas.keys():
        raise ValueError("USGS snapshot must identify its source and cover every gauge")
    for gauge in ids:
        for key in ("total_km2", "contributing_km2"):
            value = areas[gauge][key]
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("USGS areas must be positive or null")
    output.mkdir(parents=True, exist_ok=False)
    records: list[Basin] = []
    with zipfile.ZipFile(raw / "basin_timeseries_v1p2_metForcing_obsFlow.zip") as archive:

        def members(fragment: str, suffix: str) -> dict[str, str]:
            result: dict[str, str] = {}
            for name in archive.namelist():
                if fragment in name and name.endswith(suffix):
                    gauge = Path(name).name.removesuffix(suffix)
                    if gauge in result:
                        raise ValueError(f"Duplicate source member for {gauge}")
                    result[gauge] = name
            return result

        forcings = members("/basin_mean_forcing/daymet/", "_lump_cida_forcing_leap.txt")
        flows = members("/usgs_streamflow/", "_streamflow_qc.txt")
        for row in sorted(tables["topo"], key=lambda r: r["gauge_id"]):
            gauge = row["gauge_id"]
            text = archive.read(forcings[gauge]).decode()
            lines = text.splitlines()
            if lines[3].split() != [
                "Year",
                "Mnth",
                "Day",
                "Hr",
                "dayl(s)",
                "prcp(mm/day)",
                "srad(W/m2)",
                "swe(mm)",
                "tmax(C)",
                "tmin(C)",
                "vp(Pa)",
            ]:
                raise ValueError("Unexpected forcing columns/units")
            area = float(lines[2])
            latitude = float(row["gauge_lat"])
            if not np.isfinite(area) or area <= 0 or not np.isfinite(latitude) or abs(latitude) > 90:
                raise ValueError("Invalid header area or attribute latitude")
            dates, values = prepare_forcing(np.loadtxt(io.StringIO(text), skiprows=4, ndmin=2), latitude)
            dense = _dense(dates, values)
            folder = output / gauge
            folder.mkdir()
            np.save(folder / "forcing.npy", dense)
            observed = [line.split() for line in archive.read(flows[gauge]).decode().splitlines() if line.strip()]
            if any(len(r) != 6 or r[0] != gauge for r in observed):
                raise ValueError("Unexpected observation columns or gauge")
            odates = np.array([date(*map(int, r[1:4])).toordinal() for r in observed], dtype=np.int64)
            if not len(odates) or np.any(np.diff(odates) <= 0):
                raise ValueError("Observation dates must be unique and increasing")
            discharge = np.array([float(r[4]) for r in observed])
            original = _dense(odates, discharge)
            present = np.zeros(len(original), dtype=np.bool_)
            present[odates - odates[0]] = True
            flags = [r[5] for r in observed]
            qc = np.full(len(original), "", dtype=f"U{max(map(len, flags))}")
            qc[odates - odates[0]] = flags
            np.save(folder / "discharge_cfs.npy", original)
            np.save(folder / "present.npy", present)
            np.save(folder / "qc.npy", qc)
            records.append(
                Basin(
                    gauge,
                    latitude,
                    area,
                    areas[gauge]["total_km2"],
                    areas[gauge]["contributing_km2"],
                    int(dates[0]),
                    int(dates[-1] + 1),
                    int(odates[0]),
                    int(odates[-1] + 1),
                    forcings[gauge],
                    flows[gauge],
                    len(dense) - len(dates),
                    int((~np.isfinite(dense).all(axis=1)).sum()),
                    int(((discharge < 0) | ~np.isfinite(discharge)).sum()),
                    int((~present).sum()),
                )
            )
    metadata = {
        "format": 1,
        "source": "https://zenodo.org/records/15529996",
        "usgs_source": snapshot["source"],
        "forcing_variables": ["precipitation", "mean_temperature", "oudin_pet"],
        "forcing_units": ["mm/day", "degrees C", "mm/day"],
        "observation_units": "cfs",
        "calendar": "Gregorian; ordinals; stop exclusive",
        "pet": "Oudin2005 Eq3; FAO56 Eqs21–25; attribute gauge_lat; (tmin+tmax)/2; lambda=2.45; solar constant=0.0820; astronomy denominator=365",
        "discharge_conversion": "cfs * 0.028316846592 * 86400 / header_area_m2 * 1000; negative/nonfinite rows become NaN only in converted values",
        "basins": [asdict(b) for b in records],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


class Camels:
    """Metadata-only open; mmap individual basin files and copy only requested days."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.metadata: dict[str, Any] = json.loads((directory / "metadata.json").read_text())
        if self.metadata["format"] != 1:
            raise ValueError("Unsupported CAMELS prepared format")
        self.basins = tuple(Basin(**b) for b in self.metadata["basins"])
        self.ids = tuple(b.gauge_id for b in self.basins)
        self._basins = {b.gauge_id: b for b in self.basins}

    def _basin(self, gauge: str) -> Basin:
        if gauge not in self._basins:
            raise ValueError(f"Unknown CAMELS basin: {gauge}")
        return self._basins[gauge]

    def _load(self, gauge: str, variable: str) -> np.memmap:
        return np.load(self.directory / gauge / f"{variable}.npy", mmap_mode="r", allow_pickle=False)

    def observations(self, gauge: str) -> Observations:
        """Full observation timeline, independent of forcing coverage."""
        b = self._basin(gauge)
        q = self._load(gauge, "discharge_cfs")
        return Observations(
            np.arange(b.observation_start, b.observation_stop, dtype=np.int64),
            q,
            self._convert(q, b),
            self._load(gauge, "qc"),
            self._load(gauge, "present"),
        )

    @staticmethod
    def _convert(q: Array, basin: Basin) -> Array:
        return np.where((q >= 0) & np.isfinite(q), q * 0.028316846592 * 86400 / basin.header_area_m2 * 1000, np.nan)

    def select(self, ids: list[str], start: str, stop: str) -> Selection:
        """Materialize [start, stop) ISO dates in caller basin order, one zone each.

        Reuse the returned Forcing across parameter sets with Parameters.basin_index.
        Duplicate/empty IDs, uncovered dates and invalid forcing are rejected.
        """
        first, last = date.fromisoformat(start).toordinal(), date.fromisoformat(stop).toordinal()
        if not ids or len(set(ids)) != len(ids) or last <= first:
            raise ValueError("Select unique nonempty basins and a nonempty date window")
        basins = [self._basin(g) for g in ids]
        if any(first < b.forcing_start or last > b.forcing_stop for b in basins):
            raise ValueError("Selection exceeds forcing coverage")
        n = last - first
        values = np.empty((len(ids), n, 3))
        q = np.full((n, len(ids)), np.nan)
        present = np.zeros(q.shape, dtype=np.bool_)
        flags = [self._load(g, "qc") for g in ids]
        qc = np.full(q.shape, "", dtype=np.result_type(*(f.dtype for f in flags)))
        converted = np.empty_like(q)
        for i, b in enumerate(basins):
            values[i] = self._load(b.gauge_id, "forcing")[first - b.forcing_start : last - b.forcing_start]
            a, z = max(first, b.observation_start), min(last, b.observation_stop)
            if z > a:
                src = slice(a - b.observation_start, z - b.observation_start)
                dest = slice(a - first, z - first)
                q[dest, i] = self._load(b.gauge_id, "discharge_cfs")[src]
                present[dest, i] = self._load(b.gauge_id, "present")[src]
                qc[dest, i] = flags[i][src]
            converted[:, i] = self._convert(q[:, i], b)
        if not np.isfinite(values).all() or np.any(values[:, :, (0, 2)] < 0):
            raise ValueError("Selection contains invalid/missing forcing; no repair policy is applied")
        return Selection(
            np.arange(first, last, dtype=np.int64),
            q,
            converted,
            qc,
            present,
            tuple(ids),
            Forcing(values.transpose(1, 0, 2), np.ones(len(ids)), np.arange(len(ids) + 1, dtype=np.int64)),
        )
