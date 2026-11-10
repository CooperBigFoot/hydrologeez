"""Broad CAMELS meteorology, separate from experiment features and normalization."""

import csv
import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .camels import prepare_forcing

PRODUCTS = ("daymet", "maurer", "nldas")
FIELDS = ("dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp", "tmean", "pet")
UNITS = ("s", "mm/day", "W/m2", "mm", "degrees C", "degrees C", "Pa", "degrees C", "mm/day")
HEADER = [
    "year",
    "mnth",
    "day",
    "hr",
    "dayl(s)",
    "prcp(mm/day)",
    "srad(w/m2)",
    "swe(mm)",
    "tmax(c)",
    "tmin(c)",
    "vp(pa)",
]


def prepare_meteorology(raw: Path, output: Path) -> None:
    """Prepare all three products once. Preserve source values, gaps and provenance.

    Per-basin/product float32 arrays omit verified zero SWE and duplicate temperatures; source sentinels are not
    overwritten. The reader supplies separate validity and row-presence masks.
    Existing traditional daily and attribute stores are neither changed nor copied.
    """
    with (raw / "camels_topo.txt").open() as stream:
        rows = sorted(csv.DictReader(stream, delimiter=";"), key=lambda row: row["gauge_id"])
    ids = [row["gauge_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Need unique nonempty attribute basin identities")
    for name in ("name", "clim", "hydro", "soil", "geol", "vege"):
        with (raw / f"camels_{name}.txt").open() as stream:
            current = sorted(row["gauge_id"] for row in csv.DictReader(stream, delimiter=";"))
        if current != ids:
            raise ValueError("Attribute tables must agree on unique gauge identities")
    fields = {
        f"{product}.{field}": {
            "unit": unit,
            "product": product,
            "derivation": "source field"
            if field not in ("tmean", "pet")
            else "(tmin+tmax)/2"
            if field == "tmean"
            else "Oudin2005 Eq3 / FAO56; attribute gauge_lat; mean Tmin/Tmax; lambda=2.45; solar constant=0.0820; denominator=365",
        }
        for product in PRODUCTS
        for field, unit in zip(FIELDS, UNITS, strict=True)
    }
    output.mkdir(parents=True, exist_ok=False)
    records: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(raw / "basin_timeseries_v1p2_metForcing_obsFlow.zip") as archive:
        for product in PRODUCTS:
            members: dict[str, str] = {}
            for name in archive.namelist():
                if f"/basin_mean_forcing/{product}/" in name and name.endswith("_forcing_leap.txt"):
                    gauge = Path(name).name.split("_")[0]
                    if gauge in members:
                        raise ValueError(f"Duplicate {product} source for {gauge}")
                    members[gauge] = name
            for row in rows:
                gauge = row["gauge_id"]
                text = archive.read(members[gauge]).decode()
                lines = text.splitlines()
                if lines[3].lower().split() not in (HEADER, HEADER[4:]):
                    raise ValueError("Unexpected forcing columns/units")
                source = np.loadtxt(io.StringIO(text), skiprows=4, ndmin=2)
                latitude = float(row["gauge_lat"])
                if not np.isfinite(latitude) or abs(latitude) > 90:
                    raise ValueError("Invalid attribute latitude")
                dates, physical = prepare_forcing(source, latitude)
                values = np.column_stack((source[:, 4:], physical[:, 1:])).astype(np.float32)
                constants: dict[str, float] = {"swe": 0.0} if np.all(source[:, 7] == 0) else {}
                aliases: dict[str, str] = {}
                if np.array_equal(source[:, 8], source[:, 9]):
                    aliases["tmin"] = "tmax"
                    if np.array_equal(values[:, 7], values[:, 4], equal_nan=True):
                        aliases["tmean"] = "tmax"
                stored_fields = [field for field in FIELDS if field not in constants and field not in aliases]
                dense = np.full((int(dates[-1] - dates[0] + 1), len(stored_fields)), np.nan, dtype=np.float32)
                dense[dates - dates[0]] = values[:, [FIELDS.index(field) for field in stored_fields]]
                present = np.zeros(len(dense), dtype=np.bool_)
                present[dates - dates[0]] = True
                folder = output / product
                folder.mkdir(exist_ok=True)
                np.save(folder / f"{gauge}.npy", dense)
                np.save(folder / f"{gauge}-present.npy", present)
                records[f"{product}.{gauge}"] = {
                    "stored_fields": stored_fields,
                    "constants": constants,
                    "aliases": aliases,
                    "start": int(dates[0]),
                    "stop": int(dates[-1] + 1),
                    "member": members[gauge],
                    "header": lines[:4],
                    "hours": np.unique(source[:, 3]).tolist(),
                    "missing_dates": int((~present).sum()),
                }
    metadata = {
        "format": 1,
        "ids": ids,
        "fields": fields,
        "records": records,
        "source": "basin_timeseries_v1p2_metForcing_obsFlow.zip",
        "dtype": "float32",
        "interpretation": {
            "time": "Source Hr retained in each record; no subdaily or timezone inference from daily labels",
            "temperature": "Source Tmax/Tmin names retained; exact duplicates are aliases, not independently observed extrema",
            "swe": "Verified all-zero source SWE represented as a constant, not interpreted as observed absence of snow",
        },
        "calendar": "Gregorian ordinals; stop exclusive",
        "missing": "nonfinite or <= -999; negative non-temperature fields invalid",
    }
    (output / "meteorology.json").write_text(json.dumps(metadata, indent=2) + "\n")


@dataclass(frozen=True)
class Meteorology:
    """Resident caller-ordered data. Values retain sentinels; valid marks usable values."""

    ids: tuple[str, ...]
    dates: NDArray[np.int64]
    features: tuple[str, ...]
    values: NDArray[np.float32]  # [basin, day, feature]
    valid: NDArray[np.bool_]
    present: NDArray[np.bool_]  # source row, per selected feature/product


class CamelsMeteorology:
    """Metadata-only open followed by one resident selection, reused for batches."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.metadata: dict[str, Any] = json.loads((directory / "meteorology.json").read_text())
        if self.metadata["format"] != 1:
            raise ValueError("Unsupported meteorology format")
        self.ids: tuple[str, ...] = tuple(self.metadata["ids"])

    def select(self, ids: Sequence[str], start: str, stop: str, features: Sequence[str]) -> Meteorology:
        """Load only named features and dates; never fit scaling or read discharge.

        Dates outside a product's record become missing. The window must stay
        inside the combined selected-product coverage; no intersection truncation.
        """
        first, last = date.fromisoformat(start).toordinal(), date.fromisoformat(stop).toordinal()
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(self.ids) or last <= first:
            raise ValueError("Select known unique basins and a nonempty date window")
        if not features or len(set(features)) != len(features) or not set(features) <= self.metadata["fields"].keys():
            raise ValueError("Select known unique meteorological fields")
        records = [
            self.metadata["records"][f"{feature.split('.')[0]}.{gauge}"] for gauge in ids for feature in features
        ]
        if first < min(r["start"] for r in records) or last > max(r["stop"] for r in records):
            raise ValueError("Selection exceeds combined product coverage")
        values = np.full((len(ids), last - first, len(features)), np.nan, dtype=np.float32)
        present = np.zeros(values.shape, dtype=np.bool_)
        for product in PRODUCTS:
            columns = [
                (j, FIELDS.index(feature.split(".")[1]))
                for j, feature in enumerate(features)
                if feature.startswith(product + ".")
            ]
            if not columns:
                continue
            for i, gauge in enumerate(ids):
                record = self.metadata["records"][f"{product}.{gauge}"]
                a, z = max(first, record["start"]), min(last, record["stop"])
                if z <= a:
                    continue
                source = np.load(self.directory / product / f"{gauge}.npy", mmap_mode="r", allow_pickle=False)
                rows = np.load(self.directory / product / f"{gauge}-present.npy", mmap_mode="r", allow_pickle=False)
                src, dest = slice(a - record["start"], z - record["start"]), slice(a - first, z - first)
                for j, k in columns:
                    field = FIELDS[k]
                    if field in record["constants"]:
                        values[i, dest, j] = np.where(rows[src], record["constants"][field], np.nan)
                    else:
                        stored = record["aliases"].get(field, field)
                        values[i, dest, j] = source[src, record["stored_fields"].index(stored)]
                    present[i, dest, j] = rows[src]
        valid = np.isfinite(values) & (values > -999) & present
        for j, feature in enumerate(features):
            if feature.split(".")[1] not in ("tmax", "tmin", "tmean"):
                valid[:, :, j] &= values[:, :, j] >= 0
        return Meteorology(tuple(ids), np.arange(first, last, dtype=np.int64), tuple(features), values, valid, present)
