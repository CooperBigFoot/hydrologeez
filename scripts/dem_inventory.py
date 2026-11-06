"""Freeze current USGS 3DEP 1 arc-second files intersecting CONUS."""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pyproj
import requests
import shapefile
import shapely
from shapely.geometry import box, shape
from shapely.ops import transform

from scripts.daymet_archive import checksum
from scripts.daymet_mask import BOUNDARY_URL, STATEFP

BASE = "https://prd-tnm.s3.amazonaws.com/"
CURRENT = "StagedProducts/Elevation/1/TIFF/current/"
PREFIX = "dem/3dep-1arcsec"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass(frozen=True)
class Source:
    tile: str
    name: str
    size: int
    etag: str
    modified: str

    @property
    def url(self) -> str:
        return BASE + CURRENT + self.name


def conus(boundary: Path) -> shapely.Geometry:
    with zipfile.ZipFile(boundary) as archive:
        projection = archive.read("cb_2024_us_state_500k.prj").decode()
    geometries: dict[str, shapely.Geometry] = {}
    reader = shapefile.Reader(str(boundary))
    for item in reader.iterShapeRecords():
        assert item.record is not None and item.shape is not None
        state = str(item.record["STATEFP"])
        if state in STATEFP:
            if state in geometries:
                raise ValueError("Duplicate state in boundary")
            geometries[state] = shape(item.shape.__geo_interface__)
    if sorted(geometries) != STATEFP:
        raise ValueError("Boundary must contain all 48 states and DC")
    project = pyproj.Transformer.from_crs(projection, "EPSG:4269", always_xy=True)
    return transform(project.transform, shapely.union_all(list(geometries.values())))


def select(sources: list[Source], boundary: shapely.Geometry) -> tuple[list[Source], dict[str, list[str]]]:
    west, south, east, north = boundary.bounds
    cells = set()
    for x in range(math.floor(west), math.ceil(east)):
        for y in range(math.floor(south), math.ceil(north)):
            if boundary.intersection(box(x, y, x + 1, y + 1)).area > 0:
                cells.add(f"n{y + 1:02d}w{-x:03d}")
    selected = [
        source
        for source in sources
        if source.tile in cells and Path(source.name).name.startswith(f"USGS_1_{source.tile}.")
    ]
    missing = {}
    for tile in sorted(cells):
        extensions = {Path(s.name).name.removeprefix(f"USGS_1_{tile}.") for s in selected if s.tile == tile}
        absent = sorted({"tif", "xml"} - extensions)
        if absent:
            missing[tile] = absent
    return selected, missing


def discover(session: requests.Session, boundary: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    sources: list[Source] = []
    token = ""
    page = 0
    while True:
        params = {"list-type": "2", "prefix": CURRENT}
        if token:
            params["continuation-token"] = token
        response = session.get(BASE, params=params, timeout=(30, 120))
        response.raise_for_status()
        page += 1
        (directory / f"listing-{page:04d}.xml").write_bytes(response.content)
        listing = ET.fromstring(response.content)
        for item in listing.findall("s:Contents", NS):
            key = item.findtext("s:Key", namespaces=NS) or ""
            relative = key.removeprefix(CURRENT)
            tile = relative.split("/")[0]
            if not re.fullmatch(r"n\d{2}w\d{3}", tile) or relative.endswith("/") or relative.endswith("/Thumbs.db"):
                continue
            sources.append(
                Source(
                    tile,
                    relative,
                    int(item.findtext("s:Size", namespaces=NS) or "0"),
                    item.findtext("s:ETag", namespaces=NS) or "",
                    item.findtext("s:LastModified", namespaces=NS) or "",
                )
            )
        if listing.findtext("s:IsTruncated", namespaces=NS) == "false":
            break
        token = listing.findtext("s:NextContinuationToken", namespaces=NS) or ""
        if not token:
            raise ValueError("Truncated listing without continuation token")
    selected, missing = select(sources, conus(boundary))
    if not selected:
        raise ValueError("Empty CONUS selection")
    record = {
        "snapshot_utc": datetime.now(UTC).isoformat(),
        "source_prefix": BASE + CURRENT,
        "boundary_url": BOUNDARY_URL,
        "boundary_sha256": checksum(boundary),
        "states": STATEFP,
        "selection": "positive-area nominal one-degree cell intersection; native margins preserved",
        "coverage_limitations": "Census 500k generalized coastline; absent cells require investigation, not presumed land gaps",
        "missing": missing,
        "files": [asdict(source) for source in selected],
    }
    (directory / "inventory.json").write_text(json.dumps(record, indent=2) + "\n")
    (directory / boundary.name).write_bytes(boundary.read_bytes())
    print(f"{len(selected)} objects, {sum(s.size for s in selected)} bytes; {len(missing)} cells need review")


def inventory(path: Path) -> tuple[list[Source], dict[str, list[str]]]:
    record = json.loads(path.read_text())
    if record["source_prefix"] != BASE + CURRENT:
        raise ValueError("Expected current USGS 1 arc-second inventory")
    sources = [Source(**row) for row in record["files"]]
    if not sources or len({s.name for s in sources}) != len(sources):
        raise ValueError("Empty or duplicate inventory")
    for source in sources:
        if (
            not re.fullmatch(r"n\d{2}w\d{3}/[^/]+", source.name)
            or not Path(source.name).name.startswith(f"USGS_1_{source.tile}.")
            or source.name.split("/")[0] != source.tile
            or source.size <= 0
            or not re.fullmatch(r'"[0-9a-f]{32}(?:-\d+)?"', source.etag)
        ):
            raise ValueError("Invalid source identity")
    return sources, record["missing"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with requests.Session() as session:
        discover(session, args.boundary, args.directory)
