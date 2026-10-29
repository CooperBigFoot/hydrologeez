"""Acquire unchanged Daymet provider files against a frozen CMR inventory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from scripts.daymet_archive import BLOCK, checksum

VARIABLES = ("prcp", "tmin", "tmax")


class EarthdataSession(requests.Session):
    """Credentials only on NASA source and login hosts, never CDN redirects."""

    def rebuild_auth(self, prepared_request: requests.PreparedRequest, response: requests.Response) -> None:
        target = urlparse(prepared_request.url or "")
        if (
            target.scheme != "https"
            or target.port not in (None, 443)
            or target.hostname not in {"urs.earthdata.nasa.gov", "data.ornldaac.earthdata.nasa.gov"}
        ):
            prepared_request.headers.pop("Authorization", None)
        elif self.auth is not None:
            prepared_request.prepare_auth(self.auth)


@dataclass(frozen=True)
class Source:
    variable: str
    year: int
    url: str
    size: int
    sha256: str

    @property
    def name(self) -> str:
        return Path(urlparse(self.url).path).name


def inventory(path: Path) -> list[Source]:
    record = json.loads(path.read_text())
    if record["collection"] != "C2532426483-ORNL_CLOUD" or record["version"] != "4.1":
        raise ValueError("Expected Daymet V4 R1 collection")
    sources = [Source(row["variable"], row["year"], row["url"], row["bytes"], row["sha256"]) for row in record["files"]]
    years = sorted({source.year for source in sources})
    if not years or years != list(range(1980, years[-1] + 1)):
        raise ValueError("Inventory must contain continuous full history from 1980")
    for year in years:
        if sorted(source.variable for source in sources if source.year == year) != sorted(VARIABLES):
            raise ValueError("Expected exactly one annual source per variable")
    for source in sources:
        expected = f"https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data/daymet_v4_daily_na_{source.variable}_{source.year}.nc"
        if source.url != expected or source.size <= 0 or len(source.sha256) != 64:
            raise ValueError("Invalid provider identity/integrity metadata")
    return sources


def download(session: requests.Session, source: Source, target: Path) -> None:
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        with session.get(source.url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for block in response.iter_content(BLOCK):
                    stream.write(block)
        if partial.stat().st_size != source.size or checksum(partial) != source.sha256:
            raise ValueError(f"Provider integrity mismatch: {source.name}")
        partial.replace(target)
    if target.stat().st_size != source.size or checksum(target) != source.sha256:
        raise ValueError(f"Local integrity mismatch: {source.name}")
