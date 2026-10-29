"""Freeze the complete current CMR Daymet V4 R1 three-variable inventory."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import requests

from scripts.daymet_acquire import inventory


def discover(session: requests.Session, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    page = 1
    while True:
        response = session.get(
            "https://cmr.earthdata.nasa.gov/search/granules.umm_json",
            params={"collection_concept_id": "C2532426483-ORNL_CLOUD", "page_size": 2000, "page_num": page},
            timeout=90,
        )
        response.raise_for_status()
        path = directory / f"cmr-page-{page:04d}.json"
        if path.exists():
            raise ValueError("Discovery directory already contains a snapshot")
        path.write_bytes(response.content)
        entries = response.json()["items"]
        for entry in entries:
            data = entry["umm"]
            match = re.fullmatch(
                r"Daymet_Daily_V4R1.daymet_v4_daily_na_(prcp|tmin|tmax)_(\d{4})\.nc", data["GranuleUR"]
            )
            if match is None:
                continue
            variable, year = match.groups()
            info = data["DataGranule"]["ArchiveAndDistributionInformation"][0]
            if info["Checksum"]["Algorithm"].replace("-", "").lower() != "sha256":
                raise ValueError("Expected provider SHA256")
            urls = [
                u["URL"]
                for u in data["RelatedUrls"]
                if u["Type"] == "GET DATA" and u["URL"].endswith(f"daymet_v4_daily_na_{variable}_{year}.nc")
            ]
            if len(urls) != 1:
                raise ValueError("Expected one original source URL")
            files.append(
                {
                    "name": data["GranuleUR"],
                    "variable": variable,
                    "year": int(year),
                    "bytes": int(info["SizeInBytes"]),
                    "sha256": info["Checksum"]["Value"],
                    "url": urls[0],
                    "coverage": data["TemporalExtent"]["RangeDateTime"],
                    "cmr_revision": entry["meta"],
                }
            )
        if len(entries) < 2000:
            break
        page += 1
    snapshot = directory / "inventory.json"
    snapshot.write_text(
        json.dumps(
            {
                "snapshot_utc": datetime.now(UTC).isoformat(),
                "collection": "C2532426483-ORNL_CLOUD",
                "version": "4.1",
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )
    sources = inventory(snapshot)
    print(
        f"{len(sources)} files; {min(s.year for s in sources)}–{max(s.year for s in sources)}; {sum(s.size for s in sources)} bytes"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    discover(requests.Session(), parser.parse_args().directory)
