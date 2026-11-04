"""Sequential, content-verified transfer of frozen native USGS DEM files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import TypedDict, cast
from xml.etree import ElementTree as ET

import boto3
import requests
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError

from scripts.daymet_archive import BLOCK, Archive, checksum
from scripts.dem_inventory import PREFIX, Source, inventory


class Receipt(TypedDict):
    key: str
    bytes: int
    sha256: str
    verification: str
    etag: str
    source_url: str
    modified: str
    provider_checksums: dict[str, str]
    metadata: dict[str, str]
    release_identity: dict[str, str]


class Failure(TypedDict):
    name: str
    error: str
    detail: str


class Report(TypedDict):
    inventory_sha256: str
    missing_cells: dict[str, list[str]]
    objects: list[Receipt]
    failed: list[Failure]
    unverifiable_releases: dict[str, str]
    complete: bool


def download(session: requests.Session, source: Source, target: Path) -> dict[str, str]:
    md5 = hashlib.md5()
    size = 0
    with session.get(
        source.url, headers={"If-Match": source.etag, "x-amz-checksum-mode": "ENABLED"}, stream=True, timeout=(30, 120)
    ) as response:
        response.raise_for_status()
        if response.headers.get("ETag") != source.etag:
            raise ValueError("Source identity changed")
        integrity = {k: v for k, v in response.headers.items() if k.lower().startswith("x-amz-checksum")}
        with target.open("wb") as stream:
            for block in response.iter_content(BLOCK):
                size += len(block)
                if size > source.size:
                    raise ValueError("Source integrity size overflow")
                md5.update(block)
                stream.write(block)
    if size != source.size or ("-" not in source.etag and md5.hexdigest() != source.etag.strip('"')):
        raise ValueError("Source integrity mismatch")
    return integrity


def metadata(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    fields = {
        name: root.findtext(".//" + name) or ""
        for name in (
            "title",
            "pubdate",
            "begdate",
            "enddate",
            "networkr",
            "horizdn",
            "altdatum",
            "altunits",
            "latres",
            "longres",
        )
    }
    if (
        fields["horizdn"] != "North American Datum of 1983"
        or fields["altdatum"] != "North American Vertical Datum of 1988"
        or fields["altunits"] not in {"meters", "metres"}
        or any(abs(abs(float(fields[name])) - 1 / 3600) > 1e-9 for name in ("latres", "longres"))
    ):
        raise ValueError("Unexpected DEM reference system, units or resolution")
    return fields


def release_identity(session: requests.Session, fields: dict[str, str], source: Source) -> dict[str, str]:
    url = fields["networkr"]
    if not re.fullmatch(
        rf"USGS 1 (?:Arc Second|arc-second) {source.tile} (?:\d{{8}}|1 x 1 degree)", fields.get("title", "")
    ):
        raise ValueError("Metadata release tile/product mismatch")
    expected = (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/historical/"
        f"{source.tile}/USGS_1_{source.tile}_{fields['pubdate']}.tif"
    )
    if url != expected and "/historical/" in url:
        raise ValueError("Metadata release dated link mismatch")
    if url != expected:
        return {"status": "unverifiable", "reason": "No dated official TIFF link matching publication date", "url": url}
    response = session.head(url, timeout=(30, 120))
    response.raise_for_status()
    if response.headers.get("ETag") != source.etag or int(response.headers.get("Content-Length", "0")) != source.size:
        raise ValueError("Metadata release differs from frozen current TIFF")
    return {"status": "matched", "url": url, "etag": source.etag}


def receipt(archive: Archive, key: str) -> Receipt | None:
    try:
        response = archive.client.get_object(Bucket=archive.bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise
    try:
        return cast(Receipt, json.loads(response["Body"].read()))
    finally:
        response["Body"].close()


def run(snapshot: Path, stage: Path, archive: Archive, session: requests.Session) -> Report:
    sources, missing = inventory(snapshot)
    identity = checksum(snapshot)
    provenance = f"{PREFIX}/inventories/{identity}"
    stage.mkdir(parents=True, exist_ok=True)
    archive.put(snapshot, f"{provenance}/inventory.json")
    report: Report = {
        "inventory_sha256": identity,
        "missing_cells": missing,
        "objects": [],
        "failed": [],
        "unverifiable_releases": {},
        "complete": False,
    }
    target = stage / "provider.part"
    record_path = stage / "receipt.json"
    for source in sources:
        key = f"{PREFIX}/native/{source.name}"
        receipt_key = f"{provenance}/release-checked-receipts/{source.name}.json"
        try:
            saved = receipt(archive, receipt_key)
            if saved is not None:
                if saved["etag"] != source.etag or saved["bytes"] != source.size or saved["key"] != key:
                    raise ValueError("Receipt identity mismatch")
                if archive.digest(key) == (source.size, saved["sha256"]):
                    report["objects"].append(saved)
                    if saved.get("release_identity", {}).get("status") == "unverifiable":
                        report["unverifiable_releases"][source.tile] = saved["release_identity"]["reason"]
                    continue
                raise ValueError("Destination missing or corrupt despite receipt")
            target.unlink(missing_ok=True)
            if shutil.disk_usage(stage).free < source.size + BLOCK:
                raise OSError("Insufficient staging disk")
            for attempt in range(3):
                try:
                    provider_checksums = download(session, source, target)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
            details = metadata(target) if source.name.endswith(f"USGS_1_{source.tile}.xml") else {}
            release = {}
            if details:
                raster = next((s for s in sources if s.name == f"{source.tile}/USGS_1_{source.tile}.tif"), None)
                if raster is None:
                    raise ValueError("Metadata has no selected TIFF")
                release = release_identity(session, details, raster)
            uploaded = archive.put(target, key)
            saved = Receipt(
                key=str(uploaded["key"]),
                bytes=int(uploaded["bytes"]),
                sha256=str(uploaded["sha256"]),
                verification=str(uploaded["verification"]),
                etag=source.etag,
                source_url=source.url,
                modified=source.modified,
                provider_checksums=provider_checksums,
                metadata=details,
                release_identity=release,
            )
            record_path.write_text(json.dumps(saved, indent=2) + "\n")
            archive.put(record_path, receipt_key)
            report["objects"].append(saved)
            if release.get("status") == "unverifiable":
                report["unverifiable_releases"][source.tile] = release["reason"]
        except (
            requests.RequestException,
            ValueError,
            OSError,
            ClientError,
            ET.ParseError,
            S3UploadFailedError,
        ) as error:
            report["failed"].append({"name": source.name, "error": type(error).__name__, "detail": str(error)})
        finally:
            target.unlink(missing_ok=True)
        (stage / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{source.name}: {len(report['objects'])} verified; {len(report['failed'])} failed", flush=True)
    report["complete"] = (
        not missing
        and not report["failed"]
        and not report["unverifiable_releases"]
        and len(report["objects"]) == len(sources)
    )
    (stage / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Reports are observations, not immutable receipts; keep each run separately.
    report_path = stage / "report.json"
    archive.put(report_path, f"{provenance}/reports/{checksum(report_path)}.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    client = boto3.client("s3", endpoint_url="https://fsn1.your-objectstorage.com", region_name="fsn1")
    with requests.Session() as session:
        report = run(args.inventory, args.stage, Archive(client), session)
    if not report["complete"]:
        raise SystemExit("Acquisition incomplete: inspect report.json; no completion claimed")


if __name__ == "__main__":
    main()
