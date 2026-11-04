"""Sequential, content-verified transfer of frozen native USGS DEM files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import boto3
import requests
from botocore.exceptions import ClientError

from scripts.daymet_archive import BLOCK, Archive, checksum
from scripts.dem_inventory import PREFIX, Source, inventory


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


def receipt(archive: Archive, key: str) -> dict | None:
    try:
        response = archive.client.get_object(Bucket=archive.bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise
    try:
        return json.loads(response["Body"].read())
    finally:
        response["Body"].close()


def run(snapshot: Path, stage: Path, archive: Archive, session: requests.Session) -> dict:
    sources, missing = inventory(snapshot)
    identity = checksum(snapshot)
    provenance = f"{PREFIX}/inventories/{identity}"
    stage.mkdir(parents=True, exist_ok=True)
    archive.put(snapshot, f"{provenance}/inventory.json")
    report: dict = {"inventory_sha256": identity, "missing_cells": missing, "objects": [], "failed": []}
    target = stage / "provider.part"
    record_path = stage / "receipt.json"
    for source in sources:
        key = f"{PREFIX}/native/{source.name}"
        receipt_key = f"{provenance}/receipts/{source.name}.json"
        try:
            saved = receipt(archive, receipt_key)
            if saved is not None:
                if saved["etag"] != source.etag or saved["bytes"] != source.size or saved["key"] != key:
                    raise ValueError("Receipt identity mismatch")
                if archive.digest(key) == (source.size, saved["sha256"]):
                    report["objects"].append(saved)
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
            uploaded = archive.put(target, key)
            saved = dict(
                uploaded,
                etag=source.etag,
                source_url=source.url,
                modified=source.modified,
                provider_checksums=provider_checksums,
                metadata=details,
            )
            record_path.write_text(json.dumps(saved, indent=2) + "\n")
            archive.put(record_path, receipt_key)
            report["objects"].append(saved)
        except (requests.RequestException, ValueError, OSError, ClientError) as error:
            report["failed"].append({"name": source.name, "error": type(error).__name__, "detail": str(error)})
        finally:
            target.unlink(missing_ok=True)
        (stage / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{source.name}: {len(report['objects'])} verified; {len(report['failed'])} failed", flush=True)
    report["complete"] = not missing and not report["failed"] and len(report["objects"]) == len(sources)
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
