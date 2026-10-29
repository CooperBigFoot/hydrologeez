"""Bounded, restartable acquisition and native-grid preparation of Daymet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import boto3
import numpy as np
import requests
from dotenv import load_dotenv

from scripts.daymet_acquire import EarthdataSession, download, inventory
from scripts.daymet_archive import Archive, checksum
from scripts.daymet_mask import BOUNDARY_URL, STATEFP, conus_mask
from scripts.daymet_prepare import prepare_month


def verified_receipt(archive: Archive, key: str, expected: dict[str, tuple[int, str] | None]) -> dict | None:
    from botocore.exceptions import ClientError

    try:
        response = archive.client.get_object(Bucket=archive.bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise
    try:
        receipt = json.loads(response["Body"].read())
    finally:
        response["Body"].close()
    objects = receipt.get("objects", [])
    if len(objects) != len(expected) or {item["key"] for item in objects} != expected.keys():
        raise ValueError("Receipt coverage does not match expected delivery")
    for item in objects:
        declared = (item["bytes"], item["sha256"])
        provider = expected[item["key"]]
        if provider is not None and declared != provider:
            raise ValueError("Receipt integrity does not match provider inventory")
        if archive.digest(item["key"]) != declared:
            raise ValueError(f"Receipt integrity mismatch: {item['key']}")
    return receipt


def save_receipt(archive: Archive, stage: Path, key: str, record: dict) -> None:
    path = stage / "receipt.json"
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    archive.put(path, key)


def run(
    snapshot: Path, metadata: list[Path], boundary: Path, stage: Path, archive: Archive, session: requests.Session
) -> None:
    sources = inventory(snapshot)
    stage.mkdir(parents=True, exist_ok=True)
    # A processing namespace binds inputs, boundary and exact scripts, not delivery identity.
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("daymet*.py")):
        code.update(path.name.encode())
        code.update(path.read_bytes())
    repository = Path(__file__).resolve().parent.parent
    dependencies = [repository / "uv.lock", repository / "pyproject.toml"]
    provenance = {
        "inventory_sha256": checksum(snapshot),
        "boundary_sha256": checksum(boundary),
        "code_sha256": code.hexdigest(),
        "dependencies": {path.name: checksum(path) for path in dependencies},
        "boundary_url": BOUNDARY_URL,
        "states": STATEFP,
        "mask": "projected native cell-center covers; Census2024 500k; source missing retained",
        "astronomy": "FAO56 Eqs21–25 supplementary to Oudin2005 Eq3; not exact Morton reproduction",
    }
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    prefix = f"prepared/{identity}"
    archive.put(snapshot, f"{prefix}/provenance/inventory.json")
    archive.put(boundary, f"{prefix}/provenance/{boundary.name}")
    for path in metadata:
        archive.put(path, f"{prefix}/provenance/provider/{path.name}")
    for path in dependencies:
        archive.put(path, f"{prefix}/provenance/{path.name}")
    for path in sorted(Path(__file__).parent.glob("daymet*.py")):
        archive.put(path, f"{prefix}/provenance/scripts/{path.name}")
    save_receipt(archive, stage, f"{prefix}/provenance/processing.json", provenance)
    print(f"Archive prefix: s3://{archive.bucket}/{prefix}", flush=True)
    all_objects = []
    for year in sorted({source.year for source in sources}):
        annual_key = f"{prefix}/{year}/complete.json"
        annual = [source for source in sources if source.year == year]
        expected: dict[str, tuple[int, str] | None] = {
            f"raw/Daymet_Daily_V4R1/{source.name}": (source.size, source.sha256) for source in annual
        }
        expected[f"{prefix}/{year}/conus-mask.npy"] = None
        expected.update({f"{prefix}/{year}/{month:02d}.nc": None for month in range(1, 13)})
        complete = verified_receipt(archive, annual_key, expected)
        if complete is not None:
            all_objects.extend(complete["objects"])
            for source in sources:
                if source.year == year:
                    (stage / source.name).unlink(missing_ok=True)
            for month in range(1, 13):
                (stage / f"daymet_{year}_{month:02d}.nc").unlink(missing_ok=True)
            print(f"{year}: verified existing annual delivery", flush=True)
            continue
        if (
            shutil.disk_usage(stage).free
            < sum(source.size for source in annual if not (stage / source.name).exists()) + 15_000_000_000
        ):
            raise ValueError("Insufficient stage disk for annual sources plus monthly prepared output")
        paths = {}
        objects = []
        for source in annual:
            path = stage / source.name
            download(session, source, path)
            objects.append(archive.put(path, f"raw/Daymet_Daily_V4R1/{source.name}"))
            paths[source.variable] = path
        mask = conus_mask(paths["prcp"], boundary)
        mask_path = stage / "conus-mask.npy"
        np.save(mask_path, mask, allow_pickle=False)
        objects.append(archive.put(mask_path, f"{prefix}/{year}/conus-mask.npy"))
        for month in range(1, 13):
            key = f"{prefix}/{year}/{month:02d}"
            receipt = verified_receipt(archive, f"{key}.json", {f"{key}.nc": None})
            if receipt is None:
                output = stage / f"daymet_{year}_{month:02d}.nc"
                prepare_month(paths, year, month, mask, output)
                item = archive.put(output, f"{key}.nc")
                receipt = {"objects": [item]}
                save_receipt(archive, stage, f"{key}.json", receipt)
            (stage / f"daymet_{year}_{month:02d}.nc").unlink(missing_ok=True)
            objects.extend(receipt["objects"])
            print(f"{year}-{month:02d}: content verified", flush=True)
        save_receipt(archive, stage, annual_key, {"year": year, "native_days": 365, "objects": objects})
        all_objects.extend(objects)
        for path in paths.values():
            path.unlink()
        mask_path.unlink()
    save_receipt(
        archive, stage, f"{prefix}/complete.json", {"objects": all_objects, "years": sorted({s.year for s in sources})}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--provider-metadata", type=Path, nargs="+", required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv()
    session = EarthdataSession()
    session.auth = (os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"])
    client = boto3.client("s3", endpoint_url="https://fsn1.your-objectstorage.com", region_name="fsn1")
    run(args.inventory, args.provider_metadata, args.boundary, args.stage, Archive(client), session)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Request exceptions can contain signed redirect URLs. Keep secrets out of logs.
        print(f"Daymet stopped: {type(error).__name__}; no completion claimed. Check inputs and restart.", flush=True)
        raise SystemExit(1) from None
