"""Bounded, restartable acquisition and native-grid preparation of Daymet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import boto3
import numpy as np
import requests
from dotenv import load_dotenv

from scripts.daymet_acquire import EarthdataSession, download_annual, inventory
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


# Audited acquisition-only migration from the deployed three-download workflow.
RESUMABLE_CODE_SHA256 = "49bda9f2e68637cb4fd37646149ecfcca47d53868ffde1886dae920a5993d2a2"


def resume_processing(archive: Archive, identity: str, provenance: dict, repository: Path) -> dict:
    """Accept only the reviewed predecessor, never an arbitrary namespace override."""
    if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Invalid processing provenance identity")
    prefix = f"prepared/{identity}/provenance"

    def read(key: str) -> bytes:
        response = archive.client.get_object(Bucket=archive.bucket, Key=key)
        try:
            return response["Body"].read()
        finally:
            response["Body"].close()

    original = json.loads(read(f"{prefix}/processing.json"))
    if hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest() != identity:
        raise ValueError("Processing provenance identity mismatch")
    if original.keys() != provenance.keys() or original.get("code_sha256") != RESUMABLE_CODE_SHA256:
        raise ValueError("Unrecognized processing provenance schema/revision")
    for key in provenance.keys() - {"code_sha256"}:
        if original[key] != provenance[key]:
            raise ValueError(f"Processing provenance mismatch: {key}")
    digest = hashlib.sha256()
    for name in (
        "daymet.py",
        "daymet_acquire.py",
        "daymet_archive.py",
        "daymet_inventory.py",
        "daymet_mask.py",
        "daymet_prepare.py",
    ):
        content = read(f"{prefix}/scripts/{name}")
        digest.update(name.encode())
        digest.update(content)
        if name in {"daymet_mask.py", "daymet_prepare.py"} and content != (repository / "scripts" / name).read_bytes():
            raise ValueError("Scientific processing provenance mismatch")
    if digest.hexdigest() != RESUMABLE_CODE_SHA256:
        raise ValueError("Archived script provenance mismatch")
    for name, expected in original["dependencies"].items():
        if hashlib.sha256(read(f"{prefix}/{name}")).hexdigest() != expected:
            raise ValueError("Archived dependency provenance mismatch")
    return original


def run(
    snapshot: Path,
    metadata: list[Path],
    boundary: Path,
    stage: Path,
    archive: Archive,
    session_factory: Callable[[], requests.Session],
    resume: str | None = None,
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
    if resume is not None:
        resume_processing(archive, resume, provenance, repository)
        revision = identity
        identity = resume
        provenance_prefix = f"prepared/{identity}/provenance/revisions/{revision}"
    else:
        provenance_prefix = f"prepared/{identity}/provenance"
    prefix = f"prepared/{identity}"
    archive.put(snapshot, f"{provenance_prefix}/inventory.json")
    archive.put(boundary, f"{provenance_prefix}/{boundary.name}")
    for path in metadata:
        archive.put(path, f"{provenance_prefix}/provider/{path.name}")
    for path in dependencies:
        archive.put(path, f"{provenance_prefix}/{path.name}")
    for path in sorted(Path(__file__).parent.glob("daymet*.py")):
        archive.put(path, f"{provenance_prefix}/scripts/{path.name}")
    save_receipt(archive, stage, f"{provenance_prefix}/processing.json", provenance)
    if resume is not None:
        save_receipt(
            archive,
            stage,
            f"{provenance_prefix}/resume.json",
            {"original_identity": resume, "revision_identity": revision},
        )
    print(f"Archive prefix: s3://{archive.bucket}/{prefix}", flush=True)
    all_objects = []
    pending: Future[None] | None = None
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        try:
            years = sorted({source.year for source in sources})
            for position, year in enumerate(years):
                if pending is not None:
                    pending.result()
                    pending = None
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
                download_annual(annual, stage, session_factory)
                if position + 1 < len(years):
                    following = [source for source in sources if source.year == years[position + 1]]
                    needed = sum(source.size for source in following if not (stage / source.name).exists())
                    if shutil.disk_usage(stage).free >= needed + 15_000_000_000:
                        pending = prefetch.submit(download_annual, following, stage, session_factory)
                paths = {}
                objects = []
                for source in annual:
                    path = stage / source.name
                    objects.append(archive.put(path, f"raw/Daymet_Daily_V4R1/{source.name}"))
                    paths[source.variable] = path
                mask = conus_mask(paths["prcp"], boundary)
                mask_path = stage / "conus-mask.npy"
                np.save(mask_path, mask, allow_pickle=False)
                objects.append(archive.put(mask_path, f"{prefix}/{year}/conus-mask.npy"))
                for month in range(1, 13):
                    if pending is not None and pending.done():
                        pending.result()
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
                if pending is not None and pending.done():
                    pending.result()
                save_receipt(archive, stage, annual_key, {"year": year, "native_days": 365, "objects": objects})
                all_objects.extend(objects)
                for path in paths.values():
                    path.unlink()
                mask_path.unlink()
        finally:
            if pending is not None:
                pending.cancel()
    save_receipt(
        archive, stage, f"{prefix}/complete.json", {"objects": all_objects, "years": sorted({s.year for s in sources})}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--provider-metadata", type=Path, nargs="+", required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--resume-processing", help="Verified existing processing identity to preserve")
    args = parser.parse_args()
    load_dotenv()
    credentials = (os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"])

    def session_factory() -> requests.Session:
        session = EarthdataSession()
        session.auth = credentials
        return session

    client = boto3.client("s3", endpoint_url="https://fsn1.your-objectstorage.com", region_name="fsn1")
    run(
        args.inventory,
        args.provider_metadata,
        args.boundary,
        args.stage,
        Archive(client),
        session_factory,
        args.resume_processing,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Request exceptions can contain signed redirect URLs. Keep secrets out of logs.
        print(f"Daymet stopped: {type(error).__name__}; no completion claimed. Check inputs and restart.", flush=True)
        raise SystemExit(1) from None
