"""Acquire bounded source-date scopes and publish verified USGS cache snapshots."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import boto3
import rivretrieve as rr

from scripts.usgs_acquire import (
    Scope,
    acquire_scope,
    atomic_write,
    cached_outcome_ids,
    completed_scope,
    plan_scopes,
    validate_workspace,
)
from scripts.usgs_archive import USGSArchive, workspace_lock
from scripts.usgs_inventory import build_inventory
from scripts.usgs_parts import acquire_part, plan_parts, verify_part, verify_parts
from scripts.usgs_quota import reconcile, reserve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire-scope", help="Bounded pilot; not a complete CONUS inventory")
    acquire.add_argument("workspace", type=Path)
    acquire.add_argument("--station", required=True)
    acquire.add_argument("--start", required=True)
    acquire.add_argument("--end", required=True)
    inventory = sub.add_parser("inventory", help="Discover CONUS series and resolve bounded history probes")
    inventory.add_argument("workspace", type=Path)
    inventory.add_argument("boundary", type=Path)
    inventory.add_argument("--max-history-probes", type=int, default=0)
    inventory.add_argument("--refine-locations", action="store_true")
    plan = sub.add_parser("plan", help="Freeze scopes only after geography/history is resolved")
    plan.add_argument("workspace", type=Path)
    plan.add_argument("--max-days", type=int, required=True)
    run = sub.add_parser("acquire-plan", help="Resume at most this many station/date scopes")
    run.add_argument("workspace", type=Path)
    run.add_argument("--max-scopes", type=int, required=True)
    for name in ("verify", "recover"):
        operation = sub.add_parser(name)
        operation.add_argument("workspace", type=Path)
    for name in ("publish", "restore-files", "verify-remote"):
        operation = sub.add_parser(name)
        operation.add_argument("workspace", type=Path)
        operation.add_argument("generation")
        operation.add_argument("--profile")
        if name == "verify-remote":
            operation.add_argument(
                "--complete", action="store_true", help="Certify fully acquired multi-root delivery after readback"
            )
    for name in ("plan-parts", "acquire-part", "verify-part", "verify-parts"):
        operation = sub.add_parser(name)
        operation.add_argument("workspace", type=Path)
        if name in {"acquire-part", "verify-part"}:
            operation.add_argument("part")
    quota = sub.add_parser(
        "reserve-source-usage", help="Charge recent external/uncertain source requests before part workers"
    )
    quota.add_argument("workspace", type=Path)
    quota.add_argument("label")
    quota.add_argument("attempts", type=int)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    started = monotonic()
    if args.command == "reserve-source-usage":
        reservation = reserve(workspace, args.label, args.attempts)
        reconcile(workspace, reservation, args.attempts)
        result = {"reservation": reservation}
    elif args.command in {"acquire-part", "verify-part"}:
        with workspace_lock(workspace, shared=True):
            result = (
                acquire_part(workspace, args.part)
                if args.command == "acquire-part"
                else verify_part(workspace, args.part)
            )
    elif args.command in {"publish", "restore-files", "verify-remote"}:
        session = boto3.Session(profile_name=args.profile)
        client = session.client("s3", endpoint_url="https://fsn1.your-objectstorage.com", region_name="fsn1")
        archive = USGSArchive(client)

        def validate_delivery(path: Path) -> dict[str, object]:
            return verify_parts(path) if (path / "manifests/parts-plan.json").exists() else validate_workspace(path)

        if args.command == "publish":
            result = archive.publish(workspace, args.generation, validate=validate_delivery)
        elif args.command == "restore-files":
            result = archive.restore(args.generation, workspace, validate=validate_delivery)
            result["native_local_cache_usable"] = True
        else:
            credentials = session.get_credentials()
            if credentials is None:
                raise ValueError("S3 credentials unavailable")
            frozen = credentials.get_frozen_credentials()
            if frozen.access_key is None or frozen.secret_key is None:
                raise ValueError("S3 credentials incomplete")
            os.environ["RIVRETRIEVE_S3_ACCESS_KEY_ID"] = frozen.access_key
            os.environ["RIVRETRIEVE_S3_SECRET_ACCESS_KEY"] = frozen.secret_key
            if frozen.token:
                os.environ["RIVRETRIEVE_S3_SESSION_TOKEN"] = frozen.token
            else:
                os.environ.pop("RIVRETRIEVE_S3_SESSION_TOKEN", None)
                os.environ.pop("AWS_SESSION_TOKEN", None)
            os.environ["RIVRETRIEVE_S3_ENDPOINT_URL"] = "https://fsn1.your-objectstorage.com"
            os.environ["RIVRETRIEVE_S3_REGION"] = "fsn1"
            cache_root = f"s3://{archive.bucket}/{archive.prefix}/cache/{args.generation}"
            with workspace_lock(workspace):
                result = (
                    verify_parts(workspace, cache_root=cache_root)
                    if (workspace / "manifests/parts-plan.json").exists()
                    else validate_workspace(workspace, cache_root=cache_root)
                )
                if args.complete:
                    result = archive.complete_delivery(workspace, args.generation, result)
    else:
        with workspace_lock(workspace):
            os.environ["RIVRETRIEVE_CACHE_DIR"] = str(workspace / "cache")
            if args.command == "plan-parts":
                result = plan_parts(workspace)
            elif args.command == "verify-parts":
                result = verify_parts(workspace)
            elif args.command == "acquire-scope":
                if (workspace / "manifests/parts-plan.json").exists():
                    raise ValueError("Original cache frozen by parts plan")
                result = acquire_scope(workspace, Scope(args.station, args.start, args.end))
            elif args.command == "inventory":
                inventory_result = build_inventory(
                    args.boundary,
                    workspace,
                    max_history_probes=args.max_history_probes,
                    refine_locations=args.refine_locations,
                )
                result = {
                    "series": len(inventory_result["series"]),
                    "stations": len(inventory_result["stations"]),
                    "unresolved": len(inventory_result["unresolved"]),
                    "history_failures": sum(row["history_status"] == "failed" for row in inventory_result["series"]),
                    "location_refinement_status": inventory_result.get("location_refinement", {}).get("status"),
                }
            elif args.command == "plan":
                inventory_path = workspace / "manifests/inventory.json"
                scopes = plan_scopes(json.loads(inventory_path.read_text()), args.max_days)
                from scripts.daymet_archive import checksum

                plan_record = {
                    "inventory_sha256": checksum(inventory_path),
                    "max_days": args.max_days,
                    "scopes": [asdict(scope) for scope in scopes],
                }
                path = workspace / "manifests/plan.json"
                if path.exists() and json.loads(path.read_text()) != plan_record:
                    raise ValueError("Existing acquisition plan differs; use a fresh workspace")
                atomic_write(path, json.dumps(plan_record, indent=2).encode())
                result = {"planned_scopes": len(scopes)}
            elif args.command == "acquire-plan":
                if (workspace / "manifests/parts-plan.json").exists():
                    raise ValueError("Original cache frozen by parts plan")
                if args.max_scopes < 1:
                    raise ValueError("max-scopes must be positive")
                plan_record = json.loads((workspace / "manifests/plan.json").read_text())
                from scripts.daymet_archive import checksum

                if checksum(workspace / "manifests/inventory.json") != plan_record["inventory_sha256"]:
                    raise ValueError("Inventory changed after planning")
                acquired = 0
                initial_outcomes: set[str] | None = cached_outcome_ids(workspace)
                for item in plan_record["scopes"]:
                    scope = Scope(**item)
                    if completed_scope(workspace, scope, cached_outcomes=initial_outcomes) is not None:
                        continue
                    initial_outcomes = None
                    acquire_scope(workspace, scope)
                    acquired += 1
                    if acquired >= args.max_scopes:
                        break
                result = {"acquired_scopes": acquired}
            elif args.command == "verify":
                result = validate_workspace(workspace)
            else:
                result = {"recovery": str(rr.recover_cache("usgs_nwis"))}
    print(json.dumps({"elapsed_seconds": monotonic() - started, "result": result}, default=str, indent=2))


if __name__ == "__main__":
    main()
