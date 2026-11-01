"""Acquire bounded source-date scopes and publish verified USGS cache snapshots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import monotonic

import boto3
import rivretrieve as rr

from scripts.usgs_acquire import Scope, acquire_scope, validate_workspace
from scripts.usgs_archive import USGSArchive, workspace_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire-scope", help="Bounded pilot; not a complete CONUS inventory")
    acquire.add_argument("workspace", type=Path)
    acquire.add_argument("--station", required=True)
    acquire.add_argument("--start", required=True)
    acquire.add_argument("--end", required=True)
    for name in ("verify", "recover"):
        operation = sub.add_parser(name)
        operation.add_argument("workspace", type=Path)
    for name in ("publish", "restore"):
        operation = sub.add_parser(name)
        operation.add_argument("workspace", type=Path)
        operation.add_argument("generation")
        operation.add_argument("--profile")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    started = monotonic()
    if args.command in {"publish", "restore"}:
        session = boto3.Session(profile_name=args.profile)
        client = session.client("s3", endpoint_url="https://fsn1.your-objectstorage.com", region_name="fsn1")
        archive = USGSArchive(client)
        if args.command == "publish":
            result = archive.publish(workspace, args.generation, validate=validate_workspace)
        else:
            result = archive.restore(args.generation, workspace, validate=validate_workspace)
    else:
        with workspace_lock(workspace):
            os.environ["RIVRETRIEVE_CACHE_DIR"] = str(workspace / "cache")
            if args.command == "acquire-scope":
                result = acquire_scope(workspace, Scope(args.station, args.start, args.end))
            elif args.command == "verify":
                result = validate_workspace(workspace)
            else:
                result = {"recovery": str(rr.recover_cache("usgs_nwis"))}
    print(json.dumps({"elapsed_seconds": monotonic() - started, "result": result}, default=str, indent=2))


if __name__ == "__main__":
    main()
