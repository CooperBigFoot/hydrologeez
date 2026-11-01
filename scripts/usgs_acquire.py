"""Bounded USGS acquisition; a cache write alone never completes a scope."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import rivretrieve as rr
from polars.testing import assert_frame_equal
from rivretrieve._internal.observations import ObservationResult

from scripts.daymet_archive import checksum

RR_REVISION = "cfdf6ddc73b0eedda8794386f951248d367ff2fe"


@dataclass(frozen=True)
class Scope:
    station: str
    start: str
    end: str

    def __post_init__(self) -> None:
        if not self.station or date.fromisoformat(self.end) < date.fromisoformat(self.start):
            raise ValueError("Invalid station/date scope")

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def split_scope(station: str, start: str, end: str, max_days: int) -> Iterator[Scope]:
    if max_days < 1:
        raise ValueError("max_days must be positive")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    while first <= last:
        stop = min(last, first + timedelta(days=max_days - 1))
        yield Scope(station, first.isoformat(), stop.isoformat())
        first = stop + timedelta(days=1)


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + ".tmp")
    with stage.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    stage.replace(path)


def quality_rows(result: ObservationResult) -> Iterator[dict[str, Any]]:
    """Yield unchanged publisher properties, including padded dates, not a QC view."""
    for entry in result.receipts.entries:
        if entry.authorship.value == "publisher_payload":
            document = json.loads(entry.content)
            for feature in document["features"]:
                yield feature["properties"]


def validate_result(result: ObservationResult) -> None:
    entries = result.receipts.entries
    if not entries or any(e.authorship.value != "publisher_payload" for e in entries):
        raise ValueError("Scope requires original publisher receipts, not cache excerpts")
    if not result.outcomes or any(o.status.value not in {"success", "empty", "no_match"} for o in result.outcomes):
        raise ValueError("Scope has failed or unresolved retrieval outcomes")
    if any(issue.severity in {"warning", "error"} for issue in result.issues):
        raise ValueError("Scope has unresolved source issues")
    source_keys = {(p["time_series_id"], p["time"]) for p in quality_rows(result)}
    identities = {s.series_id: s.identity.published_id for s in result.source_series}
    for row in result.data.iter_rows(named=True):
        if row["time_zone"] != "unknown":
            raise ValueError("USGS daily date must not be relabelled UTC")
        if (identities[row["series_id"]], row["time"].date().isoformat()) not in source_keys:
            raise ValueError("Observation lacks matching publisher receipt identity/date")


def acquire_scope(workspace: Path, scope: Scope) -> dict[str, Any]:
    """Caller holds workspace lock. Refresh scopes lacking a valid bundle commit."""
    os.environ["RIVRETRIEVE_CACHE_DIR"] = str((workspace / "cache").resolve())
    marker = workspace / "manifests/scopes" / f"{scope.key}.json"
    record = completed_scope(workspace, scope)
    if record is not None:
        return record
    selection = rr.find(
        provider="usgs_nwis", station=scope.station, quantity="discharge", frequency="daily", statistic="mean"
    )
    result = rr.fetch(selection, start=scope.start, end=scope.end, cache="refresh", receipts=True)
    # Preserve every acquired attempt, including failed partial results, for diagnosis.
    bundle = workspace / "receipts" / f"{scope.key}-{uuid4().hex}.rrbundle"
    atomic_write(bundle, rr.to_bundle(result))
    restored = rr.from_bundle(bundle.read_bytes())
    if not isinstance(restored, ObservationResult):
        raise ValueError("Expected result bundle")
    record = {
        "scope": asdict(scope),
        "status": "complete",
        "bundle": str(bundle.relative_to(workspace)),
        "sha256": checksum(bundle),
        "rows": result.data.height,
        "outcomes": [o.model_dump(mode="json") for o in result.outcomes],
        "rr_revision": RR_REVISION,
        "uv_lock_sha256": checksum(Path(__file__).resolve().parent.parent / "uv.lock"),
        "script_sha256": checksum(Path(__file__)),
    }
    try:
        validate_result(restored)
    except ValueError as error:
        record.update(status="failed", reason=str(error))
        atomic_write(marker, json.dumps(record, sort_keys=True, indent=2).encode())
        raise
    atomic_write(marker, json.dumps(record, sort_keys=True, indent=2).encode())
    return record


def bundle_records(workspace: Path) -> Iterator[tuple[dict[str, Any], ObservationResult]]:
    for marker in sorted((workspace / "manifests/scopes").glob("*.json")):
        record = json.loads(marker.read_text())
        if record["status"] != "complete":
            raise ValueError(f"Unfinished scope: {marker.name}")
        bundle = workspace / record["bundle"]
        if checksum(bundle) != record["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {marker.name}")
        result = rr.from_bundle(bundle.read_bytes())
        if not isinstance(result, ObservationResult):
            raise ValueError("Expected observation bundle")
        validate_result(result)
        yield record, result


def validate_bundles(workspace: Path) -> dict[str, Any]:
    records = [record for record, _ in bundle_records(workspace)]
    if not records:
        raise ValueError("No completed acquisition scopes")
    plan = workspace / "manifests/plan.json"
    complete = False
    if plan.exists():
        planned = json.loads(plan.read_text())["scopes"]
        complete = {Scope(**r["scope"]).key for r in records} >= {Scope(**s).key for s in planned}
    return {"scopes": len(records), "rows": sum(r["rows"] for r in records), "planned_scopes_complete": complete}


def validate_workspace(workspace: Path, *, cache_root: str | None = None) -> dict[str, Any]:
    """Verify native reads; reuse can attempt HTTP but a fallback result is rejected."""
    os.environ["RIVRETRIEVE_CACHE_DIR"] = cache_root or str((workspace / "cache").resolve())
    audit = rr.audit_cache("usgs_nwis")
    summary = validate_bundles(workspace)
    for record, result in bundle_records(workspace):
        scope = Scope(**record["scope"])
        selection = rr.find(
            provider="usgs_nwis", station=scope.station, quantity="discharge", frequency="daily", statistic="mean"
        )
        held = rr.fetch(selection, start=scope.start, end=scope.end, cache="reuse", receipts=True)
        if not held.receipts.entries or any(e.authorship.value != "store_excerpt" for e in held.receipts.entries):
            raise ValueError("Restore readback required source acquisition instead of cache reuse")
        keys = ["station_id", "series_id", "facts_id", "time"]
        assert_frame_equal(held.data.sort(keys), result.data.sort(keys))
    return {**summary, "cache_audit": str(audit)}


def plan_scopes(inventory: dict[str, Any], max_days: int) -> list[Scope]:
    if inventory["unresolved"]:
        raise ValueError("Inventory has unresolved selection/history; resolve before planning full acquisition")
    return [
        scope
        for station in inventory["stations"]
        for scope in split_scope(station["station"], station["start"], station["end"], max_days)
    ]


def cached_outcome_ids(workspace: Path) -> set[str]:
    """Read validated active and inventory-support acquisition identities."""
    os.environ["RIVRETRIEVE_CACHE_DIR"] = str((workspace / "cache").resolve())
    status = rr.cache_status("usgs_nwis")
    cached = {outcome.outcome_id for outcome in getattr(status.manifest, "outcomes", ())}
    # RR validates these references against retained primary/supporting outcomes.
    cached.update(
        reference.removeprefix("retrieval-outcome:")
        for inventory in getattr(status.manifest, "inventories", ())
        for reference in inventory.evidence
        if reference.startswith("retrieval-outcome:")
    )
    return cached


def completed_scope(workspace: Path, scope: Scope, *, cached_outcomes: set[str] | None = None) -> dict[str, Any] | None:
    """Check bundle and cache; a supplied snapshot is valid only before cache mutation."""
    marker = workspace / "manifests/scopes" / f"{scope.key}.json"
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    bundle = workspace / record["bundle"]
    if cached_outcomes is None:
        cached_outcomes = cached_outcome_ids(workspace)
    required_outcomes = {outcome["outcome_id"] for outcome in record["outcomes"]}
    if (
        record["status"] == "complete"
        and bundle.is_file()
        and checksum(bundle) == record["sha256"]
        and required_outcomes
        and required_outcomes <= cached_outcomes
    ):
        return record
    return None
