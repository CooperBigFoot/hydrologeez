"""Independent native cache parts for a frozen USGS acquisition inventory."""

from __future__ import annotations

import gc
import json
import math
import os
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import rivretrieve as rr
from polars.testing import assert_frame_equal
from rivretrieve._internal.observations import ObservationResult

from scripts.daymet_archive import checksum
from scripts.usgs_acquire import (
    RR_REVISION,
    Scope,
    atomic_write,
    cached_outcome_ids,
    completed_scope,
    split_scope,
    validate_result,
)
from scripts.usgs_archive import workspace_lock
from scripts.usgs_quota import reconcile, reserve

MAX_STATIONS = 50
MAX_ROWS = 750_000
PADDING_DAYS = 4
PAGE_ROWS = 10_000


def part_size(scopes: list[dict[str, Any]]) -> dict[str, Any]:
    start = min(s["start"] for s in scopes)
    end = max(s["end"] for s in scopes)
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1 + PADDING_DAYS
    series = {s["station"]: len(s["series"]) for s in scopes}
    return {
        "start": start,
        "end": end,
        "estimated_rows": days * sum(series.values()),
        "estimated_pages": sum(math.ceil(days * count / PAGE_ROWS) for count in series.values()),
    }


def make_parts(scopes: list[dict[str, Any]], *, max_rows: int = MAX_ROWS) -> list[dict[str, Any]]:
    """Size against the padded shared request window, never omit source dates."""
    parts: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for scope in scopes:
        days = max_rows // len(scope["series"]) - PADDING_DAYS
        if days < 1:
            raise ValueError("Series count exceeds part capacity")
        for piece in split_scope(scope["station"], scope["start"], scope["end"], days):
            row = {**scope, "start": piece.start, "end": piece.end}
            candidate = [*current, row]
            if current and (
                len(candidate) > MAX_STATIONS
                or any(s["station"] == piece.station for s in current)
                or part_size(candidate)["estimated_rows"] > max_rows
            ):
                parts.append({"scopes": current, **part_size(current)})
                current = []
            current.append(row)
    if current:
        parts.append({"scopes": current, **part_size(current)})
    for index, part in enumerate(parts):
        part["id"] = f"part-{index:05}"
    return parts


def check_assignments(planned: list[dict[str, Any]], assigned: list[dict[str, Any]]) -> None:
    def intervals(rows: list[dict[str, Any]]) -> dict[str, list[tuple[date, date]]]:
        grouped: dict[str, list[tuple[date, date]]] = {}
        for row in rows:
            first, last = date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
            if last < first:
                raise ValueError("Invalid coverage interval")
            grouped.setdefault(row["station"], []).append((first, last))
        result: dict[str, list[tuple[date, date]]] = {}
        for station, spans in grouped.items():
            merged: list[tuple[date, date]] = []
            for first, last in sorted(spans):
                if merged and first <= merged[-1][1]:
                    raise ValueError("Duplicate logical coverage")
                if merged and first == merged[-1][1] + timedelta(days=1):
                    merged[-1] = (merged[-1][0], last)
                else:
                    merged.append((first, last))
            result[station] = merged
        return result

    if intervals(planned) != intervals(assigned):
        raise ValueError("Assignments do not equal planned coverage")


def load_plan(workspace: Path) -> dict[str, Any]:
    plan = json.loads((workspace / "manifests/parts-plan.json").read_text())
    for name in ("inventory", "plan"):
        if checksum(workspace / f"manifests/{name}.json") != plan[f"{name}_sha256"]:
            raise ValueError("Frozen source plan changed")
    assigned = [r["scope"] for r in plan["original"]] + [s for part in plan["parts"] for s in part["scopes"]]
    check_assignments(json.loads((workspace / "manifests/plan.json").read_text())["scopes"], assigned)
    stations = {s["station"]: s for s in json.loads((workspace / "manifests/inventory.json").read_text())["stations"]}
    for part in plan["parts"]:
        if any(set(scope["series"]) != set(stations[scope["station"]]["series"]) for scope in part["scopes"]):
            raise ValueError("Part source identities differ from frozen inventory")
        if any(part[key] != value for key, value in part_size(part["scopes"]).items()):
            raise ValueError("Part request or estimates changed")
    return plan


def plan_parts(workspace: Path) -> dict[str, Any]:
    path = workspace / "manifests/parts-plan.json"
    if path.exists():
        return load_plan(workspace)
    inventory = json.loads((workspace / "manifests/inventory.json").read_text())
    if inventory["unresolved"]:
        raise ValueError("Unresolved inventory")
    plan = json.loads((workspace / "manifests/plan.json").read_text())
    if plan["inventory_sha256"] != checksum(workspace / "manifests/inventory.json"):
        raise ValueError("Frozen inventory changed")
    stations = {s["station"]: s for s in inventory["stations"]}
    cached = cached_outcome_ids(workspace)
    original, remaining = [], []
    for item in plan["scopes"]:
        scope = Scope(**item)
        record = completed_scope(workspace, scope, cached_outcomes=cached)
        if record is not None:
            original.append(
                {"scope": asdict(scope), "marker_sha256": checksum(workspace / f"manifests/scopes/{scope.key}.json")}
            )
        else:
            if (workspace / f"manifests/scopes/{scope.key}.json").exists():
                raise ValueError("Existing scope lost backing; resolve before freezing independent parts")
            remaining.append({**item, "series": stations[scope.station]["series"]})
    result = {
        "inventory_sha256": checksum(workspace / "manifests/inventory.json"),
        "plan_sha256": checksum(workspace / "manifests/plan.json"),
        "policy": {"max_stations": MAX_STATIONS, "max_rows": MAX_ROWS, "padding_days": PADDING_DAYS},
        "original": original,
        "parts": make_parts(remaining),
    }
    check_assignments(
        plan["scopes"], [r["scope"] for r in original] + [s for part in result["parts"] for s in part["scopes"]]
    )
    atomic_write(path, json.dumps(result, indent=2).encode())
    return result


def get_part(workspace: Path, part_id: str) -> dict[str, Any]:
    return next(part for part in load_plan(workspace)["parts"] if part["id"] == part_id)


def acquire_part(workspace: Path, part_id: str) -> dict[str, Any]:
    part = get_part(workspace, part_id)
    directory = workspace / "manifests/parts" / part_id
    with workspace_lock(directory):
        marker = directory / "acquired.json"
        if marker.exists():
            record = json.loads(marker.read_text())
            bundle = workspace / record["bundle"]
            if not bundle.exists() or checksum(bundle) != record["sha256"]:
                raise ValueError("Acquired bundle missing or corrupt; explicit recovery required")
            if not (directory / "quota-reconciled.json").exists():
                raise ValueError("Acquired part has unresolved quota; explicit reconciliation required")
            return record
        attempt = directory / "attempt.json"
        if attempt.exists():
            raise ValueError("Interrupted part attempt requires explicit recovery; not automatically refetched")
        if not os.environ.get("USGS_API_KEY"):
            raise ValueError("Independent part quota requires USGS_API_KEY")
        reservation = reserve(workspace, part_id, part["estimated_pages"] * 3)
        atomic_write(attempt, json.dumps({"part": part, "reservation": reservation}).encode())
        os.environ["RIVRETRIEVE_CACHE_DIR"] = str(workspace / "cache/parts" / part_id)
        selection = rr.pick(
            rr.find(provider="usgs_nwis", quantity="discharge", frequency="daily", statistic="mean"),
            station=[scope["station"] for scope in part["scopes"]],
        )
        assert not isinstance(selection, ObservationResult)
        result = rr.fetch(selection, start=part["start"], end=part["end"], cache="refresh", receipts=True)
        bundle = workspace / "receipts/parts" / f"{part_id}.rrbundle"
        atomic_write(bundle, rr.to_bundle(result))
        calls = result.provenance.calls_made
        identities = {str(call["call_id"]) for call in calls if call.get("call_id")}
        actual = len(identities)
        record = {
            "status": "acquired",
            "part": part,
            "bundle": str(bundle.relative_to(workspace)),
            "sha256": checksum(bundle),
            "rr_revision": RR_REVISION,
            "uv_lock_sha256": checksum(Path(__file__).resolve().parent.parent / "uv.lock"),
            "rows": result.data.height,
            "exceeds_row_estimate": result.data.height > part["estimated_rows"],
            "actual_attempts": actual,
            "outcomes": [o.model_dump(mode="json") for o in result.outcomes],
        }
        atomic_write(marker, json.dumps(record, indent=2).encode())
        if not actual or len(identities) != len(calls):
            raise ValueError("Attempt provenance incomplete; quota reservation retained")
        reconcile(workspace, reservation, actual)
        atomic_write(
            directory / "quota-reconciled.json", json.dumps({"reservation": reservation, "actual": actual}).encode()
        )
        if any(o.status.value not in {"success", "empty", "no_match"} for o in result.outcomes) or any(
            issue.severity in {"warning", "error"} for issue in result.issues
        ):
            raise ValueError("Part retrieval failed; saved evidence requires review")
        return record


def verify_part(
    workspace: Path, part_id: str, *, cache_root: str | None = None, record_completion: bool = True
) -> dict[str, Any]:
    part = get_part(workspace, part_id)
    directory = workspace / "manifests/parts" / part_id
    with workspace_lock(directory):
        if not (directory / "quota-reconciled.json").exists():
            raise ValueError("Part quota remains unresolved")
        record = json.loads((directory / "acquired.json").read_text())
        if record["part"] != part:
            raise ValueError("Acquired part differs from frozen assignment")
        bundle = workspace / record["bundle"]
        if checksum(bundle) != record["sha256"]:
            raise ValueError("Part bundle checksum mismatch")
        result = rr.from_bundle(bundle.read_bytes())
        if not isinstance(result, ObservationResult):
            raise ValueError("Expected observation bundle")
        validate_result(result)
        if {o.station_id for o in result.outcomes} != {s["station"] for s in part["scopes"]}:
            raise ValueError("Requested stations not accounted by outcomes")
        identities = {s.series_id: s.identity.published_id for s in result.source_series}
        for scope in part["scopes"]:
            accounted = {
                identities[o.series_id]
                for o in result.outcomes
                if o.station_id == scope["station"]
                and o.series_id in identities
                and o.status.value in {"success", "empty"}
            }
            if not set(scope["series"]) <= accounted:
                raise ValueError("Known source series lacks successful or empty outcome")
        discovered = [s.model_dump(mode="json") for s in result.source_series]
        frame = result.data
        del result
        gc.collect()
        os.environ["RIVRETRIEVE_CACHE_DIR"] = cache_root or str(workspace / "cache/parts" / part_id)
        audit = rr.audit_cache("usgs_nwis")
        keys = ["station_id", "series_id", "facts_id", "time"]
        for scope in part["scopes"]:
            selection = rr.find(
                provider="usgs_nwis",
                station=scope["station"],
                quantity="discharge",
                frequency="daily",
                statistic="mean",
            )
            held = rr.fetch(selection, start=part["start"], end=part["end"], cache="reuse", receipts=True)
            if not held.receipts.entries or any(e.authorship.value != "store_excerpt" for e in held.receipts.entries):
                raise ValueError("Verification attempted source fallback")
            assert_frame_equal(
                frame.filter(pl.col("station_id") == scope["station"]).sort(keys),
                held.data.sort(keys),
                check_exact=True,
            )
            del held
        verified = {
            "status": "complete",
            "bundle_sha256": record["sha256"],
            "part": part,
            "source_series": discovered,
            "cache_audit": str(audit),
            "rows": frame.height,
        }
        if record_completion:
            atomic_write(directory / "verified.json", json.dumps(verified, indent=2).encode())
        return verified


def verify_parts(workspace: Path, *, cache_root: str | None = None) -> dict[str, Any]:
    """Verify exact logical coverage and each independent root, not one merged store."""
    from scripts.usgs_acquire import validate_workspace

    plan = load_plan(workspace)
    if cache_root:
        os.environ["RIVRETRIEVE_CACHE_DIR"] = cache_root
        status = rr.cache_status("usgs_nwis")
        original_ids = {o.outcome_id for o in getattr(status.manifest, "outcomes", ())}
        original_ids.update(
            ref.removeprefix("retrieval-outcome:")
            for item in getattr(status.manifest, "inventories", ())
            for ref in item.evidence
            if ref.startswith("retrieval-outcome:")
        )
    else:
        original_ids = cached_outcome_ids(workspace)
    for original in plan["original"]:
        scope = Scope(**original["scope"])
        marker = workspace / f"manifests/scopes/{scope.key}.json"
        if (
            checksum(marker) != original["marker_sha256"]
            or completed_scope(workspace, scope, cached_outcomes=original_ids) is None
        ):
            raise ValueError("Original acquisition backing changed or missing")
    original_summary = validate_workspace(workspace, cache_root=cache_root) if plan["original"] else None
    rows = 0
    for part in plan["parts"]:
        verified = verify_part(
            workspace,
            part["id"],
            cache_root=f"{cache_root}/parts/{part['id']}" if cache_root else None,
            record_completion=False,
        )
        rows += verified["rows"]
    return {
        "original": original_summary,
        "parts": len(plan["parts"]),
        "rows": rows,
        "planned_scopes_complete": True,
        "parts_plan_sha256": checksum(workspace / "manifests/parts-plan.json"),
    }
