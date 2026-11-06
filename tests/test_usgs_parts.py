"""Independent parts preserve the frozen station/date coverage."""

from datetime import date
from pathlib import Path

import pytest
from rivretrieve._internal.observations import ObservationResult

from scripts.usgs_parts import make_parts


def test_union_window_capacity_and_station_page_rounding() -> None:
    scopes = [
        {"station": "01", "start": "2000-01-01", "end": "2000-01-02", "series": ["a"]},
        {"station": "02", "start": "2020-01-01", "end": "2020-01-02", "series": ["b"]},
    ]
    parts = make_parts(scopes, max_rows=100)
    assert len(parts) == 2  # Sum of original history is four, but union span is large.
    assert [p["estimated_pages"] for p in parts] == [1, 1]
    same = make_parts([scopes[0], {**scopes[0], "station": "03"}])
    assert same[0]["estimated_pages"] == 2  # Two station requests, not ceil(total/10000).


def test_long_history_splits_without_gaps_or_duplicate_coverage() -> None:
    scopes = [{"station": "01", "start": "2000-01-01", "end": "2000-01-20", "series": ["a", "b"]}]
    parts = make_parts(scopes, max_rows=24)
    covered = [scope for part in parts for scope in part["scopes"]]
    assert [(s["start"], s["end"]) for s in covered] == [
        ("2000-01-01", "2000-01-08"),
        ("2000-01-09", "2000-01-16"),
        ("2000-01-17", "2000-01-20"),
    ]
    assert all(p["estimated_rows"] <= 24 for p in parts)
    assert all(s["series"] == ["a", "b"] for s in covered)
    assert sum((date.fromisoformat(s["end"]) - date.fromisoformat(s["start"])).days + 1 for s in covered) == 20


def test_assignment_rejects_gap_duplicate_and_wrong_station() -> None:

    from scripts.usgs_parts import check_assignments

    planned = [{"station": "01", "start": "2000-01-01", "end": "2000-01-04"}]
    assigned = [
        {"station": "01", "start": "2000-01-01", "end": "2000-01-02"},
        {"station": "01", "start": "2000-01-03", "end": "2000-01-04"},
    ]
    check_assignments(planned, assigned)
    for bad in [assigned[:1], assigned + assigned[:1], [*assigned, {**assigned[0], "station": "02"}]]:
        with pytest.raises(ValueError, match="coverage"):
            check_assignments(planned, bad)


def test_acquired_part_is_not_refetched_and_needs_separate_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path
    from unittest.mock import Mock

    import rivretrieve as rr

    from scripts.daymet_archive import checksum
    from scripts.usgs_parts import acquire_part

    result = rr.from_bundle((Path(__file__).parent / "fixtures/usgs_daily.rrbundle").read_bytes())
    part = {
        "id": "part-00000",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "estimated_pages": 1,
        "estimated_rows": 6,
        "scopes": [
            {
                "station": "07374000",
                "start": "2024-01-01",
                "end": "2024-01-02",
                "series": ["c9d823a2491f4b639656a11b35a7625d"],
            }
        ],
    }
    monkeypatch.setattr("scripts.usgs_parts.load_plan", lambda _: {"parts": [part]})
    monkeypatch.setattr("scripts.usgs_parts.reserve", lambda *args: "reservation", raising=False)
    monkeypatch.setattr("scripts.usgs_parts.reconcile", lambda *args: None, raising=False)
    fetch = Mock(return_value=result)
    monkeypatch.setattr(rr, "fetch", fetch)
    monkeypatch.setenv("USGS_API_KEY", "test-key")
    first = acquire_part(tmp_path, part["id"])
    assert first["status"] == "acquired"
    assert checksum(tmp_path / first["bundle"]) == first["sha256"]
    assert acquire_part(tmp_path, part["id"]) == first
    fetch.assert_called_once()
    assert not (tmp_path / "manifests/parts/part-00000/verified.json").exists()
    quota_marker = tmp_path / "manifests/parts/part-00000/quota-reconciled.json"
    saved_quota = quota_marker.read_bytes()
    quota_marker.unlink()
    with pytest.raises(ValueError, match="unresolved quota"):
        acquire_part(tmp_path, part["id"])
    quota_marker.write_bytes(saved_quota)
    (tmp_path / first["bundle"]).unlink()

    with pytest.raises(ValueError, match="bundle"):
        acquire_part(tmp_path, part["id"])
    fetch.assert_called_once()


def test_part_verification_requires_native_read_and_preserves_known_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from dataclasses import replace
    from pathlib import Path

    import rivretrieve as rr
    from rivretrieve._internal.observations import ReceiptAuthorship

    from scripts.daymet_archive import checksum
    from scripts.usgs_parts import verify_part

    payload = (Path(__file__).parent / "fixtures/usgs_daily.rrbundle").read_bytes()
    result = rr.from_bundle(payload)
    assert isinstance(result, ObservationResult)
    part = {
        "id": "part-00000",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "scopes": [{"station": "07374000", "series": ["c9d823a2491f4b639656a11b35a7625d"]}],
    }
    monkeypatch.setattr("scripts.usgs_parts.load_plan", lambda _: {"parts": [part]})
    bundle = tmp_path / "receipts/parts/part-00000.rrbundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(payload)
    directory = tmp_path / "manifests/parts/part-00000"
    directory.mkdir(parents=True)
    (directory / "quota-reconciled.json").write_text("{}")
    (directory / "acquired.json").write_text(
        json.dumps({"part": part, "bundle": str(bundle.relative_to(tmp_path)), "sha256": checksum(bundle)})
    )
    monkeypatch.setattr(rr, "audit_cache", lambda _: "audited")
    monkeypatch.setattr(rr, "fetch", lambda *args, **kwargs: result)
    with pytest.raises(ValueError, match="fallback"):
        verify_part(tmp_path, part["id"])
    assert not (directory / "verified.json").exists()
    receipts = replace(
        result.receipts,
        entries=tuple(replace(e, authorship=ReceiptAuthorship.STORE_EXCERPT) for e in result.receipts.entries),
    )
    held = result.model_copy(update={"receipts": receipts})
    monkeypatch.setattr(rr, "fetch", lambda *args, **kwargs: held)
    assert verify_part(tmp_path, part["id"])["status"] == "complete"
    part["scopes"][0]["series"].append("not-returned")
    (directory / "acquired.json").write_text(
        json.dumps({"part": part, "bundle": str(bundle.relative_to(tmp_path)), "sha256": checksum(bundle)})
    )
    with pytest.raises(ValueError, match="Known source"):
        verify_part(tmp_path, part["id"])


def test_freeze_requires_original_backing_and_maps_remaining(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from scripts.daymet_archive import checksum
    from scripts.usgs_acquire import Scope
    from scripts.usgs_parts import load_plan, plan_parts

    directory = tmp_path / "manifests"
    directory.mkdir()
    scopes = [{"station": s, "start": "2000-01-01", "end": "2000-01-02"} for s in ["01", "02"]]
    (directory / "inventory.json").write_text(
        json.dumps({"unresolved": [], "stations": [{**s, "series": [s["station"]]} for s in scopes]})
    )
    (directory / "plan.json").write_text(
        json.dumps({"scopes": scopes, "inventory_sha256": checksum(directory / "inventory.json")})
    )
    marker = directory / "scopes" / f"{Scope(**scopes[0]).key}.json"
    marker.parent.mkdir()
    marker.write_text("{}")
    monkeypatch.setattr("scripts.usgs_parts.cached_outcome_ids", lambda _: set())
    monkeypatch.setattr("scripts.usgs_parts.completed_scope", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="lost backing"):
        plan_parts(tmp_path)
    monkeypatch.setattr(
        "scripts.usgs_parts.completed_scope", lambda _, scope, **kwargs: {} if scope.station == "01" else None
    )
    plan = plan_parts(tmp_path)
    assert [r["scope"] for r in plan["original"]] == scopes[:1]
    assert [s["station"] for p in plan["parts"] for s in p["scopes"]] == ["02"]
    assert load_plan(tmp_path) == plan
    (directory / "inventory.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        load_plan(tmp_path)
