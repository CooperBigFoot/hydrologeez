"""Acquisition commits require recoverable source quality, not only cached values."""

import json
from pathlib import Path

import pytest
import rivretrieve as rr
from rivretrieve._internal.observations import ObservationResult
from rivretrieve._internal.source_series import RetrievalOutcome

from scripts.usgs_acquire import Scope, acquire_scope, quality_rows

FIXTURE = Path(__file__).parent / "fixtures/usgs_daily.rrbundle"


def test_quality_retains_source_dates_and_independent_flags() -> None:
    result = rr.from_bundle(FIXTURE.read_bytes())
    assert isinstance(result, ObservationResult)
    rows = list(quality_rows(result))
    requested = [r for r in rows if r["time"] in ("2024-01-01", "2024-01-02")]
    assert len(requested) == 2
    assert {r["time_series_id"] for r in requested} == {"c9d823a2491f4b639656a11b35a7625d"}
    assert all(r["approval_status"] == "Approved" and r["qualifier"] == ["ESTIMATED"] for r in requested)


def test_missing_bundle_reacquires_with_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = rr.from_bundle(FIXTURE.read_bytes())
    assert isinstance(result, ObservationResult)
    calls: list[str] = []
    from types import SimpleNamespace

    monkeypatch.setattr(
        rr, "cache_status", lambda _: SimpleNamespace(manifest=SimpleNamespace(outcomes=result.outcomes))
    )

    def fetch(*args: object, **kwargs: object) -> ObservationResult:
        calls.append(str(kwargs["cache"]))
        return result

    monkeypatch.setattr(rr, "fetch", fetch)
    scope = Scope("07374000", "2024-01-01", "2024-01-02")
    record = acquire_scope(tmp_path, scope)
    assert record["status"] == "complete"
    acquire_scope(tmp_path, scope)
    assert calls == ["refresh"]
    (tmp_path / str(record["bundle"])).unlink()
    acquire_scope(tmp_path, scope)
    assert calls == ["refresh", "refresh"]


def test_cache_excerpt_cannot_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = rr.from_bundle(FIXTURE.read_bytes())
    assert isinstance(result, ObservationResult)
    result = result.model_copy(update={"receipts": result.receipts.__class__(result.receipts.provider_id)})
    monkeypatch.setattr(rr, "fetch", lambda *args, **kwargs: result)
    scope = Scope("07374000", "2024-01-01", "2024-01-02")
    with pytest.raises(ValueError, match="publisher"):
        acquire_scope(tmp_path, scope)
    assert json.loads((tmp_path / "manifests/scopes" / f"{scope.key}.json").read_text())["status"] == "failed"


def test_failed_scope_retains_bundle_and_failure_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = rr.from_bundle(FIXTURE.read_bytes())
    assert isinstance(result, ObservationResult)
    result = result.model_copy(update={"outcomes": ()})
    monkeypatch.setattr(rr, "fetch", lambda *args, **kwargs: result)
    scope = Scope("07374000", "2024-01-01", "2024-01-02")
    with pytest.raises(ValueError, match="outcomes"):
        acquire_scope(tmp_path, scope)
    record = json.loads((tmp_path / "manifests/scopes" / f"{scope.key}.json").read_text())
    assert record["status"] == "failed"
    assert (tmp_path / record["bundle"]).is_file()


def test_scope_windows_are_inclusive_without_calendar_gaps() -> None:
    from scripts.usgs_acquire import split_scope

    assert list(split_scope("07374000", "2024-02-28", "2024-03-02", 2)) == [
        Scope("07374000", "2024-02-28", "2024-02-29"),
        Scope("07374000", "2024-03-01", "2024-03-02"),
    ]


def test_missing_cache_reacquires_even_with_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = rr.from_bundle(FIXTURE.read_bytes())
    assert isinstance(result, ObservationResult)
    calls: list[str] = []

    def fetch(*args: object, **kwargs: object) -> ObservationResult:
        calls.append(str(kwargs["cache"]))
        (tmp_path / "cache/usgs_nwis/store").mkdir(parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(rr, "fetch", fetch)
    scope = Scope("07374000", "2024-01-01", "2024-01-02")
    acquire_scope(tmp_path, scope)
    import shutil

    shutil.rmtree(tmp_path / "cache")
    acquire_scope(tmp_path, scope)
    assert calls == ["refresh", "refresh"]


def test_verification_blocks_network_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    from scripts.daymet_archive import checksum
    from scripts.usgs_acquire import validate_workspace

    bundle = tmp_path / "receipts/result.rrbundle"
    bundle.parent.mkdir()
    bundle.write_bytes(FIXTURE.read_bytes())
    marker = tmp_path / "manifests/scopes/scope.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "rows": 2,
                "bundle": "receipts/result.rrbundle",
                "sha256": checksum(bundle),
                "scope": {"station": "07374000", "start": "2024-01-01", "end": "2024-01-02"},
            }
        )
    )
    monkeypatch.setattr(rr, "audit_cache", lambda _: None)

    def fetch(*args: object, **kwargs: object) -> ObservationResult:
        requests.Session().get("https://example.invalid/not-allowed")
        raise AssertionError("Network should have been denied")

    monkeypatch.setattr(rr, "fetch", fetch)

    def deny_http(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Source HTTP forbidden")

    monkeypatch.setattr(requests.sessions.Session, "send", deny_http)
    with pytest.raises(RuntimeError, match="Source HTTP forbidden"):
        validate_workspace(tmp_path)


def test_plan_refuses_unresolved_inventory_and_preserves_station_spans() -> None:
    from scripts.usgs_acquire import plan_scopes

    inventory = {"stations": [{"station": "07374000", "start": "2024-01-01", "end": "2024-01-03"}], "unresolved": []}
    assert plan_scopes(inventory, 2) == [
        Scope("07374000", "2024-01-01", "2024-01-02"),
        Scope("07374000", "2024-01-03", "2024-01-03"),
    ]
    inventory["unresolved"] = [{"station": "unknown"}]
    with pytest.raises(ValueError, match="unresolved"):
        plan_scopes(inventory, 2)


def test_cli_plan_binds_inventory_and_refuses_changed_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.usgs import main

    inventory = tmp_path / "manifests/inventory.json"
    inventory.parent.mkdir()
    inventory.write_text(
        json.dumps(
            {"stations": [{"station": "07374000", "start": "2024-01-01", "end": "2024-01-03"}], "unresolved": []}
        )
    )
    monkeypatch.setattr("sys.argv", ["usgs", "plan", str(tmp_path), "--max-days", "2"])
    main()
    plan = json.loads((tmp_path / "manifests/plan.json").read_text())
    assert len(plan["scopes"]) == 2
    assert json.loads(capsys.readouterr().out)["result"]["planned_scopes"] == 2
    monkeypatch.setattr("sys.argv", ["usgs", "plan", str(tmp_path), "--max-days", "3"])
    with pytest.raises(ValueError, match="Existing acquisition plan differs"):
        main()


def test_bundle_only_restore_validation_needs_no_native_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.usgs_acquire import validate_bundles

    result = rr.from_bundle(FIXTURE.read_bytes())
    assert isinstance(result, ObservationResult)
    monkeypatch.setattr(rr, "fetch", lambda *args, **kwargs: result)
    acquire_scope(tmp_path, Scope("07374000", "2024-01-01", "2024-01-02"))
    monkeypatch.setattr(
        rr, "audit_cache", lambda _: pytest.fail("Bundle verification must not audit copied native cache")
    )
    assert validate_bundles(tmp_path) == {"scopes": 1, "rows": 2, "planned_scopes_complete": False}


def test_lost_cache_does_not_skip_scopes_after_partial_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from scripts.usgs_acquire import completed_scope

    result = rr.from_bundle(FIXTURE.read_bytes())
    assert isinstance(result, ObservationResult)
    cached: list[RetrievalOutcome] = []
    monkeypatch.setattr(rr, "cache_status", lambda _: SimpleNamespace(manifest=SimpleNamespace(outcomes=tuple(cached))))

    def fetch(*args: object, **kwargs: object) -> ObservationResult:
        outcome = result.outcomes[0].model_copy(update={"outcome_id": str(kwargs["start"])})
        cached.append(outcome)
        return result.model_copy(update={"outcomes": (outcome,)})

    monkeypatch.setattr(rr, "fetch", fetch)
    first = Scope("07374000", "2024-01-01", "2024-01-02")
    second = Scope("07374000", "2024-01-03", "2024-01-04")
    acquire_scope(tmp_path, first)
    acquire_scope(tmp_path, second)
    assert completed_scope(tmp_path, second) is not None
    cached.clear()
    assert completed_scope(tmp_path, first) is None
    acquire_scope(tmp_path, first)
    assert completed_scope(tmp_path, second) is None
    acquire_scope(tmp_path, second)
    assert completed_scope(tmp_path, second) is not None


def test_completion_retains_multipage_inventory_support(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from scripts.usgs_acquire import completed_scope

    result = rr.from_bundle(FIXTURE.read_bytes())
    assert isinstance(result, ObservationResult)
    aggregate = result.outcomes[0]
    page = aggregate.model_copy(update={"outcome_id": "page-acquisition"})
    result = result.model_copy(update={"outcomes": (page, aggregate)})
    monkeypatch.setattr(rr, "fetch", lambda *args, **kwargs: result)
    scope = Scope("07374000", "2024-01-01", "2024-01-02")
    record = acquire_scope(tmp_path, scope)
    # Public cache_status omits supporting_outcomes but exposes validated references.
    inventory = result.inventories[0].model_copy(
        update={
            "evidence": (
                "Exact publisher pages; finite requested observation scope, not historical availability",
                "retrieval-outcome:page-acquisition",
            )
        }
    )
    manifest = SimpleNamespace(outcomes=(aggregate,), inventories=(inventory,))
    monkeypatch.setattr(rr, "cache_status", lambda _: SimpleNamespace(manifest=manifest))
    assert completed_scope(tmp_path, scope) == record
    manifest.outcomes = ()
    assert completed_scope(tmp_path, scope) is None
    manifest.outcomes = (aggregate,)
    manifest.inventories = ()
    assert completed_scope(tmp_path, scope) is None
