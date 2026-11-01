from pathlib import Path

import pytest

from scripts import usgs_quota as quota


def test_reservations_enforce_shared_budget_and_reconcile(tmp_path: Path) -> None:
    first = quota.reserve(tmp_path, "part-a", 600)
    with pytest.raises(ValueError, match="budget"):
        quota.reserve(tmp_path, "part-b", 301)
    with pytest.raises(ValueError, match="outstanding"):
        quota.reserve(tmp_path, "part-a", 1)
    quota.reconcile(tmp_path, first, 400)
    quota.reserve(tmp_path, "part-b", 500)
    with pytest.raises(ValueError, match="budget"):
        quota.reserve(tmp_path, "part-c", 1)


def test_reconciled_reservations_expire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quota.time, "time", lambda: 1000.0)
    reservation = quota.reserve(tmp_path, "part-a", 900)
    quota.reconcile(tmp_path, reservation, 900)
    monkeypatch.setattr(quota.time, "time", lambda: 4599.0)
    with pytest.raises(ValueError, match="budget"):
        quota.reserve(tmp_path, "part-b", 1)
    monkeypatch.setattr(quota.time, "time", lambda: 4600.0)
    quota.reserve(tmp_path, "part-a", 900)


def test_budget_rejection_reports_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quota.time, "time", lambda: 1000.0)
    reservation = quota.reserve(tmp_path, "part-a", 900)
    quota.reconcile(tmp_path, reservation, 900)
    monkeypatch.setattr(quota.time, "time", lambda: 1100.0)
    with pytest.raises(ValueError, match="3500 seconds"):
        quota.reserve(tmp_path, "part-b", 1)


def test_concurrent_reservations_cannot_exceed_budget(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)

    def attempt(part_id: str) -> bool:
        barrier.wait()
        try:
            quota.reserve(tmp_path, part_id, 600)
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, ("part-a", "part-b"))) == [False, True]


def test_pending_reservation_does_not_expire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quota.time, "time", lambda: 1000.0)
    reservation = quota.reserve(tmp_path, "part-a", 900)
    monkeypatch.setattr(quota.time, "time", lambda: 8200.0)
    with pytest.raises(ValueError, match="reconciliation"):
        quota.reserve(tmp_path, "part-b", 1)
    with pytest.raises(ValueError, match="outstanding"):
        quota.reserve(tmp_path, "part-a", 1)
    quota.reconcile(tmp_path, reservation, 900)
    monkeypatch.setattr(quota.time, "time", lambda: 11799.0)
    with pytest.raises(ValueError, match="1 seconds"):
        quota.reserve(tmp_path, "part-b", 1)
    monkeypatch.setattr(quota.time, "time", lambda: 11800.0)
    quota.reserve(tmp_path, "part-b", 900)
