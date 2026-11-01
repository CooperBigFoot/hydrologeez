"""Shared rolling request budget for USGS acquisition parts."""

from __future__ import annotations

import fcntl
import json
import math
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict, cast
from uuid import uuid4

from scripts.usgs_acquire import atomic_write

KEYED_HOURLY_LIMIT = 1000
BUDGET = 900
WINDOW_SECONDS = 3600


class Reservation(TypedDict):
    part_id: str
    attempts: int
    timestamp: float
    reconciled: bool


@contextmanager
def _ledger(workspace: Path) -> Generator[dict[str, Reservation]]:
    path = workspace / "manifests/quota.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            records = cast(dict[str, Reservation], json.loads(path.read_text())) if path.exists() else {}
            yield records
            atomic_write(path, json.dumps(records, indent=2).encode())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def reserve(workspace: Path, part_id: str, attempts: int) -> str:
    """Reserve worst-case attempts before HTTP; reject rather than wait."""
    if attempts < 1 or attempts > BUDGET:
        raise ValueError(f"Reservation attempts must be between 1 and {BUDGET}")
    with _ledger(workspace) as records:
        now = time.time()
        active = [row for row in records.values() if not row["reconciled"] or now - row["timestamp"] < WINDOW_SECONDS]
        if any(row["part_id"] == part_id and not row["reconciled"] for row in active):
            raise ValueError("Part has an outstanding reservation")
        used = sum(row["attempts"] for row in active)
        if used + attempts > BUDGET:
            pending = sum(row["attempts"] for row in active if not row["reconciled"])
            if pending + attempts > BUDGET:
                raise ValueError("USGS request budget exhausted; outstanding reservations require reconciliation")
            wait = 0
            completed = [row for row in active if row["reconciled"]]
            for row in sorted(completed, key=lambda row: row["timestamp"]):
                used -= row["attempts"]
                wait = math.ceil(row["timestamp"] + WINDOW_SECONDS - now)
                if used + attempts <= BUDGET:
                    break
            raise ValueError(f"USGS request budget exhausted; retry after {wait} seconds")
        reservation_id = uuid4().hex
        records[reservation_id] = {"part_id": part_id, "attempts": attempts, "timestamp": now, "reconciled": False}
    return reservation_id


def reconcile(workspace: Path, reservation_id: str, actual: int) -> None:
    """Charge observed attempts for a full window after completion."""
    if actual < 0:
        raise ValueError("Actual attempts cannot be negative")
    with _ledger(workspace) as records:
        row = records[reservation_id]
        if row["reconciled"]:
            if row["attempts"] != actual:
                raise ValueError("Reservation already reconciled with different attempts")
            return
        row.update(attempts=actual, timestamp=time.time(), reconciled=True)
