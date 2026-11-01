"""Locked, content-verified deliveries of native USGS cache and publisher bundles."""

from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from scripts.daymet_archive import Archive, checksum


@contextmanager
def workspace_lock(workspace: Path) -> Generator[None]:
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".acquisition.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("USGS workspace is busy") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class USGSArchive(Archive):
    prefix = "streamflow/usgs"

    @staticmethod
    def _generation(generation: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", generation):
            raise ValueError("Invalid generation")

    def _key(self, generation: str, relative: str) -> str:
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] not in {"cache", "receipts", "manifests"}
        ):
            raise ValueError("Invalid archive path")
        return f"{self.prefix}/{path.parts[0]}/{generation}/{'/'.join(path.parts[1:])}"

    def publish(
        self, workspace: Path, generation: str, *, validate: Callable[[Path], object] | None = None
    ) -> dict[str, Any]:
        """Freeze writers for upload; the marker certifies delivery, not full acquisition."""
        self._generation(generation)
        with workspace_lock(workspace):
            if self.digest(self._key(generation, "manifests/complete.json")) is not None:
                raise ValueError("Generation is already complete; use a new generation")
            if validate is not None:
                validate(workspace)
            files: list[dict[str, Any]] = []
            for directory in ("cache", "receipts", "manifests"):
                base = workspace / directory
                if base.is_symlink():
                    raise ValueError(f"Symlink in workspace: {base}")
                if not base.is_dir():
                    raise ValueError(f"Missing workspace directory: {directory}")
                for path in sorted(base.rglob("*")):
                    if path.is_symlink():
                        raise ValueError(f"Symlink in workspace: {path}")
                    if not path.is_file():
                        continue
                    relative = path.relative_to(workspace).as_posix()
                    if relative == "manifests/complete.json":
                        raise ValueError("Reserved completion manifest path")
                    receipt = self.put(path, self._key(generation, relative))
                    files.append({"path": relative, **receipt})
            manifest = {"generation": generation, "files": files}
            with TemporaryDirectory() as temporary:
                marker = Path(temporary) / "complete.json"
                marker.write_text(json.dumps(manifest, sort_keys=True) + "\n")
                self.put(marker, self._key(generation, "manifests/complete.json"))
            return manifest

    def restore(
        self, generation: str, workspace: Path, *, validate: Callable[[Path], object] | None = None
    ) -> dict[str, Any]:
        """Restore only a completed delivery into a new or empty workspace."""
        self._generation(generation)
        if workspace.exists() and any(workspace.iterdir()):
            raise ValueError("Restore requires an empty workspace")
        with workspace_lock(workspace):
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(generation, "manifests/complete.json"))
            try:
                manifest = json.loads(response["Body"].read())
            finally:
                response["Body"].close()
            if manifest["generation"] != generation:
                raise ValueError("Generation mismatch")
            seen: set[str] = set()
            for entry in manifest["files"]:
                relative = entry["path"]
                key = self._key(generation, relative)
                if key != entry["key"] or relative in seen:
                    raise ValueError("Invalid archive inventory")
                seen.add(relative)
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, key, str(path))
                if path.stat().st_size != entry["bytes"] or checksum(path) != entry["sha256"]:
                    raise ValueError(f"Object integrity mismatch: {key}")
            for directory in ("cache", "receipts", "manifests"):
                (workspace / directory).mkdir(exist_ok=True)
            if validate is not None:
                validate(workspace)
            return manifest
