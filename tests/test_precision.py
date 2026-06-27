"""Tests for the import-time float64 enforcement and the conftest x64 seam."""

import os
import subprocess
import sys
from pathlib import Path

import jax

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _import_hydrologeez_subprocess(x64_value: str | None) -> subprocess.CompletedProcess:
    """Import hydrologeez in a FRESH python process with a controlled env.

    ``x64_value=None`` => JAX_ENABLE_X64 is unset; otherwise it is set to the
    given string. The parent's JAX_ENABLE_X64 (set by conftest) is stripped so
    this never leaks into the child.
    """
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    if x64_value is not None:
        env["JAX_ENABLE_X64"] = x64_value
    return subprocess.run(
        [sys.executable, "-c", "import hydrologeez"],
        capture_output=True,
        text=True,
        env=env,
    )


def test_import_raises_when_x64_disabled():
    """With x64 OFF, importing hydrologeez RAISES (proving no silent flip)."""
    result = _import_hydrologeez_subprocess(None)
    assert result.returncode != 0
    assert "JAX_ENABLE_X64=1" in result.stderr


def test_import_succeeds_when_x64_enabled():
    """A second fresh process with JAX_ENABLE_X64=1 imports successfully."""
    result = _import_hydrologeez_subprocess("1")
    assert result.returncode == 0, result.stderr


def test_inprocess_x64_enabled_by_conftest():
    """The in-process suite has x64 ON via the conftest seam alone.

    No test module called jax.config.update; the env-var seam did it.
    """
    assert jax.config.jax_enable_x64 is True  # ty: ignore[unresolved-attribute]
