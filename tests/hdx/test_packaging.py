from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from hydrologeez.hdx._polars import require_polars


def test_import_hydrologeez_succeeds() -> None:
    importlib.import_module("hydrologeez")


def test_import_hydrologeez_hdx_succeeds() -> None:
    importlib.import_module("hydrologeez.hdx")


def test_import_hydrologeez_does_not_import_polars() -> None:
    subprocess.run(
        [sys.executable, "-c", "import hydrologeez, sys; assert 'polars' not in sys.modules"],
        check=True,
    )


def test_import_hydrologeez_hdx_does_not_import_polars() -> None:
    subprocess.run(
        [sys.executable, "-c", "import hydrologeez.hdx, sys; assert 'polars' not in sys.modules"],
        check=True,
    )


def test_require_polars_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "polars", None)
    with pytest.raises(ImportError, match=r"hydrologeez\[hdx\]"):
        require_polars()
