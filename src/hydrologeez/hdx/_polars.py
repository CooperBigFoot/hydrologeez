from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_INSTALL_HINT = "hydrologeez HDX I/O requires polars. Install it with: pip install 'hydrologeez[hdx]'"


def require_polars() -> ModuleType:
    """Import and return the polars module, or raise a clear install error.

    Called at function entry by the loader/writer so that importing
    ``hydrologeez`` / ``hydrologeez.hdx`` never forces polars to be installed.
    """
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(_INSTALL_HINT) from exc
    return pl


def require_pyarrow() -> ModuleType:
    try:
        import pyarrow
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch pattern
        raise ImportError(_INSTALL_HINT) from exc
    return pyarrow
