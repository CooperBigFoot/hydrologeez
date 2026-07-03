"""HDX I/O layer for hydrologeez.

Canonical, format-agnostic ingestion/emission of HDX scalar datasets. The
numerical kernel never depends on a file format; polars is an OPTIONAL extra
imported lazily at call-time. This package __init__ re-exports ONLY the pure
vocabulary API — the from_hdx/to_hdx entry points are lifted to the top-level
``hydrologeez`` package in a later step.
"""

from __future__ import annotations

from hydrologeez.hdx.vocabulary import DEFAULT_FORCING_FIELDS, DEFAULT_TARGET_FIELD, Vocabulary

__all__ = ["DEFAULT_FORCING_FIELDS", "DEFAULT_TARGET_FIELD", "Vocabulary"]
