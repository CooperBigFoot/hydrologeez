from __future__ import annotations

import pytest

from hydrologeez.hdx.vocabulary import Vocabulary


def test_default_vocabulary_resolves_canonical_fields() -> None:
    vocabulary = Vocabulary()

    assert vocabulary.resolve("precip") == "precip"
    assert vocabulary.resolve("pet") == "pet"
    assert vocabulary.resolve("temp") == "temp"
    assert vocabulary.resolve("streamflow") == "streamflow"
    assert vocabulary.resolve("unknown_col") is None


def test_vocabulary_roles() -> None:
    vocabulary = Vocabulary()

    assert vocabulary.role_of("precip") == "forcing"
    assert vocabulary.role_of("pet") == "forcing"
    assert vocabulary.role_of("temp") == "forcing"
    assert vocabulary.role_of("streamflow") == "target"
    with pytest.raises(ValueError):
        vocabulary.role_of("nope")


def test_vocabulary_overrides_merge_over_defaults() -> None:
    vocabulary = Vocabulary(overrides={"P": "precip", "Q": "streamflow"})

    assert vocabulary.resolve("P") == "precip"
    assert vocabulary.resolve("Q") == "streamflow"
    assert vocabulary.resolve("pet") == "pet"
