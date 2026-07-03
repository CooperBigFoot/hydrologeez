from __future__ import annotations

from dataclasses import dataclass, field

# Canonical dynamic field names hydrologeez understands.
DEFAULT_FORCING_FIELDS: tuple[str, ...] = ("precip", "pet", "temp")
DEFAULT_TARGET_FIELD: str = "streamflow"


@dataclass(frozen=True)
class Vocabulary:
    """Maps dataset column names -> canonical hydrologeez field names.

    HDX is role-opaque; hydrologeez owns this semantic layer. By default the
    canonical names map to themselves; ``overrides`` remaps foreign dataset
    column names (e.g. ``{"P": "precip", "Q": "streamflow"}``) and takes
    precedence over the defaults.
    """

    overrides: dict[str, str] = field(default_factory=dict)

    def resolve(self, column: str) -> str | None:
        """Return the canonical field name for a dataset column, else None.

        Precedence: an explicit override wins; otherwise a column whose name is
        already a canonical field (a forcing field or the target) maps to
        itself; any other column is unmapped -> None (the loader ignores it).
        """
        if column in self.overrides:
            return self.overrides[column]
        if column in DEFAULT_FORCING_FIELDS or column == DEFAULT_TARGET_FIELD:
            return column
        return None

    def role_of(self, canonical: str) -> str:
        """Classify a CANONICAL field name into "forcing" | "target".

        Raise ValueError for a name that is neither a forcing field nor the
        target (statics are handled separately by the loader, not here).
        """
        if canonical in DEFAULT_FORCING_FIELDS:
            return "forcing"
        if canonical == DEFAULT_TARGET_FIELD:
            return "target"
        raise ValueError(f"Unknown canonical field: {canonical}")
