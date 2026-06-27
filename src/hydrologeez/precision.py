"""Import-time float64 (x64) enforcement for hydrologeez.

hydrologeez models accumulate long store recurrences and compute
metric-stability-sensitive reductions; they REQUIRE JAX 64-bit precision.

This module performs a loud, fail-fast check. It NEVER silently flips
``jax.config`` (that would mutate global state and disrespect other libraries
in the user's process). If x64 is off, it raises with a one-line fix.
"""

from __future__ import annotations

import jax

ENABLE_FIX_MESSAGE = (
    "hydrologeez requires JAX 64-bit precision, but it is disabled. "
    "Enable it BEFORE importing jax or hydrologeez, e.g. set the environment "
    "variable JAX_ENABLE_X64=1, or call "
    "jax.config.update('jax_enable_x64', True) before the first import. "
    "hydrologeez does NOT flip this for you (no silent global x64 mutation)."
)


def enforce_float64() -> None:
    """Raise ``RuntimeError`` if JAX x64 is not enabled.

    Performs NO silent ``jax.config`` flip: it raises instead of
    flipping-and-proceeding.
    """
    if not jax.config.jax_enable_x64:  # ty: ignore[unresolved-attribute]
        raise RuntimeError(ENABLE_FIX_MESSAGE)
