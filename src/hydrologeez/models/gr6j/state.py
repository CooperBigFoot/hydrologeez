"""GR6J state PyTree."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .constants import UH1_LEN, UH2_LEN


class State(eqx.Module):
    """GR6J model state (an Equinox PyTree).

    Fields
    ------
    production_store : 0-d array
        Soil moisture store S [mm].
    routing_store : 0-d array
        Routing store R [mm].
    exponential_store : 0-d array
        Exponential store Exp [mm] (may be negative).
    uh1 : (20,) array
        UH1 delay-line buffer.
    uh2 : (40,) array
        UH2 delay-line buffer.
    """

    production_store: Array
    routing_store: Array
    exponential_store: Array
    uh1: Array
    uh2: Array

    def to_flat(self) -> Array:
        """Pack into the Rust 63-element layout [S, R, Exp, uh1, uh2]."""
        return jnp.concatenate(
            [
                jnp.atleast_1d(self.production_store),
                jnp.atleast_1d(self.routing_store),
                jnp.atleast_1d(self.exponential_store),
                self.uh1,
                self.uh2,
            ]
        )

    @classmethod
    def from_flat(cls, arr: Array) -> State:
        """Reconstruct from the Rust 63-element layout."""
        flat = jnp.asarray(arr, dtype=jnp.float64)
        return cls(
            production_store=flat[0],
            routing_store=flat[1],
            exponential_store=flat[2],
            uh1=flat[3 : 3 + UH1_LEN],
            uh2=flat[3 + UH1_LEN : 3 + UH1_LEN + UH2_LEN],
        )
