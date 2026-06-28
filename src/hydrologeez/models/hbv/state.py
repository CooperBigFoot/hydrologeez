"""HBV-Light single-zone state PyTree."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .constants import ROUTING_BUFFER_SIZE


class HBVState(eqx.Module):
    """HBV-Light single-zone model state (an Equinox PyTree).

    Fields (Rust state.rs:11-21 single-zone view)
    ---------------------------------------------
    zone_sp : 0-d array
        Snow pack SP [mm] (zone_states[0][0]).
    zone_lw : 0-d array
        Liquid water in snow LW/WC [mm] (zone_states[0][1]).
    zone_sm : 0-d array
        Soil moisture SM [mm] (zone_states[0][2]).
    upper_zone : 0-d array
        Upper groundwater storage SUZ [mm].
    lower_zone : 0-d array
        Lower groundwater storage SLZ [mm].
    routing_buffer : (7,) array
        Triangular-UH convolution delay line.
    """

    zone_sp: Array
    zone_lw: Array
    zone_sm: Array
    upper_zone: Array
    lower_zone: Array
    routing_buffer: Array

    def to_flat(self) -> Array:
        """Pack into the Rust 12-element layout [SP, LW, SM, SUZ, SLZ, b0..b6]."""
        return jnp.concatenate(
            [
                jnp.atleast_1d(self.zone_sp),
                jnp.atleast_1d(self.zone_lw),
                jnp.atleast_1d(self.zone_sm),
                jnp.atleast_1d(self.upper_zone),
                jnp.atleast_1d(self.lower_zone),
                self.routing_buffer,
            ]
        )

    @classmethod
    def from_flat(cls, arr: Array) -> HBVState:
        """Reconstruct from the Rust 12-element layout."""
        flat = jnp.asarray(arr, dtype=jnp.float64)
        return cls(
            zone_sp=flat[0],
            zone_lw=flat[1],
            zone_sm=flat[2],
            upper_zone=flat[3],
            lower_zone=flat[4],
            routing_buffer=flat[5 : 5 + ROUTING_BUFFER_SIZE],
        )
