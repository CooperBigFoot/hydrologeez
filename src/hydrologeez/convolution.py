"""Shared delay-line convolution for unit-hydrograph / routing kernels.

One delay-line step shared by GR6J's two unit hydrographs (and, later, HBV's
MAXBAS routing). Pure JAX, shape-agnostic, ``lax.scan``/``vmap``/``grad``-safe.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def convolve_delay_line(buffer: Array, kernel: Array, inflow: Array) -> tuple[Array, Array]:
    """Advance a delay-line convolution one step (read the head AFTER the shift).

    Shifts the buffer one slot toward the outlet, injects ``kernel * inflow``,
    then reads the new head. This is the same-day ordinate-1 term (no forced
    one-step lag), matching airGR ``MOD_GR6J`` (shift ``StUH`` first, then read
    ``StUH1(1)``/``StUH2(1)``).

    Parameters
    ----------
    buffer : Array
        Current delay-line state (length ``n``).
    kernel : Array
        Unit-hydrograph / routing ordinates (length ``n``; sums to ~1).
    inflow : Array
        Scalar inflow entering the delay line this step.

    Returns
    -------
    tuple[Array, Array]
        ``(output, new_buffer)`` where ``output = new_buffer[0] = buffer[1] +
        kernel[0] * inflow`` and ``new_buffer`` is the advanced state.
    """
    shifted = jnp.concatenate([buffer[1:], jnp.zeros((1,), dtype=buffer.dtype)])
    new_buffer = shifted + kernel * inflow
    output = new_buffer[0]
    return output, new_buffer
