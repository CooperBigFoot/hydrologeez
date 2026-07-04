"""GR6J process free functions.

Pure, differentiable JAX implementations. Branches use jnp.where for
diff-cleanliness; all functions are jit-able with finite gradients.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from hydrologeez.convolution import convolve_delay_line

from .constants import EXP_BRANCH_THRESHOLD, MAX_EXP_ARG, MAX_TANH_ARG, NH, PERC_CONSTANT, D


def production_store_update(
    precip: Array, pet: Array, production_store: Array, x1: Array
) -> tuple[Array, Array, Array, Array]:
    """Update the production store.

    Returns (new_store, actual_et, net_rainfall_pn, effective_rainfall_pr).
    Mirrors processes.rs:15-63.
    """
    store_ratio = production_store / x1

    net_evap = pet - precip
    scaled_evap = jnp.minimum(net_evap / x1, MAX_TANH_ARG)
    t_evap = jnp.tanh(scaled_evap)
    evap_from_store = production_store * ((2.0 - store_ratio) * t_evap) / (1.0 + (1.0 - store_ratio) * t_evap)
    store_evap = production_store - evap_from_store
    et_evap = evap_from_store + precip

    net_rainfall = precip - pet
    scaled_precip = jnp.minimum(net_rainfall / x1, MAX_TANH_ARG)
    t_precip = jnp.tanh(scaled_precip)
    storage_infiltration = x1 * ((1.0 - store_ratio * store_ratio) * t_precip) / (1.0 + store_ratio * t_precip)
    effective_rainfall = net_rainfall - storage_infiltration
    store_rain = production_store + storage_infiltration

    evap_dominant = precip < pet
    new_store = jnp.where(evap_dominant, store_evap, store_rain)
    actual_et = jnp.where(evap_dominant, et_evap, pet)
    net_rainfall_pn = jnp.where(evap_dominant, 0.0, net_rainfall)
    effective_rainfall_pr = jnp.where(evap_dominant, 0.0, effective_rainfall)
    return new_store, actual_et, net_rainfall_pn, effective_rainfall_pr


def percolation(production_store: Array, x1: Array) -> tuple[Array, Array]:
    """Percolation from the production store. Mirrors processes.rs:71-81."""
    store = jnp.maximum(production_store, 0.0)
    ratio4 = (store / x1) ** 4
    perc = store * (1.0 - (1.0 + ratio4 / PERC_CONSTANT) ** (-0.25))
    return store - perc, perc


def groundwater_exchange(routing_store: Array, x2: Array, x3: Array, x5: Array) -> Array:
    """F = x2 * (R / x3 - x5). Mirrors processes.rs:87-89."""
    return x2 * (routing_store / x3 - x5)


def routing_store_update(
    routing_store: Array, uh1_output: Array, exchange: Array, x3: Array
) -> tuple[Array, Array, Array]:
    """Update routing store. Returns (new_store, qr, actual_exchange).

    Inflow is (1 - C) * q9 = 0.6 * q9 (supplied by caller).
    Mirrors processes.rs:98-123.
    """
    tmp = routing_store + uh1_output + exchange
    positive = tmp >= 0.0
    actual_exchange = jnp.where(positive, exchange, -(routing_store + uh1_output))
    store = jnp.where(positive, tmp, 0.0)
    ratio4 = (store / x3) ** 4
    qr = jnp.where(store > 0.0, store * (1.0 - (1.0 + ratio4) ** (-0.25)), 0.0)
    return store - qr, qr, actual_exchange


def exponential_store_update(exp_store: Array, uh1_output: Array, exchange: Array, x6: Array) -> tuple[Array, Array]:
    """Update exponential store. Returns (new_store, qrexp).

    Inflow is C * q9 = 0.4 * q9 (supplied by caller).
    3-branch softplus with AR clamped to [-33, 33]. Mirrors processes.rs:132-157.
    """
    store = exp_store + uh1_output + exchange
    ar = jnp.clip(store / x6, -MAX_EXP_ARG, MAX_EXP_ARG)
    large_pos = store + x6 / jnp.exp(ar)
    large_neg = x6 * jnp.exp(ar)
    normal = x6 * jnp.log1p(jnp.exp(ar))
    qrexp = jnp.where(
        ar > EXP_BRANCH_THRESHOLD,
        large_pos,
        jnp.where(ar < -EXP_BRANCH_THRESHOLD, large_neg, normal),
    )
    return store - qrexp, qrexp


def direct_branch(uh2_output: Array, exchange: Array) -> tuple[Array, Array]:
    """Direct branch. Returns (qd, actual_exchange). Mirrors processes.rs:165-173."""
    combined = uh2_output + exchange
    positive = combined >= 0.0
    qd = jnp.where(positive, combined, 0.0)
    actual_exchange = jnp.where(positive, exchange, -uh2_output)
    return qd, actual_exchange


def _ss1(i: Array, x4: Array) -> Array:
    """UH1 S-curve. Mirrors unit_hydrographs.rs:10-18."""
    below = (i / x4) ** D
    return jnp.where(i <= 0.0, 0.0, jnp.where(i < x4, below, 1.0))


def _ss2(i: Array, x4: Array) -> Array:
    """UH2 S-curve. Mirrors unit_hydrographs.rs:21-31."""
    ratio = i / x4
    half = 0.5 * ratio**D
    upper = 1.0 - 0.5 * jnp.maximum(2.0 - ratio, 0.0) ** D
    return jnp.where(
        i <= 0.0,
        0.0,
        jnp.where(i <= x4, half, jnp.where(i < 2.0 * x4, upper, 1.0)),
    )


def compute_uh_ordinates(x4: Array) -> tuple[Array, Array]:
    """Masked-kernel UH ordinates: uh1 length 20, uh2 length 40."""
    i1 = jnp.arange(1, NH + 1, dtype=jnp.float64)
    uh1 = _ss1(i1, x4) - _ss1(i1 - 1.0, x4)
    i2 = jnp.arange(1, 2 * NH + 1, dtype=jnp.float64)
    uh2 = _ss2(i2, x4) - _ss2(i2 - 1.0, x4)
    return uh1, uh2


def convolve_uh(states: Array, ordinates: Array, input_value: Array) -> tuple[Array, Array]:
    """One-step delay-line convolution (read output AFTER the shift).

    Returns (output, new_states). Delegates to the shared
    :func:`hydrologeez.convolution.convolve_delay_line`; matches airGR MOD_GR6J
    (shift StUH first, then read StUH1(1)/StUH2(1) — same-day ordinate-1 term,
    no forced one-step lag).
    """
    return convolve_delay_line(states, ordinates, input_value)
