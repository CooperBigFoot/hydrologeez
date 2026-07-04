"""HBV-Light process free functions.

Pure, differentiable JAX implementations. All branches use jnp.where for
diff-cleanliness; all functions are jit-able with finite gradients.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from hydrologeez.convolution import convolve_delay_line

from .constants import ROUTING_BUFFER_SIZE


def partition_precipitation(precip: Array, temp: Array, tt: Array, sfcf: Array) -> tuple[Array, Array]:
    """Partition precip into (p_rain, p_snow). Mirrors processes.rs:10-16.

    Strict temp > tt => all rain. temp <= tt (incl. ==) => all snow * sfcf.
    """
    is_rain = temp > tt
    p_rain = jnp.where(is_rain, precip, 0.0)
    p_snow = jnp.where(is_rain, 0.0, sfcf * precip)
    return p_rain, p_snow


def compute_melt(temp: Array, tt: Array, cfmax: Array, snow_pack: Array) -> Array:
    """Degree-day melt = min(cfmax*(temp-tt), snow_pack) when temp>tt else 0.

    Mirrors processes.rs:22-29. Clamp on start-of-step snow_pack.
    """
    melt = jnp.minimum(cfmax * (temp - tt), snow_pack)
    return jnp.where(temp > tt, melt, 0.0)


def compute_refreezing(temp: Array, tt: Array, cfmax: Array, cfr: Array, liquid_water: Array) -> Array:
    """Refreeze = min(cfr*cfmax*(tt-temp), liquid_water) when temp<tt else 0.

    Mirrors processes.rs:35-42. Clamp on start-of-step liquid_water.
    """
    refreeze = jnp.minimum(cfr * cfmax * (tt - temp), liquid_water)
    return jnp.where(temp < tt, refreeze, 0.0)


def update_snow_pack(
    sp: Array, lw: Array, p_snow: Array, melt: Array, refreeze: Array, cwh: Array
) -> tuple[Array, Array, Array]:
    """Update snow pack + liquid water; return (new_sp, new_lw, outflow).

    Mirrors processes.rs:48-73. lw_max uses the PRE-clamp new_sp; outflow is the
    excess above lw_max (and new_lw is set to lw_max when exceeded); SP and LW are
    floored at 0 AFTER lw_max is computed.
    """
    new_sp = sp + p_snow - melt + refreeze
    new_lw_pre = lw + melt - refreeze
    lw_max = cwh * new_sp
    exceeds = new_lw_pre > lw_max
    outflow = jnp.where(exceeds, new_lw_pre - lw_max, 0.0)
    new_lw = jnp.where(exceeds, lw_max, new_lw_pre)
    new_sp = jnp.maximum(new_sp, 0.0)
    new_lw = jnp.maximum(new_lw, 0.0)
    return new_sp, new_lw, outflow


def compute_recharge(soil_input: Array, sm: Array, fc: Array, beta: Array) -> Array:
    """recharge = soil_input * clamp(sm/fc, 0, 1)**beta; 0 if fc<=0 or soil_input<=0.

    Mirrors processes.rs:79-85. Uses start-of-step sm. Diff-clean double-where so
    grad w.r.t. beta is finite when the saturation ratio is 0 (avoids 0**beta in
    the differentiated path; the masked value is exactly 0).
    """
    safe_fc = jnp.where(fc > 0.0, fc, 1.0)
    sr = jnp.clip(sm / safe_fc, 0.0, 1.0)
    sr_safe = jnp.where(sr > 0.0, sr, 1.0)
    pow_term = jnp.where(sr > 0.0, sr_safe**beta, 0.0)
    recharge = soil_input * pow_term
    guard = (fc <= 0.0) | (soil_input <= 0.0)
    return jnp.where(guard, 0.0, recharge)


def compute_actual_et(pet: Array, sm: Array, fc: Array, lp: Array) -> Array:
    """ET = pet if sm>=lp*fc else pet*sm/(lp*fc); capped at max(sm,0); 0 if fc<=0 or lp<=0.

    Mirrors processes.rs:91-105. Uses start-of-step sm (same as recharge).
    """
    lp_threshold = lp * fc
    safe_thr = jnp.where(lp_threshold > 0.0, lp_threshold, 1.0)
    reduced = pet * sm / safe_thr
    et_act = jnp.where(sm >= lp_threshold, pet, reduced)
    et_act = jnp.minimum(et_act, jnp.maximum(sm, 0.0))
    guard = (fc <= 0.0) | (lp <= 0.0)
    return jnp.where(guard, 0.0, et_act)


def update_soil_moisture(
    sm: Array, soil_input: Array, recharge: Array, et_act: Array, fc: Array
) -> tuple[Array, Array]:
    """Update soil moisture; return (new_sm, overflow).

    ``new_sm_raw = sm + (soil_input - recharge) - et_act``. The above-field-capacity
    excess ``overflow = max(new_sm_raw - fc, 0)`` is RETURNED so the caller routes
    it to upper-zone recharge (mass-conserving; Seibert & Vis 2012) instead of
    discarding it. ``new_sm`` is clamped to ``[0, fc]``; the 0-floor over-draw
    behaviour (standard forward-Euler HBV) is unchanged.
    """
    new_sm_raw = sm + (soil_input - recharge) - et_act
    overflow = jnp.maximum(new_sm_raw - fc, 0.0)
    new_sm = jnp.clip(new_sm_raw, 0.0, fc)
    return new_sm, overflow


def upper_zone_outflows(suz: Array, k0: Array, k1: Array, uzl: Array) -> tuple[Array, Array]:
    """q0 = k0*max(suz-uzl, 0) (threshold suz>uzl); q1 = k1*suz. routing.rs:13-17."""
    q0 = jnp.where(suz > uzl, k0 * (suz - uzl), 0.0)
    q1 = k1 * suz
    return q0, q1


def compute_percolation(suz: Array, perc_max: Array) -> Array:
    """perc = min(perc_max, max(suz, 0)). routing.rs:23-25. Reads OLD suz."""
    return jnp.minimum(perc_max, jnp.maximum(suz, 0.0))


def update_upper_zone(suz: Array, recharge: Array, q0: Array, q1: Array, perc: Array) -> Array:
    """new_suz = max(suz + recharge - q0 - q1 - perc, 0). routing.rs:29-32."""
    return jnp.maximum(suz + recharge - q0 - q1 - perc, 0.0)


def lower_zone_outflow(slz: Array, k2: Array) -> Array:
    """q2 = k2 * slz (reads OLD slz). routing.rs:36-38."""
    return k2 * slz


def update_lower_zone(slz: Array, perc: Array, q2: Array) -> Array:
    """new_slz = max(slz + perc - q2, 0). routing.rs:42-45."""
    return jnp.maximum(slz + perc - q2, 0.0)


def compute_triangular_weights(maxbas: Array) -> Array:
    """Differentiable fixed-length-7 normalized triangular UH weights.

    Mirrors routing.rs:52-92 but as a SMOOTH function of maxbas over a fixed
    length-7 kernel. The underlying density is a triangle peaking at half=maxbas/2:
    rising 2t/maxbas^2 on [0, half], falling 2(maxbas-t)/maxbas^2 on [half, maxbas].
    Per-bin integral over [i, min(i+1, maxbas)]; bins with i >= maxbas are inactive
    (t_end <= t_start) -- that is the mask. Raw weights integrate to 0.5, so the
    explicit normalize-by-sum (LOAD-BEARING) doubles them to sum 1.0.

    Differentiability: the two limb ``if`` guards are replaced by min/max clamps so
    each limb contributes 0 outside its support without a branch; the active mask is
    a jnp.where. Ordinate VALUES are continuous across integer maxbas (a newly
    activated bin starts at 0); the GRADIENT has a kink at integer maxbas because
    n=ceil(maxbas) activates a new bin there -- this is finite-but-kinked, NOT
    gradient-continuous (analogous to GR6J integer-x4).
    """
    i = jnp.arange(ROUTING_BUFFER_SIZE, dtype=jnp.float64)
    t_start = i
    t_end = jnp.minimum(i + 1.0, maxbas)
    half = maxbas / 2.0
    maxbas_sq = maxbas * maxbas

    r_lo = jnp.minimum(t_start, half)
    r_hi = jnp.minimum(t_end, half)
    rising = (r_hi * r_hi - r_lo * r_lo) / maxbas_sq

    f_lo = jnp.maximum(t_start, half)
    f_hi = jnp.maximum(t_end, half)
    falling = 2.0 * (f_hi - f_lo) / maxbas - (f_hi * f_hi - f_lo * f_lo) / maxbas_sq

    w_raw = rising + falling
    w = jnp.where(t_end > t_start, w_raw, 0.0)
    total = jnp.sum(w)
    return jnp.where(total > 0.0, w / total, w)


def convolve_routing(buffer: Array, weights: Array, qgw: Array) -> tuple[Array, Array]:
    """Read-after-shift length-7 delay line; return (qsim, new_buffer).

    Delegates to the shared :func:`hydrologeez.convolution.convolve_delay_line`:
    shift the buffer one slot, inject ``weights * qgw``, then read the new head (the
    same-day ordinate-1 term; no forced one-step lag), matching the published
    HBV-Light routing (Seibert & Vis 2012; Seibert 2005 manual Eq. 6). Shares the
    same helper as GR6J's unit-hydrograph convolution.
    """
    return convolve_delay_line(buffer, weights, qgw)
