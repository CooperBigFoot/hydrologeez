"""Per-process oracle parity + MAXBAS kernel tests for single-zone HBV-Light."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.testing import assert_allclose

from hydrologeez.models.hbv import (
    compute_actual_et,
    compute_melt,
    compute_percolation,
    compute_recharge,
    compute_refreezing,
    compute_triangular_weights,
    convolve_routing,
    lower_zone_outflow,
    partition_precipitation,
    update_lower_zone,
    update_snow_pack,
    update_soil_moisture,
    update_upper_zone,
    upper_zone_outflows,
)
from hydrologeez.models.hbv.constants import ROUTING_BUFFER_SIZE
from hydrologeez.models.hbv.state import HBVState

FIXTURES = Path(__file__).parents[2] / "fixtures"
RTOL = 1e-4
ATOL = 1e-6

RUN_FIXTURES = ("hbv_camels_06224000.npz", "hbv_camels_06224000_maxbas25.npz")
KERNEL_FIXTURE = "hbv_triangular_weights.npz"

FLUX_KEYS = (
    "precip",
    "temp",
    "pet",
    "precip_rain",
    "precip_snow",
    "snow_pack",
    "snow_melt",
    "liquid_water_in_snow",
    "snow_input",
    "soil_moisture",
    "recharge",
    "actual_et",
    "upper_zone",
    "lower_zone",
    "q0",
    "q1",
    "q2",
    "percolation",
    "qgw",
    "streamflow",
)


def reference_step(
    state: HBVState, params: Array, precip: Array, pet: Array, temp: Array, uh_weights: Array
) -> tuple[HBVState, dict[str, Array]]:
    """Wire the free functions in run.rs single-zone order (run.rs:191-228)."""
    tt, cfmax, sfcf, cwh, cfr, fc, lp, beta, k0, k1, k2, perc_max, uzl, _maxbas = (params[i] for i in range(14))

    sp, lw, sm = state.zone_sp, state.zone_lw, state.zone_sm
    p_rain, p_snow = partition_precipitation(precip, temp, tt, sfcf)
    melt = compute_melt(temp, tt, cfmax, sp)
    refreeze = compute_refreezing(temp, tt, cfmax, cfr, lw)
    new_sp, new_lw, snow_outflow = update_snow_pack(sp, lw, p_snow, melt, refreeze, cwh)
    snow_input = p_rain + snow_outflow

    recharge = compute_recharge(snow_input, sm, fc, beta)
    et_act = compute_actual_et(pet, sm, fc, lp)
    new_sm = update_soil_moisture(sm, snow_input, recharge, et_act, fc)

    suz = state.upper_zone
    slz = state.lower_zone
    q0, q1 = upper_zone_outflows(suz, k0, k1, uzl)
    perc = compute_percolation(suz, perc_max)
    new_suz = update_upper_zone(suz, recharge, q0, q1, perc)
    q2 = lower_zone_outflow(slz, k2)
    new_slz = update_lower_zone(slz, perc, q2)
    qgw = q0 + q1 + q2

    qsim, new_buffer = convolve_routing(state.routing_buffer, uh_weights, qgw)

    new_state = HBVState(
        zone_sp=new_sp,
        zone_lw=new_lw,
        zone_sm=new_sm,
        upper_zone=new_suz,
        lower_zone=new_slz,
        routing_buffer=new_buffer,
    )
    fluxes = {
        "precip": precip,
        "temp": temp,
        "pet": pet,
        "precip_rain": p_rain,
        "precip_snow": p_snow,
        "snow_pack": new_sp,
        "snow_melt": melt,
        "liquid_water_in_snow": new_lw,
        "snow_input": snow_input,
        "soil_moisture": new_sm,
        "recharge": recharge,
        "actual_et": et_act,
        "upper_zone": new_suz,
        "lower_zone": new_slz,
        "q0": q0,
        "q1": q1,
        "q2": q2,
        "percolation": perc,
        "qgw": qgw,
        "streamflow": qsim,
    }
    return new_state, fluxes


def run_series(params: Array, precip: Array, pet: Array, temp: Array) -> dict[str, Array]:
    maxbas = params[13]
    uh_weights = compute_triangular_weights(maxbas)
    init = HBVState(
        zone_sp=jnp.asarray(0.0, dtype=jnp.float64),
        zone_lw=jnp.asarray(0.0, dtype=jnp.float64),
        zone_sm=0.5 * params[5],
        upper_zone=jnp.asarray(0.0, dtype=jnp.float64),
        lower_zone=jnp.asarray(0.0, dtype=jnp.float64),
        routing_buffer=jnp.zeros(ROUTING_BUFFER_SIZE, dtype=jnp.float64),
    )

    def body(state: HBVState, forcing: tuple[Array, Array, Array]) -> tuple[HBVState, dict[str, Array]]:
        p, e, t = forcing
        return reference_step(state, params, p, e, t, uh_weights)

    _, fluxes = jax.lax.scan(body, init, (precip, pet, temp))
    return fluxes


def _load_run(name: str):
    data = np.load(FIXTURES / name, allow_pickle=False)
    params = jnp.asarray(data["params"], dtype=jnp.float64)
    precip = jnp.asarray(data["precip"], dtype=jnp.float64)
    pet = jnp.asarray(data["pet"], dtype=jnp.float64)
    temp = jnp.asarray(data["temp"], dtype=jnp.float64)
    return data, params, precip, pet, temp


def test_per_process_parity_both_fixtures() -> None:
    for name in RUN_FIXTURES:
        data, params, precip, pet, temp = _load_run(name)
        fluxes = run_series(params, precip, pet, temp)
        for key in FLUX_KEYS:
            assert_allclose(np.asarray(fluxes[key]), data[key], rtol=RTOL, atol=ATOL, err_msg=f"flux {key} ({name})")


def test_snow_routine_non_vacuous() -> None:
    data, params, precip, pet, temp = _load_run("hbv_camels_06224000_maxbas25.npz")
    fluxes = run_series(params, precip, pet, temp)
    assert np.count_nonzero(np.asarray(fluxes["precip_snow"])) > 0
    assert np.count_nonzero(np.asarray(fluxes["snow_melt"])) > 0


def test_maxbas_kernel_table_parity() -> None:
    data = np.load(FIXTURES / KERNEL_FIXTURE, allow_pickle=False)
    grid = data["maxbas_grid"]
    weights_ref = data["weights"]
    for idx, maxbas in enumerate(grid):
        w = compute_triangular_weights(jnp.asarray(maxbas, dtype=jnp.float64))
        assert w.shape == (ROUTING_BUFFER_SIZE,)
        assert_allclose(np.asarray(w), weights_ref[idx], rtol=RTOL, atol=ATOL)
        assert_allclose(np.asarray(jnp.sum(w)), 1.0, atol=1e-6)


def test_maxbas_kernel_gradient_finite_and_value_continuity() -> None:
    data = np.load(FIXTURES / KERNEL_FIXTURE, allow_pickle=False)

    def g(maxbas: Array) -> Array:
        w = compute_triangular_weights(maxbas)
        return jnp.sum(jnp.arange(ROUTING_BUFFER_SIZE, dtype=jnp.float64) * w)

    for maxbas in [3.0 - 1e-3, 3.0, 3.0 + 1e-3, *data["maxbas_grid"]]:
        grad = jax.grad(g)(jnp.asarray(maxbas, dtype=jnp.float64))
        assert np.isfinite(np.asarray(grad))

    left = compute_triangular_weights(jnp.asarray(3.0 - 1e-3, dtype=jnp.float64))
    right = compute_triangular_weights(jnp.asarray(3.0 + 1e-3, dtype=jnp.float64))
    assert_allclose(np.asarray(left), np.asarray(right), atol=1e-2)


def test_routing_one_step_lag() -> None:
    weights = compute_triangular_weights(jnp.asarray(2.0, dtype=jnp.float64))
    buffer = jnp.zeros(ROUTING_BUFFER_SIZE, dtype=jnp.float64)
    qsim, _new = convolve_routing(buffer, weights, jnp.asarray(50.0, dtype=jnp.float64))
    assert_allclose(np.asarray(qsim), 0.0, atol=0.0)
    data, params, precip, pet, temp = _load_run("hbv_camels_06224000.npz")
    fluxes = run_series(params, precip, pet, temp)
    assert_allclose(np.asarray(fluxes["streamflow"])[0], 0.0, atol=0.0)


def test_jit_and_finite_gradients() -> None:
    data, params, precip, pet, temp = _load_run("hbv_camels_06224000.npz")
    jitted = jax.jit(run_series)(params, precip, pet, temp)
    assert np.asarray(jitted["streamflow"]).shape == data["streamflow"].shape

    def loss(p: Array) -> Array:
        return jnp.sum(run_series(p, precip, pet, temp)["streamflow"])

    grads = jax.grad(loss)(params)
    assert np.all(np.isfinite(np.asarray(grads)))
