from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.testing import assert_allclose

from hydrologeez.models.gr6j import (
    compute_uh_ordinates,
    convolve_uh,
    direct_branch,
    exponential_store_update,
    groundwater_exchange,
    percolation,
    production_store_update,
    routing_store_update,
)
from hydrologeez.models.gr6j.constants import PARAM_NAMES, UH1_LEN, UH2_LEN, B, C
from hydrologeez.models.gr6j.state import State

RTOL = 1e-4
ATOL = 1e-8
FIXTURES = Path(__file__).parents[2] / "fixtures"
FLUX_KEYS = (
    "pet",
    "precip",
    "production_store",
    "net_rainfall",
    "storage_infiltration",
    "actual_et",
    "percolation",
    "effective_rainfall",
    "q9",
    "q1",
    "routing_store",
    "exchange",
    "actual_exchange_routing",
    "actual_exchange_direct",
    "actual_exchange_total",
    "qr",
    "qrexp",
    "exponential_store",
    "qd",
    "streamflow",
)


def as_param_vector(raw: np.ndarray) -> Array:
    """Coerce fixture params into length-6 float64 vector in PARAM_NAMES order."""
    arr = np.asarray(raw)
    if arr.dtype.names is not None:
        arr = np.array([np.asarray(arr[name], dtype=float).reshape(()) for name in PARAM_NAMES])
    else:
        arr = np.asarray(arr, dtype=float).reshape(-1)
    assert arr.shape == (6,), f"expected length-6 params in x1..x6 order, got shape {arr.shape}"
    return jnp.asarray(arr, dtype=jnp.float64)


def reference_step(
    state: State,
    params: Array,
    precip: Array,
    pet: Array,
    uh1_ord: Array,
    uh2_ord: Array,
) -> tuple[State, dict[str, Array]]:
    x1, x2, x3, _x4, x5, x6 = (params[i] for i in range(6))
    new_s1, actual_et, pn, pr = production_store_update(precip, pet, state.production_store, x1)
    storage_infiltration = jnp.where(precip >= pet, pn - pr, 0.0)
    new_s, perc = percolation(new_s1, x1)
    total_eff = pr + perc
    q9, new_uh1 = convolve_uh(state.uh1, uh1_ord, B * total_eff)
    q1, new_uh2 = convolve_uh(state.uh2, uh2_ord, (1.0 - B) * total_eff)
    f = groundwater_exchange(state.routing_store, x2, x3, x5)
    new_r, qr, aex_routing = routing_store_update(state.routing_store, (1.0 - C) * q9, f, x3)
    new_exp, qrexp = exponential_store_update(state.exponential_store, C * q9, f, x6)
    qd, aex_direct = direct_branch(q1, f)
    streamflow = jnp.maximum(qr + qrexp + qd, 0.0)
    new_state = State(
        production_store=new_s,
        routing_store=new_r,
        exponential_store=new_exp,
        uh1=new_uh1,
        uh2=new_uh2,
    )
    fluxes = {
        "pet": pet,
        "precip": precip,
        "production_store": new_s,
        "net_rainfall": pn,
        "storage_infiltration": storage_infiltration,
        "actual_et": actual_et,
        "percolation": perc,
        "effective_rainfall": total_eff,
        "q9": q9,
        "q1": q1,
        "routing_store": new_r,
        "exchange": f,
        "actual_exchange_routing": aex_routing,
        "actual_exchange_direct": aex_direct,
        "actual_exchange_total": aex_routing + aex_direct,
        "qr": qr,
        "qrexp": qrexp,
        "exponential_store": new_exp,
        "qd": qd,
        "streamflow": streamflow,
    }
    return new_state, fluxes


def run_series(params: Array, precip: Array, pet: Array) -> dict[str, Array]:
    uh1_ord, uh2_ord = compute_uh_ordinates(params[3])
    init = State(
        production_store=0.3 * params[0],
        routing_store=0.5 * params[2],
        exponential_store=jnp.asarray(0.0, dtype=jnp.float64),
        uh1=jnp.zeros(UH1_LEN, dtype=jnp.float64),
        uh2=jnp.zeros(UH2_LEN, dtype=jnp.float64),
    )

    def body(state: State, forcing: tuple[Array, Array]) -> tuple[State, dict[str, Array]]:
        p, e = forcing
        new_state, fluxes = reference_step(state, params, p, e, uh1_ord, uh2_ord)
        return new_state, fluxes

    _, fluxes = jax.lax.scan(body, init, (precip, pet))
    return fluxes


def test_canonical_per_process_parity() -> None:
    data = np.load(FIXTURES / "gr6j_camels_06224000.npz")
    params = as_param_vector(data["params"])
    precip = jnp.asarray(data["precip"], dtype=jnp.float64)
    pet = jnp.asarray(data["pet"], dtype=jnp.float64)

    fluxes = run_series(params, precip, pet)

    for key in FLUX_KEYS:
        assert_allclose(np.asarray(fluxes[key]), data[key], rtol=RTOL, atol=ATOL)


def test_exchange_limb_parity_and_non_vacuity() -> None:
    data = np.load(FIXTURES / "gr6j_camels_06224000_exchange.npz")
    params = as_param_vector(data["params"])
    assert_allclose(np.asarray(params[1]), 1.0)
    assert_allclose(np.asarray(params[4]), 0.5)
    precip = jnp.asarray(data["precip"], dtype=jnp.float64)
    pet = jnp.asarray(data["pet"], dtype=jnp.float64)

    fluxes = run_series(params, precip, pet)

    for key in ("exchange", "actual_exchange_routing", "actual_exchange_direct", "actual_exchange_total"):
        assert_allclose(np.asarray(fluxes[key]), data[key], rtol=RTOL, atol=ATOL)
    actual_routing = np.asarray(fluxes["actual_exchange_routing"])
    exchange = np.asarray(fluxes["exchange"])
    assert np.any(actual_routing != exchange)
    assert np.any(np.isclose(actual_routing, exchange))


def test_masked_kernel_uh_table_parity() -> None:
    data = np.load(FIXTURES / "gr6j_uh_ordinates.npz")

    for idx, x4 in enumerate(data["x4_grid"]):
        uh1, uh2 = compute_uh_ordinates(jnp.asarray(x4, dtype=jnp.float64))
        assert uh1.shape == (UH1_LEN,)
        assert uh2.shape == (UH2_LEN,)
        assert_allclose(np.asarray(uh1), data["uh1_ord"][idx], rtol=RTOL, atol=ATOL)
        assert_allclose(np.asarray(uh2), data["uh2_ord"][idx], rtol=RTOL, atol=ATOL)
        assert_allclose(np.asarray(jnp.sum(uh1)), 1.0, atol=1e-6)
        assert_allclose(np.asarray(jnp.sum(uh2)), 1.0, atol=1e-6)


def test_uh_finite_gradient_and_ordinate_value_continuity_at_integer_x4() -> None:
    data = np.load(FIXTURES / "gr6j_uh_ordinates.npz")

    def g(x4: Array) -> Array:
        return jnp.sum(jnp.arange(UH1_LEN, dtype=jnp.float64) * compute_uh_ordinates(x4)[0])

    for x4 in [2.0 - 1e-3, 2.0, 2.0 + 1e-3, *data["x4_grid"]]:
        grad = jax.grad(g)(jnp.asarray(x4, dtype=jnp.float64))
        assert np.isfinite(np.asarray(grad))

    uh1_left, uh2_left = compute_uh_ordinates(jnp.asarray(2.0 - 1e-3, dtype=jnp.float64))
    uh1_right, uh2_right = compute_uh_ordinates(jnp.asarray(2.0 + 1e-3, dtype=jnp.float64))
    assert_allclose(np.asarray(uh1_left), np.asarray(uh1_right), atol=1e-2)
    assert_allclose(np.asarray(uh2_left), np.asarray(uh2_right), atol=1e-2)


def test_positive_branch_exponential_store_parity() -> None:
    data = np.load(FIXTURES / "gr6j_step_branches.npz")
    params = as_param_vector(data["params"])
    uh1_ord = jnp.asarray(data["uh1_ord"], dtype=jnp.float64)
    uh2_ord = jnp.asarray(data["uh2_ord"], dtype=jnp.float64)
    ars = []

    for k in range(data["input_states"].shape[0]):
        state = State.from_flat(data["input_states"][k])
        new_state, fluxes = reference_step(
            state,
            params,
            jnp.asarray(data["precip"][k], dtype=jnp.float64),
            jnp.asarray(data["pet"][k], dtype=jnp.float64),
            uh1_ord,
            uh2_ord,
        )
        assert_allclose(np.asarray(fluxes["qrexp"]), data["qrexp"][k], rtol=RTOL, atol=ATOL)
        assert_allclose(np.asarray(new_state.exponential_store), data["exponential_store"][k], rtol=RTOL, atol=ATOL)

        store_exp = data["input_states"][k, 2] + C * np.asarray(fluxes["q9"]) + np.asarray(fluxes["exchange"])
        ars.append(np.clip(store_exp / np.asarray(params[5]), -33.0, 33.0))

    ar = np.asarray(ars)
    assert np.any(ar > 7.0)
    assert np.any(ar >= 33.0)


def test_jit_and_finite_gradients() -> None:
    data = np.load(FIXTURES / "gr6j_camels_06224000.npz")
    params = as_param_vector(data["params"])
    precip = jnp.asarray(data["precip"], dtype=jnp.float64)
    pet = jnp.asarray(data["pet"], dtype=jnp.float64)

    jitted_fluxes = jax.jit(run_series)(params, precip, pet)
    assert np.asarray(jitted_fluxes["streamflow"]).shape == data["streamflow"].shape

    def loss(p: Array) -> Array:
        return jnp.sum(run_series(p, precip, pet)["streamflow"])

    assert np.all(np.isfinite(np.asarray(jax.grad(loss)(params))))
