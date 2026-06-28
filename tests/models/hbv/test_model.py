"""Structural, state-roundtrip, autodiff, jit, and batch smoke tests for HBVModel."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from hydrologeez.models.hbv.model import HBVFluxes, HBVForcing, HBVModel
from hydrologeez.models.hbv.state import HBVState

FIXTURES = Path(__file__).parents[2] / "fixtures"
PARAM_FIELDS = (
    "tt",
    "cfmax",
    "sfcf",
    "cwh",
    "cfr",
    "fc",
    "lp",
    "beta",
    "k0",
    "k1",
    "k2",
    "perc",
    "uzl",
    "maxbas",
)


def _model_from_params(params):
    p = jnp.asarray(params)
    return HBVModel(
        tt=p[0],
        cfmax=p[1],
        sfcf=p[2],
        cwh=p[3],
        cfr=p[4],
        fc=p[5],
        lp=p[6],
        beta=p[7],
        k0=p[8],
        k1=p[9],
        k2=p[10],
        perc=p[11],
        uzl=p[12],
        maxbas=p[13],
    )


def _canonical():
    data = np.load(FIXTURES / "hbv_camels_06224000.npz", allow_pickle=False)
    model = _model_from_params(data["params"])
    n = 400
    forcing = HBVForcing(
        precip=jnp.asarray(data["precip"][:n]),
        pet=jnp.asarray(data["pet"][:n]),
        temp=jnp.asarray(data["temp"][:n]),
    )
    target = jnp.asarray(data["streamflow"][:n])
    return model, forcing, target, n


def test_init_state_sm_half_fc_rest_zero():
    model, _f, _t, _n = _canonical()
    s = model.init_state()
    npt.assert_array_equal(np.asarray(s.zone_sm), 0.5 * np.asarray(model.fc))
    npt.assert_array_equal(np.asarray(s.zone_sp), 0.0)
    npt.assert_array_equal(np.asarray(s.zone_lw), 0.0)
    npt.assert_array_equal(np.asarray(s.upper_zone), 0.0)
    npt.assert_array_equal(np.asarray(s.lower_zone), 0.0)
    npt.assert_array_equal(np.asarray(s.routing_buffer), np.zeros(model.routing_buffer_size))


def test_state_flat_layout_and_roundtrip():
    # Distinct values prove the Rust 12-layout order [SP, LW, SM, SUZ, SLZ, b0..b6].
    s = HBVState(
        zone_sp=jnp.asarray(1.0),
        zone_lw=jnp.asarray(2.0),
        zone_sm=jnp.asarray(3.0),
        upper_zone=jnp.asarray(4.0),
        lower_zone=jnp.asarray(5.0),
        routing_buffer=jnp.asarray([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    )
    flat = s.to_flat()
    npt.assert_array_equal(np.asarray(flat), np.arange(1.0, 13.0))
    npt.assert_array_equal(np.asarray(HBVState.from_flat(flat).to_flat()), np.asarray(flat))


def test_transition_single_step_is_finite_and_typed():
    model, forcing, _t, _n = _canonical()
    state = model.init_state()
    one = HBVForcing(precip=forcing.precip[0], pet=forcing.pet[0], temp=forcing.temp[0])
    new_state, fluxes = model.transition(state, one)
    assert isinstance(new_state, HBVState)
    assert isinstance(fluxes, HBVFluxes)
    assert np.isfinite(np.asarray(fluxes.streamflow))


def test_run_shapes_and_fluxes_streamflow_identity():
    model, forcing, _t, n = _canonical()
    obs, fluxes, final = model.run(forcing, return_fluxes=True)
    assert np.asarray(obs).shape == (n,)
    npt.assert_allclose(np.asarray(obs), np.asarray(fluxes.streamflow), rtol=0, atol=0)
    assert isinstance(final, HBVState)
    npt.assert_array_equal(np.asarray(final.routing_buffer).shape, (model.routing_buffer_size,))


def test_run_is_jit_compatible():
    model, forcing, _t, _n = _canonical()
    jitted = eqx.filter_jit(lambda m, f: m.run(f))
    npt.assert_allclose(np.asarray(jitted(model, forcing)), np.asarray(model.run(forcing)), rtol=1e-10, atol=1e-12)


def test_batch_run_equals_per_member_loop():
    model, forcing, _t, _n = _canonical()
    forcings = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), forcing)
    batched = np.asarray(model.batch_run(forcings))
    looped = np.stack([np.asarray(model.run(forcing)), np.asarray(model.run(forcing))])
    npt.assert_allclose(batched, looped, rtol=1e-10, atol=1e-12)


def test_param_batch_vmap_runs():
    _model, forcing, _t, n = _canonical()
    data = np.load(FIXTURES / "hbv_camels_06224000.npz", allow_pickle=False)
    pbatch = jnp.stack([jnp.asarray(data["params"]), jnp.asarray(data["params"])])

    def run_from_pvec(pvec):
        model = HBVModel(
            tt=pvec[0],
            cfmax=pvec[1],
            sfcf=pvec[2],
            cwh=pvec[3],
            cfr=pvec[4],
            fc=pvec[5],
            lp=pvec[6],
            beta=pvec[7],
            k0=pvec[8],
            k1=pvec[9],
            k2=pvec[10],
            perc=pvec[11],
            uzl=pvec[12],
            maxbas=pvec[13],
        )
        return model.run(forcing)

    out = np.asarray(jax.vmap(run_from_pvec)(pbatch))
    assert out.shape == (2, n)
    assert np.all(np.isfinite(out))


def test_autodiff_gradients_are_finite():
    model, forcing, target, _n = _canonical()

    @eqx.filter_grad
    def loss(m):
        return jnp.mean((m.run(forcing) - target) ** 2)

    grads = loss(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves
    for leaf in leaves:
        assert np.all(np.isfinite(np.asarray(leaf)))
