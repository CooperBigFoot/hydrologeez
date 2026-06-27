"""Structural, autodiff, jit, and batch smoke tests for the GR6J SSM model."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from hydrologeez.models.gr6j.model import GR6J, GR6JForcing
from hydrologeez.models.gr6j.state import State

FIXTURES = Path(__file__).parents[2] / "fixtures"


def _canonical():
    data = np.load(FIXTURES / "gr6j_camels_06224000.npz", allow_pickle=False)
    params = data["params"]
    model = GR6J(
        x1=jnp.asarray(params[0]),
        x2=jnp.asarray(params[1]),
        x3=jnp.asarray(params[2]),
        x4=jnp.asarray(params[3]),
        x5=jnp.asarray(params[4]),
        x6=jnp.asarray(params[5]),
    )
    n = 200
    forcing = GR6JForcing(precip=jnp.asarray(data["precip"][:n]), pet=jnp.asarray(data["pet"][:n]))
    target = jnp.asarray(data["streamflow"][:n])
    return model, forcing, target, n


def test_transition_single_step_is_finite_and_typed():
    model, forcing, _target, _n = _canonical()
    state = model.init_state()
    one = GR6JForcing(precip=forcing.precip[0], pet=forcing.pet[0])
    new_state, fluxes = model.transition(state, one)
    assert isinstance(new_state, State)
    assert np.isfinite(np.asarray(fluxes.streamflow))
    assert np.asarray(fluxes.streamflow) >= 0.0


def test_run_shapes_and_state_roundtrip():
    model, forcing, _target, n = _canonical()
    obs, fluxes, final = model.run(forcing, return_fluxes=True)
    assert np.asarray(obs).shape == (n,)
    assert np.asarray(fluxes.streamflow).shape == (n,)
    npt.assert_allclose(np.asarray(obs), np.asarray(fluxes.streamflow), rtol=0, atol=0)
    assert isinstance(final, State)
    assert np.asarray(final.production_store).shape == ()
    assert np.asarray(final.uh1).shape == (model.nh,)
    assert np.asarray(final.uh2).shape == (2 * model.nh,)


def test_observation_operator_swap_changes_observable():
    model, forcing, _target, _n = _canonical()
    default_obs = np.asarray(model.run(forcing))
    swapped = np.asarray(model.run(forcing, observation_operator=lambda state, fluxes: fluxes.exponential_store))
    assert not np.allclose(default_obs, swapped)


def test_run_is_jit_compatible():
    model, forcing, _target, _n = _canonical()
    jitted = eqx.filter_jit(lambda m, f: m.run(f))
    npt.assert_allclose(np.asarray(jitted(model, forcing)), np.asarray(model.run(forcing)), rtol=1e-10, atol=1e-12)


def test_batch_run_equals_per_member_loop():
    model, forcing, _target, _n = _canonical()
    forcings = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), forcing)
    batched = np.asarray(model.batch_run(forcings))
    looped = np.stack([np.asarray(model.run(forcing)), np.asarray(model.run(forcing))])
    npt.assert_allclose(batched, looped, rtol=1e-10, atol=1e-12)


def test_param_batch_vmap_runs():
    _model, forcing, _target, n = _canonical()
    data = np.load(FIXTURES / "gr6j_camels_06224000.npz", allow_pickle=False)
    pbatch = jnp.stack([jnp.asarray(data["params"]), jnp.asarray(data["params"])])

    def run_from_pvec(pvec):
        model = GR6J(x1=pvec[0], x2=pvec[1], x3=pvec[2], x4=pvec[3], x5=pvec[4], x6=pvec[5])
        return model.run(forcing)

    out = np.asarray(jax.vmap(run_from_pvec)(pbatch))
    assert out.shape == (2, n)
    assert np.all(np.isfinite(out))


def test_autodiff_gradients_are_finite():
    model, forcing, target, _n = _canonical()

    @eqx.filter_grad
    def loss(model):
        return jnp.mean((model.run(forcing) - target) ** 2)

    grads = loss(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves
    for leaf in leaves:
        assert np.all(np.isfinite(np.asarray(leaf)))
