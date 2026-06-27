"""Tests for the abstract StateSpaceModel base via an analytic dummy model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from hydrologeez.ssm import StateSpaceModel


class DummyFluxes(eqx.Module):
    """Minimal fluxes PyTree exposing two distinct named fluxes."""

    streamflow: jax.Array
    inflow: jax.Array


class LinearReservoir(StateSpaceModel):
    """Discrete linear reservoir: s_{t+1} = a * s_t + u_t.

    Default observable (``streamflow``) is the reservoir storage after the step;
    ``inflow`` is the per-step forcing (a deliberately different flux).
    """

    a: float
    s0: float

    def init_state(self):
        return jnp.asarray(self.s0)

    def transition(self, state, forcing):
        s_next = self.a * state + forcing
        return s_next, DummyFluxes(streamflow=s_next, inflow=forcing)


def inflow_observation(state, fluxes):
    """Non-default observation operator: return the inflow flux instead."""
    return fluxes.inflow


def test_run_matches_closed_form():
    a, s0, u, T = 0.7, 2.0, 1.5, 50
    model = LinearReservoir(a=a, s0=s0)
    forcing = jnp.full((T,), u)
    observed = np.asarray(model.run(forcing))
    t = np.arange(1, T + 1)
    expected = a**t * s0 + u * (1.0 - a**t) / (1.0 - a)
    npt.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_batch_run_equals_per_member_loop():
    model = LinearReservoir(a=0.5, s0=1.0)
    rng = np.random.default_rng(0)
    batch = jnp.asarray(rng.standard_normal((4, 30)))
    batched = np.asarray(model.batch_run(batch))
    looped = np.stack([np.asarray(model.run(batch[i])) for i in range(batch.shape[0])])
    npt.assert_allclose(batched, looped, rtol=1e-12, atol=1e-12)


def test_observation_operator_swap_changes_observable():
    model = LinearReservoir(a=0.6, s0=0.0)
    forcing = jnp.asarray(np.linspace(1.0, 5.0, 20))
    default_obs = np.asarray(model.run(forcing))
    swapped_obs = np.asarray(model.run(forcing, observation_operator=inflow_observation))
    # The swapped operator returns the inflow (== forcing) and differs from default.
    npt.assert_allclose(swapped_obs, np.asarray(forcing), rtol=1e-12, atol=1e-12)
    assert not np.allclose(default_obs, swapped_obs)


def test_run_is_jit_compatible():
    model = LinearReservoir(a=0.8, s0=1.0)
    forcing = jnp.ones((10,))
    jitted = eqx.filter_jit(lambda m, f: m.run(f))
    npt.assert_allclose(
        np.asarray(jitted(model, forcing)),
        np.asarray(model.run(forcing)),
        rtol=1e-12,
        atol=1e-12,
    )
