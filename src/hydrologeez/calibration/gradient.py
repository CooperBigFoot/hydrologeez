"""Gradient-based calibration: optax + jax.grad through the model run."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from hydrologeez.calibration.adapter import array_to_model, bounds_array, params_to_array

_EPS = 1e-6


def _to_unconstrained(x: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    z = jnp.clip((x - lo) / (hi - lo), _EPS, 1.0 - _EPS)
    return jnp.log(z) - jnp.log1p(-z)


def _to_bounded(u: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    return lo + (hi - lo) * jax.nn.sigmoid(u)


def calibrate_gradient(
    template: eqx.Module,
    forcing,
    observed: jax.Array,
    *,
    simulate: Callable[[eqx.Module, object], jax.Array],
    loss_term: Callable[[jax.Array, jax.Array], jax.Array],
    warmup: int = 365,
    n_steps: int = 200,
    learning_rate: float = 5e-2,
    optimizer: optax.GradientTransformation | None = None,
) -> tuple[eqx.Module, jax.Array]:
    """Minimize ``loss_term(observed, sim)`` over x1..x6 via optax.

    Parameters
    ----------
    template : eqx.Module
        Model providing structure and starting x1..x6.
    forcing : object
        Stacked forcing consumed by ``simulate``.
    observed : jax.Array
        Observed streamflow series.
    simulate : Callable
        ``(model, forcing) -> streamflow array`` (e.g. ``lambda m, f: m.run(f)``).
    loss_term : Callable
        ``(obs, sim) -> scalar`` to MINIMIZE (e.g. ``lambda o, s: 1.0 - nse(o, s)``).
    warmup : int
        Leading steps excluded from the loss.
    n_steps, learning_rate, optimizer
        Optimization controls.

    Returns
    -------
    (calibrated_model, losses) where losses is a (n_steps,) float64 array.
    """
    lo, hi = bounds_array()
    x0 = params_to_array(template)
    u = _to_unconstrained(x0, lo, hi)
    opt = optimizer if optimizer is not None else optax.adam(learning_rate)
    opt_state = opt.init(u)

    obs_eval = observed[warmup:]

    def loss_fn(u_vec: jax.Array) -> jax.Array:
        theta = _to_bounded(u_vec, lo, hi)
        model = array_to_model(template, theta)
        sim = simulate(model, forcing)
        return loss_term(obs_eval, sim[warmup:])

    @jax.jit
    def step(u_vec, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(u_vec)
        updates, opt_state = opt.update(grads, opt_state)
        u_vec = optax.apply_updates(u_vec, updates)
        return u_vec, opt_state, loss

    losses = []
    for _ in range(n_steps):
        u, opt_state, loss = step(u, opt_state)
        losses.append(loss)

    final_model = array_to_model(template, _to_bounded(u, lo, hi))
    return final_model, jnp.asarray(losses, dtype=jnp.float64)
