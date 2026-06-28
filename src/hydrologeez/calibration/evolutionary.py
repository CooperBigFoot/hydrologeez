"""Derivative-free calibration plumbing: vmapped batched evaluator for ctrl-freak."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hydrologeez.calibration.adapter import GR6J_SPEC, ParamSpec, array_to_model


def make_objective(
    template: eqx.Module,
    forcing,
    observed: jax.Array,
    *,
    simulate: Callable[[eqx.Module, object], jax.Array],
    objective_term: Callable[[jax.Array, jax.Array], jax.Array],
    warmup: int = 365,
    param_spec: ParamSpec | None = None,
) -> Callable[[jax.Array], jax.Array]:
    """Build a per-individual objective ``theta(6,) -> scalar | (n_obj,)`` in code-bounds space.

    ``objective_term(obs, sim)`` returns a scalar (GA) or a (n_obj,) vector (NSGA-II);
    LOWER is better in both cases.
    """
    spec = param_spec if param_spec is not None else GR6J_SPEC
    obs_eval = observed[warmup:]

    def evaluate(theta: jax.Array) -> jax.Array:
        model = array_to_model(template, jnp.asarray(theta, dtype=jnp.float64), spec)
        sim = simulate(model, forcing)
        return objective_term(obs_eval, sim[warmup:])

    return evaluate


def make_batch_evaluator(
    evaluate: Callable[[jax.Array], jax.Array],
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a per-individual ``evaluate`` into a ctrl-freak ``evaluate_batch`` callable.

    Receives the full (n, n_params) population matrix and returns objectives via a
    single vmapped, jit-compiled call -- the batched path, NOT a per-individual loop.
    GA: returns (n,). NSGA-II: returns (n, n_obj).
    """
    vmapped = eqx.filter_jit(jax.vmap(evaluate))

    def evaluate_batch(pop_matrix: np.ndarray) -> np.ndarray:
        return np.asarray(vmapped(jnp.asarray(pop_matrix, dtype=jnp.float64)))

    return evaluate_batch
