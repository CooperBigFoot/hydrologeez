"""Adapter between a GR6J Equinox module's params and a flat float64 array.

Two equivalent views are provided:

* ``model_to_flat`` / ``flat_to_model`` -- the generic Equinox view via
  ``eqx.partition(model, eqx.is_inexact_array)`` + ``jax.flatten_util.ravel_pytree``
  (the DESIGN-mandated mechanism).
* ``params_to_array`` / ``array_to_model`` -- an explicit, ordered ``x1..x6``
  view used as the canonical column order for the ctrl-freak ``(pop_size, n_params)``
  population matrix.

The two agree element-wise (asserted in tests).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

# Canonical parameter order = the column order of the ctrl-freak population matrix.
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6")

# Code bounds (constants.rs:99-105); x6 uses code bounds [1, 50] for parity.
LOWER_BOUNDS: tuple[float, ...] = (1.0, -5.0, 1.0, 0.5, -4.0, 1.0)
UPPER_BOUNDS: tuple[float, ...] = (2500.0, 5.0, 1000.0, 10.0, 4.0, 50.0)


def bounds_array() -> tuple[jax.Array, jax.Array]:
    """Return ``(lower, upper)`` float64 arrays of shape (6,) in PARAM_NAMES order."""
    return (
        jnp.asarray(LOWER_BOUNDS, dtype=jnp.float64),
        jnp.asarray(UPPER_BOUNDS, dtype=jnp.float64),
    )


def params_to_array(model: eqx.Module) -> jax.Array:
    """Extract the (6,) float64 vector of x1..x6 from a model, in PARAM_NAMES order."""
    return jnp.asarray([jnp.asarray(getattr(model, n)) for n in PARAM_NAMES], dtype=jnp.float64)


def array_to_model(template: eqx.Module, theta: jax.Array) -> eqx.Module:
    """Rebuild a model from ``template`` with x1..x6 replaced by ``theta`` (6,).

    The replacement leaves are ``jnp`` (inexact) arrays, so the rebuilt model's
    x1..x6 are always inexact-array leaves regardless of how ``template`` was built.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    return eqx.tree_at(
        lambda m: [getattr(m, n) for n in PARAM_NAMES],
        template,
        [theta[i] for i in range(len(PARAM_NAMES))],
    )


def model_to_flat(model: eqx.Module):
    """Generic Equinox flatten: returns (flat float64 array, aux) where aux reconstructs."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    flat, unravel = ravel_pytree(params)
    return jnp.asarray(flat, dtype=jnp.float64), (unravel, static)


def flat_to_model(flat: jax.Array, aux) -> eqx.Module:
    """Inverse of ``model_to_flat``."""
    unravel, static = aux
    return eqx.combine(unravel(jnp.asarray(flat, dtype=jnp.float64)), static)
