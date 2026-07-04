"""Adapter between an Equinox model's params and a flat float64 array.

Two equivalent views are provided:

* ``model_to_flat`` / ``flat_to_model`` -- the generic Equinox view via
  ``eqx.partition(model, eqx.is_inexact_array)`` + ``jax.flatten_util.ravel_pytree``
  (the DESIGN-mandated mechanism). Model-agnostic; behavior unchanged.
* ``params_to_array`` / ``array_to_model`` -- an explicit, ordered named-param view
  used as the canonical column order for the ctrl-freak ``(pop_size, n_params)``
  population matrix. Parameterised by a ``ParamSpec`` (names + bounds) defaulting to
  ``GR6J_SPEC`` so existing GR6J call sites are byte-identical.

The two views agree element-wise (asserted in tests).
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from hydrologeez.models.hbv import constants as hbv_constants


@dataclass(frozen=True)
class ParamSpec:
    """Ordered parameter names plus per-name lower/upper calibration bounds.

    ``names``/``lower``/``upper`` are positionally aligned: ``lower[i]``/``upper[i]``
    are the calibration bounds for the parameter ``names[i]``.
    """

    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]


# --- GR6J (default spec; values unchanged from the original module) ----------
# Canonical parameter order = the column order of the ctrl-freak population matrix.
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6")

# Code bounds; x6 uses [1, 50].
LOWER_BOUNDS: tuple[float, ...] = (1.0, -5.0, 1.0, 0.5, -4.0, 1.0)
UPPER_BOUNDS: tuple[float, ...] = (2500.0, 5.0, 1000.0, 10.0, 4.0, 50.0)

GR6J_SPEC: ParamSpec = ParamSpec(names=PARAM_NAMES, lower=LOWER_BOUNDS, upper=UPPER_BOUNDS)


# --- HBV-Light single-zone (14 params, canonical order tt..maxbas) -----------
# Names + bounds are the single source of truth in models/hbv/constants.py.
# Only ``maxbas`` is hard-validated; the other 13 bounds are advisory (calibration only).
def _hbv_bounds() -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    names = tuple(str(n) for n in hbv_constants.PARAM_NAMES)
    pb = hbv_constants.PARAM_BOUNDS
    lower = tuple(float(pb[n][0]) for n in names)
    upper = tuple(float(pb[n][1]) for n in names)
    return names, lower, upper


_HBV_NAMES, _HBV_LOWER, _HBV_UPPER = _hbv_bounds()
HBV_SPEC: ParamSpec = ParamSpec(names=_HBV_NAMES, lower=_HBV_LOWER, upper=_HBV_UPPER)


def bounds_array(spec: ParamSpec = GR6J_SPEC) -> tuple[jax.Array, jax.Array]:
    """Return ``(lower, upper)`` float64 arrays in ``spec.names`` order."""
    return (
        jnp.asarray(spec.lower, dtype=jnp.float64),
        jnp.asarray(spec.upper, dtype=jnp.float64),
    )


def params_to_array(model: eqx.Module, spec: ParamSpec = GR6J_SPEC) -> jax.Array:
    """Extract the named-param vector from a model in ``spec.names`` order."""
    return jnp.asarray([jnp.asarray(getattr(model, n)) for n in spec.names], dtype=jnp.float64)


def array_to_model(template: eqx.Module, theta: jax.Array, spec: ParamSpec = GR6J_SPEC) -> eqx.Module:
    """Rebuild a model from ``template`` with ``spec.names`` replaced by ``theta``.

    Replacement leaves are ``jnp`` (inexact) arrays so the rebuilt model's named
    params are always inexact-array leaves regardless of how ``template`` was built.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    return eqx.tree_at(
        lambda m: [getattr(m, n) for n in spec.names],
        template,
        [theta[i] for i in range(len(spec.names))],
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
