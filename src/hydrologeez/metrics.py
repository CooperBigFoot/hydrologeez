"""JAX-native differentiable hydrological performance metrics.

All metrics are pure free functions on ``(obs, sim)`` JAX arrays and are written
to be ``jax.grad``-clean: the logarithmic transform is guarded with a small
additive constant so ``log`` and its derivative stay finite at zero flow. float64
is required process-wide and is enforced at ``import hydrologeez``; the test suite
enables it via ``tests/conftest.py`` before any jax import.

Signature convention: every metric takes ``(obs, sim)`` in that order. Gradient
calibration differentiates w.r.t. ``sim`` (``argnums=1``).
"""

import jax.numpy as jnp
from jax import Array

__all__ = ["kge", "lognse", "mae", "nse", "pbias", "rmse"]

# Additive constant for the logarithmic transform in ``lognse``: keeps ``log``
# finite at zero flow and bounds its derivative ``1 / (x + LOG_EPS)``.
LOG_EPS: float = 1e-6


def nse(obs: Array, sim: Array) -> Array:
    """Nash-Sutcliffe Efficiency: ``1 - SS_res / SS_tot``."""
    obs = jnp.asarray(obs)
    sim = jnp.asarray(sim)
    numerator = jnp.sum(jnp.square(sim - obs))
    denominator = jnp.sum(jnp.square(obs - jnp.mean(obs)))
    return 1.0 - numerator / denominator


def rmse(obs: Array, sim: Array) -> Array:
    """Root Mean Squared Error."""
    obs = jnp.asarray(obs)
    sim = jnp.asarray(sim)
    return jnp.sqrt(jnp.mean(jnp.square(sim - obs)))


def mae(obs: Array, sim: Array) -> Array:
    """Mean Absolute Error."""
    obs = jnp.asarray(obs)
    sim = jnp.asarray(sim)
    return jnp.mean(jnp.abs(sim - obs))


def pbias(obs: Array, sim: Array) -> Array:
    """Percent bias (hydroGOF convention): ``100 * sum(sim - obs) / sum(obs)``."""
    obs = jnp.asarray(obs)
    sim = jnp.asarray(sim)
    return 100.0 * jnp.sum(sim - obs) / jnp.sum(obs)


def lognse(obs: Array, sim: Array, eps: float = LOG_EPS) -> Array:
    """Nash-Sutcliffe Efficiency on log-transformed flows.

    Uses ``log(x + eps)`` so the transform and its gradient remain finite at zero
    flow. ``eps`` defaults to :data:`LOG_EPS`.
    """
    obs = jnp.asarray(obs)
    sim = jnp.asarray(sim)
    log_obs = jnp.log(obs + eps)
    log_sim = jnp.log(sim + eps)
    numerator = jnp.sum(jnp.square(log_sim - log_obs))
    denominator = jnp.sum(jnp.square(log_obs - jnp.mean(log_obs)))
    return 1.0 - numerator / denominator


def kge(obs: Array, sim: Array) -> Array:
    """Kling-Gupta Efficiency (Gupta et al., 2009).

    ``KGE = 1 - sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)`` where ``r`` is
    the Pearson correlation, ``alpha = std(sim) / std(obs)`` is the variability
    ratio and ``beta = mean(sim) / mean(obs)`` is the bias ratio (population
    statistics, ``ddof=0``).
    """
    obs = jnp.asarray(obs)
    sim = jnp.asarray(sim)
    mean_obs = jnp.mean(obs)
    mean_sim = jnp.mean(sim)
    obs_anom = obs - mean_obs
    sim_anom = sim - mean_sim
    r = jnp.sum(obs_anom * sim_anom) / jnp.sqrt(jnp.sum(jnp.square(obs_anom)) * jnp.sum(jnp.square(sim_anom)))
    alpha = jnp.std(sim) / jnp.std(obs)
    beta = mean_sim / mean_obs
    return 1.0 - jnp.sqrt(jnp.square(r - 1.0) + jnp.square(alpha - 1.0) + jnp.square(beta - 1.0))
