"""Reference-parity and gradient-finiteness tests for hydrologeez.metrics.

x64 is enabled by tests/conftest.py before any jax import; this module must not
set JAX_ENABLE_X64 or call jax.config.update itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hydrologeez.metrics import LOG_EPS, kge, lognse, mae, nse, pbias, rmse


# --- independent NumPy references (closed-form, ddof=0) ----------------------
def _nse_np(obs: np.ndarray, sim: np.ndarray) -> float:
    return 1.0 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)


def _rmse_np(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def _mae_np(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.mean(np.abs(sim - obs)))


def _pbias_np(obs: np.ndarray, sim: np.ndarray) -> float:
    return 100.0 * np.sum(sim - obs) / np.sum(obs)


def _lognse_np(obs: np.ndarray, sim: np.ndarray, eps: float = LOG_EPS) -> float:
    log_obs = np.log(obs + eps)
    log_sim = np.log(sim + eps)
    return 1.0 - np.sum((log_sim - log_obs) ** 2) / np.sum((log_obs - log_obs.mean()) ** 2)


def _kge_np(obs: np.ndarray, sim: np.ndarray) -> float:
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


_JAX_METRICS = {"nse": nse, "kge": kge, "lognse": lognse, "pbias": pbias, "rmse": rmse, "mae": mae}
_NP_METRICS = {
    "nse": _nse_np,
    "kge": _kge_np,
    "lognse": _lognse_np,
    "pbias": _pbias_np,
    "rmse": _rmse_np,
    "mae": _mae_np,
}


@pytest.fixture
def obs_sim() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260627)
    obs = rng.uniform(0.1, 50.0, size=256)
    sim = np.clip(obs + rng.normal(0.0, 3.0, size=256), 1e-3, None)
    return obs, sim


@pytest.mark.parametrize("name", list(_JAX_METRICS))
def test_metric_matches_numpy_reference(name: str, obs_sim: tuple[np.ndarray, np.ndarray]) -> None:
    obs, sim = obs_sim
    got = np.asarray(_JAX_METRICS[name](jnp.asarray(obs), jnp.asarray(sim)))
    expected = _NP_METRICS[name](obs, sim)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_perfect_fit_identities(obs_sim: tuple[np.ndarray, np.ndarray]) -> None:
    obs, _ = obs_sim
    x = jnp.asarray(obs)
    np.testing.assert_allclose(np.asarray(nse(x, x)), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(kge(x, x)), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(lognse(x, x)), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rmse(x, x)), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(mae(x, x)), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(pbias(x, x)), 0.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", list(_JAX_METRICS))
def test_gradient_finite_wrt_sim(name: str, obs_sim: tuple[np.ndarray, np.ndarray]) -> None:
    obs, sim = obs_sim
    grad = jax.grad(_JAX_METRICS[name], argnums=1)(jnp.asarray(obs), jnp.asarray(sim))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_lognse_gradient_finite_at_zero_and_small_flows() -> None:
    obs = jnp.asarray([0.0, 0.0, 1e-8, 0.5, 2.0, 10.0, 0.01, 25.0])
    sim = jnp.asarray([0.0, 1e-9, 0.0, 0.4, 2.3, 9.0, 0.0, 27.0])
    grad = jax.grad(lognse, argnums=1)(obs, sim)
    assert np.all(np.isfinite(np.asarray(grad)))
