"""End-to-end hydrologeez GR6J example."""

import os

os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path  # noqa: E402
from typing import cast  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from hydrologeez import calibrate_evolutionary  # noqa: E402
from hydrologeez.calibration import GR6J_SPEC  # noqa: E402
from hydrologeez.metrics import kge, nse, rmse  # noqa: E402
from hydrologeez.models.gr6j import GR6J, GR6JForcing  # noqa: E402


def _find_fixtures_dir() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / "tests" / "fixtures"
        if (candidate / "gr6j_camels_06224000.npz").exists():
            return candidate
    raise FileNotFoundError("could not locate tests/fixtures; run from the hydrologeez repo root")


def _model_from_params(params: np.ndarray) -> GR6J:
    return GR6J(
        x1=jnp.asarray(params[0], dtype=jnp.float64),
        x2=jnp.asarray(params[1], dtype=jnp.float64),
        x3=jnp.asarray(params[2], dtype=jnp.float64),
        x4=jnp.asarray(params[3], dtype=jnp.float64),
        x5=jnp.asarray(params[4], dtype=jnp.float64),
        x6=jnp.asarray(params[5], dtype=jnp.float64),
    )


def forward_and_parity() -> tuple[GR6J, GR6JForcing, int]:
    data = np.load(_find_fixtures_dir() / "gr6j_camels_06224000.npz", allow_pickle=False)
    warmup = int(data["warmup_length"])
    model = _model_from_params(np.asarray(data["params"], dtype=np.float64))
    forcing = GR6JForcing(
        precip=jnp.asarray(data["precip"], dtype=jnp.float64),
        pet=jnp.asarray(data["pet"], dtype=jnp.float64),
    )

    sim = np.asarray(model.run(forcing))
    oracle = np.asarray(data["streamflow"], dtype=np.float64)

    max_rel = float(np.max(np.abs(sim[warmup:] - oracle[warmup:]) / (np.abs(oracle[warmup:]) + 1e-8)))
    nse_val = float(nse(jnp.asarray(oracle[warmup:]), jnp.asarray(sim[warmup:])))
    kge_val = float(kge(jnp.asarray(oracle[warmup:]), jnp.asarray(sim[warmup:])))
    print(f"[parity] max relative streamflow error vs Rust oracle: {max_rel:.2e}")
    print(f"[metrics] NSE = {nse_val:.6f}; KGE = {kge_val:.6f} (post warm-up)")
    return model, forcing, warmup


def gradient_demo(true_model: GR6J, forcing: GR6JForcing, warmup: int) -> None:
    target = jax.lax.stop_gradient(true_model.run(forcing))[warmup:]

    def loss_fn(x1: jax.Array) -> jax.Array:
        model = eqx.tree_at(lambda m: m.x1, true_model, x1)
        return rmse(target, model.run(forcing)[warmup:])

    value_and_grad = eqx.filter_jit(jax.value_and_grad(loss_fn))
    x1 = jnp.asarray(true_model.x1 * 1.4, dtype=jnp.float64)
    opt = optax.adam(2.0)
    opt_state = opt.init(x1)
    first_loss = float(loss_fn(x1))
    loss = first_loss
    for _ in range(35):
        loss_value, grad = value_and_grad(x1)
        updates, opt_state = opt.update(grad, opt_state, x1)
        updated_x1 = cast(jax.Array, optax.apply_updates(x1, updates))
        x1 = jnp.clip(updated_x1, 1.0, 2500.0)
        loss = float(loss_value)

    final_loss = float(loss_fn(x1))
    print(
        f"[gradient] x1 recovered {float(x1):.2f} (true {float(true_model.x1):.2f}); "
        f"RMSE {first_loss:.4f} -> {final_loss:.4f}"
    )
    if final_loss > loss:
        print(f"[gradient] best iterated RMSE was {loss:.4f}")


def evolutionary_demo(true_model: GR6J, forcing: GR6JForcing, warmup: int) -> None:
    observed = true_model.run(forcing)
    _best_model, result = calibrate_evolutionary(
        true_model,
        forcing,
        observed,
        objective_term=lambda o, s: 1.0 - nse(o, s),
        param_spec=GR6J_SPEC,
        pop_size=16,
        n_generations=20,
        seed=0,
        warmup=warmup,
    )
    _best_x, best_fit = result.best
    print(f"[ctrl-freak GA] best objective {best_fit:.6f}")


def main() -> None:
    model, forcing, warmup = forward_and_parity()
    gradient_demo(model, forcing, warmup)
    evolutionary_demo(model, forcing, warmup)


main()
