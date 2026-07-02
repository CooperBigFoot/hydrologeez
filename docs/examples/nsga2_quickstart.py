"""End-to-end hydrologeez NSGA-II multi-objective calibration example."""

import os

os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hydrologeez import calibrate_nsga2  # noqa: E402
from hydrologeez.calibration import GR6J_SPEC  # noqa: E402
from hydrologeez.metrics import nse, rmse  # noqa: E402
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


def main() -> None:
    data = np.load(_find_fixtures_dir() / "gr6j_camels_06224000.npz", allow_pickle=False)
    warmup = int(data["warmup_length"])
    true_model = _model_from_params(np.asarray(data["params"], dtype=np.float64))
    forcing = GR6JForcing(
        precip=jnp.asarray(data["precip"], dtype=jnp.float64),
        pet=jnp.asarray(data["pet"], dtype=jnp.float64),
    )
    observed = true_model.run(forcing)

    front_models, result = calibrate_nsga2(
        true_model,
        forcing,
        observed,
        objective_term=lambda o, s: jnp.asarray([1.0 - nse(o, s), rmse(o, s)]),
        param_spec=GR6J_SPEC,
        pop_size=16,
        n_generations=20,
        seed=0,
        warmup=warmup,
    )

    objectives = result.pareto_front.objectives
    print(f"[NSGA-II] Pareto front: {len(front_models)} model(s)")
    for i, obj in enumerate(np.asarray(objectives)):
        print(f"  member {i}: 1-NSE={obj[0]:.4f}  RMSE={obj[1]:.4f}")


if __name__ == "__main__":
    main()
