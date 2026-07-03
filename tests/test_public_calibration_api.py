from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

import hydrologeez
from hydrologeez.calibration import GR6J_SPEC, calibrate_evolutionary, calibrate_nsga2
from hydrologeez.metrics import nse, rmse
from hydrologeez.models.gr6j import GR6J, GR6JForcing

FX = Path(__file__).parent / "fixtures" / "gr6j_camels_06224000.npz"


def _tiny_problem():
    d = np.load(FX, allow_pickle=False)
    p = np.asarray(d["params"], dtype=np.float64)
    model = GR6J(
        x1=jnp.asarray(p[0], dtype=jnp.float64),
        x2=jnp.asarray(p[1], dtype=jnp.float64),
        x3=jnp.asarray(p[2], dtype=jnp.float64),
        x4=jnp.asarray(p[3], dtype=jnp.float64),
        x5=jnp.asarray(p[4], dtype=jnp.float64),
        x6=jnp.asarray(p[5], dtype=jnp.float64),
    )
    n = 500
    forcing = GR6JForcing(
        precip=jnp.asarray(d["precip"][:n], dtype=jnp.float64),
        pet=jnp.asarray(d["pet"][:n], dtype=jnp.float64),
    )
    observed = model.run(forcing)
    return model, forcing, observed


def test_calibrate_evolutionary_requires_param_spec():
    model, forcing, observed = _tiny_problem()
    calibrate = cast(Any, calibrate_evolutionary)
    with pytest.raises(TypeError):
        calibrate(model, forcing, observed, objective_term=lambda o, s: 1.0 - nse(o, s), pop_size=8)


def test_calibrate_nsga2_requires_param_spec():
    model, forcing, observed = _tiny_problem()
    calibrate = cast(Any, calibrate_nsga2)
    with pytest.raises(TypeError):
        calibrate(
            model,
            forcing,
            observed,
            objective_term=lambda o, s: jnp.asarray([1.0 - nse(o, s), rmse(o, s)]),
            pop_size=8,
        )


def test_calibrate_evolutionary_pop_size_validation():
    model, forcing, observed = _tiny_problem()
    for bad in (7, 0, -2):
        with pytest.raises(ValueError, match="pop_size must be an even, positive integer"):
            calibrate_evolutionary(
                model,
                forcing,
                observed,
                objective_term=lambda o, s: 1.0 - nse(o, s),
                param_spec=GR6J_SPEC,
                pop_size=bad,
                n_generations=1,
                seed=0,
                warmup=50,
            )


def test_calibrate_nsga2_pop_size_validation():
    model, forcing, observed = _tiny_problem()
    for bad in (7, 0, -2):
        with pytest.raises(ValueError, match="pop_size must be an even, positive integer"):
            calibrate_nsga2(
                model,
                forcing,
                observed,
                objective_term=lambda o, s: jnp.asarray([1.0 - nse(o, s), rmse(o, s)]),
                param_spec=GR6J_SPEC,
                pop_size=bad,
                n_generations=1,
                seed=0,
                warmup=50,
            )


def test_calibrate_nsga2_batch_returns_n_by_nobj_shape(monkeypatch):
    model, forcing, observed = _tiny_problem()
    captured = {}

    class _StopError(Exception):
        pass

    def fake_nsga2(*, init, evaluate, crossover, mutate, pop_size, n_generations, seed, evaluate_batch, **kwargs):
        rng = np.random.default_rng(seed)
        pop = np.stack([init(rng) for _ in range(pop_size)])
        out = np.asarray(evaluate_batch(pop))
        captured["pop_shape"] = pop.shape
        captured["obj_shape"] = out.shape
        raise _StopError

    monkeypatch.setattr("hydrologeez.calibration.api.nsga2", fake_nsga2)
    with pytest.raises(_StopError):
        calibrate_nsga2(
            model,
            forcing,
            observed,
            objective_term=lambda o, s: jnp.asarray([1.0 - nse(o, s), rmse(o, s)]),
            param_spec=GR6J_SPEC,
            pop_size=8,
            n_generations=2,
            seed=0,
            warmup=50,
        )
    assert captured["pop_shape"] == (8, 6)
    assert captured["obj_shape"] == (8, 2)


def test_top_level_import_surface():
    assert callable(hydrologeez.calibrate_evolutionary)
    assert callable(hydrologeez.calibrate_nsga2)
    for name in ("kge", "lognse", "mae", "nse", "pbias", "rmse"):
        assert callable(getattr(hydrologeez, name))
    assert callable(hydrologeez.metrics.nse)
    for name in (
        "calibrate_evolutionary",
        "calibrate_nsga2",
        "metrics",
        "kge",
        "lognse",
        "mae",
        "nse",
        "pbias",
        "rmse",
    ):
        assert name in hydrologeez.__all__
