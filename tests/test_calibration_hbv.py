"""HBV dual-calibration tests: 14-D adapter round-trip, gradient recovery, batched==loop, ctrl-freak e2e."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from ctrl_freak.algorithms.ga import ga
from ctrl_freak.algorithms.nsga2 import nsga2

from hydrologeez.calibration import (
    HBV_SPEC,
    array_to_model,
    bounds_array,
    calibrate_gradient,
    flat_to_model,
    make_batch_evaluator,
    make_objective,
    model_to_flat,
    params_to_array,
)
from hydrologeez.metrics import nse
from hydrologeez.models.hbv import HBVForcing, HBVModel

FIXTURE = Path(__file__).parent / "fixtures" / "hbv_camels_06224000.npz"
N_PARAMS = 14


def _load_fixture():
    return np.load(FIXTURE, allow_pickle=False)


def _build_model_from_params(params) -> HBVModel:
    p = np.asarray(params, dtype=np.float64)
    return HBVModel(
        tt=jnp.asarray(p[0], dtype=jnp.float64),
        cfmax=jnp.asarray(p[1], dtype=jnp.float64),
        sfcf=jnp.asarray(p[2], dtype=jnp.float64),
        cwh=jnp.asarray(p[3], dtype=jnp.float64),
        cfr=jnp.asarray(p[4], dtype=jnp.float64),
        fc=jnp.asarray(p[5], dtype=jnp.float64),
        lp=jnp.asarray(p[6], dtype=jnp.float64),
        beta=jnp.asarray(p[7], dtype=jnp.float64),
        k0=jnp.asarray(p[8], dtype=jnp.float64),
        k1=jnp.asarray(p[9], dtype=jnp.float64),
        k2=jnp.asarray(p[10], dtype=jnp.float64),
        perc=jnp.asarray(p[11], dtype=jnp.float64),
        uzl=jnp.asarray(p[12], dtype=jnp.float64),
        maxbas=jnp.asarray(p[13], dtype=jnp.float64),
    )


def _build_forcing(data) -> HBVForcing:
    return HBVForcing(
        precip=jnp.asarray(data["precip"], dtype=jnp.float64),
        pet=jnp.asarray(data["pet"], dtype=jnp.float64),
        temp=jnp.asarray(data["temp"], dtype=jnp.float64),
    )


def _simulate(model, forcing):
    return model.run(forcing)


def test_hbv_adapter_roundtrips_exactly():
    data = _load_fixture()
    model = _build_model_from_params(data["params"])

    theta = params_to_array(model, spec=HBV_SPEC)
    np.testing.assert_array_equal(np.asarray(theta), np.asarray(data["params"], dtype=np.float64))

    rebuilt = array_to_model(model, theta, spec=HBV_SPEC)
    np.testing.assert_array_equal(np.asarray(params_to_array(rebuilt, spec=HBV_SPEC)), np.asarray(theta))

    flat, aux = model_to_flat(model)
    assert flat.shape == (N_PARAMS,)
    np.testing.assert_allclose(np.sort(np.asarray(flat)), np.sort(np.asarray(theta)))
    back = flat_to_model(flat, aux)
    np.testing.assert_array_equal(
        np.asarray(params_to_array(back, spec=HBV_SPEC)),
        np.asarray(params_to_array(model, spec=HBV_SPEC)),
    )


def test_hbv_gradient_calibration_recovers_synthetic_params():
    data = _load_fixture()
    warmup = int(data["warmup_length"])
    forcing = _build_forcing(data)
    true_theta = jnp.asarray(data["params"], dtype=jnp.float64)
    true_model = array_to_model(_build_model_from_params(data["params"]), true_theta, spec=HBV_SPEC)
    observed = _simulate(true_model, forcing)

    lo, hi = bounds_array(HBV_SPEC)
    # Mild perturbation of the true params (clipped into bounds) keeps the 14-D
    # synthetic-flow recovery inside the basin of attraction.
    start_theta = jnp.clip(true_theta * 0.9, lo, hi)
    start = array_to_model(true_model, start_theta, spec=HBV_SPEC)

    def loss_term(o, s):
        return 1.0 - nse(o, s)

    calibrated, losses = calibrate_gradient(
        start,
        forcing,
        observed,
        simulate=_simulate,
        loss_term=loss_term,
        warmup=warmup,
        n_steps=800,
        learning_rate=5e-2,
        param_spec=HBV_SPEC,
    )
    losses = np.asarray(losses)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]

    fit = _simulate(calibrated, forcing)
    recovered_nse = float(nse(observed[warmup:], jnp.asarray(fit)[warmup:]))
    assert recovered_nse > 0.9


def test_hbv_vmapped_batch_equals_per_individual_loop():
    data = _load_fixture()
    warmup = int(data["warmup_length"])
    forcing = _build_forcing(data)
    observed = jnp.asarray(data["streamflow"], dtype=jnp.float64)
    template = _build_model_from_params(data["params"])
    evaluate = make_objective(
        template,
        forcing,
        observed,
        simulate=_simulate,
        objective_term=lambda o, s: 1.0 - nse(o, s),
        warmup=warmup,
        param_spec=HBV_SPEC,
    )
    evaluate_batch = make_batch_evaluator(evaluate)

    lo, hi = np.asarray(bounds_array(HBV_SPEC)[0]), np.asarray(bounds_array(HBV_SPEC)[1])
    rng = np.random.default_rng(0)
    pop = rng.uniform(lo, hi, size=(8, N_PARAMS))

    per_individual = np.stack([np.asarray(evaluate(jnp.asarray(p))) for p in pop])
    batched = evaluate_batch(pop)
    np.testing.assert_allclose(batched, per_individual, rtol=1e-6, atol=1e-9)


def _evolutionary_setup(data, objective_term):
    warmup = int(data["warmup_length"])
    forcing = _build_forcing(data)
    observed = jnp.asarray(data["streamflow"], dtype=jnp.float64)
    template = _build_model_from_params(data["params"])
    evaluate = make_objective(
        template,
        forcing,
        observed,
        simulate=_simulate,
        objective_term=objective_term,
        warmup=warmup,
        param_spec=HBV_SPEC,
    )
    evaluate_batch = make_batch_evaluator(evaluate)
    lo, hi = np.asarray(bounds_array(HBV_SPEC)[0]), np.asarray(bounds_array(HBV_SPEC)[1])

    def init(rng):
        return rng.uniform(lo, hi)

    def crossover(p1, p2):
        return 0.5 * (p1 + p2)

    def mutate(x):
        return np.clip(x, lo, hi)

    def trap_evaluate(_x):
        raise AssertionError("per-individual evaluate entered; batched path NOT used")

    return evaluate_batch, init, crossover, mutate, trap_evaluate


def test_hbv_ga_runs_end_to_end_via_evaluate_batch():
    data = _load_fixture()
    evaluate_batch, init, crossover, mutate, trap = _evolutionary_setup(data, lambda o, s: 1.0 - nse(o, s))
    seen = {}

    def spy(pop_matrix):
        seen["shape"] = np.asarray(pop_matrix).shape
        return evaluate_batch(pop_matrix)

    result = ga(
        init=init,
        evaluate=trap,
        crossover=crossover,
        mutate=mutate,
        pop_size=8,
        n_generations=3,
        seed=0,
        evaluate_batch=spy,
    )
    assert seen["shape"] == (8, N_PARAMS)
    assert result.population.x.shape == (8, N_PARAMS)
    _best_x, best_fit = result.best
    assert np.isfinite(best_fit)


def test_hbv_nsga2_runs_end_to_end_via_evaluate_batch():
    data = _load_fixture()
    evaluate_batch, init, crossover, mutate, trap = _evolutionary_setup(
        data,
        lambda o, s: jnp.asarray([1.0 - nse(o, s), jnp.sqrt(jnp.mean((o - s) ** 2))]),
    )
    seen = {}

    def spy(pop_matrix):
        out = evaluate_batch(pop_matrix)
        seen["shape"] = np.asarray(out).shape
        return out

    result = nsga2(
        init=init,
        evaluate=trap,
        crossover=crossover,
        mutate=mutate,
        pop_size=8,
        n_generations=3,
        seed=0,
        evaluate_batch=spy,
    )
    assert seen["shape"] == (8, 2)
    assert len(result.pareto_front) >= 1
