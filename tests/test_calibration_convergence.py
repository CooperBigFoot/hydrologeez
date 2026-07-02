from pathlib import Path

import jax.numpy as jnp
import numpy as np
from ctrl_freak.algorithms.ga import ga
from ctrl_freak.algorithms.nsga2 import nsga2

from hydrologeez.calibration import (
    GR6J_SPEC,
    HBV_SPEC,
    bounds_array,
    calibrate_evolutionary,
    calibrate_nsga2,
    make_batch_evaluator,
    make_objective,
)
from hydrologeez.metrics import nse, rmse
from hydrologeez.models.gr6j import GR6J, GR6JForcing
from hydrologeez.models.hbv import HBVForcing, HBVModel
from hydrologeez.models.hbv.constants import PARAM_NAMES as HBV_PARAM_NAMES

GR6J_FX = Path(__file__).parent / "fixtures" / "gr6j_camels_06224000.npz"
HBV_FX = Path(__file__).parent / "fixtures" / "hbv_camels_06224000.npz"
SEED = 42


def _simulate(model, forcing):
    return model.run(forcing)


def _gr6j_problem():
    d = np.load(GR6J_FX, allow_pickle=False)
    warmup = int(d["warmup_length"])
    p = np.asarray(d["params"], dtype=np.float64)
    true_model = GR6J(
        x1=jnp.asarray(p[0], dtype=jnp.float64),
        x2=jnp.asarray(p[1], dtype=jnp.float64),
        x3=jnp.asarray(p[2], dtype=jnp.float64),
        x4=jnp.asarray(p[3], dtype=jnp.float64),
        x5=jnp.asarray(p[4], dtype=jnp.float64),
        x6=jnp.asarray(p[5], dtype=jnp.float64),
    )
    forcing = GR6JForcing(
        precip=jnp.asarray(d["precip"], dtype=jnp.float64),
        pet=jnp.asarray(d["pet"], dtype=jnp.float64),
    )
    observed = _simulate(true_model, forcing)
    return true_model, forcing, observed, warmup, GR6J_SPEC


def _hbv_problem():
    d = np.load(HBV_FX, allow_pickle=False)
    warmup = int(d["warmup_length"])
    p = np.asarray(d["params"], dtype=np.float64)
    true_model = HBVModel(
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
    assert tuple(HBV_PARAM_NAMES) == tuple(HBV_SPEC.names)
    forcing = HBVForcing(
        precip=jnp.asarray(d["precip"], dtype=jnp.float64),
        pet=jnp.asarray(d["pet"], dtype=jnp.float64),
        temp=jnp.asarray(d["temp"], dtype=jnp.float64),
    )
    observed = _simulate(true_model, forcing)
    return true_model, forcing, observed, warmup, HBV_SPEC


def _noop_run(algo, problem, objective_term, *, pop_size, n_generations, seed):
    template, forcing, observed, warmup, spec = problem
    evaluate = make_objective(
        template,
        forcing,
        observed,
        simulate=_simulate,
        objective_term=objective_term,
        warmup=warmup,
        param_spec=spec,
    )
    evaluate_batch = make_batch_evaluator(evaluate)
    lo = np.asarray(bounds_array(spec)[0], dtype=float)
    hi = np.asarray(bounds_array(spec)[1], dtype=float)

    def init(rng):
        return rng.uniform(lo, hi)

    def noop_crossover(p1, p2):
        return 0.5 * (p1 + p2)

    def noop_mutate(x):
        return np.clip(x, lo, hi)

    def trap(_x):
        raise AssertionError("per-individual evaluate entered; batched path NOT used")

    return algo(
        init=init,
        evaluate=trap,
        crossover=noop_crossover,
        mutate=noop_mutate,
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
        evaluate_batch=evaluate_batch,
    )


def ga_obj(o, s):
    return 1.0 - nse(o, s)


def nsga_obj(o, s):
    return jnp.asarray([1.0 - nse(o, s), rmse(o, s)])


GR6J_GA_POP, GR6J_GA_GEN, T_GR6J_GA = 16, 20, 0.336849213839
HBV_GA_POP, HBV_GA_GEN, T_HBV_GA = 16, 20, 0.227335446541
GR6J_NS_POP, GR6J_NS_GEN, T_GR6J_NS = 16, 20, 0.256068055938
HBV_NS_POP, HBV_NS_GEN, T_HBV_NS = 24, 30, 0.143225637062


def test_gr6j_ga_converges():
    template, forcing, observed, warmup, spec = _gr6j_problem()
    _best_model, ga_result = calibrate_evolutionary(
        template,
        forcing,
        observed,
        objective_term=ga_obj,
        param_spec=spec,
        simulate=_simulate,
        pop_size=GR6J_GA_POP,
        n_generations=GR6J_GA_GEN,
        seed=SEED,
        warmup=warmup,
    )
    _best_x, best_fit = ga_result.best
    assert best_fit < T_GR6J_GA


def test_gr6j_ga_noop_operators_do_not_converge():
    result = _noop_run(ga, _gr6j_problem(), ga_obj, pop_size=GR6J_GA_POP, n_generations=GR6J_GA_GEN, seed=SEED)
    _best_x, best_fit = result.best
    assert best_fit >= T_GR6J_GA


def test_hbv_ga_converges():
    template, forcing, observed, warmup, spec = _hbv_problem()
    _best_model, ga_result = calibrate_evolutionary(
        template,
        forcing,
        observed,
        objective_term=ga_obj,
        param_spec=spec,
        simulate=_simulate,
        pop_size=HBV_GA_POP,
        n_generations=HBV_GA_GEN,
        seed=SEED,
        warmup=warmup,
    )
    _best_x, best_fit = ga_result.best
    assert best_fit < T_HBV_GA


def test_hbv_ga_noop_operators_do_not_converge():
    result = _noop_run(ga, _hbv_problem(), ga_obj, pop_size=HBV_GA_POP, n_generations=HBV_GA_GEN, seed=SEED)
    _best_x, best_fit = result.best
    assert best_fit >= T_HBV_GA


def test_gr6j_nsga2_converges():
    template, forcing, observed, warmup, spec = _gr6j_problem()
    _front_models, ns_result = calibrate_nsga2(
        template,
        forcing,
        observed,
        objective_term=nsga_obj,
        param_spec=spec,
        simulate=_simulate,
        pop_size=GR6J_NS_POP,
        n_generations=GR6J_NS_GEN,
        seed=SEED,
        warmup=warmup,
    )
    objectives = ns_result.pareto_front.objectives
    assert objectives is not None
    best_primary = float(objectives[:, 0].min())
    assert best_primary < T_GR6J_NS


def test_gr6j_nsga2_noop_operators_do_not_converge():
    result = _noop_run(nsga2, _gr6j_problem(), nsga_obj, pop_size=GR6J_NS_POP, n_generations=GR6J_NS_GEN, seed=SEED)
    best_primary = float(result.pareto_front.objectives[:, 0].min())
    assert best_primary >= T_GR6J_NS


def test_hbv_nsga2_converges():
    template, forcing, observed, warmup, spec = _hbv_problem()
    _front_models, ns_result = calibrate_nsga2(
        template,
        forcing,
        observed,
        objective_term=nsga_obj,
        param_spec=spec,
        simulate=_simulate,
        pop_size=HBV_NS_POP,
        n_generations=HBV_NS_GEN,
        seed=SEED,
        warmup=warmup,
    )
    objectives = ns_result.pareto_front.objectives
    assert objectives is not None
    best_primary = float(objectives[:, 0].min())
    assert best_primary < T_HBV_NS


def test_hbv_nsga2_noop_operators_do_not_converge():
    result = _noop_run(nsga2, _hbv_problem(), nsga_obj, pop_size=HBV_NS_POP, n_generations=HBV_NS_GEN, seed=SEED)
    best_primary = float(result.pareto_front.objectives[:, 0].min())
    assert best_primary >= T_HBV_NS
