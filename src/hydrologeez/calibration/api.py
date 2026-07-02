"""Public evolutionary calibration API built on ctrl-freak's bounded operators."""

from collections.abc import Callable
from typing import NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ctrl_freak.algorithms.ga import ga
from ctrl_freak.algorithms.nsga2 import nsga2
from ctrl_freak.operators.standard import polynomial_mutation, sbx_crossover
from ctrl_freak.results import GAResult, NSGA2Result

from hydrologeez.calibration.adapter import ParamSpec, array_to_model, bounds_array
from hydrologeez.calibration.evolutionary import make_batch_evaluator, make_objective


def _raise_batched_only(_theta: np.ndarray) -> NoReturn:
    raise AssertionError("per-individual evaluate must not run: calibrate_* wires the batched evaluate_batch path")


def _validate_pop_size(pop_size: int) -> None:
    if pop_size <= 0 or pop_size % 2 != 0:
        raise ValueError(f"pop_size must be an even, positive integer (got {pop_size})")


def make_bounded_operators(
    param_spec: ParamSpec,
    *,
    eta_crossover: float = 15.0,
    eta_mutation: float = 20.0,
    mutation_prob: float | None = None,
) -> tuple[
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """Build SBX crossover and polynomial mutation with per-parameter bounds."""
    lo, hi = bounds_array(param_spec)
    bounds = (np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    crossover = sbx_crossover(eta=eta_crossover, bounds=bounds)
    mutate = polynomial_mutation(eta=eta_mutation, prob=mutation_prob, bounds=bounds)
    return crossover, mutate


def _assemble(
    template: eqx.Module,
    forcing,
    observed: jax.Array,
    *,
    objective_term: Callable[[jax.Array, jax.Array], jax.Array],
    param_spec: ParamSpec,
    simulate: Callable[[eqx.Module, object], jax.Array],
    warmup: int,
    eta_crossover: float,
    eta_mutation: float,
    mutation_prob: float | None,
):
    evaluate = make_objective(
        template,
        forcing,
        observed,
        simulate=simulate,
        objective_term=objective_term,
        warmup=warmup,
        param_spec=param_spec,
    )
    evaluate_batch = make_batch_evaluator(evaluate)
    crossover, mutate = make_bounded_operators(
        param_spec,
        eta_crossover=eta_crossover,
        eta_mutation=eta_mutation,
        mutation_prob=mutation_prob,
    )
    lo_arr, hi_arr = bounds_array(param_spec)
    lo = np.asarray(lo_arr, dtype=float)
    hi = np.asarray(hi_arr, dtype=float)

    def init(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(lo, hi)

    return evaluate_batch, init, crossover, mutate


def calibrate_evolutionary(
    template: eqx.Module,
    forcing,
    observed: jax.Array,
    *,
    objective_term: Callable[[jax.Array, jax.Array], jax.Array],
    param_spec: ParamSpec,
    simulate: Callable[[eqx.Module, object], jax.Array] = lambda m, f: m.run(f),
    pop_size: int = 16,
    n_generations: int = 20,
    seed: int | None = None,
    warmup: int = 365,
    eta_crossover: float = 15.0,
    eta_mutation: float = 20.0,
    mutation_prob: float | None = None,
    select: str = "tournament",
    survive: str = "elitist",
) -> tuple[eqx.Module, GAResult]:
    """Calibrate one model with a bounded single-objective genetic algorithm."""
    _validate_pop_size(pop_size)
    evaluate_batch, init, crossover, mutate = _assemble(
        template,
        forcing,
        observed,
        objective_term=objective_term,
        param_spec=param_spec,
        simulate=simulate,
        warmup=warmup,
        eta_crossover=eta_crossover,
        eta_mutation=eta_mutation,
        mutation_prob=mutation_prob,
    )
    result = ga(
        init=init,
        evaluate=_raise_batched_only,
        crossover=crossover,
        mutate=mutate,
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
        select=select,
        survive=survive,
        evaluate_batch=evaluate_batch,
    )
    best_x, _best_fit = result.best
    best_model = array_to_model(template, jnp.asarray(best_x, dtype=jnp.float64), param_spec)
    return best_model, result


def calibrate_nsga2(
    template: eqx.Module,
    forcing,
    observed: jax.Array,
    *,
    objective_term: Callable[[jax.Array, jax.Array], jax.Array],
    param_spec: ParamSpec,
    simulate: Callable[[eqx.Module, object], jax.Array] = lambda m, f: m.run(f),
    pop_size: int = 16,
    n_generations: int = 20,
    seed: int | None = None,
    warmup: int = 365,
    eta_crossover: float = 15.0,
    eta_mutation: float = 20.0,
    mutation_prob: float | None = None,
    select: str = "crowded",
    survive: str = "nsga2",
) -> tuple[list[eqx.Module], NSGA2Result]:
    """Calibrate a Pareto front with bounded NSGA-II.

    ``objective_term`` must return a real ``(n_obj,)`` vector so the batched evaluator
    forwards an exact ``(n, n_obj)`` matrix to NSGA-II. The returned front can contain
    one model when near-colinear objectives collapse the rank-0 front to a single point.
    """
    _validate_pop_size(pop_size)
    evaluate_batch, init, crossover, mutate = _assemble(
        template,
        forcing,
        observed,
        objective_term=objective_term,
        param_spec=param_spec,
        simulate=simulate,
        warmup=warmup,
        eta_crossover=eta_crossover,
        eta_mutation=eta_mutation,
        mutation_prob=mutation_prob,
    )
    result = nsga2(
        init=init,
        evaluate=_raise_batched_only,
        crossover=crossover,
        mutate=mutate,
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
        select=select,
        survive=survive,
        evaluate_batch=evaluate_batch,
    )
    front = result.pareto_front
    front_models = [array_to_model(template, jnp.asarray(x, dtype=jnp.float64), param_spec) for x in front.x]
    return front_models, result
