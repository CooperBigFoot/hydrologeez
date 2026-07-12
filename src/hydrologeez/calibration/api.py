"""Public evolutionary calibration API built on ctrl-freak's bounded operators."""

from collections.abc import Callable
from typing import Any, NoReturn

import numpy as np
import torch
from ctrl_freak.algorithms.ga import ga
from ctrl_freak.algorithms.nsga2 import nsga2
from ctrl_freak.operators.standard import polynomial_mutation, sbx_crossover
from ctrl_freak.results import GAResult, NSGA2Result
from torch import nn

from hydrologeez.calibration.adapter import ParamSpec, array_to_model, bounds_array
from hydrologeez.calibration.evolutionary import ObjectiveKind, make_batch_evaluator, make_objective


def _raise_batched_only(_theta: np.ndarray) -> NoReturn:
    raise AssertionError("per-individual evaluate must not run: calibrate_* wires the batched evaluate_batch path")


def _validate_pop_size(pop_size: int) -> None:
    if pop_size <= 0 or pop_size % 2 != 0:
        raise ValueError(f"pop_size must be an even, positive integer (got {pop_size})")


def _dtype_device(template: nn.Module) -> tuple[torch.dtype, torch.device]:
    try:
        parameter = next(template.parameters())
    except StopIteration as error:
        raise ValueError("template must have at least one registered parameter") from error
    return parameter.dtype, parameter.device


def make_bounded_operators(
    param_spec: ParamSpec,
    *,
    eta_crossover: float = 15.0,
    eta_mutation: float = 20.0,
    mutation_prob: float | None = None,
) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Build SBX crossover and polynomial mutation with per-parameter bounds."""
    lo, hi = bounds_array(param_spec)
    bounds = (lo.detach().cpu().numpy(), hi.detach().cpu().numpy())
    return (
        sbx_crossover(eta=eta_crossover, bounds=bounds),
        polynomial_mutation(eta=eta_mutation, prob=mutation_prob, bounds=bounds),
    )


def _assemble(
    template: nn.Module,
    forcing: Any,
    observed: torch.Tensor,
    *,
    objective_term: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    objective_kind: ObjectiveKind,
    param_spec: ParamSpec,
    simulate: Callable[[nn.Module, Any, dict[str, torch.Tensor]], torch.Tensor],
    warmup: int,
    eta_crossover: float,
    eta_mutation: float,
    mutation_prob: float | None,
):
    dtype, device = _dtype_device(template)
    evaluate = make_objective(
        template,
        forcing,
        observed,
        simulate=simulate,
        objective_term=objective_term,
        warmup=warmup,
        param_spec=param_spec,
    )
    evaluate_batch = make_batch_evaluator(evaluate, dtype=dtype, device=device, objective_kind=objective_kind)
    crossover, mutate = make_bounded_operators(
        param_spec, eta_crossover=eta_crossover, eta_mutation=eta_mutation, mutation_prob=mutation_prob
    )
    lo_tensor, hi_tensor = bounds_array(param_spec, dtype=dtype, device=device)
    lo, hi = lo_tensor.detach().cpu().numpy(), hi_tensor.detach().cpu().numpy()

    def init(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(lo, hi)

    return evaluate_batch, init, crossover, mutate, dtype, device


def calibrate_evolutionary(
    template: nn.Module,
    forcing: Any,
    observed: torch.Tensor,
    *,
    objective_term: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    param_spec: ParamSpec,
    simulate: Callable[[nn.Module, Any, dict[str, torch.Tensor]], torch.Tensor] = lambda m, f, p: m.run(f, p),
    pop_size: int = 16,
    n_generations: int = 20,
    seed: int | None = None,
    warmup: int = 365,
    eta_crossover: float = 15.0,
    eta_mutation: float = 20.0,
    mutation_prob: float | None = None,
    select: str = "tournament",
    survive: str = "elitist",
) -> tuple[nn.Module, GAResult]:
    """Calibrate one model with GA; custom ``simulate`` must accept explicit parameters."""
    _validate_pop_size(pop_size)
    evaluate_batch, init, crossover, mutate, dtype, device = _assemble(
        template,
        forcing,
        observed,
        objective_term=objective_term,
        objective_kind="ga",
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
    best_x, _ = result.best
    best = torch.as_tensor(best_x, dtype=dtype, device=device)
    return array_to_model(template, best, param_spec), result


def calibrate_nsga2(
    template: nn.Module,
    forcing: Any,
    observed: torch.Tensor,
    *,
    objective_term: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    param_spec: ParamSpec,
    simulate: Callable[[nn.Module, Any, dict[str, torch.Tensor]], torch.Tensor] = lambda m, f, p: m.run(f, p),
    pop_size: int = 16,
    n_generations: int = 20,
    seed: int | None = None,
    warmup: int = 365,
    eta_crossover: float = 15.0,
    eta_mutation: float = 20.0,
    mutation_prob: float | None = None,
    select: str = "crowded",
    survive: str = "nsga2",
) -> tuple[list[nn.Module], NSGA2Result]:
    """Calibrate a Pareto front; custom ``simulate`` must accept explicit parameters."""
    _validate_pop_size(pop_size)
    evaluate_batch, init, crossover, mutate, dtype, device = _assemble(
        template,
        forcing,
        observed,
        objective_term=objective_term,
        objective_kind="nsga2",
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
    models = [
        array_to_model(template, torch.as_tensor(x, dtype=dtype, device=device), param_spec)
        for x in result.pareto_front.x
    ]
    return models, result
