from __future__ import annotations

import copy
from typing import cast

import numpy as np
import pytest
import torch
from ctrl_freak.results import GAResult, NSGA2Result

from hydrologeez.calibration.adapter import ParamSpec, params_to_array
from hydrologeez.calibration.api import calibrate_evolutionary, calibrate_nsga2
from hydrologeez.calibration.evolutionary import make_batch_evaluator, make_objective
from hydrologeez.models.gr6j.model import GR6J, GR6JForcing

DTYPE = torch.float64
POP_SIZE = 10
GENERATIONS = 6
WARMUP = 4


def _model(values: tuple[float, ...]) -> GR6J:
    tensors = tuple(torch.tensor(value, dtype=DTYPE) for value in values)
    return GR6J(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5])


@pytest.fixture
def problem() -> tuple[GR6JForcing, torch.Tensor, GR6J]:
    time = torch.arange(48, dtype=DTYPE)
    forcing = GR6JForcing(
        precip=(2.0 + 12.0 * ((time % 7) < 2) + 5.0 * ((time % 13) == 0)).to(DTYPE)[None, :],
        pet=(1.5 + 0.8 * torch.sin(time / 6.0))[None, :],
    )
    truth = _model((450.0, 1.2, 120.0, 2.2, -0.5, 6.0))
    template = _model((2200.0, -4.0, 850.0, 8.5, 3.5, 42.0))
    with torch.no_grad():
        observed = truth.run(forcing)
    return forcing, observed, template


def _mse(observed: torch.Tensor, simulated: torch.Tensor) -> torch.Tensor:
    return torch.mean((simulated - observed) ** 2, dim=-1)


def _vector_loss(observed: torch.Tensor, simulated: torch.Tensor) -> torch.Tensor:
    error = simulated - observed
    return torch.stack((torch.mean(error**2, dim=-1), torch.mean(torch.abs(error), dim=-1)), dim=-1)


def _losses(model: GR6J, forcing: GR6JForcing, observed: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return _vector_loss(observed[:, WARMUP:], model.run(forcing)[:, WARMUP:])[0]


def _assert_bounds(model: torch.nn.Module) -> None:
    spec = ParamSpec.from_model(model)
    theta = params_to_array(model, spec)
    lower = torch.tensor(spec.lower, dtype=DTYPE)
    upper = torch.tensor(spec.upper, dtype=DTYPE)
    assert bool(torch.all((theta >= lower) & (theta <= upper)))


def test_ga_real_batched_path_improves(problem: tuple[GR6JForcing, torch.Tensor, GR6J]) -> None:
    forcing, observed, template = problem
    calls: list[tuple[int, dict[str, torch.Size], bool]] = []

    def simulate(model, population_forcing, parameters):
        calls.append(
            (
                population_forcing.precip.shape[0],
                {key: value.shape for key, value in parameters.items()},
                torch.is_grad_enabled(),
            )
        )
        return model.run(population_forcing, parameters)

    calibrated, result = calibrate_evolutionary(
        template,
        forcing,
        observed,
        objective_term=_mse,
        simulate=simulate,
        pop_size=POP_SIZE,
        n_generations=GENERATIONS,
        seed=7,
        warmup=WARMUP,
    )
    assert isinstance(result, GAResult)
    assert result.evaluations == POP_SIZE * (GENERATIONS + 1)
    assert np.isfinite(result.fitness).all()
    assert len(calls) == GENERATIONS + 1
    assert all(batch == POP_SIZE and not grad for batch, _, grad in calls)
    assert all(all(shape == torch.Size([POP_SIZE]) for shape in shapes.values()) for _, shapes, _ in calls)
    _assert_bounds(calibrated)
    assert _losses(cast(GR6J, calibrated), forcing, observed)[0] < _losses(template, forcing, observed)[0]


def test_nsga2_real_batched_path_dominates_template(problem: tuple[GR6JForcing, torch.Tensor, GR6J]) -> None:
    forcing, observed, template = problem
    calls = 0

    def simulate(model, population_forcing, parameters):
        nonlocal calls
        calls += 1
        assert population_forcing.precip.shape[0] == POP_SIZE
        assert all(value.shape == (POP_SIZE,) for value in parameters.values())
        assert not torch.is_grad_enabled()
        return model.run(population_forcing, parameters)

    models, result = calibrate_nsga2(
        template,
        forcing,
        observed,
        objective_term=_vector_loss,
        simulate=simulate,
        pop_size=POP_SIZE,
        n_generations=GENERATIONS,
        seed=11,
        warmup=WARMUP,
    )
    front = result.pareto_front
    assert isinstance(result, NSGA2Result)
    assert result.evaluations == POP_SIZE * (GENERATIONS + 1)
    assert result.population.objectives is not None
    assert result.population.objectives.shape == (POP_SIZE, 2)
    assert np.isfinite(result.population.objectives).all()
    assert len(front) > 0 and len(models) == len(front)
    assert calls == GENERATIONS + 1
    for model in models:
        _assert_bounds(model)
    baseline = _losses(template, forcing, observed)
    assert any(
        bool(torch.all(loss <= baseline) and torch.any(loss < baseline))
        for loss in (_losses(cast(GR6J, m), forcing, observed) for m in models)
    )


def test_numpy_boundary_shapes_and_template_immutability(problem: tuple[GR6JForcing, torch.Tensor, GR6J]) -> None:
    forcing, observed, template = problem
    spec = ParamSpec.from_model(template)
    before = copy.deepcopy(template.state_dict())
    calls = 0

    def simulate(model, population_forcing, parameters):
        nonlocal calls
        calls += 1
        assert not torch.is_grad_enabled()
        return model.run(population_forcing, parameters)

    candidates = np.array([spec.lower, spec.upper, (450, 2.2, 1.2, 120, -0.5, 6)], dtype=float)
    scalar = make_objective(template, forcing, observed, simulate=simulate, objective_term=_mse, warmup=WARMUP)
    scalar_batch = make_batch_evaluator(scalar, dtype=DTYPE, device=torch.device("cpu"), objective_kind="ga")
    output = scalar_batch(candidates)
    assert output.shape == (3,) and np.issubdtype(output.dtype, np.number)
    assert calls == 1
    vector = make_objective(template, forcing, observed, simulate=simulate, objective_term=_vector_loss, warmup=WARMUP)
    vector_output = make_batch_evaluator(vector, dtype=DTYPE, device=torch.device("cpu"), objective_kind="nsga2")(
        candidates
    )
    assert vector_output.shape == (3, 2) and calls == 2
    for name, value in template.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_invalid_inputs_and_objective_shapes(problem: tuple[GR6JForcing, torch.Tensor, GR6J]) -> None:
    forcing, observed, template = problem

    def simulate(model, batch, parameters):
        return model.run(batch, parameters)

    with pytest.raises(ValueError, match="warmup"):
        make_objective(template, forcing, observed, simulate=simulate, objective_term=_mse, warmup=48)
    bad_forcing = GR6JForcing(forcing.precip.expand(2, -1), forcing.pet.expand(2, -1))
    with pytest.raises(ValueError, match="source forcing batch"):
        make_objective(template, bad_forcing, observed, simulate=simulate, objective_term=_mse, warmup=0)
    evaluate = make_objective(template, forcing, observed, simulate=simulate, objective_term=_mse, warmup=0)
    with pytest.raises(ValueError, match="theta must have shape"):
        evaluate(torch.zeros((3, 5), dtype=DTYPE))
    candidates = np.tile(np.array((450.0, 2.2, 1.2, 120.0, -0.5, 6.0)), (3, 1))
    ga_bad = make_objective(
        template,
        forcing,
        observed,
        simulate=simulate,
        objective_term=lambda o, s: torch.stack((_mse(o, s), _mse(o, s)), dim=-1),
        warmup=0,
    )
    with pytest.raises(ValueError, match="GA objective"):
        make_batch_evaluator(ga_bad, dtype=DTYPE, device=torch.device("cpu"), objective_kind="ga")(candidates)
    with pytest.raises(ValueError, match="NSGA-II objective"):
        make_batch_evaluator(evaluate, dtype=DTYPE, device=torch.device("cpu"), objective_kind="nsga2")(candidates)
