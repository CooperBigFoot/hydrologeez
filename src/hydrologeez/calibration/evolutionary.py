"""Derivative-free calibration through one batched, no-grad Torch model run."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from hydrologeez.calibration.adapter import GR6J_SPEC, ParamSpec, array_to_parameters

ObjectiveKind = Literal["ga", "nsga2"]


def _template_dtype_device(template: nn.Module) -> tuple[torch.dtype, torch.device]:
    try:
        parameter = next(template.parameters())
    except StopIteration as error:
        raise ValueError("template must have at least one registered parameter") from error
    return parameter.dtype, parameter.device


def _expand_one_basin(value: Any, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim < 2:
            raise ValueError("tensor leaves must have leading [batch, time] dimensions")
        if value.shape[0] != 1:
            raise ValueError(f"source forcing batch must be 1 (got {value.shape[0]})")
        return value.expand(batch_size, *value.shape[1:])
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _expand_one_basin(getattr(value, field.name), batch_size)
                for field in dataclasses.fields(value)
            }
        )
    if isinstance(value, dict):
        return type(value)((key, _expand_one_basin(item, batch_size)) for key, item in value.items())
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_expand_one_basin(item, batch_size) for item in value))
    if isinstance(value, tuple):
        return tuple(_expand_one_basin(item, batch_size) for item in value)
    if isinstance(value, list):
        return [_expand_one_basin(item, batch_size) for item in value]
    raise TypeError(f"forcing contains unsupported non-tensor leaf {type(value).__name__}")


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    leaves: list[torch.Tensor] = []

    def collect(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            leaves.append(item)
            return item
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field in dataclasses.fields(item):
                collect(getattr(item, field.name))
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)
        else:
            raise TypeError(f"forcing contains unsupported non-tensor leaf {type(item).__name__}")
        return item

    collect(value)
    return leaves


def make_objective(
    template: nn.Module,
    forcing: Any,
    observed: torch.Tensor,
    *,
    simulate: Callable[[nn.Module, Any, dict[str, torch.Tensor]], torch.Tensor],
    objective_term: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    warmup: int = 365,
    param_spec: ParamSpec | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a population objective over ``theta[B,P]``.

    ``objective_term`` receives observed and simulated tensors shaped ``[B,T_eval]``
    and returns ``[B]`` for GA or ``[B,n_obj]`` for NSGA-II. Lower is better.
    """
    spec = param_spec if param_spec is not None else GR6J_SPEC
    dtype, device = _template_dtype_device(template)
    if not isinstance(observed, torch.Tensor):
        raise TypeError("observed must be a torch.Tensor")
    if observed.ndim != 2 or observed.shape[0] != 1:
        raise ValueError(f"observed must have shape [1,T] (got {tuple(observed.shape)})")
    if observed.dtype != dtype or observed.device != device:
        raise ValueError("observed dtype and device must match the template")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0 or warmup >= observed.shape[1]:
        raise ValueError(f"warmup must be a non-negative integer smaller than {observed.shape[1]}")
    _expand_one_basin(forcing, 1)
    leaves = _tensor_leaves(forcing)
    if not leaves:
        raise ValueError("forcing must contain at least one tensor")
    if any(leaf.dtype != dtype or leaf.device != device for leaf in leaves):
        raise ValueError("forcing dtype and device must match the template")
    if any(leaf.shape[1] != observed.shape[1] for leaf in leaves):
        raise ValueError("forcing and observed must have the same time length")

    def evaluate(theta: torch.Tensor) -> torch.Tensor:
        if not isinstance(theta, torch.Tensor):
            raise TypeError("theta must be a torch.Tensor")
        if theta.ndim != 2 or theta.shape[1] != len(spec.names):
            raise ValueError(f"theta must have shape [B,{len(spec.names)}] (got {tuple(theta.shape)})")
        if theta.dtype != dtype or theta.device != device:
            raise ValueError("theta dtype and device must match the template")
        batch_size = theta.shape[0]
        population_forcing = _expand_one_basin(forcing, batch_size)
        obs_eval = observed.expand(batch_size, -1)[:, warmup:]
        parameters = array_to_parameters(theta, spec)
        simulated = simulate(template, population_forcing, parameters)
        if not isinstance(simulated, torch.Tensor) or simulated.shape != observed.expand(batch_size, -1).shape:
            actual = getattr(simulated, "shape", type(simulated).__name__)
            raise ValueError(f"simulation must have shape {(batch_size, observed.shape[1])} (got {actual})")
        objectives = objective_term(obs_eval, simulated[:, warmup:])
        if not isinstance(objectives, torch.Tensor):
            raise TypeError("objective_term must return a torch.Tensor")
        if objectives.device != device:
            raise ValueError("objective tensor must be on the evaluator device")
        if objectives.ndim == 0 or objectives.shape[0] != batch_size:
            raise ValueError(
                f"objective first dimension must be population size {batch_size} (got {tuple(objectives.shape)})"
            )
        if not objectives.is_floating_point() or not torch.isfinite(objectives).all():
            raise ValueError("objectives must be finite real-valued tensors")
        return objectives

    return evaluate


def make_batch_evaluator(
    evaluate: Callable[[torch.Tensor], torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
    objective_kind: ObjectiveKind,
) -> Callable[[np.ndarray], np.ndarray]:
    """Bridge a NumPy population to a validated batched Torch objective."""

    def evaluate_batch(pop_matrix: np.ndarray) -> np.ndarray:
        theta = torch.as_tensor(pop_matrix, dtype=dtype, device=device)
        with torch.no_grad():
            objectives = evaluate(theta)
        shape = tuple(objectives.shape)
        if objective_kind == "ga":
            if objectives.ndim == 2 and objectives.shape[1] == 1:
                objectives = objectives[:, 0]
            elif objectives.ndim != 1:
                raise ValueError(f"GA objective must have shape [B] or [B,1] (got {shape})")
        elif objectives.ndim != 2 or objectives.shape[1] < 2:
            raise ValueError(f"NSGA-II objective must have shape [B,n_obj>=2] (got {shape})")
        result = objectives.detach().cpu().numpy()
        if not np.issubdtype(result.dtype, np.number) or not np.isrealobj(result):
            raise TypeError("objective output must be a real-valued numeric NumPy array")
        return result

    return evaluate_batch
