"""Bounded gradient calibration with ``torch.optim``."""

import math
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

from hydrologeez.calibration.adapter import (
    GR6J_SPEC,
    ParamSpec,
    array_to_model,
    array_to_parameters,
    bounds_array,
    params_to_array,
)
from hydrologeez.ssm import StateSpaceModel

OptimizerFactory = Callable[[list[nn.Parameter]], torch.optim.Optimizer]

_EPS = 1e-6


def _to_unconstrained(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    z = torch.clamp((x - lo) / (hi - lo), _EPS, 1.0 - _EPS)
    return torch.log(z) - torch.log1p(-z)


def _to_bounded(u: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(u)


def _make_optimizer(
    optimizer: type[torch.optim.Optimizer] | OptimizerFactory | None,
    parameter: nn.Parameter,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if optimizer is None:
        result = torch.optim.Adam([parameter], lr=learning_rate)
    elif isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        constructor = cast(Any, optimizer)
        result = constructor([parameter], lr=learning_rate)
    else:
        result = optimizer([parameter])
    if not isinstance(result, torch.optim.Optimizer):
        raise TypeError("optimizer factory must return a torch.optim.Optimizer")
    bound = [item for group in result.param_groups for item in group["params"]]
    if len(bound) != 1 or bound[0] is not parameter:
        raise ValueError("optimizer must be bound to exactly the calibration parameter")
    return result


def calibrate_gradient(
    template: StateSpaceModel,
    forcing: object,
    observed: torch.Tensor,
    *,
    loss_term: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    warmup: object | None = None,
    n_steps: int = 200,
    learning_rate: float = 5e-2,
    optimizer: type[torch.optim.Optimizer] | OptimizerFactory | None = None,
    param_spec: ParamSpec | None = None,
) -> tuple[StateSpaceModel, torch.Tensor]:
    """Minimize ``loss_term(observed, simulated)`` within code bounds.

    ``optimizer`` may be an optimizer class, which receives ``learning_rate``, or
    a configured factory, which owns its hyperparameters. It must not be an
    optimizer instance already bound to other tensors.
    """
    if not isinstance(template, nn.Module) or not callable(getattr(template, "run", None)):
        raise TypeError("template must be a torch.nn.Module with a callable run method")
    if not isinstance(observed, torch.Tensor):
        raise TypeError("observed must be a torch.Tensor")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")

    spec = param_spec if param_spec is not None else GR6J_SPEC
    x0 = params_to_array(template, spec)
    if x0.ndim != 1:
        raise ValueError("selected starting parameters must be one-dimensional")
    if observed.dtype != x0.dtype or observed.device != x0.device:
        raise ValueError("observations and selected parameters must share dtype and device")
    lo, hi = bounds_array(spec, dtype=x0.dtype, device=x0.device)
    u = nn.Parameter(_to_unconstrained(x0, lo, hi).detach().clone())
    opt = _make_optimizer(optimizer, u, learning_rate)

    losses: list[torch.Tensor] = []
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        theta = _to_bounded(u, lo, hi)
        parameters = array_to_parameters(theta, spec)
        simulated = template.run(
            forcing,
            parameters=parameters,
            warmup=warmup,
            warmup_parameters=parameters if warmup is not None else None,
        )
        loss = loss_term(observed, simulated)
        if loss.ndim != 0:
            raise ValueError("loss must be a scalar")
        if not bool(torch.isfinite(loss).item()):
            raise ValueError("loss must be finite")
        losses.append(loss.detach())
        loss.backward()
        opt.step()

    final_theta = _to_bounded(u.detach(), lo, hi)
    calibrated = cast(StateSpaceModel, array_to_model(template, final_theta, spec))
    return calibrated, torch.stack(losses).detach()
