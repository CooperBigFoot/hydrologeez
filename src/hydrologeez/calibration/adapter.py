"""Bridge hydrological parameter specifications to PyTorch modules and tensors.

Two views are provided. ``params_to_array`` and ``array_to_parameters`` use an
ordered model-derived ``ParamSpec``; the latter is the differentiable functional view.
``array_to_model`` is instead an independent module-reconstruction view.
``model_to_flat`` and ``flat_to_model`` generically flatten and reconstruct all
registered parameters in registration order.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ParamSpec:
    """Ordered parameter names plus per-name lower/upper calibration bounds.

    ``names``/``lower``/``upper`` are positionally aligned: ``lower[i]``/``upper[i]``
    are the calibration bounds for the parameter ``names[i]``.
    """

    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @classmethod
    def from_model(cls, model: Any) -> ParamSpec:
        """Build a validated specification from a module's ordered parameter bounds."""
        module = _require_module(model, "model")
        parameter_bounds = getattr(module, "parameter_bounds", None)
        if not isinstance(parameter_bounds, Mapping):
            raise TypeError("model.parameter_bounds must be a mapping")
        names = tuple(parameter_bounds)
        if not all(isinstance(name, str) for name in names):
            raise TypeError("model.parameter_bounds keys must be strings")
        bounds = tuple(parameter_bounds.values())
        spec = cls(
            names=names,
            lower=tuple(float(bound[0]) for bound in bounds),
            upper=tuple(float(bound[1]) for bound in bounds),
        )
        _validate_spec(spec)
        return spec


@dataclass(frozen=True)
class _ParameterRecord:
    name: str
    shape: torch.Size
    numel: int
    requires_grad: bool


@dataclass(frozen=True)
class _FlatAux:
    template: nn.Module
    records: tuple[_ParameterRecord, ...]
    dtype: torch.dtype
    device: torch.device


def _validate_spec(spec: ParamSpec) -> None:
    size = len(spec.names)
    if size == 0 or len(spec.lower) != size or len(spec.upper) != size:
        raise ValueError("spec names, lower, and upper must have equal nonzero lengths")
    if len(set(spec.names)) != size:
        raise ValueError("spec parameter names must be unique")
    if any(
        not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper
        for lower, upper in zip(spec.lower, spec.upper, strict=True)
    ):
        raise ValueError("spec bounds must be finite with lower strictly below upper")


def _require_module(value: Any, label: str) -> nn.Module:
    if not isinstance(value, nn.Module):
        raise TypeError(f"{label} must be a torch.nn.Module")
    return value


def _validate_theta(theta: torch.Tensor, spec: ParamSpec) -> None:
    _validate_spec(spec)
    if not isinstance(theta, torch.Tensor):
        raise TypeError("theta must be a torch.Tensor")
    if theta.ndim == 0:
        raise ValueError("theta must have at least one dimension")
    if theta.shape[-1] != len(spec.names):
        raise ValueError(f"theta final dimension must have length {len(spec.names)}")


def _install_parameter(module: nn.Module, name: str, parameter: nn.Parameter) -> None:
    parts = name.split(".")
    target = module
    for part in parts[:-1]:
        child = target.get_submodule(part)
        target = child
    setattr(target, parts[-1], parameter)


def bounds_array(
    spec: ParamSpec,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Any:
    """Return fresh lower and upper tensors in specification order."""
    _validate_spec(spec)
    return torch.tensor(spec.lower, dtype=dtype, device=device), torch.tensor(spec.upper, dtype=dtype, device=device)


def params_to_array(model: Any, spec: ParamSpec | None = None) -> Any:
    """Stack exact top-level registered parameters in specification order."""
    module = _require_module(model, "model")
    resolved_spec = ParamSpec.from_model(module) if spec is None else spec
    _validate_spec(resolved_spec)
    if any("." in name for name in resolved_spec.names):
        raise ValueError("params_to_array only accepts top-level parameter names")
    registered = dict(module.named_parameters())
    missing = [name for name in resolved_spec.names if name not in registered]
    if missing:
        raise ValueError(f"missing registered parameters: {', '.join(missing)}")
    parameters: list[torch.Tensor] = [registered[name] for name in resolved_spec.names]
    first = parameters[0]
    if any(parameter.shape != first.shape for parameter in parameters[1:]):
        raise ValueError("selected parameters must have identical shapes")
    if any(parameter.dtype != first.dtype for parameter in parameters[1:]):
        raise ValueError("selected parameters must have identical dtypes")
    if any(parameter.device != first.device for parameter in parameters[1:]):
        raise ValueError("selected parameters must be on the same device")
    return torch.stack(parameters, dim=-1)


def array_to_parameters(
    theta: torch.Tensor,
    spec: ParamSpec,
) -> dict[str, torch.Tensor]:
    """Return differentiable named tensor views of the final parameter axis."""
    _validate_theta(theta, spec)
    return {name: theta[..., index] for index, name in enumerate(spec.names)}


def array_to_model(template: Any, theta: Any, spec: ParamSpec | None = None) -> Any:
    """Reconstruct a module with independent registered parameters."""
    module = _require_module(template, "template")
    resolved_spec = ParamSpec.from_model(module) if spec is None else spec
    _validate_theta(theta, resolved_spec)
    if any("." in name for name in resolved_spec.names):
        raise ValueError("array_to_model only accepts top-level parameter names")
    registered = dict(module.named_parameters())
    missing = [name for name in resolved_spec.names if name not in registered]
    if missing:
        raise ValueError(f"missing registered parameters: {', '.join(missing)}")
    rebuilt = copy.deepcopy(module)
    for index, name in enumerate(resolved_spec.names):
        replacement = nn.Parameter(theta[..., index].detach().clone(), requires_grad=registered[name].requires_grad)
        setattr(rebuilt, name, replacement)
    return rebuilt


def model_to_flat(model: Any) -> tuple[torch.Tensor, Any]:
    """Flatten all registered parameters in registration order without detaching."""
    module = _require_module(model, "model")
    named = list(module.named_parameters())
    if not named:
        raise ValueError("model must have at least one registered parameter")
    first = named[0][1]
    if any(parameter.dtype != first.dtype for _, parameter in named[1:]):
        raise ValueError("registered parameters must have identical dtypes")
    if any(parameter.device != first.device for _, parameter in named[1:]):
        raise ValueError("registered parameters must be on the same device")
    records = tuple(
        _ParameterRecord(name, parameter.shape, parameter.numel(), parameter.requires_grad) for name, parameter in named
    )
    flat = torch.cat([parameter.reshape(-1) for _, parameter in named])
    aux = _FlatAux(copy.deepcopy(module), records, first.dtype, first.device)
    return flat, aux


def flat_to_model(flat: torch.Tensor, aux: Any) -> Any:
    """Reconstruct an independent module from generic flatten metadata."""
    if not isinstance(flat, torch.Tensor):
        raise TypeError("flat must be a torch.Tensor")
    if not isinstance(aux, _FlatAux):
        raise TypeError("aux must be metadata returned by model_to_flat")
    if flat.ndim != 1:
        raise ValueError("flat must be one-dimensional")
    total = sum(record.numel for record in aux.records)
    if flat.numel() != total:
        raise ValueError(f"flat must contain exactly {total} values")
    if flat.dtype != aux.dtype:
        raise ValueError("flat dtype does not match reconstruction metadata")
    if flat.device != aux.device:
        raise ValueError("flat device does not match reconstruction metadata")
    rebuilt = copy.deepcopy(aux.template)
    offset = 0
    for record in aux.records:
        value = flat[offset : offset + record.numel].reshape(record.shape).detach().clone()
        _install_parameter(rebuilt, record.name, nn.Parameter(value, requires_grad=record.requires_grad))
        offset += record.numel
    return rebuilt
