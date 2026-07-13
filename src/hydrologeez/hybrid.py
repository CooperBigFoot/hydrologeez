"""Pure-Torch primitives for neural-parameterized state-space models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def _ordered_names(names: Sequence[str], *, quadrant: str) -> tuple[str, ...]:
    ordered = tuple(names)
    seen: set[str] = set()
    for name in ordered:
        if name in seen:
            raise ValueError(f"{quadrant} input names contain duplicate {name!r}")
        seen.add(name)
    return ordered


def _resolve_indices(
    available: tuple[str, ...],
    requested: Sequence[str | int],
    *,
    quadrant: str,
) -> list[int]:
    indices: list[int] = []
    for selector in requested:
        if isinstance(selector, str):
            try:
                index = available.index(selector)
            except ValueError as error:
                raise ValueError(f"unknown {quadrant} feature {selector!r}") from error
        elif type(selector) is int:
            index = selector
            if index < 0 or index >= len(available):
                raise ValueError(f"{quadrant} feature index {index} is outside [0, {len(available)})")
        else:
            raise TypeError(f"{quadrant} selectors must be names or integer indices")
        if index in indices:
            raise ValueError(f"{quadrant} selectors resolve to duplicate index {index}")
        indices.append(index)
    return indices


def select_features(
    scalar_dynamic: torch.Tensor | None,
    scalar_static: torch.Tensor | None,
    *,
    dynamic_inputs: Sequence[str],
    static_inputs: Sequence[str],
    dynamic_features: Sequence[str | int] = (),
    static_features: Sequence[str | int] = (),
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Select named or indexed scalar feature columns in requested order."""
    dynamic_names = _ordered_names(dynamic_inputs, quadrant="scalar_dynamic")
    static_names = _ordered_names(static_inputs, quadrant="scalar_static")
    dynamic_indices = _resolve_indices(dynamic_names, dynamic_features, quadrant="scalar_dynamic")
    static_indices = _resolve_indices(static_names, static_features, quadrant="scalar_static")

    if scalar_dynamic is not None:
        if scalar_dynamic.ndim != 3:
            raise ValueError("scalar_dynamic must have shape [B, T, F_dynamic]")
        if scalar_dynamic.shape[-1] != len(dynamic_names):
            raise ValueError("scalar_dynamic width does not match the ordered dynamic input names")
    elif dynamic_indices:
        raise ValueError("scalar_dynamic is required by the requested dynamic features")

    if scalar_static is not None:
        if scalar_static.ndim != 2:
            raise ValueError("scalar_static must have shape [B, F_static]")
        if scalar_static.shape[-1] != len(static_names):
            raise ValueError("scalar_static width does not match the ordered static input names")
    elif static_indices:
        raise ValueError("scalar_static is required by the requested static features")

    if scalar_dynamic is not None and scalar_static is not None and scalar_dynamic.shape[0] != scalar_static.shape[0]:
        raise ValueError("scalar_dynamic and scalar_static batch sizes must match")

    selected_dynamic = scalar_dynamic[..., dynamic_indices] if scalar_dynamic is not None and dynamic_indices else None
    selected_static = scalar_static[..., static_indices] if scalar_static is not None and static_indices else None
    return selected_dynamic, selected_static


def select_network_inputs(
    scalar_dynamic: torch.Tensor | None,
    scalar_static: torch.Tensor | None,
    *,
    dynamic_inputs: Sequence[str],
    static_inputs: Sequence[str],
    dynamic_features: Sequence[str | int] = (),
    static_features: Sequence[str | int] = (),
) -> torch.Tensor:
    """Select network features and return them as one time-aligned tensor."""
    selected_dynamic, selected_static = select_features(
        scalar_dynamic,
        scalar_static,
        dynamic_inputs=dynamic_inputs,
        static_inputs=static_inputs,
        dynamic_features=dynamic_features,
        static_features=static_features,
    )
    if selected_dynamic is None and selected_static is None:
        raise ValueError("at least one network input feature must be requested")
    if selected_dynamic is None:
        if scalar_dynamic is None:
            raise ValueError("scalar_dynamic is required to define the network input time dimension")
        assert selected_static is not None
        return selected_static.unsqueeze(1).expand(-1, scalar_dynamic.shape[1], -1)
    if selected_static is None:
        return selected_dynamic
    expanded_static = selected_static.unsqueeze(1).expand(-1, selected_dynamic.shape[1], -1)
    return torch.cat((selected_dynamic, expanded_static), dim=-1)


def bounded_parameters(
    raw: torch.Tensor,
    parameter_bounds: Mapping[str, tuple[float, float]],
) -> dict[str, torch.Tensor]:
    """Map raw channels to an ordered dictionary of strictly bounded parameters."""
    if raw.ndim not in (1, 2, 3):
        raise ValueError("raw parameters must have shape [P], [B, P], or [B, T, P]")
    if not raw.is_floating_point():
        raise TypeError("raw parameters must use a floating-point dtype")

    ordered_bounds = tuple(parameter_bounds.items())
    if raw.shape[-1] != len(ordered_bounds):
        raise ValueError(
            f"raw parameter width {raw.shape[-1]} does not match parameter_bounds width {len(ordered_bounds)}"
        )

    squashed = torch.sigmoid(raw)
    parameters: dict[str, torch.Tensor] = {}
    for index, (name, (low, high)) in enumerate(ordered_bounds):
        if not low < high:
            raise ValueError(f"parameter {name!r} must have an increasing bound interval")
        lower = raw.new_tensor(low)
        upper = raw.new_tensor(high)
        mapped = lower + (upper - lower) * squashed[..., index]
        interior_lower = torch.nextafter(lower, upper)
        interior_upper = torch.nextafter(upper, lower)
        parameters[name] = torch.clamp(mapped, min=interior_lower, max=interior_upper)
    return parameters
