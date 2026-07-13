"""Pure-Torch primitives for neural-parameterized state-space models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from hydrologeez.ssm import StateSpaceModel

if TYPE_CHECKING:
    import hcx


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


class NeuralParameterForecastModel(nn.Module):
    """Run neural-parameterized state-space components as one hcx point forecast."""

    def __init__(
        self,
        wrapped_model: StateSpaceModel,
        network: nn.Module,
        *,
        dynamic_inputs: Sequence[str],
        static_inputs: Sequence[str],
        physics_forcing: Mapping[str, str | int],
        network_dynamic_inputs: Sequence[str | int],
        network_static_inputs: Sequence[str | int],
        forcing_factory: Callable[..., Any],
        output_specification: hcx.OutputSpecification[Any],
        n_components: int = 1,
    ) -> None:
        super().__init__()
        if not physics_forcing:
            raise ValueError("at least one physics-forcing field is required")
        if output_specification.raw_head_width != 1:
            raise ValueError("the discharge adapter requires a point output specification of width 1")
        if type(n_components) is not int or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        parameter_bounds = getattr(wrapped_model, "parameter_bounds", None)
        if not isinstance(parameter_bounds, Mapping):
            raise TypeError("wrapped_model must expose parameter_bounds")
        if n_components > 1 and not callable(getattr(wrapped_model, "run_components", None)):
            raise TypeError("wrapped_model must provide a callable run_components for n_components > 1")

        self.wrapped_model = wrapped_model
        self.network = network
        self.parameter_bounds = cast(Mapping[str, tuple[float, float]], parameter_bounds)
        self.n_components = n_components
        self.dynamic_inputs = tuple(dynamic_inputs)
        self.static_inputs = tuple(static_inputs)
        self.physics_forcing = tuple(physics_forcing.items())
        self.network_dynamic_inputs = tuple(network_dynamic_inputs)
        self.network_static_inputs = tuple(network_static_inputs)
        self.forcing_factory = forcing_factory
        self.output_specification = output_specification
        self.consumed_quadrants = (
            ("scalar_dynamic", "scalar_static") if self.network_static_inputs else ("scalar_dynamic",)
        )

    def forward(self, batch: hcx.Batch) -> hcx.Forecast:
        scalar_dynamic = batch.scalar_dynamic
        if scalar_dynamic is None:
            raise ValueError("scalar_dynamic is required for physics forcing")
        if batch.target.ndim != 2:
            raise ValueError("target must have shape [B, T_out]")
        if batch.target.shape[0] != scalar_dynamic.shape[0]:
            raise ValueError("target and scalar_dynamic batch sizes must match")

        forcing_fields = tuple(field for field, _ in self.physics_forcing)
        forcing_selectors = tuple(selector for _, selector in self.physics_forcing)
        selected_forcing, _ = select_features(
            scalar_dynamic,
            batch.scalar_static,
            dynamic_inputs=self.dynamic_inputs,
            static_inputs=self.static_inputs,
            dynamic_features=forcing_selectors,
        )
        assert selected_forcing is not None
        forcing_tensor: torch.Tensor = selected_forcing

        network_inputs = select_network_inputs(
            scalar_dynamic,
            batch.scalar_static,
            dynamic_inputs=self.dynamic_inputs,
            static_inputs=self.static_inputs,
            dynamic_features=self.network_dynamic_inputs,
            static_features=self.network_static_inputs,
        )
        raw_parameters = self.network(network_inputs)
        if not isinstance(raw_parameters, torch.Tensor):
            raise TypeError("network must return a torch.Tensor")

        batch_size, input_length = scalar_dynamic.shape[:2]
        parameter_width = len(self.parameter_bounds)
        expected_width = self.n_components * parameter_width
        if raw_parameters.ndim != 3 or raw_parameters.shape != (batch_size, input_length, expected_width):
            raise ValueError(
                "network output must have shape "
                f"[B, input_length, N * P] = {(batch_size, input_length, expected_width)}; "
                f"got {tuple(raw_parameters.shape)}"
            )
        raw_parameter_sets = raw_parameters.reshape(
            batch_size,
            input_length,
            self.n_components,
            parameter_width,
        )
        bounded = bounded_parameters(
            raw_parameter_sets.reshape(batch_size, input_length * self.n_components, parameter_width),
            self.parameter_bounds,
        )
        parameters = {
            name: value.reshape(batch_size, input_length, self.n_components).permute(0, 2, 1)
            for name, value in bounded.items()
        }

        output_length = batch.target.shape[-1]
        if output_length <= 0:
            raise ValueError("target output length must be positive")
        if output_length > input_length:
            raise ValueError("input sequence is shorter than the requested output sequence")
        warmup_length = input_length - output_length

        def make_forcing(start: int, stop: int) -> Any:
            return self.forcing_factory(
                **{field: forcing_tensor[:, start:stop, index] for index, field in enumerate(forcing_fields)}
            )

        main_forcing = make_forcing(warmup_length, input_length)
        main_parameters = {name: value[:, :, warmup_length:input_length] for name, value in parameters.items()}
        warmup_forcing = make_forcing(0, warmup_length) if warmup_length else None
        warmup_parameters = (
            {name: value[:, :, :warmup_length] for name, value in parameters.items()} if warmup_length else None
        )

        # Keep the ordinary run path explicit: N=1 must be bit-identical to
        # the pre-component NeuralParameterForecastModel implementation.
        if self.n_components == 1:
            single_main = {name: value[:, 0, :] for name, value in main_parameters.items()}
            if warmup_forcing is not None:
                assert warmup_parameters is not None
                single_warmup = {name: value[:, 0, :] for name, value in warmup_parameters.items()}
                discharge = self.wrapped_model.run(
                    main_forcing,
                    parameters=single_main,
                    warmup=warmup_forcing,
                    warmup_parameters=single_warmup,
                )
            else:
                discharge = self.wrapped_model.run(main_forcing, parameters=single_main)
        else:
            run_components = cast(Callable[..., torch.Tensor], getattr(self.wrapped_model, "run_components", None))
            if warmup_forcing is not None:
                assert warmup_parameters is not None
                discharge = run_components(
                    main_forcing,
                    parameters=main_parameters,
                    n_components=self.n_components,
                    warmup=warmup_forcing,
                    warmup_parameters=warmup_parameters,
                )
            else:
                discharge = run_components(
                    main_forcing,
                    parameters=main_parameters,
                    n_components=self.n_components,
                )

        discharge = discharge[:, -output_length:]
        expected_shape = (batch_size, output_length)
        if discharge.shape != expected_shape:
            raise ValueError(f"wrapped model discharge must have shape {expected_shape}; got {tuple(discharge.shape)}")
        point_parameters = self.output_specification.parameterize(discharge[..., None])
        return self.output_specification.populate_forecast(
            point_parameters,
            sample_ids=batch.metadata.sample_ids,
            input_end_indices=batch.metadata.input_end_indices,
            target_fill_mask=batch.metadata.target_fill_mask,
        )
