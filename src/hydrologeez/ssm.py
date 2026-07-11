"""Abstract eager PyTorch state-space-model base for hydrologeez."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, overload

import torch
from torch import nn

from hydrologeez.observation import default_streamflow_observation

State = Any
Forcing = Any
Fluxes = Any
ObservationOperator = Callable[[State, Fluxes], torch.Tensor]
Parameters = Mapping[str, torch.Tensor]


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [leaf for field in dataclasses.fields(value) for leaf in _tensor_leaves(getattr(value, field.name))]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _tensor_leaves(item)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _tensor_leaves(item)]
    raise TypeError(f"unsupported container leaf: {type(value).__name__}")


def _map_structure(function: Callable[[torch.Tensor], torch.Tensor], value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return function(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{field.name: _map_structure(function, getattr(value, field.name)) for field in dataclasses.fields(value)}
        )
    if isinstance(value, dict):
        return type(value)((key, _map_structure(function, item)) for key, item in value.items())
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_map_structure(function, item) for item in value))
    if isinstance(value, tuple):
        return tuple(_map_structure(function, item) for item in value)
    if isinstance(value, list):
        return [_map_structure(function, item) for item in value]
    raise TypeError(f"unsupported container leaf: {type(value).__name__}")


def _stack(values: list[Any]) -> Any:
    if not values:
        raise ValueError("cannot stack an empty sequence")
    exemplar = values[0]
    if any(type(value) is not type(exemplar) for value in values):
        raise TypeError("inconsistent container structures across time")
    if isinstance(exemplar, torch.Tensor):
        return torch.stack(values, dim=1)
    if dataclasses.is_dataclass(exemplar) and not isinstance(exemplar, type):
        return type(exemplar)(
            **{
                field.name: _stack([getattr(value, field.name) for value in values])
                for field in dataclasses.fields(exemplar)
            }
        )
    if isinstance(exemplar, dict):
        if any(value.keys() != exemplar.keys() for value in values):
            raise TypeError("inconsistent dictionary keys across time")
        return type(exemplar)((key, _stack([value[key] for value in values])) for key in exemplar)
    if isinstance(exemplar, tuple) and hasattr(exemplar, "_fields"):
        return type(exemplar)(*(_stack([value[index] for value in values]) for index in range(len(exemplar))))
    if isinstance(exemplar, tuple):
        if any(len(value) != len(exemplar) for value in values):
            raise TypeError("inconsistent tuple lengths across time")
        return tuple(_stack([value[index] for value in values]) for index in range(len(exemplar)))
    if isinstance(exemplar, list):
        if any(len(value) != len(exemplar) for value in values):
            raise TypeError("inconsistent list lengths across time")
        return [_stack([value[index] for value in values]) for index in range(len(exemplar))]
    raise TypeError(f"unsupported container leaf: {type(exemplar).__name__}")


def _forcing_shape(forcing: Forcing, *, allow_empty: bool) -> tuple[int, int]:
    leaves = _tensor_leaves(forcing)
    if not leaves:
        raise ValueError("forcing must contain at least one tensor")
    if any(leaf.ndim < 2 for leaf in leaves):
        raise ValueError("forcing tensor leaves must have leading [batch, time] dimensions")
    shape = (leaves[0].shape[0], leaves[0].shape[1])
    if any(leaf.shape[:2] != shape for leaf in leaves[1:]):
        raise ValueError("forcing tensor leaves must share a common batch/time prefix")
    if not allow_empty and shape[1] == 0:
        raise ValueError("main forcing must contain at least one timestep")
    return shape


def _validate_parameters(parameters: Parameters, batch_size: int, time_steps: int) -> None:
    for name, value in parameters.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"parameter {name!r} must be a torch.Tensor")
        if value.ndim > 2:
            raise ValueError(f"parameter {name!r} must have shape [], [B], or [B, T]")
        if value.ndim >= 1 and value.shape[0] != batch_size:
            raise ValueError(f"parameter {name!r} batch size does not match forcing")
        if value.ndim == 2 and value.shape[1] != time_steps:
            raise ValueError(f"parameter {name!r} time size does not match forcing")


def _parameters_at(parameters: Parameters, time: int) -> dict[str, torch.Tensor]:
    return {name: value[:, time] if value.ndim == 2 else value for name, value in parameters.items()}


class StateSpaceModel(nn.Module, abc.ABC):
    """Abstract eager, batched discrete-time hydrological model."""

    if TYPE_CHECKING:
        init_state: Any
        transition: Any
    else:

        @abc.abstractmethod
        def init_state(self, parameters: Parameters, *, batch_size: int) -> State:
            raise NotImplementedError

        @abc.abstractmethod
        def transition(
            self,
            state: State,
            forcing: Forcing,
            parameters: Parameters,
        ) -> tuple[State, Fluxes]:
            raise NotImplementedError

    @overload
    def run(
        self,
        forcing: Forcing,
        parameters: Parameters | None = None,
        *,
        warmup: Forcing | None = None,
        warmup_parameters: Parameters | None = None,
        observation_operator: ObservationOperator = default_streamflow_observation,
        return_fluxes: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        forcing: Forcing,
        parameters: Parameters | None = None,
        *,
        warmup: Forcing | None = None,
        warmup_parameters: Parameters | None = None,
        observation_operator: ObservationOperator = default_streamflow_observation,
        return_fluxes: Literal[True],
    ) -> tuple[torch.Tensor, Fluxes, State]: ...

    def run(
        self,
        forcing: Forcing,
        parameters: Parameters | None = None,
        *,
        warmup: Forcing | None = None,
        warmup_parameters: Parameters | None = None,
        observation_operator: ObservationOperator = default_streamflow_observation,
        return_fluxes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Fluxes, State]:
        """Run an eager fold over batched, time-leading forcing."""
        batch_size, time_steps = _forcing_shape(forcing, allow_empty=False)
        resolved = dict(self.named_parameters()) if parameters is None else parameters
        _validate_parameters(resolved, batch_size, time_steps)

        if warmup is None:
            state = self.init_state(resolved, batch_size=batch_size)
        else:
            warmup_batch, warmup_steps = _forcing_shape(warmup, allow_empty=True)
            if warmup_batch != batch_size:
                raise ValueError("warmup batch size must match main forcing")
            resolved_warmup = resolved if warmup_parameters is None else warmup_parameters
            _validate_parameters(resolved_warmup, batch_size, warmup_steps)
            state = self.init_state(resolved_warmup, batch_size=batch_size)
            with torch.no_grad():
                for time in range(warmup_steps):
                    state, _ = self.transition(
                        state,
                        _map_structure(lambda leaf, time=time: leaf[:, time], warmup),
                        _parameters_at(resolved_warmup, time),
                    )
            state = _map_structure(torch.Tensor.detach, state)

        observations = []
        fluxes = []
        for time in range(time_steps):
            state, step_fluxes = self.transition(
                state,
                _map_structure(lambda leaf, time=time: leaf[:, time], forcing),
                _parameters_at(resolved, time),
            )
            observations.append(observation_operator(state, step_fluxes))
            fluxes.append(step_fluxes)

        stacked_observations = _stack(observations)
        if return_fluxes:
            return stacked_observations, _stack(fluxes), state
        return stacked_observations
