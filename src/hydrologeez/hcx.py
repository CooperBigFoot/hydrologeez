"""Dev-only hcx adapters for hydrologeez models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from hydrologeez.hybrid import NeuralParameterForecastModel
from hydrologeez.models.gr6j import GR6J, GR6JForcing
from hydrologeez.models.hbv import HBVForcing, HBVModel

if TYPE_CHECKING:
    from hcx import OutputSpecification


def _feature_indices(dynamic_inputs: Sequence[str], required: tuple[str, ...]) -> dict[str, int]:
    names = tuple(dynamic_inputs)
    if len(names) != len(set(names)):
        raise ValueError("dynamic_inputs must not contain duplicates")
    if set(names) != set(required):
        raise ValueError(f"dynamic_inputs must contain exactly {required!r}")
    return {name: names.index(name) for name in required}


def _point_forecast(prediction: torch.Tensor, batch: Any) -> Any:
    from hcx import Point

    output_length = batch.target.shape[-1]
    if prediction.shape[1] < output_length:
        raise ValueError("input sequence is shorter than the requested output sequence")
    point = Point()
    parameters = point.parameterize(prediction[:, -output_length:, None])
    return point.populate_forecast(
        parameters,
        sample_ids=batch.metadata.sample_ids,
        input_end_indices=batch.metadata.input_end_indices,
        target_fill_mask=batch.metadata.target_fill_mask,
    )


def _validate_size(key: str, value: object, *, allow_zero: bool = False) -> None:
    valid_range = value >= 0 if isinstance(value, int) else False
    if isinstance(value, bool) or not isinstance(value, int) or not valid_range or (not allow_zero and value == 0):
        requirement = "a nonnegative integer" if allow_zero else "a positive integer"
        raise ValueError(f"{key} must be {requirement}; got {value!r}")


def _validate_names(key: str, names: list[str], size: int) -> tuple[str, ...]:
    if len(names) != size:
        raise ValueError(f"{key} count must equal its supplied size {size}; got {len(names)}")
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"{key} names must be strings; got {name!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"{key} names must be unique; got {names!r}")
    return tuple(names)


def _parse_factory_config(model_config: dict[str, object]) -> tuple[int, int]:
    allowed = {"hidden_size", "n_components"}
    for key, value in model_config.items():
        if key not in allowed:
            raise ValueError(f"unknown model config key {key!r} with value {value!r}")

    hidden_size = model_config.get("hidden_size", 16)
    n_components = model_config.get("n_components", 1)
    _validate_size("hidden_size", hidden_size)
    _validate_size("n_components", n_components)
    assert isinstance(hidden_size, int) and not isinstance(hidden_size, bool)
    assert isinstance(n_components, int) and not isinstance(n_components, bool)
    return hidden_size, n_components


def _default_hbv() -> HBVModel:
    def scalar(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32)

    return HBVModel(
        tt=scalar(0.0),
        cfmax=scalar(3.0),
        sfcf=scalar(1.0),
        cwh=scalar(0.1),
        cfr=scalar(0.05),
        fc=scalar(150.0),
        lp=scalar(0.7),
        beta=scalar(2.0),
        k0=scalar(0.2),
        k1=scalar(0.05),
        k2=scalar(0.01),
        perc=scalar(1.0),
        uzl=scalar(10.0),
        maxbas=scalar(3.0),
    )


class GR6JForecastModel(nn.Module):
    """Expose a GR6J module through the hcx point-forecast contract."""

    consumed_quadrants = ("scalar_dynamic",)

    def __init__(self, model: GR6J, *, dynamic_inputs: Sequence[str]) -> None:
        super().__init__()
        self.model = model
        self._indices = _feature_indices(dynamic_inputs, ("precip", "pet"))

    def forward(self, batch: Any) -> Any:
        scalar_dynamic = batch.scalar_dynamic
        if scalar_dynamic is None:
            raise ValueError("scalar_dynamic is required")
        forcing = GR6JForcing(
            precip=scalar_dynamic[..., self._indices["precip"]],
            pet=scalar_dynamic[..., self._indices["pet"]],
        )
        return _point_forecast(self.model.run(forcing), batch)


class HBVForecastModel(nn.Module):
    """Expose an HBV module through the hcx point-forecast contract."""

    consumed_quadrants = ("scalar_dynamic",)

    def __init__(self, model: HBVModel, *, dynamic_inputs: Sequence[str]) -> None:
        super().__init__()
        self.model = model
        self._indices = _feature_indices(dynamic_inputs, ("precip", "pet", "temp"))

    def forward(self, batch: Any) -> Any:
        scalar_dynamic = batch.scalar_dynamic
        if scalar_dynamic is None:
            raise ValueError("scalar_dynamic is required")
        forcing = HBVForcing(
            precip=scalar_dynamic[..., self._indices["precip"]],
            pet=scalar_dynamic[..., self._indices["pet"]],
            temp=scalar_dynamic[..., self._indices["temp"]],
        )
        return _point_forecast(self.model.run(forcing), batch)


def create_model(
    model_config: dict[str, object],
    *,
    dynamic_inputs: list[str],
    static_inputs: list[str],
    input_size: int,
    static_size: int,
    output_size: int,
    output_specification: OutputSpecification[object],
) -> torch.nn.Module:
    from hcx import ForecastModel

    _validate_size("input_size", input_size)
    _validate_size("static_size", static_size, allow_zero=True)
    _validate_size("output_size", output_size)
    if output_size != 1:
        raise ValueError(f"output_size must be 1 for the single discharge series; got {output_size!r}")

    dynamic_names = _validate_names("dynamic_inputs", dynamic_inputs, input_size)
    static_names = _validate_names("static_inputs", static_inputs, static_size)
    missing_forcing = {"precip", "pet", "temp"} - set(dynamic_names)
    if missing_forcing:
        raise ValueError(f"dynamic_inputs are missing required HBV forcing names {sorted(missing_forcing)!r}")
    if output_specification.raw_head_width != 1:
        raise ValueError("dPL-HBV requires a point output specification with raw_head_width == 1")

    hidden_size, n_components = _parse_factory_config(model_config)
    hbv = _default_hbv()
    network = nn.Sequential(
        nn.Linear(input_size + static_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, n_components * len(hbv.parameter_bounds)),
    )
    model = NeuralParameterForecastModel(
        hbv,
        network,
        dynamic_inputs=dynamic_names,
        static_inputs=static_names,
        physics_forcing={"precip": "precip", "pet": "pet", "temp": "temp"},
        network_dynamic_inputs=dynamic_names,
        network_static_inputs=static_names,
        forcing_factory=HBVForcing,
        output_specification=output_specification,
        n_components=n_components,
    )
    if not isinstance(model, ForecastModel):
        raise TypeError("constructed dPL-HBV model does not satisfy hcx.ForecastModel")
    return model


__all__ = ["GR6JForecastModel", "HBVForecastModel", "create_model"]
