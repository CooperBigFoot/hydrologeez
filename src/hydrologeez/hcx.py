"""Dev-only hcx adapters for hydrologeez models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from hydrologeez.models.gr6j import GR6J, GR6JForcing
from hydrologeez.models.hbv import HBVForcing, HBVModel


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


__all__ = ["GR6JForecastModel", "HBVForecastModel"]
