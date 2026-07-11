"""Tests for the default observation operator."""

from dataclasses import dataclass

import torch

from hydrologeez.observation import default_streamflow_observation


@dataclass
class Fluxes:
    streamflow: torch.Tensor
    other: torch.Tensor


def test_default_observation_returns_exact_streamflow_object():
    streamflow = torch.tensor([1.0], dtype=torch.float64)
    other = torch.tensor([2.0], dtype=torch.float64)
    result = default_streamflow_observation(object(), Fluxes(streamflow, other))
    assert result is streamflow
    assert result is not other
