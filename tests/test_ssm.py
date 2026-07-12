"""Tests for the eager PyTorch StateSpaceModel base."""

from dataclasses import dataclass
from typing import cast

import pytest
import torch
from torch import nn

from hydrologeez.ssm import StateSpaceModel


@dataclass(frozen=True)
class DummyFluxes:
    streamflow: torch.Tensor
    inflow: torch.Tensor


class LinearReservoir(StateSpaceModel):
    def __init__(self, a: float = 0.7, s0: float = 0.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.register_buffer("s0", torch.tensor(s0, dtype=torch.float64))

    def init_state(self, parameters, *, batch_size):
        return cast(torch.Tensor, self.s0).expand(batch_size)

    def transition(self, state, forcing, parameters):
        next_state = parameters["a"] * state + forcing
        return next_state, DummyFluxes(streamflow=next_state, inflow=forcing)


def inflow_observation(state, fluxes):
    return fluxes.inflow


def _manual(forcing, a, s0=0.0):
    state = torch.full((forcing.shape[0],), s0, dtype=forcing.dtype)
    outputs = []
    for time in range(forcing.shape[1]):
        step_a = a[:, time] if a.ndim == 2 else a
        state = step_a * state + forcing[:, time]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def test_native_batch_run_matches_independent_calculation():
    model = LinearReservoir(a=0.5, s0=1.0)
    forcing = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 10
    observed = model.run(forcing)
    expected = _manual(forcing, model.a, s0=1.0)
    assert observed.shape == (3, 4)
    torch.testing.assert_close(observed, expected)
    assert not hasattr(model, "batch_run")


def test_observation_operator_swap_returns_inflow():
    model = LinearReservoir(a=0.6)
    forcing = torch.linspace(1.0, 5.0, 10, dtype=torch.float64).reshape(2, 5)
    default = model.run(forcing)
    swapped = model.run(forcing, observation_operator=inflow_observation)
    torch.testing.assert_close(swapped, forcing)
    assert not torch.allclose(default, swapped)


def test_return_fluxes_tuple_contract():
    model = LinearReservoir(a=0.5)
    forcing = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    result = model.run(forcing, return_fluxes=True)
    assert len(result) == 3
    observations, fluxes, final_state = result
    assert fluxes.streamflow.shape == (2, 4)
    assert fluxes.inflow.shape == (2, 4)
    assert final_state.shape == (2,)
    torch.testing.assert_close(fluxes.inflow, forcing)
    torch.testing.assert_close(fluxes.streamflow, observations)
    torch.testing.assert_close(final_state, observations[:, -1])


def test_registered_parameter_fallback_is_used_and_differentiable():
    model = LinearReservoir(a=0.8, s0=1.0)
    assert dict(model.named_parameters())["a"] is model.a
    loss = model.run(torch.ones((2, 5), dtype=torch.float64)).sum()
    loss.backward()
    assert model.a.grad is not None
    assert torch.isfinite(model.a.grad)
    assert model.a.grad != 0


def test_explicit_per_basin_parameter_and_gradient():
    model = LinearReservoir(s0=1.0)
    forcing = torch.ones((2, 4), dtype=torch.float64)
    a = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    observed = model.run(forcing, {"a": a})
    torch.testing.assert_close(observed, _manual(forcing, a, s0=1.0))
    observed.sum().backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()


def test_explicit_per_timestep_parameter_and_mismatch_rejection():
    model = LinearReservoir(s0=1.0)
    forcing = torch.ones((2, 3), dtype=torch.float64)
    a = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=torch.float64, requires_grad=True)
    observed = model.run(forcing, {"a": a})
    torch.testing.assert_close(observed, _manual(forcing, a, s0=1.0))
    observed.sum().backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    with pytest.raises(ValueError, match="time size"):
        model.run(forcing, {"a": torch.ones((2, 4), dtype=torch.float64)})


def test_warmup_changes_state_but_is_detached_from_main_loss():
    model = LinearReservoir(s0=0.0)
    warmup = torch.ones((2, 3), dtype=torch.float64, requires_grad=True)
    forcing = torch.ones((2, 2), dtype=torch.float64, requires_grad=True)
    warmup_a = torch.full((2,), 0.5, dtype=torch.float64, requires_grad=True)
    main_a = torch.full((2,), 0.8, dtype=torch.float64, requires_grad=True)

    observed = model.run(forcing, {"a": main_a}, warmup=warmup, warmup_parameters={"a": warmup_a})
    without_warmup = model.run(forcing, {"a": main_a})
    assert not torch.allclose(observed, without_warmup)
    observed.sum().backward()
    assert warmup.grad is None
    assert warmup_a.grad is None
    assert forcing.grad is not None and torch.isfinite(forcing.grad).all() and torch.any(forcing.grad != 0)
    assert main_a.grad is not None and torch.isfinite(main_a.grad).all() and torch.any(main_a.grad != 0)


def test_empty_main_forcing_is_rejected():
    with pytest.raises(ValueError, match="at least one timestep"):
        LinearReservoir().run(torch.empty((2, 0), dtype=torch.float64))
