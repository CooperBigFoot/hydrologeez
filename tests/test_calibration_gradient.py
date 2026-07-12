from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from torch import nn

import hydrologeez.calibration as calibration
from hydrologeez.calibration.adapter import (
    GR6J_SPEC,
    HBV_SPEC,
    ParamSpec,
    array_to_model,
    bounds_array,
    params_to_array,
)
from hydrologeez.calibration.gradient import _to_bounded, _to_unconstrained, calibrate_gradient
from hydrologeez.metrics import rmse
from hydrologeez.models.gr6j.model import GR6J, GR6JForcing
from hydrologeez.models.hbv.model import HBVForcing, HBVModel

DTYPE = torch.float64


def _gr6j(theta: torch.Tensor) -> GR6J:
    return GR6J(*(value for value in theta))


def _hbv(theta: torch.Tensor) -> HBVModel:
    return HBVModel(**dict(zip(HBV_SPEC.names, theta, strict=True)))


def _gr6j_forcing(steps: int, *, requires_grad: bool = False) -> GR6JForcing:
    precip = torch.rand((1, steps), dtype=DTYPE)
    precip = torch.where(precip > 0.35, precip * 18.0, torch.zeros_like(precip)).detach()
    pet = (0.5 + 3.0 * torch.rand((1, steps), dtype=DTYPE)).detach()
    return GR6JForcing(precip.requires_grad_(requires_grad), pet.requires_grad_(requires_grad))


def _hbv_forcing(steps: int) -> HBVForcing:
    precip = (1.0 + 9.0 * torch.rand((1, steps), dtype=DTYPE)).detach()
    pet = (0.3 + 2.0 * torch.rand((1, steps), dtype=DTYPE)).detach()
    temp = (5.0 * torch.sin(torch.linspace(-2.0, 4.0, steps, dtype=DTYPE))).unsqueeze(0).detach()
    return HBVForcing(precip, pet, temp)


def _normalized_error(theta: torch.Tensor, truth: torch.Tensor, spec: ParamSpec) -> torch.Tensor:
    lo, hi = bounds_array(spec, dtype=DTYPE)
    return torch.mean(torch.abs((theta - truth) / (hi - lo)))


def _assert_history(history: torch.Tensor, steps: int) -> None:
    assert history.shape == (steps,)
    assert history.dtype == DTYPE and history.device.type == "cpu"
    assert not history.requires_grad and history.grad_fn is None
    assert torch.isfinite(history).all()


def _assert_inside(model: nn.Module, spec: ParamSpec) -> None:
    theta = params_to_array(model, spec)
    lo, hi = bounds_array(spec, dtype=theta.dtype, device=theta.device)
    assert torch.all(theta >= lo) and torch.all(theta <= hi)


def test_gr6j_recovers_seeded_synthetic_case() -> None:
    torch.manual_seed(1101)
    warmup, main = _gr6j_forcing(6), _gr6j_forcing(36)
    truth = torch.tensor([420.0, -0.8, 110.0, 2.2, 0.1, 14.0], dtype=DTYPE)
    start = torch.tensor([650.0, 0.3, 180.0, 3.1, -2.0, 20.0], dtype=DTYPE)
    true_model = array_to_model(_gr6j(truth), truth)
    template = array_to_model(true_model, start)
    with torch.no_grad():
        observed = true_model.run(main, warmup=warmup)
        baseline = rmse(observed, template.run(main, warmup=warmup))

    calibrated, history = calibrate_gradient(
        template, main, observed, loss_term=rmse, warmup=warmup, n_steps=100, learning_rate=1e-3
    )
    final_theta = params_to_array(calibrated)
    with torch.no_grad():
        final_loss = rmse(observed, calibrated.run(main, warmup=warmup))
    _assert_history(history, 100)
    _assert_inside(calibrated, GR6J_SPEC)
    assert final_loss < 0.75 * baseline and final_loss < 0.75 * history[0]
    assert _normalized_error(final_theta, truth, GR6J_SPEC) < _normalized_error(start, truth, GR6J_SPEC)


def test_hbv_recovers_seeded_synthetic_case() -> None:
    torch.manual_seed(1102)
    warmup, main = _hbv_forcing(6), _hbv_forcing(40)
    truth = torch.tensor([0.2, 4.5, 0.95, 0.08, 0.06, 320.0, 0.7, 2.5, 0.35, 0.12, 0.04, 2.0, 35.0, 3.6], dtype=DTYPE)
    lo, hi = bounds_array(HBV_SPEC, dtype=DTYPE)
    start = truth + 0.08 * (hi - lo) * torch.tensor([1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=DTYPE)
    true_model = array_to_model(_hbv(truth), truth, HBV_SPEC)
    template = array_to_model(true_model, start, HBV_SPEC)
    with torch.no_grad():
        observed = true_model.run(main, warmup=warmup)
        baseline = rmse(observed, template.run(main, warmup=warmup))

    calibrated, history = calibrate_gradient(
        template,
        main,
        observed,
        loss_term=rmse,
        warmup=warmup,
        n_steps=100,
        learning_rate=5e-2,
        param_spec=HBV_SPEC,
    )
    final_theta = params_to_array(calibrated, HBV_SPEC)
    with torch.no_grad():
        final_loss = rmse(observed, calibrated.run(main, warmup=warmup))
    _assert_history(history, 100)
    _assert_inside(calibrated, HBV_SPEC)
    assert final_loss < 0.75 * baseline and final_loss < 0.75 * history[0]
    assert _normalized_error(final_theta, truth, HBV_SPEC) < _normalized_error(start, truth, HBV_SPEC)


def test_transforms_are_literal_round_trip_safe_and_differentiable() -> None:
    lo = torch.tensor([-2.0, 1.0, 10.0, -4.0], dtype=DTYPE)
    hi = torch.tensor([3.0, 9.0, 20.0, 6.0], dtype=DTYPE)
    x = torch.tensor([-2.0, 9.0, 15.0, 3.25], dtype=DTYPE)
    z = torch.clamp((x - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
    expected_u = torch.log(z) - torch.log1p(-z)
    u = _to_unconstrained(x, lo, hi)
    torch.testing.assert_close(u, expected_u)
    torch.testing.assert_close(_to_bounded(u, lo, hi), lo + (hi - lo) * torch.sigmoid(u))
    torch.testing.assert_close(_to_bounded(u[2:], lo[2:], hi[2:]), x[2:])
    bounded = _to_bounded(u, lo, hi)
    assert torch.isfinite(u).all() and torch.all(bounded > lo) and torch.all(bounded < hi)
    leaf = torch.zeros(4, dtype=DTYPE, requires_grad=True)
    _to_bounded(leaf, lo, hi).sum().backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all() and torch.all(leaf.grad != 0)


def test_functional_calibration_does_not_mutate_template() -> None:
    torch.manual_seed(1103)
    forcing = _gr6j_forcing(10)
    truth = torch.tensor([420.0, -0.8, 110.0, 2.2, 0.1, 14.0], dtype=DTYPE)
    template = _gr6j(torch.tensor([600.0, 0.0, 160.0, 3.0, -0.5, 19.0], dtype=DTYPE))
    before = params_to_array(template).detach().clone()
    with torch.no_grad():
        observed = _gr6j(truth).run(forcing)
    calibrated, _ = calibrate_gradient(template, forcing, observed, loss_term=rmse, n_steps=3)
    torch.testing.assert_close(params_to_array(template), before)
    assert calibrated is not template
    assert all(isinstance(parameter, nn.Parameter) and parameter.is_leaf for parameter in calibrated.parameters())
    assert not torch.equal(params_to_array(calibrated), before)


def test_warmup_is_no_grad_and_main_forcing_is_differentiable() -> None:
    torch.manual_seed(1104)
    warmup, main = _gr6j_forcing(4, requires_grad=True), _gr6j_forcing(8, requires_grad=True)
    template = _gr6j(torch.tensor([420.0, -0.8, 110.0, 2.2, 0.1, 14.0], dtype=DTYPE))
    with torch.no_grad():
        observed = template.run(
            GR6JForcing(main.precip.detach(), main.pet.detach()),
            warmup=GR6JForcing(warmup.precip.detach(), warmup.pet.detach()),
        )
    _, history = calibrate_gradient(template, main, observed, loss_term=rmse, warmup=warmup, n_steps=1)
    assert main.precip.grad is not None and main.pet.grad is not None
    assert warmup.precip.grad is None and warmup.pet.grad is None
    _assert_history(history, 1)


@pytest.mark.parametrize(
    "optimizer",
    [torch.optim.SGD, lambda parameters: torch.optim.Adam(parameters, lr=1e-2)],
)
def test_optimizer_class_and_configured_factory(
    optimizer: type[torch.optim.Optimizer] | Callable[[list[nn.Parameter]], torch.optim.Optimizer],
) -> None:
    torch.manual_seed(1105)
    forcing = _gr6j_forcing(5)
    template = _gr6j(torch.tensor([420.0, -0.8, 110.0, 2.2, 0.1, 14.0], dtype=DTYPE))
    with torch.no_grad():
        observed = template.run(forcing)
    _, history = calibrate_gradient(template, forcing, observed, loss_term=rmse, n_steps=2, optimizer=optimizer)
    _assert_history(history, 2)


def test_input_and_loss_validation() -> None:
    torch.manual_seed(1106)
    forcing = _gr6j_forcing(4)
    model = _gr6j(torch.tensor([420.0, -0.8, 110.0, 2.2, 0.1, 14.0], dtype=DTYPE))
    observed = model.run(forcing).detach()
    with pytest.raises(ValueError, match="n_steps"):
        calibrate_gradient(model, forcing, observed, loss_term=rmse, n_steps=0)
    with pytest.raises(ValueError, match="learning_rate"):
        calibrate_gradient(model, forcing, observed, loss_term=rmse, learning_rate=float("nan"))
    with pytest.raises(TypeError, match="template"):
        calibrate_gradient(cast(Any, object()), forcing, observed, loss_term=rmse)
    with pytest.raises(TypeError, match="observed"):
        calibrate_gradient(model, forcing, cast(Any, object()), loss_term=rmse)
    batched = array_to_model(model, params_to_array(model).repeat(2, 1))
    with pytest.raises(ValueError, match="one-dimensional"):
        calibrate_gradient(batched, forcing, observed, loss_term=rmse)
    with pytest.raises(ValueError, match="scalar"):
        calibrate_gradient(model, forcing, observed, loss_term=lambda obs, sim: sim, n_steps=1)
    with pytest.raises(ValueError, match="finite"):
        calibrate_gradient(model, forcing, observed, loss_term=lambda obs, sim: sim.sum() * torch.nan, n_steps=1)
    with pytest.raises(TypeError, match="must return"):
        calibrate_gradient(
            model,
            forcing,
            observed,
            loss_term=rmse,
            optimizer=cast(Any, lambda parameters: object()),
            n_steps=1,
        )
    other = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
    with pytest.raises(ValueError, match="exactly"):
        calibrate_gradient(
            model,
            forcing,
            observed,
            loss_term=rmse,
            optimizer=lambda parameters: torch.optim.Adam([other]),
            n_steps=1,
        )


def test_import_surface_retains_transitional_names() -> None:
    assert calibration.calibrate_gradient
    assert calibration.array_to_parameters
    assert calibration.calibrate_evolutionary and calibration.calibrate_nsga2
