"""Golden trajectory tests for the Torch GR6J process boundary."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from hydrologeez.models.gr6j import GR6J, GR6JForcing, ProductionProcess, ResponseProcess, RoutingProcess
from hydrologeez.models.gr6j.constants import B, C, D
from hydrologeez.models.gr6j.processes import (
    PhysicalProduction,
    PhysicalResponse,
    PhysicalRouting,
    _ss1,
    _ss2,
    compute_uh_ordinates,
    convolve_uh,
    direct_branch,
    exponential_store_update,
    groundwater_exchange,
    percolation,
    production_store_update,
    routing_store_update,
)
from hydrologeez.models.gr6j.state import State
from hydrologeez.processes import Process

GOLDEN = Path(__file__).parent / "golden" / "gr6j.npz"


class _ZeroProduction(ProductionProcess):
    def forward(
        self,
        precip: torch.Tensor,
        pet: torch.Tensor,
        production_store: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del pet, x1
        zero = torch.zeros_like(precip)
        return production_store, zero, zero, zero, zero, zero


def test_process_slot_contracts() -> None:
    contracts = (ProductionProcess, RoutingProcess, ResponseProcess)
    physical = (PhysicalProduction, PhysicalRouting, PhysicalResponse)

    for contract, implementation in zip(contracts, physical, strict=True):
        assert inspect.isabstract(contract)
        assert issubclass(contract, Process)
        assert implementation.__bases__ == (contract,)
        assert implementation.introduces == {}


@pytest.fixture(scope="module")
def data():
    with np.load(GOLDEN, allow_pickle=False) as fixture:
        return {key: fixture[key] for key in fixture.files}


def tensor(value):
    return torch.from_numpy(np.array(value, copy=True)).to(torch.float64)


def scenarios(data):
    shape = (2, 2, 2, 12)

    def forcing(key):
        return tensor(np.broadcast_to(data[key][:, None, None, :], shape))

    def parameter(index):
        return tensor(np.broadcast_to(data["parameters"][None, :, None, index, None], shape))

    return forcing, parameter


def previous(data, key):
    current = data[f"state.{key}"]
    initial = data[f"initial_state.{key}"]
    first = (
        np.broadcast_to(initial[None, None, :, None, :], current.shape[:-2] + (1,) + current.shape[-1:])
        if current.ndim == 5
        else np.broadcast_to(initial[None, None, :, None], current.shape[:-1] + (1,))
    )
    return tensor(
        np.concatenate(
            (first, current[..., :-1, :] if current.ndim == 5 else current[..., :-1]),
            axis=-2 if current.ndim == 5 else -1,
        )
    )


def close(actual, expected):
    torch.testing.assert_close(
        actual,
        tensor(expected) if isinstance(expected, np.ndarray) else expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_default_slots_match_free_function_assembly(data):
    forcing, parameter = scenarios(data)
    precip = forcing("forcing.precip")
    pet = forcing("forcing.pet")
    old_production_store = previous(data, "production_store")

    s_after_ps, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(
        precip, pet, old_production_store, parameter(0)
    )
    storage_infiltration = net_rainfall_pn - effective_rainfall_pr
    s_after_perc, percolation_amount = percolation(s_after_ps, parameter(0))
    total_effective_rainfall = effective_rainfall_pr + percolation_amount
    expected_production = (
        s_after_perc,
        actual_et,
        net_rainfall_pn,
        storage_infiltration,
        percolation_amount,
        total_effective_rainfall,
    )
    actual_production = PhysicalProduction()(precip, pet, old_production_store, parameter(0))
    for actual, expected in zip(actual_production, expected_production, strict=True):
        close(actual, expected)

    uh1_ord, uh2_ord = compute_uh_ordinates(parameter(3))
    q9, uh1 = convolve_uh(previous(data, "uh1"), uh1_ord, B * total_effective_rainfall)
    q1, uh2 = convolve_uh(previous(data, "uh2"), uh2_ord, (1.0 - B) * total_effective_rainfall)
    expected_routing = (q9, q1, uh1, uh2)
    actual_routing = PhysicalRouting()(
        previous(data, "uh1"),
        previous(data, "uh2"),
        total_effective_rainfall,
        parameter(3),
    )
    for actual, expected in zip(actual_routing, expected_routing, strict=True):
        close(actual, expected)

    old_routing_store = previous(data, "routing_store")
    old_exponential_store = previous(data, "exponential_store")
    exchange_f = groundwater_exchange(old_routing_store, parameter(1), parameter(2), parameter(4))
    new_routing_store, qr, actual_exchange_routing = routing_store_update(
        old_routing_store, (1.0 - C) * q9, exchange_f, parameter(2)
    )
    new_exp_store, qrexp = exponential_store_update(old_exponential_store, C * q9, exchange_f, parameter(5))
    qd, actual_exchange_direct = direct_branch(q1, exchange_f)
    streamflow = torch.clamp_min(qr + qrexp + qd, 0.0)
    actual_exchange_total = actual_exchange_routing + actual_exchange_direct + exchange_f
    expected_response = (
        new_routing_store,
        qr,
        actual_exchange_routing,
        new_exp_store,
        qrexp,
        qd,
        actual_exchange_direct,
        exchange_f,
        streamflow,
        actual_exchange_total,
    )
    actual_response = PhysicalResponse()(
        old_routing_store,
        old_exponential_store,
        q9,
        q1,
        parameter(1),
        parameter(2),
        parameter(4),
        parameter(5),
    )
    for actual, expected in zip(actual_response, expected_response, strict=True):
        close(actual, expected)


def test_injected_production_slot_changes_run_behavior():
    default_x1 = torch.tensor(350.0, dtype=torch.float64)
    default_x2 = torch.tensor(-1.2, dtype=torch.float64)
    default_x3 = torch.tensor(90.0, dtype=torch.float64)
    default_x4 = torch.tensor(2.4, dtype=torch.float64)
    default_x5 = torch.tensor(0.15, dtype=torch.float64)
    default_x6 = torch.tensor(12.0, dtype=torch.float64)
    injected_x1 = torch.tensor(350.0, dtype=torch.float64)
    injected_x2 = torch.tensor(-1.2, dtype=torch.float64)
    injected_x3 = torch.tensor(90.0, dtype=torch.float64)
    injected_x4 = torch.tensor(2.4, dtype=torch.float64)
    injected_x5 = torch.tensor(0.15, dtype=torch.float64)
    injected_x6 = torch.tensor(12.0, dtype=torch.float64)
    forcing = GR6JForcing(
        precip=torch.tensor([[0.0, 5.0, 20.0, 3.0, 12.0, 0.0]], dtype=torch.float64),
        pet=torch.tensor([[2.0, 1.0, 0.5, 3.0, 1.5, 4.0]], dtype=torch.float64),
    )
    replacement = _ZeroProduction()
    default = GR6J(
        default_x1,
        default_x2,
        default_x3,
        default_x4,
        default_x5,
        default_x6,
        20,
    )
    injected = GR6J(
        injected_x1,
        injected_x2,
        injected_x3,
        injected_x4,
        injected_x5,
        injected_x6,
        20,
        production=replacement,
    )

    default_streamflow, default_fluxes, _ = default.run(forcing, return_fluxes=True)
    injected_streamflow, injected_fluxes, _ = injected.run(forcing, return_fluxes=True)

    assert injected.production is replacement
    assert injected.nh == 20
    torch.testing.assert_close(
        injected_fluxes.effective_rainfall,
        torch.zeros_like(injected_fluxes.effective_rainfall),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.allclose(
        default_fluxes.effective_rainfall,
        injected_fluxes.effective_rainfall,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not torch.allclose(default_streamflow, injected_streamflow, rtol=1e-12, atol=1e-12)


def test_state_round_trip_and_validation(data):
    state = State(
        *(
            tensor(data[f"initial_state.{key}"])
            for key in ("production_store", "routing_store", "exponential_store", "uh1", "uh2")
        )
    )
    flat = state.to_flat()
    assert flat.shape == (2, 63)
    restored = State.from_flat(flat)
    for key in state.__dataclass_fields__:
        close(getattr(restored, key), getattr(state, key))
    with pytest.raises(ValueError, match="63"):
        State.from_flat(torch.zeros(2, 62))


def test_state_flat_autograd():
    leaves = [torch.randn(2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    leaves += [torch.randn(2, width, dtype=torch.float64, requires_grad=True) for width in (20, 40)]
    State(*leaves).to_flat().sum().backward()
    assert all(value.grad is not None for value in leaves)


def test_store_processes_match_golden(data):
    forcing, parameter = scenarios(data)
    precip, pet = forcing("forcing.precip"), forcing("forcing.pet")
    production = previous(data, "production_store")
    result = production_store_update(precip, pet, production, parameter(0))
    expected_pr = data["flux.effective_rainfall"] - data["flux.percolation"]
    for actual, expected in zip(
        result,
        (
            data["state.production_store"] + data["flux.percolation"],
            data["flux.actual_et"],
            data["flux.net_rainfall"],
            expected_pr,
        ),
        strict=True,
    ):
        close(actual, expected)
    close(result[2] - result[3], data["flux.storage_infiltration"])
    perc = percolation(tensor(data["state.production_store"] + data["flux.percolation"]), parameter(0))
    close(perc[0], data["state.production_store"])
    close(perc[1], data["flux.percolation"])


def test_routing_processes_match_golden(data):
    _, parameter = scenarios(data)
    exchange = groundwater_exchange(previous(data, "routing_store"), parameter(1), parameter(2), parameter(4))
    close(exchange, data["flux.exchange"])
    routed = routing_store_update(
        previous(data, "routing_store"), (1.0 - C) * tensor(data["flux.q9"]), exchange, parameter(2)
    )
    for actual, key in zip(routed, ("state.routing_store", "flux.qr", "flux.actual_exchange_routing"), strict=True):
        close(actual, data[key])
    exponential = exponential_store_update(
        previous(data, "exponential_store"), C * tensor(data["flux.q9"]), exchange, parameter(5)
    )
    close(exponential[0], data["state.exponential_store"])
    close(exponential[0], data["flux.exponential_store"])
    close(exponential[1], data["flux.qrexp"])
    direct = direct_branch(tensor(data["flux.q1"]), exchange)
    close(direct[0], data["flux.qd"])
    close(direct[1], data["flux.actual_exchange_direct"])
    close(
        tensor(data["flux.actual_exchange_routing"] + data["flux.actual_exchange_direct"] + data["flux.exchange"]),
        data["flux.actual_exchange_total"],
    )


def numpy_ss1(i, x4):
    return np.where(i <= 0, 0, np.where(i < x4, (i / x4) ** D, 1))


def numpy_ss2(i, x4):
    ratio = i / x4
    return np.where(
        i <= 0, 0, np.where(i <= x4, 0.5 * ratio**D, np.where(i < 2 * x4, 1 - 0.5 * np.maximum(2 - ratio, 0) ** D, 1))
    )


def test_s_curves_and_ordinates(data):
    x4 = data["parameters"][:, 3, None]
    grid = np.stack(
        (
            [0, x4[0, 0] / 2, x4[0, 0], 1.5 * x4[0, 0], 2 * x4[0, 0], 2.5 * x4[0, 0]],
            [0, x4[1, 0] / 2, x4[1, 0], 1.5 * x4[1, 0], 2 * x4[1, 0], 2.5 * x4[1, 0]],
        )
    )
    close(_ss1(tensor(grid), tensor(x4)), tensor(numpy_ss1(grid, x4)))
    close(_ss2(tensor(grid), tensor(x4)), tensor(numpy_ss2(grid, x4)))
    tx4 = tensor(data["parameters"][:, 3])
    uh1, uh2 = compute_uh_ordinates(tx4)
    assert uh1.shape == (2, 20) and uh2.shape == (2, 40)
    assert uh1.dtype == torch.float64 and uh1.device.type == "cpu"
    i1 = np.arange(1, 21)[None, :]
    i2 = np.arange(1, 41)[None, :]
    close(uh1, tensor(numpy_ss1(i1, x4) - numpy_ss1(i1 - 1, x4)))
    close(uh2, tensor(numpy_ss2(i2, x4) - numpy_ss2(i2 - 1, x4)))
    close(uh1.sum(-1), torch.ones(2, dtype=torch.float64))
    close(uh2.sum(-1), torch.ones(2, dtype=torch.float64))


def test_convolutions_match_golden(data):
    _, parameter = scenarios(data)
    x4 = parameter(3)
    uh1, uh2 = compute_uh_ordinates(x4)
    result1 = convolve_uh(previous(data, "uh1"), uh1, B * tensor(data["flux.effective_rainfall"]))
    close(result1[0], data["flux.q9"])
    close(result1[1], data["state.uh1"])
    result2 = convolve_uh(previous(data, "uh2"), uh2, (1.0 - B) * tensor(data["flux.effective_rainfall"]))
    close(result2[0], data["flux.q1"])
    close(result2[1], data["state.uh2"])


def test_float32_dtype_smoke():
    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert production_store_update(values, values / 2, values, values * 100)[0].dtype == torch.float32
    assert percolation(values, values * 100)[0].dtype == torch.float32
    assert compute_uh_ordinates(values)[0].dtype == torch.float32


@pytest.mark.parametrize(
    "function,values",
    [
        (production_store_update, [3.0, 2.0, 40.0, 300.0]),
        (percolation, [40.0, 300.0]),
        (groundwater_exchange, [20.0, 1.0, 100.0, 0.1]),
        (routing_store_update, [20.0, 2.0, 0.1, 100.0]),
        (exponential_store_update, [2.0, 1.0, 0.1, 5.0]),
        (direct_branch, [2.0, 0.1]),
        (convolve_uh, [[1.0, 2.0, 3.0], [0.2, 0.3, 0.5], 2.0]),
    ],
)
def test_process_autograd(function, values):
    inputs = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in values]
    outputs = function(*inputs)
    tensors = outputs if isinstance(outputs, tuple) else (outputs,)
    sum(value.sum() for value in tensors).backward()
    for value in inputs:
        assert value.grad is not None and torch.isfinite(value.grad).all()


def test_s_curve_and_kernel_autograd():
    for function in (_ss1, _ss2):
        x4 = torch.tensor(2.3, dtype=torch.float64, requires_grad=True)
        function(torch.tensor(1.1, dtype=torch.float64), x4).backward()
        assert x4.grad is not None and torch.isfinite(x4.grad)
    x4 = torch.tensor([2.3, 3.7], dtype=torch.float64, requires_grad=True)
    outputs = compute_uh_ordinates(x4)
    (outputs[0].square().sum() + outputs[1].square().sum()).backward()
    assert x4.grad is not None and torch.isfinite(x4.grad).all()
