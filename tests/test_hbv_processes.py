import inspect
from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch

from hydrologeez.models.hbv import (
    HBVFluxes,
    HBVForcing,
    HBVModel,
    PhysicalResponseProcess,
    PhysicalRoutingProcess,
    PhysicalSnowProcess,
    PhysicalSoilProcess,
    ResponseProcess,
    RoutingProcess,
    SnowProcess,
    SoilProcess,
    constants,
    processes,
)
from hydrologeez.models.hbv.state import HBVState
from hydrologeez.processes import Process

GOLDEN = Path(__file__).parent / "golden" / "hbv.npz"
PARAM_NAMES = ("tt", "cfmax", "sfcf", "cwh", "cfr", "fc", "lp", "beta", "k0", "k1", "k2", "perc", "uzl", "maxbas")


def tensor(value: np.ndarray | float) -> torch.Tensor:
    return torch.as_tensor(np.array(value, copy=True), dtype=torch.float64, device="cpu")


def _slot_model(
    *,
    snow: SnowProcess | None = None,
    soil: SoilProcess | None = None,
    response: ResponseProcess | None = None,
    routing: RoutingProcess | None = None,
) -> HBVModel:
    return HBVModel(
        tt=tensor(0.0),
        cfmax=tensor(3.0),
        sfcf=tensor(1.0),
        cwh=tensor(0.1),
        cfr=tensor(0.05),
        fc=tensor(100.0),
        lp=tensor(0.7),
        beta=tensor(1.0),
        k0=tensor(0.3),
        k1=tensor(0.1),
        k2=tensor(0.05),
        perc=tensor(2.0),
        uzl=tensor(1.0),
        maxbas=tensor(3.0),
        snow=snow,
        soil=soil,
        response=response,
        routing=routing,
    )


def _slot_forcing() -> HBVForcing:
    return HBVForcing(
        precip=torch.full((1, 12), 10.0, dtype=torch.float64, device="cpu"),
        pet=torch.zeros((1, 12), dtype=torch.float64, device="cpu"),
        temp=torch.full((1, 12), 5.0, dtype=torch.float64, device="cpu"),
    )


class NoRechargeSoil(SoilProcess):
    introduces: ClassVar[dict[str, tuple[float, float]]] = {
        "fc": (50.0, 700.0),
        "lp": (0.3, 1.0),
        "beta": (1.0, 6.0),
    }

    def forward(
        self,
        soil_input: torch.Tensor,
        pet: torch.Tensor,
        soil_moisture: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del soil_input, pet, parameters
        zeros = torch.zeros_like(soil_moisture)
        return soil_moisture, zeros, zeros


def test_process_slot_contracts() -> None:
    contracts = (SnowProcess, SoilProcess, ResponseProcess, RoutingProcess)
    physical = (PhysicalSnowProcess, PhysicalSoilProcess, PhysicalResponseProcess, PhysicalRoutingProcess)
    expected = (
        {
            "tt": (-2.5, 2.5),
            "cfmax": (0.5, 10.0),
            "sfcf": (0.4, 1.4),
            "cwh": (0.0, 0.2),
            "cfr": (0.0, 0.2),
        },
        {"fc": (50.0, 700.0), "lp": (0.3, 1.0), "beta": (1.0, 6.0)},
        {
            "k0": (0.05, 0.99),
            "k1": (0.01, 0.5),
            "k2": (0.001, 0.2),
            "perc": (0.0, 6.0),
            "uzl": (0.0, 100.0),
        },
        {"maxbas": (1.0, 7.0)},
    )

    for contract, implementation, introduces in zip(contracts, physical, expected, strict=True):
        assert inspect.isabstract(contract)
        assert issubclass(contract, Process)
        assert implementation.__bases__ == (contract,)
        assert implementation.introduces == introduces
        assert "introduces" in implementation.__dict__
    assert len({id(implementation.introduces) for implementation in physical}) == len(physical)


def test_default_parameter_bounds_match_canonical_constants() -> None:
    model = _slot_model()
    expected = {name: constants.PARAM_BOUNDS[name] for name in constants.PARAM_NAMES}

    assert model.parameter_bounds == expected
    assert tuple(model.parameter_bounds) == constants.PARAM_NAMES == PARAM_NAMES
    assert tuple(dict(model.named_parameters())) == tuple(model.parameter_bounds)


def test_replacing_soil_changes_parameter_names_bounds_and_registration() -> None:
    class ReplacementSoil(SoilProcess):
        introduces: ClassVar[dict[str, tuple[float, float]]] = {
            "fc": (25.0, 800.0),
            "lp": (0.2, 0.9),
            "soil_scale": (0.0, 2.0),
        }

        def forward(
            self,
            soil_input: torch.Tensor,
            pet: torch.Tensor,
            soil_moisture: torch.Tensor,
            parameters: Mapping[str, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del pet
            recharge = soil_input * parameters["soil_scale"]
            return soil_moisture, recharge, torch.zeros_like(soil_moisture)

    default = _slot_model()
    replacement = ReplacementSoil()
    injected = HBVModel(
        tt=tensor(0.0),
        cfmax=tensor(3.0),
        sfcf=tensor(1.0),
        cwh=tensor(0.1),
        cfr=tensor(0.05),
        fc=tensor(100.0),
        lp=tensor(0.7),
        k0=tensor(0.3),
        k1=tensor(0.1),
        k2=tensor(0.05),
        perc=tensor(2.0),
        uzl=tensor(1.0),
        maxbas=tensor(3.0),
        soil_scale=tensor(0.5),
        soil=replacement,
    )

    expected_names = (
        "tt",
        "cfmax",
        "sfcf",
        "cwh",
        "cfr",
        "fc",
        "lp",
        "soil_scale",
        "k0",
        "k1",
        "k2",
        "perc",
        "uzl",
        "maxbas",
    )
    assert injected.soil is replacement
    assert tuple(injected.parameter_bounds) == expected_names
    assert set(injected.parameter_bounds) != set(default.parameter_bounds)
    assert injected.parameter_bounds["fc"] == (25.0, 800.0)
    assert injected.parameter_bounds["lp"] == (0.2, 0.9)
    assert injected.parameter_bounds["fc"] != default.parameter_bounds["fc"]
    assert injected.parameter_bounds["lp"] != default.parameter_bounds["lp"]
    assert tuple(dict(injected.named_parameters())) == expected_names
    assert "beta" not in dict(injected.named_parameters())
    torch.testing.assert_close(injected.soil_scale, tensor(0.5))


def test_process_introduces_preserves_insertion_order() -> None:
    class OrderedProcess(Process):
        introduces = {"second": (-1.0, 1.0), "first": (0.0, 2.0)}

    assert tuple(OrderedProcess.introduces) == ("second", "first")


@pytest.mark.parametrize(
    ("introduces", "exception", "message"),
    [
        ([], TypeError, "introduces must be an insertion-ordered dict"),
        ({"": (0.0, 1.0)}, ValueError, "introduced parameter names must be non-empty strings"),
        ({"x": [0.0, 1.0]}, ValueError, "bounds for 'x' must be a \\(low, high\\) tuple"),
        ({"x": (0.0,)}, ValueError, "bounds for 'x' must be a \\(low, high\\) tuple"),
        ({"x": (float("nan"), 1.0)}, ValueError, "bounds for 'x' must be finite numbers"),
        ({"x": (0.0, float("inf"))}, ValueError, "bounds for 'x' must be finite numbers"),
        ({"x": (1.0, 1.0)}, ValueError, "bounds for 'x' must satisfy low < high"),
        ({"x": (2.0, 1.0)}, ValueError, "bounds for 'x' must satisfy low < high"),
    ],
)
def test_process_introduces_validation(introduces: object, exception: type[Exception], message: str) -> None:
    with pytest.raises(exception, match=message):
        type("InvalidProcess", (Process,), {"introduces": introduces})


def test_physical_slots_are_registered_parameterless_and_match_explicit_defaults() -> None:
    implicit = _slot_model()
    explicit = _slot_model(
        snow=PhysicalSnowProcess(),
        soil=PhysicalSoilProcess(),
        response=PhysicalResponseProcess(),
        routing=PhysicalRoutingProcess(),
    )

    assert isinstance(implicit.snow, PhysicalSnowProcess)
    assert isinstance(implicit.soil, PhysicalSoilProcess)
    assert isinstance(implicit.response, PhysicalResponseProcess)
    assert isinstance(implicit.routing, PhysicalRoutingProcess)
    assert {"snow", "soil", "response", "routing"} <= dict(implicit.named_modules()).keys()
    for slot in (implicit.snow, implicit.soil, implicit.response, implicit.routing):
        assert tuple(slot.parameters()) == ()
        assert tuple(slot.buffers()) == ()
    assert tuple(dict(implicit.named_parameters())) == PARAM_NAMES
    assert tuple(dict(explicit.named_parameters())) == PARAM_NAMES

    implicit_observations, implicit_fluxes, implicit_state = implicit.run(_slot_forcing(), return_fluxes=True)
    explicit_observations, explicit_fluxes, explicit_state = explicit.run(_slot_forcing(), return_fluxes=True)
    torch.testing.assert_close(implicit_observations, explicit_observations, rtol=1e-12, atol=1e-12)
    for field in fields(HBVFluxes):
        torch.testing.assert_close(
            getattr(implicit_fluxes, field.name),
            getattr(explicit_fluxes, field.name),
            rtol=1e-12,
            atol=1e-12,
        )
    for field in fields(HBVState):
        torch.testing.assert_close(
            getattr(implicit_state, field.name),
            getattr(explicit_state, field.name),
            rtol=1e-12,
            atol=1e-12,
        )


def test_injected_soil_changes_transition_and_run() -> None:
    replacement = NoRechargeSoil()
    default = _slot_model()
    injected = _slot_model(soil=replacement)
    assert injected.soil is replacement

    direct_forcing = HBVForcing(
        precip=tensor(np.array([10.0])),
        pet=tensor(np.array([0.0])),
        temp=tensor(np.array([5.0])),
    )
    default_state = default.init_state(dict(default.named_parameters()), batch_size=1)
    injected_state = injected.init_state(dict(injected.named_parameters()), batch_size=1)
    default_next, default_fluxes = default.transition(
        default_state,
        direct_forcing,
        dict(default.named_parameters()),
    )
    injected_next, injected_fluxes = injected.transition(
        injected_state,
        direct_forcing,
        dict(injected.named_parameters()),
    )
    assert torch.count_nonzero(default_fluxes.recharge)
    torch.testing.assert_close(injected_fluxes.recharge, torch.zeros_like(injected_fluxes.recharge))
    assert not torch.allclose(default_next.zone_sm, injected_next.zone_sm)

    default_observations, default_run_fluxes, default_final = default.run(_slot_forcing(), return_fluxes=True)
    injected_observations, injected_run_fluxes, injected_final = injected.run(_slot_forcing(), return_fluxes=True)
    assert torch.count_nonzero(default_run_fluxes.recharge)
    torch.testing.assert_close(injected_run_fluxes.recharge, torch.zeros_like(injected_run_fluxes.recharge))
    assert not torch.allclose(default_observations, injected_observations)
    assert not torch.allclose(default_final.zone_sm, injected_final.zone_sm)


def assert_golden(actual: torch.Tensor, expected: np.ndarray) -> None:
    torch.testing.assert_close(actual, tensor(expected), rtol=1e-12, atol=1e-12)


def _inputs(fixture: np.lib.npyio.NpzFile) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    scenario_shape = fixture["state.zone_sp"].shape
    forcing = {
        name: tensor(np.broadcast_to(fixture[f"forcing.{name}"][:, None, None, :], scenario_shape))
        for name in ("precip", "pet", "temp")
    }
    params = {
        name: tensor(np.broadcast_to(fixture["parameters"][None, :, None, None, index], scenario_shape))
        for index, name in enumerate(PARAM_NAMES)
    }
    return forcing, params


def _start(fixture: np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    initial = fixture[f"initial_state.{field}"]
    state = fixture[f"state.{field}"]
    if field == "routing_buffer":
        first = np.broadcast_to(initial[None, None, :, None, :], state.shape[:-2] + (1, state.shape[-1]))
        return np.concatenate((first, state[..., :-1, :]), axis=-2)
    first = np.broadcast_to(initial[None, None, :, None], state.shape[:-1] + (1,))
    return np.concatenate((first, state[..., :-1]), axis=-1)


def test_snow_processes_match_all_golden_scenarios() -> None:
    with np.load(GOLDEN, allow_pickle=False) as fixture:
        forcing, p = _inputs(fixture)
        start_sp, start_lw = tensor(_start(fixture, "zone_sp")), tensor(_start(fixture, "zone_lw"))
        rain, snow = processes.partition_precipitation(forcing["precip"], forcing["temp"], p["tt"], p["sfcf"])
        melt = processes.compute_melt(forcing["temp"], p["tt"], p["cfmax"], start_sp)
        refreeze_expected = (
            fixture["state.zone_sp"]
            - _start(fixture, "zone_sp")
            - fixture["flux.precip_snow"]
            + fixture["flux.snow_melt"]
        )
        refreeze = processes.compute_refreezing(forcing["temp"], p["tt"], p["cfmax"], p["cfr"], start_lw)
        new_sp, new_lw, outflow = processes.update_snow_pack(start_sp, start_lw, snow, melt, refreeze, p["cwh"])
        assert_golden(rain, fixture["flux.precip_rain"])
        assert_golden(snow, fixture["flux.precip_snow"])
        assert_golden(melt, fixture["flux.snow_melt"])
        assert_golden(refreeze, refreeze_expected)
        assert_golden(new_sp, fixture["state.zone_sp"])
        assert_golden(new_sp, fixture["flux.snow_pack"])
        assert_golden(new_lw, fixture["state.zone_lw"])
        assert_golden(new_lw, fixture["flux.liquid_water_in_snow"])
        assert_golden(outflow, fixture["flux.snow_input"] - fixture["flux.precip_rain"])


def test_soil_processes_match_all_golden_scenarios() -> None:
    with np.load(GOLDEN, allow_pickle=False) as fixture:
        forcing, p = _inputs(fixture)
        start_sm = tensor(_start(fixture, "zone_sm"))
        soil_input = tensor(fixture["flux.snow_input"])
        base = processes.compute_recharge(soil_input, start_sm, p["fc"], p["beta"])
        et = processes.compute_actual_et(forcing["pet"], start_sm, p["fc"], p["lp"])
        new_sm, overflow = processes.update_soil_moisture(start_sm, soil_input, base, et, p["fc"])
        assert_golden(et, fixture["flux.actual_et"])
        assert_golden(new_sm, fixture["state.zone_sm"])
        assert_golden(new_sm, fixture["flux.soil_moisture"])
        assert_golden(base + overflow, fixture["flux.recharge"])


def test_response_and_routing_match_all_golden_scenarios() -> None:
    with np.load(GOLDEN, allow_pickle=False) as fixture:
        _, p = _inputs(fixture)
        suz, slz = tensor(_start(fixture, "upper_zone")), tensor(_start(fixture, "lower_zone"))
        q0, q1 = processes.upper_zone_outflows(suz, p["k0"], p["k1"], p["uzl"])
        perc = processes.compute_percolation(suz, p["perc"])
        new_suz = processes.update_upper_zone(suz, tensor(fixture["flux.recharge"]), q0, q1, perc)
        q2 = processes.lower_zone_outflow(slz, p["k2"])
        new_slz = processes.update_lower_zone(slz, perc, q2)
        for actual, key in (
            (q0, "flux.q0"),
            (q1, "flux.q1"),
            (perc, "flux.percolation"),
            (new_suz, "state.upper_zone"),
            (new_suz, "flux.upper_zone"),
            (q2, "flux.q2"),
            (new_slz, "state.lower_zone"),
            (new_slz, "flux.lower_zone"),
        ):
            assert_golden(actual, fixture[key])
        np.testing.assert_allclose(
            fixture["flux.qgw"],
            fixture["flux.q0"] + fixture["flux.q1"] + fixture["flux.q2"],
            rtol=1e-12,
            atol=1e-12,
        )
        weights = processes.compute_triangular_weights(p["maxbas"])
        assert weights.shape == fixture["state.routing_buffer"].shape
        assert weights.dtype == torch.float64 and weights.device.type == "cpu"
        assert bool(torch.all(weights >= 0))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), rtol=1e-12, atol=1e-12)
        qsim, buffer = processes.convolve_routing(
            tensor(_start(fixture, "routing_buffer")), weights, tensor(fixture["flux.qgw"])
        )
        assert_golden(qsim, fixture["flux.streamflow"])
        assert_golden(buffer, fixture["state.routing_buffer"])


def test_process_boundaries() -> None:
    x = tensor(np.array([-1.0, 0.0, 1.0]))
    rain, snow = processes.partition_precipitation(torch.ones_like(x), x, tensor(0.0), tensor(2.0))
    torch.testing.assert_close(rain, tensor(np.array([0.0, 0.0, 1.0])))
    torch.testing.assert_close(snow, tensor(np.array([2.0, 2.0, 0.0])))
    torch.testing.assert_close(
        processes.compute_melt(x, tensor(0.0), tensor(3.0), tensor(np.array([9.0, 9.0, 2.0]))),
        tensor(np.array([0.0, 0.0, 2.0])),
    )
    torch.testing.assert_close(
        processes.compute_refreezing(x, tensor(0.0), tensor(3.0), tensor(0.5), tensor(np.array([1.0, 9.0, 9.0]))),
        tensor(np.array([1.0, 0.0, 0.0])),
    )
    recharge = processes.compute_recharge(
        tensor(np.array([1.0, 0.0, -1.0, 1.0])),
        tensor(np.array([0.0, 2.0, 2.0, 2.0])),
        tensor(np.array([4.0, 4.0, 4.0, 0.0])),
        torch.ones(4, dtype=torch.float64),
    )
    torch.testing.assert_close(recharge, torch.zeros_like(recharge))
    et = processes.compute_actual_et(
        tensor(np.array([4.0, 4.0, 4.0, 4.0])),
        tensor(np.array([1.0, 5.0, 2.0, 2.0])),
        tensor(np.array([10.0, 10.0, 0.0, 10.0])),
        tensor(np.array([0.5, 0.5, 0.5, 0.0])),
    )
    torch.testing.assert_close(et, tensor(np.array([0.8, 4.0, 0.0, 0.0])))
    sm, overflow = processes.update_soil_moisture(
        tensor(np.array([1.0, 9.0])), tensor(np.array([-5.0, 5.0])), tensor(0.0), tensor(0.0), tensor(10.0)
    )
    torch.testing.assert_close(sm, tensor(np.array([0.0, 10.0])))
    torch.testing.assert_close(overflow, tensor(np.array([0.0, 4.0])))
    q0, _ = processes.upper_zone_outflows(tensor(2.0), tensor(0.5), tensor(0.1), tensor(2.0))
    torch.testing.assert_close(q0, tensor(0.0))
    torch.testing.assert_close(processes.compute_percolation(tensor(-2.0), tensor(1.0)), tensor(0.0))
    torch.testing.assert_close(
        processes.update_upper_zone(tensor(1.0), tensor(0.0), tensor(2.0), tensor(0.0), tensor(0.0)), tensor(0.0)
    )
    torch.testing.assert_close(processes.update_lower_zone(tensor(1.0), tensor(0.0), tensor(2.0)), tensor(0.0))
    sp, lw, _ = processes.update_snow_pack(tensor(0.0), tensor(0.0), tensor(0.0), tensor(2.0), tensor(0.0), tensor(0.1))
    torch.testing.assert_close(sp, tensor(0.0))
    torch.testing.assert_close(lw, tensor(0.0))


@pytest.mark.parametrize("maxbas", [1.0, 3.0, 6.5, 7.0])
def test_triangular_weight_boundaries(maxbas: float) -> None:
    weights = processes.compute_triangular_weights(tensor(maxbas))
    assert weights.shape == (7,)
    assert bool(torch.isfinite(weights).all()) and bool((weights >= 0).all())
    torch.testing.assert_close(weights.sum(), tensor(1.0), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", [(12,), (3, 12)])
def test_state_round_trip_and_gradient(shape: tuple[int, ...]) -> None:
    value = torch.arange(int(np.prod(shape)), dtype=torch.float64).reshape(shape).requires_grad_()
    state = HBVState.from_flat(value)
    result = state.to_flat()
    torch.testing.assert_close(result, value)
    assert result.shape == shape and result.dtype == value.dtype and result.device.type == value.device.type
    result.sum().backward()
    assert value.grad is not None and bool(torch.isfinite(value.grad).all())
    with pytest.raises(FrozenInstanceError):
        state.zone_sp = tensor(0.0)  # ty: ignore[invalid-assignment]


def test_state_preserves_float32_and_available_device() -> None:
    requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = torch.arange(24, dtype=torch.float32, device=requested_device).reshape(2, 12).requires_grad_()
    result = HBVState.from_flat(value).to_flat()
    assert result.dtype == torch.float32 and result.device.type == requested_device.type
    result.sum().backward()
    assert value.grad is not None and bool(torch.isfinite(value.grad).all())


def _v(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


@pytest.mark.parametrize(
    "call",
    [
        lambda a: processes.partition_precipitation(*a[:4]),
        lambda a: processes.compute_melt(*a[:4]),
        lambda a: processes.compute_refreezing(*a[:5]),
        lambda a: processes.update_snow_pack(*a[:6]),
        lambda a: processes.compute_recharge(*a[:4]),
        lambda a: processes.compute_actual_et(*a[:4]),
        lambda a: processes.update_soil_moisture(*a[:5]),
        lambda a: processes.upper_zone_outflows(*a[:4]),
        lambda a: processes.compute_percolation(*a[:2]),
        lambda a: processes.update_upper_zone(*a[:5]),
        lambda a: processes.lower_zone_outflow(*a[:2]),
        lambda a: processes.update_lower_zone(*a[:3]),
    ],
)
def test_process_autograd_is_finite(
    call: Callable[[list[torch.Tensor]], torch.Tensor | tuple[torch.Tensor, ...]],
) -> None:
    args = [_v(v) for v in (2.3, 1.1, 1.7, 0.4, 0.3, 0.2)]
    output = call(args)
    values = output if isinstance(output, tuple) else (output,)
    loss = values[0].sum()
    for value in values[1:]:
        loss = loss + value.sum()
    loss.backward()
    for arg in args[: max(index for index, item in enumerate(args) if item.grad_fn is None) + 1]:
        if arg.grad is not None:
            assert bool(torch.isfinite(arg.grad).all())


def test_routing_autograd_and_guarded_recharge_gradient() -> None:
    maxbas = _v(3.4)
    weights = processes.compute_triangular_weights(maxbas)
    buffer = torch.linspace(0.1, 0.7, 7, dtype=torch.float64, requires_grad=True)
    qgw = _v(2.2)
    qsim, new_buffer = processes.convolve_routing(buffer, weights, qgw)
    (qsim + new_buffer.sum()).backward()
    for value in (maxbas, buffer, qgw):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all())
    beta = _v(1.3)
    processes.compute_recharge(_v(2.0), tensor(0.0), _v(10.0), beta).backward()
    assert beta.grad is not None and bool(torch.isfinite(beta.grad))
