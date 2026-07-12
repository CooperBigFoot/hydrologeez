import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hydrologeez.models.hbv import HBVFluxes, HBVForcing, HBVModel, HBVState

GOLDEN_PATH = Path(__file__).parent / "golden" / "hbv.npz"
RTOL = 1e-12
ATOL = 1e-12
PARAMETER_NAMES = (
    "tt",
    "cfmax",
    "sfcf",
    "cwh",
    "cfr",
    "fc",
    "lp",
    "beta",
    "k0",
    "k1",
    "k2",
    "perc",
    "uzl",
    "maxbas",
)
STATE_FIELDS = (
    "zone_sp",
    "zone_lw",
    "zone_sm",
    "upper_zone",
    "lower_zone",
    "routing_buffer",
)
FLUX_FIELDS = (
    "precip",
    "temp",
    "pet",
    "precip_rain",
    "precip_snow",
    "snow_pack",
    "snow_melt",
    "liquid_water_in_snow",
    "snow_input",
    "soil_moisture",
    "recharge",
    "actual_et",
    "upper_zone",
    "lower_zone",
    "q0",
    "q1",
    "q2",
    "percolation",
    "qgw",
    "streamflow",
)


def _tensor(array: Any) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64, device="cpu")


def _model(parameter_row: np.ndarray) -> HBVModel:
    return HBVModel(**dict(zip(PARAMETER_NAMES, _tensor(parameter_row), strict=True)))


def _forcing(fixture: np.lib.npyio.NpzFile, index: torch.Tensor) -> HBVForcing:
    return HBVForcing(
        precip=_tensor(fixture["forcing.precip"])[index],
        pet=_tensor(fixture["forcing.pet"])[index],
        temp=_tensor(fixture["forcing.temp"])[index],
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_complete_golden_fixture() -> None:
    with np.load(GOLDEN_PATH, allow_pickle=False) as fixture:
        forcing_index = torch.arange(2).repeat_interleave(4)
        parameter_index = torch.arange(2).repeat_interleave(2).repeat(2)
        initial_index = torch.arange(2).repeat(4)
        forcing = _forcing(fixture, forcing_index)
        parameter_table = _tensor(fixture["parameters"])
        parameters = {name: parameter_table[parameter_index, column] for column, name in enumerate(PARAMETER_NAMES)}
        state = HBVState(**{field: _tensor(fixture[f"initial_state.{field}"])[initial_index] for field in STATE_FIELDS})
        model = _model(fixture["parameters"][0])
        states: list[HBVState] = []
        fluxes: list[HBVFluxes] = []
        for time in range(12):
            step_forcing = HBVForcing(
                precip=forcing.precip[:, time],
                pet=forcing.pet[:, time],
                temp=forcing.temp[:, time],
            )
            state, step_fluxes = model.transition(state, step_forcing, parameters)
            states.append(state)
            fluxes.append(step_fluxes)

        for field in STATE_FIELDS:
            actual = torch.stack([getattr(item, field) for item in states], dim=1)
            source = fixture[f"state.{field}"]
            expected = _tensor(source.reshape((8,) + source.shape[3:]))
            _assert_close(actual, expected)
        for field in FLUX_FIELDS:
            actual = torch.stack([getattr(item, field) for item in fluxes], dim=1)
            source = fixture[f"flux.{field}"]
            expected = _tensor(source.reshape((8,) + source.shape[3:]))
            _assert_close(actual, expected)
        streamflow = torch.stack([item.streamflow for item in fluxes], dim=1)
        _assert_close(streamflow, _tensor(fixture["streamflow"].reshape(8, 12)))
        for field in STATE_FIELDS:
            source = fixture[f"final_state.{field}"]
            expected = _tensor(source.reshape((8,) + source.shape[3:]))
            _assert_close(getattr(state, field), expected)


def test_run_uses_registered_parameters_and_returns_fluxes() -> None:
    with np.load(GOLDEN_PATH, allow_pickle=False) as fixture:
        model = _model(fixture["parameters"][0])
        forcing = _forcing(fixture, torch.tensor([0]))
        observations, fluxes, final_state = model.run(forcing, return_fluxes=True)

        assert tuple(dict(model.named_parameters())) == PARAMETER_NAMES
        assert observations.shape == (1, 12)
        assert final_state.routing_buffer.shape == (1, 7)
        for field in FLUX_FIELDS:
            actual = getattr(fluxes, field)
            assert actual.shape == (1, 12)
            _assert_close(actual, _tensor(fixture[f"flux.{field}"][0, 0, 0])[None])
        _assert_close(observations, _tensor(fixture["streamflow"][0, 0, 0])[None])
        _assert_close(observations, fluxes.streamflow)
        for field in STATE_FIELDS:
            _assert_close(
                getattr(final_state, field),
                _tensor(fixture[f"final_state.{field}"][0, 0, 0])[None],
            )


def test_run_accepts_scalar_per_basin_and_per_timestep_parameters() -> None:
    with np.load(GOLDEN_PATH, allow_pickle=False) as fixture:
        model = _model(fixture["parameters"][0])
        forcing = _forcing(fixture, torch.arange(2))
        table = _tensor(fixture["parameters"])
        per_basin = {name: table[:, column] for column, name in enumerate(PARAMETER_NAMES)}
        mixed = dict(per_basin)
        mixed["sfcf"] = table[0, 2]
        per_basin["sfcf"] = table[0, 2].expand(2)
        mixed["tt"] = table[:, 0, None].expand(2, 12)

        mixed_result = model.run(forcing, mixed, return_fluxes=True)
        basin_result = model.run(forcing, per_basin, return_fluxes=True)
        for mixed_item, basin_item in zip(mixed_result, basin_result, strict=True):
            for field in dataclasses.fields(mixed_item) if dataclasses.is_dataclass(mixed_item) else ():
                _assert_close(getattr(mixed_item, field.name), getattr(basin_item, field.name))
        _assert_close(mixed_result[0], basin_result[0])


def test_warmup_changes_state_and_detaches_warmup_graph() -> None:
    with np.load(GOLDEN_PATH, allow_pickle=False) as fixture:
        model = _model(fixture["parameters"][0])
        warmup_leaves = [
            _tensor([[8.0, 6.0, 4.0]]).requires_grad_(),
            _tensor([[1.0, 1.0, 1.0]]).requires_grad_(),
            _tensor([[2.0, 3.0, 4.0]]).requires_grad_(),
        ]
        main_leaves = [
            _tensor([[7.0, 5.0]]).requires_grad_(),
            _tensor([[1.0, 1.0]]).requires_grad_(),
            _tensor([[3.0, 4.0]]).requires_grad_(),
        ]
        warmup = HBVForcing(*warmup_leaves)
        main = HBVForcing(*main_leaves)
        row = _tensor(fixture["parameters"][0])
        warmup_parameters = {
            name: row[column].detach().clone().requires_grad_() for column, name in enumerate(PARAMETER_NAMES)
        }
        main_parameters = {
            name: row[column].detach().clone().requires_grad_() for column, name in enumerate(PARAMETER_NAMES)
        }
        warmed = model.run(
            main,
            main_parameters,
            warmup=warmup,
            warmup_parameters=warmup_parameters,
        )
        unwarmed = model.run(main, main_parameters)
        assert not torch.allclose(warmed, unwarmed)
        warmed.sum().backward()
        assert all(leaf.grad is None for leaf in warmup_leaves)
        assert all(value.grad is None for value in warmup_parameters.values())
        assert all(leaf.grad is not None and torch.isfinite(leaf.grad).all() for leaf in main_leaves)
        main_gradients = [value.grad for value in main_parameters.values() if value.grad is not None]
        assert main_gradients
        assert all(torch.isfinite(gradient).all() for gradient in main_gradients)
        assert any(torch.count_nonzero(gradient) for gradient in main_gradients)


def test_complete_model_has_finite_parameter_gradients() -> None:
    values = _tensor([0.0, 3.0, 1.0, 0.1, 0.05, 100.0, 0.7, 2.0, 0.3, 0.1, 0.05, 2.0, 1.0, 3.0])
    model = _model(values.numpy())
    forcing = HBVForcing(
        precip=_tensor([[10.0, 8.0, 0.0, 12.0, 6.0, 4.0]]).requires_grad_(),
        pet=_tensor([[0.5, 0.5, 1.0, 1.0, 1.0, 1.0]]).requires_grad_(),
        temp=_tensor([[-3.0, -2.0, 3.0, 4.0, 5.0, 3.0]]).requires_grad_(),
    )
    observations, fluxes, final_state = model.run(forcing, return_fluxes=True)
    loss = observations.sum()
    loss = loss + 1e-3 * (
        fluxes.snow_pack.sum()
        + fluxes.snow_melt.sum()
        + fluxes.recharge.sum()
        + fluxes.actual_et.sum()
        + fluxes.qgw.sum()
        + final_state.zone_sm.sum()
        + final_state.upper_zone.sum()
        + final_state.lower_zone.sum()
        + final_state.routing_buffer.sum()
    )
    loss.backward()

    gradients = []
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        gradients.append(parameter.grad)
    assert len(gradients) == 14
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    for leaf in dataclasses.astuple(forcing):
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()
