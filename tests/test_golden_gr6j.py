from collections.abc import Mapping
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hydrologeez.models.gr6j import GR6J, GR6JFluxes, GR6JForcing, State

STATE_FIELDS = ("production_store", "routing_store", "exponential_store", "uh1", "uh2")
FLUX_FIELDS = (
    "pet",
    "precip",
    "production_store",
    "net_rainfall",
    "storage_infiltration",
    "actual_et",
    "percolation",
    "effective_rainfall",
    "q9",
    "q1",
    "routing_store",
    "exchange",
    "actual_exchange_routing",
    "actual_exchange_direct",
    "actual_exchange_total",
    "qr",
    "qrexp",
    "exponential_store",
    "qd",
    "streamflow",
)
PARAMETER_NAMES = ("x1", "x2", "x3", "x4", "x5", "x6")
GOLDEN_PATH = Path(__file__).parent / "golden" / "gr6j.npz"


def _tensor(value: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def _assert_golden(actual: torch.Tensor, expected: torch.Tensor, key: str) -> None:
    assert actual.dtype == torch.float64, key
    assert actual.device.type == "cpu", key
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12, msg=key)


class _FixtureInitialStateGR6J(GR6J):
    def __init__(
        self,
        initial_state: State,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
    ) -> None:
        super().__init__(x1, x2, x3, x4, x5, x6)
        self._initial_state = initial_state

    def init_state(
        self,
        parameters: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> State:
        assert self._initial_state.production_store.shape[0] == batch_size
        return self._initial_state


def _state_observation(state: State, fluxes: GR6JFluxes) -> torch.Tensor:
    del fluxes
    return state.to_flat()


def _scalar_parameters() -> dict[str, torch.Tensor]:
    return {
        "x1": torch.tensor(350.0, dtype=torch.float64),
        "x2": torch.tensor(-1.2, dtype=torch.float64),
        "x3": torch.tensor(90.0, dtype=torch.float64),
        "x4": torch.tensor(2.4, dtype=torch.float64),
        "x5": torch.tensor(0.15, dtype=torch.float64),
        "x6": torch.tensor(12.0, dtype=torch.float64),
    }


def test_complete_golden_trajectory() -> None:
    with np.load(GOLDEN_PATH, allow_pickle=False) as fixture:
        scenarios = list(product(range(2), range(2), range(2)))
        forcing_indices = [scenario[0] for scenario in scenarios]
        parameter_indices = [scenario[1] for scenario in scenarios]
        initial_indices = [scenario[2] for scenario in scenarios]
        fixture_parameters = _tensor(fixture["parameters"])
        parameters = {
            name: fixture_parameters[parameter_indices, column] for column, name in enumerate(PARAMETER_NAMES)
        }
        forcing = GR6JForcing(
            precip=_tensor(fixture["forcing.precip"])[forcing_indices],
            pet=_tensor(fixture["forcing.pet"])[forcing_indices],
        )
        initial_state = State(
            **{field: _tensor(fixture[f"initial_state.{field}"])[initial_indices] for field in STATE_FIELDS}
        )
        model = _FixtureInitialStateGR6J(
            initial_state,
            parameters["x1"],
            parameters["x2"],
            parameters["x3"],
            parameters["x4"],
            parameters["x5"],
            parameters["x6"],
        )

        states, fluxes, final_state = model.run(
            forcing,
            parameters,
            observation_operator=_state_observation,
            return_fluxes=True,
        )

        assert states.shape[:2] == (8, 12)
        state_results = {
            "production_store": states[..., 0],
            "routing_store": states[..., 1],
            "exponential_store": states[..., 2],
            "uh1": states[..., 3:23],
            "uh2": states[..., 23:63],
        }
        for field in STATE_FIELDS:
            actual = state_results[field]
            assert actual.shape[:2] == (8, 12), field
            _assert_golden(actual, _tensor(fixture[f"state.{field}"]).reshape(actual.shape), f"state.{field}")
        for field in FLUX_FIELDS:
            actual = getattr(fluxes, field)
            assert actual.shape == (8, 12), field
            _assert_golden(actual, _tensor(fixture[f"flux.{field}"]).reshape(8, 12), f"flux.{field}")
        _assert_golden(fluxes.streamflow, _tensor(fixture["streamflow"]).reshape(8, 12), "streamflow")
        for field in STATE_FIELDS:
            actual = getattr(final_state, field)
            assert actual.shape[0] == 8, field
            _assert_golden(
                actual,
                _tensor(fixture[f"final_state.{field}"]).reshape(actual.shape),
                f"final_state.{field}",
            )


def test_warmup_carries_state_into_main_forcing() -> None:
    parameters = _scalar_parameters()
    precip = torch.tensor([[0.0, 0.0, 1.5, 12.0, 30.0, 4.0]], dtype=torch.float64)
    pet = torch.tensor([[2.0, 3.0, 2.5, 1.0, 0.5, 2.0]], dtype=torch.float64)
    model = GR6J(
        parameters["x1"],
        parameters["x2"],
        parameters["x3"],
        parameters["x4"],
        parameters["x5"],
        parameters["x6"],
    )
    forcing = GR6JForcing(precip=precip, pet=pet)

    full = model.run(forcing)
    warmed = model.run(
        GR6JForcing(precip=precip[:, 3:6], pet=pet[:, 3:6]),
        warmup=GR6JForcing(precip=precip[:, 0:3], pet=pet[:, 0:3]),
    )
    main_only = model.run(GR6JForcing(precip=precip[:, 3:6], pet=pet[:, 3:6]))

    torch.testing.assert_close(warmed, full[:, 3:6], rtol=1e-12, atol=1e-12)
    assert not torch.allclose(main_only, full[:, 3:6], rtol=1e-12, atol=1e-12)


def test_registered_parameters_receive_gradients() -> None:
    parameters = _scalar_parameters()
    model = GR6J(
        parameters["x1"],
        parameters["x2"],
        parameters["x3"],
        parameters["x4"],
        parameters["x5"],
        parameters["x6"],
    )
    forcing = GR6JForcing(
        precip=torch.tensor([[0.0, 5.0, 20.0, 3.0, 12.0, 0.0]], dtype=torch.float64),
        pet=torch.tensor([[2.0, 1.0, 0.5, 3.0, 1.5, 4.0]], dtype=torch.float64),
    )

    loss = model.run(forcing).sum()
    loss.backward()

    named_parameters = dict(model.named_parameters())
    assert tuple(named_parameters) == tuple(model.parameter_bounds) == ("x1", "x4", "x2", "x3", "x5", "x6")
    assert all(isinstance(getattr(model, name), nn.Parameter) for name in PARAMETER_NAMES)
    gradients = [parameter.grad for parameter in named_parameters.values()]
    assert all(gradient is not None for gradient in gradients)
    present_gradients = [gradient for gradient in gradients if gradient is not None]
    assert all(torch.isfinite(gradient).all() for gradient in present_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in present_gradients)
