from dataclasses import fields, replace

import pytest
import torch
from hcx import Forecast, Point, make_synthetic_batch
from torch import nn

from hydrologeez.hybrid import (
    NeuralParameterForecastModel,
    bounded_parameters,
    select_features,
    select_network_inputs,
)
from hydrologeez.models.hbv import HBVFluxes, HBVForcing, HBVModel, HBVState


def _model() -> HBVModel:
    return HBVModel(
        tt=torch.tensor(0.0, dtype=torch.float64),
        cfmax=torch.tensor(3.0, dtype=torch.float64),
        sfcf=torch.tensor(1.0, dtype=torch.float64),
        cwh=torch.tensor(0.1, dtype=torch.float64),
        cfr=torch.tensor(0.05, dtype=torch.float64),
        fc=torch.tensor(100.0, dtype=torch.float64),
        lp=torch.tensor(0.7, dtype=torch.float64),
        beta=torch.tensor(2.0, dtype=torch.float64),
        k0=torch.tensor(0.3, dtype=torch.float64),
        k1=torch.tensor(0.1, dtype=torch.float64),
        k2=torch.tensor(0.05, dtype=torch.float64),
        perc=torch.tensor(2.0, dtype=torch.float64),
        uzl=torch.tensor(1.0, dtype=torch.float64),
        maxbas=torch.tensor(3.0, dtype=torch.float64),
    )


def _main_forcing() -> HBVForcing:
    return HBVForcing(
        precip=torch.tensor([[4.0, 6.0, 0.0], [1.0, 8.0, 3.0]], dtype=torch.float64),
        pet=torch.tensor([[0.5, 0.7, 0.8], [0.4, 0.6, 0.7]], dtype=torch.float64),
        temp=torch.tensor([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], dtype=torch.float64),
    )


def _warmup_forcing() -> HBVForcing:
    return HBVForcing(
        precip=torch.tensor([[8.0, 2.0], [2.0, 5.0]], dtype=torch.float64),
        pet=torch.tensor([[0.5, 0.5], [0.4, 0.4]], dtype=torch.float64),
        temp=torch.tensor([[-2.0, 3.0], [1.0, 4.0]], dtype=torch.float64),
    )


def _scalar_parameters(model: HBVModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def _per_basin_parameters(parameters: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    return {name: value.expand(batch_size).clone() for name, value in parameters.items()}


def _constant_time_parameters(
    parameters: dict[str, torch.Tensor], batch_size: int, time_steps: int
) -> dict[str, torch.Tensor]:
    return {name: value.expand(batch_size, time_steps).clone() for name, value in parameters.items()}


def _component_routing_oracles(
    model: HBVModel,
    forcing: HBVForcing,
    parameters: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, n_components, time_steps = parameters["maxbas"].shape
    flat_batch_size = batch_size * n_components
    initial_parameters = {name: value[:, :, 0].reshape(flat_batch_size) for name, value in parameters.items()}
    state = model.init_state(initial_parameters, batch_size=flat_batch_size)
    shared_routing_buffer = state.routing_buffer.reshape(batch_size, n_components, model.routing_buffer_size)[:, 0, :]
    independent_routing_buffer = state.routing_buffer
    average_before: list[torch.Tensor] = []
    average_after: list[torch.Tensor] = []

    for time in range(time_steps):
        step_parameters = {name: value[:, :, time] for name, value in parameters.items()}
        flat_parameters = {name: value.reshape(flat_batch_size) for name, value in step_parameters.items()}
        step_forcing = HBVForcing(
            precip=forcing.precip[:, time, None].expand(batch_size, n_components).reshape(flat_batch_size),
            pet=forcing.pet[:, time, None].expand(batch_size, n_components).reshape(flat_batch_size),
            temp=forcing.temp[:, time, None].expand(batch_size, n_components).reshape(flat_batch_size),
        )
        pre_routing = model._pre_routing(state, step_forcing, flat_parameters)

        mean_qgw = pre_routing.qgw.reshape(batch_size, n_components).mean(dim=1)
        mean_routing_parameters = {name: step_parameters[name].mean(dim=1) for name in model.routing.introduces}
        shared_streamflow, shared_routing_buffer = model._apply_routing(
            mean_qgw,
            shared_routing_buffer,
            mean_routing_parameters,
        )
        component_streamflow, independent_routing_buffer = model._apply_routing(
            pre_routing.qgw,
            independent_routing_buffer,
            flat_parameters,
        )
        average_before.append(shared_streamflow)
        average_after.append(component_streamflow.reshape(batch_size, n_components).mean(dim=1))
        state = HBVState(
            zone_sp=pre_routing.snow_pack,
            zone_lw=pre_routing.liquid_water_in_snow,
            zone_sm=pre_routing.soil_moisture,
            upper_zone=pre_routing.upper_zone,
            lower_zone=pre_routing.lower_zone,
            routing_buffer=independent_routing_buffer,
        )

    return torch.stack(average_before, dim=1), torch.stack(average_after, dim=1)


def _pre_refactor_hbv_transition(
    model: HBVModel,
    state: HBVState,
    forcing: HBVForcing,
    parameters: dict[str, torch.Tensor],
) -> tuple[HBVState, HBVFluxes]:
    precip = forcing.precip
    pet = forcing.pet
    temp = forcing.temp

    p_rain, p_snow, new_sp, melt, new_lw, snow_input = model.snow(
        precip,
        temp,
        state.zone_sp,
        state.zone_lw,
        parameters,
    )
    new_sm, recharge_total, et_act = model.soil(
        snow_input,
        pet,
        state.zone_sm,
        parameters,
    )
    new_suz, new_slz, q0, q1, q2, perc, qgw = model.response(
        state.upper_zone,
        state.lower_zone,
        recharge_total,
        parameters,
    )
    qsim, new_buffer = model.routing(state.routing_buffer, qgw, parameters)

    new_state = HBVState(
        zone_sp=new_sp,
        zone_lw=new_lw,
        zone_sm=new_sm,
        upper_zone=new_suz,
        lower_zone=new_slz,
        routing_buffer=new_buffer,
    )
    fluxes = HBVFluxes(
        precip=precip,
        temp=temp,
        pet=pet,
        precip_rain=p_rain,
        precip_snow=p_snow,
        snow_pack=new_sp,
        snow_melt=melt,
        liquid_water_in_snow=new_lw,
        snow_input=snow_input,
        soil_moisture=new_sm,
        recharge=recharge_total,
        actual_et=et_act,
        upper_zone=new_suz,
        lower_zone=new_slz,
        q0=q0,
        q1=q1,
        q2=q2,
        percolation=perc,
        qgw=qgw,
        streamflow=qsim,
    )
    return new_state, fluxes


def test_hbv_run_components_singleton_is_exact_run_identity_with_warmup() -> None:
    model = _model()
    main = _main_forcing()
    warmup = _warmup_forcing()
    scalar = _scalar_parameters(model)
    per_basin = _per_basin_parameters(scalar, batch_size=2)
    main_time = _constant_time_parameters(scalar, batch_size=2, time_steps=3)
    warmup_time = _constant_time_parameters(scalar, batch_size=2, time_steps=2)
    main_time["fc"] = torch.tensor([[90.0, 100.0, 110.0], [105.0, 115.0, 125.0]], dtype=torch.float64)
    main_time["maxbas"] = torch.tensor([[2.0, 2.5, 3.0], [4.0, 4.5, 5.0]], dtype=torch.float64)
    warmup_time["fc"] = torch.tensor([[80.0, 85.0], [110.0, 120.0]], dtype=torch.float64)
    warmup_time["maxbas"] = torch.tensor([[1.5, 2.0], [5.0, 5.5]], dtype=torch.float64)

    for parameter_form in (scalar, per_basin, main_time):
        expected = model.run(main, parameters=parameter_form)
        actual = model.run_components(main, parameter_form, n_components=1)
        assert torch.equal(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    full_main = {name: value[:, None, :] for name, value in main_time.items()}
    full_warmup = {name: value[:, None, :] for name, value in warmup_time.items()}
    expected_warmed = model.run(
        main,
        parameters=main_time,
        warmup=warmup,
        warmup_parameters=warmup_time,
    )
    actual_warmed = model.run_components(
        main,
        full_main,
        n_components=1,
        warmup=warmup,
        warmup_parameters=full_warmup,
    )
    assert torch.equal(actual_warmed, expected_warmed)
    torch.testing.assert_close(actual_warmed, expected_warmed, rtol=1e-12, atol=1e-12)

    differentiable_main = {name: value.detach().clone().requires_grad_() for name, value in full_main.items()}
    differentiable_warmup = {name: value.detach().clone().requires_grad_() for name, value in full_warmup.items()}
    differentiable_output = model.run_components(
        main,
        differentiable_main,
        n_components=1,
        warmup=warmup,
        warmup_parameters=differentiable_warmup,
    )
    differentiable_output.sum().backward()
    assert all(value.grad is None for value in differentiable_warmup.values())
    fc_gradient = differentiable_main["fc"].grad
    assert fc_gradient is not None
    assert torch.isfinite(fc_gradient).all()
    assert torch.count_nonzero(fc_gradient) > 0


def test_hbv_run_components_averages_qgw_before_one_shared_routing_step() -> None:
    model = _model()
    forcing = HBVForcing(
        precip=torch.tensor([[8.0, 0.0, 12.0, 3.0, 0.0, 6.0, 1.0, 9.0]], dtype=torch.float64),
        pet=torch.tensor([[0.4, 0.6, 0.5, 0.8, 0.7, 0.3, 0.5, 0.6]], dtype=torch.float64),
        temp=torch.tensor([[-1.0, 2.0, 4.0, 1.0, 3.0, -0.5, 2.5, 5.0]], dtype=torch.float64),
    )
    batch_size = 1
    n_components = 2
    time_steps = forcing.precip.shape[1]
    component_fraction = torch.tensor([[[0.3], [0.7]]], dtype=torch.float64)
    parameters = {
        name: (low + component_fraction * (high - low)).expand(batch_size, n_components, time_steps).clone()
        for name, (low, high) in model.parameter_bounds.items()
    }
    assert all(not torch.equal(value[:, 0, :], value[:, 1, :]) for value in parameters.values())
    assert not torch.equal(parameters["maxbas"][:, 0, :], parameters["maxbas"][:, 1, :])

    actual = model.run_components(forcing, parameters, n_components=n_components)
    average_before, average_after = _component_routing_oracles(model, forcing, parameters)

    assert actual.shape == (batch_size, time_steps)
    assert actual.dtype == forcing.precip.dtype
    assert actual.device == forcing.precip.device
    torch.testing.assert_close(actual, average_before, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(actual, average_after, rtol=1e-12, atol=1e-12)


def test_hbv_transition_internal_stages_match_pre_refactor_transition() -> None:
    model = _model()
    parameters = _per_basin_parameters(_scalar_parameters(model), batch_size=2)
    parameters["fc"] = torch.tensor([90.0, 120.0], dtype=torch.float64)
    parameters["maxbas"] = torch.tensor([2.5, 4.5], dtype=torch.float64)
    state = HBVState(
        zone_sp=torch.tensor([2.0, 1.0], dtype=torch.float64),
        zone_lw=torch.tensor([0.2, 0.1], dtype=torch.float64),
        zone_sm=torch.tensor([40.0, 80.0], dtype=torch.float64),
        upper_zone=torch.tensor([5.0, 3.0], dtype=torch.float64),
        lower_zone=torch.tensor([7.0, 9.0], dtype=torch.float64),
        routing_buffer=torch.tensor(
            [
                [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            ],
            dtype=torch.float64,
        ),
    )
    main = _main_forcing()
    forcing = HBVForcing(
        precip=main.precip[:, 0],
        pet=main.pet[:, 0],
        temp=main.temp[:, 0],
    )

    expected_state, expected_fluxes = _pre_refactor_hbv_transition(
        model,
        state,
        forcing,
        parameters,
    )
    actual_state, actual_fluxes = model.transition(state, forcing, parameters)
    pre_routing = model._pre_routing(state, forcing, parameters)
    staged_streamflow, staged_routing_buffer = model._apply_routing(
        pre_routing.qgw,
        state.routing_buffer,
        parameters,
    )

    for field in fields(HBVState):
        torch.testing.assert_close(
            getattr(actual_state, field.name),
            getattr(expected_state, field.name),
            rtol=1e-12,
            atol=1e-12,
        )
    for field in fields(HBVFluxes):
        torch.testing.assert_close(
            getattr(actual_fluxes, field.name),
            getattr(expected_fluxes, field.name),
            rtol=1e-12,
            atol=1e-12,
        )
        if field.name != "streamflow":
            torch.testing.assert_close(
                getattr(pre_routing, field.name),
                getattr(expected_fluxes, field.name),
                rtol=1e-12,
                atol=1e-12,
            )
    torch.testing.assert_close(
        staged_streamflow,
        expected_fluxes.streamflow,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        staged_routing_buffer,
        expected_state.routing_buffer,
        rtol=1e-12,
        atol=1e-12,
    )


def test_hbv_run_accepts_time_varying_parameters_during_warmup() -> None:
    model = _model()
    scalar_parameters = _scalar_parameters(model)
    main_parameters = {
        name: value.requires_grad_()
        for name, value in _constant_time_parameters(scalar_parameters, batch_size=2, time_steps=3).items()
    }
    warmup_parameters = {
        name: value.requires_grad_()
        for name, value in _constant_time_parameters(scalar_parameters, batch_size=2, time_steps=2).items()
    }
    main_parameters["fc"] = torch.tensor(
        [[90.0, 100.0, 110.0], [105.0, 115.0, 125.0]], dtype=torch.float64, requires_grad=True
    )
    warmup_parameters["fc"] = torch.tensor([[80.0, 85.0], [110.0, 120.0]], dtype=torch.float64, requires_grad=True)

    observations = model.run(
        _main_forcing(),
        parameters=main_parameters,
        warmup=_warmup_forcing(),
        warmup_parameters=warmup_parameters,
    )

    assert observations.shape == (2, 3)
    observations.sum().backward()
    assert all(value.grad is None for value in warmup_parameters.values())
    fc_gradient = main_parameters["fc"].grad
    assert fc_gradient is not None
    assert torch.isfinite(fc_gradient).all()
    assert torch.count_nonzero(fc_gradient) > 0


def test_constant_time_parameters_match_scalar_and_per_basin_parameters() -> None:
    model = _model()
    scalar_parameters = _scalar_parameters(model)
    per_basin_parameters = _per_basin_parameters(scalar_parameters, batch_size=2)
    main_time_parameters = _constant_time_parameters(scalar_parameters, batch_size=2, time_steps=3)
    warmup_time_parameters = _constant_time_parameters(scalar_parameters, batch_size=2, time_steps=2)
    main = _main_forcing()
    warmup = _warmup_forcing()

    scalar_main = model.run(main, parameters=scalar_parameters)
    per_basin_main = model.run(main, parameters=per_basin_parameters)
    time_varying_main = model.run(main, parameters=main_time_parameters)
    torch.testing.assert_close(per_basin_main, scalar_main, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(time_varying_main, scalar_main, rtol=1e-12, atol=1e-12)

    scalar_warmed = model.run(
        main,
        parameters=scalar_parameters,
        warmup=warmup,
        warmup_parameters=scalar_parameters,
    )
    per_basin_warmed = model.run(
        main,
        parameters=per_basin_parameters,
        warmup=warmup,
        warmup_parameters=per_basin_parameters,
    )
    time_varying_warmed = model.run(
        main,
        parameters=main_time_parameters,
        warmup=warmup,
        warmup_parameters=warmup_time_parameters,
    )
    torch.testing.assert_close(per_basin_warmed, scalar_warmed, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(time_varying_warmed, scalar_warmed, rtol=1e-12, atol=1e-12)


def test_select_features_resolves_names_and_indices_in_requested_order() -> None:
    scalar_dynamic = torch.tensor(
        [
            [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]],
            [[3.0, 30.0, 300.0], [4.0, 40.0, 400.0]],
        ]
    )
    scalar_static = torch.tensor([[5.0, 50.0], [6.0, 60.0]])

    selected_dynamic, selected_static = select_features(
        scalar_dynamic,
        scalar_static,
        dynamic_inputs=("precip", "pet", "temp"),
        static_inputs=("elevation", "area"),
        dynamic_features=("temp", 0),
        static_features=("area",),
    )

    torch.testing.assert_close(
        selected_dynamic,
        torch.tensor(
            [
                [[100.0, 1.0], [200.0, 2.0]],
                [[300.0, 3.0], [400.0, 4.0]],
            ]
        ),
    )
    torch.testing.assert_close(selected_static, torch.tensor([[50.0], [60.0]]))


def test_select_network_inputs_expands_static_features_across_time() -> None:
    scalar_dynamic = torch.tensor([[[1.0, 10.0], [2.0, 20.0]]])
    scalar_static = torch.tensor([[100.0, 1000.0]])

    selected = select_network_inputs(
        scalar_dynamic,
        scalar_static,
        dynamic_inputs=("precip", "temp"),
        static_inputs=("area", "elevation"),
        dynamic_features=("temp",),
        static_features=("area",),
    )

    torch.testing.assert_close(selected, torch.tensor([[[10.0, 100.0], [20.0, 100.0]]]))


def test_select_features_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown scalar_dynamic feature 'pet'"):
        select_features(
            torch.zeros(2, 3, 1),
            None,
            dynamic_inputs=("precip",),
            static_inputs=(),
            dynamic_features=("pet",),
        )


@pytest.mark.parametrize(
    ("dynamic_inputs", "dynamic_features", "message"),
    [
        (("precip", "precip"), ("precip",), "input names contain duplicate"),
        (("precip",), (1,), "feature index 1 is outside"),
        (("precip",), ("precip", 0), "selectors resolve to duplicate"),
    ],
)
def test_select_features_rejects_invalid_selector_definitions(
    dynamic_inputs: tuple[str, ...],
    dynamic_features: tuple[str | int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_features(
            torch.zeros(2, 3, len(dynamic_inputs)),
            None,
            dynamic_inputs=dynamic_inputs,
            static_inputs=(),
            dynamic_features=dynamic_features,
        )


def test_select_features_rejects_missing_quadrants_and_width_mismatches() -> None:
    with pytest.raises(ValueError, match="scalar_static is required"):
        select_features(
            torch.zeros(2, 3, 1),
            None,
            dynamic_inputs=("precip",),
            static_inputs=("area",),
            static_features=("area",),
        )
    with pytest.raises(ValueError, match="scalar_dynamic width does not match"):
        select_features(
            torch.zeros(2, 3, 2),
            None,
            dynamic_inputs=("precip",),
            static_inputs=(),
            dynamic_features=("precip",),
        )


@pytest.mark.parametrize(
    ("raw_shape", "parameter_shape"),
    [
        ((14,), ()),
        ((2, 14), (2,)),
        ((2, 3, 14), (2, 3)),
    ],
)
def test_bounded_parameters_preserves_order_bounds_and_supported_shapes(
    raw_shape: tuple[int, ...], parameter_shape: tuple[int, ...]
) -> None:
    model = _model()
    raw = torch.linspace(-2.0, 2.0, int(torch.tensor(raw_shape).prod().item())).reshape(raw_shape)

    parameters = bounded_parameters(raw, model.parameter_bounds)

    assert tuple(parameters) == tuple(model.parameter_bounds)
    for name, value in parameters.items():
        low, high = model.parameter_bounds[name]
        assert value.shape == parameter_shape
        assert torch.all(value > low)
        assert torch.all(value < high)


def test_bounded_parameters_width_follows_hbv_parameter_bounds_slot_order() -> None:
    model = _model()
    expected_names = (
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

    parameters = bounded_parameters(torch.zeros(14), model.parameter_bounds)

    assert tuple(model.parameter_bounds) == expected_names
    assert tuple(parameters) == expected_names
    with pytest.raises(ValueError, match="raw parameter width 13.*width 14"):
        bounded_parameters(torch.zeros(13), model.parameter_bounds)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_parameters_preserves_dtype_device_and_global_default(
    dtype: torch.dtype,
) -> None:
    model = _model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.zeros(2, 3, 14, dtype=dtype, device=device)
    default_dtype = torch.get_default_dtype()

    parameters = bounded_parameters(raw, model.parameter_bounds)

    assert torch.get_default_dtype() == default_dtype
    assert all(value.dtype == dtype for value in parameters.values())
    assert all(value.device == device for value in parameters.values())


class _TimeVaryingParameterNetwork(nn.Module):
    def __init__(self, input_width: int, parameter_width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_width, parameter_width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_neural_parameter_forecast_model_returns_point_forecast() -> None:
    batch = make_synthetic_batch(
        batch_size=2,
        input_length=5,
        output_length=3,
        scalar_dynamic_features=3,
        scalar_static_features=1,
        include_gridded_dynamic=False,
        include_gridded_static=False,
        dtype=torch.float64,
    )
    scalar_dynamic = torch.tensor(
        [
            [
                [4.0, 0.5, 2.0],
                [6.0, 0.7, 3.0],
                [0.0, 0.8, 4.0],
                [3.0, 0.6, 1.0],
                [2.0, 0.4, 2.0],
            ],
            [
                [1.0, 0.4, 1.0],
                [8.0, 0.6, 2.0],
                [3.0, 0.7, 3.0],
                [5.0, 0.5, 4.0],
                [0.0, 0.3, 2.0],
            ],
        ],
        dtype=batch.target.dtype,
        device=batch.target.device,
    )
    batch = replace(batch, scalar_dynamic=scalar_dynamic)
    hbv = _model().to(dtype=batch.target.dtype, device=batch.target.device)
    network = _TimeVaryingParameterNetwork(
        input_width=4,
        parameter_width=len(hbv.parameter_bounds),
    ).to(dtype=batch.target.dtype, device=batch.target.device)
    model = NeuralParameterForecastModel(
        hbv,
        network,
        dynamic_inputs=("precip", "pet", "temp"),
        static_inputs=("area",),
        physics_forcing={"precip": "precip", "pet": "pet", "temp": "temp"},
        network_dynamic_inputs=("precip", "pet", "temp"),
        network_static_inputs=("area",),
        forcing_factory=HBVForcing,
        output_specification=Point(),
    )

    forecast = model(batch)

    assert isinstance(forecast, Forecast)
    assert forecast.prediction.shape == (batch.target.shape[0], batch.target.shape[-1])
    assert forecast.prediction.dtype == batch.target.dtype
    assert forecast.prediction.device == batch.target.device
    assert torch.isfinite(forecast.prediction).all()
    assert forecast.sample_ids is batch.metadata.sample_ids
    assert forecast.input_end_indices is batch.metadata.input_end_indices
    assert forecast.target_fill_mask is batch.metadata.target_fill_mask


def test_neural_parameter_forecast_model_trains_on_synthetic_batch() -> None:
    torch.manual_seed(1729)
    device = torch.device("cpu")
    dtype = torch.float64
    dynamic_inputs = ("precip", "pet", "temp")
    static_inputs = ("area",)
    batch = make_synthetic_batch(
        batch_size=4,
        input_length=12,
        output_length=6,
        scalar_dynamic_features=len(dynamic_inputs),
        scalar_static_features=len(static_inputs),
        include_gridded_dynamic=False,
        include_gridded_static=False,
        dtype=dtype,
        device=device,
        seed=1729,
    )
    hbv = _model().to(dtype=dtype, device=device)
    network = _TimeVaryingParameterNetwork(
        input_width=len(dynamic_inputs) + len(static_inputs),
        parameter_width=len(hbv.parameter_bounds),
    ).to(dtype=dtype, device=device)
    model = NeuralParameterForecastModel(
        hbv,
        network,
        dynamic_inputs=dynamic_inputs,
        static_inputs=static_inputs,
        physics_forcing={"precip": "precip", "pet": "pet", "temp": "temp"},
        network_dynamic_inputs=dynamic_inputs,
        network_static_inputs=static_inputs,
        forcing_factory=HBVForcing,
        output_specification=Point(),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
    losses: list[float] = []

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        forecast = model(batch)
        assert forecast.prediction.dtype == batch.target.dtype == dtype
        assert forecast.prediction.device == batch.target.device == device
        loss = torch.mean((forecast.prediction - batch.target) ** 2)
        loss.backward()

        nonzero_weight_gradient = False
        for name, parameter in network.named_parameters():
            if name.endswith("weight"):
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
                nonzero_weight_gradient |= bool(torch.count_nonzero(parameter.grad))
        assert nonzero_weight_gradient

        optimizer.step()
        losses.append(loss.detach().item())

    early_loss = sum(losses[:5]) / 5
    late_loss = sum(losses[-5:]) / 5
    assert late_loss < early_loss
