import pytest
import torch

from hydrologeez.hybrid import (
    bounded_parameters,
    select_features,
    select_network_inputs,
)
from hydrologeez.models.hbv import HBVForcing, HBVModel


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
