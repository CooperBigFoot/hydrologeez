from __future__ import annotations

import pytest
import torch
from torch import nn

import hydrologeez.calibration as calibration
from hydrologeez.calibration.adapter import (
    GR6J_SPEC,
    HBV_SPEC,
    LOWER_BOUNDS,
    PARAM_NAMES,
    UPPER_BOUNDS,
    ParamSpec,
    array_to_model,
    array_to_parameters,
    bounds_array,
    flat_to_model,
    model_to_flat,
    params_to_array,
)
from hydrologeez.models.gr6j.model import GR6J
from hydrologeez.models.hbv import constants as hbv_constants
from hydrologeez.models.hbv.model import HBVModel


def _gr6j(*, nh: int = 17) -> GR6J:
    values = (11.0, -2.0, 7.0, 3.0, -1.0, 5.0)
    return GR6J(*(torch.tensor(value, dtype=torch.float64) for value in values), nh=nh)


def _hbv() -> HBVModel:
    values = (2.0, 9.0, 0.7, 0.15, 0.03, 600.0, 0.4, 5.0, 0.8, 0.2, 0.1, 4.0, 80.0, 3.0)
    return HBVModel(
        **{name: torch.tensor(value, dtype=torch.float64) for name, value in zip(HBV_SPEC.names, values, strict=True)}
    )


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.second = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64), requires_grad=False)
        self.first = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        self.extra = nn.Parameter(torch.tensor([8.0, 9.0], dtype=torch.float64))
        self.register_buffer("offset", torch.tensor([5.0], dtype=torch.float64))
        self.label = "structure"


TINY_SPEC = ParamSpec(("first", "second"), (-10.0, -20.0), (10.0, 20.0))


def test_specs_have_exact_canonical_values() -> None:
    assert PARAM_NAMES == ("x1", "x2", "x3", "x4", "x5", "x6")
    assert LOWER_BOUNDS == (1.0, -5.0, 1.0, 0.5, -4.0, 1.0)
    assert UPPER_BOUNDS == (2500.0, 5.0, 1000.0, 10.0, 4.0, 50.0)
    assert (GR6J_SPEC.lower[-1], GR6J_SPEC.upper[-1]) == (1.0, 50.0)
    assert HBV_SPEC.names == hbv_constants.PARAM_NAMES
    assert len(HBV_SPEC.names) == 14
    assert tuple(zip(HBV_SPEC.lower, HBV_SPEC.upper, strict=True)) == tuple(
        hbv_constants.PARAM_BOUNDS[name] for name in hbv_constants.PARAM_NAMES
    )
    assert (HBV_SPEC.lower[-1], HBV_SPEC.upper[-1]) == (1.0, 7.0)


def test_bounds_array_dtype_and_device() -> None:
    lower, upper = bounds_array()
    assert lower.dtype == upper.dtype == torch.float64
    assert lower.device.type == upper.device.type == "cpu"
    torch.testing.assert_close(lower, torch.tensor(LOWER_BOUNDS, dtype=torch.float64))
    torch.testing.assert_close(upper, torch.tensor(UPPER_BOUNDS, dtype=torch.float64))

    requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    explicit_lower, explicit_upper = bounds_array(HBV_SPEC, dtype=torch.float32, device=requested)
    assert explicit_lower.dtype == explicit_upper.dtype == torch.float32
    assert explicit_lower.device.type == explicit_upper.device.type == torch.device(requested).type
    torch.testing.assert_close(explicit_lower.cpu(), torch.tensor(HBV_SPEC.lower, dtype=torch.float32))
    torch.testing.assert_close(explicit_upper.cpu(), torch.tensor(HBV_SPEC.upper, dtype=torch.float32))
    assert lower.data_ptr() != bounds_array()[0].data_ptr()


@pytest.mark.parametrize(
    ("model", "spec", "expected"),
    [
        (_gr6j(), GR6J_SPEC, (11.0, -2.0, 7.0, 3.0, -1.0, 5.0)),
        (_hbv(), HBV_SPEC, (2.0, 9.0, 0.7, 0.15, 0.03, 600.0, 0.4, 5.0, 0.8, 0.2, 0.1, 4.0, 80.0, 3.0)),
    ],
)
def test_params_to_array_uses_spec_order(model: nn.Module, spec: ParamSpec, expected: tuple[float, ...]) -> None:
    result = params_to_array(model, spec)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    result.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_array_to_parameters_is_differentiable_view_for_scalar_and_batch() -> None:
    theta = torch.arange(1, 7, dtype=torch.float64, requires_grad=True)
    parameters = array_to_parameters(theta)
    assert tuple(parameters) == GR6J_SPEC.names
    assert all(value.ndim == 0 for value in parameters.values())
    loss = torch.stack([(index + 1) * value.square() for index, value in enumerate(parameters.values())]).sum()
    loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all() and torch.count_nonzero(theta.grad) == theta.numel()

    batched = torch.arange(18, dtype=torch.float64).reshape(3, 6).requires_grad_()
    mapped = array_to_parameters(batched)
    assert all(value.shape == (3,) for value in mapped.values())
    torch.testing.assert_close(mapped["x4"], batched[:, 3])
    assert mapped["x4"].untyped_storage().data_ptr() == batched.untyped_storage().data_ptr()
    mapped["x4"].sum().backward()
    assert batched.grad is not None
    torch.testing.assert_close(batched.grad[:, 3], torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize(("model", "spec"), [(_gr6j(), GR6J_SPEC), (_hbv(), HBV_SPEC)])
def test_named_model_round_trip(model: nn.Module, spec: ParamSpec) -> None:
    theta = params_to_array(model, spec) + 0.25
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    rebuilt = array_to_model(model, theta, spec)
    assert tuple(dict(rebuilt.named_parameters())) == spec.names
    torch.testing.assert_close(params_to_array(rebuilt, spec), theta)
    for name, parameter in rebuilt.named_parameters():
        assert isinstance(parameter, nn.Parameter)
        assert parameter.dtype == theta.dtype and parameter.device == theta.device
        assert parameter.requires_grad == flags[name]
    torch.testing.assert_close(params_to_array(model, spec), theta - 0.25)


def test_batched_reconstruction_and_independence() -> None:
    template = _gr6j(nh=23)
    theta = torch.arange(12, dtype=torch.float64).reshape(2, 6)
    rebuilt = array_to_model(template, theta)
    assert rebuilt.nh == template.nh == 23
    assert all(parameter.shape == (2,) for parameter in rebuilt.parameters())
    torch.testing.assert_close(params_to_array(rebuilt), theta)
    with torch.no_grad():
        rebuilt.x1.add_(100.0)
        template.x2.add_(50.0)
    torch.testing.assert_close(template.x1, torch.tensor(11.0, dtype=torch.float64))
    torch.testing.assert_close(rebuilt.x2, theta[:, 1])


def test_partial_reconstruction_preserves_unselected_structure() -> None:
    template = _Tiny().eval()
    rebuilt = array_to_model(template, torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64), TINY_SPEC)
    assert not rebuilt.training
    assert rebuilt.label == template.label == "structure"
    torch.testing.assert_close(rebuilt.extra, template.extra)
    torch.testing.assert_close(rebuilt.offset, template.offset)
    assert rebuilt.extra is not template.extra and rebuilt.offset is not template.offset
    with torch.no_grad():
        rebuilt.extra.add_(1.0)
        rebuilt.offset.add_(1.0)
    torch.testing.assert_close(template.extra, torch.tensor([8.0, 9.0], dtype=torch.float64))
    torch.testing.assert_close(template.offset, torch.tensor([5.0], dtype=torch.float64))
    assert rebuilt.first.requires_grad
    assert not rebuilt.second.requires_grad


def test_hbv_reconstruction_preserves_structure_and_mode() -> None:
    template = _hbv().eval()
    rebuilt = array_to_model(template, params_to_array(template, HBV_SPEC), HBV_SPEC)
    assert rebuilt.n_zones == template.n_zones == 1
    assert rebuilt.routing_buffer_size == template.routing_buffer_size == 7
    assert not rebuilt.training


def test_generic_flatten_round_trip_registration_order_and_independence() -> None:
    source = _Tiny().eval()
    flat, aux = model_to_flat(source)
    expected = torch.tensor([3.0, 4.0, 1.0, 2.0, 8.0, 9.0], dtype=torch.float64)
    torch.testing.assert_close(flat, expected)
    flat.sum().backward()
    assert source.first.grad is not None and source.extra.grad is not None
    assert source.second.grad is None

    candidate = expected + 10.0
    rebuilt = flat_to_model(candidate, aux)
    assert tuple(dict(rebuilt.named_parameters())) == ("second", "first", "extra")
    assert tuple(parameter.shape for parameter in rebuilt.parameters()) == ((2,), (2,), (2,))
    assert [parameter.requires_grad for parameter in rebuilt.parameters()] == [False, True, True]
    assert rebuilt.label == "structure" and not rebuilt.training
    torch.testing.assert_close(rebuilt.offset, torch.tensor([5.0], dtype=torch.float64))
    torch.testing.assert_close(model_to_flat(rebuilt)[0], candidate)
    torch.testing.assert_close(model_to_flat(source)[0], expected)
    with torch.no_grad():
        rebuilt.first.add_(100.0)
        rebuilt.offset.add_(100.0)
    torch.testing.assert_close(source.first, torch.tensor([1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(source.offset, torch.tensor([5.0], dtype=torch.float64))
    second = flat_to_model(candidate, aux)
    torch.testing.assert_close(second.first, torch.tensor([11.0, 12.0], dtype=torch.float64))


@pytest.mark.parametrize(("model", "spec"), [(_gr6j(), GR6J_SPEC), (_hbv(), HBV_SPEC)])
def test_generic_and_canonical_orders_agree_for_models(model: nn.Module, spec: ParamSpec) -> None:
    torch.testing.assert_close(model_to_flat(model)[0], params_to_array(model, spec))


@pytest.mark.parametrize(
    "spec",
    [
        ParamSpec((), (), ()),
        ParamSpec(("x",), (), (1.0,)),
        ParamSpec(("x", "x"), (0.0, 0.0), (1.0, 1.0)),
        ParamSpec(("x",), (float("nan"),), (1.0,)),
        ParamSpec(("x",), (2.0,), (1.0,)),
    ],
)
def test_malformed_specs_are_rejected(spec: ParamSpec) -> None:
    with pytest.raises(ValueError):
        bounds_array(spec)


def test_named_view_and_reconstruction_errors() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        params_to_array(object())
    with pytest.raises(TypeError, match="torch.nn.Module"):
        array_to_model(object(), torch.zeros(6))
    with pytest.raises(TypeError, match="torch.Tensor"):
        array_to_parameters([1.0] * 6)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="at least one"):
        array_to_parameters(torch.tensor(1.0))
    with pytest.raises(ValueError, match="final dimension"):
        array_to_parameters(torch.zeros(5))

    plain = nn.Module()
    plain.x1 = torch.tensor(1.0)
    with pytest.raises(ValueError, match="missing registered"):
        params_to_array(plain, ParamSpec(("x1",), (0.0,), (2.0,)))
    with pytest.raises(ValueError, match="top-level"):
        array_to_model(_Tiny(), torch.zeros(1), ParamSpec(("child.x",), (0.0,), (1.0,)))
    with pytest.raises(ValueError, match="top-level"):
        params_to_array(_Tiny(), ParamSpec(("child.x",), (0.0,), (1.0,)))


def test_heterogeneous_selected_parameters_are_rejected() -> None:
    module = nn.Module()
    module.a = nn.Parameter(torch.ones(2, dtype=torch.float64))
    module.b = nn.Parameter(torch.ones(1, dtype=torch.float64))
    spec = ParamSpec(("a", "b"), (0.0, 0.0), (2.0, 2.0))
    with pytest.raises(ValueError, match="shapes"):
        params_to_array(module, spec)
    module.b = nn.Parameter(torch.ones(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="dtypes"):
        params_to_array(module, spec)


def test_generic_flatten_errors() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        model_to_flat(object())
    with pytest.raises(ValueError, match="at least one"):
        model_to_flat(nn.Module())
    mixed = nn.Module()
    mixed.a = nn.Parameter(torch.ones(1, dtype=torch.float64))
    mixed.b = nn.Parameter(torch.ones(1, dtype=torch.float32))
    with pytest.raises(ValueError, match="dtypes"):
        model_to_flat(mixed)

    if torch.cuda.is_available():
        mixed.b = nn.Parameter(torch.ones(1, dtype=torch.float64, device="cuda"))
        with pytest.raises(ValueError, match="same device"):
            model_to_flat(mixed)

    flat, aux = model_to_flat(_Tiny())
    with pytest.raises(TypeError, match="torch.Tensor"):
        flat_to_model([1.0], aux)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="metadata"):
        flat_to_model(flat, object())
    with pytest.raises(ValueError, match="one-dimensional"):
        flat_to_model(flat.reshape(2, 3), aux)
    with pytest.raises(ValueError, match="exactly"):
        flat_to_model(flat[:-1], aux)
    with pytest.raises(ValueError, match="dtype"):
        flat_to_model(flat.float(), aux)
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="device"):
            flat_to_model(flat.cuda(), aux)


def test_calibration_import_preserves_existing_adapter_exports() -> None:
    for name in (
        "ParamSpec",
        "PARAM_NAMES",
        "GR6J_SPEC",
        "HBV_SPEC",
        "bounds_array",
        "params_to_array",
        "array_to_model",
        "model_to_flat",
        "flat_to_model",
    ):
        assert hasattr(calibration, name)
