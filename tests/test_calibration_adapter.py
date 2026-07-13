from __future__ import annotations

import pytest
import torch
from torch import nn

import hydrologeez.calibration as calibration
from hydrologeez.calibration.adapter import (
    ParamSpec,
    array_to_model,
    array_to_parameters,
    bounds_array,
    flat_to_model,
    model_to_flat,
    params_to_array,
)
from hydrologeez.models.gr6j.model import GR6J
from hydrologeez.models.hbv.model import HBVModel

HBV_PARAMETER_NAMES = (
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


def _gr6j(*, nh: int = 17) -> GR6J:
    values = (11.0, -2.0, 7.0, 3.0, -1.0, 5.0)
    return GR6J(*(torch.tensor(value, dtype=torch.float64) for value in values), nh=nh)


def _hbv() -> HBVModel:
    values = (2.0, 9.0, 0.7, 0.15, 0.03, 600.0, 0.4, 5.0, 0.8, 0.2, 0.1, 4.0, 80.0, 3.0)
    return HBVModel(
        **{
            name: torch.tensor(value, dtype=torch.float64)
            for name, value in zip(HBV_PARAMETER_NAMES, values, strict=True)
        }  # ty: ignore[invalid-argument-type]
    )


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.second = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64), requires_grad=False)
        self.first = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        self.extra = nn.Parameter(torch.tensor([8.0, 9.0], dtype=torch.float64))
        self.register_buffer("offset", torch.tensor([5.0], dtype=torch.float64))
        self.label = "structure"


class _Bounded(nn.Module):
    parameter_bounds = {"alpha": (-2.0, 3.0), "replacement": (0.25, 8.0)}

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.replacement = nn.Parameter(torch.tensor(1.0))


TINY_SPEC = ParamSpec(("first", "second"), (-10.0, -20.0), (10.0, 20.0))


def test_specs_have_exact_model_derived_values() -> None:
    gr6j = _gr6j()
    hbv = _hbv()
    gr6j_spec = ParamSpec.from_model(gr6j)
    hbv_spec = ParamSpec.from_model(hbv)

    assert gr6j_spec.names == tuple(gr6j.parameter_bounds) == ("x1", "x4", "x2", "x3", "x5", "x6")
    assert gr6j_spec.lower == (1.0, 0.5, -5.0, 1.0, -4.0, 1.0)
    assert gr6j_spec.upper == (2500.0, 10.0, 5.0, 1000.0, 4.0, 50.0)
    assert tuple(zip(gr6j_spec.lower, gr6j_spec.upper, strict=True)) == tuple(gr6j.parameter_bounds.values())
    assert hbv_spec.names == tuple(hbv.parameter_bounds) == HBV_PARAMETER_NAMES
    assert len(hbv_spec.names) == 14
    assert tuple(zip(hbv_spec.lower, hbv_spec.upper, strict=True)) == tuple(hbv.parameter_bounds.values())
    assert (hbv_spec.lower[-1], hbv_spec.upper[-1]) == (1.0, 7.0)


def test_param_spec_from_model_preserves_order_and_bounds() -> None:
    model = _Bounded()
    assert ParamSpec.from_model(model) == ParamSpec(
        names=("alpha", "replacement"),
        lower=(-2.0, 0.25),
        upper=(3.0, 8.0),
    )


def test_param_spec_from_model_rejects_invalid_sources() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        ParamSpec.from_model(object())
    with pytest.raises(TypeError, match="must be a mapping"):
        ParamSpec.from_model(nn.Module())

    non_string = nn.Module()
    non_string.parameter_bounds = {1: (0.0, 1.0)}  # ty: ignore[unresolved-attribute]
    with pytest.raises(TypeError, match="keys must be strings"):
        ParamSpec.from_model(non_string)

    malformed = nn.Module()
    malformed.parameter_bounds = {"x": (1.0, 1.0)}  # ty: ignore[unresolved-attribute]
    with pytest.raises(ValueError, match="lower strictly below upper"):
        ParamSpec.from_model(malformed)


def test_bounds_array_dtype_and_device() -> None:
    gr6j_spec = ParamSpec.from_model(_gr6j())
    hbv_spec = ParamSpec.from_model(_hbv())
    lower, upper = bounds_array(gr6j_spec)
    assert lower.dtype == upper.dtype == torch.float64
    assert lower.device.type == upper.device.type == "cpu"
    torch.testing.assert_close(lower, torch.tensor(gr6j_spec.lower, dtype=torch.float64))
    torch.testing.assert_close(upper, torch.tensor(gr6j_spec.upper, dtype=torch.float64))

    requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    explicit_lower, explicit_upper = bounds_array(hbv_spec, dtype=torch.float32, device=requested)
    assert explicit_lower.dtype == explicit_upper.dtype == torch.float32
    assert explicit_lower.device.type == explicit_upper.device.type == torch.device(requested).type
    torch.testing.assert_close(explicit_lower.cpu(), torch.tensor(hbv_spec.lower, dtype=torch.float32))
    torch.testing.assert_close(explicit_upper.cpu(), torch.tensor(hbv_spec.upper, dtype=torch.float32))
    assert lower.data_ptr() != bounds_array(gr6j_spec)[0].data_ptr()


GR6J_ROW_MODEL = _gr6j()
HBV_ROW_MODEL = _hbv()


@pytest.mark.parametrize(
    ("model", "spec", "expected"),
    [
        (GR6J_ROW_MODEL, ParamSpec.from_model(GR6J_ROW_MODEL), (11.0, 3.0, -2.0, 7.0, -1.0, 5.0)),
        (
            HBV_ROW_MODEL,
            ParamSpec.from_model(HBV_ROW_MODEL),
            (2.0, 9.0, 0.7, 0.15, 0.03, 600.0, 0.4, 5.0, 0.8, 0.2, 0.1, 4.0, 80.0, 3.0),
        ),
    ],
)
def test_params_to_array_uses_spec_order(model: nn.Module, spec: ParamSpec, expected: tuple[float, ...]) -> None:
    result = params_to_array(model, spec)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    result.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_array_to_parameters_is_differentiable_view_for_scalar_and_batch() -> None:
    spec = ParamSpec.from_model(_gr6j())
    theta = torch.arange(1, 7, dtype=torch.float64, requires_grad=True)
    parameters = array_to_parameters(theta, spec)
    assert tuple(parameters) == spec.names == ("x1", "x4", "x2", "x3", "x5", "x6")
    assert all(value.ndim == 0 for value in parameters.values())
    loss = torch.stack([(index + 1) * value.square() for index, value in enumerate(parameters.values())]).sum()
    loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all() and torch.count_nonzero(theta.grad) == theta.numel()

    batched = torch.arange(18, dtype=torch.float64).reshape(3, 6).requires_grad_()
    mapped = array_to_parameters(batched, spec)
    assert all(value.shape == (3,) for value in mapped.values())
    torch.testing.assert_close(mapped["x4"], batched[:, 1])
    assert mapped["x4"].untyped_storage().data_ptr() == batched.untyped_storage().data_ptr()
    mapped["x4"].sum().backward()
    assert batched.grad is not None
    torch.testing.assert_close(batched.grad[:, 1], torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize("model", [_gr6j(), _hbv()])
def test_named_model_round_trip(model: nn.Module) -> None:
    spec = ParamSpec.from_model(model)
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
        dict(rebuilt.named_parameters())["x1"].add_(100.0)
        dict(template.named_parameters())["x2"].add_(50.0)
    torch.testing.assert_close(template.x1, torch.tensor(11.0, dtype=torch.float64))
    torch.testing.assert_close(rebuilt.x2, theta[:, 2])


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
    spec = ParamSpec.from_model(template)
    rebuilt = array_to_model(template, params_to_array(template, spec), spec)
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


@pytest.mark.parametrize("model", [_gr6j(), _hbv()])
def test_generic_and_model_derived_orders_agree_for_models(model: nn.Module) -> None:
    spec = ParamSpec.from_model(model)
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
    spec = ParamSpec.from_model(_gr6j())
    with pytest.raises(TypeError, match="torch.nn.Module"):
        params_to_array(object())
    with pytest.raises(TypeError, match="torch.nn.Module"):
        array_to_model(object(), torch.zeros(6))
    with pytest.raises(TypeError, match="torch.Tensor"):
        array_to_parameters([1.0] * 6, spec)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="at least one"):
        array_to_parameters(torch.tensor(1.0), spec)
    with pytest.raises(ValueError, match="final dimension"):
        array_to_parameters(torch.zeros(5), spec)

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
        "bounds_array",
        "params_to_array",
        "array_to_model",
        "model_to_flat",
        "flat_to_model",
    ):
        assert hasattr(calibration, name)
    assert hasattr(calibration.ParamSpec, "from_model")
