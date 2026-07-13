import subprocess
import sys
from typing import cast

import pytest
import torch
from hcx import ModelFactory, OutputSpecification, Point, assert_conforms, make_synthetic_batch
from torch import nn

from hydrologeez.hcx import GR6JForecastModel, HBVForecastModel, create_model
from hydrologeez.hybrid import NeuralParameterForecastModel
from hydrologeez.models.gr6j import GR6J
from hydrologeez.models.hbv import HBVForcing, HBVModel


def _scalar(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def test_gr6j_hcx_conformance() -> None:
    dynamic_inputs = ["precip", "pet"]
    batch = make_synthetic_batch(
        scalar_dynamic_features=len(dynamic_inputs),
        include_scalar_static=False,
        include_gridded_dynamic=False,
        include_gridded_static=False,
    )
    model = GR6JForecastModel(
        GR6J(
            x1=_scalar(350.0),
            x2=_scalar(0.0),
            x3=_scalar(90.0),
            x4=_scalar(1.7),
            x5=_scalar(0.0),
            x6=_scalar(4.5),
        ),
        dynamic_inputs=dynamic_inputs,
    )

    assert_conforms(model, batch)


def test_hbv_hcx_conformance() -> None:
    dynamic_inputs = ["precip", "pet", "temp"]
    batch = make_synthetic_batch(
        scalar_dynamic_features=len(dynamic_inputs),
        include_scalar_static=False,
        include_gridded_dynamic=False,
        include_gridded_static=False,
    )
    model = HBVForecastModel(
        HBVModel(
            tt=_scalar(0.0),
            cfmax=_scalar(3.0),
            sfcf=_scalar(1.0),
            cwh=_scalar(0.1),
            cfr=_scalar(0.05),
            fc=_scalar(150.0),
            lp=_scalar(0.7),
            beta=_scalar(2.0),
            k0=_scalar(0.2),
            k1=_scalar(0.05),
            k2=_scalar(0.01),
            perc=_scalar(1.0),
            uzl=_scalar(10.0),
            maxbas=_scalar(3.0),
        ),
        dynamic_inputs=dynamic_inputs,
    )

    assert_conforms(model, batch)


def test_neural_parameter_hbv_hcx_conformance() -> None:
    torch.manual_seed(1729)
    dynamic_inputs = ("precip", "pet", "temp")
    static_inputs = ("area",)
    batch = make_synthetic_batch(
        input_length=8,
        output_length=3,
        scalar_dynamic_features=len(dynamic_inputs),
        scalar_static_features=len(static_inputs),
        include_gridded_dynamic=False,
        include_gridded_static=False,
        seed=1729,
    )
    hbv = HBVModel(
        tt=_scalar(0.0),
        cfmax=_scalar(3.0),
        sfcf=_scalar(1.0),
        cwh=_scalar(0.1),
        cfr=_scalar(0.05),
        fc=_scalar(150.0),
        lp=_scalar(0.7),
        beta=_scalar(2.0),
        k0=_scalar(0.2),
        k1=_scalar(0.05),
        k2=_scalar(0.01),
        perc=_scalar(1.0),
        uzl=_scalar(10.0),
        maxbas=_scalar(3.0),
    )
    network = nn.Sequential(
        nn.Linear(len(dynamic_inputs) + len(static_inputs), 8),
        nn.Tanh(),
        nn.Linear(8, len(hbv.parameter_bounds)),
    )
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

    forecast = assert_conforms(model, batch)

    assert forecast.prediction.shape == (batch.target.shape[0], batch.target.shape[-1])


def test_plain_hydrologeez_import_does_not_import_hcx() -> None:
    code = "import sys; import hydrologeez; assert 'hcx' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("n_components", [1, 3])
def test_neural_parameter_hbv_component_counts_conform(n_components: int) -> None:
    torch.manual_seed(1729)
    dynamic_inputs = ("precip", "pet", "temp")
    static_inputs = ("area",)
    batch = make_synthetic_batch(
        input_length=8,
        output_length=3,
        scalar_dynamic_features=len(dynamic_inputs),
        scalar_static_features=len(static_inputs),
        include_gridded_dynamic=False,
        include_gridded_static=False,
        seed=1729,
    )
    hbv = HBVModel(
        tt=_scalar(0.0),
        cfmax=_scalar(3.0),
        sfcf=_scalar(1.0),
        cwh=_scalar(0.1),
        cfr=_scalar(0.05),
        fc=_scalar(150.0),
        lp=_scalar(0.7),
        beta=_scalar(2.0),
        k0=_scalar(0.2),
        k1=_scalar(0.05),
        k2=_scalar(0.01),
        perc=_scalar(1.0),
        uzl=_scalar(10.0),
        maxbas=_scalar(3.0),
    )
    network = nn.Sequential(
        nn.Linear(len(dynamic_inputs) + len(static_inputs), 8),
        nn.Tanh(),
        nn.Linear(8, n_components * len(hbv.parameter_bounds)),
    )
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
        n_components=n_components,
    )

    forecast = assert_conforms(model, batch)

    assert forecast.prediction.shape == (batch.target.shape[0], batch.target.shape[-1])


def test_create_model_returns_conforming_dpl_hbv() -> None:
    torch.manual_seed(1729)
    dynamic_inputs = ["pet", "humidity", "temp", "precip"]
    static_inputs = ["area", "elevation"]
    batch = make_synthetic_batch(
        input_length=8,
        output_length=3,
        scalar_dynamic_features=len(dynamic_inputs),
        scalar_static_features=len(static_inputs),
        include_gridded_dynamic=False,
        include_gridded_static=False,
        seed=1729,
    )

    model = create_model(
        {"hidden_size": 8, "n_components": 2},
        dynamic_inputs=dynamic_inputs,
        static_inputs=static_inputs,
        input_size=len(dynamic_inputs),
        static_size=len(static_inputs),
        output_size=1,
        output_specification=cast(OutputSpecification[object], Point()),
    )

    assert isinstance(model, NeuralParameterForecastModel)
    assert model.dynamic_inputs == tuple(dynamic_inputs)
    assert model.static_inputs == tuple(static_inputs)
    assert model.n_components == 2
    network = model.network
    assert isinstance(network, nn.Sequential)
    output_layer = network[-1]
    assert isinstance(output_layer, nn.Linear)
    assert output_layer.out_features == 2 * len(model.parameter_bounds)
    forecast = assert_conforms(model, batch)
    assert forecast.prediction.shape == (batch.target.shape[0], batch.target.shape[-1])


def test_create_model_accepts_canonical_factory_keyword_call() -> None:
    factory: ModelFactory = create_model

    model = factory(
        {},
        dynamic_inputs=["precip", "pet", "temp"],
        static_inputs=[],
        input_size=3,
        static_size=0,
        output_size=1,
        output_specification=cast(OutputSpecification[object], Point()),
    )

    assert isinstance(model, NeuralParameterForecastModel)
    assert model.n_components == 1


@pytest.mark.parametrize("output_size", [0, 2])
def test_create_model_rejects_invalid_output_size(output_size: int) -> None:
    with pytest.raises(ValueError, match="output_size"):
        create_model(
            {},
            dynamic_inputs=["precip", "pet", "temp"],
            static_inputs=[],
            input_size=3,
            static_size=0,
            output_size=output_size,
            output_specification=cast(OutputSpecification[object], Point()),
        )
