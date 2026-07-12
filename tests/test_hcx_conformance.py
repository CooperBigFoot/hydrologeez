import subprocess
import sys

import torch
from hcx import assert_conforms, make_synthetic_batch

from hydrologeez.hcx import GR6JForecastModel, HBVForecastModel
from hydrologeez.models.gr6j import GR6J
from hydrologeez.models.hbv import HBVModel


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


def test_plain_hydrologeez_import_does_not_import_hcx() -> None:
    code = "import sys; import hydrologeez; assert 'hcx' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
