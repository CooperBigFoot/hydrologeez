"""End-to-end oracle parity for single-zone HBV-Light vs the committed Rust fixtures."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from hydrologeez.models.hbv.model import HBVForcing, HBVModel

FIXTURES = Path(__file__).parents[2] / "fixtures"

PARAM_FIELDS = (
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

FLUX_KEYS = (
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

RTOL = 1e-4
ATOL = 1e-6


def _model_from_params(params):
    p = jnp.asarray(params)
    return HBVModel(
        tt=p[0],
        cfmax=p[1],
        sfcf=p[2],
        cwh=p[3],
        cfr=p[4],
        fc=p[5],
        lp=p[6],
        beta=p[7],
        k0=p[8],
        k1=p[9],
        k2=p[10],
        perc=p[11],
        uzl=p[12],
        maxbas=p[13],
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "hbv_camels_06224000.npz",
        "hbv_camels_06224000_maxbas25.npz",
        "hbv_camels_06224000_overflow.npz",
    ],
)
def test_full_series_parity(fixture_name):
    data = np.load(FIXTURES / fixture_name, allow_pickle=False)
    model = _model_from_params(data["params"])
    forcing = HBVForcing(
        precip=jnp.asarray(data["precip"]),
        pet=jnp.asarray(data["pet"]),
        temp=jnp.asarray(data["temp"]),
    )
    obs, fluxes, _final = model.run(forcing, return_fluxes=True)

    warmup = int(data["warmup_length"])
    np.testing.assert_allclose(np.asarray(obs)[warmup:], data["streamflow"][warmup:], rtol=RTOL, atol=ATOL)
    for key in FLUX_KEYS:
        got = np.asarray(getattr(fluxes, key))[warmup:]
        np.testing.assert_allclose(
            got,
            data[key][warmup:],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"flux {key} parity ({fixture_name})",
        )
