"""End-to-end oracle parity for GR6J vs the committed Rust fixtures."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from hydrologeez.models.gr6j.model import GR6J, GR6JForcing
from hydrologeez.models.gr6j.state import State

FIXTURES = Path(__file__).parents[2] / "fixtures"

FLUX_KEYS = (
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

RTOL = 1e-4
ATOL = 1e-6


def _model_from_params(params):
    return GR6J(
        x1=jnp.asarray(params[0]),
        x2=jnp.asarray(params[1]),
        x3=jnp.asarray(params[2]),
        x4=jnp.asarray(params[3]),
        x5=jnp.asarray(params[4]),
        x6=jnp.asarray(params[5]),
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["gr6j_camels_06224000.npz", "gr6j_camels_06224000_exchange.npz"],
)
def test_full_series_parity(fixture_name):
    data = np.load(FIXTURES / fixture_name, allow_pickle=False)
    model = _model_from_params(data["params"])
    forcing = GR6JForcing(precip=jnp.asarray(data["precip"]), pet=jnp.asarray(data["pet"]))
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


def test_step_branch_parity_positive_softplus_and_ar_clamp():
    data = np.load(FIXTURES / "gr6j_step_branches.npz", allow_pickle=False)
    model = _model_from_params(data["params"])
    states = data["input_states"]
    qrexp_got = []
    expstore_got = []

    for i in range(states.shape[0]):
        row = states[i]
        state = State(
            production_store=jnp.asarray(row[0]),
            routing_store=jnp.asarray(row[1]),
            exponential_store=jnp.asarray(row[2]),
            uh1=jnp.asarray(row[3:23]),
            uh2=jnp.asarray(row[23:63]),
        )
        _new_state, fluxes = model.transition(
            state,
            GR6JForcing(precip=jnp.asarray(data["precip"][i]), pet=jnp.asarray(data["pet"][i])),
        )
        qrexp_got.append(np.asarray(fluxes.qrexp))
        expstore_got.append(np.asarray(fluxes.exponential_store))

    np.testing.assert_allclose(np.array(qrexp_got), data["qrexp"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.array(expstore_got), data["exponential_store"], rtol=RTOL, atol=ATOL)
    ar = data["input_states"][:, 2] / data["params"][5]
    assert np.any(ar > 7.0) and np.any(ar >= 33.0)
