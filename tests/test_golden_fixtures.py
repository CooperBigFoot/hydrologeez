from pathlib import Path

import numpy as np
import pytest
from golden.generate import (
    GR6J_INITIAL_SCALARS,
    GR6J_INITIAL_UH1,
    GR6J_INITIAL_UH2,
    GR6J_PARAMETERS,
    GR6J_PET,
    GR6J_PRECIP,
    HBV_INITIAL_ROUTING_BUFFER,
    HBV_INITIAL_SCALARS,
    HBV_PARAMETERS,
    HBV_PET,
    HBV_PRECIP,
    HBV_TEMP,
    build_gr6j_fixture,
    build_hbv_fixture,
)

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"

GR6J_SHAPES = {
    "schema_version": (),
    "forcing.precip": (2, 12),
    "forcing.pet": (2, 12),
    "parameters": (2, 6),
    "initial_state.production_store": (2,),
    "initial_state.routing_store": (2,),
    "initial_state.exponential_store": (2,),
    "initial_state.uh1": (2, 20),
    "initial_state.uh2": (2, 40),
    **{f"state.{field}": (2, 2, 2, 12) for field in ("production_store", "routing_store", "exponential_store")},
    "state.uh1": (2, 2, 2, 12, 20),
    "state.uh2": (2, 2, 2, 12, 40),
    **{
        f"flux.{field}": (2, 2, 2, 12)
        for field in (
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
    },
    "streamflow": (2, 2, 2, 12),
    **{f"final_state.{field}": (2, 2, 2) for field in ("production_store", "routing_store", "exponential_store")},
    "final_state.uh1": (2, 2, 2, 20),
    "final_state.uh2": (2, 2, 2, 40),
}

HBV_SHAPES = {
    "schema_version": (),
    "forcing.precip": (2, 12),
    "forcing.pet": (2, 12),
    "forcing.temp": (2, 12),
    "parameters": (2, 14),
    **{f"initial_state.{field}": (2,) for field in ("zone_sp", "zone_lw", "zone_sm", "upper_zone", "lower_zone")},
    "initial_state.routing_buffer": (2, 7),
    **{f"state.{field}": (2, 2, 2, 12) for field in ("zone_sp", "zone_lw", "zone_sm", "upper_zone", "lower_zone")},
    "state.routing_buffer": (2, 2, 2, 12, 7),
    **{
        f"flux.{field}": (2, 2, 2, 12)
        for field in (
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
    },
    "streamflow": (2, 2, 2, 12),
    **{f"final_state.{field}": (2, 2, 2) for field in ("zone_sp", "zone_lw", "zone_sm", "upper_zone", "lower_zone")},
    "final_state.routing_buffer": (2, 2, 2, 7),
}

AUTHORED_INPUTS = {
    "gr6j": {
        "forcing.precip": GR6J_PRECIP,
        "forcing.pet": GR6J_PET,
        "parameters": GR6J_PARAMETERS,
        "initial_state.production_store": GR6J_INITIAL_SCALARS[:, 0],
        "initial_state.routing_store": GR6J_INITIAL_SCALARS[:, 1],
        "initial_state.exponential_store": GR6J_INITIAL_SCALARS[:, 2],
        "initial_state.uh1": GR6J_INITIAL_UH1,
        "initial_state.uh2": GR6J_INITIAL_UH2,
    },
    "hbv": {
        "forcing.precip": HBV_PRECIP,
        "forcing.pet": HBV_PET,
        "forcing.temp": HBV_TEMP,
        "parameters": HBV_PARAMETERS,
        "initial_state.zone_sp": HBV_INITIAL_SCALARS[:, 0],
        "initial_state.zone_lw": HBV_INITIAL_SCALARS[:, 1],
        "initial_state.zone_sm": HBV_INITIAL_SCALARS[:, 2],
        "initial_state.upper_zone": HBV_INITIAL_SCALARS[:, 3],
        "initial_state.lower_zone": HBV_INITIAL_SCALARS[:, 4],
        "initial_state.routing_buffer": HBV_INITIAL_ROUTING_BUFFER,
    },
}


@pytest.mark.parametrize(
    ("name", "shapes", "builder"),
    [("gr6j", GR6J_SHAPES, build_gr6j_fixture), ("hbv", HBV_SHAPES, build_hbv_fixture)],
)
def test_golden_fixture_schema_and_regeneration(name, shapes, builder):
    with np.load(GOLDEN_DIRECTORY / f"{name}.npz", allow_pickle=False) as committed:
        assert set(committed.files) == set(shapes)
        for key, shape in shapes.items():
            assert committed[key].shape == shape
            assert committed[key].dtype == np.dtype("float64")
            assert np.isfinite(committed[key]).all()
        assert committed["schema_version"] == np.float64(1.0)
        np.testing.assert_array_equal(committed["streamflow"], committed["flux.streamflow"])
        for key, expected in AUTHORED_INPUTS[name].items():
            np.testing.assert_array_equal(committed[key], expected)

        regenerated = builder()
        assert list(regenerated) == committed.files
        for key in committed.files:
            if key == "schema_version" or key in AUTHORED_INPUTS[name]:
                np.testing.assert_array_equal(regenerated[key], committed[key])
            else:
                np.testing.assert_allclose(regenerated[key], committed[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "state_fields"),
    [
        ("gr6j", ("production_store", "routing_store", "exponential_store", "uh1", "uh2")),
        ("hbv", ("zone_sp", "zone_lw", "zone_sm", "upper_zone", "lower_zone", "routing_buffer")),
    ],
)
def test_final_states_equal_last_transition(name, state_fields):
    with np.load(GOLDEN_DIRECTORY / f"{name}.npz", allow_pickle=False) as fixture:
        for field in state_fields:
            np.testing.assert_array_equal(
                fixture[f"final_state.{field}"],
                fixture[f"state.{field}"][..., -1, :]
                if field in {"uh1", "uh2", "routing_buffer"}
                else fixture[f"state.{field}"][..., -1],
            )


def test_populated_delay_lines_are_covered():
    assert np.count_nonzero(GR6J_INITIAL_UH1[1]) > 0
    assert np.count_nonzero(GR6J_INITIAL_UH2[1]) > 0
    assert np.count_nonzero(HBV_INITIAL_ROUTING_BUFFER[1]) > 0
    assert {"state.uh1", "state.uh2", "final_state.uh1", "final_state.uh2"} <= GR6J_SHAPES.keys()
    assert {"state.routing_buffer", "final_state.routing_buffer"} <= HBV_SHAPES.keys()
