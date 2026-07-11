from pathlib import Path

import numpy as np

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"

GR6J_KEYS = {
    "schema_version",
    "forcing.precip",
    "forcing.pet",
    "parameters",
    "initial_state.production_store",
    "initial_state.routing_store",
    "initial_state.exponential_store",
    "initial_state.uh1",
    "initial_state.uh2",
    "state.production_store",
    "state.routing_store",
    "state.exponential_store",
    "state.uh1",
    "state.uh2",
    "flux.pet",
    "flux.precip",
    "flux.production_store",
    "flux.net_rainfall",
    "flux.storage_infiltration",
    "flux.actual_et",
    "flux.percolation",
    "flux.effective_rainfall",
    "flux.q9",
    "flux.q1",
    "flux.routing_store",
    "flux.exchange",
    "flux.actual_exchange_routing",
    "flux.actual_exchange_direct",
    "flux.actual_exchange_total",
    "flux.qr",
    "flux.qrexp",
    "flux.exponential_store",
    "flux.qd",
    "flux.streamflow",
    "streamflow",
    "final_state.production_store",
    "final_state.routing_store",
    "final_state.exponential_store",
    "final_state.uh1",
    "final_state.uh2",
}

HBV_KEYS = {
    "schema_version",
    "forcing.precip",
    "forcing.pet",
    "forcing.temp",
    "parameters",
    "initial_state.zone_sp",
    "initial_state.zone_lw",
    "initial_state.zone_sm",
    "initial_state.upper_zone",
    "initial_state.lower_zone",
    "initial_state.routing_buffer",
    "state.zone_sp",
    "state.zone_lw",
    "state.zone_sm",
    "state.upper_zone",
    "state.lower_zone",
    "state.routing_buffer",
    "flux.precip",
    "flux.temp",
    "flux.pet",
    "flux.precip_rain",
    "flux.precip_snow",
    "flux.snow_pack",
    "flux.snow_melt",
    "flux.liquid_water_in_snow",
    "flux.snow_input",
    "flux.soil_moisture",
    "flux.recharge",
    "flux.actual_et",
    "flux.upper_zone",
    "flux.lower_zone",
    "flux.q0",
    "flux.q1",
    "flux.q2",
    "flux.percolation",
    "flux.qgw",
    "flux.streamflow",
    "streamflow",
    "final_state.zone_sp",
    "final_state.zone_lw",
    "final_state.zone_sm",
    "final_state.upper_zone",
    "final_state.lower_zone",
    "final_state.routing_buffer",
}


def _assert_common_integrity(fixture: np.lib.npyio.NpzFile, expected_keys: set[str]) -> None:
    assert set(fixture.files) == expected_keys
    for key in fixture.files:
        assert fixture[key].dtype == np.dtype("float64")
        assert np.isfinite(fixture[key]).all()
    np.testing.assert_array_equal(fixture["schema_version"], np.array(1.0, dtype=np.float64))
    np.testing.assert_array_equal(fixture["streamflow"], fixture["flux.streamflow"])


def _assert_time_series_shapes(
    fixture: np.lib.npyio.NpzFile,
    forcing_fields: tuple[str, ...],
    state_fields: tuple[str, ...],
    flux_fields: tuple[str, ...],
    buffer_widths: dict[str, int],
) -> None:
    forcing_shape = fixture[f"forcing.{forcing_fields[0]}"].shape
    assert forcing_shape[0] == 2
    time_steps = forcing_shape[1]
    for field in forcing_fields:
        assert fixture[f"forcing.{field}"].shape == forcing_shape

    scenario_shape = (2, 2, 2)
    for field in state_fields:
        tail = (buffer_widths[field],) if field in buffer_widths else ()
        assert fixture[f"state.{field}"].shape == scenario_shape + (time_steps,) + tail
        assert fixture[f"final_state.{field}"].shape == scenario_shape + tail
        expected_final = (
            fixture[f"state.{field}"][..., -1, :] if field in buffer_widths else fixture[f"state.{field}"][..., -1]
        )
        np.testing.assert_array_equal(fixture[f"final_state.{field}"], expected_final)
    for field in flux_fields:
        assert fixture[f"flux.{field}"].shape == scenario_shape + (time_steps,)
    assert fixture["streamflow"].shape == scenario_shape + (time_steps,)


def test_gr6j_golden_fixture_integrity() -> None:
    with np.load(GOLDEN_DIRECTORY / "gr6j.npz", allow_pickle=False) as fixture:
        _assert_common_integrity(fixture, GR6J_KEYS)
        assert fixture["parameters"].shape == (2, 6)
        for field in ("production_store", "routing_store", "exponential_store"):
            assert fixture[f"initial_state.{field}"].shape == (2,)
        assert fixture["initial_state.uh1"].shape == (2, 20)
        assert fixture["initial_state.uh2"].shape == (2, 40)
        assert np.count_nonzero(fixture["initial_state.uh1"][1]) > 0
        assert np.count_nonzero(fixture["initial_state.uh2"][1]) > 0
        _assert_time_series_shapes(
            fixture,
            forcing_fields=("precip", "pet"),
            state_fields=("production_store", "routing_store", "exponential_store", "uh1", "uh2"),
            flux_fields=(
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
            ),
            buffer_widths={"uh1": 20, "uh2": 40},
        )


def test_hbv_golden_fixture_integrity() -> None:
    with np.load(GOLDEN_DIRECTORY / "hbv.npz", allow_pickle=False) as fixture:
        _assert_common_integrity(fixture, HBV_KEYS)
        assert fixture["parameters"].shape == (2, 14)
        for field in ("zone_sp", "zone_lw", "zone_sm", "upper_zone", "lower_zone"):
            assert fixture[f"initial_state.{field}"].shape == (2,)
        assert fixture["initial_state.routing_buffer"].shape == (2, 7)
        assert np.count_nonzero(fixture["initial_state.routing_buffer"][1]) > 0
        _assert_time_series_shapes(
            fixture,
            forcing_fields=("precip", "pet", "temp"),
            state_fields=(
                "zone_sp",
                "zone_lw",
                "zone_sm",
                "upper_zone",
                "lower_zone",
                "routing_buffer",
            ),
            flux_fields=(
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
            ),
            buffer_widths={"routing_buffer": 7},
        )
