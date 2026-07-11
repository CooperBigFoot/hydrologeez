from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hydrologeez.models.gr6j import GR6J, GR6JForcing, State
from hydrologeez.models.hbv import HBVForcing, HBVModel, HBVState

GR6J_PRECIP = np.array(
    [
        [0.0, 0.0, 1.5, 12.0, 30.0, 4.0, 0.0, 0.0, 8.0, 2.0, 0.0, 18.0],
        [5.0, 2.0, 0.0, 0.0, 45.0, 60.0, 3.0, 1.0, 0.0, 25.0, 0.5, 0.0],
    ],
    dtype=np.float64,
)
GR6J_PET = np.array(
    [
        [2.0, 3.0, 2.5, 1.0, 0.5, 2.0, 4.0, 5.0, 1.5, 2.5, 3.5, 1.0],
        [1.0, 1.5, 3.0, 4.0, 0.0, 0.5, 2.5, 3.5, 4.5, 1.0, 2.0, 3.0],
    ],
    dtype=np.float64,
)
GR6J_PARAMETERS = np.array(
    [[350.0, -1.2, 90.0, 2.4, 0.15, 12.0], [700.0, 2.1, 180.0, 5.5, -0.35, 35.0]],
    dtype=np.float64,
)
GR6J_INITIAL_SCALARS = np.array([[105.0, 45.0, 0.0], [260.0, 120.0, -8.0]], dtype=np.float64)
GR6J_INITIAL_UH1 = np.stack(
    [
        np.zeros(20, dtype=np.float64),
        np.arange(1.0, 21.0, dtype=np.float64) / 100.0,
    ]
)
GR6J_INITIAL_UH2 = np.stack(
    [
        np.zeros(40, dtype=np.float64),
        np.arange(40.0, 0.0, -1.0, dtype=np.float64) / 200.0,
    ]
)

HBV_PRECIP = np.array(
    [
        [0.0, 6.0, 10.0, 0.0, 14.0, 3.0, 0.0, 20.0, 5.0, 0.0, 8.0, 1.0],
        [12.0, 4.0, 0.0, 18.0, 2.0, 25.0, 0.0, 0.0, 7.0, 16.0, 3.0, 0.0],
    ],
    dtype=np.float64,
)
HBV_PET = np.array(
    [
        [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 2.0, 1.0, 0.5, 1.5, 2.0, 1.0],
        [0.5, 1.0, 1.5, 2.5, 3.5, 2.0, 1.0, 0.5, 1.0, 2.0, 2.5, 1.5],
    ],
    dtype=np.float64,
)
HBV_TEMP = np.array(
    [
        [-4.0, -2.0, 0.0, 1.0, 3.0, 6.0, 2.0, -1.0, -3.0, 4.0, 7.0, 0.0],
        [2.5, 0.5, -2.5, -5.0, 1.5, 5.0, 8.0, 3.0, -1.0, 0.0, 4.5, 6.0],
    ],
    dtype=np.float64,
)
HBV_PARAMETERS = np.array(
    [
        [0.0, 3.5, 1.05, 0.10, 0.05, 220.0, 0.70, 2.0, 0.35, 0.08, 0.02, 1.5, 25.0, 3.0],
        [1.0, 6.0, 0.85, 0.05, 0.12, 450.0, 0.90, 4.5, 0.60, 0.18, 0.08, 4.0, 60.0, 6.5],
    ],
    dtype=np.float64,
)
HBV_INITIAL_SCALARS = np.array(
    [[0.0, 0.0, 110.0, 0.0, 0.0], [35.0, 2.5, 300.0, 80.0, 140.0]],
    dtype=np.float64,
)
HBV_INITIAL_ROUTING_BUFFER = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.75, 1.50, 1.25, 1.00, 0.75, 0.50, 0.25],
    ],
    dtype=np.float64,
)

GR6J_STATE_FIELDS = (
    "production_store",
    "routing_store",
    "exponential_store",
    "uh1",
    "uh2",
)
GR6J_FLUX_FIELDS = (
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
HBV_STATE_FIELDS = (
    "zone_sp",
    "zone_lw",
    "zone_sm",
    "upper_zone",
    "lower_zone",
    "routing_buffer",
)
HBV_FLUX_FIELDS = (
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


def _numpy(value: jax.Array | np.ndarray | float) -> np.ndarray:
    return np.array(value, dtype=np.float64, copy=True)


def _scan(model, initial_state, forcing):
    def step(state, one_forcing):
        new_state, fluxes = model.transition(state, one_forcing)
        return new_state, (new_state, fluxes)

    return jax.lax.scan(step, initial_state, forcing)


def build_gr6j_fixture() -> dict[str, np.ndarray]:
    scenarios = []
    for forcing_index in range(2):
        forcing_scenarios = []
        for parameter_index in range(2):
            parameters = GR6J_PARAMETERS[parameter_index]
            model = GR6J(
                x1=jnp.asarray(parameters[0]),
                x2=jnp.asarray(parameters[1]),
                x3=jnp.asarray(parameters[2]),
                x4=jnp.asarray(parameters[3]),
                x5=jnp.asarray(parameters[4]),
                x6=jnp.asarray(parameters[5]),
            )
            parameter_scenarios = []
            for initial_index in range(2):
                initial_scalars = GR6J_INITIAL_SCALARS[initial_index]
                initial_state = State(
                    production_store=jnp.asarray(initial_scalars[0]),
                    routing_store=jnp.asarray(initial_scalars[1]),
                    exponential_store=jnp.asarray(initial_scalars[2]),
                    uh1=jnp.asarray(GR6J_INITIAL_UH1[initial_index]),
                    uh2=jnp.asarray(GR6J_INITIAL_UH2[initial_index]),
                )
                forcing = GR6JForcing(
                    jnp.asarray(GR6J_PRECIP[forcing_index]),
                    jnp.asarray(GR6J_PET[forcing_index]),
                )
                final_state, (states, fluxes) = _scan(model, initial_state, forcing)
                parameter_scenarios.append((states, fluxes, final_state))
            forcing_scenarios.append(parameter_scenarios)
        scenarios.append(forcing_scenarios)

    result = {
        "schema_version": _numpy(1.0),
        "forcing.precip": _numpy(GR6J_PRECIP),
        "forcing.pet": _numpy(GR6J_PET),
        "parameters": _numpy(GR6J_PARAMETERS),
        "initial_state.production_store": _numpy(GR6J_INITIAL_SCALARS[:, 0]),
        "initial_state.routing_store": _numpy(GR6J_INITIAL_SCALARS[:, 1]),
        "initial_state.exponential_store": _numpy(GR6J_INITIAL_SCALARS[:, 2]),
        "initial_state.uh1": _numpy(GR6J_INITIAL_UH1),
        "initial_state.uh2": _numpy(GR6J_INITIAL_UH2),
    }
    for field in GR6J_STATE_FIELDS:
        result[f"state.{field}"] = _numpy(
            np.stack(
                [
                    np.stack([np.stack([getattr(scenarios[i][j][k][0], field) for k in range(2)]) for j in range(2)])
                    for i in range(2)
                ]
            )
        )
    for field in GR6J_FLUX_FIELDS:
        result[f"flux.{field}"] = _numpy(
            np.stack(
                [
                    np.stack([np.stack([getattr(scenarios[i][j][k][1], field) for k in range(2)]) for j in range(2)])
                    for i in range(2)
                ]
            )
        )
    result["streamflow"] = _numpy(result["flux.streamflow"])
    for field in GR6J_STATE_FIELDS:
        result[f"final_state.{field}"] = _numpy(
            np.stack(
                [
                    np.stack([np.stack([getattr(scenarios[i][j][k][2], field) for k in range(2)]) for j in range(2)])
                    for i in range(2)
                ]
            )
        )
    np.testing.assert_array_equal(result["streamflow"], result["flux.streamflow"])
    return result


def build_hbv_fixture() -> dict[str, np.ndarray]:
    scenarios = []
    for forcing_index in range(2):
        forcing_scenarios = []
        for parameter_index in range(2):
            parameters = HBV_PARAMETERS[parameter_index]
            model = HBVModel(
                tt=jnp.asarray(parameters[0]),
                cfmax=jnp.asarray(parameters[1]),
                sfcf=jnp.asarray(parameters[2]),
                cwh=jnp.asarray(parameters[3]),
                cfr=jnp.asarray(parameters[4]),
                fc=jnp.asarray(parameters[5]),
                lp=jnp.asarray(parameters[6]),
                beta=jnp.asarray(parameters[7]),
                k0=jnp.asarray(parameters[8]),
                k1=jnp.asarray(parameters[9]),
                k2=jnp.asarray(parameters[10]),
                perc=jnp.asarray(parameters[11]),
                uzl=jnp.asarray(parameters[12]),
                maxbas=jnp.asarray(parameters[13]),
            )
            parameter_scenarios = []
            for initial_index in range(2):
                initial_scalars = HBV_INITIAL_SCALARS[initial_index]
                initial_state = HBVState(
                    zone_sp=jnp.asarray(initial_scalars[0]),
                    zone_lw=jnp.asarray(initial_scalars[1]),
                    zone_sm=jnp.asarray(initial_scalars[2]),
                    upper_zone=jnp.asarray(initial_scalars[3]),
                    lower_zone=jnp.asarray(initial_scalars[4]),
                    routing_buffer=jnp.asarray(HBV_INITIAL_ROUTING_BUFFER[initial_index]),
                )
                forcing = HBVForcing(
                    jnp.asarray(HBV_PRECIP[forcing_index]),
                    jnp.asarray(HBV_PET[forcing_index]),
                    jnp.asarray(HBV_TEMP[forcing_index]),
                )
                final_state, (states, fluxes) = _scan(model, initial_state, forcing)
                parameter_scenarios.append((states, fluxes, final_state))
            forcing_scenarios.append(parameter_scenarios)
        scenarios.append(forcing_scenarios)

    result = {
        "schema_version": _numpy(1.0),
        "forcing.precip": _numpy(HBV_PRECIP),
        "forcing.pet": _numpy(HBV_PET),
        "forcing.temp": _numpy(HBV_TEMP),
        "parameters": _numpy(HBV_PARAMETERS),
        "initial_state.zone_sp": _numpy(HBV_INITIAL_SCALARS[:, 0]),
        "initial_state.zone_lw": _numpy(HBV_INITIAL_SCALARS[:, 1]),
        "initial_state.zone_sm": _numpy(HBV_INITIAL_SCALARS[:, 2]),
        "initial_state.upper_zone": _numpy(HBV_INITIAL_SCALARS[:, 3]),
        "initial_state.lower_zone": _numpy(HBV_INITIAL_SCALARS[:, 4]),
        "initial_state.routing_buffer": _numpy(HBV_INITIAL_ROUTING_BUFFER),
    }
    for field in HBV_STATE_FIELDS:
        result[f"state.{field}"] = _numpy(
            np.stack(
                [
                    np.stack([np.stack([getattr(scenarios[i][j][k][0], field) for k in range(2)]) for j in range(2)])
                    for i in range(2)
                ]
            )
        )
    for field in HBV_FLUX_FIELDS:
        result[f"flux.{field}"] = _numpy(
            np.stack(
                [
                    np.stack([np.stack([getattr(scenarios[i][j][k][1], field) for k in range(2)]) for j in range(2)])
                    for i in range(2)
                ]
            )
        )
    result["streamflow"] = _numpy(result["flux.streamflow"])
    for field in HBV_STATE_FIELDS:
        result[f"final_state.{field}"] = _numpy(
            np.stack(
                [
                    np.stack([np.stack([getattr(scenarios[i][j][k][2], field) for k in range(2)]) for j in range(2)])
                    for i in range(2)
                ]
            )
        )
    np.testing.assert_array_equal(result["streamflow"], result["flux.streamflow"])
    return result


def main() -> None:
    output_directory = Path(__file__).parent
    np.savez(output_directory / "gr6j.npz", **build_gr6j_fixture())  # ty: ignore[invalid-argument-type]
    np.savez(output_directory / "hbv.npz", **build_hbv_fixture())  # ty: ignore[invalid-argument-type]


if __name__ == "__main__":
    main()
