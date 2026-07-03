"""Generate and verify HBV-Light oracle fixtures from the hydrologeez model.

Writes three self-describing .npz artifacts into <repo>/tests/fixtures/.
Use --verify to rebuild the fixtures in memory and compare them to the
committed artifacts without writing any .npz files.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np

from hydrologeez.models.hbv import processes
from hydrologeez.models.hbv.model import HBVForcing, HBVModel

BASIN_ID = "camels_06224000"
WARMUP_LENGTH = 365
ROUTING_BUFFER_SIZE = 7
PARAM_NAMES = np.array(
    [
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
    ]
)

CANONICAL_PARAMS = np.array(
    [
        0.0,
        3.5,
        1.0,
        0.1,
        0.05,
        250.0,
        0.7,
        2.0,
        0.3,
        0.1,
        0.05,
        2.0,
        20.0,
        3.0,
    ],
    dtype=np.float64,
)
MAXBAS25_PARAMS = np.array(
    [
        0.5,
        5.0,
        1.1,
        0.1,
        0.05,
        300.0,
        0.7,
        2.5,
        0.4,
        0.15,
        0.04,
        2.5,
        25.0,
        2.5,
    ],
    dtype=np.float64,
)

MAXBAS_GRID = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 5.0, 7.0])

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

FixtureData = Mapping[str, np.ndarray]
FixtureBuilder = Callable[[FixtureData], dict[str, np.ndarray]]


def _model_from_params(params: np.ndarray) -> HBVModel:
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


def forcing_from_fixture(npz: FixtureData) -> dict[str, np.ndarray]:
    return {"precip": npz["precip"], "pet": npz["pet"], "temp": npz["temp"]}


def build_hbv_run(
    params: np.ndarray,
    precip: np.ndarray,
    pet: np.ndarray,
    temp: np.ndarray,
    param_names: np.ndarray = PARAM_NAMES,
    warmup_length: np.ndarray | int = WARMUP_LENGTH,
    basin_id: np.ndarray | str = BASIN_ID,
) -> dict[str, np.ndarray]:
    model = _model_from_params(params)
    _obs, fluxes, _final = model.run(
        HBVForcing(
            precip=jnp.asarray(precip),
            pet=jnp.asarray(pet),
            temp=jnp.asarray(temp),
        ),
        return_fluxes=True,
    )

    fixture = {
        "params": np.asarray(params, dtype=np.float64),
        "param_names": np.asarray(param_names),
        "warmup_length": np.asarray(warmup_length),
        "basin_id": np.asarray(basin_id),
    }
    fixture.update({key: np.asarray(getattr(fluxes, key)) for key in FLUX_KEYS})
    return fixture


def build_hbv_weights_table() -> dict[str, np.ndarray]:
    weights = np.zeros((MAXBAS_GRID.size, ROUTING_BUFFER_SIZE), dtype=np.float64)
    for i, maxbas in enumerate(MAXBAS_GRID):
        weights[i] = np.asarray(processes.compute_triangular_weights(jnp.asarray(maxbas)))
    return {"maxbas_grid": MAXBAS_GRID, "weights": weights}


def _build_run_from_fixture(npz: FixtureData) -> dict[str, np.ndarray]:
    forcing = forcing_from_fixture(npz)
    return build_hbv_run(
        params=npz["params"],
        precip=forcing["precip"],
        pet=forcing["pet"],
        temp=forcing["temp"],
        param_names=npz["param_names"],
        warmup_length=npz["warmup_length"],
        basin_id=npz["basin_id"],
    )


FIXTURE_BUILDERS: dict[str, FixtureBuilder] = {
    "hbv_camels_06224000.npz": _build_run_from_fixture,
    "hbv_camels_06224000_maxbas25.npz": _build_run_from_fixture,
    "hbv_triangular_weights.npz": lambda _npz: build_hbv_weights_table(),
}


def _assert_arrays_match(name: str, key: str, got: np.ndarray, ref: np.ndarray) -> None:
    err_msg = f"{name}:{key}"
    if np.issubdtype(ref.dtype, np.floating):
        np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL, err_msg=err_msg)
    else:
        np.testing.assert_array_equal(got, ref, err_msg=err_msg)


def verify_fixtures(out: Path) -> bool:
    ok = True
    for name, builder in FIXTURE_BUILDERS.items():
        with np.load(out / name, allow_pickle=False) as data:
            rebuilt = builder(data)
            try:
                if set(rebuilt) != set(data.files):
                    missing = sorted(set(data.files) - set(rebuilt))
                    extra = sorted(set(rebuilt) - set(data.files))
                    raise AssertionError(f"{name}: keys differ missing={missing} extra={extra}")
                for key, got in rebuilt.items():
                    _assert_arrays_match(name, key, got, data[key])
            except AssertionError as exc:
                ok = False
                print(f"FAIL {name}: {exc}")
            else:
                print(f"PASS {name}")
    return ok


def write_fixtures(out: Path) -> None:
    for name, builder in FIXTURE_BUILDERS.items():
        with np.load(out / name, allow_pickle=False) as data:
            rebuilt = builder(data)
        if name in {"hbv_camels_06224000.npz", "hbv_camels_06224000_maxbas25.npz"}:
            np.savez(
                out / name,
                params=rebuilt["params"],
                param_names=rebuilt["param_names"],
                warmup_length=rebuilt["warmup_length"],
                basin_id=rebuilt["basin_id"],
                precip=rebuilt["precip"],
                temp=rebuilt["temp"],
                pet=rebuilt["pet"],
                precip_rain=rebuilt["precip_rain"],
                precip_snow=rebuilt["precip_snow"],
                snow_pack=rebuilt["snow_pack"],
                snow_melt=rebuilt["snow_melt"],
                liquid_water_in_snow=rebuilt["liquid_water_in_snow"],
                snow_input=rebuilt["snow_input"],
                soil_moisture=rebuilt["soil_moisture"],
                recharge=rebuilt["recharge"],
                actual_et=rebuilt["actual_et"],
                upper_zone=rebuilt["upper_zone"],
                lower_zone=rebuilt["lower_zone"],
                q0=rebuilt["q0"],
                q1=rebuilt["q1"],
                q2=rebuilt["q2"],
                percolation=rebuilt["percolation"],
                qgw=rebuilt["qgw"],
                streamflow=rebuilt["streamflow"],
            )
        else:
            np.savez(
                out / name,
                maxbas_grid=rebuilt["maxbas_grid"],
                weights=rebuilt["weights"],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "fixtures",
    )
    args = parser.parse_args()

    out: Path = args.out
    if args.verify:
        if not verify_fixtures(out):
            sys.exit(1)
        return

    out.mkdir(parents=True, exist_ok=True)
    write_fixtures(out)
    print(f"Wrote 3 HBV fixtures to {out}")


if __name__ == "__main__":
    main()
