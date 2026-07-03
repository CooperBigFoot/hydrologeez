"""Generate and verify GR6J oracle fixtures from the hydrologeez model.

Writes four self-describing .npz artifacts into <repo>/tests/fixtures/.
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

from hydrologeez.models.gr6j import processes
from hydrologeez.models.gr6j.model import GR6J, GR6JForcing
from hydrologeez.models.gr6j.state import State

BASIN_ID = "camels_06224000"
WARMUP_LENGTH = 365
PARAM_NAMES = np.array(["x1", "x2", "x3", "x4", "x5", "x6"])

CANONICAL_PARAMS = np.array([350.0, 0.0, 90.0, 1.7, 0.0, 5.0], dtype=np.float64)
EXCHANGE_PARAMS = np.array([350.0, 1.0, 90.0, 1.7, 0.5, 5.0], dtype=np.float64)

UH_X4_GRID = np.array([0.5, 1.0, 1.7, 2.0, 3.5, 5.0, 7.0, 10.0])

STEP_EXP_STORES = np.array([40.0, 50.0, 165.0, 200.0])
STEP_PRECIP = 5.0
STEP_PET = 2.0

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

FixtureData = Mapping[str, np.ndarray]
FixtureBuilder = Callable[[FixtureData], dict[str, np.ndarray]]


def _model_from_params(params: np.ndarray) -> GR6J:
    p = jnp.asarray(params)
    return GR6J(
        x1=p[0],
        x2=p[1],
        x3=p[2],
        x4=p[3],
        x5=p[4],
        x6=p[5],
    )


def forcing_from_fixture(npz: FixtureData) -> dict[str, np.ndarray]:
    return {"precip": npz["precip"], "pet": npz["pet"]}


def build_gr6j_run(
    params: np.ndarray,
    precip: np.ndarray,
    pet: np.ndarray,
    param_names: np.ndarray = PARAM_NAMES,
    warmup_length: np.ndarray | int = WARMUP_LENGTH,
    basin_id: np.ndarray | str = BASIN_ID,
) -> dict[str, np.ndarray]:
    model = _model_from_params(params)
    _obs, fluxes, _final = model.run(
        GR6JForcing(precip=jnp.asarray(precip), pet=jnp.asarray(pet)),
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


def build_gr6j_uh_table() -> dict[str, np.ndarray]:
    uh1 = np.zeros((UH_X4_GRID.size, 20), dtype=np.float64)
    uh2 = np.zeros((UH_X4_GRID.size, 40), dtype=np.float64)
    for i, x4 in enumerate(UH_X4_GRID):
        o1, o2 = processes.compute_uh_ordinates(jnp.asarray(x4))
        uh1[i] = np.asarray(o1)
        uh2[i] = np.asarray(o2)
    return {"x4_grid": UH_X4_GRID, "uh1_ord": uh1, "uh2_ord": uh2}


def build_gr6j_step_branches() -> dict[str, np.ndarray]:
    params = CANONICAL_PARAMS
    model = _model_from_params(params)
    uh1, uh2 = processes.compute_uh_ordinates(jnp.asarray(params[3]))
    uh1_ord = np.asarray(uh1, dtype=np.float64)
    uh2_ord = np.asarray(uh2, dtype=np.float64)

    k = STEP_EXP_STORES.size
    input_states = np.zeros((k, 63), dtype=np.float64)
    input_states[:, 0] = 0.3 * params[0]
    input_states[:, 1] = 0.5 * params[2]
    input_states[:, 2] = STEP_EXP_STORES

    precip = np.full(k, STEP_PRECIP)
    pet = np.full(k, STEP_PET)
    qrexp = np.zeros(k)
    exp_store_out = np.zeros(k)
    for i, row in enumerate(input_states):
        state = State(
            production_store=jnp.asarray(row[0]),
            routing_store=jnp.asarray(row[1]),
            exponential_store=jnp.asarray(row[2]),
            uh1=jnp.asarray(row[3:23]),
            uh2=jnp.asarray(row[23:63]),
        )
        _new_state, fluxes = model.transition(
            state,
            GR6JForcing(precip=jnp.asarray(precip[i]), pet=jnp.asarray(pet[i])),
        )
        qrexp[i] = np.asarray(fluxes.qrexp)
        exp_store_out[i] = np.asarray(fluxes.exponential_store)

    ar_target = STEP_EXP_STORES / params[5]
    return {
        "input_states": input_states,
        "params": params,
        "param_names": PARAM_NAMES,
        "precip": precip,
        "pet": pet,
        "uh1_ord": uh1_ord,
        "uh2_ord": uh2_ord,
        "qrexp": qrexp,
        "exponential_store": exp_store_out,
        "ar_target": ar_target,
        "is_positive_branch": ar_target > 7.0,
        "is_ar_clamp": ar_target >= 33.0,
    }


def _build_run_from_fixture(npz: FixtureData) -> dict[str, np.ndarray]:
    forcing = forcing_from_fixture(npz)
    return build_gr6j_run(
        params=npz["params"],
        precip=forcing["precip"],
        pet=forcing["pet"],
        param_names=npz["param_names"],
        warmup_length=npz["warmup_length"],
        basin_id=npz["basin_id"],
    )


FIXTURE_BUILDERS: dict[str, FixtureBuilder] = {
    "gr6j_camels_06224000.npz": _build_run_from_fixture,
    "gr6j_camels_06224000_exchange.npz": _build_run_from_fixture,
    "gr6j_uh_ordinates.npz": lambda _npz: build_gr6j_uh_table(),
    "gr6j_step_branches.npz": lambda _npz: build_gr6j_step_branches(),
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
        if name in {"gr6j_camels_06224000.npz", "gr6j_camels_06224000_exchange.npz"}:
            np.savez(
                out / name,
                params=rebuilt["params"],
                param_names=rebuilt["param_names"],
                warmup_length=rebuilt["warmup_length"],
                basin_id=rebuilt["basin_id"],
                pet=rebuilt["pet"],
                precip=rebuilt["precip"],
                production_store=rebuilt["production_store"],
                net_rainfall=rebuilt["net_rainfall"],
                storage_infiltration=rebuilt["storage_infiltration"],
                actual_et=rebuilt["actual_et"],
                percolation=rebuilt["percolation"],
                effective_rainfall=rebuilt["effective_rainfall"],
                q9=rebuilt["q9"],
                q1=rebuilt["q1"],
                routing_store=rebuilt["routing_store"],
                exchange=rebuilt["exchange"],
                actual_exchange_routing=rebuilt["actual_exchange_routing"],
                actual_exchange_direct=rebuilt["actual_exchange_direct"],
                actual_exchange_total=rebuilt["actual_exchange_total"],
                qr=rebuilt["qr"],
                qrexp=rebuilt["qrexp"],
                exponential_store=rebuilt["exponential_store"],
                qd=rebuilt["qd"],
                streamflow=rebuilt["streamflow"],
            )
        elif name == "gr6j_uh_ordinates.npz":
            np.savez(
                out / name,
                x4_grid=rebuilt["x4_grid"],
                uh1_ord=rebuilt["uh1_ord"],
                uh2_ord=rebuilt["uh2_ord"],
            )
        else:
            np.savez(
                out / name,
                input_states=rebuilt["input_states"],
                params=rebuilt["params"],
                param_names=rebuilt["param_names"],
                precip=rebuilt["precip"],
                pet=rebuilt["pet"],
                uh1_ord=rebuilt["uh1_ord"],
                uh2_ord=rebuilt["uh2_ord"],
                qrexp=rebuilt["qrexp"],
                exponential_store=rebuilt["exponential_store"],
                ar_target=rebuilt["ar_target"],
                is_positive_branch=rebuilt["is_positive_branch"],
                is_ar_clamp=rebuilt["is_ar_clamp"],
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
    print(f"Wrote 4 GR6J fixtures to {out}")


if __name__ == "__main__":
    main()
