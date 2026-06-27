"""Generate golden GR6J oracle fixtures from the retired Rust pydrology core.

MUST run inside the pydrology repo environment (which ships the prebuilt `_core`
extension), NOT the hydrologeez venv, and NOT via maturin. Recommended invocation
from any cwd:

    uv run --project /Users/nicolaslazaro/Desktop/work/pydrology \
        python <hydrologeez-worktree>/scripts/generate_gr6j_fixtures.py

Writes four self-describing .npz artifacts into <repo>/tests/fixtures/.
Deterministic: no randomness, no auto-escalation. Re-running reproduces the
array contents exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd  # ty: ignore[unresolved-import]
from pydrology._core import gr6j as rust  # ty: ignore[unresolved-import]
from pydrology.models.gr6j import Parameters, run  # ty: ignore[unresolved-import]
from pydrology.types import ForcingData  # ty: ignore[unresolved-import]

PYDROLOGY_ROOT = Path("/Users/nicolaslazaro/Desktop/work/pydrology")
DATA_REL = "data/mountainous-us-basins/REGION_NAME=camels/data_type=timeseries/gauge_id=camels_06224000/data.parquet"
BASIN_ID = "camels_06224000"
WARMUP_LENGTH = 365
PARAM_NAMES = np.array(["x1", "x2", "x3", "x4", "x5", "x6"])

CANONICAL_PARAMS = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0)
EXCHANGE_PARAMS = Parameters(x1=350.0, x2=1.0, x3=90.0, x4=1.7, x5=0.5, x6=5.0)

# UH x4 grid (integer + non-integer).
UH_X4_GRID = np.array([0.5, 1.0, 1.7, 2.0, 3.5, 5.0, 7.0, 10.0])

# Crafted exponential-store inputs (x6=5 => AR = exp_store / 5).
# 40->AR8, 50->AR10 (positive softplus branch AR>7);
# 165->AR33, 200->AR40 (positive branch + the +33 AR clamp).
STEP_EXP_STORES = np.array([40.0, 50.0, 165.0, 200.0])
STEP_PRECIP = 5.0
STEP_PET = 2.0


def params_vec(p: Parameters) -> np.ndarray:
    return np.array([p.x1, p.x2, p.x3, p.x4, p.x5, p.x6], dtype=np.float64)


def load_forcing(data_path: Path) -> ForcingData:
    df = pd.read_parquet(data_path)
    return ForcingData(
        time=df["date"].to_numpy(),
        precip=df["mswep_precipitation"].to_numpy(),
        pet=df["potential_evaporation_sum_FAO_PENMAN_MONTEITH"].to_numpy(),
    )


def make_run_fixture(out: Path, name: str, params: Parameters, forcing: ForcingData) -> None:
    fluxes = run(params, forcing).fluxes.to_dict()
    np.savez(
        out / name,
        params=params_vec(params),
        param_names=PARAM_NAMES,
        warmup_length=np.array(WARMUP_LENGTH),
        basin_id=np.array(BASIN_ID),
        **fluxes,
    )


def make_uh_table(out: Path) -> None:
    uh1 = np.zeros((UH_X4_GRID.size, 20))
    uh2 = np.zeros((UH_X4_GRID.size, 40))
    for i, x4 in enumerate(UH_X4_GRID):
        o1, o2 = rust.gr6j_compute_uh_ordinates(float(x4))
        uh1[i] = np.asarray(o1)
        uh2[i] = np.asarray(o2)
    np.savez(out / "gr6j_uh_ordinates.npz", x4_grid=UH_X4_GRID, uh1_ord=uh1, uh2_ord=uh2)


def make_step_branches(out: Path) -> None:
    params = CANONICAL_PARAMS
    pvec = params_vec(params)
    x6 = params.x6
    o1, o2 = rust.gr6j_compute_uh_ordinates(params.x4)
    uh1 = np.asarray(o1, dtype=np.float64)
    uh2 = np.asarray(o2, dtype=np.float64)

    k = STEP_EXP_STORES.size
    input_states = np.zeros((k, 63), dtype=np.float64)
    input_states[:, 0] = 0.3 * params.x1  # production_store
    input_states[:, 1] = 0.5 * params.x3  # routing_store
    input_states[:, 2] = STEP_EXP_STORES  # exponential_store

    precip = np.full(k, STEP_PRECIP)
    pet = np.full(k, STEP_PET)
    qrexp = np.zeros(k)
    exp_store_out = np.zeros(k)
    for i in range(k):
        _new_state, fl = rust.gr6j_step(input_states[i], pvec, STEP_PRECIP, STEP_PET, uh1, uh2)
        qrexp[i] = float(fl["qrexp"])
        exp_store_out[i] = float(fl["exponential_store"])

    ar_target = STEP_EXP_STORES / x6
    np.savez(
        out / "gr6j_step_branches.npz",
        input_states=input_states,
        params=pvec,
        param_names=PARAM_NAMES,
        precip=precip,
        pet=pet,
        uh1_ord=uh1,
        uh2_ord=uh2,
        qrexp=qrexp,
        exponential_store=exp_store_out,
        ar_target=ar_target,
        is_positive_branch=ar_target > 7.0,
        is_ar_clamp=ar_target >= 33.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pydrology-root", type=Path, default=PYDROLOGY_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "fixtures",
    )
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    forcing = load_forcing(args.pydrology_root / DATA_REL)

    make_run_fixture(out, "gr6j_camels_06224000.npz", CANONICAL_PARAMS, forcing)
    make_run_fixture(out, "gr6j_camels_06224000_exchange.npz", EXCHANGE_PARAMS, forcing)
    make_uh_table(out)
    make_step_branches(out)
    print(f"Wrote 4 fixtures to {out}")


if __name__ == "__main__":
    main()
