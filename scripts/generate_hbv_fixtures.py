"""Generate golden HBV-Light oracle fixtures from the retired Rust pydrology core.

MUST run inside the pydrology repo environment (which ships the prebuilt `_core`
extension), NOT the hydrologeez venv, and NOT via maturin. Recommended invocation
from any cwd:

    uv run --project /Users/nicolaslazaro/Desktop/work/pydrology \
        python <hydrologeez-worktree>/scripts/generate_hbv_fixtures.py

Writes three self-describing .npz artifacts into <repo>/tests/fixtures/.
Deterministic: no randomness. Re-running reproduces the array contents exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd  # ty: ignore[unresolved-import]
from pydrology._core import hbv_light as rust  # ty: ignore[unresolved-import]
from pydrology.models.hbv_light import Parameters, run  # ty: ignore[unresolved-import]
from pydrology.types import ForcingData  # ty: ignore[unresolved-import]

PYDROLOGY_ROOT = Path("/Users/nicolaslazaro/Desktop/work/pydrology")
DATA_REL = "data/mountainous-us-basins/REGION_NAME=camels/data_type=timeseries/gauge_id=camels_06224000/data.parquet"
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

# Canonical, integer-maxbas param set (strictly in-bounds).
CANONICAL_PARAMS = Parameters(
    tt=0.0,
    cfmax=3.5,
    sfcf=1.0,
    cwh=0.1,
    cfr=0.05,
    fc=250.0,
    lp=0.7,
    beta=2.0,
    k0=0.3,
    k1=0.1,
    k2=0.05,
    perc=2.0,
    uzl=20.0,
    maxbas=3.0,
)
# Fractional-maxbas param set (maxbas=2.5 -> 3 renormalized UH weights).
MAXBAS25_PARAMS = Parameters(
    tt=0.5,
    cfmax=5.0,
    sfcf=1.1,
    cwh=0.1,
    cfr=0.05,
    fc=300.0,
    lp=0.7,
    beta=2.5,
    k0=0.4,
    k1=0.15,
    k2=0.04,
    perc=2.5,
    uzl=25.0,
    maxbas=2.5,
)

# Integer + fractional maxbas spanning the [1, 7] bounds.
MAXBAS_GRID = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 5.0, 7.0])


def params_vec(p: Parameters) -> np.ndarray:
    return np.array(
        [
            p.tt,
            p.cfmax,
            p.sfcf,
            p.cwh,
            p.cfr,
            p.fc,
            p.lp,
            p.beta,
            p.k0,
            p.k1,
            p.k2,
            p.perc,
            p.uzl,
            p.maxbas,
        ],
        dtype=np.float64,
    )


def load_forcing(data_path: Path) -> ForcingData:
    df = pd.read_parquet(data_path)
    return ForcingData(
        time=df["date"].to_numpy(),
        precip=df["mswep_precipitation"].to_numpy(),
        pet=df["potential_evaporation_sum_FAO_PENMAN_MONTEITH"].to_numpy(),
        temp=df["temperature_2m_mean"].to_numpy(),  # HBV: temp REQUIRED, field name 'temp'
    )


def make_run_fixture(out: Path, name: str, params: Parameters, forcing: ForcingData) -> None:
    fluxes = run(params, forcing).fluxes.to_dict()  # 20 keys
    np.savez(
        out / name,
        params=params_vec(params),
        param_names=PARAM_NAMES,
        warmup_length=np.array(WARMUP_LENGTH),
        basin_id=np.array(BASIN_ID),
        **fluxes,
    )


def make_weights_table(out: Path) -> None:
    weights = np.zeros((MAXBAS_GRID.size, ROUTING_BUFFER_SIZE), dtype=np.float64)
    for i, maxbas in enumerate(MAXBAS_GRID):
        w = np.asarray(rust.hbv_triangular_weights(float(maxbas)), dtype=np.float64)
        weights[i, : w.size] = w  # zero-pad beyond ceil(maxbas)
    np.savez(out / "hbv_triangular_weights.npz", maxbas_grid=MAXBAS_GRID, weights=weights)


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

    make_run_fixture(out, "hbv_camels_06224000.npz", CANONICAL_PARAMS, forcing)
    make_run_fixture(out, "hbv_camels_06224000_maxbas25.npz", MAXBAS25_PARAMS, forcing)
    make_weights_table(out)
    print(f"Wrote 3 HBV fixtures to {out}")


if __name__ == "__main__":
    main()
