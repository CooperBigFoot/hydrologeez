"""End-to-end hydrologeez HBV-Light forward run + oracle-parity example."""

import os

os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hydrologeez.metrics import kge, nse  # noqa: E402
from hydrologeez.models.hbv import HBVForcing, HBVModel  # noqa: E402

RTOL = 1e-4
ATOL = 1e-6


def _find_fixtures_dir() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / "tests" / "fixtures"
        if (candidate / "hbv_camels_06224000.npz").exists():
            return candidate
    raise FileNotFoundError("could not locate tests/fixtures; run from the hydrologeez repo root")


def _model_from_params(params: np.ndarray) -> HBVModel:
    return HBVModel(
        tt=jnp.asarray(params[0], dtype=jnp.float64),
        cfmax=jnp.asarray(params[1], dtype=jnp.float64),
        sfcf=jnp.asarray(params[2], dtype=jnp.float64),
        cwh=jnp.asarray(params[3], dtype=jnp.float64),
        cfr=jnp.asarray(params[4], dtype=jnp.float64),
        fc=jnp.asarray(params[5], dtype=jnp.float64),
        lp=jnp.asarray(params[6], dtype=jnp.float64),
        beta=jnp.asarray(params[7], dtype=jnp.float64),
        k0=jnp.asarray(params[8], dtype=jnp.float64),
        k1=jnp.asarray(params[9], dtype=jnp.float64),
        k2=jnp.asarray(params[10], dtype=jnp.float64),
        perc=jnp.asarray(params[11], dtype=jnp.float64),
        uzl=jnp.asarray(params[12], dtype=jnp.float64),
        maxbas=jnp.asarray(params[13], dtype=jnp.float64),
    )


def main() -> None:
    data = np.load(_find_fixtures_dir() / "hbv_camels_06224000.npz", allow_pickle=False)
    warmup = int(data["warmup_length"])
    model = _model_from_params(np.asarray(data["params"], dtype=np.float64))
    forcing = HBVForcing(
        precip=jnp.asarray(data["precip"], dtype=jnp.float64),
        pet=jnp.asarray(data["pet"], dtype=jnp.float64),
        temp=jnp.asarray(data["temp"], dtype=jnp.float64),
    )

    sim = np.asarray(model.run(forcing))
    oracle = np.asarray(data["streamflow"], dtype=np.float64)

    sim_eval = sim[warmup:]
    oracle_eval = oracle[warmup:]
    max_rel = float(np.max(np.abs(sim_eval - oracle_eval) / (np.abs(oracle_eval) + 1e-8)))
    nse_val = float(nse(jnp.asarray(oracle_eval), jnp.asarray(sim_eval)))
    kge_val = float(kge(jnp.asarray(oracle_eval), jnp.asarray(sim_eval)))
    print(f"[parity] max relative streamflow error vs committed oracle fixture: {max_rel:.2e}")
    print(f"[metrics] NSE = {nse_val:.6f}; KGE = {kge_val:.6f} (post warm-up)")

    np.testing.assert_allclose(sim_eval, oracle_eval, rtol=RTOL, atol=ATOL)
    print("[parity] streamflow within rtol=1e-4 atol=1e-6 of the committed oracle fixture: OK")


main()
