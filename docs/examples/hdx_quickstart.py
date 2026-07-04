"""Round-trip a hydrologeez prediction through HDX."""

import os

os.environ["JAX_ENABLE_X64"] = "1"

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hydrologeez import from_hdx, to_hdx  # noqa: E402
from hydrologeez.models.gr6j import GR6J, GR6JForcing  # noqa: E402


def _model() -> GR6J:
    return GR6J(
        x1=jnp.asarray(300.0, dtype=jnp.float64),
        x2=jnp.asarray(1.0, dtype=jnp.float64),
        x3=jnp.asarray(120.0, dtype=jnp.float64),
        x4=jnp.asarray(2.5, dtype=jnp.float64),
        x5=jnp.asarray(0.5, dtype=jnp.float64),
        x6=jnp.asarray(20.0, dtype=jnp.float64),
    )


def _forcing() -> GR6JForcing:
    return GR6JForcing(
        precip=jnp.asarray(np.linspace(2.0, 12.0, 20), dtype=jnp.float64),
        pet=jnp.asarray(np.linspace(0.5, 3.0, 20), dtype=jnp.float64),
    )


def _times(length: int) -> np.ndarray:
    start = np.datetime64("2001-01-01", "us")
    return start + np.arange(length).astype("timedelta64[D]")


def main() -> None:
    streamflow = _model().run(_forcing())
    times = _times(len(streamflow))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "prediction"
        to_hdx(
            root,
            streamflow,
            times,
            ["0001"],
            statics={"drainage_area": np.array(1234.5)},
        )
        data = from_hdx(root, forcing_type=None)

    if data.streamflow is None:
        raise RuntimeError("HDX round-trip did not reload streamflow")

    max_abs_diff = float(np.max(np.abs(np.asarray(data.streamflow) - np.asarray(streamflow))))
    drainage_area = float(data.statics["drainage_area"])
    print(
        "[hdx] round-trip max abs streamflow diff "
        f"{max_abs_diff:.2e}; basins={data.basin_ids}; drainage_area={drainage_area:.1f}"
    )


main()
