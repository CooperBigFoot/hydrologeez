from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars.testing as pl_testing
import pyarrow.parquet as pq
import pytest

from hydrologeez.hdx._polars import require_polars
from hydrologeez.hdx.writer import to_hdx


def test_to_hdx_writes_conformant_two_basin_dataset(tmp_path: Path) -> None:
    pl = require_polars()
    root = tmp_path / "prediction"
    times = [
        np.asarray(["2001-01-03", "2001-01-01", "2001-01-02"], dtype="datetime64[us]"),
        np.asarray(["2001-02-02", "2001-02-01"], dtype="datetime64[us]"),
    ]
    streamflow = np.asarray([[3.0, 1.0, 2.0], [20.0, 10.0, 999.0]], dtype=np.float64)

    to_hdx(
        root,
        streamflow,
        times,
        ["0001", "0002"],
        statics={"drainage_area": np.array([100.0, 250.0])},
    )

    expected_0001 = pl.DataFrame(
        {
            "basin_id": ["0001", "0001", "0001"],
            "time": pl.Series("time", np.asarray(["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[us]")),
            "streamflow": [1.0, 2.0, 3.0],
        },
        schema={"basin_id": pl.String, "time": pl.Datetime("us"), "streamflow": pl.Float64},
    )
    expected_0002 = pl.DataFrame(
        {
            "basin_id": ["0002", "0002"],
            "time": pl.Series("time", np.asarray(["2001-02-01", "2001-02-02"], dtype="datetime64[us]")),
            "streamflow": [10.0, 20.0],
        },
        schema={"basin_id": pl.String, "time": pl.Datetime("us"), "streamflow": pl.Float64},
    )

    actual_0001 = pl.read_parquet(root / "basin=0001" / "scalar_dynamic.parquet")
    actual_0002 = pl.read_parquet(root / "basin=0002" / "scalar_dynamic.parquet")
    schema_0001 = pq.read_schema(root / "basin=0001" / "scalar_dynamic.parquet")
    schema_static = pq.read_schema(root / "scalar_static.parquet")
    assert schema_0001.field("time").nullable is False
    assert schema_0001.field("basin_id").nullable is False
    assert schema_static.field("basin_id").nullable is False
    assert actual_0001.schema == expected_0001.schema
    assert actual_0002.schema == expected_0002.schema
    assert actual_0001["time"].null_count() == 0
    assert actual_0002["time"].null_count() == 0
    assert actual_0001["time"].is_sorted()
    assert actual_0002["time"].is_sorted()
    pl_testing.assert_frame_equal(actual_0001, expected_0001)
    pl_testing.assert_frame_equal(actual_0002, expected_0002)

    expected_static = pl.DataFrame(
        {"basin_id": ["0001", "0002"], "drainage_area": [100.0, 250.0]},
        schema={"basin_id": pl.String, "drainage_area": pl.Float64},
    )
    pl_testing.assert_frame_equal(pl.read_parquet(root / "scalar_static.parquet"), expected_static)

    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest.keys()) == {
        "format_version",
        "name",
        "created_at",
        "producer_version",
        "crs",
        "cadence",
    }
    assert manifest["format_version"] == "0.2"


def test_to_hdx_writes_basin_only_static_when_statics_none(tmp_path: Path) -> None:
    pl = require_polars()
    root = tmp_path / "no-statics"

    to_hdx(
        root,
        np.asarray([1.0, 2.0], dtype=np.float64),
        np.asarray(["2001-01-01", "2001-01-02"], dtype="datetime64[us]"),
        ["0001"],
        statics=None,
    )

    expected_static = pl.DataFrame({"basin_id": ["0001"]}, schema={"basin_id": pl.String})
    pl_testing.assert_frame_equal(pl.read_parquet(root / "scalar_static.parquet"), expected_static)


def test_to_hdx_output_passes_real_validator_when_available(tmp_path: Path) -> None:
    if os.environ.get("CI"):
        pytest.skip("real hdx validator is not required in CI")
    sibling_hdx = Path("/Users/nicolaslazaro/Desktop/work/hdx/target/debug/hdx")
    hdx_bin = str(sibling_hdx) if sibling_hdx.exists() else shutil.which("hdx")
    if hdx_bin is None:
        pytest.skip("hdx validator binary is not available")

    root = tmp_path / "prediction"
    to_hdx(
        root,
        np.asarray([1.0, 2.0], dtype=np.float64),
        np.asarray(["2001-01-01", "2001-01-02"], dtype="datetime64[us]"),
        ["0001"],
    )

    result = subprocess.run([hdx_bin, "validate", str(root)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
