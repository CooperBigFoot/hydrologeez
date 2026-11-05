"""Read-back correctness, independent of timing measurements."""

from pathlib import Path

import numpy as np
import pytest

from hydrologeez.hbv_light import Forcing, simulate
from scripts.hbv_output_writing import parameters_for, write_run


@pytest.mark.parametrize("kind", ["resident", "batch", "tile", "compressed"])
@pytest.mark.parametrize("history", [False, True])
def test_written_values_and_associations(tmp_path: Path, kind: str, history: bool) -> None:
    # Unequal basin zone counts and routing lengths catch packed-axis mixups.
    values = np.array(
        [[[8.0, -2.0, 0.0], [3.0, 2.0, 1.0], [5.0, 4.0, 2.0]], [[0.0, 5.0, 1.0], [1.0, -1.0, 0.0], [9.0, 8.0, 2.0]]] * 4
    )
    forcing = Forcing(values, np.array([1.0, 0.3, 0.7]), np.array([0, 1, 3], dtype=np.int64))
    parameters, initial, _ = parameters_for(forcing, 0, 7, 4)
    reference = simulate(forcing, parameters, initial, history=history)
    tmp_path = tmp_path / "output"
    write_run(tmp_path, forcing, ["basin-a", "basin-b"], np.arange(100, 108), 7, 4, kind, history, batch=3, window=3)
    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["basins"] == ["basin-a", "basin-b"]
    assert manifest["dates"] == list(range(100, 108))
    covered = np.zeros((8, 7), dtype=int)
    for shard in manifest["shards"]:
        a, b, start, stop = shard["rows"] + shard["days"]
        with np.load(tmp_path / shard["file"]) as stored:
            covered[start:stop, a:b] += 1
            np.testing.assert_array_equal(stored["basin_index"], np.arange(a, b) % 2)
            np.testing.assert_array_equal(stored["parameter_id"], np.arange(a, b) // 2)
            zs, ze = reference.zone_offsets[[a, b]]
            rs, re = reference.routing_offsets[[a, b]]
            np.testing.assert_array_equal(stored["zone_offsets"], reference.zone_offsets[a : b + 1] - zs)
            np.testing.assert_array_equal(stored["routing_offsets"], reference.routing_offsets[a : b + 1] - rs)
            np.testing.assert_array_equal(stored["parameters_zone"], parameters.zone[zs:ze])
            np.testing.assert_array_equal(stored["parameters_groundwater"], parameters.groundwater[a:b])
            np.testing.assert_array_equal(stored["maxbas"], parameters.maxbas[a:b])
            np.testing.assert_array_equal(stored["outlet"], reference.outlet[start:stop, a:b])
            axes = {
                "zone": (zs, ze),
                "groundwater": (a, b),
                "routing": (rs, re),
                "zone_flux": (zs, ze),
                "basin_flux": (a, b),
            }
            if history:
                assert reference.history is not None
                for name, (left, right) in axes.items():
                    np.testing.assert_array_equal(
                        stored[name], getattr(reference.history, name)[start:stop, left:right]
                    )
            else:
                assert "zone_flux" not in stored
            if stop == 8:
                for name in ("zone", "groundwater", "routing"):
                    left, right = axes[name]
                    np.testing.assert_array_equal(stored["final_" + name], getattr(reference.final, name)[left:right])
    np.testing.assert_array_equal(covered, 1)
