"""Parity with frozen outputs from the pinned NeuralHydrology implementation.

The fixture records the source hash, dependency versions, and reference invocation.
No NeuralHydrology or xarray dependency is needed to run these tests.
"""

import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest

from hydrologeez.metrics import evaluate


class ParityCase(TypedDict):
    name: str
    start_date: str
    frequency: str
    observed: list[float]
    simulated: list[float]
    expected: dict[str, float]


class ParityFixture(TypedDict):
    cases: list[ParityCase]


METRIC_NAMES = {
    "NSE": "nse",
    "MSE": "mse",
    "RMSE": "rmse",
    "KGE": "kge",
    "Alpha-NSE": "alpha",
    "Beta-KGE": "beta",
    "Beta-NSE": "beta_nse",
    "Pearson-r": "pearson",
    "FHV": "high_flow_bias",
    "FMS": "middle_flow_bias",
    "FLV": "low_flow_bias",
    "Peak-Timing": "peak_timing",
    "Missed-Peaks": "missed_peaks",
    "Peak-MAPE": "peak_mape",
}
FIXTURE = cast(
    ParityFixture,
    json.loads((Path(__file__).parent / "data" / "metrics_neuralhydrology.json").read_text()),
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_deterministic_metrics_match_neuralhydrology(case: ParityCase) -> None:
    observed = np.asarray(case["observed"], dtype=np.float64)
    simulated = np.asarray(case["simulated"], dtype=np.float64)
    dates = np.datetime64(case["start_date"], "D") + np.arange(observed.size).astype("timedelta64[D]")
    result = evaluate(observed, simulated, dates=dates)

    assert case["expected"].keys() == METRIC_NAMES.keys()
    for reference_name, metric_name in METRIC_NAMES.items():
        assert result[metric_name] == pytest.approx(case["expected"][reference_name], rel=1e-12, abs=1e-12), metric_name
