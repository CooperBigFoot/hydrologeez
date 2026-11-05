"""Connected scoring against controlled saved discharge and prepared observations."""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hydrologeez.camels import Basin
from hydrologeez.metrics import evaluate


@pytest.fixture
def stores(tmp_path: Path) -> tuple[Path, Path]:
    output, prepared = tmp_path / "output", tmp_path / "observations"
    output.mkdir()
    prepared.mkdir()
    first = date(2000, 1, 1).toordinal()
    basins = []
    for gauge, values in [("a", [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0]), ("b", [3.0] * 8)]:
        folder = prepared / gauge
        folder.mkdir()
        np.save(folder / "discharge_cfs.npy", values)
        np.save(folder / "present.npy", np.isfinite(values))
        np.save(folder / "qc.npy", np.full(8, "A"))
        basins.append(
            asdict(
                Basin(
                    gauge,
                    40.0,
                    0.028316846592 * 86400 * 1000,
                    None,
                    None,
                    first,
                    first + 1,
                    first + 2,
                    first + 10,
                    "",
                    "",
                    0,
                    0,
                    0,
                    0,
                )
            )
        )
    (prepared / "metadata.json").write_text(json.dumps({"format": 1, "observation_units": "cfs", "basins": basins}))
    # Disk order: a/p, b/q, b/p, a/q. Recording excludes the first day.
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "format": 1,
                "start": first,
                "record_start": first + 1,
                "stop": first + 11,
                "variables": ["discharge"],
                "basin_ids": ["a", "b"],
                "parameter_ids": ["p", "q", "p", "q"],
                "basin_index": [0, 1, 1, 0],
                "zone_offsets": [0, 1, 2, 3, 4],
                "routing_offsets": [0, 0, 0, 0, 0],
                "batch_size": 2,
                "metadata": {"discharge": {"units": "mm/day"}},
            }
        )
    )
    for offset in (0, 2):
        batch = output / f"batch-{offset}"
        batch.mkdir()
        np.save(batch / "parameters_zone.npy", np.zeros((2, 8)))
        np.save(batch / "parameters_groundwater.npy", np.zeros((2, 4)))
        np.save(batch / "maxbas.npy", np.ones(2))
        np.save(batch / "discharge.npy", np.arange(1.0, 11.0)[:, None] * np.array([[offset + 1.0, offset + 2.0]]))
        (batch / "complete.json").write_text(
            json.dumps({"files": {p.name: p.stat().st_size for p in batch.glob("*.npy")}})
        )
    return output, prepared


def test_identified_scores_match_complete_period_arrays(stores: tuple[Path, Path]) -> None:
    from hydrologeez.evaluation import evaluate_discharge

    output, prepared = stores
    periods = [("2000-01-02", "2000-01-12"), ("2000-01-04", "2000-01-10")]
    scores = evaluate_discharge(output, prepared, basin_ids=["b", "a"], parameter_ids=["q", "p"], periods=periods)
    assert [(s.basin_id, s.parameter_id, s.start, s.stop) for s in scores] == [
        (b, p, start, stop) for b in ["b", "a"] for p in ["q", "p"] for start, stop in periods
    ]
    for score in scores:
        obs = np.array(
            [np.nan, *([3.0] * 8 if score.basin_id == "b" else [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0]), np.nan]
        )
        multiplier = {("a", "p"): 1, ("b", "q"): 2, ("b", "p"): 3, ("a", "q"): 4}[score.basin_id, score.parameter_id]
        sl = slice(0, 10) if score.start == periods[0][0] else slice(2, 8)
        expected = evaluate(obs[sl], np.arange(1.0, 11.0)[sl] * multiplier)
        assert score.scores.keys() == expected.keys()
        assert_allclose(list(score.scores.values()), list(expected.values()), equal_nan=True)
        assert score.observed_days == np.isfinite(obs[sl]).sum()
        assert score.present_days == score.observed_days


def test_saved_evaluation_passes_complete_daily_dates(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from numpy.typing import NDArray

    from hydrologeez import evaluation

    def score(
        observed: NDArray[np.float64], simulated: NDArray[np.float64], *, dates: NDArray[np.datetime64]
    ) -> dict[str, float]:
        np.testing.assert_array_equal(dates, np.arange("2000-01-02", "2000-01-12", dtype="datetime64[D]"))
        assert observed.shape == simulated.shape == dates.shape
        np.testing.assert_array_equal(np.flatnonzero(np.isnan(observed)), [0, 3, 9])
        return {"peak_timing": 2.0}

    monkeypatch.setattr(evaluation, "evaluate", score)
    result = evaluation.evaluate_discharge(
        *stores, basin_ids=["a"], parameter_ids=["p"], periods=[("2000-01-02", "2000-01-12")]
    )[0]
    assert result.scores == {"peak_timing": 2.0}


@pytest.mark.parametrize("location,units", [("output", "cfs"), ("observations", "mm/day")])
def test_rejects_mismatched_units(stores: tuple[Path, Path], location: str, units: str) -> None:
    from hydrologeez.evaluation import evaluate_discharge

    output, prepared = stores
    path = (output if location == "output" else prepared) / "metadata.json"
    metadata = json.loads(path.read_text())
    if location == "output":
        metadata["metadata"]["discharge"]["units"] = units
    else:
        metadata["observation_units"] = units
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="units"):
        evaluate_discharge(
            output, prepared, basin_ids=["a"], parameter_ids=["p"], periods=[("2000-01-02", "2000-01-12")]
        )


@pytest.mark.parametrize(
    "period", [("2000-01-01", "2000-01-12"), ("2000-01-02", "2000-01-13"), ("2000-01-03", "2000-01-03")]
)
def test_rejects_unrecorded_dates(stores: tuple[Path, Path], period: tuple[str, str]) -> None:
    from hydrologeez.evaluation import evaluate_discharge

    with pytest.raises(ValueError, match="recorded coverage"):
        evaluate_discharge(*stores, basin_ids=["a"], parameter_ids=["p"], periods=[period])


def test_bounded_reads_without_forcing_or_simulation(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import weakref
    from unittest.mock import Mock

    from hydrologeez.evaluation import evaluate_discharge
    from hydrologeez.hbv_light import output

    original = output.OutputStore.read
    previous: weakref.ReferenceType[output.Selection] | None = None
    calls = 0

    def read(
        self: output.OutputStore,
        *,
        variables: list[str],
        basin_ids: list[str],
        parameter_ids: list[str],
        start: str,
        stop: str,
    ) -> output.Selection:
        nonlocal previous, calls
        assert previous is None or previous() is None
        assert variables == ["discharge"]
        assert len(basin_ids) == len(parameter_ids) == 1
        assert (start, stop) == ("2000-01-02", "2000-01-12")
        selected = original(
            self, variables=variables, basin_ids=basin_ids, parameter_ids=parameter_ids, start=start, stop=stop
        )
        previous = weakref.ref(selected)
        calls += 1
        return selected

    monkeypatch.setattr(output.OutputStore, "read", read)
    monkeypatch.setattr(output, "simulate", Mock(side_effect=AssertionError("must not simulate")))
    # The prepared fixture deliberately has no forcing files.
    results = evaluate_discharge(
        *stores, basin_ids=["b", "a"], parameter_ids=["q", "p"], periods=[("2000-01-02", "2000-01-12")]
    )
    assert len(results) == calls == 4


@pytest.mark.parametrize("day", [0, 1])
def test_invalid_simulations_preserve_observation_mask(stores: tuple[Path, Path], day: int) -> None:
    from hydrologeez.evaluation import evaluate_discharge

    path = stores[0] / "batch-0" / "discharge.npy"
    values = np.load(path)
    values[day, 0] = -1
    np.save(path, values)
    score = evaluate_discharge(*stores, basin_ids=["a"], parameter_ids=["p"], periods=[("2000-01-02", "2000-01-12")])[0]
    if day == 0:  # Outside observation coverage, so invalid simulation is ignored.
        assert np.isfinite(score.scores["nse"])
    else:
        assert all(np.isnan(value) for value in score.scores.values())


def test_missing_observations_report_presence_and_undefined_scores(stores: tuple[Path, Path]) -> None:
    from hydrologeez.evaluation import evaluate_discharge

    folder = stores[1] / "a"
    np.save(folder / "discharge_cfs.npy", np.full(8, -999.0))
    np.save(folder / "present.npy", np.ones(8, dtype=np.bool_))
    score = evaluate_discharge(*stores, basin_ids=["a"], parameter_ids=["p"], periods=[("2000-01-02", "2000-01-12")])[0]
    assert score.observed_days == 0
    assert score.present_days == 8
    assert all(np.isnan(value) for value in score.scores.values())
