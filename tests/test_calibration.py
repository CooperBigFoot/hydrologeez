from datetime import date

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hydrologeez.calibration import Period, scoring_mask


def test_coverage_counts_only_split_days_and_keeps_missing_excluded() -> None:
    dates = np.arange(date(2000, 12, 29).toordinal(), date(2001, 1, 5).toordinal(), dtype=np.int64)
    observed = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0])
    mask, coverage = scoring_mask(dates, observed, Period("2000-12-30", "2001-01-04"), minimum_days=2)
    assert_array_equal(mask, [False, False, False, True, True, False, False])
    assert coverage == {2000: 1, 2001: 2}


def test_population_mapping_and_candidate_specific_continuous_warmup() -> None:
    from hydrologeez.calibration import SEARCH_BOUNDS, CandidateEvaluator
    from hydrologeez.hbv_light import Forcing, simulate

    dates = np.arange(date(2000, 1, 1).toordinal(), date(2000, 1, 21).toordinal(), dtype=np.int64)
    forcing = Forcing(np.tile([4.0, 8.0, 1.0], (20, 1, 1)), np.ones(1), np.array([0, 1], dtype=np.int64))
    observed = np.arange(1.0, 21.0)
    evaluator = CandidateEvaluator(
        forcing, dates, observed, 0, Period("2000-01-06", "2000-01-15"), validation=Period("2000-01-15", "2000-01-21")
    )
    candidates = np.stack([SEARCH_BOUNDS[:, 0], SEARCH_BOUNDS[:, 1]])
    q = evaluator.discharge(candidates)
    for i in range(2):
        parameters, initial = evaluator.inputs(candidates[i : i + 1])
        warm = simulate(forcing, parameters, initial, stop=5)
        continued = simulate(forcing, parameters, warm.final, start=5)
        np.testing.assert_allclose(q[5:, i], continued.outlet[:, 0])
    one = evaluator.scores(candidates, batch_size=1)
    both = evaluator.scores(candidates, batch_size=2)
    assert one == both
    assert evaluator.coverage["training"] == {2000: 9}


def test_seeded_runs_reproduce_and_preserve_per_run_models() -> None:
    from hydrologeez.calibration import SEARCH_BOUNDS, CandidateEvaluator, calibrate
    from hydrologeez.hbv_light import Forcing

    dates = np.arange(date(2000, 1, 1).toordinal(), date(2000, 2, 1).toordinal(), dtype=np.int64)
    forcing = Forcing(np.tile([4.0, 8.0, 1.0], (31, 1, 1)), np.ones(1), np.array([0, 1], dtype=np.int64))
    evaluator = CandidateEvaluator(forcing, dates, np.arange(1.0, 32.0), 0, Period("2000-01-06", "2000-02-01"))
    first = calibrate(evaluator, seeds=[11, 22], objectives=("nse",), population_size=4, generations=1, batch_size=2)
    second = calibrate(evaluator, seeds=[11, 22], objectives=("nse",), population_size=4, generations=1, batch_size=4)
    assert len(first.solutions) == 2
    assert [s.seed for s in first.solutions] == [11, 22]
    assert [s.run for s in first.solutions] == [0, 1]
    for a, b in zip(first.solutions, second.solutions, strict=True):
        assert_array_equal(a.parameters, b.parameters)
        assert a.scores == b.scores
        assert np.all(a.parameters >= SEARCH_BOUNDS[:, 0])
        assert np.all(a.parameters <= SEARCH_BOUNDS[:, 1])
    multi = calibrate(
        evaluator, seeds=[11], objectives=("nse", "nse"), population_size=4, generations=1, retain=3, batch_size=4
    )
    assert 1 <= len(multi.solutions) <= 3
    assert len({tuple(s.parameters) for s in multi.solutions}) == len(multi.solutions)


def test_uniform_initial_population_and_invalid_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from hydrologeez.calibration import SEARCH_BOUNDS, CandidateEvaluator, calibrate
    from hydrologeez.hbv_light import Forcing

    dates = np.arange(date(2000, 1, 1).toordinal(), date(2000, 1, 11).toordinal(), dtype=np.int64)
    forcing = Forcing(np.tile([4.0, 8.0, 1.0], (10, 1, 1)), np.ones(1), np.array([0, 1], dtype=np.int64))
    evaluator = CandidateEvaluator(forcing, dates, np.ones(10), 0, Period("2000-01-01", "2000-01-11"))
    populations: list[np.ndarray] = []
    original = evaluator.scores

    def capture(
        candidates: np.ndarray, *, batch_size: int, periods: tuple[str, ...] = ("training",)
    ) -> list[dict[str, dict[str, float]]]:
        populations.append(candidates.copy())
        return original(candidates, batch_size=batch_size, periods=periods)

    monkeypatch.setattr(evaluator, "scores", capture)
    result = calibrate(evaluator, seeds=[81], objectives=("nse",), population_size=4, generations=0)
    rng = np.random.default_rng(np.random.SeedSequence(81).spawn(4)[0])
    expected = rng.uniform(SEARCH_BOUNDS[:, 0], SEARCH_BOUNDS[:, 1], size=(4, 13))
    assert_array_equal(populations[0], expected)
    assert result.solutions == ()
    assert result.runs[0].retained == 0


def test_validation_and_missing_observations_do_not_steer_optimizer() -> None:
    from hydrologeez.calibration import CandidateEvaluator, calibrate
    from hydrologeez.hbv_light import Forcing

    dates = np.arange(date(2000, 1, 1).toordinal(), date(2000, 2, 1).toordinal(), dtype=np.int64)
    forcing = Forcing(np.tile([4.0, 8.0, 1.0], (31, 1, 1)), np.ones(1), np.array([0, 1], dtype=np.int64))
    observed = np.arange(1.0, 32.0)
    observed[8] = np.nan
    changed = observed.copy()
    changed[20:] *= 100
    results = []
    for values in (observed, changed):
        evaluator = CandidateEvaluator(
            forcing, dates, values, 0, Period("2000-01-06", "2000-01-21"), validation=Period("2000-01-21", "2000-02-01")
        )
        assert not evaluator.masks["training"][8]
        results.append(
            calibrate(
                evaluator, seeds=[11], objectives=("nse",), population_size=4, generations=1, report=("validation",)
            )
        )
    assert_array_equal(results[0].solutions[0].parameters, results[1].solutions[0].parameters)
    assert results[0].solutions[0].objectives == results[1].solutions[0].objectives
    assert results[0].solutions[0].scores != results[1].solutions[0].scores
