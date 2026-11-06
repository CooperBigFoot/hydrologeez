from datetime import date

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hydrologeez.calibration import SEARCH_BOUNDS, CandidateEvaluator, JointEvaluator, Period
from hydrologeez.hbv_light import Forcing


def basin(observed: np.ndarray) -> CandidateEvaluator:
    dates = np.arange(date(2000, 1, 1).toordinal(), date(2000, 1, 9).toordinal(), dtype=np.int64)
    forcing = Forcing(np.tile([4.0, 8.0, 1.0], (8, 1, 1)), np.ones(1), np.array([0, 1], dtype=np.int64))
    return CandidateEvaluator(
        forcing, dates, observed, 0, Period("2000-01-02", "2000-01-06"), validation=Period("2000-01-06", "2000-01-09")
    )


def test_weighted_losses_convert_before_aggregation_and_fix_training_means(monkeypatch: pytest.MonkeyPatch) -> None:
    first = basin(np.array([900.0, 1.0, np.nan, 3.0, 2.0, 20.0, 30.0, 40.0]))
    second = basin(np.array([900.0, 2.0, 4.0, 6.0, 4.0, 40.0, 60.0, 80.0]))
    # Additive errors +1 and -1 yield opposite biases, which must not cancel.
    for evaluator, error in ((first, 1.0), (second, -1.0)):
        values = np.nan_to_num(evaluator.observed) + error

        def discharge(candidates: np.ndarray, values: np.ndarray = values) -> np.ndarray:
            return np.repeat(values[:, None], len(candidates), axis=1)

        monkeypatch.setattr(evaluator, "discharge", discharge)
    joint = JointEvaluator({"upstream": first, "downstream": second}, weights={"upstream": 1.0, "downstream": 3.0})
    objectives = ("rmse", "mse", "beta", "beta_nse", "alpha", "pearson", "nse")
    candidates = SEARCH_BOUNDS.mean(axis=1)[None, :]
    scores = joint.scores(candidates, batch_size=1, periods=("training", "validation"))
    losses = joint.losses(scores, objectives)
    # Masked training means: 2 and 4. Std: sqrt(2/3) and sqrt(2).
    expected = [
        0.25 / 2 + 0.75 / 4,
        0.25 / 4 + 0.75 / 16,
        0.25 / 2 + 0.75 / 4,
        0.25 / np.sqrt(2 / 3) + 0.75 / np.sqrt(2),
        0.0,
        0.0,
        0.25 / (2 / 3) + 0.75 / 2,
    ]
    assert_allclose(losses[0], expected, atol=1e-14)
    assert_allclose(joint.losses(scores, ("rmse", "mse"), period="validation")[0], expected[:2])
    assert scores[0]["upstream"]["training"]["rmse"] == 1.0
    assert joint.training_means == {"upstream": 2.0, "downstream": 4.0}
    equal = JointEvaluator({"upstream": first, "downstream": second})
    assert_allclose(equal.losses(scores, ("rmse",))[0], [0.5 / 2 + 0.5 / 4])


@pytest.mark.parametrize("weights", [{"a": 0.0, "b": 1.0}, {"a": -1.0, "b": 1.0}, {"a": np.inf, "b": 1.0}, {"a": 1.0}])
def test_invalid_weights_rejected(weights: dict[str, float]) -> None:
    evaluator = basin(np.arange(1.0, 9.0))
    with pytest.raises(ValueError, match="weights"):
        JointEvaluator({"a": evaluator, "b": evaluator}, weights=weights)


@pytest.mark.parametrize(
    "values,objective",
    [
        (np.zeros(8), "rmse"),
        (np.ones(8), "pearson"),
        (np.ones(8), "nse"),
        (np.arange(1.0, 9.0), "peak_timing"),
        (np.array([1.0, np.nan, np.nan, np.nan, 2.0, 3.0, 4.0, 5.0]), "mse"),
    ],
)
def test_observation_incompatibility_rejected_before_search(values: np.ndarray, objective: str) -> None:
    from hydrologeez.calibration import calibrate_joint

    joint = JointEvaluator({"basin": basin(values)})
    with pytest.raises(ValueError, match="basin.*" + objective):
        calibrate_joint(joint, seeds=[1], objectives=(objective,), population_size=4, generations=0)


def test_joint_search_reproduces_and_reports_independent_basin_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    from hydrologeez.calibration import calibrate_joint

    first = basin(np.array([100.0, 1.0, np.nan, 3.0, 4.0, 20.0, 30.0, 40.0]))
    second = basin(np.array([200.0, 2.0, 3.0, 6.0, 8.0, 40.0, 60.0, 80.0]))
    joint = JointEvaluator({"a": first, "b": second})
    sizes: list[int] = []
    original = first.discharge

    def capture(candidates: np.ndarray) -> np.ndarray:
        sizes.append(len(candidates))
        return original(candidates)

    monkeypatch.setattr(first, "discharge", capture)
    result = calibrate_joint(
        joint,
        batch_size=1,
        seeds=[11, 22],
        objectives=("rmse", "beta"),
        population_size=4,
        generations=1,
        retain=2,
        report=("training", "validation"),
    )
    assert max(sizes) == 1
    repeated = calibrate_joint(
        joint,
        batch_size=4,
        seeds=[11, 22],
        objectives=("rmse", "beta"),
        population_size=4,
        generations=1,
        retain=2,
        report=("training", "validation"),
    )
    assert result.solutions
    assert result.weights == {"a": 0.5, "b": 0.5}
    assert result.objectives == ("rmse", "beta")
    assert result.training_means == joint.training_means
    for a, b in zip(result.solutions, repeated.solutions, strict=True):
        assert_allclose(a.parameters, b.parameters, rtol=0, atol=0)
        assert a.seed == b.seed == [11, 22][a.run]
        assert a.losses == b.losses
        for name, evaluator in joint.basins.items():
            expected = evaluator.scores(a.parameters[None, :], batch_size=1, periods=("training", "validation"))[0]
            for period in ("training", "validation"):
                assert_allclose(list(a.scores[name][period].values()), list(expected[period].values()), equal_nan=True)
        assert_allclose(a.losses, joint.losses([a.scores], result.objectives)[0])
    single = calibrate_joint(joint, seeds=[1], objectives=("rmse",), population_size=4, generations=0)
    assert len(single.solutions) == 1


def test_candidate_undefined_in_one_basin_invalidates_whole_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    from hydrologeez.calibration import calibrate_joint

    first, second = basin(np.arange(1.0, 9.0)), basin(np.arange(2.0, 10.0))

    def constant(candidates: np.ndarray) -> np.ndarray:
        return np.ones((8, len(candidates)))

    monkeypatch.setattr(second, "discharge", constant)
    joint = JointEvaluator({"a": first, "b": second})
    result = calibrate_joint(joint, seeds=[1], objectives=("pearson",), population_size=4, generations=0)
    assert result.solutions == ()
    assert result.runs[0].retained == 0


def test_distinct_basin_forcing_states_and_validation_do_not_leak() -> None:
    from hydrologeez.calibration import calibrate_joint

    dates = np.arange(date(2000, 1, 1).toordinal(), date(2000, 1, 9).toordinal(), dtype=np.int64)
    forcing = Forcing(
        np.tile([[4.0, 8.0, 1.0], [40.0, -2.0, 0.0]], (8, 1, 1)), np.ones(2), np.array([0, 1, 2], dtype=np.int64)
    )
    evaluators = {
        str(index): CandidateEvaluator(
            forcing,
            dates,
            np.arange(1.0, 9.0) * (index + 1),
            index,
            Period("2000-01-02", "2000-01-06"),
            validation=Period("2000-01-06", "2000-01-09"),
        )
        for index in range(2)
    }
    candidates = np.stack([SEARCH_BOUNDS[:, 0], SEARCH_BOUNDS[:, 1]])
    joint = JointEvaluator(evaluators)
    scores = joint.scores(candidates, batch_size=1)
    assert not np.array_equal(evaluators["0"].discharge(candidates), evaluators["1"].discharge(candidates))
    # Each score matches a fresh independent single-basin simulation.
    for index in range(2):
        expected = evaluators[str(index)].scores(candidates, batch_size=2)
        for record, single in zip(scores, expected, strict=True):
            assert_allclose(
                list(record[str(index)]["training"].values()), list(single["training"].values()), equal_nan=True
            )
    first = calibrate_joint(
        joint, seeds=[11], objectives=("rmse",), population_size=4, generations=1, report=("validation",)
    )
    evaluators["1"].observed[5:] *= 100
    changed = JointEvaluator(evaluators)
    second = calibrate_joint(
        changed, seeds=[11], objectives=("rmse",), population_size=4, generations=1, report=("validation",)
    )
    assert first.training_means == second.training_means
    assert_allclose(first.solutions[0].parameters, second.solutions[0].parameters, atol=0, rtol=0)
    assert first.solutions[0].losses == second.solutions[0].losses
    assert first.solutions[0].scores["1"]["validation"]["rmse"] != second.solutions[0].scores["1"]["validation"]["rmse"]
