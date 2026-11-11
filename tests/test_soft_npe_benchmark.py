"""Numerical contract of the experimental CPU objective."""

import numpy as np
import pytest
import torch

from hydrologeez.losses import soft_npe_loss
from scripts.soft_npe_benchmark import SoftNPE


@pytest.mark.parametrize("softness", [0.01, 0.03, 0.1, 0.3, 1.0])
def test_matches_tensor_loss(softness: float) -> None:
    observed = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 10.0, 30.0])
    prediction = np.array([1.0, 0.0, 0.0, 3.0, 3.0, 40.0, 5.0])
    expected = soft_npe_loss(
        torch.tensor(prediction[:, None]),
        torch.tensor(observed[:, None]),
        torch.ones((7, 1), dtype=torch.bool),
        softness=softness,
    ).item()
    assert SoftNPE(observed, softness)(prediction) == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("prediction", [np.ones(4), np.array([0.0, 1.0, np.nan, 2.0]), np.array([0.0, -1.0, 2.0, 3.0])])
def test_invalid_candidates_are_never_rewarded(prediction: np.ndarray) -> None:
    assert SoftNPE(np.arange(4, dtype=float), 0.1)(prediction) == float("inf")


def test_full_period_agrees_and_is_scale_invariant() -> None:
    rng = np.random.default_rng(42)
    observed = rng.lognormal(size=3652)
    predicted = rng.lognormal(size=3652)
    expected = soft_npe_loss(
        torch.tensor(predicted[:, None]),
        torch.tensor(observed[:, None]),
        torch.ones((3652, 1), dtype=torch.bool),
        softness=0.1,
    ).item()
    actual = SoftNPE(observed, 0.1)(predicted)
    assert actual == pytest.approx(expected, abs=2e-13)
    assert SoftNPE(observed * 100, 0.1)(predicted * 100) == pytest.approx(actual, abs=2e-13)
