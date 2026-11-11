"""Discharge objectives work without a hydrological model."""

import math

import pytest

torch = pytest.importorskip("torch")
from torch.testing import assert_close  # noqa: E402


def test_masked_mse_keeps_observation_weighting() -> None:
    from hydrologeez.losses import masked_mse

    prediction = torch.tensor([[1.0, 3.0], [2.0, float("nan")]], requires_grad=True)
    target = torch.zeros_like(prediction)
    selected = torch.tensor([[True, True], [True, False]])
    loss = masked_mse(prediction, target, selected)
    assert_close(loss, torch.tensor(14 / 3))
    loss.backward()
    assert_close(prediction.grad, torch.tensor([[2 / 3, 2.0], [4 / 3, 0.0]]))


def test_soft_npe_worked_components_and_temporal_gradient() -> None:
    from hydrologeez.losses import soft_npe_loss

    # Permutation preserves mean and FDC: only the temporal component differs.
    prediction = torch.tensor([[1.0], [4.0], [2.0]], dtype=torch.float64, requires_grad=True)
    observed = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float64)

    # softness=3/7 gives temperature=1. Ascending soft ranks include self=0.5.
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    ranks = [0.5 + sigmoid(-3) + sigmoid(-1), 0.5 + sigmoid(3) + sigmoid(2), 0.5 + sigmoid(1) + sigmoid(-2)]
    centered = [r - 1.5 for r in ranks]
    correlation = (-centered[0] + centered[2]) / math.sqrt(2 * sum(r * r for r in centered))
    loss = soft_npe_loss(prediction, observed, torch.ones_like(observed, dtype=torch.bool), softness=3 / 7)
    assert_close(loss, torch.tensor(1 - correlation, dtype=torch.float64))
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.norm() > 0.01


def test_soft_npe_basin_weighting_ties_missing_and_scale() -> None:
    from hydrologeez.losses import soft_npe_loss

    # Tied two-level ranks have correlation one for every finite softness.
    # Basin 0: beta=2, identical normalized FDC => distance=1.
    # Basin 1: beta=3, identical normalized FDC => distance=2.
    target = torch.tensor([[0.0, 1.0, 5.0], [0.0, 2.0, 5.0], [2.0, float("nan"), 5.0], [2.0, 99.0, 5.0]])
    prediction = torch.tensor([[0.0, 3.0, 8.0], [0.0, 6.0, 8.0], [4.0, float("nan"), 8.0], [4.0, -99.0, 8.0]])
    selected = torch.ones_like(target, dtype=torch.bool)
    selected[-1, 1] = False
    assert_close(soft_npe_loss(prediction, target, selected), torch.tensor(1.5))
    for scale in (1e-4, 1e4):
        assert_close(soft_npe_loss(prediction * scale, target * scale, selected), torch.tensor(1.5))
    # At finite softness unequal three-level spacing changes the target ranks.
    perfect = torch.tensor([[0.0], [1.0], [4.0]], requires_grad=True)
    loss = soft_npe_loss(perfect, perfect.detach(), torch.ones_like(perfect, dtype=torch.bool), softness=1.0)
    assert loss > 0
    loss.backward()
    assert perfect.grad is not None and torch.isfinite(perfect.grad).all()


@pytest.mark.parametrize("observed", [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [float("nan"), 1.0, float("nan")]])
def test_soft_npe_no_eligible_basin(observed: list[float]) -> None:
    from hydrologeez.losses import soft_npe_loss

    target = torch.tensor(observed)[:, None]
    with pytest.raises(ValueError, match="at least two nonconstant observations"):
        soft_npe_loss(torch.ones_like(target), target, torch.ones_like(target, dtype=torch.bool))


@pytest.mark.parametrize(
    "predicted",
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, 2.0, 3.0], [1.0, float("nan"), 3.0], [1.0, float("inf"), 3.0]],
)
def test_soft_npe_model_failures_raise(predicted: list[float]) -> None:
    from hydrologeez.losses import soft_npe_loss

    target = torch.tensor([[0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="predictions"):
        soft_npe_loss(torch.tensor(predicted)[:, None], target, torch.ones_like(target, dtype=torch.bool))


@pytest.mark.parametrize("bad", [-1.0, float("inf")])
def test_soft_npe_invalid_observations_raise(bad: float) -> None:
    from hydrologeez.losses import soft_npe_loss

    target = torch.tensor([[0.0], [bad], [2.0]])
    with pytest.raises(ValueError, match="observations"):
        soft_npe_loss(torch.ones_like(target), target, torch.ones_like(target, dtype=torch.bool))


def test_soft_npe_gradcheck() -> None:
    from hydrologeez.losses import soft_npe_loss

    prediction = torch.tensor([[0.4], [3.0], [1.5], [2.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.0], [1.0], [2.0], [4.0]], dtype=torch.float64)
    selected = torch.ones_like(target, dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda p: soft_npe_loss(p, target, selected), (prediction,))


def test_soft_npe_nonzero_fdc_and_zero_loss_gradient() -> None:
    from hydrologeez.losses import soft_npe_loss

    observed = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    selected = torch.ones_like(observed, dtype=torch.bool)
    # Equally spaced triples give exactly linear soft ranks: rho=1.
    # beta=3/2; normalized FDC L1/2=1/18.
    prediction = observed + 1
    assert_close(soft_npe_loss(prediction, observed, selected), torch.tensor(math.sqrt(82) / 18, dtype=torch.float64))
    perfect = observed.clone().requires_grad_()
    loss = soft_npe_loss(perfect, observed, selected)
    assert_close(loss, torch.zeros_like(loss), atol=1e-14, rtol=0)
    loss.backward()
    assert perfect.grad is not None and torch.isfinite(perfect.grad).all()
