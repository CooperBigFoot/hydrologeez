"""Differentiable tensor losses (requires the ``hybrid`` PyTorch extra)."""

import math

import torch
from torch import Tensor


def masked_mse(prediction: Tensor, target: Tensor, selected: Tensor) -> Tensor:
    """Mean squared error over selected elements, of any matching shape."""
    if prediction.shape != target.shape or selected.shape != target.shape or selected.dtype != torch.bool:
        raise ValueError("prediction, target and boolean mask must have matching shapes")
    if not bool(selected.any()):
        raise ValueError("loss needs selected targets")
    return (prediction[selected] - target[selected]).square().mean()


def _ranks(values: Tensor) -> Tensor:
    """Exact zero-based midranks for observed targets, including ties."""
    _, inverse, counts = torch.unique(values, sorted=True, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).to(values.dtype)
    return (ends - (counts.to(values.dtype) + 1) / 2)[inverse]


def soft_npe_loss(prediction: Tensor, target: Tensor, selected: Tensor, *, softness: float = 0.1) -> Tensor:
    """Mean basin distance from ideal nonparametric KGE components.

    All inputs have shape [time, basin]; ``selected`` is boolean. NaN targets
    are missing. Observation-defined basins with fewer than two values or
    constant flows are omitted; no eligible basins raises ValueError. Scored
    negative/infinite targets and negative/nonfinite predictions are errors,
    even in omitted basins. Eligible basins need positive, nonconstant predicted
    flows (zeros in a varying series are allowed); undefined scores raise.

    Mean ratio and normalized sorted-flow agreement match exact NPE. The rank
    correlation uses exact observed midranks and pairwise sigmoid predicted
    ranks at temperature ``softness * observed mean``. This is scale invariant
    under common positive rescaling, but finite softness need not give zero at
    a perfect match. Cost is quadratic in each basin's selected time count.
    See docs/losses.md for the surrogate's limits and failure policy.
    """
    if (
        prediction.ndim != 2
        or prediction.shape != target.shape
        or selected.shape != target.shape
        or selected.dtype != torch.bool
    ):
        raise ValueError("expected matching [time, basin] tensors and boolean mask")
    if not math.isfinite(softness) or softness <= 0:
        raise ValueError("softness must be finite and positive")
    selected = selected & ~torch.isnan(target)
    if bool((~torch.isfinite(target[selected]) | (target[selected] < 0)).any()):
        raise ValueError("selected observations must be nonnegative finite values or NaN")
    if bool((~torch.isfinite(prediction[selected]) | (prediction[selected] < 0)).any()):
        raise ValueError("selected predictions must be finite and nonnegative")
    losses: list[Tensor] = []
    for basin in range(target.shape[1]):
        mask = selected[:, basin]
        observed, predicted = target[mask, basin], prediction[mask, basin]
        if observed.numel() < 2 or bool(observed.max() == observed.min()):
            continue
        if bool(predicted.max() == predicted.min()):
            raise ValueError("soft NPE needs nonconstant predictions with positive mean")
        observed_ranks = _ranks(observed)
        observed_ranks = observed_ranks - observed_ranks.mean()
        temperature = softness * observed.mean()
        predicted_ranks = torch.sigmoid((predicted[:, None] - predicted[None, :]) / temperature).sum(1)
        predicted_ranks = predicted_ranks - predicted_ranks.mean()
        denominator = observed_ranks.norm() * predicted_ranks.norm()
        if not bool(torch.isfinite(denominator)) or bool(denominator <= 0):
            raise ValueError("undefined soft rank correlation; check discharge scale and softness")
        correlation = (observed_ranks @ predicted_ranks / denominator).clamp(-1, 1)
        beta = predicted.mean() / observed.mean()
        alpha_distance = (
            0.5 * (predicted.sort().values / predicted.sum() - observed.sort().values / observed.sum()).abs().sum()
        )
        losses.append(torch.stack((beta - 1, alpha_distance, 1 - correlation)).norm())
    if not losses:
        raise ValueError("soft NPE needs a basin with at least two nonconstant observations")
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite soft NPE loss")
    return loss
