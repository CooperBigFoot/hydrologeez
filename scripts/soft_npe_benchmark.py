"""Experimental CPU soft-NPE objective; not part of the public metric catalogue."""

import math

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.stats import rankdata

Array = NDArray[np.float64]


@njit(cache=True)
def sigmoid_ranks(predicted: Array, temperature: float) -> Array:
    """All pairs, with sigmoid(-x)=1-sigmoid(x); O(n) memory, no rank approximation."""
    ranks = np.full(len(predicted), 0.5)
    for i in range(len(predicted)):
        for j in range(i):
            delta = (predicted[i] - predicted[j]) / temperature
            exponential = math.exp(-abs(delta))
            value = 1 / (1 + exponential) if delta >= 0 else exponential / (1 + exponential)
            ranks[i] += value
            ranks[j] += 1 - value
    return ranks


class SoftNPE:
    """Cache one basin's already-masked observations; invalid candidates lose."""

    def __init__(self, observed: Array, softness: float) -> None:
        if (
            observed.ndim != 1
            or len(observed) < 2
            or not np.isfinite(observed).all()
            or np.any(observed < 0)
            or np.ptp(observed) == 0
        ):
            raise ValueError("Need finite nonnegative nonconstant masked observations")
        if not math.isfinite(softness) or softness <= 0:
            raise ValueError("Need finite positive softness")
        self.mean = float(observed.mean())
        self.temperature = softness * self.mean
        self.fdc = np.sort(observed) / observed.sum()
        ranks = rankdata(observed)
        self.ranks = ranks - ranks.mean()
        self.rank_norm = float(np.linalg.norm(self.ranks))

    def __call__(self, predicted: Array) -> float:
        if predicted.shape != self.ranks.shape:
            raise ValueError("Prediction must align with masked observations")
        if not np.isfinite(predicted).all() or np.any(predicted < 0) or np.ptp(predicted) == 0:
            return float("inf")
        ranks = sigmoid_ranks(predicted, self.temperature)
        ranks -= ranks.mean()
        denominator = self.rank_norm * float(np.linalg.norm(ranks))
        if not math.isfinite(denominator) or denominator <= 0:
            return float("inf")
        correlation = float(np.clip(self.ranks @ ranks / denominator, -1, 1))
        beta = predicted.mean() / self.mean
        fdc_distance = 0.5 * np.abs(np.sort(predicted) / predicted.sum() - self.fdc).sum()
        loss = float(np.sqrt((beta - 1) ** 2 + fdc_distance**2 + (1 - correlation) ** 2))
        return loss if math.isfinite(loss) else float("inf")
