"""Real-valued GA and NSGA-II with bounded population-level evaluation.

Bounded SBX/polynomial mutation adapted from ctrl-freak (Nicolas Lazaro, 2026).
See ctrl-freak-LICENSE.txt. No worker pools or optimizer framework are required.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Population:
    parameters: Array
    losses: Array
    rank: NDArray[np.int64]
    crowding: Array


def rank_and_crowding(losses: Array) -> tuple[NDArray[np.int64], Array]:
    """Pareto fronts and normalized crowding; any invalid objective ranks last.

    Constant objectives add no crowding. Stable sorting resolves ties. The
    dominance matrix is O(population²), independent of simulation duration.
    """
    valid = np.isfinite(losses).all(axis=1)
    indices = np.flatnonzero(valid)
    values = losses[valid]
    dominates = np.all(values[:, None] <= values[None, :], axis=2) & np.any(values[:, None] < values[None, :], axis=2)
    counts = dominates.sum(axis=0)
    rank = np.full(len(losses), len(losses), dtype=np.int64)
    crowding = np.zeros(len(losses))
    remaining = np.arange(len(values))
    level = 0
    while len(remaining):
        front = remaining[counts[remaining] == 0]
        rank[indices[front]] = level
        distance = np.zeros(len(front))
        for column in values[front].T:
            order = np.argsort(column, kind="stable")
            width = column[order[-1]] - column[order[0]]
            if width > 0:
                distance[order[[0, -1]]] = np.inf
                distance[order[1:-1]] += (column[order[2:]] - column[order[:-2]]) / width
        crowding[indices[front]] = distance
        remaining = remaining[counts[remaining] != 0]
        counts -= dominates[front].sum(axis=0)
        level += 1
    rank[~valid] = level
    return rank, crowding


def _crossover(p1: Array, p2: Array, lower: Array, upper: Array, rng: np.random.Generator) -> Array:
    eta = 15.0
    y1, y2 = np.minimum(p1, p2), np.maximum(p1, p2)
    eligible = np.abs(p1 - p2) > 1e-14
    delta = np.where(eligible, y2 - y1, 1.0)
    random = rng.random(len(p1))

    def beta_q(beta: Array) -> Array:
        alpha = 2.0 - beta ** (-(eta + 1))
        return np.where(
            random <= 1 / alpha, (random * alpha) ** (1 / (eta + 1)), (1 / (2 - random * alpha)) ** (1 / (eta + 1))
        )

    c1 = 0.5 * (y1 + y2 - beta_q(1 + 2 * (y1 - lower) / delta) * delta)
    c2 = 0.5 * (y1 + y2 + beta_q(1 + 2 * (upper - y2) / delta) * delta)
    child = np.where(rng.random(len(p1)) < 0.5, c1, c2)
    return np.clip(np.where(eligible, child, p1), lower, upper)


def _mutate(x: Array, lower: Array, upper: Array, rng: np.random.Generator) -> Array:
    eta = 20.0
    mask = rng.random(len(x)) < 1 / len(x)
    width = upper - lower
    left, right = (x - lower) / width, (upper - x) / width
    u = rng.random(len(x))
    dl = (2 * u + (1 - 2 * u) * (1 - left) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    dr = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - right) ** (eta + 1)) ** (1 / (eta + 1))
    return np.clip(x + np.where(mask, np.where(u < 0.5, dl, dr) * width, 0), lower, upper)


def evolve(evaluate: Callable[[Array], Array], bounds: Array, size: int, generations: int, seed: int) -> Population:
    """Elitist GA for one loss; NSGA-II otherwise. Four independent RNG streams."""
    init, select, crossover, mutate = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    lower, upper = bounds[:, 0], bounds[:, 1]
    parameters = init.uniform(lower, upper, size=(size, len(bounds)))
    losses = evaluate(parameters)
    for _ in range(generations):
        rank, crowding = rank_and_crowding(losses)
        pairs = select.integers(size, size=(size * 2, 2))
        winners = np.where(
            (rank[pairs[:, 0]] < rank[pairs[:, 1]])
            | ((rank[pairs[:, 0]] == rank[pairs[:, 1]]) & (crowding[pairs[:, 0]] >= crowding[pairs[:, 1]])),
            pairs[:, 0],
            pairs[:, 1],
        )
        children = np.stack(
            [
                _mutate(_crossover(parameters[a], parameters[b], lower, upper, crossover), lower, upper, mutate)
                for a, b in winners.reshape(size, 2)
            ]
        )
        parameters = np.concatenate((parameters, children))
        losses = np.concatenate((losses, evaluate(children)))
        if losses.shape[1] == 1:
            order = np.argsort(losses[:, 0], kind="stable")[:size]
        else:
            rank, crowding = rank_and_crowding(losses)
            order = np.lexsort((-crowding, rank))[:size]
        parameters, losses = parameters[order], losses[order]
    rank, crowding = rank_and_crowding(losses)
    return Population(parameters, losses, rank, crowding)
