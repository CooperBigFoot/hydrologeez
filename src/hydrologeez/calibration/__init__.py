"""Seeded HBV ensemble calibration using reusable daily inputs."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate
from hydrologeez.metrics import OBJECTIVES, evaluate, objective_loss

from ._evolution import evolve

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Period:
    """ISO date interval [start, stop), scored on Gregorian calendar years."""

    start: str
    stop: str

    def ordinals(self) -> tuple[int, int]:
        first, last = date.fromisoformat(self.start).toordinal(), date.fromisoformat(self.stop).toordinal()
        if first >= last:
            raise ValueError("Period must be nonempty")
        return first, last


def scoring_mask(
    dates: NDArray[np.int64], observed: Array, period: Period, *, minimum_days: int | None = None
) -> tuple[NDArray[np.bool_], dict[int, int]]:
    """Count nonmissing days per calendar year within this split, including partial years.

    Coverage counts precede filtering. A rejected year has no scored days;
    accepted years still exclude missing observations. No QC flags are changed.
    """
    first, last = period.ordinals()
    if dates.ndim != 1 or observed.shape != dates.shape or not len(dates) or np.any(np.diff(dates) != 1):
        raise ValueError("Expected aligned, contiguous daily dates and observations")
    if first < dates[0] or last > dates[-1] + 1:
        raise ValueError("Scoring period exceeds inputs")
    if minimum_days is not None and minimum_days < 0:
        raise ValueError("minimum_days must be nonnegative")
    if np.any(np.isinf(observed)) or np.any(observed < 0):
        raise ValueError("Observations must be nonnegative finite values or NaN")
    years = np.array([date.fromordinal(int(d)).year for d in dates])
    within = (dates >= first) & (dates < last)
    mask = within & np.isfinite(observed)
    coverage = {int(year): int(np.count_nonzero(mask & (years == year))) for year in np.unique(years[within])}
    if minimum_days is not None:
        for year, count in coverage.items():
            if count < minimum_days:
                mask[years == year] = False
    return mask, coverage


# Runtime parameter order, not the presentation order of the source table.
PARAMETER_NAMES = ("TT", "CFMAX", "CFR", "CWH", "SCF", "FC", "BETA", "LP", "PERC", "Alpha", "K1", "K2", "MAXBAS")
SEARCH_BOUNDS: Array = np.array(
    [
        [-2, 2.5],
        [0.5, 10],
        [0, 0.1],
        [0, 0.2],
        [0.5, 1.2],
        [100, 1000],
        [1, 5],
        [0.3, 1],
        [0, 4],
        [0, 1],
        [0.01, 0.2],
        [0.001, 0.1],
        [1, 5],
    ],
    dtype=np.float64,
)
SEARCH_BOUNDS.setflags(write=False)


class CandidateEvaluator:
    """Reuse one basin's observations and forcing; discharge is bounded by batch size.

    Each candidate starts with empty stores on the first input day, with its own
    parameters throughout warm-up and both splits. Parameters are shared across
    all zones of the selected basin. Simulation continues through unscored gaps.
    """

    def __init__(
        self,
        forcing: Forcing,
        dates: NDArray[np.int64],
        observed: Array,
        basin_index: int,
        training: Period,
        *,
        validation: Period | None = None,
        minimum_days: int | None = None,
    ) -> None:
        if not 0 <= basin_index < len(forcing.zone_offsets) - 1 or len(dates) != len(forcing.values):
            raise ValueError("Basin and dates must match forcing")
        if np.any(np.isinf(observed)) or np.any(observed[np.isfinite(observed)] < 0):
            raise ValueError("Observed discharge must be nonnegative or missing")
        self.forcing, self.dates, self.observed, self.basin_index = forcing, dates, observed, basin_index
        self.masks: dict[str, NDArray[np.bool_]] = {}
        self.coverage: dict[str, dict[int, int]] = {}
        if validation is not None and training.ordinals()[1] > validation.ordinals()[0]:
            raise ValueError("Validation must follow training without overlap")
        for name, period in (("training", training), ("validation", validation)):
            if period is not None:
                mask, coverage = scoring_mask(dates, observed, period, minimum_days=minimum_days)
                if not mask.any():
                    raise ValueError(f"No observations remain in {name}")
                self.masks[name], self.coverage[name] = mask, coverage

    def inputs(self, candidates: Array) -> tuple[Parameters, Memory]:
        """Resolve [candidate,13] parameters and independent empty initial stores."""
        if candidates.ndim != 2 or candidates.shape[1] != len(PARAMETER_NAMES) or not len(candidates):
            raise ValueError("Expected nonempty candidates with 13 parameters")
        n = len(candidates)
        zones = int(np.diff(self.forcing.zone_offsets)[self.basin_index])
        parameters = Parameters(
            np.full(n, self.basin_index, dtype=np.int64),
            np.repeat(candidates[:, :8], zones, axis=0),
            candidates[:, 8:12].copy(),
            candidates[:, 12].copy(),
        )
        initial = Memory(
            np.zeros((n * zones, 3)), np.zeros((n, 2)), np.zeros(int((np.ceil(candidates[:, 12]) - 1).sum()))
        )
        return parameters, initial

    def discharge(self, candidates: Array) -> Array:
        """Regenerate full input-period hydrographs for explicitly supplied candidates."""
        parameters, initial = self.inputs(candidates)
        return simulate(self.forcing, parameters, initial, history=False).outlet

    def scores(
        self,
        candidates: Array,
        *,
        batch_size: int,
        periods: tuple[str, ...] = ("training",),
    ) -> list[dict[str, dict[str, float]]]:
        """Score complete periods, never average window-level rank/FDC scores."""
        if batch_size <= 0 or any(period not in self.masks for period in periods):
            raise ValueError("Positive batch size and configured scoring periods required")
        scores: list[dict[str, dict[str, float]]] = []
        for start in range(0, len(candidates), batch_size):
            discharge = self.discharge(candidates[start : start + batch_size])
            for column in discharge.T:
                scores.append(
                    {
                        period: evaluate(
                            self.observed[self.masks[period]],
                            column[self.masks[period]],
                            dates=(self.dates[self.masks[period]] - date(1970, 1, 1).toordinal()).astype(
                                "datetime64[D]"
                            ),
                        )
                        for period in periods
                    }
                )
        return scores


class JointEvaluator:
    """Explicit basin IDs mapped to independent evaluators; only parameters are shared.

    Simulate one basin at a time in candidate batches. Diagnostics retain full-period
    raw scores; losses use fixed masked training means for RMSE and MSE.
    """

    def __init__(self, basins: dict[str, CandidateEvaluator], *, weights: dict[str, float] | None = None) -> None:
        if not basins:
            raise ValueError("Select at least one basin")
        self.basins = dict(basins)
        weights = dict.fromkeys(basins, 1.0) if weights is None else weights
        values = np.array(list(weights.values()), dtype=np.float64)
        if weights.keys() != basins.keys() or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Supply positive finite weights for every selected basin")
        # Scale before summing to avoid overflow for large finite weights.
        scaled = {name: weight / float(values.max()) for name, weight in weights.items()}
        total = sum(scaled.values())
        self.weights = {name: weight / total for name, weight in scaled.items()}
        self.training_means = {
            name: float(basin.observed[basin.masks["training"]].mean()) for name, basin in basins.items()
        }

    def scores(
        self, candidates: Array, *, batch_size: int, periods: tuple[str, ...] = ("training",)
    ) -> list[dict[str, dict[str, dict[str, float]]]]:
        """Raw diagnostics indexed by candidate, basin ID, period, and metric."""
        result: list[dict[str, dict[str, dict[str, float]]]] = [{} for _ in candidates]
        for name, basin in self.basins.items():
            for record, scores in zip(
                result, basin.scores(candidates, batch_size=batch_size, periods=periods), strict=True
            ):
                record[name] = scores
        return result

    def losses(
        self,
        scores: list[dict[str, dict[str, dict[str, float]]]],
        objectives: tuple[str, ...],
        *,
        period: str = "training",
    ) -> Array:
        """Weighted mean of converted basin losses; no basin is omitted."""
        result = np.zeros((len(scores), len(objectives)))
        for row, record in enumerate(scores):
            for column, metric in enumerate(objectives):
                for name, weight in self.weights.items():
                    loss = objective_loss(metric, record[name][period][metric])
                    if metric in ("rmse", "mse"):
                        denominator = self.training_means[name] ** (2 if metric == "mse" else 1)
                        loss = loss / denominator if denominator > 0 else float("inf")
                    result[row, column] += weight * loss
        return result


@dataclass(frozen=True)
class Solution:
    """One retained model, with run identity and raw (not minimized) scores."""

    run: int
    seed: int
    parameters: Array
    objectives: tuple[float, ...]
    scores: dict[str, dict[str, float]]


@dataclass(frozen=True)
class Run:
    run: int
    seed: int
    evaluations: int
    retained: int


@dataclass(frozen=True)
class Ensemble:
    solutions: tuple[Solution, ...]
    runs: tuple[Run, ...]
    objectives: tuple[str, ...]
    coverage: dict[str, dict[int, int]]


def _validate_search(
    seeds: list[int],
    objectives: tuple[str, ...],
    population_size: int,
    generations: int,
    retain: int,
    batch_size: int,
    bounds: Array,
) -> None:
    if not objectives or any(name not in OBJECTIVES for name in objectives):
        raise ValueError("Choose supported objective metrics")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Supply unique nonnegative independent run seeds")
    if retain < 1 or batch_size < 1 or population_size < 2 or population_size % 2 or generations < 0:
        raise ValueError("Positive retention/batch, even population >=2 and nonnegative generations required")
    if bounds.shape != (13, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Bounds must have shape (13,2) with finite increasing intervals")


def _search(
    evaluate_batch: Callable[[Array], Array],
    seeds: list[int],
    objectives: tuple[str, ...],
    population_size: int,
    generations: int,
    retain: int,
    bounds: Array,
) -> Iterator[tuple[Run, Array]]:
    for run, seed in enumerate(seeds):
        population = evolve(evaluate_batch, bounds, population_size, generations, seed)
        if len(objectives) == 1:
            indices = [int(np.argmin(population.losses[:, 0]))]
        else:
            indices = sorted(
                np.flatnonzero(population.rank == 0).tolist(), key=lambda i: -float(population.crowding[i])
            )
        evaluations = population_size * (generations + 1)
        selected: list[int] = []
        seen: set[tuple[float, ...]] = set()
        for index in indices:
            key = tuple(float(v) for v in population.parameters[index])
            if np.isfinite(population.losses[index]).all() and key not in seen:
                selected.append(index)
                seen.add(key)
                if len(selected) == retain:
                    break
        yield Run(run, seed, evaluations, len(selected)), population.parameters[selected]


def calibrate(
    evaluator: CandidateEvaluator,
    *,
    seeds: list[int],
    objectives: tuple[str, ...],
    population_size: int = 100,
    generations: int = 100,
    retain: int = 1,
    batch_size: int = 64,
    report: tuple[str, ...] = ("training",),
    bounds: Array = SEARCH_BOUNDS,
) -> Ensemble:
    """Independent seeded GA (one objective) or NSGA-II runs, no global front filter.

    Uniform original-scale initialization; bounded SBX (eta=15) and polynomial
    mutation (eta=20, per-parameter probability=1/13). Retain unique parameter
    vectors from each final front, ordered by descending crowding distance then
    stable population order. Single-objective runs retain only their best model.
    Invalid-objective candidates cannot be retained; an all-invalid run returns
    zero models rather than padding an ensemble. Every selected diagnostic is
    reported independently of optimizer objectives (currently the full catalogue).
    """
    _validate_search(seeds, objectives, population_size, generations, retain, batch_size, bounds)
    if not report or any(period not in evaluator.masks for period in report):
        raise ValueError("Choose configured reporting periods")
    lower, upper = bounds[:, 0], bounds[:, 1]
    # Validate physical domains before entering optimization, including custom bounds.
    for endpoint in (lower, upper):
        parameters, initial = evaluator.inputs(endpoint[None, :])
        simulate(evaluator.forcing, parameters, initial, stop=0)

    def evaluate_batch(candidates: Array) -> Array:
        scores = evaluator.scores(candidates, batch_size=batch_size)
        return np.array([[objective_loss(name, score["training"][name]) for name in objectives] for score in scores])

    solutions: list[Solution] = []
    runs: list[Run] = []
    for run_info, candidates in _search(
        evaluate_batch, seeds, objectives, population_size, generations, retain, bounds
    ):
        if len(candidates):
            reports = evaluator.scores(
                candidates, batch_size=batch_size, periods=tuple(dict.fromkeys(("training", *report)))
            )
            for candidate, scores in zip(candidates, reports, strict=True):
                objective_values = tuple(scores["training"][name] for name in objectives)
                solutions.append(
                    Solution(
                        run_info.run,
                        run_info.seed,
                        np.asarray(candidate).copy(),
                        objective_values,
                        {p: scores[p] for p in report},
                    )
                )
        runs.append(run_info)
    return Ensemble(tuple(solutions), tuple(runs), objectives, evaluator.coverage)


@dataclass(frozen=True)
class JointSolution:
    """Shared parameter vector identified by run, seed, and its parameter values."""

    run: int
    seed: int
    parameters: Array
    losses: tuple[float, ...]
    scores: dict[str, dict[str, dict[str, float]]]


@dataclass(frozen=True)
class JointEnsemble:
    solutions: tuple[JointSolution, ...]
    runs: tuple[Run, ...]
    objectives: tuple[str, ...]
    weights: dict[str, float]
    training_means: dict[str, float]
    coverage: dict[str, dict[str, dict[int, int]]]


def calibrate_joint(
    evaluator: JointEvaluator,
    *,
    seeds: list[int],
    objectives: tuple[str, ...],
    population_size: int = 100,
    generations: int = 100,
    retain: int = 1,
    batch_size: int = 64,
    report: tuple[str, ...] = ("training",),
    bounds: Array = SEARCH_BOUNDS,
) -> JointEnsemble:
    """Seeded GA/NSGA-II with separate weighted-mean basin-loss objectives.

    Search settings and retention match ``calibrate``. Every basin must have valid
    training losses for every objective. Undefined diagnostic-only scores remain NaN.
    """
    _validate_search(seeds, objectives, population_size, generations, retain, batch_size, bounds)
    for name, basin in evaluator.basins.items():
        if not report or any(period not in basin.masks for period in report):
            raise ValueError(f"Choose configured reporting periods for {name}")
        # Perfect agreement exposes observation-only undefined metrics, including
        # insufficient peak windows and constant/too-short observation records.
        observed_scores = basin.observed[basin.masks["training"]]
        dates = (basin.dates[basin.masks["training"]] - date(1970, 1, 1).toordinal()).astype("datetime64[D]")
        reference = evaluate(observed_scores, observed_scores, dates=dates)
        for metric in objectives:
            if not np.isfinite(reference[metric]) or (
                metric in ("rmse", "mse") and evaluator.training_means[name] <= 0
            ):
                raise ValueError(f"Basin {name} has incompatible training observations for {metric}")
        for endpoint in (bounds[:, 0], bounds[:, 1]):
            parameters, initial = basin.inputs(endpoint[None, :])
            simulate(basin.forcing, parameters, initial, stop=0)

    def evaluate_batch(candidates: Array) -> Array:
        return evaluator.losses(evaluator.scores(candidates, batch_size=batch_size), objectives)

    solutions: list[JointSolution] = []
    runs: list[Run] = []
    for run, candidates in _search(evaluate_batch, seeds, objectives, population_size, generations, retain, bounds):
        if len(candidates):
            records = evaluator.scores(
                candidates, batch_size=batch_size, periods=tuple(dict.fromkeys(("training", *report)))
            )
            losses = evaluator.losses(records, objectives)
            for candidate, record, loss in zip(candidates, records, losses, strict=True):
                solutions.append(
                    JointSolution(
                        run.run,
                        run.seed,
                        np.asarray(candidate).copy(),
                        tuple(float(v) for v in np.asarray(loss)),
                        {name: {period: scores[period] for period in report} for name, scores in record.items()},
                    )
                )
        runs.append(run)
    return JointEnsemble(
        tuple(solutions),
        tuple(runs),
        objectives,
        dict(evaluator.weights),
        dict(evaluator.training_means),
        {name: basin.coverage for name, basin in evaluator.basins.items()},
    )
