"""Bounded float64 discrete Newton probe, isolated soil with supplied daily I and PET."""

import json
import platform
import resource
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit

from .model import Array


@njit(cache=True)
def sequential(inputs: Array, pet: Array, initial: Array, fc: float, beta: float, lp: float) -> Array:
    """Compiled sequential baseline; output end-of-day M, recharge R, evaporation A."""
    output = np.empty((*inputs.shape, 3))
    for b in range(inputs.shape[1]):
        moisture = initial[b]
        for t in range(inputs.shape[0]):
            recharge = inputs[t, b] * (moisture / fc) ** beta
            before = moisture + inputs[t, b] - recharge
            overflow = max(before - fc, 0.0)
            before -= overflow
            evaporation = min(before, pet[t, b] * min(before / (lp * fc), 1.0))
            moisture = before - evaporation
            output[t, b] = moisture, recharge + overflow, evaporation
    return output


def affine_scan(a: Array, b: Array, initial: Array) -> Array:
    """Inclusive Hillis–Steele scan of maps x -> a*x+b; O(T log T) work."""
    a, b = a.copy(), b.copy()
    stride = 1
    while stride < len(a):
        b[stride:] = b[stride:] + a[stride:] * b[:-stride]
        a[stride:] = a[stride:] * a[:-stride]
        stride *= 2
    return a * initial + b


def soil(m: Array, inputs: Array, pet: Array, fc: float, beta: float, lp: float) -> Array:
    recharge = inputs * (m / fc) ** beta
    before = m + inputs - recharge
    overflow = np.maximum(before - fc, 0.0)
    before -= overflow
    evaporation = np.minimum(before, pet * np.minimum(before / (lp * fc), 1.0))
    return np.stack((before - evaporation, recharge + overflow, evaporation), axis=-1)


def derivative(m: Array, inputs: Array, pet: Array, fc: float, beta: float, lp: float) -> Array:
    # Zero input removes the fractional-power term exactly, including at dry M.
    slope = np.zeros_like(m)
    active = inputs != 0
    slope[active] = inputs[active] * beta / fc * (m[active] / fc) ** (beta - 1)
    before = m + inputs - inputs * (m / fc) ** beta
    d = np.where(before < fc, 1 - slope, 0.0)
    before = np.minimum(before, fc)
    demand = pet * np.minimum(before / (lp * fc), 1.0)
    evaporation_slope = np.where(before < lp * fc, pet / (lp * fc), 0.0)
    return d * np.where(before <= demand, 0.0, 1 - evaporation_slope)


@dataclass
class Result:
    values: Array
    iterations: int
    status: str
    residual: float
    invalid_states: int


def solve(
    inputs: Array,
    pet: Array,
    initial: Array,
    fc: float,
    beta: float,
    lp: float,
    *,
    max_iterations: int = 32,
    tolerance: float = 1e-10,
) -> Result:
    """Repeat initial memory as uninformed guess; reject, never clip, invalid trials."""
    trajectory = np.broadcast_to(initial, inputs.shape).copy()
    invalid_states = 0
    status = "iteration_cap"
    iterations = 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for iteration in range(max_iterations + 1):
            previous = np.concatenate((initial[None, :], trajectory[:-1]))
            values = soil(previous, inputs, pet, fc, beta, lp)
            residual = float(np.max(np.abs(trajectory - values[..., 0])))
            if residual <= tolerance:
                status = "converged"
                break
            if iteration == max_iterations:
                break
            if beta < 1 and np.any((previous == 0) & (inputs > 0)):
                status = "singular_derivative"
                break
            a = derivative(previous, inputs, pet, fc, beta, lp)
            trial = affine_scan(a, values[..., 0] - a * previous, initial)
            iterations += 1
            invalid_states = int(np.count_nonzero(~np.isfinite(trial) | (trial < 0) | (trial > fc)))
            if invalid_states:
                status = "invalid_trial"
                break
            trajectory = trial
    # Report guessed states, but transfers evaluated at those states' predecessors.
    values[..., 0] = trajectory
    return Result(values, iterations, status, residual, invalid_states)


def main() -> None:
    """Reproducible small CPU-only feasibility matrix, not full-model throughput."""
    cases: list[dict[str, object]] = []
    tick = perf_counter()
    sequential(np.ones((2, 1)), np.ones((2, 1)), np.array([50.0]), 100.0, 1.0, 0.5)
    startup = perf_counter() - tick
    for days in (365, 3650):
        for batch in (1, 8):
            for name, beta, initial_value in (
                ("steady_nonlinear", 2.0, 50.0),
                ("wet", 2.0, 50.0),
                ("linear", 1.0, 50.0),
                ("fractional", 0.5, 50.0),
                ("dry_fractional", 0.5, 0.0),
                ("thresholds", 1.0, 0.0),
            ):
                tick = perf_counter()
                rng = np.random.default_rng(90210)
                inputs = rng.gamma(1.2, 5.0, (days, batch))
                inputs[np.arange(days) % 90 < 30] = 0.0
                inputs[30::90] = 150.0
                pet = np.broadcast_to((2.0 + np.sin(np.arange(days) / 30.0))[:, None], inputs.shape).copy()
                pet[60::90] = 100.0
                if name == "steady_nonlinear":
                    inputs[:] = 3.0
                    pet[:] = 2.0
                if name == "thresholds":
                    inputs[:] = np.resize(np.array([50.0, 0.0, 150.0, 0.0, 0.0]), days)[:, None]
                    pet[:] = np.resize(np.array([0.0, 2.0, 0.0, 100.0, 0.0]), days)[:, None]
                initial = np.full(batch, initial_value)
                preparation = perf_counter() - tick
                sequential_times: list[float] = []
                newton_times: list[float] = []
                for _ in range(5):
                    tick = perf_counter()
                    reference = sequential(inputs, pet, initial, 100.0, beta, 0.5)
                    sequential_times.append(perf_counter() - tick)
                    tick = perf_counter()
                    result = solve(inputs, pet, initial, 100.0, beta, 0.5)
                    newton_times.append(perf_counter() - tick)
                difference = np.abs(result.values - reference)
                cases.append(
                    {
                        "days": days,
                        "batch": batch,
                        "case": name,
                        "beta": beta,
                        "initial_mm": initial_value,
                        "status": result.status,
                        "iterations": result.iterations,
                        "recurrence_residual_mm": result.residual,
                        "invalid_trial_states": result.invalid_states,
                        "max_absolute_M_R_A_mm": difference.max(axis=(0, 1)).tolist(),
                        "max_relative_M_R_A": (difference / np.maximum(np.abs(reference), 1e-12))
                        .max(axis=(0, 1))
                        .tolist(),
                        "preparation_seconds": preparation,
                        "sequential_seconds": sequential_times,
                        "newton_seconds": newton_times,
                        "faithful": result.status == "converged" and bool(np.all(difference <= 1e-8)),
                    }
                )
    report = {
        "platform": platform.platform(),
        "precision": "float64",
        "seed": 90210,
        "fc_mm": 100.0,
        "lp": 0.5,
        "max_iterations": 32,
        "residual_tolerance_mm": 1e-10,
        "fidelity_tolerance_mm": 1e-8,
        "startup_seconds": startup,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cases": cases,
    }
    Path("experiments/hbv/evidence/temporal.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
