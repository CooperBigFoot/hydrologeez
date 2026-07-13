"""Torch gradient calibration and transitional derivative-free calibration."""

from hydrologeez.calibration.adapter import (
    ParamSpec,
    array_to_model,
    array_to_parameters,
    bounds_array,
    flat_to_model,
    model_to_flat,
    params_to_array,
)
from hydrologeez.calibration.api import (
    calibrate_evolutionary,
    calibrate_nsga2,
    make_bounded_operators,
)
from hydrologeez.calibration.evolutionary import (
    make_batch_evaluator,
    make_objective,
)
from hydrologeez.calibration.gradient import calibrate_gradient

__all__ = [
    "ParamSpec",
    "bounds_array",
    "model_to_flat",
    "flat_to_model",
    "params_to_array",
    "array_to_model",
    "array_to_parameters",
    "make_batch_evaluator",
    "make_objective",
    "calibrate_gradient",
    "make_bounded_operators",
    "calibrate_evolutionary",
    "calibrate_nsga2",
]
