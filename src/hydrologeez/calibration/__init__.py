"""Dual calibration stacks for hydrologeez models: gradient (optax) and derivative-free (ctrl-freak)."""

from hydrologeez.calibration.adapter import (
    GR6J_SPEC,
    HBV_SPEC,
    PARAM_NAMES,
    ParamSpec,
    array_to_model,
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
    "PARAM_NAMES",
    "ParamSpec",
    "GR6J_SPEC",
    "HBV_SPEC",
    "bounds_array",
    "model_to_flat",
    "flat_to_model",
    "params_to_array",
    "array_to_model",
    "make_batch_evaluator",
    "make_objective",
    "calibrate_gradient",
    "make_bounded_operators",
    "calibrate_evolutionary",
    "calibrate_nsga2",
]
