"""Differentiable conceptual hydrological models and utilities."""

from hydrologeez import metrics
from hydrologeez.calibration import calibrate_evolutionary, calibrate_nsga2
from hydrologeez.hdx.loader import from_hdx
from hydrologeez.hdx.writer import to_hdx
from hydrologeez.metrics import kge, lognse, mae, nse, pbias, rmse
from hydrologeez.observation import default_streamflow_observation
from hydrologeez.precision import (
    REFERENCE_DEVICE,
    REFERENCE_DTYPE,
    TRAINING_DTYPE,
    reference_defaults,
    reference_tensor,
    training_defaults,
    training_tensor,
)
from hydrologeez.ssm import StateSpaceModel

__version__ = "0.1.11"

__all__ = [
    "StateSpaceModel",
    "default_streamflow_observation",
    "calibrate_evolutionary",
    "calibrate_nsga2",
    "from_hdx",
    "to_hdx",
    "metrics",
    "kge",
    "lognse",
    "mae",
    "nse",
    "pbias",
    "rmse",
    "REFERENCE_DEVICE",
    "REFERENCE_DTYPE",
    "TRAINING_DTYPE",
    "reference_defaults",
    "reference_tensor",
    "training_defaults",
    "training_tensor",
    "__version__",
]
