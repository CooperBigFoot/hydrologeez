"""hydrologeez: differentiable conceptual hydrological models in JAX.

This package owns its package-level exports here. Importing it enforces JAX
float64 precision at import time (loud, fail-fast; no silent global x64 flip).
"""

from hydrologeez.precision import enforce_float64

# Loud, fail-fast float64 enforcement BEFORE any model import. NO silent flip.
enforce_float64()

from hydrologeez.observation import default_streamflow_observation  # noqa: E402
from hydrologeez.ssm import StateSpaceModel  # noqa: E402

__version__ = "0.1.9"

__all__ = [
    "StateSpaceModel",
    "default_streamflow_observation",
    "__version__",
]
