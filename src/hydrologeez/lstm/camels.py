"""Build standalone DischargeLSTM batches from resident CAMELS inputs."""

import torch
from torch import Tensor

from hydrologeez.hybrid.camels import CamelsInputs

from .model import ModelConfig
from .runtime import Batch


def batch_from_inputs(
    inputs: CamelsInputs,
    config: ModelConfig,
    *,
    basins: slice | list[int] = slice(None),
    window: slice = slice(None),
    evaluation_mask: Tensor | None = None,
    device: str = "cpu",
) -> Batch:
    """Slice resident CAMELS meteorology, caller-scaled attributes and discharge."""
    if config.features != inputs.features or config.attributes != inputs.attribute_names:
        raise ValueError("Model feature/attribute order differs from selected inputs")
    meteorology = inputs.meteorology[basins, window].to(device)
    attributes = inputs.attributes[basins].to(device)
    if meteorology.shape[0] == 0 or meteorology.shape[1] == 0:
        raise ValueError("Batch must contain basins and days")
    valid = None if inputs.discharge_valid is None else inputs.discharge_valid[window, basins].to(device)
    if evaluation_mask is not None:
        if valid is None or evaluation_mask.shape != valid.shape or evaluation_mask.dtype != torch.bool:
            raise ValueError("Evaluation mask needs matching authorized discharge")
        valid = valid & evaluation_mask.to(device)
    return Batch(
        meteorology,
        attributes,
        torch.zeros(meteorology.shape[:2], dtype=torch.bool, device=device),
        None if inputs.discharge is None else inputs.discharge[window, basins].to(device),
        valid,
    )
