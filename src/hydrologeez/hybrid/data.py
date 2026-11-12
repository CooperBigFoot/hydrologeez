"""Feature normalization and separate missing, hidden and padding masks."""

from dataclasses import dataclass

import torch
from torch import Tensor

from hydrologeez.losses import masked_mse as masked_mse
from hydrologeez.preprocessing import standardization


@dataclass(frozen=True)
class Normalization:
    mean: Tensor
    scale: Tensor
    features: tuple[str, ...]

    @classmethod
    def fit(cls, values: Tensor, padding: Tensor, features: tuple[str, ...]) -> "Normalization":
        """Fit only on caller-authorized meteorological pool, never implicitly."""
        if (
            values.ndim != 3
            or padding.shape != values.shape[:2]
            or len(features) != values.shape[-1]
            or len(set(features)) != len(features)
        ):
            raise ValueError("expected unique features and [basin, day, feature] values")
        valid = torch.isfinite(values) & ~padding[..., None]
        mean, scale = standardization(values.detach().cpu().double().numpy(), valid.cpu().numpy(), (0, 1))
        return cls(
            torch.as_tensor(mean, dtype=values.dtype, device=values.device),
            torch.as_tensor(scale, dtype=values.dtype, device=values.device),
            features,
        )

    @classmethod
    def identity(cls, features: tuple[str, ...]) -> "Normalization":
        """For inputs already in final model-input space, such as a PreparedDataset."""
        return cls(torch.zeros(len(features)), torch.ones(len(features)), features)


@dataclass(frozen=True)
class Prepared:
    inputs: Tensor
    normalized: Tensor
    targets: Tensor


def prepare(values: Tensor, padding: Tensor, hidden: Tensor, normalization: Normalization) -> Prepared:
    """Channels: normalized visible values, observed-visible flags, hidden flags.

    Artificially hidden values never enter inputs. Missing/padded values never
    become targets. No per-window fitting or target-derived imputation occurs.
    """
    if (
        values.ndim != 3
        or padding.shape != values.shape[:2]
        or hidden.shape != values.shape
        or values.shape[-1] != len(normalization.features)
    ):
        raise ValueError("inconsistent meteorological input shapes")
    if padding.dtype != torch.bool or hidden.dtype != torch.bool:
        raise ValueError("masks must be boolean")
    valid = torch.isfinite(values) & ~padding[..., None]
    targets = hidden & valid
    visible = valid & ~hidden
    normalized = torch.where(valid, (values - normalization.mean) / normalization.scale, 0.0)
    inputs = torch.cat((torch.where(visible, normalized, 0.0), visible.to(values.dtype), targets.to(values.dtype)), -1)
    return Prepared(inputs, normalized, targets)
