"""Conditional MAF density with an explicit physical parameter domain."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from zuko.flows import MAF


@dataclass(frozen=True)
class ModelConfig:
    parameters: tuple[str, ...]
    units: tuple[str, ...]
    attributes: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    hidden: tuple[int, ...] = (64, 64)
    transforms: int = 3


class ConditionalFlow(nn.Module):
    """Zuko MAF in logit coordinates; no learned attribute preprocessing.

    Densities require strictly interior physical values. Samples use the inverse
    mapping, not clipping; floating-point rounding can reach an endpoint.
    """

    lower: Tensor
    upper: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        n = len(config.parameters)
        if (
            not n
            or len(set(config.parameters)) != n
            or not config.attributes
            or len(set(config.attributes)) != len(config.attributes)
        ):
            raise ValueError("Select unique nonempty parameter and attribute names")
        if len(config.units) != n or len(config.lower) != n or len(config.upper) != n:
            raise ValueError("Units and bounds must match parameter order")
        if not config.hidden or min(config.hidden) < 1 or config.transforms < 1:
            raise ValueError("Hidden widths and transforms must be positive")
        self.config = config
        self.register_buffer("lower", torch.tensor(config.lower))
        self.register_buffer("upper", torch.tensor(config.upper))
        if not bool(
            torch.isfinite(self.lower).all() & torch.isfinite(self.upper).all() & (self.upper > self.lower).all()
        ):
            raise ValueError("Bounds must be finite and strictly increasing")
        self.flow = MAF(n, len(config.attributes), transforms=config.transforms, hidden_features=config.hidden)

    def _check_attributes(self, attributes: Tensor) -> None:
        if (
            attributes.ndim != 2
            or attributes.shape[0] == 0
            or attributes.shape[1] != len(self.config.attributes)
            or not bool(torch.isfinite(attributes).all())
        ):
            raise ValueError("Attributes must be finite [catchment, attribute] in configured order")

    def log_prob(self, parameters: Tensor, attributes: Tensor) -> Tensor:
        """Physical log density [example] for paired rank-two inputs."""
        self._check_attributes(attributes)
        if parameters.shape != (len(attributes), len(self.config.parameters)):
            raise ValueError("Parameters must be [example, parameter] paired with attributes")
        if not bool((torch.isfinite(parameters) & (parameters > self.lower) & (parameters < self.upper)).all()):
            raise ValueError("Parameters must be finite and strictly inside bounds; endpoints are not supported")
        width = self.upper - self.lower
        log_lower_distance = torch.log(parameters - self.lower)
        log_upper_distance = torch.log(self.upper - parameters)
        z = log_lower_distance - log_upper_distance
        log_jacobian = (log_lower_distance + log_upper_distance - torch.log(width)).sum(-1)
        return self.flow(attributes).log_prob(z) - log_jacobian

    @torch.no_grad()
    def sample(self, attributes: Tensor, count: int) -> Tensor:
        """Return physical parameters [sample, catchment, parameter]."""
        self._check_attributes(attributes)
        if count < 1:
            raise ValueError("Sample count must be positive")
        z = self.flow(attributes).sample((count,))
        return self.lower + (self.upper - self.lower) * z.sigmoid()
