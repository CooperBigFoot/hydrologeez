"""Supplied-batch likelihood training and checkpoint persistence."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .model import ConditionalFlow, ModelConfig


@dataclass(frozen=True)
class Batch:
    parameters: Tensor  # [example, parameter], physical units
    attributes: Tensor  # [example, attribute], caller-preprocessed


class Training:
    """Adam on mean negative conditional log likelihood; caller owns epochs."""

    def __init__(self, model: ConditionalFlow, *, learning_rate: float = 0.001) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss(self, batch: Batch) -> Tensor:
        return -self.model.log_prob(batch.parameters, batch.attributes).mean()

    def train_step(self, batch: Batch) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(batch)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("Missing or nonfinite gradients")
        self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self, batch: Batch) -> float:
        self.model.eval()
        return float(self.loss(batch))

    def save(self, path: str | Path) -> None:
        """Save configuration, physical mapping, weights and Adam; not RNG state.

        Caller-owned attribute preprocessing and pool/order decisions must be
        persisted by the experiment. Random samples are not replayed on load.
        """
        torch.save(
            {
                "version": 1,
                "config": asdict(self.model.config),
                "model": self.model.state_dict(),
                "learning_rate": self.learning_rate,
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "Training":
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved["version"] != 1:
            raise ValueError("Unsupported checkpoint version")
        model = ConditionalFlow(ModelConfig(**saved["config"]), dtype=saved["model"]["lower"].dtype).to(device=device)
        model.load_state_dict(saved["model"])
        training = cls(model, learning_rate=saved["learning_rate"])
        training.optimizer.load_state_dict(saved["optimizer"])
        return training
