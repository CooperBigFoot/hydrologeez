"""Training, evaluation, inference and checkpoint persistence for DischargeLSTM."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from hydrologeez.hybrid.data import Normalization
from hydrologeez.losses import masked_mse, soft_npe_loss
from hydrologeez.preprocessing import Standardization, Transform

from .model import DischargeLSTM, ModelConfig


@dataclass(frozen=True)
class Batch:
    meteorology: Tensor  # [basin, day, feature], finite on non-padded days
    attributes: Tensor  # [basin, attribute], caller-scaled
    padding: Tensor  # [basin, day], true only for right padding
    discharge: Tensor | None = None  # [day, basin]
    discharge_valid: Tensor | None = None  # [day, basin], boolean


class Training:
    """Adam updates for standalone daily discharge prediction.

    Each step owns a complete supplied batch/window; no recurrent state crosses
    batches unless callers slice/warm-up explicitly. Checkpoints persist model
    configuration, feature normalization, weights, objective and Adam state.
    """

    def __init__(
        self,
        model: DischargeLSTM,
        *,
        learning_rate: float = 0.001,
        discharge_loss: Literal["mse", "soft_npe"] = "mse",
        softness: float = 0.1,
        discharge_transform: Transform | None = None,
    ) -> None:
        if discharge_loss not in ("mse", "soft_npe"):
            raise ValueError("unknown discharge loss")
        if not math.isfinite(softness) or softness <= 0:
            raise ValueError("softness must be finite and positive")
        if discharge_transform is not None:
            if discharge_loss != "mse":
                raise ValueError("discharge_transform requires mse discharge_loss")
            if any(isinstance(step, Standardization) and step.scope != "global" for step in discharge_transform.steps):
                raise ValueError("Runtime discharge_transform requires global standardization scope")
            if model.config.output_activation != "softplus":
                raise ValueError("discharge_transform requires softplus output_activation for physical discharge")
        self.model = model
        self.learning_rate = learning_rate
        self.discharge_loss = discharge_loss
        self.softness = softness
        self.discharge_transform = discharge_transform
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss(self, batch: Batch) -> Tensor:
        if batch.discharge is None or batch.discharge_valid is None:
            raise ValueError("discharge training needs observations and validity mask")
        prediction = self.model(batch.meteorology, batch.attributes, batch.padding)
        valid = batch.discharge_valid & ~batch.padding.T
        if self.discharge_loss == "soft_npe":
            return soft_npe_loss(prediction, batch.discharge, valid, softness=self.softness)
        valid = valid & torch.isfinite(batch.discharge)
        if not bool(valid.any()):
            raise ValueError("loss needs selected targets")
        if self.discharge_transform is not None:
            predicted_target = self.discharge_transform.forward_tensor(prediction[valid])
            observed_target = self.discharge_transform.forward_tensor(batch.discharge[valid])
            return (predicted_target - observed_target).square().mean()
        return masked_mse(prediction, batch.discharge, valid)

    def train_step(self, batch: Batch) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(batch)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite training loss")
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("missing or nonfinite gradients")
        self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self, batch: Batch) -> float:
        self.model.eval()
        return float(self.loss(batch))

    @torch.no_grad()
    def predict(self, batch: Batch) -> Tensor:
        self.model.eval()
        return self.model(batch.meteorology, batch.attributes, batch.padding)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "version": 1,
                "discharge_loss": self.discharge_loss,
                "softness": self.softness,
                "discharge_transform": None if self.discharge_transform is None else self.discharge_transform.to_json(),
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
            raise ValueError("unsupported checkpoint version")
        raw_config = dict(saved["config"])
        for key in ("features", "attributes"):
            raw_config[key] = tuple(raw_config[key])
        config = ModelConfig(**raw_config)
        state = saved["model"]
        normalization = Normalization(state["mean"], state["scale"], config.features)
        model = DischargeLSTM(config, normalization).to(device=device, dtype=state["mean"].dtype)
        model.load_state_dict(state)
        raw_transform = saved.get("discharge_transform")
        training = cls(
            model,
            learning_rate=saved["learning_rate"],
            discharge_loss=saved["discharge_loss"],
            softness=saved["softness"],
            discharge_transform=None if raw_transform is None else Transform.from_json(raw_transform),
        )
        training.optimizer.load_state_dict(saved["optimizer"])
        return training
