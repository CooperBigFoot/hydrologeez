"""Run both hybrid strategies without external data: uv run --extra hybrid python -m scripts.hybrid_synthetic."""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Literal

import torch

from hydrologeez.hybrid.data import Normalization
from hydrologeez.hybrid.hbv import simulate
from hydrologeez.hybrid.model import HybridHBV, ModelConfig
from hydrologeez.hybrid.runtime import Batch, Training


def run(encoder: Literal["lstm", "kda"], device: str, directory: Path) -> dict[str, object]:
    torch.manual_seed(7)
    values = torch.rand(2, 24, 3, device=device) * torch.tensor([25.0, 12.0, 3.0], device=device)
    values[..., 1] -= 3
    padding = torch.zeros(2, 24, dtype=torch.bool, device=device)
    attributes = torch.tensor([[0.4], [0.7]], device=device)
    forcing = values.transpose(0, 1).unsqueeze(2)
    area = torch.ones(2, 1, device=device)
    zone = torch.tensor([0.0, 2.0, 0.05, 0.1, 1.0, 100.0, 1.0, 0.5], device=device).expand(2, 1, -1)
    response = torch.tensor([1.0, 0.1, 0.1, 0.05], device=device).expand(2, -1)
    target = simulate(forcing, area, zone, response, torch.full((2,), 2.7, device=device)).outlet
    batch = Batch(values, attributes, padding, forcing, area, target, torch.ones_like(target, dtype=torch.bool))
    # This synthetic pool is the training pool; evaluation fitting is never implicit.
    normalization = Normalization.fit(values, padding, ("P", "T", "PET"))
    config = ModelConfig(encoder=encoder, attributes=("aridity",), hidden=8)
    model = HybridHBV(config, normalization).to(device)
    training = Training(model, learning_rate=0.01)
    start = time.perf_counter()
    initial = training.evaluate(batch)
    direct = [training.train_step(batch) for _ in range(8)]
    final = training.evaluate(batch)
    # Independent weights for alpha; shared implementation, not shared optimization.
    alpha = Training(HybridHBV(config, normalization).to(device), stage="pretrain", learning_rate=0.01)
    hidden = torch.rand_like(values) < 0.25
    pretrain = [alpha.train_step(batch, hidden=hidden) for _ in range(8)]
    checkpoint = directory / f"{encoder}-pretrained.pt"
    alpha.save(checkpoint)
    alpha = Training.load(checkpoint, device=device)
    alpha.set_stage("frozen")
    representation = alpha.model.represent(values, attributes, padding).detach().clone()
    frozen = [alpha.train_step(batch) for _ in range(8)]
    change = (alpha.model.represent(values, attributes, padding) - representation).abs().max().item()
    prediction = alpha.predict(batch)
    checkpoint = directory / f"{encoder}-discharge.pt"
    alpha.save(checkpoint)
    restored = Training.load(checkpoint, device=device)
    difference = (restored.predict(batch).outlet - prediction.outlet).abs().max().item()
    # Restart water stores using the SAME estimated parameters and context.
    with torch.no_grad():
        parameters = alpha.model.physical_parameters(representation, padding)
        first = simulate(forcing[:12], area, parameters[:, None, :8], parameters[:, 8:12], parameters[:, 12])
        tail = simulate(
            forcing[12:], area, parameters[:, None, :8], parameters[:, 8:12], parameters[:, 12], first.final
        )
    restart_error = (torch.cat((first.outlet, tail.outlet)) - prediction.outlet).abs().max().item()
    if device == "mps":
        torch.mps.synchronize()
    return {
        "encoder": encoder,
        "device": device,
        "dtype": "float32",
        "batch": 2,
        "days": 24,
        "hidden": 8,
        "direct_initial_mse": initial,
        "direct_final_mse": final,
        "direct_steps": direct,
        "pretraining_steps": pretrain,
        "frozen_steps": frozen,
        "frozen_representation_change": change,
        "checkpoint_max_error": difference,
        "restart_max_error": restart_error,
        "seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path(".hybrid-example"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "runs": [run(encoder, args.device, args.output) for encoder in ("lstm", "kda")],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
