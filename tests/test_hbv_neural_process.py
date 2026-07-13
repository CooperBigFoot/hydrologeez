from typing import cast

import torch
from torch import nn

from hydrologeez.models.hbv import HBVForcing, HBVModel, NeuralRecharge

PHYSICAL_PARAMETER_NAMES = (
    "tt",
    "cfmax",
    "sfcf",
    "cwh",
    "cfr",
    "fc",
    "lp",
    "k0",
    "k1",
    "k2",
    "perc",
    "uzl",
    "maxbas",
)


def _tensor(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, device="cpu")


def _model(soil: NeuralRecharge) -> HBVModel:
    return HBVModel(
        tt=_tensor(0.0),
        cfmax=_tensor(3.0),
        sfcf=_tensor(1.0),
        cwh=_tensor(0.1),
        cfr=_tensor(0.05),
        fc=_tensor(100.0),
        lp=_tensor(0.7),
        k0=_tensor(0.3),
        k1=_tensor(0.1),
        k2=_tensor(0.05),
        perc=_tensor(2.0),
        uzl=_tensor(1.0),
        maxbas=_tensor(3.0),
        soil=soil,
    ).to(dtype=torch.float64, device="cpu")


def _physical_parameters(model: HBVModel) -> dict[str, torch.Tensor]:
    return {name: getattr(model, name).detach() for name in PHYSICAL_PARAMETER_NAMES}


def _forcing() -> HBVForcing:
    base_precip = torch.tensor(
        [0.0, 2.0, 8.0, 1.0, 12.0, 4.0, 0.0, 6.0, 3.0, 10.0, 1.0, 5.0],
        dtype=torch.float64,
        device="cpu",
    )
    precip = torch.stack(
        (
            base_precip,
            base_precip.roll(2),
            base_precip.roll(4),
            base_precip.roll(6),
        )
    ).repeat(1, 3)
    return HBVForcing(
        precip=precip,
        pet=torch.full((4, 36), 0.8, dtype=torch.float64, device="cpu"),
        temp=torch.full((4, 36), 5.0, dtype=torch.float64, device="cpu"),
    )


def test_neural_recharge_installs_assembles_bounds_and_runs() -> None:
    torch.manual_seed(1729)
    soil = NeuralRecharge()
    model = _model(soil)
    named_parameters = dict(model.named_parameters())

    assert model.soil is soil
    assert tuple(model.parameter_bounds) == PHYSICAL_PARAMETER_NAMES
    assert model.parameter_bounds["fc"] == (50.0, 700.0)
    assert model.parameter_bounds["lp"] == (0.3, 1.0)
    assert "beta" not in model.parameter_bounds
    assert "beta" not in named_parameters
    assert tuple(name for name in named_parameters if name.startswith("soil.")) == (
        "soil.mlp.0.weight",
        "soil.mlp.0.bias",
        "soil.mlp.2.weight",
        "soil.mlp.2.bias",
    )

    streamflow = model.run(_forcing(), parameters=_physical_parameters(model))

    assert streamflow.shape == (4, 36)
    assert streamflow.dtype == torch.float64
    assert streamflow.device.type == "cpu"
    assert torch.isfinite(streamflow).all()


def test_neural_recharge_training_reduces_loss_with_finite_weight_gradients() -> None:
    torch.manual_seed(1729)
    teacher = _model(NeuralRecharge())
    student = _model(NeuralRecharge())
    forcing = _forcing()
    teacher_parameters = _physical_parameters(teacher)
    student_parameters = _physical_parameters(student)
    student_mlp = cast(nn.Sequential, student.soil.mlp)

    with torch.no_grad():
        target = teacher.run(forcing, parameters=teacher_parameters)

    optimizer = torch.optim.Adam(student_mlp.parameters(), lr=0.03)
    optimized_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized_parameter_ids == {id(parameter) for parameter in student_mlp.parameters()}
    losses: list[float] = []

    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        prediction = student.run(forcing, parameters=student_parameters)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()

        weight_gradients = []
        for name, parameter in student_mlp.named_parameters():
            if name.endswith("weight"):
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
                weight_gradients.append(parameter.grad.reshape(-1))
        assert weight_gradients
        assert torch.count_nonzero(torch.cat(weight_gradients)) > 0

        optimizer.step()
        losses.append(loss.detach().item())

    early_loss = sum(losses[:5]) / 5
    late_loss = sum(losses[-5:]) / 5
    assert late_loss < early_loss
