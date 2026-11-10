"""Hybrid physical and learning contracts on small CPU tensors."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.testing import assert_close  # noqa: E402 -- optional dependency skip must precede import


def test_tensor_hbv_worked_day_and_restart() -> None:
    from hydrologeez.hybrid.hbv import Memory, simulate

    forcing = torch.tensor([[[[6.0, 2.0, 2.0]]]], dtype=torch.float64)
    zone = torch.tensor([[[0.0, 2.0, 0.05, 0.1, 1.0, 100.0, 1.0, 0.5]]], dtype=torch.float64)
    response = torch.tensor([[1.0, 0.0, 0.1, 0.05]], dtype=torch.float64)
    state = Memory(
        torch.tensor([[[10.0, 0.0, 50.0]]], dtype=torch.float64),
        torch.tensor([[10.0, 20.0]], dtype=torch.float64),
        torch.zeros(1, 6, dtype=torch.float64),
    )
    result = simulate(
        forcing, torch.ones(1, 1, dtype=torch.float64), zone, response, torch.tensor([2.0], dtype=torch.float64), state
    )
    assert_close(result.outlet, torch.tensor([[1.21]], dtype=torch.float64))
    assert_close(result.final.zone, torch.tensor([[[6.0, 0.6, 52.7]]], dtype=torch.float64))
    assert_close(result.final.groundwater, torch.tensor([[12.33, 19.95]], dtype=torch.float64))
    assert_close(result.final.routing, torch.tensor([[1.21, 0, 0, 0, 0, 0]], dtype=torch.float64))


def test_tensor_hbv_parity_mass_and_gradients() -> None:
    from hydrologeez.hbv_light import Forcing, Parameters
    from hydrologeez.hbv_light import Memory as NMemory
    from hydrologeez.hbv_light import simulate as reference
    from hydrologeez.hybrid.hbv import Memory, simulate

    rng = np.random.default_rng(8)
    values = rng.uniform([0, -5, 0], [30, 12, 4], (24, 2, 3))
    z = np.tile([0.3, 2.1, 0.08, 0.12, 1.1, 100.0, 1.3, 0.6], (2, 1))
    g = np.array([[1.2, 0.2, 0.08, 0.03]])
    initial = NMemory(np.array([[5.0, 0.2, 40.0], [2.0, 0.1, 30.0]]), np.array([[8.0, 12.0]]), np.zeros(2))
    expected = reference(
        Forcing(values, np.array([0.3, 0.7]), np.array([0, 2])),
        Parameters(np.array([0]), z, g, np.array([2.7])),
        initial,
        history=True,
    )
    f = torch.from_numpy(values[:, None])
    area = torch.tensor([[0.3, 0.7]], dtype=torch.float64)
    zone = torch.from_numpy(z[None]).requires_grad_()
    groundwater = torch.from_numpy(g).requires_grad_()
    maxbas = torch.tensor([2.7], dtype=torch.float64, requires_grad=True)
    state = Memory(
        torch.from_numpy(initial.zone[None]),
        torch.from_numpy(initial.groundwater),
        torch.zeros(1, 6, dtype=torch.float64),
    )
    result = simulate(f, area, zone, groundwater, maxbas, state)
    assert_close(result.outlet, torch.from_numpy(expected.outlet), atol=1e-12, rtol=1e-12)
    assert_close(result.final.zone[0], torch.from_numpy(expected.final.zone))
    assert_close(result.final.groundwater, torch.from_numpy(expected.final.groundwater))
    assert_close(result.final.routing[0, :2], torch.from_numpy(expected.final.routing))
    first = simulate(f[:9], area, zone, groundwater, maxbas, state)
    last = simulate(f[9:], area, zone, groundwater, maxbas, first.final)
    assert_close(torch.cat([first.outlet, last.outlet]), result.outlet)
    assert expected.history is not None
    inputs = (torch.from_numpy(expected.history.zone_flux[..., 3]) * area).sum()
    evaporation = (torch.from_numpy(expected.history.zone_flux[..., 2]) * area).sum()

    def storage(memory: Memory) -> torch.Tensor:
        return (memory.zone.sum(-1) * area).sum() + memory.groundwater.sum() + memory.routing.sum()

    assert_close(storage(state) + inputs, storage(result.final) + evaporation + result.outlet.sum())
    # Short smooth wet sequence, away from thresholds and caps.
    assert torch.autograd.gradcheck(
        lambda zp, gp, mb: simulate(f[:3], area, zp, gp, mb, state).outlet,
        (zone, groundwater, maxbas),
        eps=1e-6,
        atol=1e-5,
    )
    dry = simulate(torch.zeros_like(f), area, zone, groundwater, maxbas)
    dry.outlet.sum().backward()
    for tensor in (zone, groundwater, maxbas):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_preprocessing_hides_only_selected_valid_targets() -> None:
    from hydrologeez.hybrid.data import Normalization, masked_mse, prepare

    values = torch.tensor([[[1.0, 10.0], [3.0, float("nan")], [1000.0, 1000.0]]])
    padding = torch.tensor([[False, False, True]])
    norm = Normalization.fit(values, padding, ("P", "T"))
    assert_close(norm.mean, torch.tensor([2.0, 10.0]))
    hidden = torch.tensor([[[True, False], [False, True], [True, True]]])
    prepared = prepare(values, padding, hidden, norm)
    assert prepared.targets.sum() == 1
    assert prepared.inputs[0, 0, 0] == 0
    assert prepared.inputs[0, 0, 4] == 1  # synthetic hidden indicator
    assert prepared.inputs[0, 1, 5] == 0  # genuine missing is not hidden
    changed = values.clone()
    changed[0, 0, 0] = 999.0
    assert_close(prepare(changed, padding, hidden, norm).inputs, prepared.inputs)
    prediction = torch.zeros_like(values, requires_grad=True)
    loss = masked_mse(prediction, prepared.normalized, prepared.targets)
    assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert prediction.grad is not None
    assert_close(prediction.grad != 0, prepared.targets)
    with pytest.raises(ValueError, match="targets"):
        masked_mse(prediction, prepared.normalized, torch.zeros_like(hidden))


def test_kda_coordinate_decay_and_gradcheck() -> None:
    from hydrologeez.hybrid.encoders import kda_recurrence

    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[1.0, 0.0]], [[0.6, 0.8]]]], dtype=torch.float64)
    v = torch.tensor([[[[2.0]], [[3.0]]]], dtype=torch.float64)
    decay = torch.tensor([[[[0.8, 0.4]], [[0.5, 0.25]]]], dtype=torch.float64).log()
    beta = torch.ones(1, 2, 1, dtype=torch.float64)
    output, state = kda_recurrence(q, k, v, decay, beta)
    # First memory [2,0]; decayed second memory [1,0], residual 3-.6=2.4.
    assert_close(state, torch.tensor([[[[2.44], [1.92]]]], dtype=torch.float64))
    assert_close(output.flatten(), torch.tensor([2.0, 1.92], dtype=torch.float64) / 2**0.5)
    args = tuple(t.requires_grad_() for t in (q, k, v, decay, beta))
    assert torch.autograd.gradcheck(kda_recurrence, args)


@pytest.mark.parametrize("encoder", ["lstm", "kda"])
def test_training_stages_checkpoint_and_freeze(encoder: Literal["lstm", "kda"], tmp_path: Path) -> None:
    from hydrologeez.hybrid.data import Normalization
    from hydrologeez.hybrid.model import HybridHBV, ModelConfig
    from hydrologeez.hybrid.runtime import Batch, Training

    torch.manual_seed(19)
    values = torch.rand(2, 12, 3) * torch.tensor([20.0, 10.0, 2.0])
    padding = torch.zeros(2, 12, dtype=torch.bool)
    padding[1, -2:] = True
    norm = Normalization.fit(values, padding, ("P", "T", "PET"))
    model = HybridHBV(ModelConfig(encoder=encoder, features=norm.features, attributes=("aridity",), hidden=8), norm)
    batch = Batch(
        values,
        torch.ones(2, 1),
        padding,
        values.transpose(0, 1).unsqueeze(2),
        torch.ones(2, 1),
        discharge=torch.ones(12, 2),
        discharge_valid=~padding.T,
    )
    training = Training(model, stage="direct", learning_rate=0.01)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loss = training.train_step(batch)
    assert np.isfinite(loss)
    for component in ("encoder", "parameter_head"):
        assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items() if k.startswith(component))
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items() if k.startswith("pretext_head"))
    training.set_stage("pretrain")
    hidden = torch.rand_like(values) < 0.3
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert np.isfinite(training.train_step(batch, hidden=hidden))
    for component in ("encoder", "pretext_head"):
        assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items() if k.startswith(component))
    path = tmp_path / "model.pt"
    training.save(path)
    restored = Training.load(path)
    assert restored.model.config == model.config
    assert_close(restored.model.normalization.mean, norm.mean)
    assert_close(restored.predict(batch).outlet, training.predict(batch).outlet)
    restored.set_stage("frozen")
    before_repr = restored.model.represent(values, batch.attributes, padding).detach().clone()
    before = {k: v.clone() for k, v in restored.model.state_dict().items()}
    assert np.isfinite(restored.train_step(batch))
    assert not restored.model.encoder.training
    assert all(not p.requires_grad for p in restored.model.encoder.parameters())
    assert_close(restored.model.represent(values, batch.attributes, padding), before_repr, rtol=0, atol=0)
    assert all(torch.equal(v, before[k]) for k, v in restored.model.state_dict().items() if k.startswith("encoder"))
    assert any(
        not torch.equal(v, before[k]) for k, v in restored.model.state_dict().items() if k.startswith("parameter_head")
    )
    restored.save(path)
    frozen = Training.load(path)
    assert frozen.stage == "frozen"
    assert_close(frozen.predict(batch).outlet, restored.predict(batch).outlet, rtol=0, atol=0)
    assert np.isfinite(frozen.evaluate(batch))


def test_kda_pinned_fla_forward_backward() -> None:
    import json

    from hydrologeez.hybrid.encoders import kda_recurrence

    fixture = json.loads((Path(__file__).parent / "data/kda_fla.json").read_text())
    args = [torch.tensor(x, requires_grad=True) for x in fixture["inputs"]]
    output, state = kda_recurrence(*args)
    assert_close(output, torch.tensor(fixture["output"]), rtol=1e-5, atol=1e-6)
    assert_close(state, torch.tensor(fixture["state"]), rtol=1e-5, atol=1e-6)
    (output.square().sum() + state.square().sum()).backward()
    for value, expected in zip(args, fixture["gradients"], strict=True):
        assert_close(value.grad, torch.tensor(expected), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("encoder", ["lstm", "kda"])
def test_encoder_prefix_causality(encoder: Literal["lstm", "kda"]) -> None:
    from hydrologeez.hybrid.encoders import KDA, LSTM

    torch.manual_seed(2)
    model = KDA(5, 8) if encoder == "kda" else LSTM(5, 8)
    x = torch.randn(2, 6, 5)
    assert_close(model(x)[:, :3], model(x[:, :3]))
