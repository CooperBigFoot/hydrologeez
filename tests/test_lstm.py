"""Standalone discharge LSTM contracts: causal prediction, output policy, loss and checkpoints."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.testing import assert_close  # noqa: E402 -- optional dependency skip must precede import


def test_discharge_lstm_parameter_count_causality_and_output_policy() -> None:
    from hydrologeez.hybrid.data import Normalization
    from hydrologeez.lstm.model import DischargeLSTM, ModelConfig

    features = tuple(f"f{i}" for i in range(16))
    attributes = tuple(f"a{i}" for i in range(45))
    norm = Normalization(torch.zeros(16), torch.ones(16), features)
    config = ModelConfig(features=features, attributes=attributes, hidden=256, output_activation="softplus")
    model = DischargeLSTM(config, norm)
    assert sum(p.numel() for p in model.parameters()) == 326_913

    torch.manual_seed(4)
    values = torch.randn(3, 8, 16)
    attrs = torch.randn(3, 45)
    padding = torch.zeros(3, 8, dtype=torch.bool)
    full = model(values, attrs, padding)
    assert full.shape == (8, 3)  # [day, basin], matching hybrid HBV outlet convention
    assert (full > 0).all()
    # Causal: changing future meteorology cannot alter earlier discharge predictions.
    prefix = model(values[:, :4], attrs, padding[:, :4])
    assert_close(full[:4], prefix)

    # Linear activation preserves signed linear outputs without silent clipping.
    linear_model = DischargeLSTM(
        ModelConfig(features=("P",), attributes=(), hidden=4, output_activation="linear"),
        Normalization(torch.zeros(1), torch.ones(1), ("P",)),
    )
    with torch.no_grad():
        linear_model.readout.weight.zero_()
        linear_model.readout.bias.fill_(-2.5)
    out = linear_model(torch.zeros(1, 3, 1), torch.empty(1, 0), torch.zeros(1, 3, dtype=torch.bool))
    assert_close(out, torch.full((3, 1), -2.5))

    # Missing meteorological values are rejected rather than silently zero-filled without indicators.
    missing = values.clone()
    missing[0, 2, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(missing, attrs, padding)


def test_discharge_lstm_training_masking_and_checkpoint_restart(tmp_path: Path) -> None:
    from hydrologeez.hybrid.data import Normalization
    from hydrologeez.lstm.model import DischargeLSTM, ModelConfig
    from hydrologeez.lstm.runtime import Batch, Training

    torch.manual_seed(11)
    values = torch.rand(2, 10, 3) * torch.tensor([15.0, 10.0, 3.0])
    padding = torch.zeros(2, 10, dtype=torch.bool)
    padding[1, -2:] = True
    norm = Normalization.fit(values, padding, ("P", "T", "PET"))
    config = ModelConfig(features=norm.features, attributes=("aridity", "elev_mean"), hidden=8)
    model = DischargeLSTM(config, norm)

    discharge = torch.arange(1.0, 11.0)[:, None].expand(-1, 2).clone()
    discharge[:3] = 999.0  # warm-up days excluded by discharge_valid
    discharge[5, 0] = float("nan")  # missing observation
    valid = ~padding.T
    valid[:3] = False
    batch = Batch(values, torch.tensor([[0.5, 200.0], [1.2, 800.0]]), padding, discharge, valid)

    training = Training(model, learning_rate=0.01)
    before = [p.detach().clone() for p in model.parameters()]
    loss0 = training.train_step(batch)
    assert np.isfinite(loss0)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))

    # Changing unscored warm-up or padded targets does not affect evaluation loss.
    altered_q = discharge.clone()
    altered_q[:3] = -12345.0
    altered_q[-2:, 1] = 54321.0
    altered_batch = Batch(batch.meteorology, batch.attributes, batch.padding, altered_q, batch.discharge_valid)
    assert_close(torch.tensor(training.evaluate(batch)), torch.tensor(training.evaluate(altered_batch)))

    path = tmp_path / "lstm.pt"
    training.save(path)
    restored = Training.load(path)
    assert restored.model.config == config
    assert_close(restored.model.normalization.mean, norm.mean)
    assert_close(restored.predict(batch), training.predict(batch), rtol=0, atol=0)
    assert restored.evaluate(batch) == training.evaluate(batch)

    # Resumed Adam step matches uninterrupted training exactly.
    loss_a = training.train_step(batch)
    loss_b = restored.train_step(batch)
    assert loss_a == loss_b
    for a, b in zip(model.parameters(), restored.model.parameters(), strict=True):
        assert_close(a, b, rtol=0, atol=0)


def test_discharge_lstm_camels_inputs_and_soft_npe(tmp_path: Path) -> None:
    import polars as pl
    from test_camels_meteorology import meteorology_sources

    from hydrologeez.camels import Camels, CamelsAttributes, prepare
    from hydrologeez.camels_meteorology import CamelsMeteorology, prepare_meteorology
    from hydrologeez.hybrid.camels import load_inputs
    from hydrologeez.hybrid.data import Normalization
    from hydrologeez.lstm.camels import batch_from_inputs
    from hydrologeez.lstm.model import DischargeLSTM, ModelConfig
    from hydrologeez.lstm.runtime import Training

    raw = meteorology_sources(tmp_path)
    met_dir = tmp_path / "meteorology"
    daily_dir = tmp_path / "daily"
    prepare_meteorology(raw, met_dir)
    prepare(raw, daily_dir, raw / "areas.json")
    attributes = CamelsAttributes(
        pl.DataFrame({"gauge_id": ["00000001", "00000002"], "elev_mean": [0.1, 0.9], "slope_mean": [-0.5, 0.5]}),
        {
            "citation": "synthetic",
            "documentation": "synthetic",
            "fields": {},
            "missing_value": "-999",
            "source": "synthetic",
        },
    )

    features = ("daymet.prcp", "nldas.prcp", "daymet.tmean")
    inputs = load_inputs(
        CamelsMeteorology(met_dir),
        ["00000001", "00000002"],
        "2000-02-28",
        "2000-03-02",
        features=features,
        physical=("daymet.prcp", "daymet.tmean", "daymet.pet"),
        attributes=attributes,
        attribute_names=("elev_mean", "slope_mean"),
        observations=Camels(daily_dir),
    )
    config = ModelConfig(features=features, attributes=("elev_mean", "slope_mean"), hidden=8)
    mask = torch.tensor([[False, False], [True, True], [True, True]])
    batch = batch_from_inputs(inputs, config, evaluation_mask=mask)
    assert batch.meteorology.shape == (2, 3, 3)
    assert batch.discharge is not None and batch.discharge_valid is not None
    assert not batch.discharge_valid[0].any()

    norm = Normalization.fit(batch.meteorology, batch.padding, config.features)
    # Assign non-equal observations across valid days so soft_npe is defined.
    soft_batch = type(batch)(
        batch.meteorology,
        batch.attributes,
        batch.padding,
        torch.tensor([[1.0, 2.0], [2.0, 3.0], [5.0, 7.0]]),
        torch.tensor([[False, False], [True, True], [True, True]]),
    )
    training = Training(DischargeLSTM(config, norm), discharge_loss="soft_npe", softness=0.2)
    assert np.isfinite(training.train_step(soft_batch))
    pred = training.predict(batch)
    assert pred.shape == (3, 2)
    assert (pred > 0).all()

    with pytest.raises(ValueError, match="order"):
        batch_from_inputs(inputs, ModelConfig(features=features[::-1], attributes=("elev_mean", "slope_mean")))
