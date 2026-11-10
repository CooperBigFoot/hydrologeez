"""Conditional densities in physical units and resident training inputs."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")
from torch.testing import assert_close  # noqa: E402 -- optional dependency imports


def test_density_mapping_and_endpoints() -> None:
    from hydrologeez.flow.model import ConditionalFlow, ModelConfig

    model = ConditionalFlow(
        ModelConfig(("a", "b"), ("mm", "day"), ("c",), (-2.0, 10.0), (2.0, 30.0), hidden=(8,), transforms=2)
    ).double()
    theta = torch.tensor([[-1.0, 25.0], [1.0, 15.0]], dtype=torch.float64)
    attributes = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    # u=(1/4,3/4) or (3/4,1/4): product of physical Jacobians = (4*3/16)*(20*3/16).
    z = torch.tensor([[-3.0, 3.0], [3.0, -3.0]], dtype=torch.float64).sign() * torch.log(
        torch.tensor(3.0, dtype=torch.float64)
    )
    expected = model.flow(attributes).log_prob(z) - torch.log(torch.tensor(0.75 * 3.75, dtype=torch.float64))
    assert_close(model.log_prob(theta, attributes), expected)
    for value in (-2.0, 2.0, -3.0, float("nan")):
        invalid = theta.clone()
        invalid[0, 0] = value
        with pytest.raises(ValueError, match="strictly inside"):
            model.log_prob(invalid, attributes)
    samples = model.sample(attributes, 12)
    assert samples.shape == (12, 2, 2)
    assert torch.isfinite(samples).all()
    assert (samples >= model.lower).all() and (samples <= model.upper).all()


@pytest.mark.parametrize("dimensions", [(1, 2), (3, 1)])
def test_training_checkpoint(tmp_path: Path, dimensions: tuple[int, int]) -> None:
    from hydrologeez.flow.model import ConditionalFlow, ModelConfig
    from hydrologeez.flow.runtime import Batch, Training

    p, c = dimensions
    config = ModelConfig(
        tuple(f"p{i}" for i in range(p)),
        ("mm",) * p,
        tuple(f"c{i}" for i in range(c)),
        (0.0,) * p,
        (10.0,) * p,
        hidden=(8,),
        transforms=2,
    )
    model = ConditionalFlow(config)
    batch = Batch(torch.full((4, p), 3.0), torch.randn(4, c))
    training = Training(model)
    before = [v.detach().clone() for v in model.parameters()]
    assert torch.isfinite(training.loss(batch))
    training.train_step(batch)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    path = tmp_path / "checkpoint.pt"
    training.save(path)
    loaded = Training.load(path)
    assert loaded.model.config == config
    assert loaded.evaluate(batch) == training.evaluate(batch)
    # Equal next updates prove that Adam moments, not just weights, resumed.
    training.train_step(batch)
    loaded.train_step(batch)
    for a, b in zip(model.parameters(), loaded.model.parameters(), strict=True):
        assert_close(a, b, rtol=0, atol=0)
    assert loaded.model.sample(batch.attributes, 5).shape == (5, 4, p)


def test_resident_alignment_and_workflow(tmp_path: Path) -> None:
    import numpy as np
    import polars as pl

    from hydrologeez.camels import CamelsAttributes
    from hydrologeez.flow.data import load_inputs, save_examples
    from hydrologeez.flow.model import ConditionalFlow, ModelConfig
    from hydrologeez.flow.runtime import Training

    path = tmp_path / "examples.npz"
    save_examples(
        path,
        np.array([[0.0, 12.0], [1.0, 15.0], [2.0, 18.0]], dtype=np.float32),
        ids=("00000002", "00000001", "00000002"),
        parameters=("a", "b"),
        units=("mm", "day"),
    )
    attributes = CamelsAttributes(
        pl.DataFrame(
            {"gauge_id": ["00000001", "00000002"], "x": [10.0, 20.0], "y": [3.0, 4.0], "category": ["A", "B"]}
        ),
        {
            "citation": "synthetic",
            "documentation": "synthetic",
            "fields": {},
            "missing_value": "-999",
            "source": "synthetic",
        },
    )
    inputs = load_inputs(
        path, ("00000002", "00000001"), parameters=("b", "a"), attributes=attributes, attribute_names=("y", "x")
    )
    assert inputs.ids == ("00000002", "00000001", "00000002")
    assert inputs.units == ("day", "mm")
    assert inputs.parameters.device.type == "cpu"
    # Endpoints survive the reader unchanged; exact training policy belongs to model.
    assert_close(inputs.parameters, torch.tensor([[12.0, 0.0], [15.0, 1.0], [18.0, 2.0]]))
    config = ModelConfig(("b", "a"), ("day", "mm"), ("y", "x"), (10.0, -1.0), (20.0, 3.0), hidden=(8,))
    training = Training(ConditionalFlow(config))
    for field in ("category",):
        with pytest.raises(ValueError, match="numeric"):
            load_inputs(path, ("00000001",), parameters=("a",), attributes=attributes, attribute_names=(field,))
    missing = CamelsAttributes(attributes.table.with_columns(pl.lit(float("nan")).alias("x")), attributes.metadata)
    with pytest.raises(ValueError, match="finite"):
        load_inputs(path, ("00000001",), parameters=("a",), attributes=missing, attribute_names=("x",))
    path.unlink()  # No later batch may reopen storage.
    for _ in range(3):
        batch = inputs.batch(config, examples=[2, 0])
        assert_close(batch.attributes, torch.tensor([[4.0, 20.0], [4.0, 20.0]]))
        assert_close(batch.parameters, torch.tensor([[18.0, 2.0], [12.0, 0.0]]))
        training.train_step(batch)
    assert isinstance(training.evaluate(inputs.batch(config)), float)
    training.save(tmp_path / "flow.pt")
    restored = Training.load(tmp_path / "flow.pt")
    assert restored.model.sample(inputs.attributes, 7).shape == (7, 2, 2)
    wrong = ModelConfig(("a", "b"), ("mm", "day"), ("y", "x"), (-1.0, 10.0), (3.0, 20.0))
    with pytest.raises(ValueError, match="order"):
        inputs.batch(wrong)


def test_density_near_upper_bound() -> None:
    from hydrologeez.flow.model import ConditionalFlow, ModelConfig

    model = ConditionalFlow(ModelConfig(("a",), ("mm",), ("c",), (-10.0,), (1.0,), hidden=(8,)))
    theta = torch.nextafter(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    assert torch.isfinite(model.log_prob(theta, torch.zeros(1, 1))).all()
