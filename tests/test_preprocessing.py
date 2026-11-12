import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hydrologeez.preprocessing import (
    Exclusion,
    Field,
    Log,
    PhysicalForcing,
    PreparedDataset,
    Standardization,
    Transform,
    ZScore,
    fit,
    homogeneity_flags,
    prepare,
)

IDS = ("a", "b")


def field(values: list[list[float]], accepted: list[list[bool]] | None = None) -> Field:
    array = np.array(values)
    return Field(array, np.isfinite(array) if accepted is None else np.array(accepted), "mm/day")


def zscore(transform: Transform) -> Standardization:
    step = transform.steps[0]
    assert isinstance(step, Standardization)
    return step


def test_global_and_per_basin_scopes_fit_only_accepted_pooled_values() -> None:
    # Rejected (999) and unpooled (-999) values must not reach the statistics.
    source = field(
        [[1.0, 10.0], [3.0, 30.0], [999.0, 999.0], [-999.0, -999.0]],
        [[True, True], [True, True], [False, False], [True, True]],
    )
    pool = np.array([[True, True], [True, True], [True, True], [False, False]])

    shared = zscore(fit([ZScore("global")], source, IDS, pool))
    assert_allclose(shared.mean, [11.0])
    assert_allclose(shared.scale, [math.sqrt((100 + 64 + 1 + 361) / 4)])

    separate = zscore(fit([ZScore("basin")], source, IDS, pool))
    assert separate.ids == IDS
    assert_allclose(separate.mean, [2.0, 20.0])
    assert_allclose(separate.scale, [1.0, 10.0])


def test_static_statistics_count_each_authorized_basin_once() -> None:
    static = Field(np.array([2.0, 6.0, 100.0]), np.ones(3, dtype=np.bool_), "m")
    # Basin a has many authorized days, b one, c none.
    pool = np.zeros((5, 3), dtype=np.bool_)
    pool[:, 0] = True
    pool[0, 1] = True
    steps = zscore(fit([ZScore()], static, ("a", "b", "c"), pool))
    assert_allclose(steps.mean, [4.0])
    assert_allclose(steps.scale, [2.0])


def test_per_basin_transform_never_applies_to_unfitted_basin() -> None:
    source = field([[1.0, 5.0], [3.0, 7.0]])
    pool = np.array([[True, False], [True, False]])
    transform = fit([ZScore("basin")], source, IDS, pool)
    assert_allclose(transform.forward(np.array([[1.0], [3.0]]), ("a",)), [[-1.0], [1.0]])
    with pytest.raises(ValueError, match="b"):
        transform.forward(source.values, IDS)
    with pytest.raises(ValueError, match="daily"):
        fit([ZScore("basin")], Field(np.array([1.0, 2.0]), np.ones(2, dtype=np.bool_), "m"), IDS, np.ones(2, bool))


def test_invalid_inputs_and_constant_features_are_explicit() -> None:
    source = field([[0.0, 1.0], [0.0, 2.0]])
    everywhere = np.ones((2, 2), dtype=np.bool_)
    with pytest.raises(ValueError, match="log"):
        fit([Log(0.0, "mm/day")], source, IDS, everywhere)
    with pytest.raises(ValueError, match="units"):
        fit([Log(0.01, "m3/s")], source, IDS, everywhere)
    with pytest.raises(ValueError, match="pool"):
        fit([ZScore()], source, IDS, np.zeros((2, 2), dtype=np.bool_))
    with pytest.raises(ValueError, match="finite"):
        Field(np.array([[np.nan]]), np.array([[True]]), "mm/day")
    # A constant feature keeps scale 1, so its standardized values are 0.
    constant = fit([ZScore("basin")], source, IDS, everywhere)
    assert_allclose(zscore(constant).scale, [1.0, 0.5])
    assert_allclose(constant.forward(source.values, IDS)[:, 0], [0.0, 0.0])


def test_chain_applies_in_declared_order_inverts_and_round_trips() -> None:
    source = field([[0.0, 1.0], [9.0, 99.0]])
    everywhere = np.ones((2, 2), dtype=np.bool_)
    transform = fit([Log(1.0, "mm/day"), ZScore()], source, IDS, everywhere)
    logs = np.log(np.array([1.0, 2.0, 10.0, 100.0]))
    expected = ((logs - logs.mean()) / logs.std()).reshape(2, 2)
    forward = transform.forward(source.values, IDS)
    assert_allclose(forward, expected)
    assert_allclose(transform.inverse(forward, IDS), source.values, atol=1e-12)
    restored = Transform.from_json(transform.to_json())
    assert restored == transform
    assert_array_equal(restored.forward(source.values, IDS), forward)


def test_exclusion_combines_named_daily_reasons_and_basin_assessments() -> None:
    daily = {
        "negative": np.array([[True, False], [False, False]]),
        "outlier": np.array([[False, False], [False, True]]),
    }
    basin = {"inhomogeneous": np.array([False, True]), "regulated": np.array([True, False])}
    mask = Exclusion(daily=("negative",), basin=("inhomogeneous",)).mask((2, 2), daily, basin)
    assert_array_equal(mask, [[True, True], [False, True]])
    # Basin-only rules need no daily evidence; unselected evidence is not inspected.
    only_basin = Exclusion(basin=("regulated",)).mask((3, 2), {"other": np.zeros(5, dtype=np.bool_)}, basin)
    assert_array_equal(only_basin, [[True, False]] * 3)
    with pytest.raises(ValueError, match="unknown"):
        Exclusion(daily=("missing",)).mask((2, 2), daily, basin)


def prepared_example(output: Path) -> dict[str, np.ndarray]:
    """Two basins, three days; QC excludes day 2 of basin b."""
    evidence = {"outlier": np.array([[False, False], [False, False], [False, True]])}
    excluded = Exclusion(daily=("outlier",)).mask((3, 2), evidence, {})
    prcp = np.array([[1.0, 3.0], [np.nan, 5.0], [3.0, 999.0]])
    q = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 600.0]])
    area = np.array([10.0, np.nan])
    forcing = np.arange(18.0).reshape(3, 2, 3)
    inputs = {"prcp": Field(prcp, np.isfinite(prcp) & ~excluded, "mm/day")}
    attributes = {"area": Field(area, np.isfinite(area), "km2")}
    discharge = Field(q, np.isfinite(q) & ~excluded, "mm/day")
    training = np.array([[True, True], [True, True], [False, False]])
    prepare(
        output,
        ids=IDS,
        dates=np.arange(730000, 730003),
        inputs=inputs,
        attributes=attributes,
        transforms={
            "prcp": fit([ZScore()], inputs["prcp"], IDS, np.ones((3, 2), dtype=np.bool_)),
            "area": Transform("km2"),
        },
        forcing=PhysicalForcing(("prcp", "tmean", "pet"), ("mm/day", "degC", "mm/day"), forcing),
        discharge=discharge,
        discharge_transform=fit([Log(0.01, "mm/day"), ZScore()], discharge, IDS, training),
        fill_value=-7.0,
        recipe={"exclusion": {"daily": ["outlier"]}, "discharge_pool": "days 0-1"},
        provenance={"qc_archive": "synthetic"},
    )
    return {"evidence": evidence["outlier"], "forcing": forcing, "q": q}


def test_prepared_dataset_fills_after_transform_and_keeps_targets_distinct(tmp_path: Path) -> None:
    source = prepared_example(tmp_path / "prepared")
    dataset = PreparedDataset.read(tmp_path / "prepared")

    # Accepted precipitation 1, 3, 3, 5: mean 3, population sd sqrt(2).
    s = math.sqrt(2)
    assert_allclose(dataset.inputs[:, :, 0], [[-2 / s, -7.0, 0.0], [0.0, 2 / s, -7.0]], rtol=1e-6)
    assert_array_equal(dataset.inputs_was_filled[:, :, 0], [[False, True, False], [False, False, True]])
    assert_array_equal(dataset.attributes, [[10.0], [-7.0]])
    assert_array_equal(dataset.attributes_was_filled, [[False], [True]])
    assert_array_equal(dataset.model_inputs(("prcp_was_filled", "prcp"))[:, :, 0], dataset.inputs_was_filled[:, :, 0])

    assert dataset.forcing.dtype == np.float32
    assert_array_equal(dataset.forcing, source["forcing"])
    discharge = dataset.discharge
    assert discharge is not None
    assert_array_equal(discharge.physical, source["q"])
    assert_array_equal(discharge.eligible, [[True, True], [False, True], [True, False]])
    logs = np.log([1.01, 2.01, 4.01])
    assert_allclose(discharge.target[2, 0], (np.log(5.01) - logs.mean()) / logs.std())
    assert np.isnan(discharge.target[~discharge.eligible]).all()
    assert_allclose(discharge.transform.inverse(discharge.target, IDS)[discharge.eligible], [1, 2, 4, 5])

    assert_array_equal(source["evidence"], [[False, False], [False, False], [False, True]])
    assert dataset.ids == IDS and dataset.features == ("prcp",) and dataset.units == ("mm/day",)
    assert dataset.recipe["exclusion"] == {"daily": ["outlier"]}
    with pytest.raises(FileExistsError):
        prepared_example(tmp_path / "prepared")


def prepare_single(output: Path, name: str, source: Field, transform: Transform) -> None:
    prepare(
        output,
        ids=IDS,
        dates=np.arange(730000, 730000 + len(source.values)),
        inputs={name: source},
        attributes={},
        transforms={name: transform},
        forcing=PhysicalForcing(("p", "t", "pet"), ("mm/day", "degC", "mm/day"), np.ones((len(source.values), 2, 3))),
        fill_value=-3.0,
        recipe={"pools": {name: "all days"}},
        provenance={},
    )


def test_per_basin_log_chain_fills_after_transform_and_persists(tmp_path: Path) -> None:
    source = field([[0.0, 9.0], [np.nan, 99.0], [1.0, 999.0]])
    transform = fit([Log(1.0, "mm/day"), ZScore("basin")], source, IDS, np.ones((3, 2), dtype=np.bool_))
    prepare_single(tmp_path / "prepared", "x", source, transform)
    dataset = PreparedDataset.read(tmp_path / "prepared")
    # Basin a: log 1, log 2 -> -1, 1. Basin b: log 10, 100, 1000 -> -sqrt(1.5), 0, sqrt(1.5).
    r = math.sqrt(1.5)
    assert_allclose(dataset.inputs[:, :, 0], [[-1.0, -3.0, 1.0], [-r, 0.0, r]], rtol=1e-6, atol=1e-7)
    restored = dataset.transforms["x"]
    assert restored == transform
    observed = dataset.inputs[:, (0, 2), 0].T.astype(np.float64)
    assert_allclose(restored.inverse(observed, IDS), [[0.0, 9.0], [1.0, 999.0]], rtol=1e-5, atol=1e-6)
    assert_allclose(restored.forward(np.array([[999.0, 1.0]]), ("b", "a")), [[r, 1.0]])
    assert_allclose(restored.forward(np.array([99.0]), ("b",)), [0.0], atol=1e-12)


def test_prepare_rejects_unfitted_basins_and_reserved_names(tmp_path: Path) -> None:
    source = field([[1.0, 5.0], [3.0, 7.0]])
    only_a = fit([ZScore("basin")], source, IDS, np.array([[True, False], [True, False]]))
    with pytest.raises(ValueError, match="b"):
        prepare_single(tmp_path / "unfitted", "x", source, only_a)
    with pytest.raises(ValueError, match="_was_filled"):
        prepare_single(tmp_path / "reserved", "x_was_filled", source, Transform("mm/day"))
    assert not (tmp_path / "unfitted").exists() and not (tmp_path / "reserved").exists()


def test_models_consume_prepared_data_repeatedly_without_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from torch.testing import assert_close

    from hydrologeez import preprocessing
    from hydrologeez.hybrid import data as hybrid_data
    from hydrologeez.hybrid.camels import prepared_inputs
    from hydrologeez.hybrid.data import Normalization
    from hydrologeez.hybrid.model import HybridHBV, ModelConfig
    from hydrologeez.hybrid.runtime import Training
    from hydrologeez.lstm.camels import batch_from_inputs
    from hydrologeez.lstm.model import ModelConfig as LSTMConfig

    source = prepared_example(tmp_path / "prepared")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("prepared data must not be refitted or retransformed")

    monkeypatch.setattr(preprocessing, "fit", forbidden)
    monkeypatch.setattr(preprocessing, "standardization", forbidden)
    monkeypatch.setattr(preprocessing.Transform, "forward", forbidden)
    monkeypatch.setattr(hybrid_data, "standardization", forbidden)
    features = ("prcp", "prcp_was_filled")
    attributes = ("area", "area_was_filled")
    config = ModelConfig(features=features, attributes=attributes, hidden=4)
    eligible = torch.tensor([[True, True], [False, True], [True, False]])
    for _ in range(2):
        dataset = PreparedDataset.read(tmp_path / "prepared")
        inputs = prepared_inputs(dataset, features=features, attribute_names=attributes)
        batch = inputs.batch(config)
        assert_array_equal(batch.meteorology[:, :, 1].numpy(), dataset.inputs_was_filled[:, :, 0])
        assert_array_equal(batch.attributes.numpy(), [[10.0, 0.0], [-7.0, 1.0]])
        assert_array_equal(batch.forcing[:, :, 0, :].numpy(), source["forcing"].astype(np.float32))
        assert batch.discharge_valid is not None and torch.equal(batch.discharge_valid, eligible)
        filled = torch.from_numpy(dataset.inputs_was_filled[:, :, 0])
        assert batch.filled is not None
        assert torch.equal(batch.filled[:, :, 0], filled) and not batch.filled[:, :, 1].any()
        identity = Normalization.identity(features)
        visible = hybrid_data.prepare(
            batch.meteorology, batch.padding, torch.zeros_like(batch.filled), identity, batch.filled
        )
        assert torch.equal(visible.inputs[:, :, 2].bool(), ~filled)
        assert torch.equal(visible.inputs[:, :, 0][filled], torch.full((2,), -7.0))
        hidden = torch.ones_like(batch.filled)
        targets = hybrid_data.prepare(batch.meteorology, batch.padding, hidden, identity, batch.filled).targets
        assert torch.equal(targets[:, :, 0], ~filled) and targets[:, :, 1].all()

        torch.manual_seed(0)
        training = Training(HybridHBV(config, Normalization.identity(features)))
        outlet = training.predict(batch).outlet
        observed = torch.from_numpy(source["q"]).to(outlet.dtype)
        # The excluded 600 mm/day and missing values never enter the loss.
        assert_close(training.evaluate(batch), float((outlet - observed)[eligible].square().mean()))
        training.set_stage("pretrain")
        assert np.isfinite(training.evaluate(batch, hidden=hidden))

        lstm_batch = batch_from_inputs(inputs, LSTMConfig(features=features, attributes=attributes))
        assert torch.isfinite(lstm_batch.meteorology).all()
        assert lstm_batch.discharge_valid is not None and torch.equal(lstm_batch.discharge_valid, eligible)


def test_archive_evidence_feeds_caller_exclusions(tmp_path: Path) -> None:
    from hydrologeez.quality_control import (
        DischargeRecord,
        DischargeSource,
        HomogeneityQuality,
        StreamflowQualityArchive,
        write_archive,
    )

    dates = np.arange(730000, 730030, dtype=np.int64)
    varying = np.linspace(2.0, 2.29, 30)
    negative = varying.copy()
    negative[3] = -1.0
    none, no_flags = np.zeros(30, dtype=np.bool_), np.full(30, "")
    write_archive(
        tmp_path / "qc",
        DischargeSource("synthetic", "m3/s", 1.0, {}),
        [DischargeRecord("x", dates, negative, none, no_flags), DischargeRecord("y", dates, varying, none, no_flags)],
    )
    archive = StreamflowQualityArchive(tmp_path / "qc")
    ids = ("y", "x")
    daily = archive.daily(ids, "1999-09-03", "1999-10-03")  # ordinals 730000-730029
    assessed = HomogeneityQuality(
        ("x", "y"),
        "yearly",
        ("MEAN", "CV"),
        ("snht",),
        np.array([["suspect", "useful"], ["useful", "doubtful"]]),
        np.zeros((2, 2), dtype=np.int64),
        np.zeros((2, 2, 1)),
        np.zeros((2, 2, 1), dtype=np.bool_),
    )
    basin = {"mean_inhomogeneous": homogeneity_flags(assessed, ids, "MEAN", ("suspect", "doubtful"))}
    assert_array_equal(basin["mean_inhomogeneous"], [False, True])

    excluded = Exclusion(daily=("negative",)).mask(daily.discharge.shape, daily.reasons, basin)
    assert_array_equal(np.argwhere(excluded), [[3, 1]])
    whole_basin = Exclusion(basin=("mean_inhomogeneous",)).mask(daily.discharge.shape, daily.reasons, basin)
    assert whole_basin[:, 1].all() and not whole_basin[:, 0].any()
    with pytest.raises(ValueError, match="index"):
        homogeneity_flags(assessed, ids, "P50", ("suspect",))
    with pytest.raises(ValueError, match="classification"):
        homogeneity_flags(assessed, ids, "MEAN", ("bad",))
