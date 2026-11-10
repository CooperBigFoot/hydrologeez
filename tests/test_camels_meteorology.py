"""Raw products remain distinct, aligned and independently selectable."""

import zipfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from test_camels import source_files

from hydrologeez.camels_meteorology import CamelsMeteorology, prepare_meteorology


def meteorology_sources(tmp_path: Path) -> Path:
    raw, _ = source_files(tmp_path)
    path = raw / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
    with zipfile.ZipFile(path, "a") as archive:
        for product, offset in (("maurer", 10), ("nldas", 20)):
            for gauge in (1, 2):
                text = archive.read(f"x/basin_mean_forcing/daymet/01/{gauge:08}_lump_cida_forcing_leap.txt").decode()
                text = text.replace(f"12 0 {gauge} 0", f"12 0 {gauge + offset} 0")
                if product == "maurer":
                    text = text.replace(f"2000 2 29 12 0 {gauge + 1} 0 0 12 2 0\n", "")
                    text = text.replace("14 4 0", "14 4 -999")
                archive.writestr(f"x/basin_mean_forcing/{product}/01/{gauge:08}_lump_{product}_forcing_leap.txt", text)
    return raw


def test_products_preserve_source_values_gaps_and_selection_order(tmp_path: Path) -> None:
    raw = meteorology_sources(tmp_path)
    output = tmp_path / "meteorology"
    prepare_meteorology(raw, output)
    store = CamelsMeteorology(output)
    features = ("nldas.prcp", "daymet.prcp", "maurer.prcp", "maurer.vp", "daymet.tmean", "daymet.pet")
    selected = store.select(["00000002", "00000001"], "2000-02-28", "2000-03-02", features)
    assert selected.ids == ("00000002", "00000001")
    assert selected.features == features
    assert selected.values.dtype == np.float32
    assert_allclose(selected.values[:, 0, :3], [[22, 2, 12], [21, 1, 11]])
    assert_array_equal(selected.present[0, :, 2], [True, False, True])
    assert_array_equal(selected.valid[0, :, 3], [True, False, False])
    assert selected.values[0, 2, 3] == -999
    assert_allclose(selected.values[0, :, 4], [5, 7, 9])
    assert np.isfinite(selected.values[:, :, 5]).all()
    assert store.metadata["fields"]["daymet.pet"]["unit"] == "mm/day"
    assert "Oudin" in store.metadata["fields"]["daymet.pet"]["derivation"]
    for ids, start, stop, fields in [
        ([], "2000-02-28", "2000-03-01", features),
        (["00000001"], "1999-01-01", "2000-03-01", features),
        (["00000001"], "2000-02-28", "2000-03-01", ("bogus",)),
        (["00000001"], "2000-02-28", "2000-03-01", ("daymet.prcp",) * 2),
    ]:
        with pytest.raises(ValueError):
            store.select(ids, start, stop, fields)


def test_resident_inputs_separate_physical_features_and_target_access(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from hydrologeez.camels import Camels, prepare
    from hydrologeez.hybrid.camels import load_inputs
    from hydrologeez.hybrid.model import ModelConfig

    raw = meteorology_sources(tmp_path)
    output = tmp_path / "meteorology"
    prepare_meteorology(raw, output)
    daily = tmp_path / "daily"
    prepare(raw, daily, raw / "areas.json")
    features = ("nldas.prcp", "daymet.prcp", "maurer.prcp")
    physical = ("daymet.prcp", "daymet.tmean", "daymet.pet")
    store = CamelsMeteorology(output)
    inputs = load_inputs(
        store, ["00000002", "00000001"], "2000-02-28", "2000-03-02", features=features, physical=physical
    )
    config = ModelConfig(features=features)
    batch = inputs.batch(config)
    assert batch.discharge is None  # no discharge store was authorized
    assert batch.meteorology.shape == (2, 3, 3)
    assert torch.isnan(batch.meteorology[:, 1, 2]).all()
    assert_allclose(batch.forcing[:, 0, 0, 0], [2, 3, 4])
    assert_allclose(batch.forcing[:, 0, 0, 1], [5, 7, 9])
    assert not batch.padding.any()
    assert batch.meteorology.data_ptr() == inputs.meteorology.data_ptr()
    with pytest.raises(ValueError, match="order"):
        inputs.batch(ModelConfig(features=features[::-1]))
    with pytest.raises(ValueError, match="physical"):
        load_inputs(
            store,
            ["00000001"],
            "2000-02-28",
            "2000-03-02",
            features=features,
            physical=("maurer.prcp", "maurer.tmean", "maurer.pet"),
        )
    supervised = load_inputs(
        store,
        ["00000002", "00000001"],
        "2000-02-28",
        "2000-03-02",
        features=features,
        physical=physical,
        observations=Camels(daily),
    )
    observed = supervised.batch(config, simulation=slice(0, 2))
    assert_array_equal(observed.discharge_valid, [[True, True], [False, False]])
    assert supervised.observations is not None
    assert supervised.observations.qc[:, 0].tolist() == ["A:e", "", "M"]
    masked = supervised.batch(config, evaluation_mask=torch.zeros((3, 2), dtype=torch.bool)).discharge_valid
    assert masked is not None and masked.sum() == 0


def test_product_coverage_does_not_truncate_other_products(tmp_path: Path) -> None:
    raw = meteorology_sources(tmp_path)
    archive_path = raw / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
    with zipfile.ZipFile(archive_path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    for name in entries:
        if "/maurer/" in name:
            entries[name] = b"\n".join(entries[name].splitlines()[:-1]) + b"\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, contents in entries.items():
            archive.writestr(name, contents)
    output = tmp_path / "meteorology"
    prepare_meteorology(raw, output)
    selected = CamelsMeteorology(output).select(
        ["00000001"], "2000-03-01", "2000-03-02", ("daymet.prcp", "maurer.prcp")
    )
    assert selected.values[0, 0, 0] == 3
    assert np.isnan(selected.values[0, 0, 1])
    assert not selected.present[0, 0, 1]


def test_local_sources_and_hybrid_training(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import io
    from datetime import date

    from hydrologeez.camels import Camels, prepare_attributes, read_attributes
    from hydrologeez.camels_meteorology import FIELDS, PRODUCTS
    from hydrologeez.hybrid.camels import load_inputs
    from hydrologeez.hybrid.data import Normalization
    from hydrologeez.hybrid.model import HybridHBV, ModelConfig
    from hydrologeez.hybrid.runtime import Training

    base = Path(__file__).resolve().parents[1] / "data/camels-us"
    if not (base / "meteorology/meteorology.json").exists() or not (base / "prepared/metadata.json").exists():
        pytest.skip("Local prepared CAMELS stores unavailable")
    store = CamelsMeteorology(base / "meteorology")
    fields = tuple(f"{product}.{field}" for product in PRODUCTS for field in FIELDS)
    selected = store.select(["01013500"], "1980-01-01", "1980-02-01", fields)
    with zipfile.ZipFile(base / "raw/basin_timeseries_v1p2_metForcing_obsFlow.zip") as archive:
        for i, product in enumerate(PRODUCTS):
            record = store.metadata["records"][f"{product}.01013500"]
            source = np.loadtxt(io.StringIO(archive.read(record["member"]).decode()), skiprows=4)
            assert_allclose(
                selected.values[0, :, i * 9 : i * 9 + 7], source[:31, 4:].astype(np.float32), rtol=0, atol=0
            )
    assert selected.dates[0] == date(1980, 1, 1).toordinal()
    assert selected.present.all() and selected.valid.all()
    attr = tmp_path / "attributes"
    prepare_attributes(base / "raw", attr)
    features = ("daymet.prcp", "maurer.prcp", "nldas.prcp")
    inputs = load_inputs(
        store,
        ["01013500", "01022500"],
        "2000-02-01",
        "2000-03-01",
        features=features,
        physical=("daymet.prcp", "daymet.tmean", "daymet.pet"),
        attributes=read_attributes(attr),
        attribute_names=("gauge_lat",),
        observations=Camels(base / "prepared"),
    )
    config = ModelConfig(features=features, attributes=("gauge_lat",), hidden=4)
    batch = inputs.batch(config)
    normalization = Normalization.fit(batch.meteorology, batch.padding, config.features)
    for encoder in ("lstm", "kda"):
        config = ModelConfig(encoder=encoder, features=features, attributes=("gauge_lat",), hidden=4)
        for stage in ("direct", "pretrain"):
            training = Training(HybridHBV(config, normalization), stage=stage)
            hidden = torch.zeros_like(batch.meteorology, dtype=torch.bool)
            hidden[:, ::3, :] = True
            loss = training.train_step(batch, hidden=hidden if stage == "pretrain" else None)
            assert np.isfinite(loss)
            if stage == "pretrain":
                training.set_stage("frozen")
                assert np.isfinite(training.train_step(batch))
            assert torch.isfinite(training.predict(batch).outlet).all()


def test_zero_swe_and_identical_temperatures_are_not_stored_twice(tmp_path: Path) -> None:
    raw = meteorology_sources(tmp_path)
    path = raw / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    for name in entries:
        if "/nldas/" in name:
            entries[name] = (
                entries[name]
                .replace(b"10 0 0", b"10 10 0")
                .replace(b"12 2 0", b"12 12 0")
                .replace(b"14 4 0", b"14 14 0")
            )
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    output = tmp_path / "meteorology"
    prepare_meteorology(raw, output)
    store = CamelsMeteorology(output)
    record = store.metadata["records"]["nldas.00000001"]
    assert record["constants"] == {"swe": 0.0}
    assert record["aliases"] == {"tmin": "tmax", "tmean": "tmax"}
    assert np.load(output / "nldas/00000001.npy").shape == (3, 6)
    selected = store.select(["00000001"], "2000-02-28", "2000-03-02", ("nldas.swe", "nldas.tmin", "nldas.tmean"))
    assert_allclose(selected.values[0], [[0, 10, 10], [0, 12, 12], [0, 14, 14]])
