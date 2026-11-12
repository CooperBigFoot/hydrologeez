"""Bounded warm-cache CAMELS loading, batching and real-data runtime measurements."""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from hydrologeez.camels import Camels
from hydrologeez.camels_meteorology import CamelsMeteorology, Meteorology
from hydrologeez.hybrid.camels import load_inputs
from hydrologeez.hybrid.data import Normalization
from hydrologeez.hybrid.model import HybridHBV, ModelConfig
from hydrologeez.hybrid.runtime import Training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meteorology", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    store = CamelsMeteorology(args.meteorology)
    features = ("daymet.prcp", "maurer.prcp", "nldas.prcp", "daymet.srad", "daymet.vp", "daymet.tmean")
    config = ModelConfig(features=features, hidden=4)
    start = time.perf_counter()
    selected = store.select(store.ids[:128], "1990-01-01", "2000-01-01", features)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    tensor = torch.from_numpy(selected.values)
    conversion_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(500):
        _batch = tensor[:32, :365]
    slice_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(500):
        _batch = tensor[list(range(32)), :365]
    indexed_seconds = time.perf_counter() - start
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    start = time.perf_counter()
    transferred = tensor[:32, :365].to(device)
    if device == "mps":
        torch.mps.synchronize()
    transfer_seconds = time.perf_counter() - start
    start = time.perf_counter()
    inputs = load_inputs(
        store,
        store.ids[:128],
        "1990-01-01",
        "2000-01-01",
        features=features,
        physical=("daymet.prcp", "daymet.tmean", "daymet.pet"),
        observations=Camels(args.observations),
    )
    adapter_seconds = time.perf_counter() - start
    assert isinstance(inputs.source, Meteorology)
    arrays = (inputs.source.values, inputs.source.valid, inputs.source.present)
    resident_bytes = sum(array.nbytes for array in arrays)
    resident_bytes += sum(t.numel() * t.element_size() for t in (inputs.meteorology, inputs.attributes, inputs.forcing))
    if inputs.observations is not None:
        resident_bytes += sum(
            a.nbytes
            for a in (
                inputs.observations.discharge_cfs,
                inputs.observations.discharge_mm_day,
                inputs.observations.present,
                inputs.observations.qc,
            )
        )
    if inputs.discharge is not None and inputs.discharge_valid is not None:
        resident_bytes += inputs.discharge.numel() * inputs.discharge.element_size() + inputs.discharge_valid.numel()
    batch = inputs.batch(config, basins=slice(0, 2), context=slice(0, 29), simulation=slice(0, 29))
    normalization = Normalization.fit(batch.meteorology, batch.padding, config.features)
    runs: dict[str, dict[str, float]] = {}
    for encoder in ("lstm", "kda"):
        for stage in ("direct", "pretrain"):
            torch.manual_seed(0)
            model = HybridHBV(ModelConfig(encoder=encoder, features=features, hidden=4), normalization)
            training = Training(model, stage=stage)
            hidden = torch.zeros_like(batch.meteorology, dtype=torch.bool)
            hidden[:, ::3] = True
            start = time.perf_counter()
            loss = training.train_step(batch, hidden=hidden if stage == "pretrain" else None)
            result = {"first_loss": loss}
            if stage == "pretrain":
                training.set_stage("frozen")
                result["frozen_loss"] = training.train_step(batch)
            prediction = training.predict(batch)
            result["seconds"] = time.perf_counter() - start
            result["finite_predictions"] = float(torch.isfinite(prediction.outlet).all())
            runs[f"{encoder}.{stage}"] = result
    payload = {
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cache": "warm filesystem cache; not a cold-read or target VM claim",
        "torch_threads": 1,
        "selection": {
            "basins": 128,
            "start": "1990-01-01",
            "stop": "2000-01-01",
            "features": features,
            "shape": selected.values.shape,
            "dtype": str(selected.values.dtype),
            "value_bytes": selected.values.nbytes,
            "mask_bytes": selected.valid.nbytes + selected.present.nbytes,
        },
        "stored_file_bytes": sum(p.stat().st_size for p in args.meteorology.rglob("*") if p.is_file()),
        "npy_payload_bytes": sum(np.load(p, mmap_mode="r").nbytes for p in args.meteorology.rglob("*.npy")),
        "load_seconds": load_seconds,
        "zero_copy_tensor_conversion_seconds": conversion_seconds,
        "500_slice_batches_seconds": slice_seconds,
        "500_indexed_copy_batches_seconds": indexed_seconds,
        "batch_shape": list(transferred.shape),
        "device": device,
        "one_batch_transfer_seconds": transfer_seconds,
        "adapter_load_seconds": adapter_seconds,
        "adapter_resident_bytes": resident_bytes,
        "peak_process_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_note": "macOS bytes, Linux KiB; includes all selections, interpreter, torch, runtime and mapped pages",
        "bounded_runtime": runs,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
