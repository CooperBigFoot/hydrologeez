"""Tests for explicit tensor dtype and device policies."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import hydrologeez
from hydrologeez.precision import (
    REFERENCE_DEVICE,
    REFERENCE_DTYPE,
    TRAINING_DTYPE,
    enforce_float64,
    reference_defaults,
    reference_tensor,
    training_defaults,
    training_tensor,
)

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def test_import_succeeds_when_jax_x64_disabled() -> None:
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "0"
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "import hydrologeez"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_reference_defaults_are_explicit_fresh_and_local() -> None:
    before = torch.get_default_dtype()
    first = reference_defaults()
    second = reference_defaults()
    assert first == {"dtype": torch.float64, "device": torch.device("cpu")}
    assert first is not second
    assert torch.get_default_dtype() == before


@pytest.mark.parametrize(
    "data",
    [np.array([1.25, 2.5], dtype=np.float32), torch.tensor([1.25, 2.5], dtype=torch.float32)],
)
def test_reference_tensor_converts_to_cpu_float64(data: object) -> None:
    before = torch.get_default_dtype()
    result = reference_tensor(data)
    assert result.dtype == torch.float64
    assert result.device == torch.device("cpu")
    torch.testing.assert_close(result, torch.tensor([1.25, 2.5], dtype=torch.float64))
    assert torch.get_default_dtype() == before


def test_reference_tensor_preserves_autograd_connectivity() -> None:
    before = torch.get_default_dtype()
    source = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    reference_tensor(source).sum().backward()
    assert source.grad is not None
    torch.testing.assert_close(source.grad, torch.ones_like(source))
    assert torch.get_default_dtype() == before


@pytest.mark.parametrize("device", _available_devices(), ids=str)
def test_training_helpers_use_requested_device_without_global_mutation(device: torch.device) -> None:
    before = torch.get_default_dtype()
    first = training_defaults(device)
    second = training_defaults(str(device))
    assert first == {"dtype": torch.float32, "device": device}
    assert first is not second
    result = training_tensor(np.array([1.25, 2.5], dtype=np.float64), device=device)
    assert result.dtype == torch.float32
    assert result.device.type == device.type
    torch.testing.assert_close(result.cpu(), torch.tensor([1.25, 2.5], dtype=torch.float32))
    assert torch.get_default_dtype() == before


def test_legacy_enforce_float64_is_deprecated_noop() -> None:
    before = torch.get_default_dtype()
    with pytest.warns(DeprecationWarning, match="retired"):
        assert enforce_float64() is None
    assert torch.get_default_dtype() == before


def test_package_root_policy_exports_match_precision_module() -> None:
    assert hydrologeez.REFERENCE_DEVICE is REFERENCE_DEVICE
    assert hydrologeez.REFERENCE_DTYPE is REFERENCE_DTYPE
    assert hydrologeez.TRAINING_DTYPE is TRAINING_DTYPE
    assert hydrologeez.reference_defaults is reference_defaults
    assert hydrologeez.reference_tensor is reference_tensor
    assert hydrologeez.training_defaults is training_defaults
    assert hydrologeez.training_tensor is training_tensor
