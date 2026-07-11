"""Tests for the PyTorch delay-line primitive."""

import torch

from hydrologeez.convolution import convolve_delay_line


def test_unbatched_read_after_shift_exact():
    buffer = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    kernel = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    output, new_buffer = convolve_delay_line(buffer, kernel, torch.tensor(5.0, dtype=torch.float64))
    torch.testing.assert_close(new_buffer, torch.tensor([21.0, 31.5, 2.5], dtype=torch.float64))
    torch.testing.assert_close(output, torch.tensor(21.0, dtype=torch.float64))


def test_batched_buffers_shared_kernel_and_per_basin_inflow():
    buffer = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    kernel = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)
    output, new_buffer = convolve_delay_line(buffer, kernel, torch.tensor([4.0, 8.0], dtype=torch.float64))
    expected = torch.tensor([[3.0, 4.0, 2.0], [7.0, 8.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(new_buffer, expected)
    torch.testing.assert_close(output, expected[:, 0])


def test_per_basin_kernels_broadcast():
    buffer = torch.zeros((2, 3), dtype=torch.float64)
    kernel = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=torch.float64)
    output, new_buffer = convolve_delay_line(buffer, kernel, torch.tensor([2.0, 4.0], dtype=torch.float64))
    expected = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 2.0]], dtype=torch.float64)
    torch.testing.assert_close(new_buffer, expected)
    torch.testing.assert_close(output, expected[:, 0])


def test_float64_cpu_dtype_and_device_are_preserved():
    buffer = torch.zeros(3, dtype=torch.float64)
    output, new_buffer = convolve_delay_line(
        buffer, torch.ones(3, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
    )
    assert output.dtype == torch.float64 and new_buffer.dtype == torch.float64
    assert output.device.type == "cpu" and new_buffer.device.type == "cpu"


def test_gradients_reach_all_inputs():
    buffer = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    kernel = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64, requires_grad=True)
    inflow = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    output, new_buffer = convolve_delay_line(buffer, kernel, inflow)
    (output + new_buffer.sum()).backward()
    for value in (buffer, kernel, inflow):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
