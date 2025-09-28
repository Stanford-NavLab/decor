"""Tests for the GPU helper module.

We intentionally exercise the helpers on CPU hardware.  The torch code path
must match the NumPy/Numba implementation exactly so that we can switch
execution backends without affecting optimisation behaviour.
"""

import numpy as np
import pytest
import torch

from decor import bit_flip, correlation
from decor.gpu_backend import (
    codes_array_to_tensor,
    compute_delta_map,
    compute_packed_correlation,
    packed_correlation_to_numpy,
    tensor_to_codes_array,
)


@pytest.mark.parametrize("num_codes, code_length", [(2, 7), (4, 11)])
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_round_trip_and_correlation(num_codes, code_length, device_type):
    """The GPU helpers must match the CPU reference exactly."""

    device = torch.device(device_type)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this host")

    # Prepare deterministic codes for reproducibility.  The SpreadingCodes
    # helper uses ±1 values, so we mirror that convention and store them as int8.
    rng = np.random.default_rng(seed=1234)
    codes = rng.choice([-1, 1], size=(num_codes, code_length)).astype(np.int8)

    tensor = codes_array_to_tensor(codes, device=device)

    # Round-trip conversion should preserve the data exactly.
    round_trip = tensor_to_codes_array(tensor)
    assert np.array_equal(round_trip, codes)

    # Compute correlations via the new path.
    torch_corr = compute_packed_correlation(tensor)
    torch_corr_np = packed_correlation_to_numpy(torch_corr)

    # CPU reference as implemented today.
    reference = np.zeros_like(torch_corr_np)
    correlation.compute_correlation(codes.astype(np.int64), reference)

    assert np.array_equal(torch_corr_np, reference)


@pytest.mark.parametrize("num_codes, code_length, p", [(2, 7, 2.0), (3, 9, 3.0)])
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_delta_map_parity(num_codes, code_length, p, device_type):
    """Torch delta map must match the NumPy baseline exactly."""

    device = torch.device(device_type)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this host")

    rng = np.random.default_rng(seed=5678)
    codes = rng.choice([-1, 1], size=(num_codes, code_length)).astype(np.int8)

    tensor_codes = codes_array_to_tensor(codes, device=device)
    torch_corr = compute_packed_correlation(tensor_codes)

    delta_map = compute_delta_map(tensor_codes, torch_corr, p)

    # CPU reference uses the established NumPy/Numba helpers.
    corr_np = np.zeros_like(packed_correlation_to_numpy(torch_corr))
    correlation.compute_correlation(codes.astype(np.int64), corr_np)
    dest = np.zeros_like(codes, dtype=np.float64)
    bit_flip.deltas(codes, corr_np, p, dest)

    assert np.allclose(delta_map.cpu().numpy(), dest)
