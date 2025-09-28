"""PyTorch helpers for GPU execution.

This module introduces the first building block of the GPU plan.  It covers
basic tensor transfers and a parity-checked correlation routine that mirrors
``decor.correlation.compute_correlation``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.fft import fft as torch_fft, ifft as torch_ifft


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Pick a torch device.

    Parameters
    ----------
    preferred:
        A string like ``"cuda"`` or ``"cpu"``.  When ``None`` we pick ``cuda`` if
        it is available, otherwise ``cpu``.
    """

    if preferred is not None:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def codes_array_to_tensor(
    codes: np.ndarray, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Move a code family from NumPy to a torch tensor.

    We keep the values in ``int8`` form because the codes are ±1.  Kernels can
    cast to floating point when they need FFT support.

    ``SpreadingCodes`` users can call this helper with ``codes.value`` to avoid
    coupling the module back to the high-level class.
    """

    if codes.dtype != np.int8:
        raise ValueError("codes must use dtype int8 for a lossless transfer")

    if device is None:
        device = resolve_device()

    return torch.from_numpy(codes).to(device=device, dtype=torch.int8)


def tensor_to_codes_array(codes: torch.Tensor) -> np.ndarray:
    """Bring a code tensor back to NumPy."""

    # We copy here to avoid aliasing the torch storage.  This keeps the
    # ``SpreadingCodes`` cache semantics unchanged.
    return codes.to(device="cpu", dtype=torch.int8).cpu().numpy().copy()


def compute_packed_correlation(codes: torch.Tensor) -> torch.Tensor:
    """Compute auto- and cross-correlations with torch FFTs.

    This mirrors ``decor.correlation.compute_correlation`` so that downstream
    code can reuse the packed upper-triangular layout.
    """

    if codes.dtype != torch.int8:
        raise ValueError("codes tensor must have dtype torch.int8")

    if codes.dim() != 2:
        raise ValueError("codes tensor must be two-dimensional")

    num_codes, code_length = codes.shape
    if num_codes <= 0 or code_length <= 0:
        raise ValueError("codes tensor must not be empty")

    # Cast to float for the FFT.  We use float32 to match torch's fast path.
    float_codes = codes.to(dtype=torch.float32)
    # ``torch_fft`` works on complex64 when the input is float32.
    # pylint struggles with torch's lazily-populated namespaces.  We pin the
    # callable import above, yet the type checker still flags ``not-callable``.
    # Disable the warning locally to keep noise out of CI.
    spectra = torch_fft(float_codes, dim=1)  # pylint: disable=not-callable

    num_corr = (num_codes * num_codes + num_codes) // 2
    # Allocate integer output that mirrors the CPU cache layout.
    correlations = torch.empty(
        (num_corr, code_length), dtype=torch.int64, device=codes.device
    )

    # Iterate over the packed upper-triangular pairs.  This mirrors the CPU
    # nested loop so that index formulas stay identical.  The kernel is still
    # vectorised inside each iteration because the FFT work happens on device.
    idx = 0
    for i in range(num_codes):
        spectrum_i = spectra[i]
        for j in range(i, num_codes):
            # Multiply spectrum_i by the conjugate spectrum.  This works for
            # both GPU and CPU tensors.
            product = spectrum_i * torch.conj(spectra[j])
            seq = torch_ifft(product, dim=0).real  # pylint: disable=not-callable

            # Round to the nearest integer.  The FFT of ±1 sequences is exact,
            # yet we keep the rounding to guard against floating-point noise.
            seq = torch.round(seq).to(torch.int64)

            if i == j:
                # The CPU path forces the zero shift to zero.  We do the same
                # so that caches match bit-for-bit.
                seq[0] = 0

            correlations[idx] = seq
            idx += 1

    return correlations


def torch_random_code_family(
    num_codes: int,
    code_length: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate a random code family directly on the target device.

    This mirrors ``decor.util.randb`` but avoids an intermediate NumPy array,
    which makes it a useful helper once we start running the optimiser entirely
    on GPU.  The function is intentionally simple because callers can seed the
    global RNG before invoking it.
    """

    if device is None:
        device = resolve_device()

    # We generate ±1 values using torch's random utilities.  The baseline randb
    # chooses from {-1, 1} with equal probability.
    torch.manual_seed(torch.initial_seed())

    values = torch.randint(
        0, 2, (num_codes, code_length), device=device, dtype=torch.int8
    )
    values = values * 2 - 1
    return values


def packed_correlation_to_numpy(correlations: torch.Tensor) -> np.ndarray:
    """Convert a packed correlation tensor back to NumPy ints."""

    if correlations.dtype != torch.int64:
        raise ValueError("correlations tensor must have dtype torch.int64")

    # ``cpu()`` followed by ``numpy()`` materialises a view.  We copy so that
    # callers can mutate the result without touching torch-managed memory.
    return correlations.to(device="cpu", dtype=torch.int64).cpu().numpy().copy()
