"""PyTorch helpers for GPU execution.

This module introduces the first building block of the GPU plan.  It covers
basic tensor transfers and a parity-checked correlation routine that mirrors
``decor.correlation.compute_correlation``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.fft import fft as torch_fft, ifft as torch_ifft


# Cache lookup tables per (device, length, p, dtype).  The cache keeps small
# tensors on device so callers can reuse them between delta computations.
_LUT_CACHE: dict[Tuple[str, int, int, float, torch.dtype], torch.Tensor] = {}


def _packed_index(i: int, j: int, num_codes: int) -> int:
    """Return the packed upper-triangular index for (i, j).

    The GPU helpers mirror the CPU cache layout.  We keep this helper local so
    the formula stays in one place and we can reference it wherever we need to
    touch packed correlation rows.
    """

    if j < i:
        raise ValueError("Packed index helper expects i <= j")

    return i * num_codes - i * (i + 1) // 2 + j


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


def build_abs_p_lut(
    code_length: int,
    p: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Precompute |s / T|**p for s in [0, T].

    The delta map stores objective differences that depend on the absolute
    value of correlation sums divided by ``code_length``.  We build the lookup
    table on device so kernels can reuse it without round-tripping through the
    host.  Callers can cache the result and pass it to ``compute_delta_map``.
    """

    if code_length <= 0:
        raise ValueError("code_length must be positive")

    if device is None:
        device = resolve_device()

    device_index = -1 if device.index is None else int(device.index)
    cache_key = (device.type, device_index, int(code_length), float(p), dtype)

    cached = _LUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Values run from 0 to +T because correlation sums stay within ±T.  We build
    # the tensor directly on the requested device to avoid an extra copy.
    indices = torch.arange(code_length + 1, device=device, dtype=dtype)
    normalised = indices / float(code_length)
    lut = torch.pow(normalised, p)
    _LUT_CACHE[cache_key] = lut
    return lut


def compute_delta_map(
    codes: torch.Tensor,
    correlations: torch.Tensor,
    p: float,
    *,
    lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the full delta map on the selected torch device.

    This prototype mirrors ``decor.bit_flip.deltas``.  It keeps the tight loops
    in Python for clarity while relying on torch tensor ops to execute the heavy
    arithmetic on the device.  The implementation favours parity and readable
    logic over ultimate performance so that we can validate the math before
    moving to Triton kernels.
    """

    if codes.dtype != torch.int8:
        raise ValueError("codes tensor must have dtype torch.int8")

    if correlations.dtype != torch.int64:
        raise ValueError("correlations tensor must have dtype torch.int64")

    if codes.dim() != 2:
        raise ValueError("codes tensor must be two-dimensional")

    if correlations.dim() != 2:
        raise ValueError("correlations tensor must be two-dimensional")

    num_codes, code_length = codes.shape
    expected_rows = (num_codes * num_codes + num_codes) // 2
    if correlations.shape[0] != expected_rows:
        raise ValueError("correlations tensor has an unexpected packed shape")

    if correlations.shape[1] != code_length:
        raise ValueError("correlations tensor must use the same length as codes")

    device = codes.device
    if lut is None:
        lut = build_abs_p_lut(code_length, p, device=device)

    # The lookup table stores ``|s / T|**p`` so we only need integer indices.
    if lut.device != device:
        raise ValueError("lookup table must live on the same device as codes")

    if lut.dtype != torch.float64:
        # We use float64 to mirror the CPU implementation.  Keeping the check
        # explicit prevents silent dtype drift later.
        raise ValueError("lookup table must have dtype torch.float64")

    # We snapshot the integer view of the codes before mutating any entry so the
    # correlation adjustments use the pre-flip value, just like the CPU helper.
    codes_int = codes.to(torch.int64)
    correlations_int = correlations

    delta_map = torch.zeros(
        (num_codes, code_length), dtype=torch.float64, device=device
    )

    max_index = lut.shape[0] - 1
    # Reuse shift tensors to keep the inner loops simple and to avoid repeated
    # allocations inside the flip loops.
    auto_shifts = (
        torch.arange(1, code_length, device=device, dtype=torch.int64)
        if code_length > 1
        else None
    )
    full_shifts = torch.arange(code_length, device=device, dtype=torch.int64)

    for i in range(num_codes):
        code_i = codes_int[i]
        delta_row = delta_map[i]

        auto_idx = _packed_index(i, i, num_codes)
        auto_row = correlations_int[auto_idx]

        for j in range(code_length):
            x_ij = code_i[j]
            two_x_ij = 2 * x_ij

            delta_acc = torch.zeros((), device=device, dtype=torch.float64)

            if auto_shifts is not None:
                plus_idx = (j + auto_shifts) % code_length
                minus_idx = (j - auto_shifts) % code_length
                neighbours_plus = code_i[plus_idx]
                neighbours_minus = code_i[minus_idx]

                prev_vals = auto_row[auto_shifts]
                new_vals = prev_vals - two_x_ij * (neighbours_plus + neighbours_minus)

                prev_terms = lut[torch.clamp(prev_vals.abs(), max=max_index).long()]
                new_terms = lut[torch.clamp(new_vals.abs(), max=max_index).long()]
                delta_acc += (new_terms - prev_terms).sum()

            for r in range(num_codes):
                if r == i:
                    continue

                if r < i:
                    corr_idx = _packed_index(r, i, num_codes)
                else:
                    corr_idx = _packed_index(i, r, num_codes)

                corr_row = correlations_int[corr_idx]
                prev_vals = corr_row

                if r < i:
                    indices = (j + full_shifts) % code_length
                    neighbour_vals = codes_int[r][indices]
                else:
                    indices = (j - full_shifts) % code_length
                    neighbour_vals = codes_int[r][indices]

                new_vals = prev_vals - two_x_ij * neighbour_vals

                prev_terms = lut[torch.clamp(prev_vals.abs(), max=max_index).long()]
                new_terms = lut[torch.clamp(new_vals.abs(), max=max_index).long()]
                delta_acc += (new_terms - prev_terms).sum()

            delta_row[j] = delta_acc

    return delta_map


def update_corr_one_flip(
    codes: torch.Tensor,
    correlations: torch.Tensor,
    i: int,
    j: int,
) -> None:
    """Update packed correlations after flipping ``codes[i, j]`` in-place.

    We mirror ``decor.correlation.update_correlation`` but keep the arithmetic on
    device tensors so callers avoid host round-trips.  Only integer work is
    required, therefore we stay in ``torch.int64`` throughout to match the cache
    layout used elsewhere in the module.
    """

    if codes.dtype != torch.int8:
        raise ValueError("codes tensor must have dtype torch.int8")

    if correlations.dtype != torch.int64:
        raise ValueError("correlations tensor must have dtype torch.int64")

    if codes.dim() != 2 or correlations.dim() != 2:
        raise ValueError("codes and correlations tensors must be two-dimensional")

    num_codes, code_length = codes.shape
    if not (0 <= i < num_codes and 0 <= j < code_length):
        raise IndexError("flip indices fall outside the code tensor")

    expected_rows = (num_codes * num_codes + num_codes) // 2
    if correlations.shape[0] != expected_rows or correlations.shape[1] != code_length:
        raise ValueError("correlations tensor has an unexpected packed shape")

    codes_int = codes.to(torch.int64)
    x_ij = codes_int[i, j]

    # First pass: rows where i <= r.
    for r in range(i, num_codes):
        idx = _packed_index(i, r, num_codes)
        for k in range(0 if i != r else 1, code_length):
            neighbour = codes_int[r, (j - k) % code_length]
            correlations[idx, k] += -2 * x_ij * neighbour

    # Second pass: rows where i >= r.
    for r in range(0, i + 1):
        idx = _packed_index(r, i, num_codes)
        for k in range(0 if i != r else 1, code_length):
            neighbour = codes_int[r, (k + j) % code_length]
            correlations[idx, k] += -2 * x_ij * neighbour


def apply_flip_inplace(
    codes: torch.Tensor,
    correlations: torch.Tensor,
    i: int,
    j: int,
) -> None:
    """Flip ``codes[i, j]`` and update the packed correlations on device.

    Callers must ensure the tensors live on the same device and reuse the GPU
    layout described in ``compute_packed_correlation``.  The helper first updates
    the correlation cache using the pre-flip value and only then flips the bit to
    keep the operations consistent with the CPU reference implementation.
    """

    update_corr_one_flip(codes, correlations, i, j)
    codes[i, j] = torch.neg(codes[i, j])
