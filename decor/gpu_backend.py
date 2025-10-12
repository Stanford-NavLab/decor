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

# Triton is optional during CPU-only development, but the GPU kernels require it.
# We import lazily so the remainder of the module still loads when Triton is missing.
try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-GPU builders
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _update_corr_kernel(
        codes_ptr,
        correlations_ptr,
        rows_ptr,
        packed_ptr,
        start_ptr,
        direction_ptr,
        num_rows,
        code_length,
        column_index,
        x_ij,
        codes_stride_n,
        codes_stride_t,
        corr_stride_row,
        corr_stride_t,
        BLOCK_N: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Triton program that streams packed correlation rows."""

        pid_rows = tl.program_id(0)
        pid_time = tl.program_id(1)

        row_offsets = pid_rows * BLOCK_N + tl.arange(0, BLOCK_N)
        time_offsets = pid_time * BLOCK_T + tl.arange(0, BLOCK_T)

        delta_factor = tl.full((), -2, dtype=tl.int32) * x_ij

        for lane in tl.static_range(BLOCK_N):
            row_offset = row_offsets[lane]
            lane_active = row_offset < num_rows
            if not lane_active:
                continue

            row_index = tl.load(rows_ptr + row_offset)
            start_offset = tl.load(start_ptr + row_offset)
            direction = tl.load(direction_ptr + row_offset)
            corr_index = tl.load(packed_ptr + row_offset)

            k_indices = time_offsets + start_offset
            mask_valid = k_indices < code_length

            corr_ptr_lane = (
                correlations_ptr
                + corr_index * corr_stride_row
                + k_indices * corr_stride_t
            )
            current_vals = tl.load(corr_ptr_lane, mask=mask_valid, other=0)

            minus_idx = column_index - k_indices
            minus_idx = minus_idx + code_length * (minus_idx < 0)
            plus_idx = column_index + k_indices
            plus_idx = plus_idx - code_length * (plus_idx >= code_length)

            neighbour_idx = plus_idx + (minus_idx - plus_idx) * direction

            code_ptr_lane = (
                codes_ptr + row_index * codes_stride_n + neighbour_idx * codes_stride_t
            )
            neighbour_vals = tl.load(code_ptr_lane, mask=mask_valid, other=0).to(
                tl.int32
            )

            delta_vals = delta_factor * neighbour_vals
            updated_vals = current_vals + delta_vals.to(current_vals.dtype)
            tl.store(corr_ptr_lane, updated_vals, mask=mask_valid)

    @triton.jit
    def _deltas_autocorr_kernel(
        codes_ptr,
        correlations_ptr,
        lut_ptr,
        delta_ptr,
        row_index,
        corr_index,
        code_length,
        max_shift,
        codes_stride_n,
        codes_stride_t,
        corr_stride_row,
        corr_stride_t,
        shift_start,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Accumulate auto-correlation delta terms for one chunk of shifts."""

        pid_j = tl.program_id(0)

        j_offsets = pid_j * BLOCK_T + tl.arange(0, BLOCK_T)
        j_mask = j_offsets < code_length

        code_row_ptr = codes_ptr + row_index * codes_stride_n
        corr_row_ptr = correlations_ptr + corr_index * corr_stride_row

        x_vals = tl.load(
            code_row_ptr + j_offsets * codes_stride_t, mask=j_mask, other=0
        ).to(tl.int32)
        two_x_vals = 2 * x_vals

        delta_tile = tl.load(delta_ptr + j_offsets, mask=j_mask, other=0.0).to(
            tl.float64
        )

        shift_indices = shift_start + 1 + tl.arange(0, BLOCK_K)
        mask_shift = shift_indices <= max_shift

        for lane in tl.static_range(0, BLOCK_K):
            lane_active = mask_shift[lane]
            lane_active_f = lane_active.to(tl.float64)
            if not lane_active:
                continue

            shift_value = shift_indices[lane]
            prev_val = tl.load(corr_row_ptr + shift_value * corr_stride_t)
            prev_abs = tl.abs(prev_val)
            prev_abs = tl.where(prev_abs > code_length, code_length, prev_abs)
            prev_idx = prev_abs.to(tl.int32)
            prev_term = tl.load(lut_ptr + prev_idx)

            plus_idx = j_offsets + shift_value
            plus_idx = plus_idx - code_length * (plus_idx >= code_length)
            minus_idx = j_offsets - shift_value
            minus_idx = minus_idx + code_length * (minus_idx < 0)

            plus_vals = tl.load(
                code_row_ptr + plus_idx * codes_stride_t, mask=j_mask, other=0
            ).to(tl.int32)
            minus_vals = tl.load(
                code_row_ptr + minus_idx * codes_stride_t, mask=j_mask, other=0
            ).to(tl.int32)

            neighbour_sum = plus_vals + minus_vals
            delta_contrib = two_x_vals * neighbour_sum

            new_val = prev_val - delta_contrib.to(prev_val.dtype)
            new_abs = tl.abs(new_val)
            new_abs = tl.where(new_abs > code_length, code_length, new_abs)
            new_idx = new_abs.to(tl.int32)
            new_term = tl.load(lut_ptr + new_idx)

            delta_tile += lane_active_f * (new_term - prev_term)

        tl.store(delta_ptr + j_offsets, delta_tile, mask=j_mask)

    @triton.jit
    def _deltas_cross_kernel(
        codes_ptr,
        correlations_ptr,
        lut_ptr,
        delta_ptr,
        row_index,
        other_index,
        corr_index,
        code_length,
        max_shift,
        direction_flag,
        codes_stride_n,
        codes_stride_t,
        corr_stride_row,
        corr_stride_t,
        shift_start,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Accumulate cross-correlation delta terms for one neighbour chunk."""

        pid_j = tl.program_id(0)

        j_offsets = pid_j * BLOCK_T + tl.arange(0, BLOCK_T)
        j_mask = j_offsets < code_length

        row_ptr = codes_ptr + row_index * codes_stride_n
        other_ptr = codes_ptr + other_index * codes_stride_n
        corr_ptr = correlations_ptr + corr_index * corr_stride_row

        x_vals = tl.load(row_ptr + j_offsets * codes_stride_t, mask=j_mask, other=0).to(
            tl.int32
        )
        two_x_vals = 2 * x_vals

        delta_tile = tl.load(delta_ptr + j_offsets, mask=j_mask, other=0.0).to(
            tl.float64
        )

        shift_indices = shift_start + tl.arange(0, BLOCK_K)
        mask_shift = shift_indices <= max_shift

        for lane in tl.static_range(0, BLOCK_K):
            lane_active = mask_shift[lane]
            lane_active_f = lane_active.to(tl.float64)
            if not lane_active:
                continue

            shift_value = shift_indices[lane]
            prev_val = tl.load(corr_ptr + shift_value * corr_stride_t)
            prev_abs = tl.abs(prev_val)
            prev_abs = tl.where(prev_abs > code_length, code_length, prev_abs)
            prev_idx = prev_abs.to(tl.int32)
            prev_term = tl.load(lut_ptr + prev_idx)

            if direction_flag:
                neighbour_idx = j_offsets - shift_value
                neighbour_idx = neighbour_idx + code_length * (neighbour_idx < 0)
            else:
                neighbour_idx = j_offsets + shift_value
                neighbour_idx = neighbour_idx - code_length * (
                    neighbour_idx >= code_length
                )

            neighbour_vals = tl.load(
                other_ptr + neighbour_idx * codes_stride_t, mask=j_mask, other=0
            ).to(tl.int32)

            delta_contrib = two_x_vals * neighbour_vals

            new_val = prev_val - delta_contrib.to(prev_val.dtype)
            new_abs = tl.abs(new_val)
            new_abs = tl.where(new_abs > code_length, code_length, new_abs)
            new_idx = new_abs.to(tl.int32)
            new_term = tl.load(lut_ptr + new_idx)

            delta_tile += lane_active_f * (new_term - prev_term)

        tl.store(delta_ptr + j_offsets, delta_tile, mask=j_mask)


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


def _ensure_triton_available() -> None:
    """Raise a helpful error when Triton support is missing.

    We defer the import check until the kernels are used so CPU-focused tests can
    still exercise the torch-only fallbacks.  Callers receive a short hint that
    explains how to enable GPU support if Triton is not present.
    """

    if triton is None or tl is None:
        raise RuntimeError(
            "Triton support is required for the GPU kernels. Install the 'triton'"
            " package and ensure a CUDA-compatible device is available."
        )


def triton_update_corr_one_flip(
    codes: torch.Tensor,
    correlations: torch.Tensor,
    i: int,
    j: int,
    *,
    block_t: int,
    block_n: int,
) -> None:
    """Update packed correlations after flipping ``codes[i, j]`` using Triton.

    The kernel keeps the work entirely on device and streams the affected rows in
    wide tiles.  This mirrors :func:`update_corr_one_flip` but avoids Python loops
    so launch overhead stays small even for large ``code_length`` values.
    """

    _ensure_triton_available()

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

    if codes.device.type != "cuda" or correlations.device != codes.device:
        raise ValueError(
            "Triton kernels require codes and correlations on the same CUDA device"
        )

    if block_t <= 0 or block_n <= 0:
        raise ValueError("block_t and block_n must be positive")

    if block_t % 128 != 0 or block_n % 4 != 0:
        raise ValueError(
            "block_t must be a multiple of 128 and block_n a multiple of 4"
        )

    codes_int = codes.to(torch.int32)
    correlations_int = correlations

    rows = torch.empty((2 * num_codes - 1,), dtype=torch.int32, device=codes.device)
    packed = torch.empty_like(rows)
    start = torch.empty_like(rows)
    direction = torch.empty_like(rows)

    offset = 0
    for r in range(i, num_codes):
        rows[offset] = r
        packed[offset] = _packed_index(i, r, num_codes)
        start[offset] = 1 if r == i else 0
        direction[offset] = 1  # use (j - k) indexing
        offset += 1

    for r in range(0, i + 1):
        rows[offset] = r
        packed[offset] = _packed_index(r, i, num_codes)
        start[offset] = 1 if r == i else 0
        direction[offset] = 0  # use (k + j) indexing
        offset += 1

    num_rows = offset
    rows = rows[:num_rows]
    packed = packed[:num_rows]
    start = start[:num_rows]
    direction = direction[:num_rows]

    grid = (
        (num_rows + block_n - 1) // block_n,
        (code_length + block_t - 1) // block_t,
    )

    x_ij = codes_int[i, j].to(torch.int32)

    with torch.cuda.device(codes.device):
        _update_corr_kernel[grid](
            codes_int,
            correlations_int,
            rows,
            packed,
            start,
            direction,
            num_rows,
            code_length,
            j,
            x_ij,
            codes_int.stride(0),
            codes_int.stride(1),
            correlations_int.stride(0),
            correlations_int.stride(1),
            BLOCK_N=block_n,
            BLOCK_T=block_t,
        )


def triton_deltas_row(
    codes: torch.Tensor,
    correlations: torch.Tensor,
    lut: torch.Tensor,
    row_index: int,
    *,
    block_t: int,
    block_k: int,
) -> torch.Tensor:
    """Recompute a single delta row on device using Triton kernels."""

    _ensure_triton_available()

    if codes.dtype != torch.int8:
        raise ValueError("codes tensor must have dtype torch.int8")
    if correlations.dtype != torch.int64:
        raise ValueError("correlations tensor must have dtype torch.int64")
    if lut.dtype != torch.float64:
        raise ValueError("lookup table must have dtype torch.float64")

    if codes.dim() != 2 or correlations.dim() != 2:
        raise ValueError("codes and correlations tensors must be two-dimensional")

    if (
        codes.device.type != "cuda"
        or correlations.device != codes.device
        or lut.device != codes.device
    ):
        raise ValueError(
            "codes, correlations, and lut must share a CUDA device for Triton execution"
        )

    num_codes, code_length = codes.shape
    if not (0 <= row_index < num_codes):
        raise IndexError("row_index falls outside the code tensor")

    expected_rows = (num_codes * num_codes + num_codes) // 2
    if correlations.shape[0] != expected_rows or correlations.shape[1] != code_length:
        raise ValueError("correlations tensor has an unexpected packed shape")

    if lut.shape[0] != code_length + 1:
        raise ValueError("lookup table must match code_length + 1 entries")

    if block_t <= 0 or block_k <= 0:
        raise ValueError("block_t and block_k must be positive")

    if block_t % 128 != 0 or block_k % 8 != 0:
        raise ValueError(
            "block_t must be a multiple of 128 and block_k a multiple of 8"
        )

    codes_int = codes.to(torch.int32)
    correlations_int = correlations

    delta_row = torch.zeros(code_length, dtype=torch.float64, device=codes.device)

    auto_index = _packed_index(row_index, row_index, num_codes)
    auto_max_shift = code_length - 1

    grid_t = (code_length + block_t - 1) // block_t
    grid_k_auto = (auto_max_shift + block_k - 1) // block_k

    for chunk in range(grid_k_auto):
        shift_start = chunk * block_k
        with torch.cuda.device(codes.device):
            _deltas_autocorr_kernel[(grid_t,)](
                codes_int,
                correlations_int,
                lut,
                delta_row,
                row_index,
                auto_index,
                code_length,
                auto_max_shift,
                codes_int.stride(0),
                codes_int.stride(1),
                correlations_int.stride(0),
                correlations_int.stride(1),
                shift_start,
                BLOCK_T=block_t,
                BLOCK_K=block_k,
            )

    for other_index in range(num_codes):
        if other_index == row_index:
            continue

        if other_index < row_index:
            corr_index = _packed_index(other_index, row_index, num_codes)
            direction_flag = 0
        else:
            corr_index = _packed_index(row_index, other_index, num_codes)
            direction_flag = 1

        max_shift = code_length - 1
        grid_k = (max_shift + block_k - 1) // block_k

        for chunk in range(grid_k):
            shift_start = chunk * block_k
            with torch.cuda.device(codes.device):
                _deltas_cross_kernel[(grid_t,)](
                    codes_int,
                    correlations_int,
                    lut,
                    delta_row,
                    row_index,
                    other_index,
                    corr_index,
                    code_length,
                    max_shift,
                    direction_flag,
                    codes_int.stride(0),
                    codes_int.stride(1),
                    correlations_int.stride(0),
                    correlations_int.stride(1),
                    shift_start,
                    BLOCK_T=block_t,
                    BLOCK_K=block_k,
                )

    return delta_row
