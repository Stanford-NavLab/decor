"""Spreading code class"""

# pylint: disable=not-an-iterable
from typing import Optional, Tuple, List
from numbers import Number

import numpy as np

from .gold_weil import gold_codes, weil_codes
from .util import randb, argsort
from .correlation import compute_correlation, update_correlation
import torch

from .gpu_backend import (
    build_abs_p_lut,
    codes_array_to_tensor,
    apply_flip_inplace,
    compute_delta_map,
    compute_packed_correlation,
    packed_correlation_to_numpy,
    resolve_device,
)

from . import bit_flip


class SpreadingCodes:
    """A spreading code family, with functions to compute its properties."""

    def __init__(
        self,
        num_codes: Optional[int] = None,
        code_length: Optional[int] = None,
        value: Optional[np.ndarray] = None,
        p: Number = 2,
        **kwargs,
    ) -> None:
        # use a random code if value is not provided
        if value is not None:
            num_codes, code_length = value.shape
            self.value = value
        else:
            self.value = randb(num_codes, code_length)

        self.num_codes = num_codes
        self.code_length = code_length
        self.num_correlations = (num_codes**2 + num_codes) // 2
        self.p = p

        self._correlation = kwargs.get("_correlation", None)
        self._objective = kwargs.get("_objective", None)
        self._device = kwargs.get("device", None)
        self._delta = kwargs.get("_delta", None)
        self._delta_dict = kwargs.get("_delta_dict", {})

        self._approx_delta = kwargs.get("_approx_delta", None)
        self._approx_delta_dict = kwargs.get("_approx_delta_dict", {})

        # GPU caches live alongside the NumPy views so we can reuse the same
        # ``SpreadingCodes`` instance across CPU and GPU execution paths.
        self._codes_tensor = None
        self._correlation_tensor = None
        self._delta_tensor = None
        self._lut_tensor = None

    def __repr__(self) -> str:
        return f"SpreadingCode(n={self.num_codes}, T={self.code_length}, p={self.p})"

    def __getitem__(self, idx):
        return self.value[idx]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the codes."""
        return self.value.shape

    def __len__(self) -> int:
        """Return code length."""
        return self.code_length

    def copy(self):
        """Copy only the code value."""
        return SpreadingCodes(value=self.value.copy(), p=self.p)

    def deepcopy(self):
        """Return a deepcopy of the spreading codes."""

        def copy_or_none(a):
            return a.copy() if a is not None else None

        return SpreadingCodes(
            value=self.value.copy(),
            p=self.p,
            _correlation=copy_or_none(self._correlation),
            _objective=copy_or_none(self._objective),
            _delta=copy_or_none(self._delta),
            _delta_dict=copy_or_none(self._delta_dict),
            _approx_delta=copy_or_none(self._approx_delta),
            _approx_delta_dict=copy_or_none(self._approx_delta_dict),
        )

    def use_gpu(self, device: Optional[str] = None) -> None:
        """Opt into GPU-backed helpers.

        The choice is sticky for the lifetime of the instance.  Passing
        ``device=None`` disables the GPU path and brings the instance back to
        pure CPU behaviour.
        """

        if device is None:
            self._device = None
        else:
            resolved = resolve_device(device)
            if resolved.type == "cpu":
                # We support explicit CPU selection so tests can exercise the
                # torch code path without CUDA.
                self._device = resolved
            elif resolved.type == "cuda":
                self._device = resolved
            else:
                raise ValueError(f"Unsupported torch device: {resolved}")

        # Correlation caches depend on the backend, so we clear them when the
        # device policy changes.
        self._correlation = None
        self._delta = None
        self._delta_dict.clear()
        self._codes_tensor = None
        self._correlation_tensor = None
        self._delta_tensor = None
        self._lut_tensor = None

    # ------------------------------------------------------------------

    def using_gpu(self) -> bool:
        """Return True when the instance routes updates through torch helpers."""

        return self._device is not None

    # ------------------------------------------------------------------

    def _ensure_gpu_state(self) -> None:
        """Lazily materialise GPU caches when ``use_gpu`` is active.

        The helper keeps the tensor view in sync with the NumPy caches so the
        optimiser can call into GPU kernels repeatedly without rebuilding
        temporary tensors.  Each allocation happens at most once per lifecycle
        change (for example after a flip invalidates the delta map).
        """

        if self._device is None:
            raise RuntimeError("GPU state requested without calling use_gpu")

        device = self._device

        if self._codes_tensor is None:
            # Build the device copy directly from the canonical NumPy array.
            self._codes_tensor = codes_array_to_tensor(
                self.value.astype(np.int8), device=device
            )

        if self._correlation_tensor is None:
            if self._correlation is not None:
                # Mirror the NumPy cache onto the selected device.
                self._correlation_tensor = torch.tensor(
                    self._correlation,
                    device=device,
                    dtype=torch.int64,
                )
            else:
                # Build the correlation cache entirely on device and persist it
                # so future NumPy lookups stay coherent with the GPU view.
                corr_tensor = compute_packed_correlation(self._codes_tensor)
                self._correlation_tensor = corr_tensor
                self._correlation = packed_correlation_to_numpy(corr_tensor)

        if (
            self._lut_tensor is None
            or self._lut_tensor.shape[0] != self.code_length + 1
            or self._lut_tensor.device != device
        ):
            # LUT entries depend on ``code_length`` and the selected ``p`` value.
            self._lut_tensor = build_abs_p_lut(
                self.code_length,
                float(self.p),
                device=device,
            )

    def _gpu_delta_tensor(self) -> torch.Tensor:
        """Return the GPU delta map, computing it on demand."""

        self._ensure_gpu_state()
        if self._delta_tensor is None:
            self._delta_tensor = compute_delta_map(
                self._codes_tensor,
                self._correlation_tensor,
                float(self.p),
                lut=self._lut_tensor,
            )
        return self._delta_tensor

    def delta_tensor(self) -> torch.Tensor:
        """Expose the GPU delta tensor for optimiser integration."""

        if self._delta_tensor is not None:
            return self._delta_tensor

        if self._device is None:
            raise RuntimeError("delta_tensor requires use_gpu(True)")
        return self._gpu_delta_tensor()

    def _correlation_cache(self) -> np.ndarray:
        """Return the cached integer correlation sums.

        We keep correlations as raw integer sums to save memory. The cache is
        materialised lazily so callers who never inspect correlations avoid any
        extra work.
        """

        if self._correlation is None:
            if self.num_codes is None or self.code_length is None:
                raise ValueError("Cannot compute correlation: codes not initialized")
            if self._device is None:
                self._correlation = np.zeros(
                    (self.num_correlations, self.code_length), dtype=np.int64
                )
                compute_correlation(self.value, self._correlation)
            else:
                self._ensure_gpu_state()
                if self._correlation_tensor is None:
                    self._correlation_tensor = compute_packed_correlation(
                        self._codes_tensor
                    )
                self._correlation = packed_correlation_to_numpy(
                    self._correlation_tensor
                )
        return self._correlation

    def correlation(
        self,
        nonzero_autocorrelation_peak: bool = False,
        *,
        scaled: bool = True,
        copy: bool = True,
    ) -> np.ndarray:
        """Return correlations for this code family.

        Parameters
        ----------
        nonzero_autocorrelation_peak:
            When True the zero-shift autocorrelation is set to 1.0 for scaled
            output or to ``code_length`` for raw integer output.
        scaled:
            Normalise correlations by ``code_length``. When False the raw integer
            cache is returned, which is useful for in-place updates.
        copy:
            When ``scaled`` is False, choose whether to copy the integer cache
            before returning. Internal callers that plan to update the cache must
            pass ``copy=False``.
        """

        corr_int = self._correlation_cache()

        if not scaled:
            raw = corr_int.copy() if copy or nonzero_autocorrelation_peak else corr_int
            if nonzero_autocorrelation_peak:
                if raw is corr_int:
                    raw = corr_int.copy()
                for i in range(self.num_codes):
                    idx = i * self.num_codes - i * (i + 1) // 2 + i
                    raw[idx, 0] = self.code_length
            return raw

        corr_float = corr_int.astype(np.float64)
        corr_float /= self.code_length

        if nonzero_autocorrelation_peak:
            res = corr_float.copy()
            for i in range(self.num_codes):
                idx = i * self.num_codes - i * (i + 1) // 2 + i
                res[idx, 0] = 1.0
            return res

        return corr_float

    def cross_correlation(self, i: int, j: int, full=True) -> np.ndarray:
        """Return the cross-correlation between codes i and j, at all shifts,
        both negative and positive."""
        cross_correlation = self.correlation()[
            i * self.num_codes - i * (i + 1) // 2 + j
        ].copy()
        if i == j:
            cross_correlation[0] = 1.0
        if full:
            return np.hstack((np.flip(cross_correlation[1:]), cross_correlation))
        return cross_correlation

    def objective(self, pnorm: bool = False) -> float:
        """Return the objective value of the codes."""
        if self._objective is None:
            self._objective = np.sum(np.power(np.abs(self.correlation()), self.p))

        return np.power(self._objective, 1.0 / self.p) if pnorm else self._objective

    def delta(self, i: int, j: int) -> float:
        """Return the change in objective value of flipping the given bit."""
        if self._delta is not None:
            return self._delta[i, j]

        if (i, j) not in self._delta_dict:
            if self._device is None:
                self._delta_dict[i, j] = bit_flip.delta(
                    i,
                    j,
                    self.value,
                    self.correlation(scaled=False, copy=False),
                    self.p,
                )
            else:
                # Leverage the cached GPU tensor to avoid NumPy round-trips.
                delta_tensor = self._gpu_delta_tensor()
                self._delta = delta_tensor.cpu().numpy()
                self._delta_dict.clear()
                return float(self._delta[i, j])

        return self._delta_dict[i, j]

    def deltas(self) -> None:
        """Return the change in objective value of flipping each bit."""
        if self._delta is None:
            if self._device is None:
                self._delta = np.zeros(self.shape)
                bit_flip.deltas(
                    self.value,
                    self.correlation(scaled=False, copy=False),
                    self.p,
                    self._delta,
                )
            else:
                # Keep the GPU path on device to avoid expensive transfers.
                delta_tensor = self._gpu_delta_tensor()
                self._delta = delta_tensor.cpu().numpy()

        return self._delta

    def best_delta(self) -> Tuple[int, int]:
        """Return the index of the bit with the best improvement."""
        return np.unravel_index(np.argmin(self.deltas()), self.shape)

    def top_k_delta(self, k: Optional[int] = None) -> np.ndarray:
        """Return the sorted indices of the change in objective improvement."""
        return argsort(self.deltas(), k=k)

    def flip(self, i: int, j: int) -> None:
        """Flip the sign of the given bit."""
        delta = self.delta(i, j)

        if self._device is None:
            update_correlation(self.value, i, j, self._correlation_cache())
            self.value[i, j] *= -1
        else:
            self._ensure_gpu_state()
            # Mutate the cached tensors directly so callers stay on device and
            # we avoid bouncing through NumPy between flips.  The helper updates
            # the correlation cache before toggling the code entry to preserve
            # parity with the CPU control flow.
            apply_flip_inplace(self._codes_tensor, self._correlation_tensor, i, j)

            # Keep the public NumPy view in sync lazily.  We only pull the data
            # back when downstream CPU code needs it, which happens when callers
            # access ``self.value`` or ``self._correlation`` again.  The cheap
            # copies here avoid sharing memory between torch and NumPy.
            self.value = self._codes_tensor.cpu().numpy().astype(np.int8)
            self._correlation = self._correlation_tensor.cpu().numpy().astype(np.int64)

        if self._objective is not None:
            self._objective += delta

        # update all deltas or clear cache
        if self._delta is not None:
            if self._device is None:
                bit_flip.update_deltas(
                    i,
                    j,
                    self.value,
                    self.correlation(scaled=False, copy=False),
                    self.p,
                    self._delta,
                )
            else:
                # Drop the stale CPU copy so the next call recomputes via GPU.
                self._delta = None
        else:
            self._delta_dict.clear()

        if self._device is not None:
            # Clear incremental caches because the GPU helpers will rebuild
            # them on demand using ``compute_delta_map``.
            self._delta_dict.clear()
            self._delta_tensor = None


def random_code_family(
    num_codes: int,
    code_length: int,
    p: Number = 2,
    seed: Optional[int] = None,
    balanced: bool = False,
) -> SpreadingCodes:
    """Generate a random set of spreading codes."""
    if seed is not None:
        np.random.seed(seed)
    x = SpreadingCodes(num_codes, code_length, p=p)

    if balanced:
        target_sum = 0 if code_length % 2 == 0 else 1
        for i in range(num_codes):
            curr_sum = x[i].sum()
            idx = int(np.random.choice(code_length))
            while np.abs(curr_sum) > target_sum:
                if x[i, idx] == 1 and curr_sum > target_sum:
                    x.value[i, idx] *= -1

                elif x[i, idx] == -1 and curr_sum < target_sum:
                    x.value[i, idx] *= -1

                curr_sum = x[i].sum()
                idx = (idx + 1) % code_length

            assert (
                np.abs(np.sum(x[i])) == target_sum
            ), f"Code is not balanced: {np.abs(np.sum(x[i]))} != {target_sum}"

    return x


def gold_code_family(
    num_codes: int, code_length: int, p: Number = 2, indices: Optional[List[int]] = None
) -> SpreadingCodes:
    """Generate a family of Gold codes."""
    codebook = gold_codes(code_length)

    if indices is not None:
        assert len(indices) == num_codes, "Number of indices must match num_codes"
        value = codebook[indices]
    else:
        value = codebook[np.random.choice(len(codebook), num_codes, replace=False)]

    return SpreadingCodes(value=value, p=p)


def weil_code_family(
    num_codes: int, code_length: int, p: Number = 2, indices: Optional[List[int]] = None
) -> SpreadingCodes:
    """Generate a family of Weil codes."""
    codebook = weil_codes(code_length)

    if indices is not None:
        assert len(indices) == num_codes, "Number of indices must match num_codes"
        value = codebook[indices]
    else:
        value = codebook[np.random.choice(len(codebook), num_codes, replace=False)]

    return SpreadingCodes(value=value, p=p)
