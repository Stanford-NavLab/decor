"""Spreading code class"""

# pylint: disable=not-an-iterable
from typing import Optional, Tuple, List
from numbers import Number

import numpy as np

from .gold_weil import gold_codes, weil_codes
from .util import randb, argsort
from .correlation import compute_correlation, update_correlation

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
        self._delta = kwargs.get("_delta", None)
        self._delta_dict = kwargs.get("_delta_dict", {})

        self._approx_delta = kwargs.get("_approx_delta", None)
        self._approx_delta_dict = kwargs.get("_approx_delta_dict", {})

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

    def correlation(self, nonzero_autocorrelation_peak: bool = False) -> np.ndarray:
        """Return the correlation matrix of the codes. The zero-shift
        autocorrelation is by default set to zero."""
        if self._correlation is None:
            self._correlation = np.zeros((self.num_correlations, self.code_length))
            compute_correlation(self.value, self._correlation)

        if nonzero_autocorrelation_peak:
            res = self._correlation.copy()
            for i in range(self.num_codes):
                idx = i * self.num_codes - i * (i + 1) // 2 + i
                res[idx, 0] = self.code_length
            return res

        return self._correlation

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
            self._delta_dict[i, j] = bit_flip.delta(
                i, j, self.value, self.correlation(), self.p
            )

        return self._delta_dict[i, j]

    def deltas(self) -> None:
        """Return the change in objective value of flipping each bit."""
        if self._delta is None:
            self._delta = np.zeros(self.shape)
            bit_flip.deltas(self.value, self.correlation(), self.p, self._delta)

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

        update_correlation(self.value, i, j, self._correlation)
        self.value[i, j] *= -1

        if self._objective is not None:
            self._objective += delta

        # update all deltas or clear cache
        if self._delta is not None:
            bit_flip.update_deltas(
                i,
                j,
                self.value,
                self.correlation(),
                self.p,
                self._delta,
            )
        else:
            self._delta_dict.clear()


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
