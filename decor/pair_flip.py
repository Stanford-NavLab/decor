"""Compute the change in the objective function when the signs of two bits are
swapped. TODO: fixme
"""

# pylint: disable=not-an-iterable
from typing import Optional, Tuple, List
from numbers import Number

import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def delta(i, j, j2, x, correlations, obj_val, p):
    """Calculate the change in the objective function when the signs of x[i,j]
    and x[i,j2] are flipped."""
    num_codes, length = x.shape
    res = 0.0

    # change in autocorrelation
    idx = i * num_codes - i * (i + 1) // 2 + i
    for k in prange(1, length):
        prev = correlations[idx, k]
        diff = 0
        if (j + k) % length != j2:
            diff = diff - 2 * x[i, j] * x[i, (j + k) % length]
        if (j - k) % length != j2:
            diff = diff - 2 * x[i, j] * x[i, (j - k) % length]
        if (j2 + k) % length != j:
            diff = diff - 2 * x[i, j2] * x[i, (j2 + k) % length]
        if (j2 - k) % length != j:
            diff = diff - 2 * x[i, j2] * x[i, (j2 - k) % length]
        res += np.abs(prev + diff) ** p - np.abs(prev) ** p

    for r in prange(num_codes):
        if r < i:
            # change in right side cross-correlation
            idx = r * num_codes - r * (r + 1) // 2 + i
            for k in prange(length):
                prev = correlations[idx, k]
                diff = (
                    -2 * x[i, j] * x[r, (k + j) % length]
                    - 2 * x[i, j2] * x[r, (k + j2) % length]
                )
                res += np.abs(prev + diff) ** p - np.abs(prev) ** p

        elif r > i:
            # change in left side cross-correlation
            idx = i * num_codes - i * (i + 1) // 2 + r
            for k in prange(length):
                prev = correlations[idx, k]
                diff = (
                    -2 * x[i, j] * x[r, (j - k) % length]
                    - 2 * x[i, j2] * x[r, (j2 - k) % length]
                )
                res += np.abs(prev + diff) ** p - np.abs(prev) ** p

    return np.power(res + np.power(obj_val, p), 1.0 / p) - obj_val


@njit(fastmath=True, parallel=True)
def best_delta(x, correlations, obj_val, p, indices1, indices2):
    """Calculate the change in the objective function when the sign of each bit
    is flipped."""
    num_codes, length = x.shape

    out = np.zeros((len(indices1)))
    # pylint: disable=consider-using-enumerate
    for sel in prange(len(indices1)):
        i, j = indices1[sel]
        _, j2 = indices2[sel]

        res = 0.0

        # change in autocorrelation
        idx = i * num_codes - i * (i + 1) // 2 + i
        for k in prange(1, length):
            prev = correlations[idx, k]
            diff = 0
            if (j + k) % length != j2:
                diff = diff - 2 * x[i, j] * x[i, (j + k) % length]
            if (j - k) % length != j2:
                diff = diff - 2 * x[i, j] * x[i, (j - k) % length]
            if (j2 + k) % length != j:
                diff = diff - 2 * x[i, j2] * x[i, (j2 + k) % length]
            if (j2 - k) % length != j:
                diff = diff - 2 * x[i, j2] * x[i, (j2 - k) % length]
            res += np.abs(prev + diff) ** p - np.abs(prev) ** p

        for r in prange(num_codes):
            if r < i:
                # change in right side cross-correlation
                idx = r * num_codes - r * (r + 1) // 2 + i
                for k in prange(length):
                    prev = correlations[idx, k]
                    diff = (
                        -2 * x[i, j] * x[r, (k + j) % length]
                        - 2 * x[i, j2] * x[r, (k + j2) % length]
                    )
                    res += np.abs(prev + diff) ** p - np.abs(prev) ** p

            elif r > i:
                # change in left side cross-correlation
                idx = i * num_codes - i * (i + 1) // 2 + r
                for k in prange(length):
                    prev = correlations[idx, k]
                    diff = (
                        -2 * x[i, j] * x[r, (j - k) % length]
                        - 2 * x[i, j2] * x[r, (j2 - k) % length]
                    )
                    res += np.abs(prev + diff) ** p - np.abs(prev) ** p

        out[sel] = np.power(res + np.power(obj_val, p), 1.0 / p) - obj_val

    best = np.argmin(out)
    return indices1[best], indices2[best]
