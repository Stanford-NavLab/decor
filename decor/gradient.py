"""Approximate change in objective function after a bit flip"""

# pylint: disable=not-an-iterable

import numpy as np
from numba import njit, prange

from .util import norm_grad


@njit(fastmath=True, parallel=True)
def delta(i, j, x, vec):
    """Calculate the derivative of the cross-correlation of all pairs of codes
    with respect to x[i, j]. vec is the gradient of the objective function with
    respect to the correlations."""
    num_codes, code_length = x.shape
    res = 0.0
    x_ij = x[i, j]

    # Precompute the autocorrelation index
    pair_auto = i * num_codes - i * (i + 1) // 2 + i

    # Autocorrelation contribution
    for k in prange(1, code_length):
        idx = pair_auto * code_length + k
        res += vec[idx] * (x[i, (j + k) % code_length] + x[i, (j - k) % code_length])

    # Cross-correlation contributions
    for r in prange(num_codes):
        if r != i:
            if r < i:
                pair = r * num_codes - r * (r + 1) // 2 + i
            else:
                pair = i * num_codes - i * (i + 1) // 2 + r

            for k in prange(code_length):
                idx = pair * code_length + k
                if r < i:
                    res += vec[idx] * x[r, (k + j) % code_length]
                else:
                    res += vec[idx] * x[r, (j - k) % code_length]

    return -x_ij * res


@njit(fastmath=True, parallel=True)
def deltas(x, correlations, p, dest):
    """Calculate the product of the jacobian of the cross-correlation
    function with a vector. The result is stored in the pre-allocated array
    dest, which has shape (num_codes, code_length)."""
    num_codes, code_length = x.shape
    vec = norm_grad(correlations.flatten(), p, unscaled=True)
    for i in prange(num_codes):
        for j in prange(code_length):
            dest[i, j] = delta(i, j, x, vec)


@njit(fastmath=True, parallel=True)
def update_deltas(a, b, x, correlations, p, dest):
    """Update the deltas array when the sign of x[a, b] is flipped. Assume that
    x[a,b] and the correlations have already been updated."""
    num_codes, code_length = x.shape
    x_ab = x[a, b]
    vec = norm_grad(correlations.flatten(), p, unscaled=True)

    for i in prange(num_codes):
        for j in prange(code_length):
            if i == a:
                # Directly recompute using the delta function
                dest[i, j] = delta(i, j, x, vec)

            else:
                # Update only the terms that depend on x[a, b]
                res = 0.0

                if a < i:
                    idx = a * num_codes - a * (a + 1) // 2 + i
                    cross = correlations[idx]

                    for k in prange(code_length):
                        vec_k = vec[idx * code_length + k]

                        # add new value
                        res += vec_k * x[a, (j + k) % code_length]

                        # subtract old value
                        cross_k = cross[k] - 2 * x_ab * x[i, (b - k) % code_length]
                        vec_k = norm_grad(np.array([cross_k]), p, unscaled=True)[0]
                        sign = 1 if (j + k) % code_length != b else -1
                        res -= sign * vec_k * x[a, (j + k) % code_length]

                elif a > i:
                    idx = i * num_codes - i * (i + 1) // 2 + a
                    cross = correlations[idx]

                    for k in prange(code_length):
                        vec_k = vec[idx * code_length + k]

                        # add new value
                        res += vec_k * x[a, (j - k) % code_length]

                        # subtract old value
                        cross_k = cross[k] - 2 * x_ab * x[i, (b + k) % code_length]
                        vec_k = norm_grad(np.array([cross_k]), p, unscaled=True)[0]
                        sign = 1 if (j - k) % code_length != b else -1
                        res -= sign * vec_k * x[a, (j - k) % code_length]

                dest[i, j] += -x[i, j] * res
