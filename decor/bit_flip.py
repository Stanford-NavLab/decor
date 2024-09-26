"""Functions for calculating the change in the objective function when flipping a bit."""

# pylint: disable=not-an-iterable
import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def delta(i, j, x, correlations, p):
    """Calculate the change in the objective function when the sign of the given
    bit is flipped."""
    num_codes, length = x.shape
    res = 0.0
    x_ij = x[i, j]

    # Precompute indices and related values for cache locality
    idx_auto = i * num_codes - i * (i + 1) // 2 + i
    correlations_auto = correlations[idx_auto]

    # change in autocorrelation
    for k in prange(1, length):
        x_ik_plus = x[i, (j + k) % length]
        x_ik_minus = x[i, (j - k) % length]
        prev = correlations_auto[k]
        new = prev - 2 * x_ij * (x_ik_plus + x_ik_minus) / length
        res += np.abs(new) ** p - np.abs(prev) ** p

    for r in prange(num_codes):
        if r != i:
            # Precompute correlation index to improve memory access pattern
            if r < i:
                idx_corr = r * num_codes - r * (r + 1) // 2 + i
            else:
                idx_corr = i * num_codes - i * (i + 1) // 2 + r

            correlations_cross = correlations[idx_corr]

            for k in prange(length):
                prev = correlations_cross[k]
                if r < i:
                    new = prev - 2 * x_ij * x[r, (k + j) % length] / length
                else:
                    new = prev - 2 * x_ij * x[r, (j - k) % length] / length
                res += np.abs(new) ** p - np.abs(prev) ** p

    return res


@njit(fastmath=True, parallel=True)
def deltas(x, correlations, p, dest):
    """Calculate the change in the objective function when the sign of each bit
    is flipped."""
    num_codes, length = x.shape
    for i in prange(num_codes):
        for j in prange(length):
            dest[i, j] = delta(i, j, x, correlations, p)


@njit(fastmath=True, parallel=True)
def update_deltas(a, b, x, correlations, p, dest):
    """Update deltas after flipping x[a, b]. Assume that x[a, b] and
    the correlations have already been updated."""
    num_codes, length = x.shape
    x_ab = x[a, b]

    for i in prange(num_codes):
        for j in prange(length):
            if i == a:
                # Directly recompute using the delta function
                dest[i, j] = delta(i, j, x, correlations, p)
            else:
                # Update only the terms that depend on x[a, b]
                curr_delta = dest[i, j]
                x_ij = x[i, j]

                if a < i:
                    idx = a * num_codes - a * (a + 1) // 2 + i
                else:
                    idx = i * num_codes - i * (i + 1) // 2 + a

                crosscorrelation = correlations[idx]

                for k in prange(length):
                    # Add new value
                    pre = crosscorrelation[k]
                    if a < i:
                        post = pre - 2 * x_ij * x[a, (k + j) % length] / length
                    else:
                        post = pre - 2 * x_ij * x[a, (j - k) % length] / length
                    curr_delta += np.abs(post) ** p - np.abs(pre) ** p

                    # Subtract old value
                    if a < i:
                        pre -= 2 * x_ab * x[i, (b - k) % length] / length
                        sign = 1 if (k + j) % length != b else -1
                        post = pre - sign * 2 * x_ij * x[a, (k + j) % length] / length
                    else:
                        pre -= 2 * x_ab * x[i, (b + k) % length] / length
                        sign = 1 if (j - k) % length != b else -1
                        post = pre - sign * 2 * x_ij * x[a, (j - k) % length] / length

                    curr_delta -= np.abs(post) ** p - np.abs(pre) ** p

                dest[i, j] = curr_delta


@njit(fastmath=True, parallel=True)
def best_delta(x, correlations, p, indices):
    """Calculate the change in the objective function when the sign of each bit
    is flipped."""
    num_codes, length = x.shape
    out = np.zeros(len(indices))

    # pylint: disable=consider-using-enumerate
    for sel in prange(len(indices)):
        i, j = indices[sel]

        res = 0.0
        # change in autocorrelation
        idx = i * num_codes - i * (i + 1) // 2 + i
        for k in prange(1, length):
            prev = correlations[idx, k]
            new = (
                prev
                + (-2 * x[i, j] * (x[i, (j + k) % length] + x[i, (j - k) % length]))
                / length
            )
            res += np.abs(new) ** p - np.abs(prev) ** p

        for r in prange(num_codes):
            if r < i:
                # change in right side cross-correlation
                idx = r * num_codes - r * (r + 1) // 2 + i
                for k in prange(length):
                    prev = correlations[idx, k]
                    new = prev - 2 * x[i, j] * x[r, (k + j) % length] / length
                    res += np.abs(new) ** p - np.abs(prev) ** p

            elif r > i:
                # change in left side cross-correlation
                idx = i * num_codes - i * (i + 1) // 2 + r
                for k in prange(length):
                    prev = correlations[idx, k]
                    new = prev - 2 * x[i, j] * x[r, (j - k) % length] / length
                    res += np.abs(new) ** p - np.abs(prev) ** p

        out[sel] = res

    best = np.argmin(out)
    return indices[best]
