"""Accelerated auto and cross correlation"""

# pylint: disable=not-an-iterable
import numpy as np
from numpy.fft import fft, ifft
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def compute_correlation(x, dest):
    """Calculate the cross-correlation of all pairs of codes.
    x has shape (num_codes, code_length)
    dest has shape ((num_codes**2 + num_codes)//2, code_length)"""
    num_codes, code_length = x.shape

    ffts = np.empty(x.shape, dtype=np.complex128)
    for i in prange(num_codes):
        ffts[i, :] = fft(x[i])

    for i in prange(num_codes):
        for j in prange(i, num_codes):
            idx = i * num_codes - i * (i + 1) // 2 + j
            dest[idx, :] = ifft(ffts[i] * ffts[j].conj()).real / code_length
            if i == j:
                dest[idx, 0] = 0.0


@njit(fastmath=True, parallel=True)
def update_correlation(x, i, j, dest):
    """Update the auto- and cross-correlation when the sign of
    x[i, j] is flipped. The result is stored in the pre-allocated array
    dest. dest has shape ((num_codes**2 + num_codes) // 2, code_length)"""
    num_codes, code_length = x.shape
    x_ij = x[i, j]
    for r in prange(num_codes):
        if i <= r:
            idx = i * num_codes - i * (i + 1) // 2 + r
            for k in prange(i == r, code_length):
                dest[idx, k] += -2 * x_ij * x[r, (j - k) % code_length] / code_length

        if i >= r:
            idx = r * num_codes - r * (r + 1) // 2 + i
            for k in prange(i == r, code_length):
                dest[idx, k] += -2 * x_ij * x[r, (k + j) % code_length] / code_length
