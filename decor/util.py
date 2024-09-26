"""Utility functions"""

# pylint: disable=not-an-iterable
from typing import Optional
import numpy as np
from numba import njit


def randb(*shape, dtype=float):
    """Generate a random binary vector of length n."""
    return np.random.choice((-1, 1), shape).astype(dtype)


def argsort(x: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """Return the indices that would sort an array."""
    if k is None:
        idx = np.argsort(x.ravel())
        return np.vstack(
            [np.array([i, j]) for i, j in zip(*np.unravel_index(idx, x.shape))]
        )
    else:
        x_flat = x.flatten()
        top_idx_flat = np.argpartition(x_flat, k)[:k]
        sorted_idx = top_idx_flat[np.argsort(x_flat[top_idx_flat])]
        return np.array(np.unravel_index(sorted_idx, x.shape)).T


@njit(fastmath=True)
def norm_grad(x, p, unscaled=True) -> np.ndarray:
    """
    Explicitly computes the gradient of the Lp norm.
    """
    unscaled_grad = np.sign(x) * np.power(np.abs(x), p - 1)
    if unscaled:
        return unscaled_grad

    return unscaled_grad / np.power(np.linalg.norm(x.flatten(), ord=p), p - 1)
