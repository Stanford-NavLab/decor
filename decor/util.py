"""Utility functions"""

# pylint: disable=not-an-iterable
from typing import Optional
import numpy as np


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
