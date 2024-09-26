"""Bit-flip descent methods"""

from typing import List, Tuple

import numpy as np
from .spreading_codes import SpreadingCodes
from .optimizer import SpreadingCodeOptimizer
from .bit_flip import best_delta


class RandomCodeOptimizer(SpreadingCodeOptimizer):
    """Random bit flip descent."""

    def step(self, codes: SpreadingCodes) -> List[Tuple[int, int]]:
        """Cyclic bit flip descent step."""
        i = np.random.choice(codes.num_codes)
        j = np.random.choice(codes.code_length)

        if codes.delta(i, j) < 0:
            codes.flip(i, j)
            return [(i, j)]
        return []


class ColumnMajorCyclicCodeOptimizer(SpreadingCodeOptimizer):
    """Column major cyclic bit flip descent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.i: int = 0
        self.j: int = 0

    def update_index(self, num_codes: int, code_length: int) -> None:
        """Cyclic index update"""
        self.i += 1
        if self.i == num_codes:
            self.i = 0
            self.j += 1
            if self.j == code_length:
                self.j = 0

    def step(self, codes: SpreadingCodes) -> List[Tuple[int, int]]:
        """Cyclic bit flip descent step."""
        if codes.delta(self.i, self.j) < 0:
            codes.flip(self.i, self.j)
            edits = [(self.i, self.j)]
        else:
            edits = []

        self.update_index(*codes.shape)

        return edits


class RowMajorCyclicCodeOptimizer(ColumnMajorCyclicCodeOptimizer):
    """Column major cyclic bit flip descent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.i: int = 0
        self.j: int = 0

    def update_index(self, num_codes: int, code_length: int) -> None:
        """Cyclic index update"""
        self.j += 1
        if self.j == code_length:
            self.j = 0
            self.i += 1
            if self.i == num_codes:
                self.i = 0


class BiST(SpreadingCodeOptimizer):
    """BiST method."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.i: int = 0
        self.j: int = 0
        self.not_improved_row: int = 0

    def step(self, codes: SpreadingCodes) -> List[Tuple[int, int]]:
        """BiST step."""
        if codes.delta(self.i, self.j) < 0:
            codes.flip(self.i, self.j)
            edits = [(self.i, self.j)]
            self.not_improved_row = 0
        else:
            edits = []
            self.not_improved_row += 1

        # set next index to be in the same code
        self.j = (self.j + 1) % codes.code_length

        # move on to the next code if no improvement
        if self.not_improved_row == codes.code_length:
            self.i = (self.i + 1) % codes.num_codes
            self.j = 0
            self.not_improved_row = 0

        return edits


class GreedyCodeOptimizer(SpreadingCodeOptimizer):
    """Greedy bit flip descent."""

    def step(self, codes: SpreadingCodes) -> List[Tuple[int, int]]:
        """Greedy bit flip descent step."""
        i, j = codes.best_delta()

        if codes.delta(i, j) < 0:
            codes.flip(i, j)
            return [(i, j)]
        return []

    def early_stop(self) -> bool:
        """Check if the optimization should stop."""
        return self.not_improved >= 1


class TopKGreedyCodeOptimizer(SpreadingCodeOptimizer):
    """Top-K greedy bit flip descent."""

    def __init__(self, num_neighbors: int, cyclic: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.num_neighbors = num_neighbors
        self.cyclic = cyclic
        self.i = 0
        self.j = 0

    def step(self, codes: SpreadingCodes) -> List[Tuple[int, int]]:
        """Select a bit to flip."""
        flat_indices = np.random.choice(
            codes.num_codes * codes.code_length, size=self.num_neighbors, replace=False
        )
        indices = np.vstack(np.unravel_index(flat_indices, codes.shape)).T
        if self.cyclic:
            indices[0, :] = self.i, self.j
            ColumnMajorCyclicCodeOptimizer.update_index(self, *codes.shape)

        i, j = best_delta(
            codes.value,
            codes.correlation(),
            codes.p,
            indices,
        )

        if codes.delta(i, j) < 0:
            codes.flip(i, j)
            return [(i, j)]

        return []


class AdaptiveKGreedyCodeOptimizer(SpreadingCodeOptimizer):
    """Adaptive K greedy bit flip descent."""

    def __init__(
        self,
        num_initial_neighbors: int = 1,
        cyclic: bool = False,
        alpha: float = 10.0,
        increment_patience: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_neighbors = num_initial_neighbors
        self.cyclic = cyclic
        self.alpha = alpha
        self.increment_patience = increment_patience
        self.i = 0
        self.j = 0

    def step(self, codes: SpreadingCodes) -> List[Tuple[int, int]]:
        """Select a bit to flip."""
        if (
            self.num_neighbors < codes.code_length
            and self.not_improved >= self.increment_patience
        ):
            # increase number of neighbors
            self.num_neighbors += 1

        if self.num_neighbors >= self.alpha * codes.code_length:
            # switch to greedy
            self.not_improved = 0
            self.patience = 1
            return GreedyCodeOptimizer.step(self, codes)

        else:
            return TopKGreedyCodeOptimizer.step(self, codes)
