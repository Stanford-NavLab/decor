"""decor: binary spreading codes with low autocorrelation and cross-correlation."""

from rocket_fft import numpy_like

from .util import randb

from .spreading_codes import (
    SpreadingCodes,
    random_code_family,
    gold_code_family,
    weil_code_family,
)
from .bit_flip_descent import (
    RandomCodeOptimizer,
    ColumnMajorCyclicCodeOptimizer,
    RowMajorCyclicCodeOptimizer,
    BiST,
    GreedyCodeOptimizer,
    TopKGreedyCodeOptimizer,
    AdaptiveKGreedyCodeOptimizer,
)

from .optimizer import SpreadingCodeOptimizer, load_log


numpy_like()
