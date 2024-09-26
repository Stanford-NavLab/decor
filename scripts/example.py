import numpy as np
import matplotlib.pyplot as plt
from decor import (
    SpreadingCodes,
    GreedyCodeOptimizer,
    TopKGreedyCodeOptimizer,
    AdaptiveKGreedyCodeOptimizer,
    ColumnMajorCyclicCodeOptimizer,
)


if __name__ == "__main__":
    np.random.seed(0)
    x = SpreadingCodes(63, 1023, p=6)
    x.deltas()

    # jit warmup
    AdaptiveKGreedyCodeOptimizer().optimize(x.copy(), n_iter=1)

    methods = [
        # [TopKGreedyCodeOptimizer(num_neighbors=100), 10**6],
        [AdaptiveKGreedyCodeOptimizer(), 10**6],
        # [ColumnMajorCyclicCodeOptimizer(), 10**3],
        # [GreedyCodeOptimizer(), 5],
    ]

    for method, max_iters in methods:
        print(f"Running {str(method.__class__)}")
        x_method = x.copy()
        method.optimize(x_method, n_iter=max_iters)

    for method, _ in methods:
        plt.plot(method.objective_values, label=str(method.__class__))
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.show()

    for method, _ in methods:
        plt.plot(
            np.cumsum(method.iteration_times),
            method.objective_values,
            label=str(method),
        )
    plt.legend()
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Objective Value")
    plt.show()

    for method, _ in methods:
        plt.plot(
            method.iteration_times,
            label=str(method),
        )
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Iteration time (s)")
    plt.show()
