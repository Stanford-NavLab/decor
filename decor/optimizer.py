"""Spreading code optimization."""

import pickle
import os
import re
from time import perf_counter
from typing import Any, Dict, List, Tuple, Optional
from numbers import Number

import numpy as np

from tqdm import tqdm
from .spreading_codes import SpreadingCodes


class SpreadingCodeOptimizer:
    """Spreading code optimizer."""

    def __init__(
        self,
        patience: Optional[int] = None,
        log_dir: Optional[str] = None,
        log_freq_mins: Number = 10,
    ):
        self.patience = patience
        self.log_dir = log_dir
        self.log_freq_mins = log_freq_mins

        self.objective_values: List[float] = []
        self.iteration_times: List[float] = []
        self.best_objective: float = float("inf")
        self.not_improved: int = 0
        self.best_codes: Optional[SpreadingCodes] = None

    def optimize(self, codes: SpreadingCodes, n_iter: int):
        """Optimize the spreading code."""
        if self.patience is None:
            self.patience = n_iter

        initial_objective = codes.objective()
        self.best_codes = codes.copy()
        init_time = perf_counter()
        with tqdm(total=n_iter) as pbar:
            for iteration in range(n_iter):
                # step optimizer
                start_time = perf_counter()
                edits = self.step(codes)
                elapsed = perf_counter() - start_time
                self.iteration_times.append(elapsed)

                # Update objective
                objective = codes.objective()
                self.objective_values.append(objective)

                # Update best codes
                if objective < self.best_objective:
                    self.best_objective = objective
                    for i, j in edits:
                        self.best_codes.value[i, j] *= -1

                    self.not_improved = 0
                else:
                    self.not_improved += 1

                # update progress bar
                pct_improvement = (
                    100 * (initial_objective - self.best_objective) / initial_objective
                )
                msg = f"Objective: {objective:.3f}"
                msg += f" Best: {self.best_objective:.3f}"
                msg += f" improvement: {pct_improvement:.3f}%"
                pbar.set_description(msg)

                # write log to file
                if perf_counter() - init_time > 60 * self.log_freq_mins:
                    self.write_log(iteration, codes)
                    init_time = perf_counter()

                # check early stopping
                if self.early_stop():
                    break

                pbar.update(1)

        self.write_log(iteration, codes)

    def write_log(self, iteration: int, codes: SpreadingCodes) -> None:
        """Write log to file."""
        if self.log_dir is not None:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            # write code
            code_data = {
                "iteration": iteration,
                "num_codes": codes.num_codes,
                "code_length": codes.code_length,
                "p": codes.p,
                "codes": codes.value,
                "best_codes": self.best_codes.value,
                "best_objective": self.best_objective,
            }
            file_name = os.path.join(self.log_dir, "results.pkl")
            pickle.dump(code_data, open(file_name, "wb"))

            # write log
            file_name = os.path.join(self.log_dir, f"iteration_{iteration}.pkl")
            pickle.dump(self.log(), open(file_name, "wb"))
            self.objective_values.clear()
            self.iteration_times.clear()

    def log(self) -> Dict[str, Any]:
        """Generate iteration log."""
        return {
            "objectives": self.objective_values,
            "not_improved": self.not_improved,
            "iteration_times": self.iteration_times,
        }

    def step(self, codes: SpreadingCodes) -> List[Tuple[int, int]]:
        """Select a bit to flip."""
        raise NotImplementedError

    def early_stop(self) -> bool:
        """Check if the optimization should stop."""
        return self.not_improved >= self.patience


def load_log(path: str) -> Dict[str, Any]:
    """Load optimization log."""
    data = {}
    for f in os.listdir(path):
        if f.endswith(".pkl"):
            match = re.search(r"iteration_(\d+)\.pkl", f)
            if match:
                file_data = pickle.load(open(os.path.join(path, f), "rb"))
                data[int(match.group(1))] = file_data

    objective_values = []
    iteration_times = []

    for iteration in sorted(data.keys()):
        objective_values.extend(data[iteration]["objectives"])
        iteration_times.extend(data[iteration]["iteration_times"])

    res = data[max(data.keys())]
    res["objectives"] = np.array(objective_values)
    res["iteration_times"] = np.array(iteration_times)
    res["elapsed_time"] = np.cumsum(iteration_times)

    code_data = pickle.load(open(os.path.join(path, "results.pkl"), "rb"))
    res["best_codes"] = code_data["best_codes"]
    res["p"] = code_data["p"]

    return res
