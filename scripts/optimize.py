"""Dispatch code optimization.
Example call: python -m scripts.optimize --method HillClimbing --args heuristic_freq=100
"""

import argparse
import time
import os

import decor


OUTPUT_DIR = "results"


def convert_value(v: str):
    """Convert value to int, float, or str."""
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random seed", type=int, default=0)
    parser.add_argument("--n", help="Number of codes", type=int, default=31)
    parser.add_argument("--len", help="Code length", type=int, default=1023)
    parser.add_argument("--p", help="p", type=int, default=2)
    parser.add_argument("--iters", help="Maximum iterations", type=int, default=10**6)
    parser.add_argument(
        "--method", help="Method", type=str, default="AdaptiveKGreedyCodeOptimizer"
    )
    parser.add_argument("--args", nargs="*", help="Arguments in key=value format")
    args = parser.parse_args()

    # optimizer kwargs
    kwargs = {}
    if args.args:
        for arg in args.args:
            key, value = arg.split("=")
            kwargs[key] = convert_value(value)

    p = convert_value(args.p)

    num = int(time.time())
    name = f"n={args.n}_l={args.len}_p={p}_seed={args.seed}_{args.method}({str(kwargs)})_id={num}"
    log_dir = os.path.join(OUTPUT_DIR, f"{name}")

    # generate initial code
    x = decor.random_code_family(args.n, args.len, seed=args.seed, p=p)

    # run optimizer
    method = getattr(decor, args.method)

    print(f"Running {args.method} with args: {kwargs}")
    opt = method(log_dir=log_dir, **kwargs)

    # run optimization
    opt.optimize(x, args.iters)
