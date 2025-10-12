#!/usr/bin/env python3
"""Benchmark GPU delta-map paths on CPU or CUDA devices.

The tool now supports both the original torch-only reference implementation and
the Triton kernels that stream correlation updates and per-row delta refreshes.
Use the ``--backend`` flag to switch between paths while keeping timing and
reporting identical.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from decor.gpu_backend import (
    build_abs_p_lut,
    codes_array_to_tensor,
    compute_delta_map,
    compute_packed_correlation,
    resolve_device,
    triton_deltas_row,
    triton_update_corr_one_flip,
)


@dataclass
class BenchmarkCase:
    """Container for one benchmark configuration.

    Each case stores the structure of the code family that we benchmark. The
    dataclass format keeps the fields self-describing and easy to log.
    """

    num_codes: int
    code_length: int


def _parse_case(raw_value: str) -> BenchmarkCase:
    """Parse a ``NUMxLEN`` string into a :class:`BenchmarkCase` instance."""

    if "x" not in raw_value:
        raise argparse.ArgumentTypeError("Case format must look like 'NUMxLEN'")

    first, second = raw_value.lower().split("x", 1)
    try:
        num_codes = int(first)
        code_length = int(second)
    except ValueError as exc:  # pragma: no cover - guarded by argparse
        raise argparse.ArgumentTypeError("Case format must contain integers") from exc

    if num_codes <= 0 or code_length <= 0:
        raise argparse.ArgumentTypeError("Case values must be positive")

    return BenchmarkCase(num_codes=num_codes, code_length=code_length)


def _default_cases() -> List[BenchmarkCase]:
    """Return a short list of representative benchmark sizes."""

    return [
        BenchmarkCase(num_codes=31, code_length=1023),
    ]


def _build_cases(args: argparse.Namespace) -> List[BenchmarkCase]:
    """Build the ordered set of benchmark cases from user input."""

    if args.case:
        return list(args.case)

    if args.num_codes is not None and args.code_length is not None:
        return [BenchmarkCase(num_codes=args.num_codes, code_length=args.code_length)]

    if args.num_codes is not None or args.code_length is not None:
        raise SystemExit("Both --num-codes and --code-length must be supplied together")

    return _default_cases()


def _maybe_synchronize(device: torch.device) -> None:
    """Synchronise CUDA streams when needed so timings stay accurate."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compute_delta_map_triton(
    codes_tensor: torch.Tensor,
    correlations: torch.Tensor,
    lut: torch.Tensor,
    *,
    block_t: int,
    block_k: int,
) -> torch.Tensor:
    """Recompute the full delta tensor via Triton row kernels."""

    num_codes, code_length = codes_tensor.shape
    delta_map = torch.empty(
        (num_codes, code_length), dtype=torch.float64, device=codes_tensor.device
    )

    for row_index in range(num_codes):
        delta_row = triton_deltas_row(
            codes_tensor,
            correlations,
            lut,
            row_index,
            block_t=block_t,
            block_k=block_k,
        )
        delta_map[row_index].copy_(delta_row)

    return delta_map


def _execute_triton_iteration(
    codes_tensor: torch.Tensor,
    correlations: torch.Tensor,
    lut: torch.Tensor,
    *,
    block_t: int,
    block_k: int,
    update_block_t: int,
    update_block_n: int,
    flips_per_iter: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Run one Triton iteration: delta recompute plus optional random flips."""

    delta_map = _compute_delta_map_triton(
        codes_tensor,
        correlations,
        lut,
        block_t=block_t,
        block_k=block_k,
    )

    if flips_per_iter > 0:
        num_codes, code_length = codes_tensor.shape
        for _ in range(flips_per_iter):
            row = int(rng.integers(0, num_codes))
            col = int(rng.integers(0, code_length))
            triton_update_corr_one_flip(
                codes_tensor,
                correlations,
                row,
                col,
                block_t=update_block_t,
                block_n=update_block_n,
            )
            codes_tensor[row, col] = -codes_tensor[row, col]

    return delta_map


def _time_backend(
    backend: str,
    codes_tensor: torch.Tensor,
    correlations: torch.Tensor,
    p_value: float,
    lut: torch.Tensor,
    *,
    repeats: int,
    warmup: int,
    triton_block_t: int,
    triton_block_k: int,
    triton_update_block_t: int,
    triton_update_block_n: int,
    flips_per_iter: int,
    seed: int,
) -> Sequence[float]:
    """Return elapsed milliseconds for repeated runs with the chosen backend."""

    device = codes_tensor.device
    backend_name = backend.lower()
    if backend_name not in {"torch", "triton"}:
        raise ValueError(f"Unsupported backend '{backend}'")

    rng = np.random.default_rng(seed=seed)
    codes_work = codes_tensor.clone()
    correlations_work = correlations.clone()

    # Warm-up to amortise first-launch overheads and JIT compilation.
    for _ in range(max(0, warmup)):
        if backend_name == "torch":
            compute_delta_map(codes_work, correlations_work, p_value, lut=lut)
        else:
            _execute_triton_iteration(
                codes_work,
                correlations_work,
                lut,
                block_t=triton_block_t,
                block_k=triton_block_k,
                update_block_t=triton_update_block_t,
                update_block_n=triton_update_block_n,
                flips_per_iter=flips_per_iter,
                rng=rng,
            )

    timings: List[float] = []
    for _ in range(repeats):
        _maybe_synchronize(device)
        start = time.perf_counter()
        if backend_name == "torch":
            compute_delta_map(codes_work, correlations_work, p_value, lut=lut)
        else:
            _execute_triton_iteration(
                codes_work,
                correlations_work,
                lut,
                block_t=triton_block_t,
                block_k=triton_block_k,
                update_block_t=triton_update_block_t,
                update_block_n=triton_update_block_n,
                flips_per_iter=flips_per_iter,
                rng=rng,
            )
        _maybe_synchronize(device)
        end = time.perf_counter()
        timings.append((end - start) * 1_000.0)

    return timings


def _format_ms(value_ms: float) -> str:
    """Return a compact millisecond string with three decimal places."""

    return f"{value_ms:.3f} ms"


def _format_throughput(num_codes: int, code_length: int, avg_ms: float) -> str:
    """Return a human-readable throughput summary.

    The delta map visits every entry in the ``(num_codes, code_length)`` array.
    We express the throughput in millions of entries per second so the figure is
    easy to compare against future Triton versions.
    """

    entries = num_codes * code_length
    maps_per_second = 1_000.0 / avg_ms if avg_ms > 0.0 else float("inf")
    mega_entries = (entries / 1_000_000.0) * maps_per_second
    return f"{mega_entries:.2f} M entries/s"


def _prepare_inputs(
    case: BenchmarkCase,
    device: torch.device,
    p_value: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the tensors that feed each benchmark iteration."""

    rng = np.random.default_rng(seed=seed)
    codes = rng.choice([-1, 1], size=(case.num_codes, case.code_length)).astype(np.int8)

    codes_tensor = codes_array_to_tensor(codes, device=device)
    correlations = compute_packed_correlation(codes_tensor)
    lut = build_abs_p_lut(case.code_length, p_value, device=device)

    return codes_tensor, correlations, lut


def run_benchmarks(
    cases: Iterable[BenchmarkCase],
    *,
    device: torch.device,
    p_value: float,
    repeats: int,
    warmup: int,
    seed: int,
    backend: str,
    triton_block_t: int,
    triton_block_k: int,
    triton_update_block_t: int,
    triton_update_block_n: int,
    flips_per_iter: int,
) -> None:
    """Execute every benchmark case and print the collected statistics."""

    for offset, case in enumerate(cases):
        case_seed = seed + offset
        codes_tensor, correlations, lut = _prepare_inputs(
            case, device=device, p_value=p_value, seed=case_seed
        )

        timings = _time_backend(
            backend,
            codes_tensor,
            correlations,
            p_value,
            lut,
            repeats=repeats,
            warmup=warmup,
            triton_block_t=triton_block_t,
            triton_block_k=triton_block_k,
            triton_update_block_t=triton_update_block_t,
            triton_update_block_n=triton_update_block_n,
            flips_per_iter=flips_per_iter,
            seed=case_seed,
        )

        avg_ms = float(np.mean(timings))
        std_ms = float(np.std(timings))
        throughput = _format_throughput(case.num_codes, case.code_length, avg_ms=avg_ms)

        print(
            f"n={case.num_codes:4d}, T={case.code_length:6d} :: "
            f"avg {_format_ms(avg_ms)}, std {_format_ms(std_ms)} :: {throughput}"
        )


def parse_args() -> argparse.Namespace:
    """Return command-line configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the torch delta-map prototype across one or more code sizes. "
            "This enables quick comparisons between CPU, CUDA, and future Triton "
            "implementations."
        )
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device specifier. Leave empty to pick CUDA when available and "
            "fall back to CPU otherwise."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "triton"],
        default="torch",
        help=(
            "Computation backend. Use 'torch' for the reference path or 'triton' "
            "to exercise the GPU kernels."
        ),
    )
    parser.add_argument(
        "--p",
        type=float,
        default=2.0,
        help="Objective exponent. Matches the optimiser configuration.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed iterations per case. Must be positive.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of untimed warm-up runs per case."
        " The value should stay small, just enough to hide allocation hiccups.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for the RNG that populates the code families.",
    )
    parser.add_argument(
        "--num-codes",
        type=int,
        default=None,
        help="Optional single-case override for the code count.",
    )
    parser.add_argument(
        "--code-length",
        type=int,
        default=None,
        help="Optional single-case override for the code length.",
    )
    parser.add_argument(
        "--case",
        type=_parse_case,
        action="append",
        help=(
            "Append a benchmark case described as NUMxLEN. Repeat the flag to add "
            "multiple cases."
        ),
    )
    triton_group = parser.add_argument_group(
        "triton", "Configuration for Triton kernels"
    )
    triton_group.add_argument(
        "--triton-block-t",
        type=int,
        default=1024,
        help="Triton BLOCK_T for delta rows. Must be a multiple of 128.",
    )
    triton_group.add_argument(
        "--triton-block-k",
        type=int,
        default=128,
        help="Triton BLOCK_K for delta rows. Must be a multiple of 8.",
    )
    triton_group.add_argument(
        "--triton-update-block-t",
        type=int,
        default=1024,
        help="Triton BLOCK_T for correlation updates. Must be a multiple of 128.",
    )
    triton_group.add_argument(
        "--triton-update-block-n",
        type=int,
        default=32,
        help="Triton BLOCK_N for correlation updates. Must be a multiple of 4.",
    )
    parser.add_argument(
        "--flips-per-iter",
        type=int,
        default=0,
        help=(
            "Number of random flips to apply during each Triton iteration. Useful "
            "for profiling correlation updates alongside delta recomputes."
        ),
    )

    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.flips_per_iter < 0:
        parser.error("--flips-per-iter cannot be negative")

    return args


def main() -> None:
    """Entry point for the benchmarking utility."""

    args = parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA device, but torch reports no CUDA support")

    print("Benchmarking compute_delta_map" f" on device {device}" f" with p={args.p}")
    print(
        f"backend={args.backend}, warmup={args.warmup}, repeats={args.repeats}, seed={args.seed}"  # noqa: T201
    )

    cases = _build_cases(args)
    run_benchmarks(
        cases,
        device=device,
        p_value=args.p,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
        backend=args.backend,
        triton_block_t=args.triton_block_t,
        triton_block_k=args.triton_block_k,
        triton_update_block_t=args.triton_update_block_t,
        triton_update_block_n=args.triton_update_block_n,
        flips_per_iter=args.flips_per_iter,
    )


if __name__ == "__main__":
    main()
