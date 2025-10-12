#!/usr/bin/env python3
"""Benchmark the torch delta-map prototype on CPU or GPU devices.

This standalone script exercises ``decor.gpu_backend.compute_delta_map`` so we
can record baseline throughput before porting the kernels to Triton. The script
keeps the control flow intentionally simple and prints plain text summaries that
fit well into development notes or the GPU implementation log.
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


def _time_delta_map(
    codes_tensor: torch.Tensor,
    correlations: torch.Tensor,
    p_value: float,
    lut: torch.Tensor,
    *,
    repeats: int,
    warmup: int,
) -> Sequence[float]:
    """Return a list of elapsed milliseconds for repeated delta-map runs."""

    device = codes_tensor.device

    # Warm-up runs let cuBLAS / cuFFT caches settle and hide one-off allocations.
    for _ in range(warmup):
        compute_delta_map(codes_tensor, correlations, p_value, lut=lut)

    timings: List[float] = []
    for _ in range(repeats):
        _maybe_synchronize(device)
        start = time.perf_counter()
        compute_delta_map(codes_tensor, correlations, p_value, lut=lut)
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
) -> None:
    """Execute every benchmark case and print the collected statistics."""

    for offset, case in enumerate(cases):
        case_seed = seed + offset
        codes_tensor, correlations, lut = _prepare_inputs(
            case, device=device, p_value=p_value, seed=case_seed
        )

        timings = _time_delta_map(
            codes_tensor,
            correlations,
            p_value,
            lut,
            repeats=repeats,
            warmup=warmup,
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

    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    return args


def main() -> None:
    """Entry point for the benchmarking utility."""

    args = parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA device, but torch reports no CUDA support")

    print("Benchmarking compute_delta_map" f" on device {device}" f" with p={args.p}")
    print(
        f"warmup={args.warmup}, repeats={args.repeats}, seed={args.seed}"  # noqa: T201
    )

    cases = _build_cases(args)
    run_benchmarks(
        cases,
        device=device,
        p_value=args.p,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
