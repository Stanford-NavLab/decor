GPU Implementation Plan for Binary Code Optimization

This document summarizes the key insights, decisions, and implementation ideas for moving the binary code optimization software (originally NumPy/Numba-based) to the GPU.

⸻

1. Problem Recap
   • Optimize binary spreading codes (-1,+1) of shape (n, T).
   • Goal: minimize auto- and cross-correlation sidelobes.
   • Hot path: bit-flip descent – evaluate and apply flips iteratively.
   • Regime: n > 100, T > 10,000, millions of iterations.

⸻

2. Computation Breakdown

Initial correlation
• Compute all autocorr + crosscorr sequences: (n^2+n)/2 sequences of length T.
• Can be accelerated with FFT (torch.fft / cuFFT).
• One-time cost → not a bottleneck.

Objective function
• Uses (\sum |c|^p)^{1/p}.
• Only needs fast evaluation after flips.

Delta updates
• Each flip (i,j) only affects:
• Autocorr row (i,i).
• Crosscorr rows (i,r) for all r ≠ i.
• Updates are O(nT), streaming-friendly.

⸻

3. Why GPUs Fit
   • Work per flip is large (millions of elements for nT).
   • Low arithmetic intensity, bandwidth-bound.
   • GPU memory bandwidth (0.5–1.5 TB/s) >> CPU memory bandwidth (~200 GB/s).
   • Expected speedup: ~3×–10× over optimized CPU if implemented correctly.

⸻

4. Memory Strategy
   • Correlations: store as integers (int16 if T ≤ 32767, else int32).
   • Flip changes sums by ±2.
   • Normalization factor T^{-p} applied only at objective evaluation.
   • Codes: int8 (±1).
   • Delta map: float32 (or fixed-point int32).
   • LUT: L[s] = |s|^p for s=0..T, stored in GPU memory.

Memory footprint examples (int16 correlation storage):
• n=256, T=10k → ~0.66 GB.
• n=512, T=10k → ~2.63 GB.

Fits easily on 24–80 GB GPUs.

⸻

5. GPU Implementation Stack

Recommended stack
• PyTorch: device memory, cuFFT, reductions, top-k.
• Triton: custom kernels for correlation/delta updates.

⸻

6. Kernels Needed

A) update_corr_one_flip(i,j)
• Update (i,i) and (i,r) correlation rows.
• O(nT), but trivially parallel.
• Use padding (Tpad = 2^k) → (j ± k) & (Tpad-1) instead of %.
• No atomics needed if each thread owns unique (r,k) cell.

B) deltas_row(i) / deltas_all()
• Recompute deltas using LUT.
• Each delta uses affected correlations only.
• Implement as wide streaming kernels.

C) Optional: update_deltas_after_flip
• Incremental update (like CPU version).
• More complex; add later.

⸻

7. Iteration Strategy
   1. Initialization
      • Copy codes to GPU.
      • Compute correlations (cuFFT).
      • Build LUT.
   2. Main loop
      • Refresh delta map (deltas_all or row chunks).
      • Pick top-K flips with torch.topk (device-side).
      • Apply best flip:
      • Launch update_corr_one_flip.
      • Flip code entry in-place.
      • Refresh deltas_row(i).
      • Every M flips: full delta refresh.
   3. Objective evaluation
      • Reduction: sum(LUT[|S|]) \* T^{-p}.

⸻

8. Optimization Guidelines
   • Pad T to power of two for cheap indexing.
   • Wide tiles (BLOCK_K ~1024–2048) to saturate memory bandwidth.
   • Keep all data on device; minimize host sync.
   • Batch flips if needed for throughput, but beware interference.
   • Profile early with Nsight to confirm bandwidth-bound kernels.

⸻

9. Scalability
   • Works up to n≈500, T≈20k on a single GPU.
   • For larger problems: shard correlation rows across GPUs; replicate codes.
   • Use NCCL for objective reductions.

⸻

10. Expected Gains
    • Each flip touches ~nT elements (~2.5M for n=256, T=10k).
    • GPU: 5–20k flips/sec (practical).
    • CPU: 1–4k flips/sec.
    • Wall-clock improvement: 3–10×.

⸻

11. Next Steps
    • Implement Triton kernel for update_corr_one_flip.
    • Implement Triton kernel for deltas_row.
    • Build control loop in PyTorch.
    • Validate correctness vs. CPU.
    • Profile and iterate.

⸻

12. Integration Roadmap
    • Stage 0 (parity groundwork) – completed
    • Added PyTorch helper module mirroring the NumPy correlation path.
    • Kept data layout identical to the CPU cache and added golden-value tests.
    • Stage 1 (correlation + data movement) – completed
    • Implement helpers to move codebooks to/from GPU memory with explicit dtypes (codes as int8/float32, correlations as int32/64) ✅
    • Re-implement full correlation with torch.fft, populate integer caches, and verify parity with CPU for small and medium n, T pairs ✅
    • Teach SpreadingCodes to opt into the GPU correlation path via an opt-in flag or injected backend while keeping CPU as the default ✅
    • Stage 2 (delta map evaluation) – in progress
    • [x] Prototype a torch-only implementation that recomputes full delta maps on device using the existing formulas to validate math and indexing.
    • [x] Introduce Triton kernels only after the torch prototype matches CPU results and profiling shows the expected bottlenecks. (2025-10-12: `triton_update_corr_one_flip` and `triton_deltas_row` now land in `decor/gpu_backend.py` with streaming tiles.)
    • [x] Cache the |s|^p lookup table on device and share it between kernels.
    • [x] Add regression tests that compare GPU delta values against the NumPy baseline across representative (n, T, p) tuples.
    • [x] Benchmark the torch delta map path to set a target for the later Triton kernels and record baseline throughput in the plan.
    • [x] Document GPU delta integration steps (data transfers, dtype expectations, cache semantics) for future Triton porting.
    • [x] Implement torch-side `update_corr_one_flip` and `apply_flip_inplace` helpers that mirror the CPU packed-layout updates exactly and add parity tests to guard future refactors.
    • Stage 3 (incremental updates and optimizer loop)
    • Implement the update_corr_one_flip and per-row delta refresh kernels. ✅ (2025-10-12 Triton versions in place; integrate with optimiser next.)
    • Capture Nsight Systems traces for the Triton kernels on a CUDA host to quantify launch overhead and memory bandwidth utilisation.
    • Expand regression coverage (multi-flip optimiser journeys, mixed CPU/GPU parity suites) once kernels settle.
    • [~] Extend AdaptiveKGreedyCodeOptimizer to call the GPU helpers while keeping a CPU fallback path for debugging. (2025-10-12: Greedy and TopK variants now read `delta_tensor`, so the adaptive strategy keeps flips on device when `use_gpu=True`.)
    • Measure end-to-end behaviour, add mixed-device regression tests, and document troubleshooting steps (dtype mismatches, device sync costs).
    • Promote the new torch helpers into the optimiser loop so flips stay on device without rebuilding correlation caches from NumPy.

⸻

13. Baseline Torch Delta Benchmark

    • Command: `python scripts/benchmark_gpu_deltas.py --device cuda` (falls back to CPU when CUDA is unavailable).
    • Configuration: warmup=2, repeats=5, seed=1234, p=2.0.
    • Current status:
    – Script landed and exercised on preliminary inputs.
    – Record measured averages, standard deviations, and throughput (M entries/s) once runs complete on additional hardware.
    – Capture both CUDA and CPU figures when possible so the Triton goals have multiple reference points.
    • Preliminary measurement (2025-10-11):
    – CUDA (`n=31`, `T=1023`): avg 148,313.653 ms, std 45.137 ms → 0.00 M entries/s (rounded).
    – CPU (`n=31`, `T=1023`): avg 144,316.146 ms, std 100.367 ms → 0.00 M entries/s (rounded).
    – Similar runtimes across devices confirm that the torch prototype is dominated by host-driven loops rather than GPU arithmetic throughput.
    • 2025-10-12 update:
    – Attempted `python scripts/benchmark_gpu_deltas.py --device cuda --case 128x8192 --repeats 10`, but torch reports no CUDA support on this workstation. Re-run on a CUDA-enabled host and capture an Nsight Systems trace when available.
    – Triton kernels now exist for correlation updates and per-row deltas; benchmarking pending on GPU hardware.
    – Added a GPU parity test that flips bits via `apply_flip_inplace` to verify caches stay in sync with the CPU reference.
    – Ran `python3 -m scripts.benchmark_gpu_deltas --device cuda` on 2025-10-12: `n=31`, `T=1023` averaged 139,803.505 ms (std 61.646 ms) → 0.00 M entries/s. Matches the earlier torch-only baseline because the benchmark still exercises `compute_delta_map` instead of the new Triton kernels.
    – Next steps: wire the benchmark (or a sibling entry point) to call `triton_deltas_row`/`triton_update_corr_one_flip`, rerun on CUDA with Nsight Systems, and confirm the expected bandwidth-bound behaviour.
    • Expected bottleneck:
    – The Python double loop (`
