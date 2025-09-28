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
    • Stage 1 (correlation + data movement) – in progress
    • Implement helpers to move codebooks to/from GPU memory with explicit dtypes (codes as int8/float32, correlations as int32/64) ✅
    • Re-implement full correlation with torch.fft, populate integer caches, and verify parity with CPU for small and medium n, T pairs ✅
    • Teach SpreadingCodes to opt into the GPU correlation path via an opt-in flag or injected backend while keeping CPU as the default ✅
    • Stage 2 (delta map evaluation)
    • Prototype a torch-only implementation that recomputes full delta maps on device using the existing formulas to validate math and indexing.
    • Introduce Triton kernels only after the torch prototype matches CPU results and profiling shows the expected bottlenecks.
    • Cache the |s|^p lookup table on device and share it between kernels.
    • Stage 3 (incremental updates and optimizer loop)
    • Implement the update_corr_one_flip and per-row delta refresh kernels.
    • Extend AdaptiveKGreedyCodeOptimizer to call the GPU helpers while keeping a CPU fallback path for debugging.
    • Measure end-to-end behaviour, add mixed-device regression tests, and document troubleshooting steps (dtype mismatches, device sync costs).

    These stages give us a reversible path: we can ship Stage 1 for immediate FFT acceleration while Stage 2/3 bake, and we preserve confidence by running CPU/GPU parity tests throughout.

⸻

Conclusion: PyTorch + Triton provides the right balance of speed and development time. Storing correlations as integers and using LUTs for powers gives efficiency and exactness. With careful kernel design, substantial GPU speedups are achievable in the targeted problem regime.
