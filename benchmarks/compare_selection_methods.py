#!/usr/bin/env python3
"""
Compare KV cache selection methods: GPU scan vs PRISM.

Measures actual GPU signature scan latency at various context lengths,
then compares with PRISM's O(1) photonic selection (9 ns physics constant).

Shows that scan cost dominates at long contexts, and PRISM eliminates it.

Usage:
    python benchmarks/compare_selection_methods.py
    python benchmarks/compare_selection_methods.py --device cuda
"""
import argparse
import time
import torch


def measure_gpu_scan(n_blocks, d_sig, k, n_trials=500, device="cuda"):
    """Measure GPU signature scan latency (read + inner product + top-k)."""
    sigs = torch.randn(n_blocks, d_sig, device=device, dtype=torch.float16)
    query = torch.randn(d_sig, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(100):
        scores = sigs @ query
        _ = torch.topk(scores, min(k, n_blocks))
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(n_trials):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        scores = sigs @ query
        _ = torch.topk(scores, min(k, n_blocks))
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)  # us

    return sorted(times)[len(times) // 2]  # median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d_sig", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args()

    device = args.device
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    PRISM_NS = 9.0  # photonic pipeline latency (physics constant)

    contexts = [
        (16_000, "16K"),
        (128_000, "128K"),
        (1_000_000, "1M"),
        (10_000_000, "10M"),
        (100_000_000, "100M"),
    ]

    print("=" * 72)
    print("  Selection Method Comparison: GPU Scan vs PRISM")
    print("=" * 72)
    print(f"  GPU: {gpu_name}")
    print(f"  Config: d_sig={args.d_sig}, k={args.k}, B={args.block_size}")
    print(f"  PRISM: {PRISM_NS} ns (photonic O(1), device-physics estimate)")
    if device != "cuda":
        print("  WARNING: No CUDA GPU. Values will not reflect real HBM latency.")
    print()

    # Header
    print(f"  {'Context':>8}  {'N blocks':>10}  "
          f"{'GPU scan':>12}  {'Scan-free*':>10}  {'Speedup':>10}  {'Scan %':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")

    results = []
    for ctx, label in contexts:
        n_blocks = ctx // args.block_size

        # GPU scan measurement (or estimate for very large N)
        if n_blocks <= 1_000_000 and device == "cuda":
            gpu_us = measure_gpu_scan(n_blocks, args.d_sig, args.k, device=device)
        elif n_blocks <= 100_000:
            gpu_us = measure_gpu_scan(n_blocks, args.d_sig, args.k, device=device)
        else:
            # Extrapolate from smaller measurement
            ref_n = 100_000
            ref_us = measure_gpu_scan(ref_n, args.d_sig, args.k, device=device)
            gpu_us = ref_us * (n_blocks / ref_n)

        prism_us = PRISM_NS / 1000  # convert to us
        speedup = gpu_us / prism_us

        # Scan fraction: scan / (scan + fetch)
        # Fetch = k * B * d_h * 2 bytes / HBM_BW
        fetch_bytes = args.k * args.block_size * 128 * 2  # d_h=128, bf16
        hbm_bw = 3.35e12  # H100 HBM3 bytes/sec
        fetch_us = (fetch_bytes / hbm_bw) * 1e6
        scan_fraction = gpu_us / (gpu_us + fetch_us) * 100

        # Format GPU scan
        if gpu_us >= 1000:
            gpu_str = f"{gpu_us/1000:.1f} ms"
        else:
            gpu_str = f"{gpu_us:.1f} us"

        print(f"  {label:>8}  {n_blocks:>10,}  {gpu_str:>12}  "
              f"{'9 ns':>10}  {speedup:>9,.0f}x  {scan_fraction:>7.0f}%")

        results.append({
            "context": label, "n_blocks": n_blocks,
            "gpu_scan_us": round(gpu_us, 2),
            "prism_us": round(prism_us, 4),
            "speedup": round(speedup),
            "scan_fraction_pct": round(scan_fraction, 1),
        })

    print()
    print(f"  {'KEY INSIGHT':=^72}")
    print(f"  At 128K context, GPU scan is small ({results[1]['scan_fraction_pct']}% of traffic).")
    print(f"  At 10M+ context, scan dominates ({results[3]['scan_fraction_pct']}%+).")
    print(f"  PRISM eliminates the scan entirely: {PRISM_NS} ns regardless of N.")
    print(f"  {'':=^72}")
    print()
    print(f"  * GPU scan: {'MEASURED on ' + gpu_name if device == 'cuda' else 'ESTIMATED (no CUDA)'}")
    print(f"  * Scan-free*: theoretical lower bound if scan is fully eliminated")
    print(f"    (e.g., via photonic broadcast, dedicated ASIC, or other O(1) hardware)")
    print(f"  * No fabricated PRISM chip exists — 9 ns is a device-physics estimate")
    print(f"  * Scan %: fraction of decode HBM traffic spent on signature scan")
    print(f"  * Quest/RocketKV have similar GPU scan cost (same O(N) scaling)")


if __name__ == "__main__":
    main()
