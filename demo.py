#!/usr/bin/env python3
"""
PRISM Quick Demo — Photonic block selection simulation on GPU.

Runs entirely on PyTorch/CUDA. No photonic hardware needed.

Usage:
    python demo.py              # auto-detect GPU
    python demo.py --device cpu  # force CPU
"""
import argparse
import torch
from prism.simulator import PRISMSimulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_blocks", type=int, default=1024)
    parser.add_argument("--d_head", type=int, default=128)
    parser.add_argument("--d_sig", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--drift", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    n_tokens = args.n_blocks * args.block_size

    print("=" * 60)
    print("  PRISM Demo: Photonic KV Cache Block Selection")
    print("=" * 60)
    print(f"  Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Config: N={args.n_blocks} blocks, d_head={args.d_head}, "
          f"d_sig={args.d_sig}, k={args.k}")
    print(f"  MRR impairments: {args.bits}-bit quantization, "
          f"{args.drift*1000:.0f}pm thermal drift\n")

    # Simulate KV cache keys
    torch.manual_seed(42)
    keys = torch.randn(n_tokens, args.d_head, device=device)
    query = torch.randn(args.d_head, device=device)

    # Create PRISM simulator
    sim = PRISMSimulator(
        d_sig=args.d_sig, k=args.k, block_size=args.block_size,
        bits=args.bits, drift=args.drift
    )

    # Register signatures (simulates MRR programming)
    sim.register_signatures(keys)
    print(f"  Simulator: {sim}\n")

    # Run selection
    top_indices, top_scores, all_scores = sim.select_with_scores(query)

    # Compute recall vs digital baseline
    recall = sim.recall_at_k(keys, query)

    # Score correlation
    n_blocks = n_tokens // args.block_size
    blocks = keys[:n_blocks * args.block_size].reshape(n_blocks, args.block_size, -1)
    sigs = blocks.mean(dim=1)
    if sim._projection is not None:
        digital_scores = (sigs @ sim._projection) @ (query @ sim._projection)
    else:
        digital_scores = sigs @ query
    corr = torch.corrcoef(
        torch.stack([all_scores.cpu(), digital_scores.cpu()]))[0, 1].item()

    # Results
    print(f"  Results:")
    print(f"    Recall@{args.k}:        {recall:.1%}")
    print(f"    Score correlation: {corr:.4f}")
    print(f"    Traffic reduction: {args.n_blocks // args.k}x "
          f"(read {args.k}/{args.n_blocks} blocks)")
    print(f"\n  Latency comparison:")
    print(f"    PRISM (photonic):  ~9 ns")
    print(f"    GPU full-scan:     ~5 us (500x slower)")
    print(f"\n  Energy comparison:")
    print(f"    PRISM:             ~2,290 pJ")
    print(f"    GPU full-scan:     ~16,300,000 pJ (7,000x more)")
    # Stage-by-stage comparison report
    sim.register_signatures(keys)
    sim.report(gpu_name="H100", measure_gpu=(device.type == "cuda"))


if __name__ == "__main__":
    main()
