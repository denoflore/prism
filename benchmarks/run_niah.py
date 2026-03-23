#!/usr/bin/env python3
"""
NIAH Benchmark with MRR-simulated block selection.

Runs Needle-in-a-Haystack evaluation on Qwen2.5-7B with PRISM's
photonic block selection (simulated via MRR impairment model).

Usage:
    # Quick test (4K context, ~2 min on RTX 4090)
    python benchmarks/run_niah.py --quick

    # Full evaluation (4K-64K, ~30 min)
    python benchmarks/run_niah.py

    # Custom config
    python benchmarks/run_niah.py --contexts 4096 8192 --k 32 --bits 5

Requirements:
    pip install torch transformers accelerate
    # ~16GB VRAM for bf16, ~8GB for 4-bit quantized
"""
import argparse
import json
import os
import time
import numpy as np

import torch


def create_niah_prompt(tokenizer, context_length, needle_position, needle, filler):
    """Create a NIAH prompt with needle at specified position."""
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)

    n_filler_needed = context_length - len(needle_tokens) - 50  # margin
    filler_repeated = (filler_tokens * (n_filler_needed // len(filler_tokens) + 1))[:n_filler_needed]

    insert_pos = int(len(filler_repeated) * needle_position)
    tokens = filler_repeated[:insert_pos] + needle_tokens + filler_repeated[insert_pos:]
    return tokens[:context_length]


def mrr_block_select(keys, query, k, block_size, d_sig, bits, drift_sigma):
    """Simulate PRISM block selection with MRR impairments."""
    n_tokens = keys.shape[0]
    n_blocks = n_tokens // block_size

    # Compute block signatures (mean key)
    blocks = keys[:n_blocks * block_size].reshape(n_blocks, block_size, -1)
    sigs = blocks.mean(dim=1)  # [n_blocks, d_head]

    # Project to d_sig dimensions (random projection)
    if sigs.shape[1] > d_sig:
        proj = torch.randn(sigs.shape[1], d_sig, device=sigs.device, dtype=sigs.dtype)
        proj /= proj.norm(dim=0, keepdim=True)
        sigs = sigs @ proj
        q = query @ proj
    else:
        q = query

    # MRR impairments
    sigs_np = sigs.float().cpu().numpy()
    q_np = q.float().cpu().numpy()

    # Normalize to [0,1]
    w_min, w_max = sigs_np.min(), sigs_np.max()
    W = (sigs_np - w_min) / (w_max - w_min + 1e-30)

    # Quantize
    levels = 2 ** bits
    W_q = np.round(W * (levels - 1)) / (levels - 1)

    # Thermal drift
    W_d = np.clip(W_q + np.random.randn(*W_q.shape) * drift_sigma, 0, 1)

    # Reconstruct and compute scores
    S_recon = W_d * (w_max - w_min) + w_min
    scores = S_recon @ q_np

    # Top-k
    top_indices = np.argsort(-scores)[:k]
    return top_indices, n_blocks


def main():
    parser = argparse.ArgumentParser(description="NIAH with MRR block selection")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--contexts", nargs="+", type=int,
                        default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--n_positions", type=int, default=11)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--d_sig", type=int, default=32)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--drift", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 4K context only, 5 positions")
    parser.add_argument("--output", default="results/niah_results.json")
    args = parser.parse_args()

    if args.quick:
        args.contexts = [4096]
        args.n_positions = 5

    print("=" * 60)
    print("  NIAH Benchmark with PRISM Block Selection")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Contexts: {args.contexts}")
    print(f"  k={args.k}, B={args.block_size}, d={args.d_sig}")
    print(f"  MRR: {args.bits}-bit, drift={args.drift}")
    print()

    # Load model
    print("Loading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    needle = "The special magic number is 7392158."
    filler = (
        "This is a passage of filler text used to pad the context window. "
        "It contains no important information and serves only to increase "
        "the total token count. Modern technology advances rapidly. "
    )
    question = "What is the special magic number mentioned in the text?"

    results = {}
    for ctx_len in args.contexts:
        positions = np.linspace(0.05, 0.95, args.n_positions)
        hits = []

        print(f"\nContext {ctx_len//1000}K:")
        for pos in positions:
            # Create prompt
            tokens = create_niah_prompt(tokenizer, ctx_len, pos, needle, filler)
            input_ids = torch.tensor([tokens], device=device)

            # Forward pass to get KV cache
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
                past_kv = outputs.past_key_values

            # Extract keys from layer 14 (mid-layer), head 0
            layer_keys = past_kv[14][0][0, 0]  # [seq_len, d_head]

            # Query = last token's query
            q_tokens = torch.tensor([[tokens[-1]]], device=device)
            with torch.no_grad():
                q_out = model(q_tokens, past_key_values=past_kv, use_cache=False)

            # Use last hidden state as proxy query
            query_vec = layer_keys[-1]  # simplified

            # MRR block selection
            top_idx, n_blocks = mrr_block_select(
                layer_keys, query_vec, args.k, args.block_size,
                args.d_sig, args.bits, args.drift
            )

            # Check if needle block is selected
            needle_token_pos = int(len(tokens) * pos)
            needle_block = needle_token_pos // args.block_size
            hit = needle_block in top_idx

            hits.append(hit)
            status = "HIT" if hit else "MISS"
            print(f"  pos={pos:.2f} needle_block={needle_block}/{n_blocks} → {status}")

            # Cleanup
            del past_kv, outputs
            torch.cuda.empty_cache()

        accuracy = sum(hits) / len(hits) * 100
        results[str(ctx_len)] = {
            "accuracy": accuracy,
            "hits": sum(hits),
            "total": len(hits),
            "config": {
                "k": args.k, "B": args.block_size,
                "d": args.d_sig, "bits": args.bits, "drift": args.drift
            }
        }
        print(f"  Accuracy: {accuracy:.0f}% ({sum(hits)}/{len(hits)})")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print(f"\n{'SUMMARY':=^60}")
    for ctx, r in results.items():
        print(f"  {int(ctx)//1000}K: {r['accuracy']:.0f}% "
              f"({r['hits']}/{r['total']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
