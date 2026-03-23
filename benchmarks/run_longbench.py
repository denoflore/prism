#!/usr/bin/env python3
"""
LongBench-v2 Benchmark with PRISM Block Selection.

Compares full-attention vs PRISM block-sparse attention on LongBench-v2
(multiple-choice, 503 samples across 6 domains).

Usage:
    # Quick test (~20 min on RTX 5880)
    python benchmarks/run_longbench.py --quick

    # Full evaluation
    python benchmarks/run_longbench.py

    # Specific domains
    python benchmarks/run_longbench.py --domains "Multi-Document QA" "Single-Document QA"

Requirements:
    pip install torch transformers datasets accelerate
"""
import argparse
import json
import os
import time
import gc
import re

import numpy as np
import torch


def prism_block_select(hidden_states, query, k, block_size, d_sig, bits, drift_sigma):
    """Simulate PRISM photonic block selection with MRR impairments."""
    n_tokens = hidden_states.shape[0]
    n_blocks = n_tokens // block_size
    if n_blocks == 0:
        return np.arange(0)

    blocks = hidden_states[:n_blocks * block_size].reshape(n_blocks, block_size, -1)
    sigs = blocks.mean(dim=1).float().cpu().numpy()

    d_head = sigs.shape[1]
    rng = np.random.default_rng(42)
    if d_head > d_sig:
        proj = rng.standard_normal((d_head, d_sig)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        sigs = sigs @ proj
        q = query.float().cpu().numpy() @ proj
    else:
        q = query.float().cpu().numpy()

    w_min, w_max = sigs.min(), sigs.max()
    W = (sigs - w_min) / (w_max - w_min + 1e-30)
    levels = 2 ** bits
    W_q = np.round(W * (levels - 1)) / (levels - 1)
    W_d = np.clip(W_q + rng.standard_normal(W_q.shape) * drift_sigma, 0, 1)
    S_recon = W_d * (w_max - w_min) + w_min
    scores = S_recon @ q
    return np.argsort(-scores)[:k]


def build_sparse_token_mask(seq_len, block_indices, block_size, window=256):
    """Boolean mask: True = attend to this token."""
    mask = np.zeros(seq_len, dtype=bool)
    mask[max(0, seq_len - window):] = True
    for idx in block_indices:
        s = idx * block_size
        e = min(s + block_size, seq_len)
        mask[s:e] = True
    return mask


def run_mcq(model, tokenizer, prompt, choices, device, max_len=4096):
    """Run multiple-choice by scoring each choice."""
    scores = []
    for choice in choices:
        full = prompt + " " + choice
        ids = tokenizer.encode(full, return_tensors="pt",
                               truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            out = model(ids)
            # Score = log-prob of last token
            logits = out.logits[0, -1]
            score = logits.max().item()
        scores.append(score)
        del ids, out
    return ["A", "B", "C", "D"][np.argmax(scores)]


def run_mcq_sparse(model, tokenizer, prompt, choices, block_indices,
                   block_size, device, max_len=4096, window=256):
    """Run MCQ with block-sparse input (PRISM selection)."""
    scores = []
    for choice in choices:
        full = prompt + " " + choice
        ids = tokenizer.encode(full, return_tensors="pt",
                               truncation=True, max_length=max_len).to(device)
        seq_len = ids.shape[1]
        mask = build_sparse_token_mask(seq_len, block_indices, block_size, window)
        keep = torch.tensor(np.where(mask)[0], device=device)
        sparse_ids = ids[0, keep].unsqueeze(0)

        with torch.no_grad():
            out = model(sparse_ids)
            logits = out.logits[0, -1]
            score = logits.max().item()
        scores.append(score)
        del ids, sparse_ids, out
    return ["A", "B", "C", "D"][np.argmax(scores)]


def main():
    parser = argparse.ArgumentParser(description="LongBench-v2 with PRISM")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Filter by domain. Default: Multi-Document QA, Single-Document QA, Long In-context Learning")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--d_sig", type=int, default=32)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--drift", type=float, default=0.01)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="results/longbench_results.json")
    args = parser.parse_args()

    if args.quick:
        args.max_samples = 30
        args.max_len = 4096

    default_domains = ["Multi-Document QA", "Single-Document QA",
                       "Long In-context Learning"]
    domains = args.domains or default_domains

    print("=" * 65)
    print("  LongBench-v2 with PRISM Block Selection")
    print("=" * 65)
    print(f"  Model: {args.model}")
    print(f"  Domains: {domains}")
    print(f"  PRISM: k={args.k}, B={args.block_size}, d={args.d_sig}")
    print(f"  Max samples: {args.max_samples}, max_len: {args.max_len}")
    print()

    # Load model
    print("Loading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    print(f"Loaded on {device}\n")

    # Load dataset
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    print(f"LongBench-v2: {len(ds)} total samples\n")

    results = {}
    total_full_correct = 0
    total_prism_correct = 0
    total_count = 0

    for domain in domains:
        domain_samples = [s for s in ds if s["domain"] == domain]
        if not domain_samples:
            print(f"No samples for domain: {domain}")
            continue

        n = min(len(domain_samples), args.max_samples)
        domain_samples = domain_samples[:n]

        full_correct = 0
        prism_correct = 0
        evaluated = 0

        print(f"{'='*65}")
        print(f"  Domain: {domain} ({n} samples)")
        print(f"{'='*65}")

        for i, sample in enumerate(domain_samples):
            context = sample["context"]
            question = sample["question"]
            choices = [sample["choice_A"], sample["choice_B"],
                       sample["choice_C"], sample["choice_D"]]
            answer = sample["answer"]  # A/B/C/D

            prompt = f"Read the following text and answer the question.\n\n{context}\n\nQuestion: {question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer:"

            try:
                # A. Full attention
                pred_full = run_mcq(model, tokenizer, prompt, choices,
                                    device, args.max_len)
                if pred_full == answer:
                    full_correct += 1

                # B. PRISM block selection
                ids = tokenizer.encode(prompt, return_tensors="pt",
                                       truncation=True,
                                       max_length=args.max_len).to(device)
                with torch.no_grad():
                    out = model(ids, output_hidden_states=True, use_cache=False)
                    hidden = out.hidden_states[14][0]  # layer 14

                block_indices = prism_block_select(
                    hidden, hidden[-1], args.k, args.block_size,
                    args.d_sig, args.bits, args.drift)

                pred_prism = run_mcq_sparse(
                    model, tokenizer, prompt, choices, block_indices,
                    args.block_size, device, args.max_len)
                if pred_prism == answer:
                    prism_correct += 1

                evaluated += 1
                if evaluated % 10 == 0:
                    print(f"  {evaluated}/{n} | Full: {full_correct/evaluated:.1%} | "
                          f"PRISM: {prism_correct/evaluated:.1%}")

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at {i}, skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error at {i}: {str(e)[:80]}")
            finally:
                torch.cuda.empty_cache()
                gc.collect()

        if evaluated > 0:
            acc_full = full_correct / evaluated * 100
            acc_prism = prism_correct / evaluated * 100
            drop = acc_full - acc_prism
            results[domain] = {
                "full_attention": round(acc_full, 1),
                "prism_selection": round(acc_prism, 1),
                "drop": round(drop, 1),
                "n_samples": evaluated,
            }
            total_full_correct += full_correct
            total_prism_correct += prism_correct
            total_count += evaluated
            print(f"  Result: Full={acc_full:.1f}% | PRISM={acc_prism:.1f}% | "
                  f"drop={drop:+.1f}%\n")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'SUMMARY':=^65}")
    print(f"  {'Domain':<35} {'Full':>8} {'PRISM':>8} {'Drop':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for domain, r in results.items():
        print(f"  {domain:<35} {r['full_attention']:>7.1f}% "
              f"{r['prism_selection']:>7.1f}% {r['drop']:>+7.1f}%")
    if total_count > 0:
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'OVERALL':<35} "
              f"{total_full_correct/total_count*100:>7.1f}% "
              f"{total_prism_correct/total_count*100:>7.1f}% "
              f"{(total_full_correct-total_prism_correct)/total_count*100:>+7.1f}%")
    print(f"{'='*65}")
    print(f"  PRISM: k={args.k}, B={args.block_size}, d={args.d_sig}, "
          f"{args.bits}-bit, drift={args.drift}")
    print(f"  Model: {args.model}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
