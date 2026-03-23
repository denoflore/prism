"""
Phase 1: Compare recall of unsigned vs signed inner products for PRISM.

Three modes:
  A) ReLU projection (current PRISM): sigma = max(0, R*mean_k), w in [0,1]
  B) Split encoding: sigma = [sigma+; sigma-], w in [0,1], 2d MRRs
  C) Signed balanced PD (proposed): sigma = R*mean_k (signed), w in [-1,1], d MRRs

This script validates whether switching to add-drop + balanced PD
improves recall before modifying the entire paper.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_config import (OUT_DIR, COLOR_PRIMARY, COLOR_SECONDARY,
                         COLOR_TERTIARY, COLOR_THRESHOLD)

# ── Parameters ──
D_HEAD = 128       # head dimension (Qwen2.5-7B)
N_BLOCKS = 1024    # number of KV cache blocks
B = 128            # block size
K = 32             # top-k
N_TRIALS = 200     # Monte Carlo trials
SEED = 42
DIMS = [8, 16, 32, 64]  # signature dimensions to test
QUANT_BITS = 5     # MRR weight quantization


def generate_data(n_blocks, d_head, rng):
    """Generate random KV cache blocks and a query."""
    keys = rng.standard_normal((n_blocks, d_head))
    # Mean key per block (signature source)
    mean_keys = keys  # already block-level
    query = rng.standard_normal(d_head)
    # Ground truth: exact inner products
    exact_scores = mean_keys @ query
    exact_topk = set(np.argsort(-exact_scores)[:K])
    return mean_keys, query, exact_scores, exact_topk


def random_projection_matrix(d_sig, d_head, rng):
    """Random Gaussian projection matrix."""
    R = rng.standard_normal((d_sig, d_head)) / np.sqrt(d_sig)
    return R


def quantize_unsigned(w, bits):
    """Quantize weights in [0,1] to given bit precision."""
    levels = 2**bits
    w_clip = np.clip(w, 0, 1)
    return np.round(w_clip * (levels - 1)) / (levels - 1)


def quantize_signed(w, bits):
    """Quantize weights in [-1,1] to given bit precision."""
    levels = 2**bits
    w_clip = np.clip(w, -1, 1)
    # Map [-1,1] to [0, levels-1], quantize, map back
    return np.round((w_clip + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1


def mode_a_relu(mean_keys, query, R, bits):
    """Mode A: ReLU projection, unsigned [0,1] weights."""
    # Project and apply ReLU
    sigs = np.maximum(0, mean_keys @ R.T)  # (N, d_sig), non-negative
    q_proj = np.maximum(0, R @ query)       # (d_sig,), non-negative

    # Normalize signatures to [0,1] for MRR
    s_max = sigs.max()
    if s_max > 0:
        sigs_norm = sigs / s_max
    else:
        sigs_norm = sigs

    # Quantize
    sigs_q = quantize_unsigned(sigs_norm, bits)

    # Inner product (approximate)
    scores = sigs_q @ q_proj
    return scores


def mode_b_split(mean_keys, query, R, bits):
    """Mode B: Split encoding, unsigned [0,1] weights, 2d MRRs."""
    # Project (signed)
    sigs = mean_keys @ R.T   # (N, d_sig)
    q_proj = R @ query        # (d_sig,)

    # Split into positive and negative parts
    sigs_pos = np.maximum(0, sigs)
    sigs_neg = np.maximum(0, -sigs)
    q_pos = np.maximum(0, q_proj)
    q_neg = np.maximum(0, -q_proj)

    # Normalize each half to [0,1]
    s_max = max(sigs_pos.max(), sigs_neg.max(), 1e-30)
    sigs_pos_n = sigs_pos / s_max
    sigs_neg_n = sigs_neg / s_max
    q_max = max(q_pos.max(), q_neg.max(), 1e-30)

    # Quantize
    sigs_pos_q = quantize_unsigned(sigs_pos_n, bits)
    sigs_neg_q = quantize_unsigned(sigs_neg_n, bits)

    # Reconstruct: score = sig+ . q+ + sig- . q- (both terms positive)
    # This approximates the signed inner product
    scores = sigs_pos_q @ q_pos + sigs_neg_q @ q_neg
    return scores


def mode_c_signed(mean_keys, query, R, bits):
    """Mode C: Signed balanced PD, weights in [-1,1], d MRRs."""
    # Project (signed, no ReLU)
    sigs = mean_keys @ R.T   # (N, d_sig)
    q_proj = R @ query        # (d_sig,)

    # Normalize signatures to [-1,1] (symmetric)
    s_abs_max = max(np.abs(sigs).max(), 1e-30)
    sigs_norm = sigs / s_abs_max

    # Quantize in [-1,1]
    sigs_q = quantize_signed(sigs_norm, bits)

    # Inner product (signed, balanced PD)
    scores = sigs_q @ q_proj
    return scores


def compute_recall(exact_topk, approx_scores, k=K):
    """Recall@K."""
    approx_topk = set(np.argsort(-approx_scores)[:k])
    return len(exact_topk & approx_topk) / k


def run_experiment():
    """Run the full comparison experiment."""
    rng = np.random.default_rng(SEED)

    results = {}

    for d_sig in DIMS:
        recalls_a = []
        recalls_b = []
        recalls_c = []

        for trial in range(N_TRIALS):
            # Generate data
            mean_keys, query, exact_scores, exact_topk = generate_data(
                N_BLOCKS, D_HEAD, rng)

            # Random projection matrix
            R = random_projection_matrix(d_sig, D_HEAD, rng)

            # Mode A: ReLU
            scores_a = mode_a_relu(mean_keys, query, R, QUANT_BITS)
            recalls_a.append(compute_recall(exact_topk, scores_a))

            # Mode B: Split encoding
            scores_b = mode_b_split(mean_keys, query, R, QUANT_BITS)
            recalls_b.append(compute_recall(exact_topk, scores_b))

            # Mode C: Signed balanced PD
            scores_c = mode_c_signed(mean_keys, query, R, QUANT_BITS)
            recalls_c.append(compute_recall(exact_topk, scores_c))

        results[d_sig] = {
            'relu_mean': np.mean(recalls_a),
            'relu_std': np.std(recalls_a),
            'split_mean': np.mean(recalls_b),
            'split_std': np.std(recalls_b),
            'signed_mean': np.mean(recalls_c),
            'signed_std': np.std(recalls_c),
        }

        print(f"d={d_sig:3d}  "
              f"ReLU={np.mean(recalls_a):.3f}±{np.std(recalls_a):.3f}  "
              f"Split={np.mean(recalls_b):.3f}±{np.std(recalls_b):.3f}  "
              f"Signed={np.mean(recalls_c):.3f}±{np.std(recalls_c):.3f}")

    return results


def plot_results(results):
    """Plot recall comparison."""
    dims = sorted(results.keys())
    relu_means = [results[d]['relu_mean'] for d in dims]
    split_means = [results[d]['split_mean'] for d in dims]
    signed_means = [results[d]['signed_mean'] for d in dims]
    relu_stds = [results[d]['relu_std'] for d in dims]
    split_stds = [results[d]['split_std'] for d in dims]
    signed_stds = [results[d]['signed_std'] for d in dims]

    fig, ax = plt.subplots(figsize=(3.35, 2.5))

    ax.errorbar(dims, relu_means, yerr=relu_stds, fmt='o-',
                color=COLOR_THRESHOLD, label='ReLU [0,1] (current)',
                markerfacecolor='white', markeredgewidth=1.0, capsize=3)
    ax.errorbar(dims, split_means, yerr=split_stds, fmt='s--',
                color=COLOR_TERTIARY, label='Split [0,1] (2d MRRs)',
                markerfacecolor='white', markeredgewidth=1.0, capsize=3)
    ax.errorbar(dims, signed_means, yerr=signed_stds, fmt='^-',
                color=COLOR_PRIMARY, label='Signed [-1,1] (balanced PD)',
                markerfacecolor='white', markeredgewidth=1.0, capsize=3,
                linewidth=1.5)

    ax.set_xlabel('Signature dimension $d$')
    ax.set_ylabel(f'Recall@{K}')
    ax.set_ylim(0, 0.45)
    ax.set_xticks(dims)
    ax.legend(loc='upper right', fontsize=6.5)

    ax.text(0.03, 0.97,
            f'$N={N_BLOCKS}$, $K={K}$, {QUANT_BITS}-bit\n{N_TRIALS} trials',
            transform=ax.transAxes, ha='left', va='top', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                      edgecolor='black', linewidth=0.4, alpha=0.9))

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_signed_recall_comparison.pdf')
    fig.savefig(fig_path)
    print(f'Saved: {fig_path}')
    plt.close(fig)


def main():
    print("=" * 60)
    print("PRISM Signed vs Unsigned Recall Comparison")
    print(f"N_BLOCKS={N_BLOCKS}, K={K}, bits={QUANT_BITS}, trials={N_TRIALS}")
    print("=" * 60)

    results = run_experiment()

    # Save JSON
    json_path = os.path.join(OUT_DIR, '..', 'results',
                             'signed_recall_comparison.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # Convert numpy types for JSON serialization
    json_results = {str(k): v for k, v in results.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'Results saved: {json_path}')

    plot_results(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (d=32):")
    r32 = results[32]
    print(f"  ReLU [0,1]:      Recall@{K} = {r32['relu_mean']:.1%}")
    print(f"  Split [0,1]:     Recall@{K} = {r32['split_mean']:.1%}")
    print(f"  Signed [-1,1]:   Recall@{K} = {r32['signed_mean']:.1%}")
    improvement = (r32['signed_mean'] - r32['relu_mean']) / r32['relu_mean']
    print(f"  Improvement (signed vs ReLU): {improvement:+.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
