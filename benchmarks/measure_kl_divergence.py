"""
PRISM: KL Divergence Between Full and Block-Sparse Attention

Measures the KL divergence between the output distributions of full attention
and PRISM block-sparse attention, validating that block selection preserves
the attention distribution with minimal distortion.

Procedure:
  1. Generate random Q, K, V tensors (no LLM needed).
  2. Compute full attention: softmax(QK^T / sqrt(d_head)) @ V.
  3. Use PRISMSimulator to select top-k blocks.
  4. Compute block-sparse attention: same softmax but only over selected
     blocks + a recent-window of the most recent blocks.
  5. Measure KL(full || sparse) per head.

The key insight: if PRISM selects the right blocks (high recall), the
sparse attention distribution closely approximates the full distribution,
and KL divergence is near zero.

Parameters (defaults match PRISM paper):
  - N = 1024 blocks (131,072 tokens at B=128)
  - B = 128 tokens per block
  - d_head = 128 (head dimension)
  - k = 32 top blocks selected by PRISM
  - d_sig = 32 (signature/WDM channels)

Usage:
    python benchmarks/measure_kl_divergence.py
    python benchmarks/measure_kl_divergence.py --N 512 --k 64 --n_heads 8
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prism.simulator import PRISMSimulator


def full_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    d_head: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute full (dense) attention output and attention weights.

    Computes: softmax(Q @ K^T / sqrt(d_head)) @ V

    Args:
        Q: Query tensor [1, d_head] (single query token, decode step).
        K: Key tensor [seq_len, d_head].
        V: Value tensor [seq_len, d_head].
        d_head: Head dimension for scaling.

    Returns:
        output: Attention output [1, d_head].
        attn_weights: Attention distribution [1, seq_len] (probabilities).
    """
    # Q @ K^T -> [1, seq_len]
    scores = Q @ K.T / (d_head ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)  # [1, seq_len]
    output = attn_weights @ V  # [1, d_head]
    return output, attn_weights


def block_sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    d_head: int,
    selected_indices: torch.Tensor,
    block_size: int,
    recent_window: int = 4,
    N_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute block-sparse attention using only selected blocks + recent window.

    Only attends to:
      1. Top-k blocks selected by PRISM (via selected_indices).
      2. The most recent `recent_window` blocks (always included for
         local context, regardless of PRISM selection).

    Tokens not in any selected/recent block get zero attention weight.

    Args:
        Q: Query tensor [1, d_head].
        K: Key tensor [seq_len, d_head].
        V: Value tensor [seq_len, d_head].
        d_head: Head dimension.
        selected_indices: Block indices from PRISM [k].
        block_size: Tokens per block.
        recent_window: Number of recent blocks always included.
        N_blocks: Total number of blocks.

    Returns:
        output: Attention output [1, d_head].
        attn_weights: Sparse attention distribution [1, seq_len].
    """
    seq_len = K.shape[0]

    # Build set of active block indices (PRISM + recent window)
    active_blocks = set(selected_indices.cpu().tolist())
    for i in range(max(0, N_blocks - recent_window), N_blocks):
        active_blocks.add(i)

    # Build mask: True for tokens in active blocks
    mask = torch.zeros(seq_len, dtype=torch.bool, device=K.device)
    for blk_idx in active_blocks:
        start = blk_idx * block_size
        end = min(start + block_size, seq_len)
        mask[start:end] = True

    # Compute attention only on masked positions
    scores = Q @ K.T / (d_head ** 0.5)  # [1, seq_len]

    # Set non-selected positions to -inf before softmax
    scores_masked = scores.clone()
    scores_masked[:, ~mask] = float('-inf')

    attn_weights = F.softmax(scores_masked, dim=-1)  # [1, seq_len]
    # Replace NaN (all -inf) with zeros
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = attn_weights @ V  # [1, d_head]
    return output, attn_weights


def kl_divergence_attention(
    full_weights: torch.Tensor,
    sparse_weights: torch.Tensor,
    eps: float = 1e-10,
) -> float:
    """
    Compute KL(full || sparse) for attention weight distributions.

    KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))

    Where P = full attention weights, Q = sparse attention weights.
    We add eps to avoid log(0).

    Args:
        full_weights: Full attention distribution [1, seq_len].
        sparse_weights: Sparse attention distribution [1, seq_len].
        eps: Small constant for numerical stability.

    Returns:
        KL divergence (scalar, non-negative).
    """
    P = full_weights.squeeze() + eps
    Q_dist = sparse_weights.squeeze() + eps
    # Renormalize after adding eps
    P = P / P.sum()
    Q_dist = Q_dist / Q_dist.sum()

    kl = (P * (P.log() - Q_dist.log())).sum().item()
    return max(0.0, kl)  # Clamp numerical noise


def run_kl_measurement(
    N: int = 1024,
    B: int = 128,
    d_head: int = 128,
    k: int = 32,
    d_sig: int = 32,
    n_heads: int = 16,
    recent_window: int = 4,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Measure KL divergence across multiple simulated attention heads.

    For each head:
      1. Generate random K, V (one block's worth per block, N blocks).
      2. Generate a random query Q.
      3. Use PRISMSimulator to select top-k blocks.
      4. Compute full and sparse attention.
      5. Measure KL(full || sparse).

    Args:
        N: Number of KV cache blocks.
        B: Block size (tokens per block).
        d_head: Head dimension.
        k: Top-k blocks for PRISM.
        d_sig: Signature dimension (WDM channels).
        n_heads: Number of attention heads to simulate.
        recent_window: Recent blocks always included.
        seed: Random seed.
        device: PyTorch device string.

    Returns:
        Dict with per-head KL values and summary statistics.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    # Clamp k to be at most N - recent_window (to leave room)
    effective_k = min(k, N - recent_window)

    kl_values = []

    for head_idx in range(n_heads):
        # Generate random KV cache: N blocks of B tokens each
        seq_len = N * B
        K = torch.randn(seq_len, d_head, device=dev, dtype=torch.float32)
        V = torch.randn(seq_len, d_head, device=dev, dtype=torch.float32)
        Q = torch.randn(1, d_head, device=dev, dtype=torch.float32)

        # Use PRISM to select top-k blocks
        sim = PRISMSimulator(
            d_sig=d_sig, k=effective_k, block_size=B,
            bits=5, drift=0.01, det_noise=0.01,
        )
        sim.register_signatures(K)
        selected = sim.select(Q.squeeze())  # [k]

        # Full attention
        _, full_weights = full_attention(Q, K, V, d_head)

        # Block-sparse attention
        _, sparse_weights = block_sparse_attention(
            Q, K, V, d_head, selected, B,
            recent_window=recent_window, N_blocks=N,
        )

        # KL divergence
        kl = kl_divergence_attention(full_weights, sparse_weights)
        kl_values.append(kl)

    kl_tensor = torch.tensor(kl_values)

    return {
        "kl_values": kl_values,
        "mean": kl_tensor.mean().item(),
        "median": kl_tensor.median().item(),
        "std": kl_tensor.std().item(),
        "min": kl_tensor.min().item(),
        "max": kl_tensor.max().item(),
        "p99": kl_tensor.quantile(0.99).item() if len(kl_values) >= 2 else kl_tensor.max().item(),
        "n_heads": n_heads,
        "N": N,
        "B": B,
        "k": effective_k,
        "d_sig": d_sig,
        "d_head": d_head,
        "recent_window": recent_window,
    }


def print_results(results: dict) -> None:
    """Print formatted KL divergence measurement results."""
    print("=" * 75)
    print("  PRISM: KL Divergence - Full vs Block-Sparse Attention")
    print("=" * 75)
    print(f"  Blocks (N):         {results['N']}")
    print(f"  Block size (B):     {results['B']} tokens")
    print(f"  Sequence length:    {results['N'] * results['B']:,} tokens")
    print(f"  Head dimension:     {results['d_head']}")
    print(f"  Signature dim:      {results['d_sig']} (WDM channels)")
    print(f"  Top-k blocks:       {results['k']}")
    print(f"  Recent window:      {results['recent_window']} blocks")
    print(f"  Heads simulated:    {results['n_heads']}")
    total_attended = (results['k'] + results['recent_window'])
    sparsity = total_attended / results['N']
    print(f"  Attention sparsity: {total_attended}/{results['N']} blocks "
          f"({sparsity:.1%} of context)")
    print("-" * 75)

    # Per-head table
    print()
    print(f"  {'Head':>6s}  {'KL(full||sparse)':>18s}  {'Quality'}")
    print(f"  {'------':>6s}  {'------------------':>18s}  {'-------'}")
    for i, kl in enumerate(results["kl_values"]):
        if kl < 0.01:
            quality = "excellent"
        elif kl < 0.1:
            quality = "good"
        elif kl < 0.5:
            quality = "moderate"
        else:
            quality = "poor"
        print(f"  {i:>6d}  {kl:>18.6f}  {quality}")

    # Summary
    print()
    print("-" * 75)
    print(f"  {'Statistic':<25s}  {'KL Divergence':>15s}")
    print(f"  {'-'*25}  {'-'*15}")
    print(f"  {'Mean':<25s}  {results['mean']:>15.6f}")
    print(f"  {'Median':<25s}  {results['median']:>15.6f}")
    print(f"  {'Std':<25s}  {results['std']:>15.6f}")
    print(f"  {'Min':<25s}  {results['min']:>15.6f}")
    print(f"  {'Max':<25s}  {results['max']:>15.6f}")
    print(f"  {'P99':<25s}  {results['p99']:>15.6f}")

    print()
    print("-" * 75)
    if results['mean'] < 0.1:
        print("  Conclusion: Block-sparse attention closely approximates full attention.")
        print(f"  Mean KL = {results['mean']:.6f} indicates minimal distribution shift.")
        print("  PRISM block selection preserves attention quality at this sparsity level.")
    else:
        print("  Conclusion: Significant KL divergence detected.")
        print("  Consider increasing k or reducing N for better approximation.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Measure KL divergence between full and PRISM block-sparse attention"
    )
    parser.add_argument("--N", type=int, default=1024,
                        help="Number of KV cache blocks (default: 1024)")
    parser.add_argument("--B", type=int, default=128,
                        help="Block size in tokens (default: 128)")
    parser.add_argument("--d_head", type=int, default=128,
                        help="Head dimension (default: 128)")
    parser.add_argument("--k", type=int, default=32,
                        help="Top-k blocks to select (default: 32)")
    parser.add_argument("--d_sig", type=int, default=32,
                        help="Signature dimension / WDM channels (default: 32)")
    parser.add_argument("--n_heads", type=int, default=16,
                        help="Number of heads to simulate (default: 16)")
    parser.add_argument("--recent_window", type=int, default=4,
                        help="Recent blocks always included (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    args = parser.parse_args()

    results = run_kl_measurement(
        N=args.N, B=args.B, d_head=args.d_head,
        k=args.k, d_sig=args.d_sig, n_heads=args.n_heads,
        recent_window=args.recent_window,
        seed=args.seed, device=args.device,
    )
    print_results(results)


if __name__ == "__main__":
    main()
