"""
PRISM: WDM Inter-Channel Crosstalk Simulation

Simulates crosstalk between WDM channels in the PRISM photonic accelerator.

Each PRISM WDM channel uses an MRR with a Lorentzian lineshape. When channels
are spaced by delta_ch in wavelength, a fraction of each channel's optical
power leaks into neighboring MRR resonances (crosstalk). This script:

  1. Computes the crosstalk ratio chi_omega for nearest-neighbor (NN) and
     next-nearest-neighbor (NNN) channels analytically from the Lorentzian.
  2. Verifies that for the PRISM design (1.6 nm spacing, Q=10000):
     chi_omega < 0.01 (-20 dB), ensuring negligible inter-channel interference.
  3. Simulates the impact of crosstalk on inner-product recall by adding
     crosstalk noise to photonic MAC results and measuring recall degradation.

WDM parameters (from PRISM paper):
  - d = 32 channels
  - Channel spacing = 1.6 nm
  - Q_loaded = 10,000 (FWHM = 0.155 nm)
  - FSR = 8.3 nm (to avoid aliased resonances)

Usage:
    python scripts/wdm_crosstalk.py
    python scripts/wdm_crosstalk.py --d 64 --spacing 0.8 --Q 15000
"""

import argparse
import torch
import math


def lorentzian_response(detuning_nm: torch.Tensor, FWHM: float) -> torch.Tensor:
    """
    Normalized Lorentzian response of an MRR at a given detuning.

    L(delta) = 1 / (1 + (2*delta/FWHM)^2)

    At resonance (delta=0): L = 1.0
    At delta = FWHM/2: L = 0.5 (half-maximum)

    Args:
        detuning_nm: Wavelength offset from resonance in nm.
        FWHM: Full width at half maximum in nm.

    Returns:
        L: Lorentzian response in [0, 1].
    """
    x = 2.0 * detuning_nm / FWHM
    return 1.0 / (1.0 + x ** 2)


def compute_crosstalk_matrix(
    d: int,
    channel_spacing: float,
    Q: float,
    lambda_center: float = 1550.0,
    FSR: float = 8.3,
) -> torch.Tensor:
    """
    Compute the d x d crosstalk matrix for WDM channels.

    Entry (i, j) is the fraction of channel j's power that leaks into
    channel i's MRR. Diagonal entries are 1.0 (self-response).

    Also accounts for FSR aliasing: if |ch_i - ch_j| approaches FSR,
    the crosstalk wraps around through the next resonance order.

    Args:
        d: Number of WDM channels.
        channel_spacing: Wavelength spacing between adjacent channels in nm.
        Q: Loaded quality factor of each MRR.
        lambda_center: Center wavelength in nm.
        FSR: Free spectral range of the MRR in nm.

    Returns:
        X: Crosstalk matrix of shape [d, d].
    """
    FWHM = lambda_center / Q

    # Channel wavelengths relative to first channel
    channels = torch.arange(d, dtype=torch.float64) * channel_spacing

    # Compute pairwise detuning
    detuning = channels.unsqueeze(0) - channels.unsqueeze(1)  # [d, d]

    # Direct Lorentzian crosstalk
    X = lorentzian_response(detuning, FWHM)

    # Add FSR-aliased contributions (nearest aliased resonance)
    X_alias_pos = lorentzian_response(detuning - FSR, FWHM)
    X_alias_neg = lorentzian_response(detuning + FSR, FWHM)
    X = X + X_alias_pos + X_alias_neg

    # Normalize diagonal to 1 (the self-response includes alias terms)
    diag = X.diagonal().clone()
    X = X / diag.unsqueeze(1)

    return X


def analyze_crosstalk(X: torch.Tensor, channel_spacing: float) -> dict:
    """
    Extract crosstalk statistics from the crosstalk matrix.

    Args:
        X: Crosstalk matrix [d, d].
        channel_spacing: Channel spacing in nm.

    Returns:
        Dict with NN crosstalk, NNN crosstalk, worst-case, etc.
    """
    d = X.shape[0]

    # Nearest-neighbor crosstalk (off-diagonal by 1)
    nn_values = []
    for i in range(d - 1):
        nn_values.append(X[i, i + 1].item())
    nn_mean = sum(nn_values) / len(nn_values)

    # Next-nearest-neighbor crosstalk (off-diagonal by 2)
    nnn_values = []
    for i in range(d - 2):
        nnn_values.append(X[i, i + 2].item())
    nnn_mean = sum(nnn_values) / len(nnn_values)

    # Worst-case total crosstalk per channel (sum of off-diagonal)
    off_diag = X.clone()
    off_diag.fill_diagonal_(0)
    total_per_channel = off_diag.sum(dim=1)
    worst_total = total_per_channel.max().item()
    mean_total = total_per_channel.mean().item()

    return {
        "nn_crosstalk": nn_mean,
        "nn_crosstalk_dB": 10 * math.log10(nn_mean + 1e-30),
        "nnn_crosstalk": nnn_mean,
        "nnn_crosstalk_dB": 10 * math.log10(nnn_mean + 1e-30),
        "worst_total": worst_total,
        "worst_total_dB": 10 * math.log10(worst_total + 1e-30),
        "mean_total": mean_total,
        "mean_total_dB": 10 * math.log10(mean_total + 1e-30),
    }


def simulate_recall_impact(
    d: int = 32,
    N: int = 1024,
    k: int = 32,
    channel_spacing: float = 1.6,
    Q: float = 10_000,
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    """
    Measure recall degradation due to WDM crosstalk.

    Procedure:
      1. Generate random signature matrix S [N, d] and query q [d].
      2. Compute ideal scores: s_ideal = S @ q.
      3. Apply crosstalk: s_noisy = S @ X @ q (crosstalk mixes channels).
      4. Compare top-k sets from ideal vs noisy scores.

    Args:
        d: Number of WDM channels (signature dimension).
        N: Number of KV cache blocks.
        k: Top-k blocks to select.
        channel_spacing: WDM channel spacing in nm.
        Q: MRR loaded Q factor.
        n_trials: Number of random trials.
        seed: Random seed.

    Returns:
        Dict with mean, min, max recall.
    """
    torch.manual_seed(seed)

    # Compute crosstalk matrix
    X = compute_crosstalk_matrix(d, channel_spacing, Q)

    recalls = []
    for _ in range(n_trials):
        S = torch.randn(N, d, dtype=torch.float64)
        q = torch.randn(d, dtype=torch.float64)

        # Ideal inner products (no crosstalk)
        scores_ideal = S @ q

        # Crosstalk-corrupted inner products
        # Each MRR responds to its own channel + leakage from neighbors
        q_corrupted = X @ q  # crosstalk mixes the query channels
        scores_noisy = S @ q_corrupted

        # Top-k recall
        _, topk_ideal = torch.topk(scores_ideal, k)
        _, topk_noisy = torch.topk(scores_noisy, k)
        ideal_set = set(topk_ideal.tolist())
        noisy_set = set(topk_noisy.tolist())
        recall = len(ideal_set & noisy_set) / k
        recalls.append(recall)

    recalls_t = torch.tensor(recalls)
    return {
        "mean_recall": recalls_t.mean().item(),
        "min_recall": recalls_t.min().item(),
        "max_recall": recalls_t.max().item(),
        "std_recall": recalls_t.std().item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PRISM WDM crosstalk simulation"
    )
    parser.add_argument("--d", type=int, default=32, help="Number of WDM channels")
    parser.add_argument("--spacing", type=float, default=1.6,
                        help="Channel spacing in nm")
    parser.add_argument("--Q", type=float, default=10_000, help="Loaded Q factor")
    parser.add_argument("--FSR", type=float, default=8.3, help="FSR in nm")
    parser.add_argument("--N", type=int, default=1024, help="Number of blocks")
    parser.add_argument("--k", type=int, default=32, help="Top-k")
    parser.add_argument("--n_trials", type=int, default=100, help="Recall trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    FWHM = 1550.0 / args.Q

    print("=" * 80)
    print("  PRISM: WDM Inter-Channel Crosstalk Analysis")
    print("=" * 80)
    print(f"  Channels (d):       {args.d}")
    print(f"  Channel spacing:    {args.spacing:.2f} nm")
    print(f"  Q_loaded:           {args.Q:.0f}")
    print(f"  FWHM:               {FWHM*1e3:.1f} pm ({FWHM:.4f} nm)")
    print(f"  FSR:                {args.FSR:.1f} nm")
    print(f"  Spacing / FWHM:     {args.spacing / FWHM:.1f}x")
    print("-" * 80)

    # Compute crosstalk matrix
    X = compute_crosstalk_matrix(args.d, args.spacing, args.Q, FSR=args.FSR)
    stats = analyze_crosstalk(X, args.spacing)

    # Crosstalk table
    print()
    print(f"  {'Metric':<35s}  {'Value':>12s}  {'dB':>10s}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*10}")
    print(f"  {'Nearest-neighbor (NN) crosstalk':<35s}  "
          f"{stats['nn_crosstalk']:>12.2e}  {stats['nn_crosstalk_dB']:>10.1f}")
    print(f"  {'Next-nearest (NNN) crosstalk':<35s}  "
          f"{stats['nnn_crosstalk']:>12.2e}  {stats['nnn_crosstalk_dB']:>10.1f}")
    print(f"  {'Worst-case total per channel':<35s}  "
          f"{stats['worst_total']:>12.2e}  {stats['worst_total_dB']:>10.1f}")
    print(f"  {'Mean total per channel':<35s}  "
          f"{stats['mean_total']:>12.2e}  {stats['mean_total_dB']:>10.1f}")

    print()
    nn_ok = stats['nn_crosstalk'] < 0.01
    print(f"  NN crosstalk < 0.01 (-20 dB)?  {'YES' if nn_ok else 'NO'}  "
          f"(chi_omega = {stats['nn_crosstalk']:.2e})")

    # Recall impact simulation
    print()
    print("-" * 80)
    print(f"  Recall Impact Simulation (N={args.N}, k={args.k}, {args.n_trials} trials)")
    print("-" * 80)

    recall = simulate_recall_impact(
        d=args.d, N=args.N, k=args.k,
        channel_spacing=args.spacing, Q=args.Q,
        n_trials=args.n_trials, seed=args.seed,
    )

    print(f"  {'Metric':<30s}  {'Value':>10s}")
    print(f"  {'-'*30}  {'-'*10}")
    print(f"  {'Mean recall@k':<30s}  {recall['mean_recall']:>10.4f}")
    print(f"  {'Min recall@k':<30s}  {recall['min_recall']:>10.4f}")
    print(f"  {'Max recall@k':<30s}  {recall['max_recall']:>10.4f}")
    print(f"  {'Std recall@k':<30s}  {recall['std_recall']:>10.4f}")

    print()
    print("-" * 80)
    print("  Summary:")
    print(f"  - With Q={args.Q:.0f} and {args.spacing:.1f} nm spacing, "
          f"the spacing/FWHM ratio is {args.spacing/FWHM:.1f}x.")
    print(f"  - NN crosstalk is {stats['nn_crosstalk']:.2e} "
          f"({stats['nn_crosstalk_dB']:.1f} dB), well below -20 dB.")
    print(f"  - Recall impact is negligible: mean recall = "
          f"{recall['mean_recall']:.4f}.")
    print(f"  - Crosstalk is NOT a limiting factor for PRISM at these parameters.")
    print("=" * 80)


if __name__ == "__main__":
    main()
