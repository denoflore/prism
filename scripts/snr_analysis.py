#!/usr/bin/env python3
"""
Fig: SNR and Recall Analysis (two-panel).
  (a) SNR vs signature dimension d for three bank sizes (N=64, 256, 1024).
  (b) Recall@8 vs SNR with reliable-region shading.
Output: figures/fig_snr_analysis.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import (
    OUT_DIR, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY,
    COLOR_THRESHOLD,
)

# ── Physical constants (from generate_photonic_figures.py fig5) ──
q = 1.602e-19    # C
kB = 1.381e-23   # J/K
T = 300.0        # K
R_L = 50.0       # ohm
BW = 1e9         # 1 GHz
RESPONSIVITY = 1.0
TIA_NOISE = 10e-9  # A/sqrt(Hz)

# Laser power for SNR calculation
P_LASER_DBM = 13.0

# Loss model
WG_LOSS = 1.0
MOD_IL = 3.0
CONN_LOSS = 1.5

def splitter_loss(N):
    return 10.0 * np.log10(N) + 0.2 * np.ceil(np.log2(N))

def mrr_il(d):
    return 0.1 * d

def compute_snr(N, d_arr):
    """Compute electrical SNR (dB) for bank size N over array of d values."""
    loss = splitter_loss(N) + WG_LOSS + mrr_il(d_arr) + MOD_IL + CONN_LOSS
    P_rx_dBm = P_LASER_DBM - loss
    P_rx_W = 1e-3 * 10.0 ** (P_rx_dBm / 10.0)
    I_photo = RESPONSIVITY * P_rx_W

    i_shot = np.sqrt(2 * q * I_photo * BW)
    i_thermal = np.sqrt(4 * kB * T * BW / R_L)
    i_tia = TIA_NOISE * np.sqrt(BW)
    i_total = np.sqrt(i_shot**2 + i_thermal**2 + i_tia**2)

    return 20.0 * np.log10(I_photo / i_total)


def simulate_recall(snr_sweep, n_blocks=1024, d_sim=64, k=32, n_trials=1000,
                    seed=42):
    """Monte Carlo recall@k simulation as function of SNR."""
    rng = np.random.default_rng(seed)
    sigs = rng.standard_normal((n_blocks, d_sim)).astype(np.float32)
    sigs /= np.linalg.norm(sigs, axis=1, keepdims=True)
    query = rng.standard_normal(d_sim).astype(np.float32)
    query /= np.linalg.norm(query)

    true_sim = sigs @ query
    true_topk = set(np.argsort(true_sim)[-k:])

    recall_mean = []
    recall_std = []
    for snr_dB in snr_sweep:
        snr_lin = 10.0 ** (snr_dB / 20.0)
        sig_power = np.std(true_sim)
        noise_std = sig_power / snr_lin
        recalls = []
        for _ in range(n_trials):
            noisy = true_sim + rng.normal(0, noise_std, size=n_blocks)
            noisy_topk = set(np.argsort(noisy)[-k:])
            recalls.append(len(true_topk & noisy_topk) / k)
        recall_mean.append(np.mean(recalls))
        recall_std.append(np.std(recalls))
    return np.array(recall_mean), np.array(recall_std)


def main():
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.4, 5.0))

    # ── Panel (a): SNR vs d for three bank sizes ──
    d_arr = np.arange(4, 129)
    bank_sizes = [64, 256, 1024]
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY]

    for N, color in zip(bank_sizes, colors):
        snr = compute_snr(N, d_arr)
        ax_a.plot(d_arr, snr, color=color, linewidth=1.2,
                  label=f"$N = {N}$")

    # 20 dB threshold (may be outside visible range)
    ax_a.axhline(20, color=COLOR_THRESHOLD, ls="--", lw=0.8, alpha=0.8)

    ax_a.set_xlabel("Signature dimension $d$")
    ax_a.set_ylabel("SNR (dB)")
    ax_a.set_xlim(4, 128)
    ax_a.set_ylim(top=0)
    ax_a.legend(loc="upper right", fontsize=7)
    ax_a.text(-0.14, 1.04, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold")

    # ── Panel (b): Recall@32 vs SNR ──
    snr_sweep = np.arange(0, 41, 1.0)
    recall_mean, recall_std = simulate_recall(snr_sweep)

    ax_b.plot(snr_sweep, recall_mean * 100, color=COLOR_PRIMARY, linewidth=1.2)
    ax_b.fill_between(snr_sweep,
                      np.maximum((recall_mean - recall_std) * 100, 0),
                      np.minimum((recall_mean + recall_std) * 100, 100),
                      color=COLOR_PRIMARY, alpha=0.15)

    # 90% recall threshold
    ax_b.axhline(90, color=COLOR_THRESHOLD, ls="--", lw=0.8, alpha=0.8)
    # 15 dB vertical threshold
    ax_b.axvline(15, color=COLOR_THRESHOLD, ls="--", lw=0.8, alpha=0.8)

    # Reliable region shading (SNR > 15 and Recall > 90%)
    snr_fill = np.linspace(15, 40, 100)
    ax_b.fill_between(snr_fill, 90, 100, color=COLOR_SECONDARY, alpha=0.15)
    ax_b.text(27, 93, "Reliable region", fontsize=7, color=COLOR_SECONDARY)

    ax_b.set_xlabel("SNR (dB)")
    ax_b.set_ylabel("Recall@32 (%)")
    ax_b.set_xlim(0, 40)
    ax_b.set_ylim(20, 105)
    ax_b.text(-0.14, 1.04, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_snr_analysis.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
