#!/usr/bin/env python3
"""
Fig: Energy Advantage Scaling with Context Length.
  Main plot (log-log): energy ratio (PRISM / baseline) vs context length.
  Inset: absolute energy comparison (PRISM fixed vs electronic O(n)).
Output: figures/fig_scaling_projection.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from plot_config import (
    OUT_DIR, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY,
    COLOR_THRESHOLD,
)

# ── Parameters (from compute_crossover.py) ──
B = 128           # block size (tokens)
k = 32            # top-k selected blocks
d = 64            # signature dimension (nominal)
n_ref = 128_000   # reference context length

# PRISM hardware
t_prism_ns = 9.0
E_prism_dynamic_pJ = 796
P_per_MRR_uW = 0.0   # TFLN: Pockels EO is capacitive, ~0 static power

# GPU full-scan baseline at reference
E_gpu_full_uJ_ref = 50.0  # uJ at n_ref


# ── Energy models ──
def E_gpu_full(n):
    """GPU full scan energy (uJ), linear in n."""
    return E_gpu_full_uJ_ref * (n / n_ref)

def E_prism_total(n, d_val=d):
    """PRISM total energy (uJ): selection + fetch."""
    N = n / B
    n_mrr = N * d_val
    P_heater_W = n_mrr * P_per_MRR_uW * 1e-6
    E_heater_J = P_heater_W * t_prism_ns * 1e-9
    E_dynamic_J = E_prism_dynamic_pJ * 1e-12
    E_sel = (E_heater_J + E_dynamic_J) * 1e6  # uJ
    E_fetch = E_gpu_full_uJ_ref * (k * B / n_ref)
    return E_sel + E_fetch


def main():
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.4, 5.0))

    n_arr = np.logspace(np.log10(1024), np.log10(1_000_000), 400)

    # ── Panel (a): Energy ratio ──
    e_prism = np.array([E_prism_total(n) for n in n_arr])
    e_gpu = np.array([E_gpu_full(n) for n in n_arr])
    ratio = e_prism / e_gpu

    ax_a.loglog(n_arr, ratio, color=COLOR_PRIMARY, linewidth=1.2, marker="o",
                markevery=50, markersize=4)

    # Break-even line
    ax_a.axhline(1.0, color=COLOR_THRESHOLD, ls="--", lw=0.8, alpha=0.8)
    ax_a.text(1.3e3, 1.15, "Break-even (1$\\times$)", fontsize=7,
              color=COLOR_THRESHOLD)

    # 10x advantage line
    ax_a.axhline(0.1, color="#AAAAAA", ls="--", lw=0.6, alpha=0.6)
    ax_a.text(1.3e3, 0.115, "10$\\times$ advantage", fontsize=7,
              color="#AAAAAA")

    # Annotation at 1M tokens
    n_1m = 1_000_000
    e_ratio_1m = E_prism_total(n_1m) / E_gpu_full(n_1m)
    traffic_reduction = 1.0 / e_ratio_1m
    ax_a.annotate(
        f"{traffic_reduction:.0f}$\\times$ traffic\nreduction",
        xy=(n_1m, e_ratio_1m),
        xytext=(n_1m * 0.08, e_ratio_1m * 25),
        fontsize=7, fontweight="bold", color=COLOR_TERTIARY,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
        ha="center",
    )

    ax_a.set_xlabel("Context length $n$")
    ax_a.set_ylabel("Energy ratio (PRISM / baseline)")
    ax_a.set_xlim(1e3, 1.2e6)
    ax_a.set_ylim(3e-3, 5)
    ax_a.set_xticks([1e3, 1e4, 1e5, 1e6])
    ax_a.set_xticklabels(["1K", "10K", "100K", "1M"])
    ax_a.text(-0.14, 1.04, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold")

    # ── Panel (b): Absolute energy comparison ──
    e_prism_abs = np.array([E_prism_total(n) for n in n_arr])
    e_gpu_abs = np.array([E_gpu_full(n) for n in n_arr])

    ax_b.loglog(n_arr, e_prism_abs * 1e6, color=COLOR_PRIMARY,
                linewidth=1.2, label="PRISM")
    ax_b.loglog(n_arr, e_gpu_abs * 1e6, color=COLOR_TERTIARY,
                linewidth=1.2, label=r"Electronic ($O(n)$)")

    # Find and mark crossing point
    cross_idx = np.where(e_prism_abs < e_gpu_abs)[0]
    if len(cross_idx) > 0:
        ci = cross_idx[0]
        ax_b.plot(n_arr[ci], e_gpu_abs[ci] * 1e6, "o", color="black",
                  markersize=5, zorder=5)
        ax_b.fill_between(
            n_arr[ci:], e_prism_abs[ci:] * 1e6, e_gpu_abs[ci:] * 1e6,
            color=COLOR_SECONDARY, alpha=0.15, label="PRISM favorable"
        )

    ax_b.set_xlabel("Context length $n$")
    ax_b.set_ylabel("Energy per query (pJ)")
    ax_b.set_xlim(1e3, 1.2e6)
    ax_b.set_xticks([1e3, 1e4, 1e5, 1e6])
    ax_b.set_xticklabels(["1K", "10K", "100K", "1M"])
    ax_b.legend(fontsize=7, loc="upper left")
    ax_b.text(-0.14, 1.04, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_scaling_projection.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
