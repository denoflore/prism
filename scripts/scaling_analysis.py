#!/usr/bin/env python3
"""
Fig: PRISM Photonic Scaling Projections (three-panel).
  (a) MRR count vs bank size N for four d values.
  (b) Aggregate heater power vs N (SOI comparison only; TFLN EO = 0).
  (c) Chip area vs N.
Output: figures/fig_scaling_analysis.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import (
    OUT_DIR, COLORS, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY,
    COLOR_GRAY, COLOR_THRESHOLD,
)

# ── Hardware parameters (from generate_photonic_figures.py fig6) ──
FOOTPRINT_M2 = (50e-6) ** 2        # 50 um x 50 um per MRR
FOOTPRINT_MM2 = FOOTPRINT_M2 * 1e6
P_HEATER_AVG_W = 2.5e-3            # W per MRR (SOI thermo-optic, for comparison only; TFLN EO = 0)
RETICLE_MM2 = 26 * 33              # 858 mm^2 single-reticle limit
THERMAL_LIMIT_W = 200.0            # practical thermal dissipation limit


def main():
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.0, 2.5))

    N_arr = np.logspace(np.log10(32), np.log10(4096), 80)
    d_vals = [16, 32, 64, 128]
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_GRAY]
    markers = ["o", "s", "^", "D"]

    # Sparse marker indices
    mk_idx = np.linspace(0, len(N_arr) - 1, 6, dtype=int)

    for d, color, mk in zip(d_vals, colors, markers):
        n_mrr = N_arr * d
        heater_W = N_arr * d * P_HEATER_AVG_W
        area_mm2 = N_arr * d * FOOTPRINT_MM2

        # Panel (a): MRR count
        ax_a.loglog(N_arr, n_mrr, color=color, linewidth=1.2,
                    label=f"$d = {d}$")
        ax_a.plot(N_arr[mk_idx], n_mrr[mk_idx], marker=mk, color=color,
                  linewidth=0, markersize=4)

        # Panel (b): Heater power
        ax_b.loglog(N_arr, heater_W, color=color, linewidth=1.2,
                    label=f"$d = {d}$")
        ax_b.plot(N_arr[mk_idx], heater_W[mk_idx], marker=mk, color=color,
                  linewidth=0, markersize=4)

        # Panel (c): Chip area
        ax_c.loglog(N_arr, area_mm2, color=color, linewidth=1.2,
                    label=f"$d = {d}$")
        ax_c.plot(N_arr[mk_idx], area_mm2[mk_idx], marker=mk, color=color,
                  linewidth=0, markersize=4)

    # ── Panel (a) formatting ──
    ax_a.set_xlabel("Bank size $N$")
    ax_a.set_ylabel(r"MRR count ($d \times N$)")
    ax_a.set_xlim(30, 5000)
    ax_a.set_ylim(1e2, 1e6)
    ax_a.legend(loc="upper left", fontsize=6.5)
    ax_a.text(-0.20, 1.04, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold")

    # ── Panel (b) formatting ──
    ax_b.axhline(THERMAL_LIMIT_W, color=COLOR_THRESHOLD, ls="--", lw=0.8,
                 alpha=0.8)
    ax_b.text(40, THERMAL_LIMIT_W * 1.25, "200 W thermal limit",
              fontsize=6.5, color=COLOR_THRESHOLD)
    # Infeasible zone tint
    ax_b.fill_between(N_arr, THERMAL_LIMIT_W, 1e4,
                      color=COLOR_TERTIARY, alpha=0.08)

    ax_b.set_xlabel("Bank size $N$")
    ax_b.set_ylabel("Aggregate heater power (W) — SOI comparison")
    ax_b.set_xlim(30, 5000)
    ax_b.set_ylim(1, 3000)
    ax_b.legend(loc="lower right", fontsize=5.5)
    ax_b.text(-0.20, 1.04, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold")

    # ── Panel (c) formatting ──
    ax_c.axhline(RETICLE_MM2, color=COLOR_THRESHOLD, ls="--", lw=0.8,
                 alpha=0.8)
    ax_c.text(40, RETICLE_MM2 * 1.25,
              f"{RETICLE_MM2} mm$^2$ reticle limit",
              fontsize=6.5, color=COLOR_THRESHOLD)
    # Infeasible zone tint
    ax_c.fill_between(N_arr, RETICLE_MM2, 1e5,
                      color=COLOR_TERTIARY, alpha=0.08)

    ax_c.set_xlabel("Bank size $N$")
    ax_c.set_ylabel(r"Chip area (mm$^2$)")
    ax_c.set_xlim(30, 5000)
    ax_c.set_ylim(1, 1e4)
    ax_c.legend(loc="lower right", fontsize=5.5)
    ax_c.text(-0.20, 1.04, "(c)", transform=ax_c.transAxes,
              fontsize=10, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_scaling_analysis.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
