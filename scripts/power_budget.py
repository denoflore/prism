#!/usr/bin/env python3
"""
Fig: Optical Power Budget Analysis (two-panel).
  (a) Received power vs bank size N for three laser powers.
  (b) Electrical SNR vs signature dimension d for N=256 and N=1024.
Output: figures/fig_power_budget.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import (
    OUT_DIR, COLORS, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY,
    COLOR_THRESHOLD,
)

# ── Physical / link-budget parameters (from generate_photonic_figures.py) ──
PD_SENS_DBM = -20.0  # minimum detectable power (dBm)

# Loss components
WG_LOSS_DB = 1.0       # waveguide loss (5 mm at 2 dB/cm)
MOD_IL_DB = 3.0        # modulator insertion loss
CONN_LOSS_DB = 1.5     # coupling / connector loss

def splitter_loss_dB(N):
    """1:N splitter loss including excess loss."""
    return 10.0 * np.log10(N) + 0.2 * np.ceil(np.log2(N))

def mrr_il_dB(d):
    """MRR through-port insertion loss scaling with d."""
    return 0.1 * d

def total_loss_dB(N, d):
    return splitter_loss_dB(N) + WG_LOSS_DB + mrr_il_dB(d) + MOD_IL_DB + CONN_LOSS_DB

# SNR computation constants (from fig5_snr_analysis)
q = 1.602e-19    # C
kB = 1.381e-23   # J/K
T = 300.0        # K
R_L = 50.0       # ohm
BW = 1e9         # 1 GHz bandwidth
RESPONSIVITY = 1.0       # A/W
TIA_NOISE = 10e-9        # A/sqrt(Hz)
P_LASER_SNR_DBM = 13.0   # laser power used for SNR panel


def main():
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.4, 5.0))

    # ── Panel (a): Received power vs bank size N ──
    N_arr = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])
    d_fixed = 64  # fixed signature dim for this panel
    laser_powers_mW = [100, 50, 10]
    laser_labels = [r"$P_{\mathrm{laser}}$ = 100 mW",
                    r"$P_{\mathrm{laser}}$ = 50 mW",
                    r"$P_{\mathrm{laser}}$ = 10 mW"]
    line_colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY]
    markers = ["o", "s", "^"]

    for P_mW, label, color, mk in zip(laser_powers_mW, laser_labels,
                                       line_colors, markers):
        P_laser_dBm = 10.0 * np.log10(P_mW)
        loss = total_loss_dB(N_arr, d_fixed)
        P_rx_dBm = P_laser_dBm - loss
        ax_a.plot(N_arr, P_rx_dBm, marker=mk, color=color, label=label,
                  markersize=4, linewidth=1.2)

    # Minimum detectable power threshold
    ax_a.axhline(PD_SENS_DBM, color=COLOR_THRESHOLD, ls="--", lw=0.8,
                 alpha=0.8)
    ax_a.text(20, PD_SENS_DBM - 2.5, "Min. detectable power",
              fontsize=7, color=COLOR_THRESHOLD)

    ax_a.set_xscale("log", base=2)
    ax_a.set_xlabel("Bank Size $N$")
    ax_a.set_ylabel("Received Power per Detector (dBm)")
    ax_a.set_xticks(N_arr)
    ax_a.set_xticklabels([str(n) for n in N_arr])
    ax_a.set_ylim(-40, 10)
    ax_a.legend(loc="upper right", fontsize=7)
    ax_a.text(-0.14, 1.04, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold")

    # ── Panel (b): Electrical SNR vs signature dimension d ──
    d_arr = np.arange(4, 129)
    N_vals = [256, 1024]
    snr_colors = [COLOR_PRIMARY, COLOR_SECONDARY]
    snr_markers = ["o", "^"]

    for N_bank, color, mk in zip(N_vals, snr_colors, snr_markers):
        loss = total_loss_dB(N_bank, d_arr)
        P_rx_dBm = P_LASER_SNR_DBM - loss
        P_rx_W = 1e-3 * 10.0 ** (P_rx_dBm / 10.0)
        I_photo = RESPONSIVITY * P_rx_W

        i_shot = np.sqrt(2 * q * I_photo * BW)
        i_thermal = np.sqrt(4 * kB * T * BW / R_L)
        i_tia = TIA_NOISE * np.sqrt(BW)
        i_total = np.sqrt(i_shot**2 + i_thermal**2 + i_tia**2)

        SNR_dB = 20.0 * np.log10(I_photo / i_total)

        # Plot with sparse markers to keep clean
        step = 8
        ax_b.plot(d_arr, SNR_dB, color=color, linewidth=1.2,
                  label=f"$N = {N_bank}$")
        ax_b.plot(d_arr[::step], SNR_dB[::step], marker=mk, color=color,
                  linewidth=0, markersize=4)

    # Shaded reliable region (SNR > 20 dB)
    ax_b.axhline(20, color=COLOR_THRESHOLD, ls="--", lw=0.8, alpha=0.8)
    ax_b.fill_between(d_arr, 20, 45, color=COLOR_SECONDARY, alpha=0.15)
    ax_b.text(70, 37, "Reliable top-$k$ region", fontsize=7,
              color=COLOR_SECONDARY)

    ax_b.set_xlabel("Signature Dimension $d$")
    ax_b.set_ylabel("Electrical SNR (dB)")
    ax_b.set_xlim(4, 128)
    ax_b.set_ylim(0, 45)
    ax_b.legend(loc="upper left", fontsize=7)
    ax_b.text(-0.14, 1.04, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_power_budget.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
