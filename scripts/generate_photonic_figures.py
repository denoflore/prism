#!/usr/bin/env python3
r"""
Generate photonic system-level figures for the PRISM paper.

Figures produced (saved to ../figures/ relative to this script):
  Fig 1. fig_concept_comparison.pdf  -- Electronic vs PRISM comparison
  Fig 4. fig_power_budget.pdf        -- Optical power budget vs N
  Fig 5. fig_snr_analysis.pdf        -- Noise & SNR analysis (2 panels)
  Fig 6. fig_scaling_analysis.pdf    -- Scaling analysis (3 panels)
  Fig 8. fig_time_multiplex.pdf      -- Time-mux latency vs area trade-off

Usage:
    python generate_photonic_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# ── Tier 2C rcParams ─────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.4,
    'legend.frameon': True,
    'legend.edgecolor': '#CCCCCC',
    'legend.framealpha': 0.9,
})

# ── Output directory (relative to this script) ──────────────────────────────
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ── Figure sizes per Tier 2C ─────────────────────────────────────────────────
COL_W = 3.35   # single-column width (inches)
FULL_W = 7.0   # double-column width (inches)

# ── Tier 2C Color palette ────────────────────────────────────────────────────
C_PRIMARY    = '#1B3A5C'   # Dark navy
C_SECONDARY  = '#4DB8C7'   # Cyan/teal accent
C_TERTIARY   = '#B8952A'   # Muted gold
C_QUATERNARY = '#A0522D'   # Muted red-brown
C_QUINARY    = '#6B7B3A'   # Olive
C_THRESHOLD  = '#888888'   # Threshold / reference gray
C_LIGHTFILL  = '#D4E4EC'   # Light fill
C_ERRORBAND  = '#E3EFF5'   # Error band

# Block-diagram fills (Tier 2B-style, for concept_comparison)
C_OPTICAL_FILL    = '#E3EFF5'   # blue!5 equivalent
C_ELECTRONIC_FILL = '#E8E8E8'   # gray!12 equivalent
C_HYBRID_FILL     = '#D4E4EC'   # hybrid fill

# Stacked area / bar palette (using Tier 2C cycle)
STACK_COLORS = [C_PRIMARY, C_SECONDARY, C_TERTIARY, C_QUATERNARY, C_QUINARY,
                C_THRESHOLD]


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Optical Power Budget
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_power_budget():
    """Required laser power vs number of channels N, with loss breakdown."""
    N_arr = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
    d_vals = [32, 64, 128]
    PD_sens = -20.0  # dBm

    # Loss components
    def splitter_loss(N):
        return 10 * np.log10(N) + 0.2 * np.ceil(np.log2(N))

    wg_loss = 1.0        # dB (5 mm, 2 dB/cm)
    mod_il  = 3.0        # dB
    conn_loss = 1.5      # dB

    def mrr_il(d):
        return 0.1 * d   # dB per ring

    fig, (ax_main, ax_bar) = plt.subplots(
        1, 2, figsize=(FULL_W, 2.8),
        gridspec_kw={"width_ratios": [3, 1.4], "wspace": 0.35}
    )

    # --- Main panel: Total required laser power vs N ---
    markers = ["o", "s", "D"]
    colors_d = [C_PRIMARY, C_SECONDARY, C_TERTIARY]
    for i, d in enumerate(d_vals):
        total_loss = splitter_loss(N_arr) + wg_loss + mrr_il(d) + mod_il + conn_loss
        P_required = PD_sens + total_loss
        ax_main.plot(N_arr, P_required, marker=markers[i], color=colors_d[i],
                     markerfacecolor="white", markeredgewidth=1.0,
                     label=f"$d = {d}$", zorder=5)

    # Banking benefit: show N=1024 with N_bank=256 (4 banks)
    N_bank = 256
    for i, d in enumerate(d_vals):
        loss_banked = splitter_loss(N_bank) + wg_loss + mrr_il(d) + mod_il + conn_loss
        P_banked = PD_sens + loss_banked
        if i == 0:  # only label once
            ax_main.plot(1024, P_banked, marker="*", color=colors_d[i],
                         markersize=9, zorder=6, label=f"Banked ($N_b$=256)")
        else:
            ax_main.plot(1024, P_banked, marker="*", color=colors_d[i],
                         markersize=9, zorder=6)

    # PD sensitivity reference
    ax_main.axhline(PD_sens, color=C_THRESHOLD, ls="--", lw=0.7, alpha=0.6)
    ax_main.text(35, PD_sens + 0.8, "PD sensitivity = $-$20 dBm",
                 fontsize=6.5, color=C_THRESHOLD)

    # Typical laser power
    ax_main.axhline(20, color=C_TERTIARY, ls=":", lw=0.7, alpha=0.5)
    ax_main.text(35, 20.8, "Commercial laser (+20 dBm)", fontsize=6.5,
                 color=C_TERTIARY)

    ax_main.set_xscale("log", base=2)
    ax_main.set_xlabel("Number of channels $N$")
    ax_main.set_ylabel("Required laser power (dBm)")
    ax_main.set_xticks(N_arr)
    ax_main.set_xticklabels([str(n) for n in N_arr])
    ax_main.legend(loc="upper left", framealpha=0.9, edgecolor='#CCCCCC',
                   handlelength=1.5, borderpad=0.4)
    ax_main.set_ylim(-5, 35)
    ax_main.text(-0.14, 1.04, "(a)", transform=ax_main.transAxes,
                 fontsize=10, fontweight="bold")

    # --- Inset panel: stacked bar for N=256, 1024 ---
    bar_Ns = [256, 1024]
    d_bar = 32
    width = 0.35
    x_pos = np.arange(len(bar_Ns))

    components = [
        ("Splitter", [splitter_loss(n) for n in bar_Ns], STACK_COLORS[0]),
        ("Waveguide", [wg_loss] * 2, STACK_COLORS[1]),
        ("MRR IL", [mrr_il(d_bar)] * 2, STACK_COLORS[2]),
        ("Modulator", [mod_il] * 2, STACK_COLORS[3]),
        ("Coupling", [conn_loss] * 2, STACK_COLORS[4]),
    ]

    bottom = np.zeros(len(bar_Ns))
    for name, vals, col in components:
        ax_bar.bar(x_pos, vals, width=0.5, bottom=bottom, color=col,
                   edgecolor="black", linewidth=0.5, label=name)
        bottom += np.array(vals)

    # Total labels on top of each bar
    for j, n in enumerate(bar_Ns):
        ax_bar.text(x_pos[j], bottom[j] + 0.5, f"{bottom[j]:.1f} dB",
                    ha="center", fontsize=6.5, fontweight="bold", va="bottom")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"$N$={n}" for n in bar_Ns])
    ax_bar.set_ylabel("Optical loss (dB)")
    ax_bar.legend(fontsize=5.5, loc="upper center", framealpha=0.9,
                  edgecolor='#CCCCCC', borderpad=0.3,
                  bbox_to_anchor=(0.5, 1.02))
    ax_bar.set_ylim(0, bottom.max() + 8)
    ax_bar.text(-0.2, 1.04, "(b)", transform=ax_bar.transAxes,
                fontsize=10, fontweight="bold")

    path = os.path.join(OUTDIR, "fig_power_budget.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Noise & SNR Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_snr_analysis():
    """(a) SNR vs dimension d; (b) Recall@8 vs SNR."""
    fig, (ax_snr, ax_recall) = plt.subplots(1, 2, figsize=(FULL_W, 2.8),
                                             gridspec_kw={"wspace": 0.38})

    # ── Physical constants ──
    q    = 1.602e-19   # C
    kB   = 1.381e-23   # J/K
    T    = 300.0        # K
    R_L  = 50.0         # ohm
    BW   = 1e9          # 1 GHz
    resp = 1.0          # A/W (PD responsivity)
    tia_density = 10e-9 # A/sqrt(Hz), TIA input-referred noise

    # ── Panel (a): SNR vs d ──
    P_laser_dBm = 13.0  # per bank
    N_bank = 256
    d_arr = np.arange(8, 129)

    # Losses
    splitter_loss_dB = 10 * np.log10(N_bank) + 0.2 * np.ceil(np.log2(N_bank))
    wg_loss_dB = 1.0
    mod_il_dB = 3.0
    conn_loss_dB = 1.5

    # Per-channel received power vs d (MRR through-port IL scales with d)
    mrr_il_dB = 0.1 * d_arr
    total_loss_dB = splitter_loss_dB + wg_loss_dB + mrr_il_dB + mod_il_dB + conn_loss_dB
    P_rx_dBm = P_laser_dBm - total_loss_dB
    P_rx_W = 1e-3 * 10 ** (P_rx_dBm / 10)
    I_photo = resp * P_rx_W

    # Noise currents (RMS)
    i_shot = np.sqrt(2 * q * I_photo * BW)
    i_thermal = np.sqrt(4 * kB * T * BW / R_L)
    i_tia = tia_density * np.sqrt(BW)
    i_total = np.sqrt(i_shot**2 + i_thermal**2 + i_tia**2)

    SNR_dB = 20 * np.log10(I_photo / i_total)

    ax_snr.plot(d_arr, SNR_dB, color=C_PRIMARY, lw=1.5, label="Total SNR")

    # Precision thresholds
    thresholds = [(24, "4-bit"), (30, "5-bit"), (36, "6-bit")]
    th_colors = [C_QUINARY, C_TERTIARY, C_QUATERNARY]
    for (th, lbl), tc in zip(thresholds, th_colors):
        ax_snr.axhline(th, color=tc, ls="--", lw=0.7, alpha=0.7)
        ax_snr.text(d_arr[-1] + 2, th + 0.5, lbl, fontsize=6.5, color=tc,
                    va="bottom")

    # Noise breakdown (secondary lines)
    ax_snr.plot(d_arr, 20 * np.log10(I_photo / i_shot), color=C_LIGHTFILL,
                ls=":", lw=0.8, label="Shot-limited")
    ax_snr.plot(d_arr, 20 * np.log10(I_photo / i_thermal),
                color=C_SECONDARY, ls=":", lw=0.8, label="Thermal-limited")

    ax_snr.set_xlabel("Signature dimension $d$")
    ax_snr.set_ylabel("SNR (dB)")
    ax_snr.set_xlim(8, 128)
    ax_snr.set_ylim(10, 65)
    ax_snr.legend(loc="upper right", framealpha=0.9, edgecolor='#CCCCCC',
                  handlelength=1.5, fontsize=6.5)
    ax_snr.text(-0.14, 1.04, "(a)", transform=ax_snr.transAxes,
                fontsize=10, fontweight="bold")

    # ── Panel (b): Recall@8 vs SNR ──
    rng = np.random.default_rng(42)
    n_blocks = 500
    d_sim = 32
    k = 8
    n_trials = 1000

    # Generate ground-truth block signatures
    sigs = rng.standard_normal((n_blocks, d_sim)).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(sigs, axis=1, keepdims=True)
    sigs = sigs / norms

    query = rng.standard_normal(d_sim).astype(np.float32)
    query = query / np.linalg.norm(query)

    # True similarities and true top-k
    true_sim = sigs @ query
    true_topk = set(np.argsort(true_sim)[-k:])

    snr_sweep = np.arange(5, 51, 1.0)
    recall_mean = []
    recall_std = []

    for snr_dB in snr_sweep:
        snr_lin = 10 ** (snr_dB / 20)
        recalls = []
        for _ in range(n_trials):
            # Add Gaussian noise to similarities
            sig_power = np.std(true_sim)
            noise_std = sig_power / snr_lin
            noisy_sim = true_sim + rng.normal(0, noise_std, size=n_blocks)
            noisy_topk = set(np.argsort(noisy_sim)[-k:])
            recalls.append(len(true_topk & noisy_topk) / k)
        recall_mean.append(np.mean(recalls))
        recall_std.append(np.std(recalls))

    recall_mean = np.array(recall_mean)
    recall_std = np.array(recall_std)

    ax_recall.plot(snr_sweep, recall_mean, color=C_QUATERNARY, lw=1.5)
    ax_recall.fill_between(snr_sweep, recall_mean - recall_std,
                           np.minimum(recall_mean + recall_std, 1.0),
                           color=C_ERRORBAND, alpha=0.3, label=r"$\pm 1\sigma$")

    # 90% recall line
    ax_recall.axhline(0.9, color=C_THRESHOLD, ls="--", lw=0.7, alpha=0.6)
    ax_recall.text(6, 0.91, "Recall = 0.9", fontsize=6.5, color=C_THRESHOLD)

    # Mark precision thresholds
    for (th, lbl), tc in zip(thresholds, th_colors):
        ax_recall.axvline(th, color=tc, ls=":", lw=0.7, alpha=0.5)
        ax_recall.text(th + 0.5, 0.3, lbl, fontsize=6, color=tc, rotation=90,
                       va="bottom")

    ax_recall.set_xlabel("SNR (dB)")
    ax_recall.set_ylabel("Recall@8")
    ax_recall.set_xlim(5, 50)
    ax_recall.set_ylim(0.0, 1.05)
    ax_recall.legend(loc="lower right", framealpha=0.9, edgecolor='#CCCCCC')
    ax_recall.text(-0.14, 1.04, "(b)", transform=ax_recall.transAxes,
                   fontsize=10, fontweight="bold")

    path = os.path.join(OUTDIR, "fig_snr_analysis.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Scaling Analysis (three panels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_scaling_analysis():
    """(a) MRR count vs context; (b) chip area vs N; (c) SOI heater power vs N (comparison only)."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.5),
                              gridspec_kw={"wspace": 0.42})
    ax_mrr, ax_area, ax_power = axes

    B = 256  # block size in tokens
    footprint_m2 = (50e-6) ** 2  # 50 um x 50 um per MRR
    footprint_mm2 = footprint_m2 * 1e6  # -> mm^2
    P_heater_avg = 2.5e-3  # W per MRR (SOI thermo-optic, for comparison only)

    d_vals = [16, 32, 64]
    colors_d = [C_PRIMARY, C_SECONDARY, C_TERTIARY]
    markers = ["o", "s", "D"]

    # ── Panel (a): MRR count vs context length ──
    ctx_tokens = np.array([4096, 8192, 16384, 32768, 65536, 131072,
                           262144, 524288, 1048576, 2097152])
    ctx_labels = {4096: "4K", 8192: "8K", 16384: "16K", 32768: "32K",
                  65536: "64K", 131072: "128K", 262144: "256K",
                  524288: "512K", 1048576: "1M", 2097152: "2M"}

    for i, d in enumerate(d_vals):
        n_blocks = ctx_tokens / B
        n_mrr = n_blocks * d
        ax_mrr.plot(ctx_tokens, n_mrr, marker=markers[i], color=colors_d[i],
                    markerfacecolor="white", markeredgewidth=0.8,
                    label=f"$d = {d}$", markersize=3.5)

    # Feasibility bands
    ax_mrr.axhspan(0, 1e4, color=C_QUINARY, alpha=0.15)
    ax_mrr.axhspan(1e4, 1e5, color=C_TERTIARY, alpha=0.15)
    ax_mrr.axhspan(1e5, 1e7, color=C_QUATERNARY, alpha=0.10)
    ax_mrr.text(5000, 3e3, "Demonstrated", fontsize=5.5, color=C_QUINARY)
    ax_mrr.text(5000, 3e4, "Near-term", fontsize=5.5, color=C_TERTIARY)
    ax_mrr.text(5000, 3e5, "Future", fontsize=5.5, color=C_QUATERNARY)

    ax_mrr.set_xscale("log")
    ax_mrr.set_yscale("log")
    ax_mrr.set_xlabel("Context length (tokens)")
    ax_mrr.set_ylabel("MRR count")
    ax_mrr.set_xlim(3000, 3e6)
    ax_mrr.set_ylim(50, 1e7)
    ax_mrr.legend(loc="upper left", framealpha=0.9, edgecolor='#CCCCCC',
                  fontsize=6, handlelength=1.2, borderpad=0.3)
    ax_mrr.text(-0.18, 1.04, "(a)", transform=ax_mrr.transAxes,
                fontsize=10, fontweight="bold")

    # ── Panel (b): Chip area vs N ──
    N_arr = np.logspace(np.log10(32), np.log10(4096), 50)
    for i, d in enumerate([32, 64]):
        area_mm2 = N_arr * d * footprint_mm2
        ax_area.plot(N_arr, area_mm2, color=colors_d[i + 1],
                     label=f"$d = {d}$", lw=1.3)

    # Reticle limit
    reticle_mm2 = 26 * 33  # 858 mm^2
    ax_area.axhline(reticle_mm2, color=C_THRESHOLD, ls="--", lw=0.7, alpha=0.7)
    ax_area.text(40, reticle_mm2 * 1.15, f"Reticle limit ({reticle_mm2} mm$^2$)",
                 fontsize=6, color=C_THRESHOLD)

    ax_area.set_xscale("log")
    ax_area.set_yscale("log")
    ax_area.set_xlabel("Number of channels $N$")
    ax_area.set_ylabel("Chip area (mm$^2$)")
    ax_area.set_xlim(30, 5000)
    ax_area.set_ylim(0.05, 2000)
    ax_area.legend(loc="upper left", framealpha=0.9, edgecolor='#CCCCCC',
                   fontsize=6, handlelength=1.2)
    ax_area.text(-0.18, 1.04, "(b)", transform=ax_area.transAxes,
                 fontsize=10, fontweight="bold")

    # ── Panel (c): Total heater power vs N ──
    for i, d in enumerate([32, 64]):
        P_total_W = N_arr * d * P_heater_avg
        ax_power.plot(N_arr, P_total_W, color=colors_d[i + 1],
                      label=f"$d = {d}$", lw=1.3)

    # Thermal limits
    limits = [(10, "Passive (10 W)"), (50, "Active (50 W)"), (200, "Liquid (200 W)")]
    lim_colors = [C_QUINARY, C_TERTIARY, C_QUATERNARY]
    for (pw, lbl), lc in zip(limits, lim_colors):
        ax_power.axhline(pw, color=lc, ls="--", lw=0.7, alpha=0.6)
        ax_power.text(40, pw * 1.12, lbl, fontsize=5.5, color=lc)

    ax_power.set_xscale("log")
    ax_power.set_yscale("log")
    ax_power.set_xlabel("Number of channels $N$")
    ax_power.set_ylabel("Total heater power (W) — SOI comparison")
    ax_power.set_xlim(30, 5000)
    ax_power.set_ylim(1, 2000)
    ax_power.legend(loc="upper left", framealpha=0.9, edgecolor='#CCCCCC',
                    fontsize=6, handlelength=1.2)
    ax_power.text(-0.18, 1.04, "(c)", transform=ax_power.transAxes,
                  fontsize=10, fontweight="bold")

    path = os.path.join(OUTDIR, "fig_scaling_analysis.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Time-multiplexed Architecture Trade-off
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_time_multiplex():
    """Latency vs chip area for different time-mux factors."""
    N_total = 1024
    d = 32
    footprint_mm2 = (50e-6) ** 2 * 1e6  # per MRR -> mm^2

    # Base latency: single pass ~10 ns (modulation + propagation + detection)
    t_base_ns = 10.0
    # Reconfiguration overhead per mux step: ~2 ns (thermal settling negligible
    # if using pre-heated banks)
    t_reconfig_ns = 0.0  # assume banked / fast switching

    mux_factors = [1, 2, 4, 8, 16]
    areas = []
    latencies = []
    for m in mux_factors:
        N_active = N_total / m
        area = N_active * d * footprint_mm2
        latency = t_base_ns * m
        areas.append(area)
        latencies.append(latency)

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    # Plot the trade-off curve
    ax.plot(areas, latencies, "o-", color=C_PRIMARY, markerfacecolor="white",
            markeredgewidth=1.2, markersize=6, zorder=5, lw=1.5)

    # Label each point with manual offsets to avoid overlap
    label_offsets = {
        1:  (-55, -20),  # fully parallel: left-below
        2:  (15, -12),
        4:  (15, 8),
        8:  (-15, 20),   # above
        16: (-70, -20),  # left-below
    }
    for m, a, l in zip(mux_factors, areas, latencies):
        ox, oy = label_offsets.get(m, (12, 8))
        ha = "left" if ox > 0 else "left"
        ax.annotate(f"{m}$\\times$ mux\n({l:.0f} ns, {a:.1f} mm$^2$)",
                    xy=(a, l), fontsize=6, ha="left", va="bottom",
                    xytext=(ox, oy),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=C_THRESHOLD, lw=0.5),
                    color=C_PRIMARY)

    # GPU baseline
    gpu_latency = 5000  # ns
    ax.axhline(gpu_latency, color=C_QUATERNARY, ls="--", lw=1.0, alpha=0.7)
    ax.text(areas[0] * 0.6, gpu_latency * 1.12,
            f"GPU sequential scan (~{gpu_latency/1000:.0f} $\\mu$s)",
            fontsize=7, color=C_QUATERNARY, fontweight="bold")

    # Speedup annotation
    ax.annotate("", xy=(areas[-1], latencies[-1]),
                xytext=(areas[-1], gpu_latency),
                arrowprops=dict(arrowstyle="<->", color=C_QUINARY, lw=1.0))
    ax.text(areas[-1] * 1.3, np.sqrt(latencies[-1] * gpu_latency),
            f"{gpu_latency/latencies[-1]:.0f}$\\times$\nfaster",
            fontsize=7, color=C_QUINARY, fontweight="bold", ha="left", va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chip area (mm$^2$)")
    ax.set_ylabel("Selection latency (ns)")
    ax.set_xlim(0.3, 200)
    ax.set_ylim(5, 15000)

    # Custom y-ticks
    ax.set_yticks([10, 100, 1000, 5000])
    ax.set_yticklabels(["10", "100", "1000", "5000"])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    path = os.path.join(OUTDIR, "fig_time_multiplex.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Concept Comparison (Electronic vs PRISM)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_concept_comparison():
    """Side-by-side comparison: Electronic GPU scan vs PRISM photonic."""
    fig, (ax_elec, ax_phot) = plt.subplots(1, 2, figsize=(FULL_W, 3.0),
                                            gridspec_kw={"wspace": 0.12})

    for ax in (ax_elec, ax_phot):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── Helper functions ──
    def draw_box(ax, xy, w, h, text, facecolor, edgecolor="black",
                 fontsize=7, fontweight="normal", alpha=1.0, text_color="black"):
        box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.15",
                             facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=0.8, alpha=alpha, zorder=3)
        ax.add_patch(box)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color,
                zorder=4)

    def draw_arrow(ax, xy_from, xy_to, color="black", style="-|>", lw=1.0):
        arrow = FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                                mutation_scale=10, color=color, lw=lw, zorder=2)
        ax.add_patch(arrow)

    # ═══════════════════════════════════════════════════════════════════════
    # LEFT PANEL: Electronic (GPU)
    # ═══════════════════════════════════════════════════════════════════════
    ax_elec.set_title("Electronic (GPU)", fontsize=9, fontweight="bold",
                      pad=8, color="black")

    # Query
    draw_box(ax_elec, (3.5, 8.5), 3.0, 1.0, "Query $\\mathbf{q}$",
             C_OPTICAL_FILL, edgecolor="black", fontsize=8, fontweight="bold")

    # HBM Memory block
    draw_box(ax_elec, (1.0, 4.5), 8.0, 3.2, "", C_ELECTRONIC_FILL, edgecolor="black")
    ax_elec.text(5.0, 7.35, "HBM Memory", ha="center", fontsize=7.5,
                 fontweight="bold", color="black")

    # Individual signature blocks (sequential scan)
    n_blocks_show = 6
    bw = 1.05
    bh = 0.55
    x_start = 1.4
    for i in range(n_blocks_show):
        y_pos = 6.5 - i * 0.65
        fc = C_HYBRID_FILL if i < 3 else C_ERRORBAND
        draw_box(ax_elec, (x_start, y_pos), bw, bh,
                 f"$\\mathbf{{s}}_{{{i+1}}}$", fc, edgecolor="black",
                 fontsize=6)
        # Sequential arrow
        if i < n_blocks_show - 1:
            ax_elec.annotate("", xy=(x_start + bw / 2, y_pos - 0.08),
                             xytext=(x_start + bw / 2, y_pos + bh + 0.02),
                             arrowprops=dict(arrowstyle="-|>", color="black",
                                             lw=0.6, mutation_scale=7))

    # "..." for more
    ax_elec.text(x_start + bw / 2, 4.75, "...", fontsize=10, ha="center",
                 color="black", fontweight="bold")

    # Dot-product blocks
    for i in range(3):
        x_dp = 4.0 + i * 1.8
        draw_box(ax_elec, (x_dp, 5.2), 1.5, 1.8, "",
                 C_ERRORBAND, edgecolor="black", fontsize=6, alpha=0.8)
        ax_elec.text(x_dp + 0.75, 6.6, f"$\\mathbf{{q}} \\cdot \\mathbf{{s}}_{{{i+1}}}$",
                     ha="center", fontsize=6.5, color="black")
        # Score
        ax_elec.text(x_dp + 0.75, 5.5, f"score$_{{{i+1}}}$",
                     ha="center", fontsize=5.5, color="black")

    ax_elec.text(8.0, 6.0, "...", fontsize=10, ha="center", color="black",
                 fontweight="bold")

    # Arrow from query to memory
    draw_arrow(ax_elec, (5.0, 8.5), (5.0, 7.8), color="black")

    # Sequential scan annotation
    ax_elec.annotate("$O(N)$ sequential\naccess",
                     xy=(1.0, 5.5), fontsize=7, color="black",
                     fontweight="bold", ha="center",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=C_ERRORBAND,
                               edgecolor="black", alpha=0.8))

    # Top-k selection
    draw_box(ax_elec, (3.0, 2.5), 4.0, 1.2, "Top-$k$ Selection\n(sort + select)",
             C_HYBRID_FILL, edgecolor="black", fontsize=7, fontweight="bold")
    draw_arrow(ax_elec, (5.0, 4.5), (5.0, 3.7), color="black")

    # Result
    draw_box(ax_elec, (3.5, 0.8), 3.0, 1.0, "Selected blocks",
             C_HYBRID_FILL, edgecolor="black", fontsize=7.5, fontweight="bold")
    draw_arrow(ax_elec, (5.0, 2.5), (5.0, 1.8), color="black")

    # Timing
    ax_elec.text(5.0, 0.15, "$\\sim$5 $\\mu$s", fontsize=9, ha="center",
                 color="black", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="black", alpha=0.9))

    # ═══════════════════════════════════════════════════════════════════════
    # RIGHT PANEL: PRISM (Photonic)
    # ═══════════════════════════════════════════════════════════════════════
    ax_phot.set_title("PRISM (Photonic)", fontsize=9, fontweight="bold",
                      pad=8, color="black")

    # Query
    draw_box(ax_phot, (3.5, 8.5), 3.0, 1.0,
             "Query sketch $\\hat{\\mathbf{q}}$",
             C_OPTICAL_FILL, edgecolor="black", fontsize=8, fontweight="bold")

    # WDM Encoding
    draw_box(ax_phot, (3.0, 7.0), 4.0, 1.0,
             "WDM Encoding\n($d$ wavelengths)",
             C_OPTICAL_FILL, edgecolor="black", fontsize=7, fontweight="bold")
    draw_arrow(ax_phot, (5.0, 8.5), (5.0, 8.0), color="black")

    # Optical broadcast fan-out
    ax_phot.annotate("$O(1)$ parallel\nbroadcast",
                     xy=(9.2, 6.3), fontsize=7, color="black",
                     fontweight="bold", ha="center",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=C_ERRORBAND,
                               edgecolor="black", alpha=0.8))

    # Splitter
    draw_box(ax_phot, (4.0, 5.8), 2.0, 0.7, "1:$N$ Splitter",
             C_OPTICAL_FILL, edgecolor="black", fontsize=6.5, fontweight="bold")
    draw_arrow(ax_phot, (5.0, 7.0), (5.0, 6.5), color="black")

    # Fan-out arrows to MRR banks
    n_banks = 5
    bank_y = 4.2
    bank_h = 0.8
    bank_w = 1.2
    x_positions = np.linspace(0.8, 8.0, n_banks)

    for j, xp in enumerate(x_positions):
        # Fan-out arrow from splitter
        draw_arrow(ax_phot, (5.0, 5.8), (xp + bank_w / 2, bank_y + bank_h),
                   color="black", lw=0.7)
        # MRR bank
        fc = C_OPTICAL_FILL if j < n_banks - 1 else C_ERRORBAND
        lbl = f"MRR$_{{{j+1}}}$" if j < n_banks - 1 else f"MRR$_N$"
        draw_box(ax_phot, (xp, bank_y), bank_w, bank_h, lbl,
                 fc, edgecolor="black", fontsize=6)
        # PD below each bank
        draw_box(ax_phot, (xp + 0.15, 3.2), bank_w - 0.3, 0.55,
                 "PD", C_ELECTRONIC_FILL, edgecolor="black", fontsize=5.5)
        draw_arrow(ax_phot, (xp + bank_w / 2, bank_y),
                   (xp + bank_w / 2, 3.75), color="black", lw=0.6)

    # "..." between banks
    ax_phot.text(x_positions[-2] + bank_w + 0.15, bank_y + bank_h / 2,
                 "...", fontsize=10, ha="center", color="black",
                 fontweight="bold")

    # Comparator / Top-k
    draw_box(ax_phot, (2.5, 1.8), 5.0, 1.0,
             "Analog Comparator $\\rightarrow$ Top-$k$",
             C_HYBRID_FILL, edgecolor="black", fontsize=7, fontweight="bold")

    # Arrows from PDs to comparator
    for xp in [x_positions[0], x_positions[2], x_positions[-1]]:
        draw_arrow(ax_phot, (xp + bank_w / 2, 3.2),
                   (5.0, 2.8), color="black", lw=0.5)

    # Result
    draw_box(ax_phot, (3.5, 0.5), 3.0, 0.9, "Selected blocks",
             C_HYBRID_FILL, edgecolor="black", fontsize=7.5, fontweight="bold")
    draw_arrow(ax_phot, (5.0, 1.8), (5.0, 1.4), color="black")

    # Timing
    ax_phot.text(5.0, 0.0, "$\\sim$9 ns", fontsize=9, ha="center",
                 color="black", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="black", alpha=0.9))

    # Speedup annotation between panels
    fig.text(0.50, 0.06, "$\\mathbf{>500\\times}$ speedup",
             fontsize=10, ha="center", fontweight="bold", color="black",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=C_ERRORBAND,
                       edgecolor="black", alpha=0.9))

    path = os.path.join(OUTDIR, "fig_concept_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating PRISM photonic figures...")
    print()

    print("[Fig 1] Concept comparison (Electronic vs PRISM)...")
    fig1_concept_comparison()

    print("[Fig 4] Optical power budget...")
    fig4_power_budget()

    print("[Fig 5] Noise & SNR analysis...")
    fig5_snr_analysis()

    print("[Fig 6] Scaling analysis...")
    fig6_scaling_analysis()

    print("[Fig 8] Time-multiplexed architecture...")
    fig8_time_multiplex()

    print()
    print("All figures saved to:", os.path.abspath(OUTDIR))
    print("Done.")
