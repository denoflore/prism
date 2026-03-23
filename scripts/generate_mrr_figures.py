#!/usr/bin/env python3
r"""
Generate MRR simulator visualizations for the PRISM paper.

Figures produced (saved to ../figures/):
  1. fig_mrr_lorentzian.pdf      -- MRR transfer function + weight encoding
  2. fig_digital_vs_photonic.pdf -- Digital vs photonic computation comparison
  3. fig_weight_fidelity.pdf     -- Weight encoding fidelity under impairments
  4. fig_combined_sensitivity.pdf -- 2D recall heatmap (bits x drift)

Uses the same simplified MRR model as sim_hw_impairments.py for recall
calculations (normalize to [0,1], quantize, add noise, denormalize, matmul).
MRR Lorentzian physics shown in figure 1 for device illustration.

Usage:
    python scripts/generate_mrr_figures.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

# Inline minimal MRR model (avoids dependency on external prism package)
class MRRPhysicalParams:
    def __init__(self, q_loaded=10_000, extinction_ratio_db=20.0,
                 resonance_wavelength_nm=1550.0, n_eff=2.4, radius_um=5.0):
        self.q_loaded = q_loaded
        self.extinction_ratio_db = extinction_ratio_db
        self.resonance_wavelength_nm = resonance_wavelength_nm
        self.n_eff = n_eff
        self.radius_um = radius_um

class MRRArraySimulator:
    def __init__(self, n_channels=32, n_blocks=4, params=None):
        if params is None:
            params = MRRPhysicalParams()
        self.params = params
        self.n_channels = n_channels
        self.n_blocks = n_blocks
        lam0 = params.resonance_wavelength_nm
        self.fwhm_nm = lam0 / params.q_loaded
        self.er_linear = 10 ** (params.extinction_ratio_db / 10)
        self.max_detuning_nm = self.fwhm_nm * 2.0

    def lorentzian_transmission(self, detuning_nm):
        gamma = self.fwhm_nm / 2.0
        T_min = 1.0 / self.er_linear
        return 1.0 - (1.0 - T_min) / (1.0 + (detuning_nm / gamma) ** 2)

# -- Tier 2C: Data Plot rcParams ---------------------------------------------
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

# -- Tier 2C Color Palette ---------------------------------------------------
CLR_PRIMARY   = '#1B3A5C'   # dark navy
CLR_SECONDARY = '#4DB8C7'   # teal
CLR_TERTIARY  = '#B8952A'   # muted gold
CLR_QUATERNARY = '#A0522D'  # red-brown
CLR_QUINARY   = '#6B7B3A'   # olive
CLR_GRAY      = '#999999'   # silver/gray (4th in color cycle)
CLR_THRESHOLD = '#888888'   # threshold/reference lines
CLR_LIGHT_FILL = '#D4E4EC'  # light fill
CLR_ERROR_BAND = '#E3EFF5'  # error band

# -- Output directory (relative to script location) --------------------------
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- Figure sizes (Tier 2C) --------------------------------------------------
COL_W = 3.35   # single column (inches)
FULL_W = 7.0   # full/double-column width

# -- Simplified MRR impairment model (same as sim_hw_impairments.py) --------
D = 64
N_BLOCKS = 1024
K = 32


def normalize_weights(S_b):
    """Map signatures to [0,1] range for MRR weights."""
    w_min, w_max = S_b.min(), S_b.max()
    return (S_b - w_min) / (w_max - w_min + 1e-30), w_min, w_max


def uniform_quantize(w, bits):
    """Uniform quantization of weights in [0,1]."""
    levels = 2 ** bits
    w_c = np.clip(w, 0, 1)
    return np.round(w_c * (levels - 1)) / (levels - 1)


def add_thermal_drift(w, sigma):
    """Add Gaussian noise to weights (thermal drift)."""
    return np.clip(w + np.random.randn(*w.shape) * sigma, 0, 1)


def add_detector_noise(scores, sigma):
    """Add Gaussian noise to output scores."""
    return scores + np.random.randn(*scores.shape) * sigma * np.std(scores)


def mrr_impaired_scores(S_b, s_q, bits=5, drift_sigma=0.01, det_sigma=0.0):
    """Full simplified MRR pipeline: normalize -> quantize -> drift -> IP -> noise."""
    W, w_min, w_max = normalize_weights(S_b)
    W_q = uniform_quantize(W, bits)
    W_d = add_thermal_drift(W_q, drift_sigma) if drift_sigma > 0 else W_q
    S_recon = W_d * (w_max - w_min) + w_min
    scores = S_recon @ s_q
    if det_sigma > 0:
        scores = add_detector_noise(scores, det_sigma)
    return scores, W, W_q, W_d, w_min, w_max


# ===========================================================================
# Figure 1: MRR Lorentzian Transfer Function
# ===========================================================================
def fig_mrr_lorentzian():
    """Two-panel figure:
    (a) Lorentzian T vs detuning with weight annotations
    (b) Weight-to-transmission mapping for different bit precisions
    """
    print("[1/4] MRR Lorentzian transfer function...")

    params = MRRPhysicalParams(q_loaded=10_000, extinction_ratio_db=20.0)
    sim = MRRArraySimulator(n_channels=32, n_blocks=4, params=params)
    fwhm = sim.fwhm_nm
    er_depth = 1.0 - 1.0 / sim.er_linear

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 5.0))

    # -- Panel (a): Lorentzian spectrum with weight points --
    det = np.linspace(-0.5, 0.5, 1000)
    T = sim.lorentzian_transmission(det)
    ax1.fill_between(det / fwhm, T, alpha=0.15, color=CLR_LIGHT_FILL)
    ax1.plot(det / fwhm, T, color=CLR_PRIMARY, lw=1.4)

    # Mark specific weight values (using Tier 1 wavelength palette)
    # Alternating left/right placement to avoid label overlap in narrow column
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [CLR_QUATERNARY, CLR_QUINARY, CLR_SECONDARY, CLR_TERTIARY, CLR_PRIMARY]
    for w, c in zip(weights, colors):
        d = w * sim.max_detuning_nm
        T_val = sim.lorentzian_transmission(np.array([d]))[0]
        ax1.plot(d / fwhm, T_val, "o", color=c, ms=6, zorder=5)

    # Place labels at fixed y positions to avoid overlap, alternating sides
    # Collect (weight, detuning/fwhm, T_val) for annotation
    annot_data = []
    for w in weights:
        d = w * sim.max_detuning_nm
        T_val = sim.lorentzian_transmission(np.array([d]))[0]
        annot_data.append((w, d / fwhm, T_val))

    # Fixed label positions: (text_x, text_y) — manually tuned for 3.4" wide column
    label_positions = [
        (0.6, -0.08),    # w=0.00: right, below
        (1.8, 0.22),     # w=0.25: right, lowered to make room
        (1.8, 0.52),     # w=0.50: right, between 0.25 and 0.75
        (1.8, 0.78),     # w=0.75: right, mid
        (1.8, 1.02),     # w=1.00: right, top
    ]
    for (w, dx, T_val), (tx, ty) in zip(annot_data, label_positions):
        ax1.annotate(f"w={w:.2f}\nT={T_val:.2f}",
                     xy=(dx, T_val),
                     xytext=(tx, ty),
                     fontsize=6.5, color='black', ha='center',
                     arrowprops=dict(arrowstyle="-", color='gray', lw=0.4))

    # FWHM annotation
    T_on = sim.lorentzian_transmission(np.array([0.0]))[0]
    T_half = (1.0 + T_on) / 2
    ax1.annotate("", xy=(-0.5, T_half), xytext=(0.5, T_half),
                 arrowprops=dict(arrowstyle="<->", color=CLR_THRESHOLD, lw=0.8))
    ax1.text(0, T_half + 0.04, "FWHM", ha="center", fontsize=7, color='black')

    # ER annotation
    ax1.annotate("", xy=(-0.02, 1.0), xytext=(-0.02, T_on),
                 arrowprops=dict(arrowstyle="<->", color=CLR_THRESHOLD, lw=0.8))
    ax1.text(-0.55, (1.0 + T_on) / 2 + 0.35, f"ER\n{params.extinction_ratio_db:.0f} dB",
             fontsize=7, color='black', ha="center", va="center")

    ax1.axhline(y=1.0, ls=":", color=CLR_THRESHOLD, lw=0.4)
    ax1.axvline(x=0, ls=":", color=CLR_THRESHOLD, lw=0.4)
    ax1.set_xlabel(r"Detuning $\Delta\lambda$ / FWHM")
    ax1.set_ylabel("Transmission $T$")
    ax1.set_xlim(-3.0, 3.0)
    ax1.set_ylim(-0.05, 1.15)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes,
             fontsize=10, fontweight='bold', va='top')

    # -- Panel (b): Weight -> Transmission for different bits --
    w_cont = np.linspace(0, 1, 500)
    d_cont = w_cont * sim.max_detuning_nm
    T_cont = sim.lorentzian_transmission(d_cont)
    ax2.plot(w_cont, T_cont, '--', color=CLR_THRESHOLD, lw=1.0, label="Ideal (continuous)")

    bit_styles = [
        (4, "--", CLR_PRIMARY),
        (5, "-.", CLR_SECONDARY),
        (6, ":",  CLR_TERTIARY),
        (8, "-",  CLR_GRAY),
    ]
    for bits, ls, c in bit_styles:
        n_lev = 2 ** bits
        levels = np.linspace(0, 1, n_lev)
        d_q = levels * sim.max_detuning_nm
        T_q = sim.lorentzian_transmission(d_q)
        ax2.plot(levels, T_q, ls, color=c, lw=1.2,
                 label=f"{bits}-bit ({n_lev} levels)")
        if n_lev <= 64:
            ax2.plot(levels, T_q, ".", color=c, ms=3)

    ax2.set_xlabel("Programmed weight $w$ [0, 1]")
    ax2.set_ylabel("Transmission $T(w)$")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes,
             fontsize=10, fontweight='bold', va='top')

    fig.tight_layout(w_pad=2.0)
    path = os.path.join(OUTDIR, "fig_mrr_lorentzian.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 2: Digital vs Photonic Inner Product Comparison
# ===========================================================================
def fig_digital_vs_photonic():
    """Three-panel figure:
    (a) Score scatter (digital vs MRR) for 3 configs
    (b) Rank agreement with top-k highlighted (nominal config)
    (c) Score error histograms
    """
    print("[2/4] Digital vs photonic comparison...")

    np.random.seed(42)
    N = N_BLOCKS

    # Generate data
    sigs = np.random.randn(N, D)
    sigs = sigs / np.linalg.norm(sigs, axis=1, keepdims=True)
    query = np.random.randn(D)
    query = query / np.linalg.norm(query)

    # Digital exact
    digital_scores = sigs @ query
    digital_rank = np.argsort(np.argsort(-digital_scores))
    digital_topk = set(np.argsort(-digital_scores)[:K])

    # Three-series color set per Tier 2C
    configs = [
        ("Nominal (5-bit, 20pm)", 5, 0.01,  0.01,  CLR_PRIMARY),
        ("Optimistic (6-bit, 10pm)", 6, 0.005, 0.005, CLR_SECONDARY),
        ("Pessimistic (4-bit, 40pm)", 4, 0.02,  0.02,  CLR_TERTIARY),
    ]

    results = {}
    for name, bits, drift_s, det_s, color in configs:
        mrr_scores, _, _, _, _, _ = mrr_impaired_scores(
            sigs, query, bits=bits, drift_sigma=drift_s, det_sigma=det_s)
        mrr_rank = np.argsort(np.argsort(-mrr_scores))
        mrr_topk = set(np.argsort(-mrr_scores)[:K])
        recall = len(digital_topk & mrr_topk) / K
        results[name] = {
            "scores": mrr_scores, "rank": mrr_rank,
            "topk": mrr_topk, "recall": recall, "color": color
        }

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.8))

    # -- (a) Score correlation scatter --
    ax = axes[0]
    for name, r in results.items():
        corr = np.corrcoef(digital_scores, r["scores"])[0, 1]
        ax.scatter(digital_scores, r["scores"], s=4, alpha=0.4,
                   color=r["color"], rasterized=True,
                   label=f"{name}\n($\\rho$={corr:.3f})")
    # Diagonal reference
    mn, mx = digital_scores.min(), digital_scores.max()
    ax.plot([mn, mx], [mn, mx], "--", color=CLR_THRESHOLD, lw=0.8)
    ax.set_xlabel("Digital score $\\mathbf{q} \\cdot \\mathbf{s}_n$")
    ax.set_ylabel("MRR score")
    ax.text(-0.02, 1.08, '(a)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='right')
    ax.legend(fontsize=6, loc="upper left", framealpha=0.9,
              handletextpad=0.3, labelspacing=0.6)

    # -- (b) Rank agreement (nominal) --
    ax = axes[1]
    nom_name = "Nominal (5-bit, 20pm)"
    r = results[nom_name]
    is_topk_d = np.array([i in digital_topk for i in range(N)])
    is_topk_m = np.array([i in r["topk"] for i in range(N)])
    both = is_topk_d & is_topk_m
    miss = is_topk_d & ~is_topk_m
    other = ~is_topk_d

    ax.scatter(digital_rank[other], r["rank"][other],
               s=2, alpha=0.15, color=CLR_GRAY, rasterized=True, label="Other")
    ax.scatter(digital_rank[both], r["rank"][both],
               s=25, alpha=0.9, color=CLR_PRIMARY, marker="^", zorder=5,
               label=f"Top-{K} match ({both.sum()}/{K})")
    if miss.any():
        ax.scatter(digital_rank[miss], r["rank"][miss],
                   s=25, alpha=0.9, color=CLR_TERTIARY, marker="x", zorder=5,
                   label=f"Top-{K} miss ({miss.sum()}/{K})")
    ax.plot([0, N], [0, N], "--", color=CLR_THRESHOLD, lw=0.8)
    ax.set_xlabel("Digital rank")
    ax.set_ylabel("MRR rank")
    ax.text(-0.02, 1.08, '(b)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='right')
    ax.set_title(f"Nominal config (R@{K}={r['recall']:.0%})", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.set_xlim(-5, N + 5)
    ax.set_ylim(-5, N + 5)
    ax.set_aspect("equal")

    # -- (c) Error histogram --
    ax = axes[2]
    for name, r in results.items():
        err = r["scores"] - digital_scores
        err_norm = err / (np.std(digital_scores) + 1e-10)
        ax.hist(err_norm, bins=40, alpha=0.5, color=r["color"],
                density=True, label=name)
    ax.axvline(x=0, color=CLR_THRESHOLD, ls="--", lw=0.8)
    ax.set_xlabel("Normalized score error")
    ax.set_ylabel("Density")
    ax.text(-0.02, 1.08, '(c)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='right')
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9)

    fig.tight_layout(w_pad=1.5)
    path = os.path.join(OUTDIR, "fig_digital_vs_photonic.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 3: Weight Encoding Fidelity
# ===========================================================================
def fig_weight_fidelity():
    """Three-panel figure:
    (a) Ideal vs quantized weights (scatter)
    (b) Ideal vs quantized+drift weights (scatter)
    (c) Error histograms for all configs
    """
    print("[3/4] Weight encoding fidelity...")

    np.random.seed(123)
    N = 128

    sigs = np.random.randn(N, D)
    sigs = sigs / np.linalg.norm(sigs, axis=1, keepdims=True)
    W_ideal, w_min, w_max = normalize_weights(sigs)

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.8))

    # Enable all four spines (closed box) for all panels
    for a in axes:
        for spine in a.spines.values():
            spine.set_visible(True)

    # -- (a) 5-bit quantization only --
    ax = axes[0]
    W_q5 = uniform_quantize(W_ideal, 5)
    err_q5 = W_q5 - W_ideal
    rmse_q5 = np.sqrt(np.mean(err_q5 ** 2))

    ax.scatter(W_ideal.ravel(), W_q5.ravel(),
               s=0.5, alpha=0.2, color=CLR_PRIMARY, rasterized=True)
    ax.plot([0, 1], [0, 1], "--", color=CLR_THRESHOLD, lw=0.8)
    ax.set_xlabel("Ideal weight")
    ax.set_ylabel("Quantized weight")
    ax.text(0.95, 0.05, f"RMSE = {rmse_q5:.4f}",
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            color='black')
    ax.text(-0.02, 1.08, '(a)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    # -- (b) 5-bit + thermal drift --
    ax = axes[1]
    W_d = add_thermal_drift(W_q5, 0.01)
    err_d = W_d - W_ideal
    rmse_d = np.sqrt(np.mean(err_d ** 2))

    ax.scatter(W_ideal.ravel(), W_d.ravel(),
               s=0.5, alpha=0.2, color=CLR_SECONDARY, rasterized=True)
    ax.plot([0, 1], [0, 1], "--", color=CLR_THRESHOLD, lw=0.8)
    ax.set_xlabel("Ideal weight")
    ax.set_ylabel("Quantized + drifted weight")
    ax.text(0.95, 0.05, f"RMSE = {rmse_d:.4f}",
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            color='black')
    ax.text(-0.02, 1.08, '(b)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    # -- (c) Error histograms --
    ax = axes[2]
    ax.hist(err_q5.ravel(), bins=50, alpha=0.5, color=CLR_PRIMARY,
            density=True, label=f"5-bit only\n({rmse_q5:.4f})")
    ax.hist(err_d.ravel(), bins=50, alpha=0.5, color=CLR_SECONDARY,
            density=True, label=f"5-bit+20pm\n({rmse_d:.4f})")

    # 4-bit + 30pm pessimistic
    W_q4 = uniform_quantize(W_ideal, 4)
    W_d4 = add_thermal_drift(W_q4, 0.02)
    err_d4 = W_d4 - W_ideal
    rmse_d4 = np.sqrt(np.mean(err_d4 ** 2))
    ax.hist(err_d4.ravel(), bins=50, alpha=0.5, color=CLR_TERTIARY,
            density=True, label=f"4-bit+30pm\n({rmse_d4:.4f})")

    ax.axvline(x=0, color=CLR_THRESHOLD, ls="--", lw=0.8)
    ax.set_xlabel("Weight error (actual $-$ ideal)")
    ax.set_ylabel("Density")
    ax.text(-0.02, 1.08, '(c)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='right')
    ax.legend(fontsize=5.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.9, title="Config (RMSE)", title_fontsize=5.5)

    fig.tight_layout(w_pad=1.5)
    fig.subplots_adjust(right=0.78)  # make room for legend outside
    path = os.path.join(OUTDIR, "fig_weight_fidelity.pdf")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 4: Combined Impairment 2D Sensitivity Heatmap
# ===========================================================================
def fig_combined_sensitivity():
    """2D heatmap: weight bits (y) x thermal drift (x) -> recall@8.
    50 Monte Carlo trials per cell.
    """
    print("[4/4] Combined impairment sensitivity heatmap...")

    np.random.seed(77)
    N_TRIALS = 50

    bit_values = [3, 4, 5, 6, 7, 8]
    # drift sigma in normalized weight space [0, 1]
    drift_sigmas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    # Convert to approximate pm for labels (FSR=12nm, FWHM~0.155nm)
    drift_pm_labels = [f"{s*1000:.0f}" for s in drift_sigmas]

    recall_matrix = np.zeros((len(bit_values), len(drift_sigmas)))

    for bi, bits in enumerate(bit_values):
        for di, drift_s in enumerate(drift_sigmas):
            recalls = []
            for _ in range(N_TRIALS):
                s_q = np.random.randn(D)
                s_q = s_q / np.linalg.norm(s_q)
                S_b = np.random.randn(N_BLOCKS, D)
                S_b = S_b / np.linalg.norm(S_b, axis=1, keepdims=True)

                exact_ip = S_b @ s_q
                exact_topk = set(np.argsort(-exact_ip)[:K])

                mrr_s, _, _, _, _, _ = mrr_impaired_scores(
                    S_b, s_q, bits=bits, drift_sigma=drift_s, det_sigma=0.01)
                mrr_topk = set(np.argsort(-mrr_s)[:K])

                recalls.append(len(exact_topk & mrr_topk) / K)
            recall_matrix[bi, di] = np.mean(recalls)
            print(f"    bits={bits}, drift_sigma={drift_s}: "
                  f"recall@{K}={recall_matrix[bi, di]:.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.5))

    # Tier 2C: use 'coolwarm' diverging colormap
    im = ax.imshow(recall_matrix, aspect="auto", origin="lower",
                   cmap="coolwarm", vmin=0.3, vmax=1.0,
                   extent=[-0.5, len(drift_sigmas) - 0.5,
                           -0.5, len(bit_values) - 0.5])

    # Annotate cells
    for bi in range(len(bit_values)):
        for di in range(len(drift_sigmas)):
            val = recall_matrix[bi, di]
            color = "white" if val < 0.55 else "black"
            ax.text(di, bi, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")

    ax.set_xticks(range(len(drift_sigmas)))
    ax.set_xticklabels(drift_pm_labels)
    ax.set_yticks(range(len(bit_values)))
    ax.set_yticklabels([str(b) for b in bit_values])
    ax.set_xlabel(r"Thermal drift $\sigma$ ($\times 10^{-3}$ weight units)")
    ax.set_ylabel("Weight precision (bits)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(f"Recall@{K}", fontsize=8)

    # Mark operating points with black markers for visibility
    # Nominal: 5-bit, sigma=0.01
    nom_bi = bit_values.index(5)
    nom_di = drift_sigmas.index(0.01)
    ax.plot(nom_di, nom_bi, "*", ms=18, mfc="none", mec=(0.4, 0.8, 0.1, 0.5), mew=1.5,
            zorder=5)
    ax.annotate("Nominal", xy=(nom_di, nom_bi),
                xytext=(nom_di + 0.5, nom_bi + 0.45),
                fontsize=7, color='black', fontweight='bold',
                arrowprops=dict(arrowstyle="-", color='black', lw=0.4))

    # Optimistic: 6-bit, sigma=0.005
    opt_bi = bit_values.index(6)
    opt_di = drift_sigmas.index(0.005)
    ax.plot(opt_di, opt_bi, "o", ms=12, mfc="none", mec=(0.4, 0.8, 0.1, 0.5), mew=1.5,
            zorder=5)
    ax.annotate("Optimistic", xy=(opt_di, opt_bi),
                xytext=(opt_di - 0.6, opt_bi + 0.45),
                fontsize=7, color='black', fontweight='bold',
                arrowprops=dict(arrowstyle="-", color='black', lw=0.4))

    # Pessimistic: 4-bit, sigma=0.02
    pes_bi = bit_values.index(4)
    pes_di = drift_sigmas.index(0.02)
    ax.plot(pes_di, pes_bi, "^", ms=12, mfc="none", mec=(0.4, 0.8, 0.1, 0.5), mew=1.5,
            zorder=5)
    ax.annotate("Pessimistic", xy=(pes_di, pes_bi),
                xytext=(pes_di + 0.45, pes_bi - 0.45),
                fontsize=7, color='black', fontweight='bold',
                arrowprops=dict(arrowstyle="-", color='black', lw=0.4))

    ax.set_title(f"Recall@{K}: weight precision vs. thermal drift\n"
                 f"($d$={D}, $N$={N_BLOCKS}, {N_TRIALS} trials)", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig_combined_sensitivity.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save data
    data = {
        "bit_values": bit_values,
        "drift_sigmas": drift_sigmas,
        "recall_matrix": recall_matrix.tolist(),
        "params": {"d": D, "N": N_BLOCKS, "K": K, "n_trials": N_TRIALS},
    }
    json_path = os.path.join(RESULTS_DIR, "hw_combined_sensitivity.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Data: {json_path}")


# ===========================================================================
if __name__ == "__main__":
    print(f"Output: {OUTDIR}")
    print("=" * 60)
    fig_mrr_lorentzian()
    fig_digital_vs_photonic()
    fig_weight_fidelity()
    fig_combined_sensitivity()
    print("=" * 60)
    print("All figures generated!")
