"""
PRISM Hardware-Aware Simulations
Simulate MRR weight bank impairments on signature recall.
Tasks: quantization, thermal drift, detector noise, combined, energy analysis.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

# Tier 2C rcParams
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
    'lines.linewidth': 1.3,
    'lines.markersize': 4,
})

# Tier 2C color palette
COLOR_PRIMARY   = '#1B3A5C'  # dark navy
COLOR_SECONDARY = '#4DB8C7'  # cyan/teal accent
COLOR_TERTIARY  = '#B8952A'  # muted gold
COLOR_QUATERNARY = '#A0522D' # muted red-brown
COLOR_QUINARY   = '#6B7B3A'  # olive
COLOR_THRESHOLD = '#888888'  # threshold gray
COLOR_LIGHT_FILL = '#D4E4EC' # light fill
COLOR_ERROR_BAND = '#E3EFF5' # error band

# Tier 2C figure sizes
FIGWIDTH_SINGLE = 3.35   # inches, single column
FIGHEIGHT_SINGLE = 2.5

# Output directory (relative)
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Simulation parameters
D = 64          # WDM channels / signature dimension
N_BLOCKS = 1024 # number of blocks
K = 32          # top-k
N_TRIALS = 100  # trials for statistics

np.random.seed(42)

def generate_data():
    """Generate query and block signatures from N(0,1)."""
    s_q = np.random.randn(D)
    s_q = s_q / np.linalg.norm(s_q)
    S_b = np.random.randn(N_BLOCKS, D)
    S_b = S_b / np.linalg.norm(S_b, axis=1, keepdims=True)
    return s_q, S_b

def compute_recall_at_k(exact_topk, approx_topk):
    """Recall@K = |exact_topk ∩ approx_topk| / K"""
    return len(set(exact_topk) & set(approx_topk)) / K

def uniform_quantize(w, bits):
    """Uniform quantization of weights in [-1,1] to given bit precision (balanced PD)."""
    levels = 2**bits
    w_clipped = np.clip(w, -1, 1)
    w_q = np.round((w_clipped + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
    return w_q

def normalize_weights(S_b):
    """Map signatures to [-1,1] range for balanced PD weights."""
    w_abs_max = np.max(np.abs(S_b))
    if w_abs_max < 1e-30:
        w_abs_max = 1e-30
    return S_b / w_abs_max, w_abs_max

def inner_products_from_weights(w_matrix, s_q, w_abs_max):
    """Compute inner products using weight matrix (in [-1,1] space)."""
    S_b_recon = w_matrix * w_abs_max
    return S_b_recon @ s_q

# ============================================================
# TASK 1: Weight Quantization
# ============================================================
print("=" * 60)
print("TASK 1: Weight Quantization Effect on Recall")
print("=" * 60)

bit_values = [2, 3, 4, 5, 6, 7, 8]
recall_quant = {b: [] for b in bit_values}

for trial in range(N_TRIALS):
    s_q, S_b = generate_data()

    # Exact inner products
    exact_ip = S_b @ s_q
    exact_topk = np.argsort(exact_ip)[-K:]

    # Normalize to [0,1] for MRR weights
    W, w_abs_max = normalize_weights(S_b)

    for bits in bit_values:
        W_q = uniform_quantize(W, bits)
        approx_ip = inner_products_from_weights(W_q, s_q, w_abs_max)
        approx_topk = np.argsort(approx_ip)[-K:]
        recall_quant[bits].append(compute_recall_at_k(exact_topk, approx_topk))

quant_results = {}
for bits in bit_values:
    mean_r = np.mean(recall_quant[bits])
    std_r = np.std(recall_quant[bits])
    quant_results[str(bits)] = {"mean": float(mean_r), "std": float(std_r)}
    print(f"  {bits}-bit: Recall@{K} = {mean_r:.4f} ± {std_r:.4f}")

# Save data
with open(os.path.join(RESULTS_DIR, "hw_quantization.json"), "w") as f:
    json.dump({
        "description": "Recall@32 vs weight quantization bit precision",
        "parameters": {"d": D, "N_blocks": N_BLOCKS, "K": K, "n_trials": N_TRIALS},
        "bit_values": bit_values,
        "results": quant_results
    }, f, indent=2)

# Plot — Fig: Recall vs Precision (3-series per fig prompt)
fig, ax = plt.subplots(figsize=(FIGWIDTH_SINGLE, FIGHEIGHT_SINGLE))
means = [quant_results[str(b)]["mean"] for b in bit_values]
stds = [quant_results[str(b)]["std"] for b in bit_values]
ax.errorbar(bit_values, means, yerr=stds, fmt='o-', color=COLOR_PRIMARY,
            capsize=3, capthick=0.8, elinewidth=0.8, markeredgecolor='black',
            markeredgewidth=0.4, zorder=3, label='Quantization only')
ax.axhline(y=1.0, color=COLOR_TERTIARY, linestyle='--', linewidth=0.8,
           alpha=0.8, label='FP ideal (1.0)')
ax.axhline(y=0.90, color=COLOR_QUATERNARY, linestyle=':', linewidth=0.8,
           alpha=0.8, label='90% threshold')
ax.set_xlabel('Weight Precision (bits)')
ax.set_ylabel(f'Recall@{K}')
ax.set_ylim([0.30, 1.05])
ax.set_xticks(bit_values)
ax.legend(loc='lower right')
fig.savefig(os.path.join(FIG_DIR, "fig_recall_vs_precision.pdf"))
plt.close(fig)
print("  Saved: fig_recall_vs_precision.pdf\n")


# ============================================================
# TASK 2: Thermal Drift
# ============================================================
print("=" * 60)
print("TASK 2: Thermal Drift Effect on Recall")
print("=" * 60)

sigma_thermal_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
recall_thermal = {s: [] for s in sigma_thermal_values}

for trial in range(N_TRIALS):
    s_q, S_b = generate_data()
    exact_ip = S_b @ s_q
    exact_topk = np.argsort(exact_ip)[-K:]

    W, w_abs_max = normalize_weights(S_b)

    for sigma in sigma_thermal_values:
        W_noisy = W + np.random.randn(*W.shape) * sigma
        W_noisy = np.clip(W_noisy, -1, 1)  # physical: balanced PD weight in [-1,1]
        approx_ip = inner_products_from_weights(W_noisy, s_q, w_abs_max)
        approx_topk = np.argsort(approx_ip)[-K:]
        recall_thermal[sigma].append(compute_recall_at_k(exact_topk, approx_topk))

thermal_results = {}
for sigma in sigma_thermal_values:
    mean_r = np.mean(recall_thermal[sigma])
    std_r = np.std(recall_thermal[sigma])
    thermal_results[str(sigma)] = {"mean": float(mean_r), "std": float(std_r)}
    # Convert sigma to pm drift (FSR=30nm)
    pm_drift = sigma * 30e3  # sigma * FSR_in_pm
    print(f"  sigma={sigma:.3f} (~{pm_drift:.0f}pm drift): Recall@{K} = {mean_r:.4f} ± {std_r:.4f}")

with open(os.path.join(RESULTS_DIR, "hw_thermal_drift.json"), "w") as f:
    json.dump({
        "description": "Recall@32 vs thermal drift sigma (normalized weight noise)",
        "parameters": {"d": D, "N_blocks": N_BLOCKS, "K": K, "n_trials": N_TRIALS,
                       "note": "sigma in normalized weight space [-1,1]; FSR=30nm; balanced PD"},
        "sigma_values": sigma_thermal_values,
        "results": thermal_results
    }, f, indent=2)

# Plot — Fig: Recall vs Drift
fig, ax = plt.subplots(figsize=(FIGWIDTH_SINGLE, FIGHEIGHT_SINGLE))
means = [thermal_results[str(s)]["mean"] for s in sigma_thermal_values]
stds = [thermal_results[str(s)]["std"] for s in sigma_thermal_values]
pm_drifts = [s * 30e3 for s in sigma_thermal_values]  # convert to pm

ax.errorbar(sigma_thermal_values, means, yerr=stds, fmt='o-', color=COLOR_PRIMARY,
            capsize=3, capthick=0.8, elinewidth=0.8, markeredgecolor='black',
            markeredgewidth=0.4, zorder=3)

# Vertical shaded band for standard thermal stabilization (sigma <= 0.005)
ax.axvspan(0, 0.005, color=COLOR_SECONDARY, alpha=0.15, zorder=0)
ax.text(0.0025, 0.75,
        'Standard thermal\nstabilization', fontsize=7, ha='center', va='center',
        color='black')

ax.axhline(y=0.95, color=COLOR_TERTIARY, linestyle='--', linewidth=0.8,
           alpha=0.8, label='95%')
ax.axhline(y=0.90, color=COLOR_QUATERNARY, linestyle=':', linewidth=0.8,
           alpha=0.8, label='90%')

# Annotate the sigma=0.01 point
sigma_01_val = thermal_results.get("0.01", {}).get("mean")
if sigma_01_val is not None:
    ax.annotate(f'{sigma_01_val:.1%}',
                xy=(0.01, sigma_01_val),
                xytext=(0.025, sigma_01_val + 0.03),
                fontsize=8, color=COLOR_TERTIARY, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.4))

ax.set_xlabel(r'Thermal Drift $\sigma_\mathrm{th}$ (normalized)')
ax.set_ylabel(f'Recall@{K}')
ax.set_ylim([0.55, 1.05])
ax.set_xscale('log')
ax.legend(loc='lower left')

# Secondary x-axis for pm drift
ax2 = ax.twiny()
ax2.set_xscale('log')
ax2.set_xlim([s * 30e3 for s in ax.get_xlim()])
ax2.set_xlabel('Physical drift (pm)', fontsize=8)
ax2.tick_params(labelsize=8)
ax2.spines['right'].set_visible(False)

fig.savefig(os.path.join(FIG_DIR, "fig_recall_vs_drift.pdf"))
plt.close(fig)
print("  Saved: fig_recall_vs_drift.pdf\n")


# ============================================================
# TASK 3: Detector Noise (Shot Noise)
# ============================================================
print("=" * 60)
print("TASK 3: Detector Noise (Shot Noise) Effect on Recall")
print("=" * 60)

sigma_det_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
recall_det = {s: [] for s in sigma_det_values}

for trial in range(N_TRIALS):
    s_q, S_b = generate_data()
    exact_ip = S_b @ s_q
    exact_topk = np.argsort(exact_ip)[-K:]

    for sigma_det in sigma_det_values:
        # Balanced PD shot noise: both PDs contribute independently
        signal = S_b @ s_q
        noise_std = np.sqrt(2) * np.sqrt(np.abs(signal)) * sigma_det
        noisy_ip = signal + np.random.randn(N_BLOCKS) * noise_std
        approx_topk = np.argsort(noisy_ip)[-K:]
        recall_det[sigma_det].append(compute_recall_at_k(exact_topk, approx_topk))

det_results = {}
for sigma_det in sigma_det_values:
    mean_r = np.mean(recall_det[sigma_det])
    std_r = np.std(recall_det[sigma_det])
    det_results[str(sigma_det)] = {"mean": float(mean_r), "std": float(std_r)}
    print(f"  sigma_det={sigma_det:.3f}: Recall@{K} = {mean_r:.4f} ± {std_r:.4f}")

# Plot — Fig: Recall vs Noise
fig, ax = plt.subplots(figsize=(FIGWIDTH_SINGLE, FIGHEIGHT_SINGLE))
means = [det_results[str(s)]["mean"] for s in sigma_det_values]
stds = [det_results[str(s)]["std"] for s in sigma_det_values]

ax.errorbar(sigma_det_values, means, yerr=stds, fmt='o-', color=COLOR_PRIMARY,
            capsize=3, capthick=0.8, elinewidth=0.8, markeredgecolor='black',
            markeredgewidth=0.4, zorder=3)

# Confidence band (dark navy at 20% opacity)
means_arr = np.array(means)
stds_arr = np.array(stds)
ax.fill_between(sigma_det_values, means_arr - stds_arr, means_arr + stds_arr,
                color=COLOR_PRIMARY, alpha=0.20, zorder=1)

# Vertical shaded band for standard Ge-on-Si PD (sigma_det <= 0.01)
ax.axvspan(0, 0.01, color=COLOR_SECONDARY, alpha=0.15, zorder=0)
ax.text(0.005, 0.75, 'Standard\nGe-on-Si PD', fontsize=7, ha='center', va='center',
        color='black')

ax.axhline(y=0.95, color=COLOR_THRESHOLD, linestyle='--', linewidth=0.6,
           alpha=0.7, label='95% threshold')
ax.set_xlabel(r'Detector Noise $\sigma_\mathrm{det}$')
ax.set_ylabel(f'Recall@{K}')
ax.set_ylim([0.45, 1.05])
ax.set_xscale('log')
ax.legend(loc='lower left')
fig.savefig(os.path.join(FIG_DIR, "fig_recall_vs_noise.pdf"))
plt.close(fig)
print("  Saved: fig_recall_vs_noise.pdf\n")


# ============================================================
# TASK 4: Combined Impairment
# ============================================================
print("=" * 60)
print("TASK 4: Combined Impairment (6-bit + thermal + detector)")
print("=" * 60)

COMBINED_BITS = 6
COMBINED_SIGMA_TH = 0.01
COMBINED_SIGMA_DET = 0.01
recall_combined = []

for trial in range(N_TRIALS):
    s_q, S_b = generate_data()
    exact_ip = S_b @ s_q
    exact_topk = np.argsort(exact_ip)[-K:]

    W, w_abs_max = normalize_weights(S_b)

    # 1. Quantize
    W_q = uniform_quantize(W, COMBINED_BITS)

    # 2. Thermal drift
    W_noisy = W_q + np.random.randn(*W_q.shape) * COMBINED_SIGMA_TH
    W_noisy = np.clip(W_noisy, -1, 1)

    # 3. Compute inner products then add detector noise
    S_b_recon = W_noisy * w_abs_max
    signal = S_b_recon @ s_q
    noise_std = np.sqrt(2) * np.sqrt(np.abs(signal)) * COMBINED_SIGMA_DET
    noisy_ip = signal + np.random.randn(N_BLOCKS) * noise_std

    approx_topk = np.argsort(noisy_ip)[-K:]
    recall_combined.append(compute_recall_at_k(exact_topk, approx_topk))

combined_mean = float(np.mean(recall_combined))
combined_std = float(np.std(recall_combined))
print(f"  Combined: Recall@{K} = {combined_mean:.4f} ± {combined_std:.4f}")
print(f"  Config: {COMBINED_BITS}-bit quantization, sigma_th={COMBINED_SIGMA_TH}, sigma_det={COMBINED_SIGMA_DET}")

# Also compute individual contributions at same parameters for comparison
individual_at_config = {}
# 6-bit only
r6 = quant_results["6"]
individual_at_config["quantization_only"] = r6
# thermal 0.01 only
rt = thermal_results["0.01"]
individual_at_config["thermal_only"] = rt
# detector 0.01 only
rd = det_results["0.01"]
individual_at_config["detector_only"] = rd

with open(os.path.join(RESULTS_DIR, "hw_combined.json"), "w") as f:
    json.dump({
        "description": "Combined impairment: 6-bit quantization + thermal drift + detector noise",
        "parameters": {
            "d": D, "N_blocks": N_BLOCKS, "K": K, "n_trials": N_TRIALS,
            "bits": COMBINED_BITS,
            "sigma_thermal": COMBINED_SIGMA_TH,
            "sigma_detector": COMBINED_SIGMA_DET
        },
        "combined_recall": {"mean": combined_mean, "std": combined_std},
        "individual_contributions": individual_at_config,
        "note": "Individual contributions at same parameter values for comparison"
    }, f, indent=2)
print("  Saved: hw_combined.json\n")


# ============================================================
# TASK 5: Energy Analysis
# ============================================================
print("=" * 60)
print("TASK 5: Energy Analysis")
print("=" * 60)

d = 64    # WDM channels
N = 1024  # blocks
latency_ns = 9  # ns per query

# Component powers (mW)
components = {
    "Laser (CW)": {"power_mW": 10.0, "type": "static", "note": "10mW total, always on"},
    "DAC (d channels)": {"power_mW": d * 0.5, "type": "dynamic", "note": f"{d} ch x 0.5 mW"},
    "MZM modulators": {"power_mW": d * 0.2, "type": "dynamic", "note": f"{d} ch x 0.2 mW"},
    "MRR tuning (TFLN EO)": {"power_mW": 0.0, "type": "static", "note": "Pockels EO: capacitive, near-zero static power"},
    "Photodetectors": {"power_mW": N * 0.01, "type": "dynamic", "note": f"{N} x 0.01 mW"},
    "ADC": {"power_mW": N * 0.1, "type": "dynamic", "note": f"{N} x 0.1 mW"},
    "Top-k logic": {"power_mW": 1.0, "type": "dynamic", "note": "Digital comparator tree"}
}

total_power = sum(c["power_mW"] for c in components.values())
dynamic_power = sum(c["power_mW"] for c in components.values() if c["type"] == "dynamic")
static_power = sum(c["power_mW"] for c in components.values() if c["type"] == "static")

# Per-query energy
dynamic_energy_nJ = dynamic_power * latency_ns * 1e-3  # mW * ns = pJ, *1e-3 = nJ
static_energy_per_query_uJ = static_power * 1e-3  # mW * 1ms = uJ (at 1 query/ms)

# Baselines (from literature)
baselines = {
    "GPU full scan (A100, 128K)": {"energy_uJ": 50, "note": "Memory BW limited, ~2 TB/s, 7 GB KV"},
    "GPU ANN (FAISS)": {"energy_uJ": 5, "note": "Approximate nearest neighbor"},
    "NVIDIA ICMS (estimated)": {"energy_uJ": 10, "note": "In-memory compute"}
}

print(f"\n  {'Component':<25} {'Power (mW)':>12} {'Type':>10}")
print(f"  {'-'*25} {'-'*12} {'-'*10}")
for name, info in components.items():
    print(f"  {name:<25} {info['power_mW']:>12.1f} {info['type']:>10}")
print(f"  {'-'*25} {'-'*12} {'-'*10}")
print(f"  {'Total':<25} {total_power:>12.1f}")
print(f"  {'Dynamic':<25} {dynamic_power:>12.1f}")
print(f"  {'Static':<25} {static_power:>12.1f}")
print()
print(f"  Latency: {latency_ns} ns")
print(f"  Dynamic energy/query: {dynamic_energy_nJ:.2f} nJ ({dynamic_energy_nJ*1e-3:.4f} uJ)")
print(f"  Static (laser + TEC) energy @ 1 query/ms: {static_energy_per_query_uJ:.4f} uJ")
print()
print(f"  Baselines:")
for name, info in baselines.items():
    print(f"    {name}: {info['energy_uJ']} uJ/query")
print()
print(f"  PRISM vs GPU full scan: {50/static_energy_per_query_uJ:.0f}x (at 1 kHz query rate)")
print(f"  PRISM dynamic-only vs GPU ANN: {5/(dynamic_energy_nJ*1e-3):.0f}x")

energy_data = {
    "description": "PRISM energy analysis per query",
    "parameters": {"d": d, "N_blocks": N, "K": K, "latency_ns": latency_ns},
    "components": components,
    "totals": {
        "total_power_mW": total_power,
        "dynamic_power_mW": dynamic_power,
        "static_power_mW": static_power,
        "dynamic_energy_per_query_nJ": round(dynamic_energy_nJ, 4),
        "static_energy_per_query_uJ_at_1kHz": round(static_energy_per_query_uJ, 2),
        "total_energy_per_query_uJ_at_1kHz": round(static_energy_per_query_uJ + dynamic_energy_nJ * 1e-3, 4)
    },
    "electronic_baselines": baselines,
    "speedup_vs_gpu_fullscan_at_1kHz": round(50 / static_energy_per_query_uJ, 1),
    "speedup_dynamic_vs_gpu_ann": round(5 / (dynamic_energy_nJ * 1e-3), 0)
}

with open(os.path.join(RESULTS_DIR, "energy_analysis.json"), "w") as f:
    json.dump(energy_data, f, indent=2)
print("\n  Saved: energy_analysis.json")

print("\n" + "=" * 60)
print("ALL TASKS COMPLETE")
print("=" * 60)
