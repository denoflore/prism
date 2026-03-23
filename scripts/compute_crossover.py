"""
PRISM crossover analysis: compute energy/latency crossover points
vs GPU full scan, GPU ANN (FAISS), and NVIDIA ICMS baselines.
Generates figures for paper.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import json, os

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

# Tier 2C diverging colormap: navy-white-gold
cmap_diverging = LinearSegmentedColormap.from_list(
    'navy_white_gold',
    [COLOR_PRIMARY, '#FFFFFF', COLOR_TERTIARY]
)

# Output directory (relative)
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────
B = 128          # block size (tokens)
k = 32           # top-k selected blocks
d_h = 128        # head dimension (Qwen2.5-7B)
b_prec = 2       # bytes per element (bf16)

# PRISM hardware (from paper Table tab:prism_energy in hardware_analysis.tex)
PLATFORM = "TFLN"  # TFLN Pockels EO (capacitive) — zero static heater power
t_prism_ns = 9.0                     # total PRISM latency (ns)
P_per_MRR_uW = 0.0                   # TFLN: Pockels EO is capacitive, ~0 static power
P_per_MRR_uW_SOI = 24.4              # SOI thermo-optic heater power (µW) — for comparison only

# Component power parameters (from paper Table: tab:prism_energy)
# d-proportional components
P_DAC_per_ch_mW = 0.5       # DAC: 0.5 mW per WDM channel
P_MZM_per_ch_mW = 0.1       # MZM: 0.1 mW per channel (6.4/64)
# N-proportional components (balanced PD: 2N detectors)
P_PD_per_block_mW = 0.005 * 2   # PD: 0.005 mW × 2 (balanced) per block
P_ADC_per_block_mW = 0.049 * 2  # TIA+ADC: 0.049 mW × 2 (balanced) per block
# Fixed components
P_laser_mW = 100.0          # CW laser source
P_driver_mW = 5.0           # Voltage driver array
P_topk_mW = 1.0             # Top-k digital logic

def E_prism_dynamic_pJ_func(d, N):
    """Dynamic energy per query (pJ) — depends on d and N.
    Matches paper Table tab:prism_energy.
    At d=64, N=1024: returns ~2295 pJ (paper rounds to 2290)."""
    P_total_mW = (P_DAC_per_ch_mW + P_MZM_per_ch_mW) * d + \
                 (P_PD_per_block_mW + P_ADC_per_block_mW) * N + \
                 P_laser_mW + P_driver_mW + P_topk_mW
    return P_total_mW * t_prism_ns  # mW × ns = pJ

# Electronic baselines at reference point n_ref=128K
n_ref = 128_000
E_gpu_full_uJ_ref = 50.0   # GPU full scan energy (µJ) at n_ref
E_gpu_ann_uJ_ref = 5.0     # GPU ANN (FAISS) energy (µJ) at n_ref
E_icms_uJ_ref = 10.0       # NVIDIA ICMS energy (µJ) at n_ref

L_gpu_full_us_ref = 5.0    # GPU full scan latency (µs) at n_ref
L_gpu_ann_us_ref = 1.0     # GPU ANN latency (µs) at n_ref
L_icms_us_ref = 0.5        # ICMS latency (µs) at n_ref

# ─── Scaling models ──────────────────────────────────────────
def N_blocks(n):
    return n / B

# Energy models (µJ)
def E_gpu_full(n):
    """GPU full scan: linear in n (memory-bandwidth limited)"""
    return E_gpu_full_uJ_ref * (n / n_ref)

def E_gpu_ann(n):
    """GPU ANN (FAISS IVF-PQ): O(sqrt(N)) probes"""
    return E_gpu_ann_uJ_ref * np.sqrt(n / n_ref)

def E_icms(n):
    """NVIDIA ICMS: linear scan + flash read"""
    return E_icms_uJ_ref * (n / n_ref)

def E_prism_select(n, d):
    """PRISM selection energy per query (µJ)
    Dynamic (d,N-dependent) + heater power × query time
    """
    N = N_blocks(n)
    n_mrr = N * d
    P_heater_W = n_mrr * P_per_MRR_uW * 1e-6  # W
    E_heater_J = P_heater_W * t_prism_ns * 1e-9  # J
    E_dynamic_pJ = E_prism_dynamic_pJ_func(d, N)
    E_dynamic_J = E_dynamic_pJ * 1e-12  # J
    return (E_heater_J + E_dynamic_J) * 1e6  # µJ

def E_prism_total(n, d):
    """PRISM total: selection + fetch of k blocks (same GPU memory cost)"""
    E_sel = E_prism_select(n, d)
    # Fetch k blocks from GPU memory (same cost model)
    E_fetch = E_gpu_full_uJ_ref * (k * B / n_ref)  # fetch k*B tokens
    return E_sel + E_fetch

# Latency models (µs)
def L_gpu_full(n):
    return L_gpu_full_us_ref * (n / n_ref)

def L_gpu_ann(n):
    return L_gpu_ann_us_ref * np.sqrt(n / n_ref)

def L_icms(n):
    return L_icms_us_ref * (n / n_ref)

def L_prism(n):
    """PRISM latency: fixed 9ns + fetch latency for k blocks"""
    L_sel = t_prism_ns * 1e-3  # µs
    # Fetch k blocks from memory
    L_fetch = L_gpu_full_us_ref * (k * B / n_ref)  # fetch time
    return L_sel + L_fetch

# ─── Find crossover points ───────────────────────────────────
n_range = np.logspace(np.log10(1024), np.log10(10_000_000), 2000)
d_vals = [32, 64, 128]

results = {}

for d in d_vals:
    res = {"d": d}

    # Energy crossovers (where PRISM total < baseline)
    e_prism = np.array([E_prism_total(n, d) for n in n_range])
    e_full = np.array([E_gpu_full(n) for n in n_range])
    e_ann = np.array([E_gpu_ann(n) for n in n_range])
    e_icms = np.array([E_icms(n) for n in n_range])

    # Find crossover (where ratio crosses 1.0)
    ratio_full = e_prism / e_full
    ratio_ann = e_prism / e_ann
    ratio_icms = e_prism / e_icms

    def find_crossover(ratio, n_arr):
        """Find n where ratio drops below 1.0"""
        idx = np.where(ratio < 1.0)[0]
        if len(idx) == 0:
            return None
        return n_arr[idx[0]]

    n_cross_full = find_crossover(ratio_full, n_range)
    n_cross_ann = find_crossover(ratio_ann, n_range)
    n_cross_icms = find_crossover(ratio_icms, n_range)

    res["energy_crossover"] = {
        "vs_gpu_full": int(n_cross_full) if n_cross_full else "never",
        "vs_gpu_ann": int(n_cross_ann) if n_cross_ann else "never",
        "vs_icms": int(n_cross_icms) if n_cross_icms else "never",
    }

    # Latency crossovers
    l_prism = np.array([L_prism(n) for n in n_range])
    l_full = np.array([L_gpu_full(n) for n in n_range])
    l_ann = np.array([L_gpu_ann(n) for n in n_range])
    l_icms_arr = np.array([L_icms(n) for n in n_range])

    lr_full = l_prism / l_full
    lr_ann = l_prism / l_ann
    lr_icms = l_prism / l_icms_arr

    n_lat_full = find_crossover(lr_full, n_range)
    n_lat_ann = find_crossover(lr_ann, n_range)
    n_lat_icms = find_crossover(lr_icms, n_range)

    res["latency_crossover"] = {
        "vs_gpu_full": int(n_lat_full) if n_lat_full else "never",
        "vs_gpu_ann": int(n_lat_ann) if n_lat_ann else "never",
        "vs_icms": int(n_lat_icms) if n_lat_icms else "never",
    }

    # Energy ratio at key context lengths
    for n_test in [16384, 65536, 131072, 524288, 1048576]:
        e_p = E_prism_total(n_test, d)
        res[f"ratio_{n_test//1024}K"] = {
            "vs_full": round(e_p / E_gpu_full(n_test), 4),
            "vs_ann": round(e_p / E_gpu_ann(n_test), 4),
            "prism_uJ": round(e_p, 4),
            "gpu_full_uJ": round(E_gpu_full(n_test), 2),
        }

    results[f"d={d}"] = res

# Save results
with open(os.path.join(RESULTS_DIR, "crossover_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)
print("=== Crossover Results ===")
for dk, rv in results.items():
    print(f"\n{dk}:")
    print(f"  Energy crossover: {rv['energy_crossover']}")
    print(f"  Latency crossover: {rv['latency_crossover']}")
    for key in rv:
        if key.startswith("ratio_"):
            print(f"  {key}: {rv[key]}")

# ─── Figure 1: Crossover contour plot ─────────────────────────
# Tier 2C three-panel: figsize=(7.0, 2.5)
fig, axes = plt.subplots(3, 1, figsize=(3.4, 7.0), sharex=True)

n_grid = np.logspace(np.log10(2048), np.log10(4_000_000), 200)
d_grid = np.linspace(16, 128, 100)
N_n, D_d = np.meshgrid(n_grid, d_grid)

baselines_list = [
    ("GPU Full Scan", E_gpu_full),
    ("GPU ANN (FAISS)", E_gpu_ann),
    ("NVIDIA ICMS", E_icms),
]

panel_labels = ['(a)', '(b)', '(c)']

for idx, (ax, (name, e_baseline)) in enumerate(zip(axes, baselines_list)):
    ratio = np.zeros_like(N_n)
    for i in range(len(d_grid)):
        for j in range(len(n_grid)):
            ratio[i, j] = E_prism_total(n_grid[j], d_grid[i]) / e_baseline(n_grid[j])

    # Contour fill with Tier 2C diverging colormap (navy-white-gold)
    levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    cf = ax.contourf(N_n / 1000, D_d, ratio, levels=levels,
                     norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=10),
                     cmap=cmap_diverging, extend='both')

    # Crossover contour (ratio = 1) — thick black line
    cs = ax.contour(N_n / 1000, D_d, ratio, levels=[1.0],
                    colors='black', linewidths=2.0)
    ax.clabel(cs, fmt='1.0', fontsize=7)

    ax.set_xscale('log')
    if idx == 2:
        ax.set_xlabel('Context length (K tokens)')
    ax.set_ylabel('Signature dimension $d$')
    ax.set_title(f'vs. {name}', fontsize=9)

    # Panel label outside top-left
    ax.text(-0.02, 1.06, panel_labels[idx], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='right')

    # Re-enable spines for contour plots (need all four for visual containment)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Mark key points
    for d_mark in [32, 64]:
        cross = results[f"d={d_mark}"]["energy_crossover"]
        if "Full" in name:
            n_c = cross["vs_gpu_full"]
        elif "ANN" in name:
            n_c = cross["vs_gpu_ann"]
        else:
            n_c = cross["vs_icms"]
        if isinstance(n_c, int):
            ax.plot(n_c/1000, d_mark, 'k*', markersize=8, zorder=5)

    # Colorbar for each panel
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.05,
                        label=r'$C_{\mathrm{PRISM}} / C_{\mathrm{baseline}}$')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_crossover_contour.pdf"),
            bbox_inches='tight', dpi=300)
print("Saved fig_crossover_contour.pdf")

# ─── Figure 2: Scaling projection (log-log) ──────────────────
# Tier 2C two-panel: figsize=(7.0, 2.8)
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 5.0))

n_proj = np.logspace(np.log10(4096), 7, 500)  # 4K to 10M

# Tier 2C three-series colors for d values
colors_d = {32: COLOR_PRIMARY, 64: COLOR_SECONDARY, 128: COLOR_TERTIARY}

# (a) Energy ratio vs context length
for d in d_vals:
    e_p = np.array([E_prism_total(n, d) for n in n_proj])
    e_f = np.array([E_gpu_full(n) for n in n_proj])
    ratio = e_p / e_f
    ax1.loglog(n_proj / 1000, ratio, '-', color=colors_d[d],
               linewidth=1.3, label=f'$d = {d}$')

ax1.axhline(1.0, color=COLOR_THRESHOLD, linestyle='--', linewidth=0.6, alpha=0.7,
            label='Break-even')
ax1.set_xlabel('Context length (K tokens)')
ax1.set_ylabel(r'$C_{\mathrm{PRISM}} / C_{\mathrm{GPU\ full\ scan}}$')
ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')
ax1.legend(fontsize=7)
ax1.set_ylim(1e-4, 10)
ax1.text(0.05, 0.15, 'PRISM favorable', transform=ax1.transAxes,
         fontsize=7, color=COLOR_PRIMARY, fontstyle='italic', alpha=0.7)
ax1.text(0.05, 0.80, 'GPU favorable', transform=ax1.transAxes,
         fontsize=7, color=COLOR_QUATERNARY, fontstyle='italic', alpha=0.7)

# (b) Latency comparison
l_p = np.array([L_prism(n) for n in n_proj])
l_f = np.array([L_gpu_full(n) for n in n_proj])
l_a = np.array([L_gpu_ann(n) for n in n_proj])
l_i = np.array([L_icms(n) for n in n_proj])

ax2.loglog(n_proj / 1000, l_f, '-', color=COLOR_QUATERNARY, linewidth=1.3,
           label='GPU full scan')
ax2.loglog(n_proj / 1000, l_a, '--', color=COLOR_SECONDARY, linewidth=1.3,
           label='GPU ANN')
ax2.loglog(n_proj / 1000, l_i, '-.', color=COLOR_QUINARY, linewidth=1.3,
           label='NVIDIA ICMS')
ax2.loglog(n_proj / 1000, l_p, '-', color=COLOR_PRIMARY, linewidth=1.5,
           label='PRISM')

ax2.set_xlabel('Context length (K tokens)')
ax2.set_ylabel(r'Latency ($\mu$s)')
ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold', va='top')
ax2.legend(fontsize=7)

# Annotate speedup at 1M
n_1m = 1_000_000
speedup = L_gpu_full(n_1m) / L_prism(n_1m)
ax2.annotate(f'{speedup:.0f}x gap',
             xy=(n_1m/1000, L_prism(n_1m)),
             xytext=(n_1m/1000 * 0.3, L_prism(n_1m) * 5),
             fontsize=8, arrowprops=dict(arrowstyle='->', color='black', lw=0.4),
             ha='center')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_scaling_projection.pdf"),
            bbox_inches='tight', dpi=300)
print("Saved fig_scaling_projection.pdf")

# ─── Figure 3: Sensitivity analysis (SOI comparison) ─────────
fig3, ax3 = plt.subplots(figsize=(3.35, 2.5))

# NOTE: This sweep is for SOI thermo-optic platform comparison only.
# TFLN Pockels EO has ~0 heater power, so this plot shows the SOI case.
P_mrr_range = np.array([5, 10, 20, 50, 100, 200, 500])  # µW per MRR (SOI)
d_test = 64

crossovers_full = []
crossovers_ann = []
for p in P_mrr_range:
    # Temporarily override P_per_MRR_uW
    for n_test in n_range:
        N = N_blocks(n_test)
        n_mrr = N * d_test
        P_heater_W = n_mrr * p * 1e-6
        E_heater_J = P_heater_W * t_prism_ns * 1e-9
        E_dynamic_J = E_prism_dynamic_pJ_func(d_test, N) * 1e-12
        E_sel = (E_heater_J + E_dynamic_J) * 1e6
        E_fetch = E_gpu_full_uJ_ref * (k * B / n_ref)
        E_total = E_sel + E_fetch

        if E_total < E_gpu_full(n_test):
            crossovers_full.append(n_test)
            break
    else:
        crossovers_full.append(np.nan)

    for n_test in n_range:
        N = N_blocks(n_test)
        n_mrr = N * d_test
        P_heater_W = n_mrr * p * 1e-6
        E_heater_J = P_heater_W * t_prism_ns * 1e-9
        E_dynamic_J = E_prism_dynamic_pJ_func(d_test, N) * 1e-12
        E_sel = (E_heater_J + E_dynamic_J) * 1e6
        E_fetch = E_gpu_full_uJ_ref * (k * B / n_ref)
        E_total = E_sel + E_fetch

        if E_total < E_gpu_ann(n_test):
            crossovers_ann.append(n_test)
            break
    else:
        crossovers_ann.append(np.nan)

ax3.semilogy(P_mrr_range, np.array(crossovers_full)/1000, 'o-',
             color=COLOR_PRIMARY, linewidth=1.3, markersize=5,
             label='vs. GPU full scan')
ax3.semilogy(P_mrr_range, np.array(crossovers_ann)/1000, 's-',
             color=COLOR_SECONDARY, linewidth=1.3, markersize=5,
             label='vs. GPU ANN')
ax3.axhline(128, color=COLOR_THRESHOLD, linestyle=':', alpha=0.5, label='128K context')
ax3.axhline(1000, color=COLOR_THRESHOLD, linestyle='--', alpha=0.5, label='1M context')

ax3.set_xlabel(r'Heater power per MRR ($\mu$W) — SOI comparison')
ax3.set_ylabel('Crossover context length (K tokens)')
ax3.set_title(f'SOI heater sensitivity ($d = {d_test}$)', fontsize=9)
ax3.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_sensitivity_heater.pdf"),
            bbox_inches='tight', dpi=300)
print("Saved fig_sensitivity_heater.pdf")

# ─── Print summary ───────────────────────────────────────────
print("\n" + "="*60)
print("CROSSOVER SUMMARY FOR PAPER")
print("="*60)
for dk, rv in results.items():
    print(f"\n{dk}:")
    ec = rv["energy_crossover"]
    lc = rv["latency_crossover"]
    for baseline in ["vs_gpu_full", "vs_gpu_ann", "vs_icms"]:
        e_val = ec[baseline]
        l_val = lc[baseline]
        e_str = f"{e_val/1000:.1f}K" if isinstance(e_val, int) else e_val
        l_str = f"{l_val/1000:.1f}K" if isinstance(l_val, int) else l_val
        print(f"  {baseline}: energy crossover = {e_str}, latency = {l_str}")

# Heater halving effect (SOI comparison only)
d_test = 64
n_cross_base = results["d=64"]["energy_crossover"]["vs_gpu_full"]
# Compute with half heater power (SOI scenario)
orig_P = P_per_MRR_uW_SOI
half_P = orig_P / 2
# At half power, crossover shifts
for n_test in n_range:
    N = N_blocks(n_test)
    n_mrr = N * d_test
    P_heater_W = n_mrr * half_P * 1e-6
    E_heater_J = P_heater_W * t_prism_ns * 1e-9
    E_dynamic_J = E_prism_dynamic_pJ_func(d_test, N) * 1e-12
    E_sel = (E_heater_J + E_dynamic_J) * 1e6
    E_fetch = E_gpu_full_uJ_ref * (k * B / n_ref)
    E_total = E_sel + E_fetch
    if E_total < E_gpu_full(n_test):
        n_cross_half = n_test
        break

if isinstance(n_cross_base, int):
    shift_pct = abs(n_cross_half - n_cross_base) / n_cross_base * 100
    print(f"\nHeater halving effect: {n_cross_base/1000:.1f}K -> {n_cross_half/1000:.1f}K ({shift_pct:.0f}% shift)")

# ─── Figure 4: Sensitivity tornado chart ─────────────────────
fig4, ax4 = plt.subplots(figsize=(3.35, 2.8))

d_tornado = 64
QPS_TEC = 1e5  # queries per second for TEC amortization

def find_crossover_vs_gpu_full(d, P_laser=P_laser_mW, P_adc=P_ADC_per_block_mW,
                                P_dac=P_DAC_per_ch_mW, TEC_W=1.0, IL_dB=0.0):
    """Find crossover context length vs GPU full scan with parameter overrides."""
    P_laser_eff = P_laser * 10**(IL_dB / 10)  # insertion loss increases laser need
    E_TEC_per_query_pJ = TEC_W / QPS_TEC * 1e12  # W / QPS → J → pJ
    for n_test in n_range:
        N = N_blocks(n_test)
        P_total_mW = (P_dac + P_MZM_per_ch_mW) * d + \
                     (P_PD_per_block_mW + P_adc) * N + \
                     P_laser_eff + P_driver_mW + P_topk_mW
        E_dynamic_pJ = P_total_mW * t_prism_ns + E_TEC_per_query_pJ
        E_dynamic_J = E_dynamic_pJ * 1e-12
        # Heater (TFLN: 0)
        n_mrr = N * d
        P_heater_W = n_mrr * P_per_MRR_uW * 1e-6
        E_heater_J = P_heater_W * t_prism_ns * 1e-9
        E_sel_uJ = (E_heater_J + E_dynamic_J) * 1e6
        E_fetch_uJ = E_gpu_full_uJ_ref * (k * B / n_ref)
        E_total_uJ = E_sel_uJ + E_fetch_uJ
        if E_total_uJ < E_gpu_full(n_test):
            return n_test
    return n_range[-1]  # never crosses within range

# Default crossover
n_cross_default = find_crossover_vs_gpu_full(d_tornado)

# Parameter sweeps: (name, low, high, kwargs_low, kwargs_high)
sweeps = [
    ("Laser power",     dict(P_laser=50),   dict(P_laser=500)),
    ("ADC power/block", dict(P_adc=0.02),   dict(P_adc=0.5)),
    ("DAC power/ch",    dict(P_dac=0.1),    dict(P_dac=2.0)),
    ("TEC power",       dict(TEC_W=0.5),    dict(TEC_W=5.0)),
    ("Insertion loss",  dict(IL_dB=0),      dict(IL_dB=6)),
]

names = []
low_vals = []
high_vals = []

for name, kw_low, kw_high in sweeps:
    n_low = find_crossover_vs_gpu_full(d_tornado, **kw_low)
    n_high = find_crossover_vs_gpu_full(d_tornado, **kw_high)
    names.append(name)
    # "optimistic" = lower crossover (better), "pessimistic" = higher crossover (worse)
    low_vals.append(min(n_low, n_high))
    high_vals.append(max(n_low, n_high))

# Sort by total bar width (largest sensitivity first)
widths = [h - l for l, h in zip(low_vals, high_vals)]
order = np.argsort(widths)[::-1]
names = [names[i] for i in order]
low_vals = [low_vals[i] for i in order]
high_vals = [high_vals[i] for i in order]

y_pos = np.arange(len(names))

# Horizontal bars
for i in range(len(names)):
    ax4.barh(i, high_vals[i] / 1000 - n_cross_default / 1000,
             left=n_cross_default / 1000,
             height=0.5, color=COLOR_QUATERNARY, alpha=0.7,
             label='Pessimistic' if i == 0 else '')
    ax4.barh(i, low_vals[i] / 1000 - n_cross_default / 1000,
             left=n_cross_default / 1000,
             height=0.5, color=COLOR_SECONDARY, alpha=0.7,
             label='Optimistic' if i == 0 else '')

# Center line
ax4.axvline(n_cross_default / 1000, color='black', linewidth=1.0, linestyle='-', zorder=3)

ax4.set_yticks(y_pos)
ax4.set_yticklabels(names)
ax4.set_xlabel('Crossover context length (K tokens)')
ax4.set_title(f'Sensitivity tornado ($d = {d_tornado}$, vs. GPU full scan)', fontsize=9)
ax4.legend(fontsize=7, loc='lower right')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_sensitivity_tornado.pdf"),
            bbox_inches='tight', dpi=300)
print("Saved fig_sensitivity_tornado.pdf")
