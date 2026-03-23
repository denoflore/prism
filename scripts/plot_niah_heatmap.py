"""Generate NIAH accuracy heatmap for PRISM paper."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plot_config import OUT_DIR, COLOR_PRIMARY


def make_white_to_navy_cmap():
    """Sequential colormap from white to dark navy per Tier 2C."""
    return mcolors.LinearSegmentedColormap.from_list(
        'white_navy', ['#FFFFFF', COLOR_PRIMARY])


def main():
    results_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                                'mrr_niah_v2b.json')
    with open(results_path) as f:
        v2b = json.load(f)

    contexts = ['4096', '8192', '16384', '32768', '65536', '131072']
    ctx_labels = ['4K', '8K', '16K', '32K', '64K', '128K']

    configs = [
        ('full', 'Full Attn'),
        ('ideal_k8', 'Ideal k=8'),
        ('ideal_k16', 'Ideal k=16'),
        ('ideal_k32', 'Ideal k=32'),
        ('si_5bit_20pm_k8', '5b-20pm k=8'),
        ('si_5bit_20pm_k32', '5b-20pm k=32'),
        ('si_4bit_30pm_k8', '4b-30pm k=8'),
        ('si_4bit_30pm_k32', '4b-30pm k=32'),
    ]
    config_labels = [c[1] for c in configs]

    matrix = np.zeros((len(configs), len(contexts)))
    for ci, ctx in enumerate(contexts):
        for ri, (key, _) in enumerate(configs):
            matrix[ri, ci] = v2b['niah'][ctx].get(key, 0)

    cmap = make_white_to_navy_cmap()

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    im = ax.imshow(matrix, aspect='auto', cmap=cmap,
                   vmin=0, vmax=100, interpolation='nearest')
    ax.set_xticks(range(len(ctx_labels)))
    ax.set_xticklabels(ctx_labels)
    ax.set_yticks(range(len(config_labels)))
    ax.set_yticklabels(config_labels)
    ax.set_xlabel('Context Length')

    # Cell annotations: black on light, white on dark
    for ri in range(len(configs)):
        for ci in range(len(contexts)):
            val = matrix[ri, ci]
            txt_color = 'white' if val > 50 else 'black'
            ax.text(ci, ri, f'{val:.0f}', ha='center', va='center',
                    fontsize=6.5, color=txt_color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, label='Accuracy (%)')
    ax.set_title('NIAH Accuracy: MRR Block-Sparse vs Full Attention',
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_niah_heatmap.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
