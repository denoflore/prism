"""Figure: R_h Heatmap (bf16, 8K context)."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plot_config import OUT_DIR, COLOR_PRIMARY, COLOR_THRESHOLD


def make_white_to_navy_cmap():
    """Sequential colormap from white to dark navy per Tier 2C."""
    return mcolors.LinearSegmentedColormap.from_list(
        'white_navy', ['#FFFFFF', COLOR_PRIMARY])


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                             'PRISM_phase1_results.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    mat = np.array(data['qwen25_7b_rh']['8192']['rh_matrix'])  # (28, 4)
    num_layers, num_kv_heads = mat.shape

    cmap = make_white_to_navy_cmap()

    fig, ax = plt.subplots(figsize=(3.35, 6.0))

    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   interpolation='nearest', origin='lower')

    ax.set_xlabel('KV Head Index')
    ax.set_ylabel('Layer Index')
    ax.set_xticks(range(num_kv_heads))
    ax.set_xticklabels([str(i) for i in range(num_kv_heads)])

    layer_ticks = list(range(0, num_layers, 2))
    ax.set_yticks(layer_ticks)
    ax.set_yticklabels([str(i) for i in layer_ticks])

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.08)
    cbar.set_label(r'Retrieval Ratio $R_h$')
    # Mark threshold on colorbar
    cbar.ax.axhline(y=0.3, color=COLOR_THRESHOLD, linestyle='--', linewidth=0.8)

    # Cell annotations: black on light, white on dark
    for i in range(num_layers):
        for j in range(num_kv_heads):
            val = mat[i, j]
            txt_color = 'white' if val >= 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=5, color=txt_color, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_rh_heatmap.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
