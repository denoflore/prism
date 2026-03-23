"""Figure: Recall vs Block Size / Signature Dimension."""

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_config import (OUT_DIR, COLOR_PRIMARY, COLOR_SECONDARY,
                         COLOR_TERTIARY, COLOR_THRESHOLD)


def main():
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    block_sizes = [64, 128, 256]
    recall_data = {
        64:  {'R@2': 0.10, 'R@4': 0.21, 'R@8': 0.38},
        128: {'R@2': 0.16, 'R@4': 0.31, 'R@8': 0.52},
        256: {'R@2': 0.3125, 'R@4': 0.50, 'R@8': 0.7734},
    }

    ks = ['R@2', 'R@4', 'R@8']
    panel_labels = ['(a) Recall@2', '(b) Recall@4', '(c) Recall@8']
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=1.0)

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.0))

    for i, (k, label) in enumerate(zip(ks, panel_labels)):
        ax = axes[i]
        vals = [recall_data[b][k] for b in block_sizes]
        bar_colors = [cmap(norm(v)) for v in vals]
        x = np.arange(len(block_sizes))

        bars = ax.bar(x, vals, 0.65, color=bar_colors,
                      edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

        ax.set_ylabel('Recall')
        ax.set_xticks(x)
        ax.set_xticklabels([f'B={b}' for b in block_sizes])
        ax.set_ylim(0, 1.0)
        ax.set_title(label, fontsize=9, fontweight='bold', loc='left')

    axes[-1].set_xlabel('Block Size $B$')

    fig.tight_layout(h_pad=1.5)

    # Shared colorbar in dedicated axes at very bottom
    cbar_ax = fig.add_axes([0.15, 0.02, 0.55, 0.015])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Recall', fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=6)

    fig.text(0.95, 0.02, r'$d_{\mathrm{sig}}=64$, ctx=4K',
             ha='right', va='bottom', fontsize=7, color='gray')

    fig.subplots_adjust(bottom=0.09)
    path = os.path.join(OUT_DIR, 'fig_recall_comparison.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
