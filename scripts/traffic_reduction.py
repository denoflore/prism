"""Figure: Traffic Reduction vs Context Length."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plot_config import OUT_DIR, COLOR_PRIMARY, COLOR_THRESHOLD


def main():
    B = 128
    k = 32
    transfer_per_fetch = k * B  # 1024 tokens

    ctx_lengths = np.array([4096, 8192, 16384, 32768, 65536,
                            131072, 262144, 524288, 1048576])
    reduction = ctx_lengths / transfer_per_fetch

    ctx_labels_map = {
        4096:'4K', 8192:'8K', 16384:'16K', 32768:'32K', 65536:'64K',
        131072:'128K', 262144:'256K', 524288:'512K', 1048576:'1M'
    }

    fig, ax = plt.subplots(figsize=(3.35, 2.5))

    ax.plot(ctx_lengths, reduction, 'o-', color=COLOR_PRIMARY,
            markerfacecolor='white', markeredgewidth=1.0, zorder=5)

    ax.axhline(y=1, color=COLOR_THRESHOLD, linestyle='--', linewidth=0.8,
               alpha=0.7, label=r'Breakeven (1$\times$)')

    ax.axvspan(8192, 1200000, alpha=0.06, color=COLOR_PRIMARY,
               label='Recall scaling region')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Traffic Reduction Factor')
    ax.set_xticks(ctx_lengths)
    ax.set_xticklabels([ctx_labels_map[c] for c in ctx_lengths],
                       rotation=45, ha='right')

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(0.8, 600)

    ax.annotate(f'{reduction[0]:.1f}$\\times$',
                (ctx_lengths[0], reduction[0]),
                textcoords='offset points', xytext=(8, -12),
                fontsize=7, color='black')
    ax.annotate(f'{reduction[-1]:.0f}$\\times$',
                (ctx_lengths[-1], reduction[-1]),
                textcoords='offset points', xytext=(-35, 8),
                fontsize=7, color='black')

    ax.text(0.03, 0.95, r'$B=128,\; k=32$' + '\n' + r'Transfer $= kB = 4096$ tokens',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=7, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                      edgecolor='black', linewidth=0.4, alpha=0.9))

    ax.legend(loc='lower left', fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_traffic_reduction.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
