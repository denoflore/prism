"""Figure: R_h vs Context Length (retrieval head fraction scaling)."""

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_config import (OUT_DIR, COLOR_PRIMARY, COLOR_QUATERNARY,
                         COLOR_QUINARY, COLOR_THRESHOLD)


def main():
    # Qwen2.5-7B bf16
    bf16_ctx   = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    bf16_pct   = [83.9, 83.0, 91.1, 92.9,  95.5,  92.0,  98.2,   99.1]
    bf16_mean  = [0.574, 0.560, 0.627, 0.639, 0.656, 0.633, 0.796, 0.936]

    # Qwen3-8B bf16
    q3_ctx  = [2048, 4096, 8192]
    q3_pct  = [86.5, 88.2, 89.6]

    # Qwen2.5-7B 4-bit
    q4_ctx  = [2048, 4096, 8192]
    q4_pct  = [91.1, 90.2, 92.0]

    ctx_labels = {2048:'2K', 4096:'4K', 8192:'8K', 16384:'16K', 32768:'32K',
                  65536:'64K', 131072:'128K', 262144:'256K'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 5.0),
                                    gridspec_kw={'height_ratios': [3, 2]})

    # Panel (a): R_h percentage
    ax1.plot(bf16_ctx, bf16_pct, 'o-', color=COLOR_PRIMARY,
             label='Qwen2.5-7B (bf16)',
             markerfacecolor='white', markeredgewidth=1.0, zorder=5)
    ax1.plot(q3_ctx, q3_pct, 's--', color=COLOR_QUATERNARY,
             label='Qwen3-8B (bf16)',
             markerfacecolor='white', markeredgewidth=1.0, zorder=5)
    ax1.plot(q4_ctx, q4_pct, '^:', color=COLOR_QUINARY,
             label='Qwen2.5-7B (4-bit)',
             markerfacecolor='white', markeredgewidth=1.0, zorder=5)

    ax1.axhline(y=90, color=COLOR_THRESHOLD, linestyle='--', linewidth=0.8,
                alpha=0.7, label='90% threshold')

    ax1.set_xscale('log', base=2)
    ax1.set_xlim(1400, 380000)
    ax1.set_ylim(78, 101)
    ax1.set_ylabel(r'Retrieval Head Fraction $R_h$ (%)')
    ax1.set_xlabel('Context Length (tokens)')
    ax1.set_xticks(bf16_ctx)
    ax1.set_xticklabels([ctx_labels[c] for c in bf16_ctx], rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes,
             fontsize=10, fontweight='bold')

    # Panel (b): Mean R_h
    ax2.plot(bf16_ctx, bf16_mean, 'o-', color=COLOR_PRIMARY,
             label='Qwen2.5-7B (bf16)',
             markerfacecolor='white', markeredgewidth=1.0, zorder=5)

    ax2.set_xscale('log', base=2)
    ax2.set_xlim(1400, 380000)
    ax2.set_ylim(0.45, 1.0)
    ax2.set_ylabel(r'Mean $\overline{R}_h$')
    ax2.set_xlabel('Context Length (tokens)')
    ax2.set_xticks(bf16_ctx)
    ax2.set_xticklabels([ctx_labels[c] for c in bf16_ctx], rotation=45, ha='right')
    ax2.legend(loc='upper left')
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes,
             fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_rh_context_scaling.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
