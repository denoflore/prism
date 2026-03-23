"""Shared matplotlib configuration for PRISM paper figures — Tier 2C style."""

import os
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
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.3,
    'grid.color': '#E0E0E0',
    'legend.frameon': True,
    'legend.edgecolor': '#CCCCCC',
    'legend.framealpha': 0.9,
})

# Vivid color cycle
COLOR_PRIMARY   = '#2563EB'  # bright blue
COLOR_SECONDARY = '#F97316'  # orange
COLOR_TERTIARY  = '#16A34A'  # green
COLOR_GRAY      = '#6B7280'  # neutral gray
COLOR_QUATERNARY = '#DC2626' # red
COLOR_QUINARY   = '#7C3AED'  # purple
COLOR_THRESHOLD = '#888888'  # threshold gray

COLORS = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY,
          COLOR_GRAY, COLOR_QUATERNARY, COLOR_QUINARY]

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
