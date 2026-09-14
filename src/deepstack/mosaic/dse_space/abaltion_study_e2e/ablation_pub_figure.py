#!/usr/bin/env python3
"""
Generate publication-quality ablation study figure for ASPLOS/MICRO.
Option A: Grouped bar chart — normalized speedup with incremental % annotations.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ──────────────────────────────────────────────────────────────
# Global style — ACM / IEEE two-column paper
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset':   'dejavuserif',
    'font.size':          8.5,
    'axes.labelsize':     9.5,
    'axes.titlesize':     10,
    'xtick.labelsize':    7.5,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth':     0.6,
    'grid.linewidth':     0.4,
    'lines.linewidth':    0.8,
    'patch.linewidth':    0.4,
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
    'xtick.major.pad':    2,
    'ytick.major.pad':    2,
})

# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────
techniques = [
    'Baseline\n(AstraSim)',
    '+ Full\nParallel',
    '+ Flex. Parallel\nAcross Module',
    '+ Search\nOn-chip Arch.',
    '+ Comm-Comp\nOverlap',
    '+ Stacking\nDSE',
    '+ NoC\nDSE',
]

bs4_stps    = [177.1, 256.4, 256.4, 314.2, 340.5, 493.3, 494.1]
bs1024_stps = [5729, 21252, 24488, 31350, 38061, 51095, 54280]

n = len(techniques)

# Normalised speedup over baseline
bs4_norm    = [v / bs4_stps[0]    for v in bs4_stps]
bs1024_norm = [v / bs1024_stps[0] for v in bs1024_stps]

# Incremental % change
def delta_pct(vals):
    return [0.0] + [(vals[i] - vals[i-1]) / vals[i-1] * 100
                     for i in range(1, len(vals))]

bs4_delta    = delta_pct(bs4_stps)
bs1024_delta = delta_pct(bs1024_stps)

# ──────────────────────────────────────────────────────────────
# Colour palette (good contrast in colour AND grayscale)
# ──────────────────────────────────────────────────────────────
C_BS4    = '#4C72B0'   # steel blue
C_BS1024 = '#C44E52'   # muted red
EDGE     = 'white'

# ──────────────────────────────────────────────────────────────
# Figure – double-column width (≈ 7 in)
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 3.2))

x = np.arange(n)
w = 0.33

bars4    = ax.bar(x - w/2, bs4_norm,    w, color=C_BS4,    edgecolor=EDGE,
                  linewidth=0.5, label='Batch Size = 4',    zorder=3)
bars1024 = ax.bar(x + w/2, bs1024_norm, w, color=C_BS1024, edgecolor=EDGE,
                  linewidth=0.5, label='Batch Size = 1024', zorder=3)

# ── Helper: format incremental annotation ────────────────────
def fmt_delta(d, is_baseline=False, is_zero=False):
    if is_baseline:
        return '1.0\u00d7'           # 1.0×
    if is_zero:
        return '\u2014'              # em-dash
    if abs(d) >= 100:
        return f'+{d:.0f}%'
    elif abs(d) >= 10:
        return f'+{d:.0f}%'
    elif abs(d) >= 1:
        return f'+{d:.1f}%'
    else:
        return f'+{d:.1f}%'

# ── Incremental % annotations ───────────────────────────────
ann_kw = dict(textcoords='offset points', ha='center', va='bottom',
              fontsize=6.0, fontweight='medium')

for i, (bar, d) in enumerate(zip(bars4, bs4_delta)):
    h = bar.get_height()
    txt = fmt_delta(d, is_baseline=(i == 0), is_zero=(d == 0 and i > 0))
    ax.annotate(txt, xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 2), color=C_BS4, **ann_kw)

for i, (bar, d) in enumerate(zip(bars1024, bs1024_delta)):
    h = bar.get_height()
    txt = fmt_delta(d, is_baseline=(i == 0))
    ax.annotate(txt, xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 2), color=C_BS1024, **ann_kw)

# ── Total speedup text (bold, above last bars, no arrow) ─────
for bars, norm, color in [
    (bars4,    bs4_norm,    C_BS4),
    (bars1024, bs1024_norm, C_BS1024),
]:
    last = bars[-1]
    ax.annotate(
        f'{norm[-1]:.1f}\u00d7 total',
        xy=(last.get_x() + last.get_width()/2, last.get_height()),
        xytext=(0, 12), textcoords='offset points',
        ha='center', va='bottom', fontsize=6.5, fontweight='bold',
        color=color,
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor=color, linewidth=0.5, alpha=0.9),
    )

# ── Axes styling ─────────────────────────────────────────────
ax.set_ylabel('Speedup over Baseline (\u00d7)', labelpad=4)
ax.set_xticks(x)
ax.set_xticklabels(techniques, linespacing=0.85)
ax.set_xlim(-0.55, n - 0.45)
ax.set_ylim(0, max(bs1024_norm) * 1.22)

# Horizontal grid
ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax.grid(axis='y', linestyle='--', alpha=0.30, zorder=0)
ax.set_axisbelow(True)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Baseline reference line at 1×
ax.axhline(y=1.0, color='#999999', linewidth=0.5, linestyle=':', zorder=1)

# Legend — upper left, out of the way
ax.legend(loc='upper left', frameon=True, fancybox=False,
          edgecolor='#cccccc', framealpha=0.95, borderpad=0.4,
          handlelength=1.2, handletextpad=0.4)

plt.tight_layout(pad=0.3)

# ── Save ─────────────────────────────────────────────────────
outdir = os.path.dirname(os.path.abspath(__file__))
for fmt in ('pdf', 'png'):
    path = os.path.join(outdir, f'ablation_figure.{fmt}')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')

plt.close()
