"""
plot_stps_torus_mesh_switch.py — Publication-quality STPS plot for torus_mesh_switch

1×6 subplots (one per BW multiplier: 0.5×, 0.75×, 1×, 1.25×, 1.5×, 2×).
X-axis = BS (4, 64, 1024), evenly spaced.
Lines  = Latency multipliers (color-coded).
Y-axis = STPS normalized to (bw=1×, lat=1×) per BS.

Usage:
    python3 -m mosaic.dse_space.case_study_latency_vs_bandwidth.plot_stps_torus_mesh_switch <run_dir>
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── RC params: conference-quality (ISCA / MICRO / ASPLOS style) ──────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "CMU Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Config ────────────────────────────────────────────────────────────────
BASELINE = "torus_mesh_switch_1"
TARGET_BS = [4, 64, 1024]
TARGET_BW = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
TARGET_LAT = [0.25, 0.5, 1.0, 2.0, 4.0]

# Color palette: viridis (consistent with dse_fixed_parallel_analysis.py)
_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, len(TARGET_LAT)))
LAT_COLORS = {lat: _viridis[i] for i, lat in enumerate(TARGET_LAT)}
LAT_MARKERS = {
    0.25: "D",
    0.5:  "s",
    1.0:  "o",
    2.0:  "^",
    4.0:  "v",
}
LAT_LABELS = {
    0.25: "0.25×",
    0.5:  "0.5×",
    1.0:  "1×",
    2.0:  "2×",
    4.0:  "4×",
}

BW_TITLES = {
    0.5:  "BW = 0.5×",
    0.75: "BW = 0.75×",
    1.0:  "BW = 1×",
    1.25: "BW = 1.25×",
    1.5:  "BW = 1.5×",
    2.0:  "BW = 2×",
}

BS_TICK_LABELS = ["4", "64", "1024"]


def load_and_filter(run_dir):
    csv_path = os.path.join(run_dir, "lat_bw_decode_result.csv")
    df = pd.read_csv(csv_path)
    df = df[df["baseline_noc"] == BASELINE].copy()
    # Best STPS across tp_transform_moe modes per config
    idx_cols = ["latency_multiplier", "bw_multiplier", "bs"]
    df = df.loc[df.groupby(idx_cols)["scaled_stps_avg"].idxmax()].copy()
    return df


def plot_stps(df, out_dir):
    n_bw = len(TARGET_BW)
    fig, axes = plt.subplots(
        1, n_bw,
        figsize=(7.2, 1.7),
        sharey=True,
        constrained_layout=True,
    )

    bs_to_x = {bs: i for i, bs in enumerate(TARGET_BS)}

    # Per-BS baseline (bw=1x, lat=1x)
    base_stps = {}
    for bs in TARGET_BS:
        row = df[(df["bs"] == bs) &
                 (df["bw_multiplier"] == 1.0) &
                 (df["latency_multiplier"] == 1.0)]
        if not row.empty:
            base_stps[bs] = row["scaled_stps_avg"].values[0]

    for col, bw in enumerate(TARGET_BW):
        ax = axes[col]
        sub = df[df["bw_multiplier"] == bw]

        for lat in TARGET_LAT:
            sl = sub[sub["latency_multiplier"] == lat].sort_values("bs")
            if sl.empty:
                continue
            xs = [bs_to_x[b] for b in sl["bs"].values if b in bs_to_x]
            ys = [sl[sl["bs"] == b]["scaled_stps_avg"].values[0] / base_stps[b]
                  for b in sl["bs"].values if b in bs_to_x and b in base_stps]

            is_baseline = (lat == 1.0)
            ax.plot(
                xs, ys,
                color=LAT_COLORS[lat],
                marker=LAT_MARKERS[lat],
                markeredgecolor="white",
                markeredgewidth=0.3,
                markersize=3.5 if is_baseline else 3,
                label=LAT_LABELS[lat],
                zorder=3 if is_baseline else 2,
                linewidth=1.2 if is_baseline else 0.9,
            )

        # Baseline reference
        ax.axhline(y=1.0, color="#aaaaaa", linestyle="--", linewidth=0.5, zorder=1)

        # Formatting
        ax.set_xticks(range(len(TARGET_BS)))
        ax.set_xticklabels(BS_TICK_LABELS)
        ax.set_xlim(-0.3, len(TARGET_BS) - 0.7)
        ax.set_xlabel("Batch Size", labelpad=2)
        ax.set_title(BW_TITLES[bw], fontweight="bold", pad=3, fontsize=7.5)
        ax.grid(True, which="major", axis="y", linestyle="-", alpha=0.15)

        if col == 0:
            ax.set_ylabel("Normalized STPS", labelpad=3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


    # ── Shared legend at top (single row, title inline) ─────────────────
    # Invisible handle for "Latency:" label so it sits on the same line
    title_handle = Line2D([], [], color="none", marker="None", linestyle="None",
                          label="Latency Multiplier:")
    handles = [title_handle] + [
        Line2D([0], [0], color=LAT_COLORS[lat], marker=LAT_MARKERS[lat],
               markeredgecolor="white", markeredgewidth=0.3,
               markersize=4, linewidth=0.9, label=LAT_LABELS[lat])
        for lat in TARGET_LAT
    ]
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        borderpad=0.08,
        columnspacing=0.8,
        handletextpad=0.25,
        handlelength=1.2,
        # bbox_to_anchor=(0.5, 0.955),
        bbox_to_anchor=(0.5, 0.955),
    )
    # Bold the "Latency:" text
    leg.get_texts()[0].set_fontweight("bold")

    fig.get_layout_engine().set(rect=[0, 0, 1, 0.84])

    # ── Save ──────────────────────────────────────────────────────────────
    for fmt in ("pdf", "png"):
        path = os.path.join(out_dir, f"stps_lat_vs_bw_torus_mesh_switch.{fmt}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def main(run_dir):
    print(f"=== STPS Plot (torus_mesh_switch) for {run_dir} ===")
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    df = load_and_filter(run_dir)
    print(f"  Filtered rows: {len(df)}")
    plot_stps(df, out_dir)
    print("=== Done ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m mosaic.dse_space.case_study_latency_vs_bandwidth.plot_stps_torus_mesh_switch <run_dir>")
        sys.exit(1)
    main(sys.argv[1])
