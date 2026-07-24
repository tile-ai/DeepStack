"""
dse_fixed_parallel_analysis.py — Analysis for fixed-parallel lat×bw DSE

Same layout as dse_latency_sensitivity.py:
  1×6 subplots (one per BW: 0.5, 0.75, 1.0, 1.25, 1.5, 2.0).
  X-axis = BS (4, 64, 1024), lines = latency multipliers.
  Normalized to bw=1x, lat=1x.
  Draws both STPS and UTPS.

Usage:
    python3 -m mosaic.dse_space.case_study_latency_vs_bandwidth.dse_fixed_parallel_analysis <run_dir>
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
})

TARGET_BS = [4, 64, 1024]
TARGET_BW = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def load_data(run_dir: str):
    data = {}
    for phase in ("decode", "prefill"):
        result_path = os.path.join(run_dir, f"lat_bw_{phase}_result.csv")
        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
            df = pd.read_csv(result_path)
            if len(df) > 0:
                data[phase] = df
                print(f"  Loaded {phase}: {len(df)} rows")
    return data


def get_metrics(phase):
    if phase == "decode":
        return [("scaled_stps_avg", "Scaled STPS"),
                ("scaled_utps_avg", "Scaled UTPS")]
    return [("scaled_stps", "Scaled STPS"),
            ("scaled_utps", "Scaled UTPS")]


def _save(fig, out_dir, name):
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {name}.{{png,pdf}}")


# ══════════════════════════════════════════════════════════════════════════
# Plot: 1×N subplots (one per BW), X=BS, lines=latency
# ══════════════════════════════════════════════════════════════════════════

def plot_norm_by_bw(df, phase, out_dir):
    """
    For each metric (STPS & UTPS):
      1×N subplots (one per BW in TARGET_BW).
      X-axis = BS (4, 64, 1024), evenly spaced.
      Lines  = latency multipliers (color).
      Y-axis = normalized metric vs bw=1x, lat=1x (unified baseline).
    """
    metrics = get_metrics(phase)
    baselines = sorted(df["baseline_noc"].unique())
    bs_vals = TARGET_BS
    bs_to_idx = {b: i for i, b in enumerate(bs_vals)}

    for metric, metric_label in metrics:
        for bl in baselines:
            bw_present = [bw for bw in TARGET_BW
                          if not df[(df["baseline_noc"] == bl) &
                                    (df["bw_multiplier"] == bw)].empty]
            if not bw_present:
                continue

            # Unified baseline: bw=1x, lat=1x
            base_df = df[(df["baseline_noc"] == bl) &
                         (df["bw_multiplier"] == 1.0) &
                         (df["latency_multiplier"] == 1.0) &
                         (df["bs"].isin(bs_vals))]
            base_perf = base_df.groupby("bs")[metric].max()
            if base_perf.empty:
                continue

            lat_vals = sorted(df[df["baseline_noc"] == bl]["latency_multiplier"].unique())
            cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(lat_vals)))

            n_bw = len(bw_present)
            # Paper size: ~textwidth for 1-col, compact height
            fig, axes = plt.subplots(1, n_bw,
                                     figsize=(1.35 * n_bw, 1.8),
                                     squeeze=False, sharey=True)
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.82,
                                wspace=0.08)

            for ci, bw in enumerate(bw_present):
                ax = axes[0, ci]
                sub = df[(df["baseline_noc"] == bl) & (df["bw_multiplier"] == bw) &
                         (df["bs"].isin(bs_vals))]
                if sub.empty:
                    continue

                for i, lat in enumerate(lat_vals):
                    sl = sub[sub["latency_multiplier"] == lat]
                    best = sl.groupby("bs")[metric].max().sort_index()
                    normed = (best / base_perf).dropna()
                    if normed.empty:
                        continue
                    x = [bs_to_idx[b] for b in normed.index if b in bs_to_idx]
                    y = [normed[b] for b in normed.index if b in bs_to_idx]
                    ax.plot(x, y, marker="o", label=f"{lat}×")

                ax.axhline(y=1.0, color="grey", linestyle=":", linewidth=0.5)
                ax.set_xticks(range(len(bs_vals)))
                ax.set_xticklabels([str(int(b)) for b in bs_vals])
                ax.set_xlim(-0.25, len(bs_vals) - 0.75)
                ax.set_xlabel("Batch Size")
                ax.set_title(f"BW={bw}×", pad=3)
                ax.grid(alpha=0.25)
                if ci == 0:
                    ax.set_ylabel(f"Norm. {metric_label}")
                if ci == n_bw - 1:
                    ax.legend(title="Lat", loc="center left",
                              bbox_to_anchor=(1.02, 0.5), frameon=False)

            fig.suptitle(
                f"{metric_label} — Latency Sensitivity (fixed parallel)  [{bl}]",
                fontsize=9, fontweight="bold", y=0.97)
            m_tag = "stps" if "stps" in metric else "utps"
            tag = bl.replace(".", "p")
            _save(fig, out_dir, f"{phase}_norm_{m_tag}_by_bw_{tag}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main(run_dir: str):
    print(f"=== Fixed-Parallel Analysis for {run_dir} ===")
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    data = load_data(run_dir)
    if not data:
        print("No result CSVs found. Exiting.")
        return

    for phase in ("decode", "prefill"):
        if phase not in data:
            print(f"\n  Skipping {phase} (no data)")
            continue

        print(f"\n--- {phase.upper()} ---")
        df = data[phase]
        df = df[df["bs"].isin(TARGET_BS)]

        print(f"  Baselines: {sorted(df['baseline_noc'].unique())}")
        print(f"  Latency mults: {sorted(df['latency_multiplier'].unique())}")
        print(f"  BW mults: {sorted(df['bw_multiplier'].unique())}")
        print(f"  BS values: {sorted(df['bs'].unique())}")
        print(f"  Parallel configs per BS:")
        for bs in TARGET_BS:
            sub = df[df["bs"] == bs]
            if not sub.empty:
                r = sub.iloc[0]
                print(f"    BS={bs}: TP={int(r.tp)} EP={int(r.ep)} DP={int(r.dp)} PP={int(r.pp)}")

        plot_norm_by_bw(df, phase, out_dir)

    print(f"\n=== Done. Results in {out_dir}/ ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m mosaic.dse_space.case_study_latency_vs_bandwidth.dse_fixed_parallel_analysis <run_dir>")
        sys.exit(1)
    main(sys.argv[1])
