"""
dse_latency_sensitivity.py — STPS sensitivity to latency at different batch sizes

Plot A: 1×3 subplots (bw=0.5x, 1x, 2x).
        X-axis = BS (4, 64, 1024), lines = latency multipliers.
        Shows normalized STPS (vs lat=1x at same bw).

Plot B: 1×3 subplots (one per BS=4, 64, 1024), bw=1x.
        Grouped bar chart, X-axis = latency multipliers,
        Bars = TP / EP / DP / PP (log-y), showing best parallel config.

Usage:
    python3 -m mosaic.dse_space.case_study_latency_vs_bandwidth.dse_latency_sensitivity <run_dir>
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

TARGET_BS = [4, 64, 1024]
TARGET_BW = [0.5, 1.0, 2.0]

PAR_COLORS = {"TP": "#2196F3", "EP": "#4CAF50", "DP": "#FF9800", "PP": "#D32F2F"}


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
    """Return list of (col_name, display_label) for both STPS and UTPS."""
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
# Plot A: Normalized STPS — 1×3 subplots for bw=0.5x, 1x, 2x
# ══════════════════════════════════════════════════════════════════════════

def plot_norm_by_bw(df, phase, out_dir):
    """
    For each metric (STPS & UTPS):
      1×3 subplots (one per bw in 0.5x, 1x, 2x).
      X-axis = BS (4, 64, 1024), evenly spaced.
      Lines  = latency multipliers (color).
      Y-axis = normalized metric vs **bw=1x, lat=1x** (unified baseline).
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

            fig, axes = plt.subplots(1, len(bw_present),
                                     figsize=(5.5 * len(bw_present), 5),
                                     squeeze=False, sharey=True)

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
                    ax.plot(x, y, marker="o", markersize=7, linewidth=2,
                            color=cmap[i], label=f"lat={lat}x")

                ax.axhline(y=1.0, color="grey", linestyle=":", linewidth=0.8)
                ax.set_xticks(range(len(bs_vals)))
                ax.set_xticklabels([str(int(b)) for b in bs_vals])
                ax.set_xlim(-0.3, len(bs_vals) - 0.7)
                ax.set_xlabel("Batch Size (BS)")
                ax.set_title(f"BW = {bw}x")
                ax.grid(alpha=0.3)
                if ci == 0:
                    ax.set_ylabel(f"Normalized {metric_label}\n(vs bw=1x, lat=1x)")
                ax.legend(fontsize=7, title="Latency", title_fontsize=8)

            fig.suptitle(
                f"{phase.capitalize()} — {metric_label} Normalized Sensitivity  [{bl}]",
                fontsize=13, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            m_tag = "stps" if "stps" in metric else "utps"
            tag = f"{bl}".replace(".", "p")
            _save(fig, out_dir, f"{phase}_norm_{m_tag}_by_bw_{tag}")


# ══════════════════════════════════════════════════════════════════════════
# Plot B: Best parallel config bar chart — 1×3 subplots for BS=4, 64, 1024
# ══════════════════════════════════════════════════════════════════════════

def plot_best_parallel_bars(df, phase, out_dir):
    """
    1×3 subplots (one per BS in 4, 64, 1024), bw=1x.
    X-axis  = latency multipliers.
    Stacked bars = log2(TP) + log2(EP) + log2(DP) + log2(PP) = log2(total devices).
    All bars reach the same total height; the split shows how devices are allocated.
    """
    metric, _ = get_metrics(phase)[0]  # use STPS for best-config selection
    baselines = sorted(df["baseline_noc"].unique())
    par_keys = ["TP", "EP", "DP", "PP"]
    par_cols = ["tp", "ep", "dp", "pp"]

    for bl in baselines:
        sub_all = df[(df["baseline_noc"] == bl) & (df["bw_multiplier"] == 1.0) &
                     (df["bs"].isin(TARGET_BS))]
        if sub_all.empty:
            continue

        lat_vals = sorted(sub_all["latency_multiplier"].unique())
        n_lat = len(lat_vals)
        bar_w = 0.55

        fig, axes = plt.subplots(1, len(TARGET_BS),
                                 figsize=(5.5 * len(TARGET_BS), 5),
                                 squeeze=False, sharey=True)

        for bi, bs in enumerate(TARGET_BS):
            ax = axes[0, bi]
            sub = sub_all[sub_all["bs"] == bs]
            if sub.empty:
                ax.set_visible(False)
                continue

            x = np.arange(n_lat)

            # Collect best config per latency
            configs = {}  # lat -> {tp, ep, dp, pp}
            for lat in lat_vals:
                sl = sub[sub["latency_multiplier"] == lat]
                if sl.empty:
                    configs[lat] = {pc: 1 for pc in par_cols}
                    continue
                best_idx = sl[metric].idxmax()
                configs[lat] = {pc: int(sl.loc[best_idx, pc]) for pc in par_cols}

            # Stacked bar: bottom accumulates log2 values
            bottom = np.zeros(n_lat)
            for pk, pc in zip(par_keys, par_cols):
                log_vals = np.array([np.log2(max(configs[lat][pc], 1))
                                     for lat in lat_vals])
                bars = ax.bar(x, log_vals, width=bar_w, bottom=bottom,
                              color=PAR_COLORS[pk], label=pk,
                              edgecolor="white", linewidth=0.5)
                # Label with actual value in the middle of each segment
                for j, (rect, lv) in enumerate(zip(bars, log_vals)):
                    if lv >= 0.8:  # only label if segment is tall enough
                        ax.text(rect.get_x() + rect.get_width() / 2,
                                bottom[j] + lv / 2,
                                str(configs[lat_vals[j]][pc]),
                                ha="center", va="center", fontsize=8,
                                fontweight="bold", color="white")
                bottom += log_vals

            ax.set_xticks(x)
            ax.set_xticklabels([f"{l}x" for l in lat_vals])
            ax.set_xlabel("Latency Multiplier")
            ax.set_title(f"BS = {bs}")
            ax.set_ylim(0, None)
            # Y-axis ticks as 2^n
            max_y = int(np.ceil(bottom.max()))
            ax.set_yticks(range(max_y + 1))
            ax.set_yticklabels([str(2**i) for i in range(max_y + 1)])
            ax.grid(axis="y", alpha=0.3)
            if bi == 0:
                ax.set_ylabel("Device Allocation (log₂ scale)")
                ax.legend(fontsize=7, ncol=2, loc="upper right")

        fig.suptitle(
            f"{phase.capitalize()} — Best Parallel Config at BW=1x  [{bl}]",
            fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        tag = bl.replace(".", "p")
        _save(fig, out_dir, f"{phase}_best_parallel_bars_{tag}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main(run_dir: str):
    print(f"=== Latency Sensitivity vs BS — Analysis for {run_dir} ===")
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

        # Filter to target BS only
        df = df[df["bs"].isin(TARGET_BS)]
        print(f"  Baselines: {sorted(df['baseline_noc'].unique())}")
        print(f"  Latency mults: {sorted(df['latency_multiplier'].unique())}")
        print(f"  BW mults: {sorted(df['bw_multiplier'].unique())}")
        print(f"  BS values: {sorted(df['bs'].unique())}")

        plot_norm_by_bw(df, phase, out_dir)
        plot_best_parallel_bars(df, phase, out_dir)

    print(f"\n=== Done. Results in {out_dir}/ ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m mosaic.dse_space.case_study_latency_vs_bandwidth.dse_latency_sensitivity <run_dir>")
        sys.exit(1)
    main(sys.argv[1])
