"""
dse_analysis.py — Post-DSE analysis for "NoC Latency vs Bandwidth"

Usage:
    python3 -m mosaic.dse_space.case_study_latency_vs_bandwidth.dse_analysis <run_dir>

Reads:
    <run_dir>/lat_bw_decode_result.csv
    <run_dir>/lat_bw_prefill_result.csv
    <run_dir>/lat_bw_decode_invalid.csv   (optional)
    <run_dir>/lat_bw_prefill_invalid.csv  (optional)

Outputs plots & summary CSV into <run_dir>/analysis/
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

COLORS = {
    "primary": "#2196F3",
    "secondary": "#4CAF50",
    "accent": "#FF9800",
    "red": "#D32F2F",
    "purple": "#7B1FA2",
    "grey": "#9E9E9E",
    "light": "#E8EAF6",
    "teal": "#009688",
}

BASELINE_MARKERS = {"torus_mesh_switch_1": "o", "torus_mesh_mesh_3": "s"}


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_data(run_dir: str):
    data = {}
    for phase in ("decode", "prefill"):
        result_path = os.path.join(run_dir, f"lat_bw_{phase}_result.csv")
        invalid_path = os.path.join(run_dir, f"lat_bw_{phase}_invalid.csv")
        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
            df = pd.read_csv(result_path)
            if len(df) > 0:
                data[phase] = df
                print(f"  Loaded {phase}: {len(df)} rows")
        if os.path.exists(invalid_path) and os.path.getsize(invalid_path) > 0:
            df_inv = pd.read_csv(invalid_path)
            if len(df_inv) > 0:
                data[f"{phase}_invalid"] = df_inv
                print(f"  Loaded {phase}_invalid: {len(df_inv)} rows")
    return data


def get_metric(phase):
    if phase == "decode":
        return "scaled_stps_avg", "Scaled STPS (decode)"
    return "scaled_stps", "Scaled STPS (prefill)"


# ══════════════════════════════════════════════════════════════════════════
# Plot 1: 2D Heatmap — lat_mult × bw_mult → best perf
# ══════════════════════════════════════════════════════════════════════════

def plot_lat_bw_heatmap(df, phase, out_dir):
    """Heatmap: rows=latency_mult, cols=bw_mult, value=best perf across workloads."""
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    for bl in baselines:
        sub = df[df["baseline_noc"] == bl]
        pivot = sub.groupby(["latency_multiplier", "bw_multiplier"])[metric].max().unstack(fill_value=0)
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot.index) * 1.0)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x}x" for x in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y}x" for y in pivot.index], fontsize=9)
        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("Latency Multiplier")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if val > pivot.values.max() * 0.6 else "black")

        fig.colorbar(im, ax=ax, label=metric_label, shrink=0.8)
        fig.suptitle(f"{phase.capitalize()} — Best {metric_label} [{bl}]",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_lat_bw_heatmap_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 2: Per-workload heatmaps
# ══════════════════════════════════════════════════════════════════════════

def plot_lat_bw_heatmap_per_workload(df, phase, out_dir):
    """Grid of heatmaps: one per workload (bs, seq)."""
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    for bl in baselines:
        sub = df[df["baseline_noc"] == bl]
        workloads = sub.groupby(["bs", "seq"]).size().reset_index()[["bs", "seq"]]
        n_wl = len(workloads)
        ncols = min(4, n_wl)
        nrows = (n_wl + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

        lat_vals = sorted(sub["latency_multiplier"].unique())
        bw_vals = sorted(sub["bw_multiplier"].unique())

        for idx, (_, wl) in enumerate(workloads.iterrows()):
            bs, seq = wl["bs"], wl["seq"]
            ax = axes[idx // ncols, idx % ncols]
            wl_sub = sub[(sub["bs"] == bs) & (sub["seq"] == seq)]
            pivot = wl_sub.groupby(["latency_multiplier", "bw_multiplier"])[metric].max().unstack(fill_value=0)

            if pivot.empty:
                ax.set_visible(False)
                continue

            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                           interpolation="nearest", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{x}x" for x in pivot.columns], fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{y}x" for y in pivot.index], fontsize=7)
            ax.set_title(f"BS={bs}, SEQ={seq}", fontsize=9)
            ax.set_xlabel("BW mult")
            ax.set_ylabel("Lat mult")

        for idx in range(n_wl, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle(f"{phase.capitalize()} — Lat×BW Heatmaps [{bl}]",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, out_dir, f"{phase}_lat_bw_heatmap_workloads_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 3: BW sensitivity at fixed latencies
# ══════════════════════════════════════════════════════════════════════════

def plot_bw_at_fixed_latency(df, phase, out_dir):
    """For each latency_mult, plot perf vs bw_mult."""
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    bs_vals = sorted(df["bs"].unique())
    seq_vals = sorted(df["seq"].unique())
    mid_bs = bs_vals[len(bs_vals) // 2]
    mid_seq = seq_vals[len(seq_vals) // 2]

    for bl in baselines:
        sub = df[(df["baseline_noc"] == bl) & (df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        lat_vals = sorted(sub["latency_multiplier"].unique())
        cmap = plt.cm.viridis(np.linspace(0, 1, len(lat_vals)))

        for i, lat in enumerate(lat_vals):
            sl = sub[sub["latency_multiplier"] == lat]
            best = sl.groupby("bw_multiplier")[metric].max().sort_index()
            ax.plot(best.index, best.values, marker="o", markersize=4,
                    linewidth=1.5, color=cmap[i], label=f"lat={lat}x")

        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel(metric_label)
        ax.set_title(f"[{bl}] BS={mid_bs}, SEQ={mid_seq}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)

        fig.suptitle(f"{phase.capitalize()} — BW Sensitivity at Fixed Latency",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_bw_at_fixed_lat_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 4: Latency sensitivity at fixed BWs
# ══════════════════════════════════════════════════════════════════════════

def plot_lat_at_fixed_bw(df, phase, out_dir):
    """For each bw_mult, plot perf vs latency_mult."""
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    bs_vals = sorted(df["bs"].unique())
    seq_vals = sorted(df["seq"].unique())
    mid_bs = bs_vals[len(bs_vals) // 2]
    mid_seq = seq_vals[len(seq_vals) // 2]

    for bl in baselines:
        sub = df[(df["baseline_noc"] == bl) & (df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        bw_vals = sorted(sub["bw_multiplier"].unique())
        cmap = plt.cm.plasma(np.linspace(0, 1, len(bw_vals)))

        for i, bw in enumerate(bw_vals):
            sl = sub[sub["bw_multiplier"] == bw]
            best = sl.groupby("latency_multiplier")[metric].max().sort_index()
            ax.plot(best.index, best.values, marker="s", markersize=4,
                    linewidth=1.5, color=cmap[i], label=f"bw={bw}x")

        ax.set_xlabel("Latency Multiplier")
        ax.set_ylabel(metric_label)
        ax.set_title(f"[{bl}] BS={mid_bs}, SEQ={mid_seq}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.invert_xaxis()  # Lower latency = better, on left

        fig.suptitle(f"{phase.capitalize()} — Latency Sensitivity at Fixed BW",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_lat_at_fixed_bw_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 5: Normalized sensitivity — lat vs bw elasticity
# ══════════════════════════════════════════════════════════════════════════

def plot_elasticity(df, phase, out_dir):
    """Compare: doubling BW vs halving latency, which gives more perf?"""
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    fig, axes = plt.subplots(1, len(baselines), figsize=(6 * len(baselines), 5), squeeze=False)

    for bi, bl in enumerate(baselines):
        ax = axes[0, bi]
        sub = df[df["baseline_noc"] == bl]
        workloads = sub.groupby(["bs", "seq"]).size().reset_index()[["bs", "seq"]]

        # BW sensitivity: fix lat=1, vary bw
        bw_norms = []
        for _, wl in workloads.iterrows():
            sl = sub[(sub["bs"] == wl["bs"]) & (sub["seq"] == wl["seq"]) &
                     (sub["latency_multiplier"] == 1.0)]
            best = sl.groupby("bw_multiplier")[metric].max().sort_index()
            base = best.get(1.0, None)
            if base and base > 0:
                bw_norms.append(best / base)

        # Lat sensitivity: fix bw=1, vary lat
        lat_norms = []
        for _, wl in workloads.iterrows():
            sl = sub[(sub["bs"] == wl["bs"]) & (sub["seq"] == wl["seq"]) &
                     (sub["bw_multiplier"] == 1.0)]
            best = sl.groupby("latency_multiplier")[metric].max().sort_index()
            base = best.get(1.0, None)
            if base and base > 0:
                lat_norms.append(best / base)

        if bw_norms:
            bw_avg = pd.concat(bw_norms, axis=1).mean(axis=1)
            ax.plot(bw_avg.index, bw_avg.values, marker="o", markersize=5,
                    linewidth=2, color=COLORS["primary"], label="BW scaling (lat=1x)")

        if lat_norms:
            lat_avg = pd.concat(lat_norms, axis=1).mean(axis=1)
            # Invert lat axis: 0.25x lat = 4x "improvement"
            ax.plot(1.0 / lat_avg.index, lat_avg.values, marker="s", markersize=5,
                    linewidth=2, color=COLORS["red"], label="Lat scaling (bw=1x, inverted)")

        ax.axhline(y=1.0, color=COLORS["grey"], linestyle=":", linewidth=0.8)
        ax.set_xlabel("Improvement Factor (2x = double BW or halve latency)")
        ax.set_ylabel("Avg Normalized Perf")
        ax.set_title(f"[{bl}]", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)

    fig.suptitle(f"{phase.capitalize()} — BW vs Latency Elasticity (avg across workloads)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, f"{phase}_elasticity")


# ══════════════════════════════════════════════════════════════════════════
# Plot 6: Per-workload elasticity (scatter: BW gain vs lat gain)
# ══════════════════════════════════════════════════════════════════════════

def plot_per_workload_elasticity(df, phase, out_dir):
    """Scatter: x=gain from 2x BW, y=gain from 0.5x lat, per workload."""
    metric, _ = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    for bl in baselines:
        sub = df[df["baseline_noc"] == bl]
        workloads = sub.groupby(["bs", "seq"]).size().reset_index()[["bs", "seq"]]

        bw_gains = []
        lat_gains = []
        labels = []

        for _, wl in workloads.iterrows():
            bs, seq = wl["bs"], wl["seq"]
            # BW gain: 2x bw vs 1x bw (at lat=1)
            base_bw = sub[(sub["bs"] == bs) & (sub["seq"] == seq) &
                          (sub["latency_multiplier"] == 1.0) & (sub["bw_multiplier"] == 1.0)]
            up_bw = sub[(sub["bs"] == bs) & (sub["seq"] == seq) &
                        (sub["latency_multiplier"] == 1.0) & (sub["bw_multiplier"] == 2.0)]
            # Lat gain: 0.5x lat vs 1x lat (at bw=1)
            base_lat = base_bw  # same baseline
            down_lat = sub[(sub["bs"] == bs) & (sub["seq"] == seq) &
                           (sub["latency_multiplier"] == 0.5) & (sub["bw_multiplier"] == 1.0)]

            if not base_bw.empty and not up_bw.empty and not down_lat.empty:
                bv = base_bw[metric].max()
                if bv > 0:
                    bw_gains.append(up_bw[metric].max() / bv)
                    lat_gains.append(down_lat[metric].max() / bv)
                    labels.append(f"BS{bs}_S{seq}")

        if not bw_gains:
            continue

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(bw_gains, lat_gains, s=50, c=COLORS["primary"], edgecolor="black", linewidth=0.5, zorder=3)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (bw_gains[i], lat_gains[i]), fontsize=6, ha="left", va="bottom")

        # Diagonal: if above, latency matters more; below, BW matters more
        lim = max(max(bw_gains), max(lat_gains)) * 1.1
        ax.plot([0.8, lim], [0.8, lim], color=COLORS["grey"], linestyle="--", linewidth=0.8)
        ax.set_xlabel("Perf Gain from 2x BW (at lat=1x)")
        ax.set_ylabel("Perf Gain from 0.5x Latency (at bw=1x)")
        ax.set_title(f"[{bl}]")
        ax.grid(alpha=0.3)

        # Label regions
        ax.text(lim * 0.95, lim * 0.75, "BW > Lat", fontsize=9, ha="right", color=COLORS["primary"], alpha=0.6)
        ax.text(lim * 0.75, lim * 0.95, "Lat > BW", fontsize=9, ha="right", color=COLORS["red"], alpha=0.6)

        fig.suptitle(f"{phase.capitalize()} — Per-Workload: 2x BW vs 0.5x Latency [{bl}]",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_workload_elasticity_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 7: Power & area analysis
# ══════════════════════════════════════════════════════════════════════════

def plot_power_area(df, phase, out_dir):
    """Power and SM count across lat/bw configs."""
    metric, _ = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    bs_vals = sorted(df["bs"].unique())
    seq_vals = sorted(df["seq"].unique())
    mid_bs = bs_vals[len(bs_vals) // 2]
    mid_seq = seq_vals[len(seq_vals) // 2]

    for bl in baselines:
        sub = df[(df["baseline_noc"] == bl) & (df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
        if sub.empty:
            continue

        # Best perf per (lat, bw)
        best_idx = sub.groupby(["latency_multiplier", "bw_multiplier"])[metric].idxmax()
        best = sub.loc[best_idx]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # SM count heatmap
        ax = axes[0]
        pivot_sm = best.pivot_table(index="latency_multiplier", columns="bw_multiplier",
                                     values="sm_count", aggfunc="first")
        im = ax.imshow(pivot_sm.values, aspect="auto", cmap="Blues", interpolation="nearest", origin="lower")
        ax.set_xticks(range(len(pivot_sm.columns)))
        ax.set_xticklabels([f"{x}x" for x in pivot_sm.columns], fontsize=8)
        ax.set_yticks(range(len(pivot_sm.index)))
        ax.set_yticklabels([f"{y}x" for y in pivot_sm.index], fontsize=8)
        ax.set_xlabel("BW mult")
        ax.set_ylabel("Lat mult")
        ax.set_title("Feasible SM Count")
        for i in range(len(pivot_sm.index)):
            for j in range(len(pivot_sm.columns)):
                ax.text(j, i, f"{int(pivot_sm.values[i, j])}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)

        # Total power heatmap
        ax = axes[1]
        pivot_pw = best.pivot_table(index="latency_multiplier", columns="bw_multiplier",
                                     values="total_power_per_device_W", aggfunc="first")
        im = ax.imshow(pivot_pw.values, aspect="auto", cmap="OrRd", interpolation="nearest", origin="lower")
        ax.set_xticks(range(len(pivot_pw.columns)))
        ax.set_xticklabels([f"{x}x" for x in pivot_pw.columns], fontsize=8)
        ax.set_yticks(range(len(pivot_pw.index)))
        ax.set_yticklabels([f"{y}x" for y in pivot_pw.index], fontsize=8)
        ax.set_xlabel("BW mult")
        ax.set_ylabel("Lat mult")
        ax.set_title("Total Power/Device (W)")
        for i in range(len(pivot_pw.index)):
            for j in range(len(pivot_pw.columns)):
                ax.text(j, i, f"{pivot_pw.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)

        # Freq scale heatmap
        ax = axes[2]
        pivot_fs = best.pivot_table(index="latency_multiplier", columns="bw_multiplier",
                                     values="freq_scale_power", aggfunc="first")
        im = ax.imshow(pivot_fs.values, aspect="auto", cmap="RdYlGn", interpolation="nearest", origin="lower")
        ax.set_xticks(range(len(pivot_fs.columns)))
        ax.set_xticklabels([f"{x}x" for x in pivot_fs.columns], fontsize=8)
        ax.set_yticks(range(len(pivot_fs.index)))
        ax.set_yticklabels([f"{y}x" for y in pivot_fs.index], fontsize=8)
        ax.set_xlabel("BW mult")
        ax.set_ylabel("Lat mult")
        ax.set_title("Freq Scale (power)")
        for i in range(len(pivot_fs.index)):
            for j in range(len(pivot_fs.columns)):
                ax.text(j, i, f"{pivot_fs.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)

        fig.suptitle(f"{phase.capitalize()} — Power & Area [{bl}] (BS={mid_bs}, SEQ={mid_seq})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_power_area_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 8: Parallel scheme heatmap
# ══════════════════════════════════════════════════════════════════════════

def plot_parallel_scheme(df, phase, out_dir):
    """Winning parallel scheme per (lat_mult, bw_mult)."""
    metric, _ = get_metric(phase)
    par_cols = ["tp", "ep", "dp", "pp"]
    df = df.copy()
    df["par_key"] = df[par_cols].astype(str).agg("_".join, axis=1)

    baselines = sorted(df["baseline_noc"].unique())

    bs_vals = sorted(df["bs"].unique())
    seq_vals = sorted(df["seq"].unique())
    mid_bs = bs_vals[len(bs_vals) // 2]
    mid_seq = seq_vals[len(seq_vals) // 2]

    for bl in baselines:
        sub = df[(df["baseline_noc"] == bl) & (df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
        if sub.empty:
            continue

        best_idx = sub.groupby(["latency_multiplier", "bw_multiplier"])[metric].idxmax()
        best = sub.loc[best_idx]

        # Encode par_key as int for heatmap
        unique_pars = sorted(best["par_key"].unique())
        par_to_int = {p: i for i, p in enumerate(unique_pars)}
        best = best.copy()
        best["par_int"] = best["par_key"].map(par_to_int)

        pivot = best.pivot_table(index="latency_multiplier", columns="bw_multiplier",
                                  values="par_int", aggfunc="first")
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot.index) * 1.0)))
        cmap = plt.cm.Set3(np.linspace(0, 1, len(unique_pars)))
        from matplotlib.colors import ListedColormap, BoundaryNorm
        lc = ListedColormap(cmap[:len(unique_pars)])
        bounds = np.arange(-0.5, len(unique_pars), 1)
        norm = BoundaryNorm(bounds, lc.N)

        im = ax.imshow(pivot.values, aspect="auto", cmap=lc, norm=norm,
                       interpolation="nearest", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x}x" for x in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y}x" for y in pivot.index], fontsize=8)
        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("Latency Multiplier")

        # Text annotations
        par_pivot = best.pivot_table(index="latency_multiplier", columns="bw_multiplier",
                                      values="par_key", aggfunc="first")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                lat = pivot.index[i]
                bw = pivot.columns[j]
                if lat in par_pivot.index and bw in par_pivot.columns:
                    val = par_pivot.loc[lat, bw]
                    if pd.notna(val):
                        ax.text(j, i, str(val).replace("_", "\n"), ha="center", va="center", fontsize=5)

        fig.suptitle(f"{phase.capitalize()} — Winning Parallel Scheme [{bl}] (BS={mid_bs}, SEQ={mid_seq})",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_parallel_scheme_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 9: Cross-baseline comparison
# ══════════════════════════════════════════════════════════════════════════

def plot_cross_baseline(df, phase, out_dir):
    """Compare the two baselines: best perf at each (lat, bw) point."""
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())
    if len(baselines) < 2:
        return

    bs_vals = sorted(df["bs"].unique())
    seq_vals = sorted(df["seq"].unique())
    mid_bs = bs_vals[len(bs_vals) // 2]
    mid_seq = seq_vals[len(seq_vals) // 2]

    sub = df[(df["bs"] == mid_bs) & (df["seq"] == mid_seq)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # BW sweep at lat=1
    ax = axes[0]
    for bl in baselines:
        sl = sub[(sub["baseline_noc"] == bl) & (sub["latency_multiplier"] == 1.0)]
        best = sl.groupby("bw_multiplier")[metric].max().sort_index()
        marker = BASELINE_MARKERS.get(bl, "^")
        ax.plot(best.index, best.values, marker=marker, markersize=5,
                linewidth=2, label=bl.replace("torus_mesh_", ""))
    ax.set_xlabel("BW Multiplier")
    ax.set_ylabel(metric_label)
    ax.set_title("BW sweep (lat=1x)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xscale("log", base=2)

    # Lat sweep at bw=1
    ax = axes[1]
    for bl in baselines:
        sl = sub[(sub["baseline_noc"] == bl) & (sub["bw_multiplier"] == 1.0)]
        best = sl.groupby("latency_multiplier")[metric].max().sort_index()
        marker = BASELINE_MARKERS.get(bl, "^")
        ax.plot(best.index, best.values, marker=marker, markersize=5,
                linewidth=2, label=bl.replace("torus_mesh_", ""))
    ax.set_xlabel("Latency Multiplier")
    ax.set_ylabel(metric_label)
    ax.set_title("Lat sweep (bw=1x)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()

    fig.suptitle(f"{phase.capitalize()} — Cross-Baseline (BS={mid_bs}, SEQ={mid_seq})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, f"{phase}_cross_baseline")


# ══════════════════════════════════════════════════════════════════════════
# Plot 10: Time breakdown at selected (lat, bw) configs
# ══════════════════════════════════════════════════════════════════════════

def plot_time_breakdown(df, phase, out_dir):
    """Stacked bar: time breakdown for best configs along BW axis (lat=1x)."""
    metric, _ = get_metric(phase)

    bs_vals = sorted(df["bs"].unique())
    seq_vals = sorted(df["seq"].unique())
    mid_bs = bs_vals[len(bs_vals) // 2]
    mid_seq = seq_vals[len(seq_vals) // 2]
    baselines = sorted(df["baseline_noc"].unique())

    if phase == "decode":
        time_cols = ["time_dense_ffn_ms", "time_moe_ms", "time_rms_norm_ms",
                     "time_add_residual_ms", "pp_p2p_time_ms"]
        if "time_mla_2ms" in df.columns:
            time_cols = ["time_mla_2ms"] + time_cols
    else:
        time_cols = [c for c in ["time_mla_ms", "time_gqa_ms", "time_dense_ffn_ms",
                                  "time_moe_ms", "time_rms_norm_ms",
                                  "time_add_residual_ms", "pp_p2p_time_ms"]
                     if c in df.columns]

    labels = [c.replace("time_", "").replace("_ms", "").replace("_2ms", "") for c in time_cols]

    for bl in baselines:
        sub = df[(df["baseline_noc"] == bl) & (df["bs"] == mid_bs) &
                 (df["seq"] == mid_seq) & (df["latency_multiplier"] == 1.0)]
        if sub.empty:
            continue

        best = sub.loc[sub.groupby("bw_multiplier")[metric].idxmax()]
        best = best.sort_values("bw_multiplier")
        x = np.arange(len(best))
        bottom = np.zeros(len(best))
        cmap_vals = plt.cm.Set2(np.linspace(0, 1, len(time_cols)))

        fig, ax = plt.subplots(figsize=(max(6, len(best) * 0.8), 5))
        for i, col in enumerate(time_cols):
            vals = best[col].values.astype(float)
            ax.bar(x, vals, bottom=bottom, label=labels[i], color=cmap_vals[i],
                   edgecolor="white", linewidth=0.3)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"bw={m}x" for m in best["bw_multiplier"].values], fontsize=8)
        ax.set_xlabel("BW Multiplier (at lat=1x)")
        ax.set_ylabel("Time (ms)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{phase.capitalize()} — Time Breakdown [{bl}] (BS={mid_bs}, SEQ={mid_seq}, lat=1x)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_time_breakdown_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Summary CSV
# ══════════════════════════════════════════════════════════════════════════

def write_summary_csv(df, phase, out_dir):
    metric, _ = get_metric(phase)
    best_idx = df.groupby(["baseline_noc", "latency_multiplier", "bw_multiplier", "bs", "seq"])[metric].idxmax()
    best = df.loc[best_idx].sort_values(["baseline_noc", "latency_multiplier", "bw_multiplier", "bs", "seq"])

    cols = ["baseline_noc", "latency_multiplier", "bw_multiplier",
            "sm_count",
            "l1_link_bw_GBs", "l2_link_bw_GBs", "l3_link_bw_GBs",
            "bs", "seq", "tp", "ep", "dp", "pp"]
    if phase == "decode":
        cols += ["raw_utps_avg", "raw_stps_avg", "scaled_utps_avg", "scaled_stps_avg"]
    else:
        cols += ["raw_utps", "raw_stps", "scaled_utps", "scaled_stps"]
    cols += ["chip_power_W", "noc_power_per_device_W", "total_power_per_device_W",
             "hit_power_wall", "freq_scale_power"]
    cols = [c for c in cols if c in best.columns]

    out_path = os.path.join(out_dir, f"{phase}_best_configs.csv")
    best[cols].to_csv(out_path, index=False)
    print(f"  Summary CSV: {out_path} ({len(best)} rows)")


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _save(fig, out_dir, name):
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {name}.{{png,pdf}}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main(run_dir: str):
    print(f"=== Latency vs Bandwidth — Analysis for {run_dir} ===")
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

        print(f"  Baselines: {sorted(df['baseline_noc'].unique())}")
        print(f"  Latency mults: {sorted(df['latency_multiplier'].unique())}")
        print(f"  BW mults: {sorted(df['bw_multiplier'].unique())}")
        print(f"  Workloads: {len(df.groupby(['bs', 'seq']))}")
        print(f"  Total rows: {len(df)}")

        write_summary_csv(df, phase, out_dir)
        plot_lat_bw_heatmap(df, phase, out_dir)
        plot_lat_bw_heatmap_per_workload(df, phase, out_dir)
        plot_bw_at_fixed_latency(df, phase, out_dir)
        plot_lat_at_fixed_bw(df, phase, out_dir)
        plot_elasticity(df, phase, out_dir)
        plot_per_workload_elasticity(df, phase, out_dir)
        plot_power_area(df, phase, out_dir)
        plot_parallel_scheme(df, phase, out_dir)
        plot_cross_baseline(df, phase, out_dir)
        plot_time_breakdown(df, phase, out_dir)

    print(f"\n=== Done. Results in {out_dir}/ ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m mosaic.dse_space.case_study_latency_vs_bandwidth.dse_analysis <run_dir>")
        sys.exit(1)
    main(sys.argv[1])
