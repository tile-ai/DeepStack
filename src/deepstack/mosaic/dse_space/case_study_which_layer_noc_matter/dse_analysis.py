"""
dse_analysis.py — Post-DSE analysis for "Which NoC Layer's BW Matters Most?"

Usage:
    python3 -m mosaic.dse_space.case_study_which_layer_noc_matter.dse_analysis <run_dir>

Reads:
    <run_dir>/noc_bw_decode_result.csv
    <run_dir>/noc_bw_prefill_result.csv
    <run_dir>/noc_bw_decode_invalid.csv   (optional)
    <run_dir>/noc_bw_prefill_invalid.csv  (optional)

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

LAYER_COLORS = {"L1": "#2196F3", "L2": "#FF9800", "L3": "#4CAF50"}
BASELINE_MARKERS = {"torus_mesh_switch_1": "o", "torus_mesh_mesh_3": "s"}


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_data(run_dir: str):
    data = {}
    for phase in ("decode", "prefill"):
        result_path = os.path.join(run_dir, f"noc_bw_{phase}_result.csv")
        invalid_path = os.path.join(run_dir, f"noc_bw_{phase}_invalid.csv")
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
# Plot 1: Per-layer BW sensitivity — perf vs bw_multiplier for each layer
# ══════════════════════════════════════════════════════════════════════════

def plot_layer_bw_sensitivity(df, phase, out_dir):
    """For each workload, plot best perf vs bw_multiplier, one curve per scaled_layer."""
    metric, metric_label = get_metric(phase)

    workloads = df.groupby(["bs", "seq"]).size().reset_index().rename(columns={0: "cnt"})
    baselines = sorted(df["baseline_noc"].unique())

    for bl in baselines:
        sub_bl = df[df["baseline_noc"] == bl]
        n_wl = len(workloads)
        ncols = min(4, n_wl)
        nrows = (n_wl + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

        for idx, (_, wl) in enumerate(workloads.iterrows()):
            bs, seq = wl["bs"], wl["seq"]
            ax = axes[idx // ncols, idx % ncols]
            sub = sub_bl[(sub_bl["bs"] == bs) & (sub_bl["seq"] == seq)]
            if sub.empty:
                ax.set_visible(False)
                continue

            for layer in ["L1", "L2", "L3"]:
                sl = sub[sub["scaled_layer"] == layer]
                if sl.empty:
                    continue
                best = sl.groupby("bw_multiplier")[metric].max().sort_index()
                ax.plot(best.index, best.values, marker="o", markersize=4, linewidth=1.5,
                        color=LAYER_COLORS[layer], label=layer)

            ax.set_title(f"BS={bs}, SEQ={seq}", fontsize=9)
            ax.set_xlabel("BW Multiplier")
            ax.set_ylabel(metric_label)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.set_xscale("log", base=2)

        for idx in range(n_wl, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle(f"{phase.capitalize()} — Per-Layer BW Sensitivity [{bl}]",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, out_dir, f"{phase}_layer_bw_sensitivity_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 2: Heatmap — best perf per (scaled_layer × bw_multiplier)
# ══════════════════════════════════════════════════════════════════════════

def plot_layer_bw_heatmap(df, phase, out_dir):
    """Heatmap: rows=layer, cols=bw_mult, value=best perf across workloads."""
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    for bl in baselines:
        sub = df[df["baseline_noc"] == bl]
        pivot = sub.groupby(["scaled_layer", "bw_multiplier"])[metric].max().unstack(fill_value=0)
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), 3))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x}x" for x in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("Scaled Layer")
        fig.colorbar(im, ax=ax, label=metric_label, shrink=0.8)
        fig.suptitle(f"{phase.capitalize()} — Best {metric_label} [{bl}]", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_layer_bw_heatmap_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 3: Normalized sensitivity — which layer has steepest slope?
# ══════════════════════════════════════════════════════════════════════════

def plot_normalized_sensitivity(df, phase, out_dir):
    """Normalize perf to bw_mult=1.0 baseline, show relative gain per layer."""
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    # Pick representative workload (mid BS, mid SEQ)
    bs_vals = sorted(df["bs"].unique())
    seq_vals = sorted(df["seq"].unique())
    mid_bs = bs_vals[len(bs_vals) // 2]
    mid_seq = seq_vals[len(seq_vals) // 2]

    fig, axes = plt.subplots(1, len(baselines), figsize=(6 * len(baselines), 5), squeeze=False)

    for bi, bl in enumerate(baselines):
        ax = axes[0, bi]
        sub = df[(df["baseline_noc"] == bl) & (df["bs"] == mid_bs) & (df["seq"] == mid_seq)]

        for layer in ["L1", "L2", "L3"]:
            sl = sub[sub["scaled_layer"] == layer]
            if sl.empty:
                continue
            best = sl.groupby("bw_multiplier")[metric].max().sort_index()
            baseline_val = best.get(1.0, best.iloc[0])
            if baseline_val > 0:
                normalized = best / baseline_val
                ax.plot(normalized.index, normalized.values, marker="o", markersize=5,
                        linewidth=2, color=LAYER_COLORS[layer], label=layer)

        ax.axhline(y=1.0, color=COLORS["grey"], linestyle=":", linewidth=0.8)
        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("Normalized Performance (vs 1x)")
        ax.set_title(f"[{bl}]", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)

    fig.suptitle(f"{phase.capitalize()} — Normalized BW Sensitivity (BS={mid_bs}, SEQ={mid_seq})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, f"{phase}_normalized_sensitivity")


# ══════════════════════════════════════════════════════════════════════════
# Plot 4: All-workload normalized sensitivity (aggregated)
# ══════════════════════════════════════════════════════════════════════════

def plot_aggregated_sensitivity(df, phase, out_dir):
    """Average normalized perf across all workloads to get one curve per layer."""
    metric, _ = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    fig, axes = plt.subplots(1, len(baselines), figsize=(6 * len(baselines), 5), squeeze=False)

    for bi, bl in enumerate(baselines):
        ax = axes[0, bi]
        sub = df[df["baseline_noc"] == bl]
        workloads = sub.groupby(["bs", "seq"]).size().reset_index()[["bs", "seq"]]

        for layer in ["L1", "L2", "L3"]:
            all_norm = []
            for _, wl in workloads.iterrows():
                sl = sub[(sub["scaled_layer"] == layer) &
                         (sub["bs"] == wl["bs"]) & (sub["seq"] == wl["seq"])]
                best = sl.groupby("bw_multiplier")[metric].max().sort_index()
                base = best.get(1.0, None)
                if base and base > 0:
                    all_norm.append(best / base)

            if all_norm:
                combined = pd.concat(all_norm, axis=1).mean(axis=1)
                ax.plot(combined.index, combined.values, marker="o", markersize=5,
                        linewidth=2, color=LAYER_COLORS[layer], label=layer)

                # Shade min-max range
                stacked = pd.concat(all_norm, axis=1)
                lo = stacked.min(axis=1)
                hi = stacked.max(axis=1)
                ax.fill_between(combined.index, lo.values, hi.values,
                                color=LAYER_COLORS[layer], alpha=0.15)

        ax.axhline(y=1.0, color=COLORS["grey"], linestyle=":", linewidth=0.8)
        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("Avg Normalized Performance")
        ax.set_title(f"[{bl}]", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)

    fig.suptitle(f"{phase.capitalize()} — Aggregated BW Sensitivity (all workloads, mean ± range)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, f"{phase}_aggregated_sensitivity")


# ══════════════════════════════════════════════════════════════════════════
# Plot 5: capacity vs config
# ══════════════════════════════════════════════════════════════════════════

def plot_capacity_sm(df, phase, out_dir):
    """Show how feasible SM capacity changes per (layer, bw_mult)."""
    arch_df = df[
        ["baseline_noc", "scaled_layer", "bw_multiplier", "sm_count"]
    ].drop_duplicates()

    baselines = sorted(arch_df["baseline_noc"].unique())
    fig, axes = plt.subplots(1, len(baselines), figsize=(6 * len(baselines), 4.5), squeeze=False)

    for bi, bl in enumerate(baselines):
        ax = axes[0, bi]
        sub = arch_df[arch_df["baseline_noc"] == bl]

        for layer in ["L1", "L2", "L3"]:
            sl = sub[sub["scaled_layer"] == layer].sort_values("bw_multiplier")
            if sl.empty:
                continue
            # Take first (they're all same sm_count per (bl, layer, bw_mult))
            sl = sl.drop_duplicates(subset=["bw_multiplier"])
            ax.plot(sl["bw_multiplier"], sl["sm_count"], marker="o", markersize=5,
                    linewidth=1.5, color=LAYER_COLORS[layer], label=f"{layer} SM")

        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("Feasible SM Count")
        ax.set_title(f"[{bl}]", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)

    fig.suptitle(f"{phase.capitalize()} — Feasible SM Capacity vs BW Scaling",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, f"{phase}_capacity_sm_count")


# ══════════════════════════════════════════════════════════════════════════
# Plot 6: Power analysis per layer scaling
# ══════════════════════════════════════════════════════════════════════════

def plot_power_analysis(df, phase, out_dir):
    """Power breakdown and freq_scale vs bw_multiplier, per layer."""
    metric, _ = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())

    # Representative workload
    bs_vals = sorted(df["bs"].unique())
    seq_vals = sorted(df["seq"].unique())
    mid_bs = bs_vals[len(bs_vals) // 2]
    mid_seq = seq_vals[len(seq_vals) // 2]

    for bl in baselines:
        sub = df[(df["baseline_noc"] == bl) & (df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Power vs bw_mult per layer
        ax = axes[0]
        for layer in ["L1", "L2", "L3"]:
            sl = sub[sub["scaled_layer"] == layer]
            best = sl.loc[sl.groupby("bw_multiplier")[metric].idxmax()]
            best = best.sort_values("bw_multiplier")
            ax.plot(best["bw_multiplier"], best["total_power_per_device_W"],
                    marker="o", color=LAYER_COLORS[layer], label=layer, linewidth=1.5)
        ax.axhline(y=100, color=COLORS["red"], linestyle="--", linewidth=0.8, label="TDP 100W")
        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("Total Power/Device (W)")
        ax.set_title("Power vs BW Scaling")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)

        # Freq scale
        ax = axes[1]
        for layer in ["L1", "L2", "L3"]:
            sl = sub[sub["scaled_layer"] == layer]
            best = sl.loc[sl.groupby("bw_multiplier")[metric].idxmax()]
            best = best.sort_values("bw_multiplier")
            ax.plot(best["bw_multiplier"], best["freq_scale_power"],
                    marker="o", color=LAYER_COLORS[layer], label=layer, linewidth=1.5)
        ax.axhline(y=1.0, color=COLORS["grey"], linestyle=":", linewidth=0.8)
        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("Freq Scale (power)")
        ax.set_title("Frequency Scaling")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)

        # Power wall hits
        ax = axes[2]
        for layer in ["L1", "L2", "L3"]:
            sl = sub[sub["scaled_layer"] == layer]
            pw = sl.groupby("bw_multiplier")["hit_power_wall"].mean().sort_index()
            ax.plot(pw.index, pw.values * 100, marker="o",
                    color=LAYER_COLORS[layer], label=layer, linewidth=1.5)
        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("% Configs Hitting Power Wall")
        ax.set_title("Power Wall Rate")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)

        fig.suptitle(f"{phase.capitalize()} — Power Analysis [{bl}] (BS={mid_bs}, SEQ={mid_seq})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_power_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 7: Time breakdown for best config at each bw_mult
# ══════════════════════════════════════════════════════════════════════════

def plot_time_breakdown(df, phase, out_dir):
    """Stacked bar: time components for the best config at each bw_mult, per layer."""
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
        sub = df[(df["baseline_noc"] == bl) & (df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
        if sub.empty:
            continue

        layers = ["L1", "L2", "L3"]
        fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5), squeeze=False)

        for li, layer in enumerate(layers):
            ax = axes[0, li]
            sl = sub[sub["scaled_layer"] == layer]
            if sl.empty:
                ax.set_visible(False)
                continue

            best = sl.loc[sl.groupby("bw_multiplier")[metric].idxmax()]
            best = best.sort_values("bw_multiplier")
            x = np.arange(len(best))
            bottom = np.zeros(len(best))
            cmap = plt.cm.Set2(np.linspace(0, 1, len(time_cols)))

            for i, col in enumerate(time_cols):
                vals = best[col].values.astype(float)
                ax.bar(x, vals, bottom=bottom, label=labels[i], color=cmap[i],
                       edgecolor="white", linewidth=0.3)
                bottom += vals

            ax.set_xticks(x)
            ax.set_xticklabels([f"{m}x" for m in best["bw_multiplier"].values], fontsize=8)
            ax.set_xlabel("BW Multiplier")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"Scale {layer}")
            ax.legend(fontsize=6, ncol=2)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{phase.capitalize()} — Time Breakdown [{bl}] (BS={mid_bs}, SEQ={mid_seq})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_time_breakdown_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 8: Parallel scheme preference per layer scaling
# ══════════════════════════════════════════════════════════════════════════

def plot_parallel_scheme(df, phase, out_dir):
    """Which parallel scheme wins as BW changes per layer."""
    metric, _ = get_metric(phase)
    par_cols = ["tp", "ep", "dp", "pp"]
    df = df.copy()
    df["par_key"] = df[par_cols].astype(str).agg("_".join, axis=1)

    baselines = sorted(df["baseline_noc"].unique())
    for bl in baselines:
        sub = df[df["baseline_noc"] == bl]
        best_idx = sub.groupby(["scaled_layer", "bw_multiplier", "bs", "seq"])[metric].idxmax()
        best = sub.loc[best_idx]

        pivot = best.groupby(["scaled_layer", "bw_multiplier", "par_key"]).size().reset_index(name="count")
        layers = sorted(pivot["scaled_layer"].unique())

        fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5), squeeze=False)
        for li, layer in enumerate(layers):
            ax = axes[0, li]
            lp = pivot[pivot["scaled_layer"] == layer]
            lp_pivot = lp.pivot_table(index="bw_multiplier", columns="par_key",
                                       values="count", fill_value=0)
            lp_pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set3",
                          edgecolor="white", linewidth=0.3)
            ax.set_title(f"Scale {layer}")
            ax.set_xlabel("BW Multiplier")
            ax.set_ylabel("# Winning Configs")
            ax.legend(title="tp_ep_dp_pp", fontsize=5, title_fontsize=6,
                      bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{phase.capitalize()} — Winning Parallel Scheme [{bl}]",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_parallel_scheme_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 9b: Best STPS vs BW multiplier, one curve per BS (per layer)
# ══════════════════════════════════════════════════════════════════════════

def plot_best_stps_per_bs(df, phase, out_dir):
    """For each (baseline, layer), plot best STPS vs bw_multiplier with one curve per BS.

    Decode has seq=1 so curves are just per-bs.
    Prefill may have multiple seq values — each (bs, seq) gets its own curve.
    """
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())
    layers = sorted(df["scaled_layer"].unique())

    seq_vals = sorted(df["seq"].unique())
    single_seq = len(seq_vals) == 1

    for bl in baselines:
        ncols = len(layers)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)

        for li, layer in enumerate(layers):
            ax = axes[0, li]
            sub = df[(df["baseline_noc"] == bl) & (df["scaled_layer"] == layer)]
            if sub.empty:
                ax.set_visible(False)
                continue

            workloads = sub.groupby(["bs", "seq"]).size().reset_index()[["bs", "seq"]]
            cmap = plt.cm.viridis(np.linspace(0, 1, len(workloads)))

            for wi, (_, wl) in enumerate(workloads.iterrows()):
                bs, seq = wl["bs"], wl["seq"]
                sl = sub[(sub["bs"] == bs) & (sub["seq"] == seq)]
                best = sl.groupby("bw_multiplier")[metric].max().sort_index()
                label = f"BS={bs}" if single_seq else f"BS={bs},S={seq}"
                ax.plot(best.index, best.values, marker="o", markersize=4,
                        linewidth=1.5, color=cmap[wi], label=label)

            ax.set_xlabel("BW Multiplier")
            ax.set_ylabel(metric_label)
            ax.set_title(f"Scale {layer}", fontsize=10)
            ax.legend(fontsize=7, ncol=1)
            ax.grid(alpha=0.3)
            ax.set_xscale("log", base=2)

        fig.suptitle(f"{phase.capitalize()} — Best STPS per BS vs BW [{bl}]",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_best_stps_per_bs_{bl}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 9: Cross-baseline comparison (both baselines side by side)
# ══════════════════════════════════════════════════════════════════════════

def plot_cross_baseline(df, phase, out_dir):
    """Compare aggregated sensitivity across baselines on same plot."""
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())
    if len(baselines) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for li, layer in enumerate(["L1", "L2", "L3"]):
        ax = axes[li]
        for bl in baselines:
            sub = df[(df["baseline_noc"] == bl) & (df["scaled_layer"] == layer)]
            workloads = sub.groupby(["bs", "seq"]).size().reset_index()[["bs", "seq"]]

            all_norm = []
            for _, wl in workloads.iterrows():
                sl = sub[(sub["bs"] == wl["bs"]) & (sub["seq"] == wl["seq"])]
                best = sl.groupby("bw_multiplier")[metric].max().sort_index()
                base = best.get(1.0, None)
                if base and base > 0:
                    all_norm.append(best / base)

            if all_norm:
                combined = pd.concat(all_norm, axis=1).mean(axis=1)
                marker = BASELINE_MARKERS.get(bl, "^")
                ax.plot(combined.index, combined.values, marker=marker,
                        markersize=5, linewidth=2, label=bl.replace("torus_mesh_", ""))

        ax.axhline(y=1.0, color=COLORS["grey"], linestyle=":", linewidth=0.8)
        ax.set_xlabel("BW Multiplier")
        ax.set_ylabel("Avg Normalized Perf")
        ax.set_title(f"Scale {layer}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)

    fig.suptitle(f"{phase.capitalize()} — Cross-Baseline BW Sensitivity Comparison",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, f"{phase}_cross_baseline")


# ══════════════════════════════════════════════════════════════════════════
# Summary CSV
# ══════════════════════════════════════════════════════════════════════════

def write_summary_csv(df, phase, out_dir):
    metric, _ = get_metric(phase)
    best_idx = df.groupby(["baseline_noc", "scaled_layer", "bw_multiplier", "bs", "seq"])[metric].idxmax()
    best = df.loc[best_idx].sort_values(["baseline_noc", "scaled_layer", "bw_multiplier", "bs", "seq"])

    cols = ["baseline_noc", "scaled_layer", "bw_multiplier",
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
    print(f"=== Which Layer NoC BW Matters — Analysis for {run_dir} ===")
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
        print(f"  Layers: {sorted(df['scaled_layer'].unique())}")
        print(f"  BW multipliers: {sorted(df['bw_multiplier'].unique())}")
        print(f"  Workloads: {len(df.groupby(['bs', 'seq']))}")
        print(f"  Total rows: {len(df)}")

        write_summary_csv(df, phase, out_dir)
        plot_layer_bw_sensitivity(df, phase, out_dir)
        plot_layer_bw_heatmap(df, phase, out_dir)
        plot_normalized_sensitivity(df, phase, out_dir)
        plot_aggregated_sensitivity(df, phase, out_dir)
        plot_capacity_sm(df, phase, out_dir)
        plot_power_analysis(df, phase, out_dir)
        plot_time_breakdown(df, phase, out_dir)
        plot_parallel_scheme(df, phase, out_dir)
        plot_best_stps_per_bs(df, phase, out_dir)
        plot_cross_baseline(df, phase, out_dir)

    print(f"\n=== Done. Results in {out_dir}/ ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m mosaic.dse_space.case_study_which_layer_noc_matter.dse_analysis <run_dir>")
        sys.exit(1)
    main(sys.argv[1])
