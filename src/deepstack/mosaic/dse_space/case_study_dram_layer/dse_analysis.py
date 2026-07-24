"""
dse_analysis.py — Post-DSE analysis & visualization for DRAM layer case study.

Usage:
    python3 -m mosaic.dse_space.case_study_dram_layer.dse_analysis <run_dir>

    e.g.:
    python3 -m mosaic.dse_space.case_study_dram_layer.dse_analysis \
        mosaic/dse_space/case_study_dram_layer/runs/your-run-id

Reads:
    <run_dir>/dram_layer_decode_result.csv
    <run_dir>/dram_layer_prefill_result.csv   (optional)
    <run_dir>/dram_layer_decode_invalid.csv   (optional, for validity rate)
    <run_dir>/dram_layer_prefill_invalid.csv  (optional)

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

# ── Style ─────────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════
# Data loading & preprocessing
# ══════════════════════════════════════════════════════════════════════════

def load_data(run_dir: str):
    """Load decode and prefill CSVs, return dict of DataFrames."""
    data = {}
    for phase in ("decode", "prefill"):
        result_path = os.path.join(run_dir, f"dram_layer_{phase}_result.csv")
        invalid_path = os.path.join(run_dir, f"dram_layer_{phase}_invalid.csv")
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


def add_derived_columns(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Add useful derived columns."""
    df = df.copy()
    df["config_key"] = (
        "m" + df["dram_total_layers"].astype(str) +
        "_n" + df["dram_active_layers"].astype(str)
    )
    df["sm_config"] = (
        "smem" + df["smem_capacity_KiB"].astype(str) +
        "K_l1_" + df["l1_throughput_Bpc"].astype(str)
    )
    df["total_mem_GiB"] = df["max_activation/GiB"] + df["mem_weight/GiB"] + df["kv_cache/GiB"]
    df["mem_util_pct"] = df["total_mem_GiB"] / df["ddr_capacity_GB"] * 100

    if phase == "decode":
        # Best metric for decode: scaled_stps_avg (higher = better)
        df["perf_metric"] = df["scaled_stps_avg"]
        df["perf_metric_name"] = "stps"
    else:
        # Best metric for prefill: scaled_utps (higher = better)
        df["perf_metric"] = df["scaled_utps"]
        df["perf_metric_name"] = "utps"

    # Parallel scheme string
    df["par_str"] = (
        "tp" + df["tp"].astype(str) +
        "_ep" + df["ep"].astype(str) +
        "_dp" + df["dp"].astype(str) +
        "_pp" + df["pp"].astype(str)
    )
    return df


# ══════════════════════════════════════════════════════════════════════════
# Plot 1: Performance vs DRAM layers (best config per m)
# ══════════════════════════════════════════════════════════════════════════

def plot_perf_vs_layers(df: pd.DataFrame, phase: str, out_dir: str):
    """For each workload (bs, seq), find best perf per m and plot."""
    metric = "scaled_stps_avg" if phase == "decode" else "scaled_stps"
    metric_label = "Scaled STPS" if "stps" in metric else "Scaled UTPS"

    workloads = df.groupby(["bs", "seq"]).size().reset_index().rename(columns={0: "count"})
    n_wl = len(workloads)
    if n_wl == 0:
        return

    ncols = min(4, n_wl)
    nrows = (n_wl + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                              squeeze=False)

    for idx, (_, wl_row) in enumerate(workloads.iterrows()):
        bs, seq = wl_row["bs"], wl_row["seq"]
        ax = axes[idx // ncols, idx % ncols]

        sub = df[(df["bs"] == bs) & (df["seq"] == seq)]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Best perf per (m, n)
        best = sub.loc[sub.groupby(["dram_total_layers", "dram_active_layers"])[metric].idxmax()]
        best = best.sort_values(["dram_total_layers", "dram_active_layers"])

        # Group by m, plot different n as separate series
        for n_val, grp in best.groupby("dram_active_layers"):
            grp = grp.sort_values("dram_total_layers")
            label = f"n={n_val}" if len(best["dram_active_layers"].unique()) > 1 else None
            ax.plot(grp["dram_total_layers"], grp[metric], marker="o", markersize=4,
                    linewidth=1.5, label=label)

        ax.set_title(f"BS={bs}, SEQ={seq}", fontsize=10)
        ax.set_xlabel("DRAM Layers (m)")
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.3)
        if len(best["dram_active_layers"].unique()) > 1:
            ax.legend(fontsize=7, ncol=2)

    # Hide unused axes
    for idx in range(n_wl, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"{phase.capitalize()} — {metric_label} vs DRAM Layers (best config per m,n)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, f"{phase}_perf_vs_layers")


# ══════════════════════════════════════════════════════════════════════════
# Plot 2: Heatmap — best perf per (m, workload)
# ══════════════════════════════════════════════════════════════════════════

def plot_perf_heatmap(df: pd.DataFrame, phase: str, out_dir: str):
    """Heatmap: rows = m (DRAM layers), cols = workload, value = best perf."""
    metric = "scaled_stps_avg" if phase == "decode" else "scaled_stps"

    df["workload"] = "BS" + df["bs"].astype(str) + "_S" + df["seq"].astype(str)
    pivot = df.groupby(["dram_total_layers", "workload"])[metric].max().unstack(fill_value=0)

    if pivot.empty or pivot.shape[0] < 2:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8), max(4, len(pivot.index) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"m={m}" for m in pivot.index], fontsize=8)
    ax.set_xlabel("Workload")
    ax.set_ylabel("DRAM Layers")
    fig.colorbar(im, ax=ax, label=metric, shrink=0.8)
    fig.suptitle(f"{phase.capitalize()} — Best {metric} per (m, workload)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, f"{phase}_perf_heatmap")


# ══════════════════════════════════════════════════════════════════════════
# Plot 3: Time breakdown for best configs
# ══════════════════════════════════════════════════════════════════════════

def plot_time_breakdown(df: pd.DataFrame, phase: str, out_dir: str):
    """Stacked bar: time breakdown for best-perf config per m (fixed workload)."""
    metric = "scaled_stps_avg" if phase == "decode" else "scaled_stps"

    # Pick a representative workload (medium BS, medium seq)
    bs_vals = sorted(df["bs"].unique())
    seq_vals = sorted(df["seq"].unique())
    mid_bs = bs_vals[len(bs_vals) // 2]
    mid_seq = seq_vals[len(seq_vals) // 2]
    sub = df[(df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
    if sub.empty:
        # Fallback: use whatever workload has most data
        counts = df.groupby(["bs", "seq"]).size()
        mid_bs, mid_seq = counts.idxmax()
        sub = df[(df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
    if sub.empty:
        return

    # Best config per m
    best_idx = sub.groupby("dram_total_layers")[metric].idxmax()
    best = sub.loc[best_idx].sort_values("dram_total_layers")

    if phase == "decode":
        time_cols = ["time_dense_ffn_ms", "time_moe_ms", "time_rms_norm_ms",
                     "time_add_residual_ms", "pp_p2p_time_ms"]
        # Also include MLA time from middle kv point
        if "time_mla_2ms" in best.columns:
            best = best.copy()
            best["time_mla_ms"] = best["time_mla_2ms"]
            time_cols = ["time_mla_ms"] + time_cols
        labels = [c.replace("time_", "").replace("_ms", "") for c in time_cols]
    else:
        time_cols = [c for c in ["time_mla_ms", "time_gqa_ms", "time_dense_ffn_ms",
                                  "time_moe_ms", "time_rms_norm_ms",
                                  "time_add_residual_ms", "pp_p2p_time_ms"]
                     if c in best.columns]
        labels = [c.replace("time_", "").replace("_ms", "") for c in time_cols]

    if not time_cols:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(best) * 0.6), 5))
    x = np.arange(len(best))
    bottom = np.zeros(len(best))
    cmap = plt.cm.Set2(np.linspace(0, 1, len(time_cols)))

    for i, col in enumerate(time_cols):
        vals = best[col].values.astype(float)
        ax.bar(x, vals, bottom=bottom, label=labels[i], color=cmap[i], edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"m={int(r['dram_total_layers'])}\nn={int(r['dram_active_layers'])}\nSM={int(r['sm_count'])}"
                         for _, r in best.iterrows()], fontsize=7)
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("DRAM Config")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{phase.capitalize()} Time Breakdown (BS={mid_bs}, SEQ={mid_seq}, best config per m)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, f"{phase}_time_breakdown")


# ══════════════════════════════════════════════════════════════════════════
# Plot 4: Power & thermal analysis
# ══════════════════════════════════════════════════════════════════════════

def plot_power_analysis(df: pd.DataFrame, phase: str, out_dir: str):
    """Power, freq scale, and area vs m for best configs."""
    metric = "scaled_stps_avg" if phase == "decode" else "scaled_stps"

    # Pick representative workload
    counts = df.groupby(["bs", "seq"]).size()
    mid_bs, mid_seq = counts.idxmax()
    sub = df[(df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
    if sub.empty:
        return

    best_idx = sub.groupby("dram_total_layers")[metric].idxmax()
    best = sub.loc[best_idx].sort_values("dram_total_layers")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    m_vals = best["dram_total_layers"].values

    # (0,0) Power breakdown
    ax = axes[0, 0]
    ax.bar(m_vals, best["chip_power_W"].values, label="Chip Power", color=COLORS["primary"], alpha=0.8)
    ax.bar(m_vals, best["noc_power_per_device_W"].values, bottom=best["chip_power_W"].values,
           label="NoC Power", color=COLORS["accent"], alpha=0.8)
    ax.axhline(y=100, color=COLORS["red"], linestyle="--", linewidth=1, label="TDP (100W)")
    ax.set_xlabel("DRAM Layers (m)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Per-Device Power")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # (0,1) Frequency scaling
    ax = axes[0, 1]
    ax.plot(m_vals, best["freq_scale_thermal"].values, marker="s", color=COLORS["red"],
            label="Thermal freq scale", linewidth=1.5)
    ax.plot(m_vals, best["freq_scale_power"].values, marker="^", color=COLORS["accent"],
            label="Power freq scale", linewidth=1.5)
    ax.plot(m_vals, best["total_freq_scale"].values, marker="o", color=COLORS["primary"],
            label="Total freq scale", linewidth=2)
    ax.axhline(y=1.0, color=COLORS["grey"], linestyle=":", linewidth=0.8)
    ax.set_xlabel("DRAM Layers (m)")
    ax.set_ylabel("Freq Scale Factor")
    ax.set_title("Frequency Scaling")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (1,0) Feasible SM capacity
    ax = axes[1, 0]
    ax.plot(
        m_vals,
        best["sm_count"].values,
        marker="D",
        color=COLORS["accent"],
        linewidth=1.5,
        label="Feasible SMs",
    )
    ax.set_xlabel("DRAM Layers (m)")
    ax.set_ylabel("SM Count")
    ax.set_title("Reference Capacity")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # (1,1) Perf vs Power (Pareto)
    ax = axes[1, 1]
    sc = ax.scatter(best["total_power_per_device_W"].values, best[metric].values,
                    c=m_vals, cmap="viridis", s=60, edgecolor="black", linewidth=0.5, zorder=3)
    for _, r in best.iterrows():
        ax.annotate(f"m={int(r['dram_total_layers'])}",
                    (r["total_power_per_device_W"], r[metric]),
                    fontsize=6, ha="center", va="bottom")
    ax.set_xlabel("Total Power per Device (W)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Performance vs Power")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="DRAM layers (m)", shrink=0.7)

    fig.suptitle(f"{phase.capitalize()} — Power & Area Analysis (BS={mid_bs}, SEQ={mid_seq})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, f"{phase}_power_analysis")


# ══════════════════════════════════════════════════════════════════════════
# Plot 5: SM config impact (smem_cap × l1_tp)
# ══════════════════════════════════════════════════════════════════════════

def plot_sm_config_impact(df: pd.DataFrame, phase: str, out_dir: str):
    """For each m, show how smem_cap and l1_tp affect perf."""
    metric = "scaled_stps_avg" if phase == "decode" else "scaled_stps"

    # Pick representative workload
    counts = df.groupby(["bs", "seq"]).size()
    mid_bs, mid_seq = counts.idxmax()
    sub = df[(df["bs"] == mid_bs) & (df["seq"] == mid_seq)]
    if sub.empty:
        return

    # Best perf per (m, smem, l1)
    best = sub.groupby(["dram_total_layers", "smem_capacity_KiB", "l1_throughput_Bpc"])[metric].max().reset_index()

    m_vals = sorted(best["dram_total_layers"].unique())
    if len(m_vals) < 2:
        return

    smem_vals = sorted(best["smem_capacity_KiB"].unique())
    l1_vals = sorted(best["l1_throughput_Bpc"].unique())

    # Subplots: one per smem_cap, lines per l1_tp, x = m
    ncols = len(smem_vals)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4), sharey=True, squeeze=False)

    for ci, smem in enumerate(smem_vals):
        ax = axes[0, ci]
        for l1 in l1_vals:
            sub2 = best[(best["smem_capacity_KiB"] == smem) & (best["l1_throughput_Bpc"] == l1)]
            sub2 = sub2.sort_values("dram_total_layers")
            if not sub2.empty:
                ax.plot(sub2["dram_total_layers"], sub2[metric], marker="o", markersize=4,
                        linewidth=1.5, label=f"L1={l1} B/c")
        ax.set_title(f"SMEM = {smem} KiB")
        ax.set_xlabel("DRAM Layers (m)")
        if ci == 0:
            ax.set_ylabel(metric.replace("_", " "))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{phase.capitalize()} — SM Config Impact (BS={mid_bs}, SEQ={mid_seq})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, f"{phase}_sm_config_impact")


# ══════════════════════════════════════════════════════════════════════════
# Plot 6: Validity rate — what fraction of configs can fit the model
# ══════════════════════════════════════════════════════════════════════════

def plot_validity_rate(df_valid: pd.DataFrame, df_invalid: pd.DataFrame,
                       phase: str, out_dir: str):
    """Bar chart: valid vs invalid scheme count per m."""
    valid_counts = df_valid.groupby("dram_total_layers").size().rename("valid")
    invalid_counts = df_invalid.groupby("dram_total_layers").size().rename("invalid")
    combined = pd.concat([valid_counts, invalid_counts], axis=1).fillna(0)
    combined = combined.sort_index()

    fig, ax = plt.subplots(figsize=(max(6, len(combined) * 0.5), 4))
    x = np.arange(len(combined))
    w = 0.6
    ax.bar(x, combined["valid"], w, label="Valid", color=COLORS["secondary"])
    ax.bar(x, combined["invalid"], w, bottom=combined["valid"], label="Invalid", color=COLORS["red"], alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"m={int(m)}" for m in combined.index], fontsize=8)
    ax.set_ylabel("# Parallel Scheme Combos")
    ax.set_xlabel("DRAM Layers")
    ax.set_title(f"{phase.capitalize()} — Valid vs Invalid Configs per m")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add validity % annotation
    for i, m in enumerate(combined.index):
        total = combined.loc[m, "valid"] + combined.loc[m, "invalid"]
        if total > 0:
            pct = combined.loc[m, "valid"] / total * 100
            ax.text(i, total, f"{pct:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save(fig, out_dir, f"{phase}_validity_rate")


# ══════════════════════════════════════════════════════════════════════════
# Plot 7: BW utilization — effective BW vs m
# ══════════════════════════════════════════════════════════════════════════

def plot_bw_breakdown(df: pd.DataFrame, phase: str, out_dir: str):
    """Per unique (m, n, smem, l1), show BW breakdown as stacked bars."""
    cols = ["dram_total_layers", "dram_active_layers", "smem_capacity_KiB",
            "l1_throughput_Bpc", "ddr_peak_bw_TBs", "ddr_eff_bw_TBs",
            "is_l1_bound", "littles_law_limited", "sm_count"]
    arch_df = df[cols].drop_duplicates().sort_values(
        ["dram_total_layers", "dram_active_layers", "smem_capacity_KiB", "l1_throughput_Bpc"])

    # Focus on m=n configs for clarity
    full_connect = arch_df[arch_df["dram_total_layers"] == arch_df["dram_active_layers"]]
    if full_connect.empty:
        full_connect = arch_df

    # Group by (smem, l1)
    smem_vals = sorted(full_connect["smem_capacity_KiB"].unique())
    l1_vals = sorted(full_connect["l1_throughput_Bpc"].unique())

    nrows = len(smem_vals)
    ncols = len(l1_vals)
    if nrows == 0 or ncols == 0:
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows),
                              sharex=True, squeeze=False)

    for ri, smem in enumerate(smem_vals):
        for ci, l1 in enumerate(l1_vals):
            ax = axes[ri, ci]
            sub = full_connect[(full_connect["smem_capacity_KiB"] == smem) &
                               (full_connect["l1_throughput_Bpc"] == l1)]
            sub = sub.sort_values("dram_total_layers")
            if sub.empty:
                ax.set_visible(False)
                continue

            m_vals = sub["dram_total_layers"].values
            peak = sub["ddr_peak_bw_TBs"].values
            eff = sub["ddr_eff_bw_TBs"].values

            x = np.arange(len(m_vals))
            w = 0.6

            # Stacked: eff (blue), peak-eff (light)
            ax.bar(x, eff, w, color=COLORS["primary"], label="Eff BW")
            ax.bar(x, peak - eff, w, bottom=eff, color=COLORS["light"], edgecolor=COLORS["grey"],
                   linewidth=0.3, label="Peak - Eff")

            # L1 bound markers
            l1_bound_mask = sub["is_l1_bound"].values
            if np.any(l1_bound_mask):
                ax.scatter(x[l1_bound_mask], eff[l1_bound_mask] * 0.5,
                           marker="x", color=COLORS["red"], s=30, zorder=5, label="L1 bound")

            ax.set_xticks(x)
            ax.set_xticklabels([str(m) for m in m_vals], fontsize=7)
            ax.set_title(f"SMEM={smem}K, L1={l1}B/c", fontsize=9, pad=2)
            ax.grid(axis="y", alpha=0.3)

            if ci == 0:
                ax.set_ylabel("BW (TB/s)", fontsize=9)
            if ri == nrows - 1:
                ax.set_xlabel("DRAM Layers (m=n)")

    # Shared legend
    handles = [
        Patch(facecolor=COLORS["primary"], label="Effective BW"),
        Patch(facecolor=COLORS["light"], edgecolor=COLORS["grey"], label="Peak - Eff"),
        Line2D([0], [0], marker="x", color=COLORS["red"], linestyle="None",
               markersize=5, label="L1 Bound"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"{phase.capitalize()} — BW Breakdown (m=n, single chip)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, f"{phase}_bw_breakdown")


# ══════════════════════════════════════════════════════════════════════════
# Plot 8: Parallel scheme analysis
# ══════════════════════════════════════════════════════════════════════════

def plot_parallel_scheme(df: pd.DataFrame, phase: str, out_dir: str):
    """Show which parallel scheme wins per (m, workload)."""
    metric = "scaled_stps_avg" if phase == "decode" else "scaled_stps"

    best_idx = df.groupby(["dram_total_layers", "bs", "seq"])[metric].idxmax()
    best = df.loc[best_idx]

    # Summarize parallel scheme frequency per m
    par_cols = ["tp", "ep", "dp", "pp"]
    best["par_key"] = best[par_cols].astype(str).agg("_".join, axis=1)

    pivot = best.groupby(["dram_total_layers", "par_key"]).size().unstack(fill_value=0)
    if pivot.empty or pivot.shape[0] < 2:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.index) * 0.6), 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set3", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("DRAM Layers (m)")
    ax.set_ylabel("Count (workloads)")
    ax.set_title(f"{phase.capitalize()} — Winning Parallel Scheme per m")
    ax.legend(title="tp_ep_dp_pp", fontsize=6, title_fontsize=7,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, f"{phase}_parallel_scheme")


# ══════════════════════════════════════════════════════════════════════════
# Summary CSV: best config per (m, workload)
# ══════════════════════════════════════════════════════════════════════════

def write_summary_csv(df: pd.DataFrame, phase: str, out_dir: str):
    """Write summary CSV: best perf config per (m, bs, seq)."""
    metric = "scaled_stps_avg" if phase == "decode" else "scaled_stps"

    best_idx = df.groupby(["dram_total_layers", "dram_active_layers", "bs", "seq"])[metric].idxmax()
    best = df.loc[best_idx].sort_values(["dram_total_layers", "bs", "seq"])

    summary_cols = [
        "dram_total_layers", "dram_active_layers", "sm_count",
        "smem_capacity_KiB", "l1_throughput_Bpc",
        "ddr_eff_bw_TBs", "ddr_capacity_GB",
        "freq_scale_thermal",
        "bs", "seq",
        "tp", "ep", "dp", "pp",
    ]
    if phase == "decode":
        summary_cols += ["raw_utps_avg", "raw_stps_avg", "scaled_utps_avg", "scaled_stps_avg"]
    else:
        summary_cols += ["raw_utps", "raw_stps", "scaled_utps", "scaled_stps"]
    summary_cols += [
        "chip_power_W", "total_power_per_device_W",
        "hit_power_wall", "total_freq_scale",
    ]

    # Only include columns that exist
    summary_cols = [c for c in summary_cols if c in best.columns]

    out_path = os.path.join(out_dir, f"{phase}_best_configs.csv")
    best[summary_cols].to_csv(out_path, index=False)
    print(f"  Summary CSV: {out_path} ({len(best)} rows)")


# ══════════════════════════════════════════════════════════════════════════
# Plot 9: Perf scaling — normalized to baseline (m=4, n=4)
# ══════════════════════════════════════════════════════════════════════════

def plot_perf_scaling(df: pd.DataFrame, phase: str, out_dir: str):
    """Normalized perf (vs m=4 baseline) for each workload."""
    metric = "scaled_stps_avg" if phase == "decode" else "scaled_stps"

    workloads = df.groupby(["bs", "seq"]).size().reset_index().rename(columns={0: "count"})
    if len(workloads) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, wl_row in workloads.iterrows():
        bs, seq = wl_row["bs"], wl_row["seq"]
        sub = df[(df["bs"] == bs) & (df["seq"] == seq)]
        best_per_m = sub.groupby("dram_total_layers")[metric].max()

        # Normalize to smallest m available (or m=4 if exists)
        baseline_m = 4 if 4 in best_per_m.index else best_per_m.index.min()
        baseline_val = best_per_m.get(baseline_m, best_per_m.iloc[0])
        if baseline_val > 0:
            normalized = best_per_m / baseline_val
            ax.plot(normalized.index, normalized.values, marker="o", markersize=4,
                    linewidth=1.5, label=f"BS={bs},S={seq}", alpha=0.8)

    ax.axhline(y=1.0, color=COLORS["grey"], linestyle=":", linewidth=0.8)
    ax.set_xlabel("DRAM Layers (m)")
    ax.set_ylabel(f"Normalized {metric.replace('_', ' ')} (vs m={baseline_m})")
    ax.set_title(f"{phase.capitalize()} — Perf Scaling vs Baseline (m={baseline_m})")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, f"{phase}_perf_scaling")


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
    print(f"=== DSE Analysis for {run_dir} ===")
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
        df = add_derived_columns(data[phase], phase)

        # Summary
        print(f"  m range: {df['dram_total_layers'].min()} - {df['dram_total_layers'].max()}")
        print(f"  Unique configs: {len(df.groupby(['dram_total_layers','dram_active_layers','smem_capacity_KiB','l1_throughput_Bpc']))}")
        print(f"  Workloads: {len(df.groupby(['bs','seq']))}")
        print(f"  Power wall hits: {df['hit_power_wall'].sum()} / {len(df)}")

        write_summary_csv(df, phase, out_dir)
        plot_perf_vs_layers(df, phase, out_dir)
        plot_perf_heatmap(df, phase, out_dir)
        plot_time_breakdown(df, phase, out_dir)
        plot_power_analysis(df, phase, out_dir)
        plot_sm_config_impact(df, phase, out_dir)
        plot_bw_breakdown(df, phase, out_dir)
        plot_parallel_scheme(df, phase, out_dir)
        plot_perf_scaling(df, phase, out_dir)

        # Validity rate (if invalid CSV exists)
        inv_key = f"{phase}_invalid"
        if inv_key in data:
            plot_validity_rate(df, data[inv_key], phase, out_dir)

    print(f"\n=== Done. Results in {out_dir}/ ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m mosaic.dse_space.case_study_dram_layer.dse_analysis <run_dir>")
        print("  e.g.: python3 -m mosaic.dse_space.case_study_dram_layer.dse_analysis "
              "mosaic/dse_space/case_study_dram_layer/runs/your-run-id")
        sys.exit(1)
    main(sys.argv[1])
