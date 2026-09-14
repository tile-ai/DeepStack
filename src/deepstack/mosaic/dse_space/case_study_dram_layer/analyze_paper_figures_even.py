#!/usr/bin/env python3
"""
Paper-quality analysis of DRAM layer DSE results — EVEN m,n only.
Filters to configurations where both dram_total_layers and dram_active_layers
are even, with m up to 14. Heatmap axes: 2, 4, 6, 8, 10, 12, 14.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# ── Constants ──────────────────────────────────────────────────────────────
POWER_CAP = 100.0       # W per device
NUM_DEVICES = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures_paper_even")
DATA_DIR = os.path.join(BASE_DIR, "lfs")
os.makedirs(FIG_DIR, exist_ok=True)

EVEN_M_VALUES = [2, 4, 6, 8, 10, 12, 14]

# Plot style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BS_COLORS = {
    1: "#1f77b4", 4: "#ff7f0e", 16: "#2ca02c", 64: "#d62728",
    256: "#9467bd", 1024: "#8c564b", 4096: "#e377c2", 16384: "#7f7f7f",
}


# ── Data loading & processing ─────────────────────────────────────────────
def load_and_process(csv_name, scaled_col, raw_col):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    # Filter: even m, even n, m <= 14
    df = df[(df["dram_total_layers"] % 2 == 0) &
            (df["dram_active_layers"] % 2 == 0) &
            (df["dram_total_layers"] <= 14)]
    over = df["total_power_per_device_W"] > POWER_CAP
    df["effective_power_W"] = df["total_power_per_device_W"].copy()
    df.loc[over, "effective_power_W"] = POWER_CAP
    df["effective_stps"] = df[raw_col].copy()
    df.loc[over, "effective_stps"] = df.loc[over, scaled_col]
    df["system_power_W"] = df["effective_power_W"] * NUM_DEVICES
    df["tokens_per_joule"] = df["effective_stps"] / df["system_power_W"]
    df["mn_label"] = "m=" + df["dram_total_layers"].astype(str) + ",n=" + df["dram_active_layers"].astype(str)
    df["parallelism"] = ("tp" + df["tp"].astype(str) + "_ep" + df["ep"].astype(str) +
                          "_dp" + df["dp"].astype(str) + "_pp" + df["pp"].astype(str))
    return df


def find_best_per_bs(df, metric, bs_col="bs"):
    idx = df.groupby(bs_col)[metric].idxmax()
    return df.loc[idx].sort_values(bs_col).reset_index(drop=True)


def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved {name}.pdf/png")


# ── Figure 1: Pareto Scatter ──────────────────────────────────────────────
def fig_pareto_scatter(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    for ax, df, title in [
        (axes[0], pf, "Prefill (seq=1024)"),
        (axes[1], dc, "Decode"),
    ]:
        bs_vals = sorted(df["bs"].unique())
        best_perf = find_best_per_bs(df, "effective_stps")
        best_eff = find_best_per_bs(df, "tokens_per_joule")

        for bs in bs_vals:
            sub = df[df["bs"] == bs]
            c = BS_COLORS.get(bs, "#333333")
            ax.scatter(sub["tokens_per_joule"], sub["effective_stps"],
                       s=1.5, alpha=0.08, c=c, rasterized=True)

        for bs in bs_vals:
            c = BS_COLORS.get(bs, "#333333")
            bp = best_perf[best_perf["bs"] == bs]
            be = best_eff[best_eff["bs"] == bs]
            if not bp.empty:
                ax.scatter(bp["tokens_per_joule"], bp["effective_stps"],
                           marker="*", s=80, c=c, edgecolors="black", linewidths=0.5, zorder=10)
            if not be.empty:
                ax.scatter(be["tokens_per_joule"], be["effective_stps"],
                           marker="D", s=35, c=c, edgecolors="black", linewidths=0.5, zorder=10)

        ax.set_xlabel("Energy Efficiency (tokens/J)")
        ax.set_ylabel("System Throughput (tokens/s)")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")

    handles = []
    for bs in sorted(set(pf["bs"].unique()) | set(dc["bs"].unique())):
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=BS_COLORS.get(bs, "#333"),
                              markersize=5, label=f"bs={bs}"))
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                          markeredgecolor="black", markersize=10, label="Best perf"))
    handles.append(Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                          markeredgecolor="black", markersize=6, label="Best tok/J"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12),
               frameon=True, fancybox=True)
    fig.tight_layout()
    _save(fig, "fig1_pareto_scatter_even")


# ── Figure 2: Arch comparison ─────────────────────────────────────────────
def fig_arch_comparison(pf, dc):
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.5))

    for row, (df, phase) in enumerate([(pf, "Prefill"), (dc, "Decode")]):
        bp = find_best_per_bs(df, "effective_stps")
        be = find_best_per_bs(df, "tokens_per_joule")
        bs_vals = sorted(df["bs"].unique())
        x = np.arange(len(bs_vals))
        w = 0.35

        for col, (metric, ylabel) in enumerate([
            ("dram_active_layers", "Active DRAM Layers (n)"),
            ("l1_throughput_Bpc", "L1 Throughput (B/cycle)"),
            ("sm_count", "SM Count"),
        ]):
            ax = axes[row, col]
            ax.bar(x - w/2, bp[metric].values, w, label="Best Perf", color="#2196F3", edgecolor="black", linewidth=0.3)
            ax.bar(x + w/2, be[metric].values, w, label="Best Tok/J", color="#FF9800", edgecolor="black", linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in bs_vals], rotation=45, ha="right")
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(ylabel)
            if col == 0:
                ax.annotate(phase, xy=(-0.35, 0.5), xycoords="axes fraction",
                            fontsize=11, fontweight="bold", rotation=90, va="center")
            ax.set_xlabel("Batch Size")
            if row == 0 and col == 2:
                ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    _save(fig, "fig2_arch_comparison_even")


# ── Figure 3: Distributed strategy ────────────────────────────────────────
def fig_parallel_strategy(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    strategy_cols = ["tp", "ep", "dp", "pp"]
    strat_colors = {"tp": "#e74c3c", "ep": "#3498db", "dp": "#2ecc71", "pp": "#f39c12"}

    for ax, df, title in [(axes[0], pf, "Prefill"), (axes[1], dc, "Decode")]:
        bp = find_best_per_bs(df, "effective_stps")
        be = find_best_per_bs(df, "tokens_per_joule")
        bs_vals = sorted(df["bs"].unique())

        y_labels = []
        y_pos = []
        idx = 0
        for bi, bs in enumerate(bs_vals):
            y_labels.extend([f"bs={bs} perf", f"bs={bs} eff"])
            y_pos.extend([idx, idx + 1])
            idx += 2.5

        y_pos = np.array(y_pos)

        for si, strat in enumerate(strategy_cols):
            perf_vals = np.log2(bp[strat].values.astype(float) + 1)
            eff_vals = np.log2(be[strat].values.astype(float) + 1)
            interleaved = []
            for p, e in zip(perf_vals, eff_vals):
                interleaved.extend([p, e])
            left = np.zeros(len(y_pos))
            if si > 0:
                for sj in range(si):
                    prev_strat = strategy_cols[sj]
                    pv = np.log2(bp[prev_strat].values.astype(float) + 1)
                    ev = np.log2(be[prev_strat].values.astype(float) + 1)
                    il = []
                    for p, e in zip(pv, ev):
                        il.extend([p, e])
                    left += np.array(il)
            ax.barh(y_pos, interleaved, left=left, height=0.8,
                    color=strat_colors[strat], edgecolor="white", linewidth=0.3,
                    label=strat.upper())

        for bi, bs in enumerate(bs_vals):
            bp_row = bp[bp["bs"] == bs].iloc[0]
            be_row = be[be["bs"] == bs].iloc[0]
            perf_str = f"tp{int(bp_row.tp)}/ep{int(bp_row.ep)}/dp{int(bp_row.dp)}/pp{int(bp_row.pp)}"
            eff_str = f"tp{int(be_row.tp)}/ep{int(be_row.ep)}/dp{int(be_row.dp)}/pp{int(be_row.pp)}"
            total_perf = sum(np.log2(bp_row[s] + 1) for s in strategy_cols)
            total_eff = sum(np.log2(be_row[s] + 1) for s in strategy_cols)
            ax.text(total_perf + 0.3, y_pos[bi*2], perf_str, va="center", fontsize=5.5)
            ax.text(total_eff + 0.3, y_pos[bi*2+1], eff_str, va="center", fontsize=5.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=6.5)
        ax.set_xlabel("log2(parallelism + 1)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=7)
        ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, "fig3_parallel_strategy_even")


# ── Figure 4: Tokens/J box plot by mn ──────────────────────────────────────
def fig_tokj_by_mn(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    for ax, df, title in [(axes[0], pf, "Prefill"), (axes[1], dc, "Decode")]:
        max_bs = df["bs"].max()
        sub = df[df["bs"] == max_bs].copy()
        mn_labels = sorted(sub["mn_label"].unique(),
                           key=lambda x: (int(x.split(",")[0].split("=")[1]),
                                          int(x.split(",")[1].split("=")[1])))
        data = [sub[sub["mn_label"] == mn]["tokens_per_joule"].values for mn in mn_labels]

        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                        medianprops=dict(color="red", linewidth=1.2))
        for i, (patch, mn) in enumerate(zip(bp["boxes"], mn_labels)):
            m = int(mn.split(",")[0].split("=")[1])
            n = int(mn.split(",")[1].split("=")[1])
            ratio = n / m
            patch.set_facecolor(plt.cm.viridis(ratio))
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(mn_labels) + 1))
        ax.set_xticklabels(mn_labels, rotation=70, ha="right", fontsize=5.5)
        ax.set_ylabel("Energy Efficiency (tokens/J)")
        ax.set_title(f"{title} (bs={max_bs})")

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("n/m ratio", fontsize=7)

    fig.tight_layout()
    _save(fig, "fig4_tokj_by_mn_even")


# ── Figure 5: mn heatmap — fixed axes 2,4,6,...,14 ────────────────────────
def fig_mn_heatmap(pf, dc):
    """
    Heatmap with fixed even axes: m,n in {2,4,6,8,10,12,14}.
    2x2: (prefill stps, prefill tok/J) / (decode stps, decode tok/J)
    using the largest bs.
    """
    even_vals = EVEN_M_VALUES
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    for row, (df, phase) in enumerate([(pf, "Prefill"), (dc, "Decode")]):
        max_bs = df["bs"].max()
        sub = df[df["bs"] == max_bs]
        m_col, n_col = "dram_total_layers", "dram_active_layers"

        for col, (metric, mlabel) in enumerate([
            ("effective_stps", "Best Throughput (stps)"),
            ("tokens_per_joule", "Best Efficiency (tok/J)"),
        ]):
            ax = axes[row, col]
            pivot = sub.groupby([m_col, n_col])[metric].max().reset_index()
            pt = pivot.pivot(index=m_col, columns=n_col, values=metric)
            # Reindex to fixed even values
            pt = pt.reindex(index=even_vals, columns=even_vals)

            im = ax.imshow(pt.values, aspect="auto", cmap="YlOrRd", origin="lower")
            ax.set_xticks(range(len(even_vals)))
            ax.set_xticklabels(even_vals, fontsize=8)
            ax.set_yticks(range(len(even_vals)))
            ax.set_yticklabels(even_vals, fontsize=8)
            ax.set_xlabel("Active layers (n)")
            ax.set_ylabel("Total layers (m)")
            ax.set_title(f"{phase} bs={max_bs}\n{mlabel}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate cells
            thresh = np.nanmean(pt.values)
            for yi in range(len(even_vals)):
                for xi in range(len(even_vals)):
                    val = pt.values[yi, xi]
                    if not np.isnan(val):
                        fmt = f"{val:.0f}" if val > 100 else f"{val:.3f}"
                        color = "white" if val > thresh else "black"
                        ax.text(xi, yi, fmt, ha="center", va="center", fontsize=6, color=color)

    fig.tight_layout()
    _save(fig, "fig5_mn_heatmap_even")


# ── Figure 5b: mn heatmap per bs ──────────────────────────────────────────
def fig_mn_heatmap_per_bs(pf, dc):
    """
    Heatmap of best stps and tokens/J per (m,n) for each bs value.
    Even axes only: 2,4,6,8,10,12,14.
    """
    even_vals = EVEN_M_VALUES

    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        bs_vals = sorted(df["bs"].unique())
        m_col, n_col = "dram_total_layers", "dram_active_layers"

        for metric, mlabel, suffix in [
            ("effective_stps", "Best stps", "stps"),
            ("tokens_per_joule", "Best tokens/J", "tokj"),
        ]:
            n_bs = len(bs_vals)
            ncols = min(n_bs, 4)
            nrows = (n_bs + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)

            for i, bs in enumerate(bs_vals):
                ax = axes[i // ncols][i % ncols]
                bs_sub = df[df["bs"] == bs]
                pivot = bs_sub.groupby([m_col, n_col])[metric].max().reset_index()
                pt = pivot.pivot(index=m_col, columns=n_col, values=metric)
                pt = pt.reindex(index=even_vals, columns=even_vals)

                im = ax.imshow(pt.values, aspect="auto", cmap="YlOrRd", origin="lower")
                ax.set_xticks(range(len(even_vals)))
                ax.set_xticklabels(even_vals, fontsize=7)
                ax.set_yticks(range(len(even_vals)))
                ax.set_yticklabels(even_vals, fontsize=7)
                ax.set_xlabel("n (active layers)")
                ax.set_ylabel("m (total layers)")
                ax.set_title(f"bs={bs}", fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                thresh = np.nanmean(pt.values[~np.isnan(pt.values)]) if np.any(~np.isnan(pt.values)) else 0
                for yi in range(len(even_vals)):
                    for xi in range(len(even_vals)):
                        val = pt.values[yi, xi]
                        if not np.isnan(val):
                            fmt = f"{val:.0f}" if val > 100 else f"{val:.4f}"
                            color = "white" if val > thresh else "black"
                            ax.text(xi, yi, fmt, ha="center", va="center", fontsize=5.5, color=color)

            for i in range(n_bs, nrows * ncols):
                axes[i // ncols][i % ncols].set_visible(False)

            fig.suptitle(f"{phase_name} — {mlabel} by (m, n), even only", fontsize=13, y=1.01)
            fig.tight_layout()
            _save(fig, f"fig5b_{prefix}_mn_heatmap_{suffix}_even")


# ── Figure 6: CDF ─────────────────────────────────────────────────────────
def fig_tokj_cdf(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for ax, df, title in [(axes[0], pf, "Prefill"), (axes[1], dc, "Decode")]:
        for bs in sorted(df["bs"].unique()):
            vals = df[df["bs"] == bs]["tokens_per_joule"].dropna().sort_values()
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals / vals.max(), cdf, label=f"bs={bs}",
                    color=BS_COLORS.get(bs, "#333"), linewidth=1.0)
        ax.set_xlabel("tokens/J (normalized to max per bs)")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.legend(fontsize=6, loc="lower right")
        ax.set_xlim(0, 1.05)

    fig.tight_layout()
    _save(fig, "fig6_tokj_cdf_even")


# ── Figure 7: DDR BW impact ───────────────────────────────────────────────
def fig_ddr_bw_impact(pf, dc):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0))

    for row, (df, phase) in enumerate([(pf, "Prefill"), (dc, "Decode")]):
        for col, (metric, ylabel) in enumerate([
            ("effective_stps", "Best Throughput (stps)"),
            ("tokens_per_joule", "Best Efficiency (tok/J)"),
        ]):
            ax = axes[row, col]
            for bs in sorted(df["bs"].unique()):
                sub = df[df["bs"] == bs]
                best = sub.groupby("ddr_peak_bw_TBs")[metric].max().reset_index()
                ax.plot(best["ddr_peak_bw_TBs"], best[metric], marker="o", markersize=3,
                        label=f"bs={bs}", color=BS_COLORS.get(bs, "#333"), linewidth=1.0)
            ax.set_xlabel("DDR Peak BW (TB/s)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{phase}")
            if row == 0 and col == 1:
                ax.legend(fontsize=5.5, loc="best")

    fig.tight_layout()
    _save(fig, "fig7_ddr_bw_impact_even")


# ── Summary Table ─────────────────────────────────────────────────────────
def fig_summary_table(pf, dc):
    meta_cols = ["Phase", "BS", "Objective"]
    perf_cols = ["effective_stps", "tokens_per_joule"]
    dram_cols = ["dram_total_layers", "dram_active_layers"]
    hw_cols = [
        "sm_count", "smem_capacity_KiB",
        "l1_throughput_Bpc",
        "ddr_peak_bw_TBs", "ddr_eff_bw_TBs", "ddr_capacity_GB",
        "littles_law_required_buf_KiB", "littles_law_limited", "is_l1_bound",
    ]
    par_cols = ["tp", "ep", "dp", "pp", "sp", "cp", "fsdp"]
    power_cols = [
        "total_power_per_device_W", "chip_power_W", "noc_power_per_device_W",
        "effective_power_W", "system_power_W",
        "hit_power_wall", "freq_scale_power",
        "thermal_resistance_CpW", "freq_scale_thermal", "total_freq_scale",
    ]
    other_cols = ["arch", "noc", "model", "minibatch", "seq"]
    src_cols = dram_cols + hw_cols + par_cols + power_cols + other_cols + perf_cols

    rows = []
    for df, phase in [(pf, "Prefill"), (dc, "Decode")]:
        bp = find_best_per_bs(df, "effective_stps")
        be = find_best_per_bs(df, "tokens_per_joule")

        for objective, best_df in [("Best Perf", bp), ("Best Tok/J", be)]:
            for _, r in best_df.iterrows():
                row = {"Phase": phase, "BS": int(r.bs), "Objective": objective}
                for c in src_cols:
                    if c in r.index:
                        val = r[c]
                        if isinstance(val, (np.integer, int)):
                            row[c] = int(val)
                        elif isinstance(val, (np.floating, float)):
                            row[c] = round(float(val), 6)
                        else:
                            row[c] = val
                rows.append(row)

    summary = pd.DataFrame(rows)
    col_order = meta_cols + [c for c in summary.columns if c not in meta_cols]
    summary = summary[col_order]

    print("\n" + "=" * 120)
    print("  BEST CONFIGS SUMMARY TABLE (even m,n only, m<=14)")
    print("=" * 120)
    display_cols = [
        "Phase", "BS", "Objective", "effective_stps", "tokens_per_joule",
        "dram_total_layers", "dram_active_layers", "sm_count",
        "l1_throughput_Bpc", "ddr_peak_bw_TBs", "ddr_capacity_GB",
        "tp", "ep", "dp", "pp",
        "total_power_per_device_W", "effective_power_W",
        "hit_power_wall", "total_freq_scale",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].to_string(index=False))

    csv_path = os.path.join(FIG_DIR, "best_configs_summary_even.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved full table ({len(summary.columns)} columns) to {csv_path}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("Loading data (even m,n only, m<=14)...")
    pf = load_and_process(
        "dram_layer_prefill_result_dpsk_mn1to12_seq1024.csv",
        "scaled_stps", "raw_stps")
    dc = load_and_process(
        "dram_layer_decode_result_dpsk_mn1to12.csv",
        "scaled_stps_avg", "raw_stps_avg")

    print(f"Prefill: {len(pf)} rows, bs={sorted(pf['bs'].unique())}")
    print(f"Decode:  {len(dc)} rows, bs={sorted(dc['bs'].unique())}")

    mn_pf = pf[["dram_total_layers", "dram_active_layers"]].drop_duplicates().sort_values(
        ["dram_total_layers", "dram_active_layers"])
    mn_dc = dc[["dram_total_layers", "dram_active_layers"]].drop_duplicates().sort_values(
        ["dram_total_layers", "dram_active_layers"])
    print(f"Prefill (m,n) combos: {len(mn_pf)}")
    for _, r in mn_pf.iterrows():
        print(f"  m={int(r['dram_total_layers']):>2}, n={int(r['dram_active_layers']):>2}")
    print(f"Decode (m,n) combos: {len(mn_dc)}")
    for _, r in mn_dc.iterrows():
        print(f"  m={int(r['dram_total_layers']):>2}, n={int(r['dram_active_layers']):>2}")

    summary = fig_summary_table(pf, dc)

    print("\nGenerating paper figures (even only)...")
    fig_pareto_scatter(pf, dc)
    fig_arch_comparison(pf, dc)
    fig_parallel_strategy(pf, dc)
    fig_tokj_by_mn(pf, dc)
    fig_mn_heatmap(pf, dc)
    fig_mn_heatmap_per_bs(pf, dc)
    fig_tokj_cdf(pf, dc)
    fig_ddr_bw_impact(pf, dc)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
