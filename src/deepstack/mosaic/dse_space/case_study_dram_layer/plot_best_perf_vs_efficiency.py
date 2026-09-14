#!/usr/bin/env python3
"""
Dual-metric figures v5 — only Method 2 and Method 6.
  Method 2: v3 layout (rows=bs, cols=PF stps/PF tokJ/DC stps/DC tokJ),
            but squished shorter vertically.
  Method 6: up-down split, rows=Prefill/Decode, cols=bs4/bs1024, landscape.
  + bar charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
import os

POWER_CAP = 100.0
NUM_DEVICES = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures_best_perf_vs_efficiency")
DATA_DIR = os.path.join(BASE_DIR, "lfs")
os.makedirs(FIG_DIR, exist_ok=True)

EVEN_M_VALUES = [2, 4, 6, 8, 10, 12, 14]  # stacked (y-axis)
EVEN_N_VALUES = [2, 4, 6, 8, 10, 12]      # connected (x-axis)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
})

CMAP_STPS = LinearSegmentedColormap.from_list(
    "stps_warm", ["#FFF8F0", "#FFE0B2", "#FFB74D", "#F4511E", "#B71C1C"])
CMAP_TOKJ = LinearSegmentedColormap.from_list(
    "tokj_teal", ["#F0F9F4", "#B2DFDB", "#4DB6AC", "#00897B", "#004D40"])
CMAP_STPS_SOFT = LinearSegmentedColormap.from_list(
    "stps_soft", ["#FFFBF5", "#FFECD2", "#FFD19A", "#FF9E6D", "#E65100"])
CMAP_TOKJ_SOFT = LinearSegmentedColormap.from_list(
    "tokj_soft", ["#F5FCFA", "#C8E6E0", "#80CBC4", "#26A69A", "#00695C"])

XLABEL = "Connected DRAM Layers"
YLABEL_PLAIN = "Stacked DRAM Layers"


def load_and_process(csv_name, scaled_col, raw_col):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
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
    return df


def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved {name}")


def get_best_mn(df, bs, metric):
    sub = df[df["bs"] == bs]
    best_per_mn = sub.groupby(["dram_total_layers", "dram_active_layers"])[metric].max().reset_index()
    pt = best_per_mn.pivot(index="dram_total_layers", columns="dram_active_layers", values=metric)
    pt = pt.reindex(index=EVEN_M_VALUES, columns=EVEN_N_VALUES)
    best_row = best_per_mn.loc[best_per_mn[metric].idxmax()]
    return pt, int(best_row["dram_total_layers"]), int(best_row["dram_active_layers"]), best_row[metric]


def fmt_val(v, metric):
    if metric == "effective_stps":
        if v >= 1e6:   return f"{v/1e6:.1f}M"
        elif v >= 1e4: return f"{v/1e3:.0f}K"
        elif v >= 1e3: return f"{v/1e3:.1f}K"
        else:          return f"{v:.0f}"
    else:
        if v >= 100:   return f"{v:.0f}"
        elif v >= 10:  return f"{v:.1f}"
        elif v >= 1:   return f"{v:.2f}"
        else:          return f"{v:.3f}"


def text_color_for_cmap(cmap, norm, value):
    rgba = cmap(norm(value))
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < 0.5 else "black"


# ======================================================================
# METHOD 2: v3 layout, squished vertically
#   rows = bs(4,1024), cols = PF stps | PF tokJ | DC stps | DC tokJ
#   Same as v3 but shorter figure height
# ======================================================================
def method2(pf, dc):
    even_m = EVEN_M_VALUES
    even_n = EVEN_N_VALUES
    bs_list = [4, 1024]

    # v3 was 15×7, squish to 15×5
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.0))

    metrics_info = [
        ("effective_stps", "System Throughput", CMAP_STPS),
        ("tokens_per_joule", "Tokens/J", CMAP_TOKJ),
    ]
    all_configs = {}

    for row, bs in enumerate(bs_list):
        for phase_idx, (df, phase) in enumerate([(pf, "Prefill"), (dc, "Decode")]):
            for met_idx, (metric, mlabel, cmap) in enumerate(metrics_info):
                col = phase_idx * 2 + met_idx
                ax = axes[row][col]

                pt, bm, bn, bv = get_best_mn(df, bs, metric)
                vals = pt.values
                all_configs[(phase, bs, metric)] = (bm, bn, bv)

                vmin_val, vmax_val = np.nanmin(vals), np.nanmax(vals)
                norm = Normalize(vmin=vmin_val, vmax=vmax_val)
                im = ax.imshow(vals, aspect="auto", cmap=cmap, origin="lower", norm=norm)

                bm_idx = even_m.index(bm)
                bn_idx = even_n.index(bn)

                for yi in range(len(even_m)):
                    for xi in range(len(even_n)):
                        v = vals[yi, xi]
                        if np.isnan(v): continue
                        tc = text_color_for_cmap(cmap, norm, v)
                        is_best = (yi == bm_idx and xi == bn_idx)
                        ax.text(xi, yi, fmt_val(v, metric),
                                ha="center", va="center",
                                fontsize=8 if is_best else 6.5,
                                color=tc,
                                fontweight="bold" if is_best else "normal",
                                zorder=6)

                # Best cell: white outline + colored border (double border for contrast)
                if metric == "effective_stps":
                    best_color = "#D32F2F"
                    best_label = "Best Throughput"
                else:
                    best_color = "#1565C0"
                    best_label = "Best Power Efficiency"
                # Outer white border for contrast against any background
                rect_w = mpatches.FancyBboxPatch(
                    (bn_idx - 0.49, bm_idx - 0.49), 0.98, 0.98,
                    boxstyle="round,pad=0.03",
                    facecolor="none", edgecolor="white", linewidth=4.5, zorder=7)
                ax.add_patch(rect_w)
                # Inner colored border
                rect = mpatches.FancyBboxPatch(
                    (bn_idx - 0.47, bm_idx - 0.47), 0.94, 0.94,
                    boxstyle="round,pad=0.03",
                    facecolor="none", edgecolor=best_color, linewidth=2.5, zorder=8)
                ax.add_patch(rect)

                # Label at bottom-right of panel
                ax.text(0.97, 0.03, best_label,
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6, fontweight="bold", color=best_color,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor=best_color, alpha=0.9, linewidth=0.8),
                        zorder=9)

                ax.set_xticks(range(len(even_n)))
                ax.set_xticklabels(even_n, fontsize=8)
                ax.set_yticks(range(len(even_m)))
                ax.set_yticklabels(even_m, fontsize=8)

                if row == 1:
                    ax.set_xlabel(XLABEL, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"$\\bf{{BS={bs}}}$\n{YLABEL_PLAIN}", fontsize=10)
                if row == 0:
                    ax.set_title(f"$\\bf{{{phase}}}$\n{mlabel}", fontsize=10)

                # Colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             format=lambda x, _: fmt_val(x, metric))


    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    _save(fig, "method2")
    return all_configs


# ======================================================================
# METHOD 6: Up-down split, landscape
#   rows=Prefill/Decode, cols=bs4/bs1024, colorbars on right
# ======================================================================
def method6(pf, dc):
    even_m = EVEN_M_VALUES
    even_n = EVEN_N_VALUES
    bs_list = [4, 1024]
    phases = [("Prefill", pf), ("Decode", dc)]

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.04],
                           hspace=0.25, wspace=0.25)

    all_configs = {}

    for row, (phase, df) in enumerate(phases):
        for col, bs in enumerate(bs_list):
            ax = fig.add_subplot(gs[row, col])

            pt_stps, bm_s, bn_s, bv_s = get_best_mn(df, bs, "effective_stps")
            pt_tokj, bm_j, bn_j, bv_j = get_best_mn(df, bs, "tokens_per_joule")
            all_configs[(phase, bs, "effective_stps")] = (bm_s, bn_s, bv_s)
            all_configs[(phase, bs, "tokens_per_joule")] = (bm_j, bn_j, bv_j)

            vals_s = pt_stps.values
            vals_j = pt_tokj.values
            s_norm = Normalize(vmin=np.nanmin(vals_s), vmax=np.nanmax(vals_s))
            j_norm = Normalize(vmin=np.nanmin(vals_j), vmax=np.nanmax(vals_j))

            for yi in range(len(even_m)):
                for xi in range(len(even_n)):
                    vs = vals_s[yi, xi]
                    vj = vals_j[yi, xi]
                    if np.isnan(vs) and np.isnan(vj):
                        continue
                    cx, cy = xi, yi

                    # Top half: stps
                    if not np.isnan(vs):
                        r = mpatches.FancyBboxPatch(
                            (cx-0.47, cy+0.01), 0.94, 0.46,
                            boxstyle="square,pad=0",
                            facecolor=CMAP_STPS_SOFT(s_norm(vs)),
                            edgecolor="none", zorder=1)
                        ax.add_patch(r)
                        tc = text_color_for_cmap(CMAP_STPS_SOFT, s_norm, vs)
                        ax.text(cx, cy+0.23, fmt_val(vs, "effective_stps"),
                                ha="center", va="center", fontsize=6,
                                color=tc, fontweight="bold", zorder=5)

                    # Bottom half: tok/J
                    if not np.isnan(vj):
                        r = mpatches.FancyBboxPatch(
                            (cx-0.47, cy-0.47), 0.94, 0.46,
                            boxstyle="square,pad=0",
                            facecolor=CMAP_TOKJ_SOFT(j_norm(vj)),
                            edgecolor="none", zorder=1)
                        ax.add_patch(r)
                        tc = text_color_for_cmap(CMAP_TOKJ_SOFT, j_norm, vj)
                        ax.text(cx, cy-0.23, fmt_val(vj, "tokens_per_joule"),
                                ha="center", va="center", fontsize=5.5,
                                color=tc, fontweight="bold", zorder=5)

                    # Divider
                    ax.plot([cx-0.47, cx+0.47], [cy, cy],
                            color="white", linewidth=0.3, alpha=0.3, zorder=3)

                    # Cell border
                    border = mpatches.FancyBboxPatch(
                        (cx-0.49, cy-0.49), 0.98, 0.98,
                        boxstyle="round,pad=0.01",
                        facecolor="none", edgecolor="#BDBDBD",
                        linewidth=0.4, zorder=4)
                    ax.add_patch(border)

            # Best throughput: red border + label
            bm_s_idx, bn_s_idx = even_m.index(bm_s), even_n.index(bn_s)
            rect_s = mpatches.FancyBboxPatch(
                (bn_s_idx-0.52, bm_s_idx-0.52), 1.04, 1.04,
                boxstyle="round,pad=0.03",
                facecolor="none", edgecolor="#D32F2F", linewidth=2.5, zorder=10)
            ax.add_patch(rect_s)

            # Best tok/J: blue border
            bm_j_idx, bn_j_idx = even_m.index(bm_j), even_n.index(bn_j)
            same = (bm_j_idx == bm_s_idx and bn_j_idx == bn_s_idx)
            rect_j = mpatches.FancyBboxPatch(
                (bn_j_idx-0.56, bm_j_idx-0.56), 1.12, 1.12,
                boxstyle="round,pad=0.03",
                facecolor="none", edgecolor="#1565C0", linewidth=2,
                linestyle="--" if same else "-", zorder=10)
            ax.add_patch(rect_j)

            ax.set_xlim(-0.55, len(even_n)-0.45)
            ax.set_ylim(-0.55, len(even_m)-0.45)
            ax.set_xticks(range(len(even_n)))
            ax.set_xticklabels(even_n, fontsize=7)
            ax.set_yticks(range(len(even_m)))
            ax.set_yticklabels(even_m, fontsize=7)
            ax.set_facecolor("#FAFAFA")

            if row == 1:
                ax.set_xlabel(XLABEL, fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"$\\bf{{{phase}}}$\n{YLABEL_PLAIN}", fontsize=9)
            if row == 0:
                ax.set_title(f"$\\bf{{BS={bs}}}$", fontsize=9)

    # Vertical colorbars — labels on the left side
    cax_s = fig.add_subplot(gs[0, 2])
    sm_s = plt.cm.ScalarMappable(cmap=CMAP_STPS_SOFT, norm=Normalize(0, 1))
    sm_s.set_array([])
    cb_s = plt.colorbar(sm_s, cax=cax_s, orientation="vertical")
    cb_s.set_ticks([0, 1]); cb_s.set_ticklabels(["Low", "High"], fontsize=7)
    cax_s.yaxis.set_ticks_position("left")
    cax_s.yaxis.set_label_position("left")
    cax_s.set_ylabel("System\nThroughput", fontsize=7, rotation=90,
                      labelpad=-5, va="center")

    cax_j = fig.add_subplot(gs[1, 2])
    sm_j = plt.cm.ScalarMappable(cmap=CMAP_TOKJ_SOFT, norm=Normalize(0, 1))
    sm_j.set_array([])
    cb_j = plt.colorbar(sm_j, cax=cax_j, orientation="vertical")
    cb_j.set_ticks([0, 1]); cb_j.set_ticklabels(["Low", "High"], fontsize=7)
    cax_j.yaxis.set_ticks_position("left")
    cax_j.yaxis.set_label_position("left")
    cax_j.set_ylabel("Tokens/J", fontsize=7, rotation=90,
                      labelpad=-5, va="center")

    # Compact legend at top — 4 items in one row
    handles = [
        mpatches.Patch(facecolor=CMAP_STPS_SOFT(0.6), edgecolor="#9E9E9E",
                       linewidth=0.5, label="System Throughput"),
        mpatches.Patch(facecolor=CMAP_TOKJ_SOFT(0.6), edgecolor="#9E9E9E",
                       linewidth=0.5, label="Tokens/J"),
        mpatches.FancyBboxPatch((0,0), 1, 1, boxstyle="round",
                                facecolor="none", edgecolor="#D32F2F",
                                linewidth=2, label="Best Throughput"),
        mpatches.FancyBboxPatch((0,0), 1, 1, boxstyle="round",
                                facecolor="none", edgecolor="#1565C0",
                                linewidth=2, label="Best Tokens/J"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.995), frameon=True, fontsize=7,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

    _save(fig, "method6_ud")
    return all_configs


# ======================================================================
# BAR CHARTS
# ======================================================================
def bar_chart_summary(all_configs):
    bs_list = [4, 1024]
    phases = ["Prefill", "Decode"]
    scenarios = [f"{p}\nBS={b}" for b in bs_list for p in phases]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax_idx, (metric, mlabel, c_m, c_n) in enumerate([
        ("effective_stps", "Throughput-Optimal", "#E65100", "#FFB74D"),
        ("tokens_per_joule", "Efficiency-Optimal", "#00695C", "#80CBC4"),
    ]):
        ax = axes[ax_idx]
        opt_m, opt_n = [], []
        for bs in bs_list:
            for phase in phases:
                bm, bn, _ = all_configs[(phase, bs, metric)]
                opt_m.append(bm); opt_n.append(bn)
        x = np.arange(len(scenarios)); w = 0.32
        bars_m = ax.bar(x-w/2, opt_m, w, label="Stacked Layers",
                        color=c_m, edgecolor="black", linewidth=0.4)
        bars_n = ax.bar(x+w/2, opt_n, w, label="Connected Layers",
                        color=c_n, edgecolor="black", linewidth=0.4)
        for bar in bars_m:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{int(bar.get_height())}", ha="center", fontsize=10,
                    fontweight="bold", color=c_m)
        for bar in bars_n:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{int(bar.get_height())}", ha="center", fontsize=10,
                    fontweight="bold", color=c_n)
        ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=9)
        ax.set_ylabel("Layer Count", fontsize=10)
        ax.set_title(f"({chr(97+ax_idx)}) {mlabel}", fontweight="bold", fontsize=11)
        ax.set_ylim(0, 14)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "bar_summary")


def bar_chart_perf_vs_eff(all_configs):
    bs_list = [4, 1024]
    phases = ["Prefill", "Decode"]
    scenarios = [f"{p}\nBS={b}" for b in bs_list for p in phases]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax_idx, (dim, dlabel) in enumerate([
        ("m", "Stacked DRAM Layers"), ("n", "Connected DRAM Layers"),
    ]):
        ax = axes[ax_idx]
        pv, ev = [], []
        for bs in bs_list:
            for phase in phases:
                pm, pn, _ = all_configs[(phase, bs, "effective_stps")]
                em, en, _ = all_configs[(phase, bs, "tokens_per_joule")]
                pv.append(pm if dim == "m" else pn)
                ev.append(em if dim == "m" else en)
        x = np.arange(len(scenarios)); w = 0.32
        ax.bar(x-w/2, pv, w, label="Throughput-optimal",
               color="#E65100", edgecolor="black", linewidth=0.4)
        ax.bar(x+w/2, ev, w, label="Efficiency-optimal",
               color="#00695C", edgecolor="black", linewidth=0.4)
        for i, v in enumerate(pv):
            ax.text(x[i]-w/2+w/2, v+0.3, f"{v}", ha="center", fontsize=10,
                    fontweight="bold", color="#E65100")
        for i, v in enumerate(ev):
            ax.text(x[i]+w/2, v+0.3, f"{v}", ha="center", fontsize=10,
                    fontweight="bold", color="#00695C")
        for i in range(len(scenarios)):
            if pv[i] != ev[i]:
                ymax = max(pv[i], ev[i]) + 1.6
                ax.annotate("diff", xy=(x[i], ymax), ha="center",
                            fontsize=8, color="#C62828", fontstyle="italic",
                            fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=9)
        ax.set_ylabel(dlabel, fontsize=10)
        ax.set_title(f"({chr(97+ax_idx)}) Optimal {dlabel}",
                     fontweight="bold", fontsize=11)
        ax.set_ylim(0, 14)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "bar_perf_vs_eff")


def main():
    print("Loading data...")
    pf = load_and_process(
        "dram_layer_prefill_result_dpsk_mn1to12_seq1024.csv",
        "scaled_stps", "raw_stps")
    dc = load_and_process(
        "dram_layer_decode_result_dpsk_mn1to12.csv",
        "scaled_stps_avg", "raw_stps_avg")
    print(f"Prefill: {len(pf)} rows | Decode: {len(dc)} rows")

    c2 = method2(pf, dc)
    c6 = method6(pf, dc)
    bar_chart_summary(c6)
    bar_chart_perf_vs_eff(c6)

    print(f"\nAll saved to: {FIG_DIR}")
    for bs in [4, 1024]:
        for phase in ["Prefill", "Decode"]:
            pm, pn, _ = c6[(phase, bs, "effective_stps")]
            em, en, _ = c6[(phase, bs, "tokens_per_joule")]
            match = "SAME" if (pm==em and pn==en) else "DIFF"
            print(f"  {phase:>7s} bs={bs:>5d}: Perf({pm},{pn}) Eff({em},{en}) [{match}]")


if __name__ == "__main__":
    main()
