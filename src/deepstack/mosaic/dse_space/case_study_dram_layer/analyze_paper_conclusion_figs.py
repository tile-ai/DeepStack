#!/usr/bin/env python3
"""
Targeted figures for two key conclusions:
  C1: bs=4 vs bs=1024, prefill vs decode → different optimal (m,n)
  C2: Best throughput vs best tok/J → different optimal (m,n)

Multiple layout ideas for MICRO/ASPLOS.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import griddata
import os

POWER_CAP = 100.0
NUM_DEVICES = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures_conclusion")
DATA_DIR = os.path.join(BASE_DIR, "lfs")
os.makedirs(FIG_DIR, exist_ok=True)

EVEN_M_VALUES = [2, 4, 6, 8, 10, 12, 14]

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
    """Get best (m,n) per (bs, metric) and the full heatmap pivot."""
    sub = df[df["bs"] == bs]
    best_per_mn = sub.groupby(["dram_total_layers", "dram_active_layers"])[metric].max().reset_index()
    pt = best_per_mn.pivot(index="dram_total_layers", columns="dram_active_layers", values=metric)
    pt = pt.reindex(index=EVEN_M_VALUES, columns=EVEN_M_VALUES)
    best_row = best_per_mn.loc[best_per_mn[metric].idxmax()]
    return pt, int(best_row["dram_total_layers"]), int(best_row["dram_active_layers"]), best_row[metric]


# ======================================================================
# PLAN A: 2×2 heatmap — rows=bs(4,1024), cols=Prefill/Decode
#   Each cell has TWO markers: ★=best throughput, ◆=best tok/J
#   → One figure covers BOTH conclusions
# ======================================================================
def plan_a_heatmap_2x2(pf, dc):
    """2×2 heatmap: (bs=4, bs=1024) × (Prefill, Decode).
    Each panel marks ★=best stps, ◆=best tok/J."""
    even = EVEN_M_VALUES
    bs_list = [4, 1024]

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.8))

    for row, bs in enumerate(bs_list):
        for col, (df, phase) in enumerate([(pf, "Prefill"), (dc, "Decode")]):
            ax = axes[row][col]

            # Throughput heatmap as background
            pt_stps, bm_s, bn_s, bv_s = get_best_mn(df, bs, "effective_stps")
            pt_tokj, bm_j, bn_j, bv_j = get_best_mn(df, bs, "tokens_per_joule")

            # Normalize throughput for color
            vals = pt_stps.values
            im = ax.imshow(vals, aspect="auto", cmap="YlOrRd", origin="lower")

            ax.set_xticks(range(len(even)))
            ax.set_xticklabels(even)
            ax.set_yticks(range(len(even)))
            ax.set_yticklabels(even)

            # Annotate cell values (compact)
            vmax = np.nanmax(vals)
            vmin = np.nanmin(vals)
            thresh = (vmax + vmin) / 2
            for yi in range(len(even)):
                for xi in range(len(even)):
                    v = vals[yi, xi]
                    if not np.isnan(v):
                        if v >= 1e6:
                            txt = f"{v/1e6:.1f}M"
                        elif v >= 1e4:
                            txt = f"{v/1e3:.0f}K"
                        elif v >= 1e3:
                            txt = f"{v/1e3:.1f}K"
                        else:
                            txt = f"{v:.0f}"
                        color = "white" if v > thresh else "black"
                        ax.text(xi, yi, txt, ha="center", va="center",
                                fontsize=5.5, color=color)

            # Mark best throughput: blue star
            bm_s_idx = even.index(bm_s)
            bn_s_idx = even.index(bn_s)
            ax.scatter(bn_s_idx, bm_s_idx, marker="*", s=280, c="#2196F3",
                       edgecolors="white", linewidths=1.2, zorder=10)

            # Mark best tok/J: green diamond
            bm_j_idx = even.index(bm_j)
            bn_j_idx = even.index(bn_j)
            ax.scatter(bn_j_idx, bm_j_idx, marker="D", s=100, c="#4CAF50",
                       edgecolors="white", linewidths=1.2, zorder=10)

            # Labels
            if row == 1:
                ax.set_xlabel("Active layers (n)")
            if col == 0:
                ax.set_ylabel(f"bs={bs}\nTotal layers (m)", fontweight="bold")
            if row == 0:
                ax.set_title(phase, fontweight="bold", fontsize=12)

            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         format=lambda x, _: f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}")

    # Legend
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2196F3",
               markeredgecolor="white", markersize=14, label="Best Throughput (m,n)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#4CAF50",
               markeredgecolor="white", markersize=8, label="Best Tokens/J (m,n)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=9)

    fig.suptitle("Optimal DRAM Layer Configuration Varies by\nWorkload Phase, Batch Size, and Objective",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(h_pad=1.0)
    _save(fig, "planA_heatmap_2x2")


# ======================================================================
# PLAN B: 2×2 3D surface — same layout but 3D, dual markers
# ======================================================================
def plan_b_3d_surface_2x2(pf, dc):
    """2×2 3D surface: (bs=4, bs=1024) × (Prefill, Decode)."""
    bs_list = [4, 1024]

    fig = plt.figure(figsize=(10, 9))

    for row, bs in enumerate(bs_list):
        for col, (df, phase, cmap_name) in enumerate([
            (pf, "Prefill", "YlOrRd"), (dc, "Decode", "YlGnBu")
        ]):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"]).agg(
                stps=("effective_stps", "max"),
                tokj=("tokens_per_joule", "max"),
            ).reset_index()

            m_pts = best["dram_total_layers"].values.astype(float)
            n_pts = best["dram_active_layers"].values.astype(float)
            z_pts = best["stps"].values

            m_fine = np.linspace(2, 14, 40)
            n_fine = np.linspace(2, 12, 35)
            M, N = np.meshgrid(m_fine, n_fine)

            try:
                Z = griddata((m_pts, n_pts), z_pts, (M, N), method="cubic")
                Z[N > M] = np.nan
                ax.plot_surface(M, N, Z, cmap=cmap_name, alpha=0.8,
                                edgecolor="gray", linewidth=0.15,
                                rstride=2, cstride=2, antialiased=True)
            except Exception:
                ax.scatter(m_pts, n_pts, z_pts, c=z_pts, cmap=cmap_name, s=30)

            # Best throughput: blue star
            best_s = best.loc[best["stps"].idxmax()]
            ax.scatter([best_s["dram_total_layers"]], [best_s["dram_active_layers"]],
                       [best_s["stps"]], s=250, c="#2196F3", marker="*",
                       edgecolors="white", linewidths=1.2, zorder=10, depthshade=False)
            ax.text(best_s["dram_total_layers"], best_s["dram_active_layers"],
                    best_s["stps"] * 1.06,
                    f"Perf({int(best_s['dram_total_layers'])},{int(best_s['dram_active_layers'])})",
                    fontsize=7, fontweight="bold", color="#1565C0")

            # Best tok/J: green diamond
            # Need to find best tok/J from full data
            best_j_data = sub.groupby(["dram_total_layers", "dram_active_layers"])["tokens_per_joule"].max().reset_index()
            best_j_row = best_j_data.loc[best_j_data["tokens_per_joule"].idxmax()]
            bm_j = best_j_row["dram_total_layers"]
            bn_j = best_j_row["dram_active_layers"]
            # Get the throughput at this (m,n) for z position
            z_at_j = best[(best["dram_total_layers"] == bm_j) & (best["dram_active_layers"] == bn_j)]["stps"].values
            z_j = z_at_j[0] if len(z_at_j) > 0 else best_s["stps"] * 0.5
            ax.scatter([bm_j], [bn_j], [z_j], s=120, c="#4CAF50", marker="D",
                       edgecolors="white", linewidths=1.2, zorder=10, depthshade=False)
            ax.text(bm_j, bn_j, z_j * 1.06,
                    f"Eff({int(bm_j)},{int(bn_j)})",
                    fontsize=7, fontweight="bold", color="#2E7D32")

            ax.set_xlabel("m", fontsize=9, labelpad=2)
            ax.set_ylabel("n", fontsize=9, labelpad=2)
            ax.set_zlabel("tok/s", fontsize=9, labelpad=2)
            ax.set_title(f"{phase}, bs={bs}", fontweight="bold", fontsize=11)
            ax.view_init(elev=25, azim=-135)
            ax.tick_params(labelsize=7)

    fig.suptitle("Throughput Landscape: Optimal (m,n) Differs by Phase, BS, and Objective",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    _save(fig, "planB_3d_surface_2x2")


# ======================================================================
# PLAN C: Compact summary figure — "arrow diagram"
#   Left: 4 mini-heatmaps (2×2). Right: summary table/diagram.
# ======================================================================
def plan_c_summary_with_table(pf, dc):
    """Left: 2×2 mini heatmaps. Right: summary comparison table."""
    even = EVEN_M_VALUES
    bs_list = [4, 1024]

    fig = plt.figure(figsize=(11, 5.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.35)

    configs = {}  # store for summary

    for row, bs in enumerate(bs_list):
        for col, (df, phase) in enumerate([(pf, "Prefill"), (dc, "Decode")]):
            ax = fig.add_subplot(gs[row, col])

            pt_stps, bm_s, bn_s, bv_s = get_best_mn(df, bs, "effective_stps")
            pt_tokj, bm_j, bn_j, bv_j = get_best_mn(df, bs, "tokens_per_joule")

            configs[(phase, bs, "perf")] = (bm_s, bn_s)
            configs[(phase, bs, "eff")] = (bm_j, bn_j)

            vals = pt_stps.values
            im = ax.imshow(vals, aspect="auto", cmap="YlOrRd", origin="lower")
            ax.set_xticks(range(len(even)))
            ax.set_xticklabels(even, fontsize=7)
            ax.set_yticks(range(len(even)))
            ax.set_yticklabels(even, fontsize=7)

            # Annotate
            vmax, vmin = np.nanmax(vals), np.nanmin(vals)
            thresh = (vmax + vmin) / 2
            for yi in range(len(even)):
                for xi in range(len(even)):
                    v = vals[yi, xi]
                    if not np.isnan(v):
                        if v >= 1e4:
                            txt = f"{v/1e3:.0f}K"
                        elif v >= 1e3:
                            txt = f"{v/1e3:.1f}K"
                        else:
                            txt = f"{v:.0f}"
                        color = "white" if v > thresh else "black"
                        ax.text(xi, yi, txt, ha="center", va="center",
                                fontsize=5, color=color)

            # Markers
            ax.scatter(even.index(bn_s), even.index(bm_s), marker="*", s=250,
                       c="#2196F3", edgecolors="white", linewidths=1.2, zorder=10)
            ax.scatter(even.index(bn_j), even.index(bm_j), marker="D", s=80,
                       c="#4CAF50", edgecolors="white", linewidths=1.2, zorder=10)

            if row == 1:
                ax.set_xlabel("Active (n)", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"bs={bs}\nTotal (m)", fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(phase, fontweight="bold", fontsize=11)

    # Right panel: summary table
    ax_table = fig.add_subplot(gs[:, 2])
    ax_table.axis("off")

    # Build table data
    table_data = []
    row_labels = []

    for bs in bs_list:
        for phase in ["Prefill", "Decode"]:
            pm, pn = configs[(phase, bs, "perf")]
            em, en = configs[(phase, bs, "eff")]
            table_data.append([
                f"({pm}, {pn})",
                f"({em}, {en})",
                "Same" if (pm == em and pn == en) else "Different"
            ])
            row_labels.append(f"{phase}\nbs={bs}")

    col_labels = ["Best Perf\n(m, n)", "Best Eff\n(m, n)", "Match?"]

    table = ax_table.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0.0, 0.15, 1.0, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Color the "Match?" column
    for i, row in enumerate(table_data):
        cell = table[i + 1, 2]
        if row[2] == "Same":
            cell.set_facecolor("#C8E6C9")
        else:
            cell.set_facecolor("#FFCDD2")
        # Color perf/eff columns
        table[i + 1, 0].set_facecolor("#BBDEFB")
        table[i + 1, 1].set_facecolor("#C8E6C9")

    # Header colors
    for j in range(3):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(fontweight="bold")

    # Title for table
    ax_table.set_title("Optimal Config Summary", fontweight="bold", fontsize=11, pad=10)

    # Conclusion annotations
    ax_table.text(0.5, 0.05,
                  "C1: Prefill vs Decode → different optimal layers\n"
                  "C2: Throughput vs Efficiency → different optimal layers",
                  ha="center", va="center", fontsize=9, fontstyle="italic",
                  transform=ax_table.transAxes,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#FBC02D"))

    # Legend
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2196F3",
               markeredgecolor="white", markersize=14, label="★ Best Throughput"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#4CAF50",
               markeredgecolor="white", markersize=8, label="◆ Best Tokens/J"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.35, -0.03), frameon=True, fontsize=9)

    _save(fig, "planC_summary_with_table")


# ======================================================================
# PLAN D: Single (m,n) grid with ALL 8 optimal points plotted
#   One clean plot showing how different scenarios land in different spots.
# ======================================================================
def plan_d_single_grid_all_optima(pf, dc):
    """One (m,n) grid with all 8 optimal points: 2 phases × 2 bs × 2 objectives."""
    even = EVEN_M_VALUES
    bs_list = [4, 1024]

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Draw the valid (m,n) grid
    for m in even:
        for n in even:
            if n <= m:
                ax.scatter(n, m, s=300, c="#f0f0f0", edgecolors="#cccccc",
                           linewidths=0.5, zorder=1)

    # n <= m diagonal shading
    ax.fill_between([0.5, 15], [0.5, 15], [15, 15], alpha=0.05, color="gray")
    ax.plot([0.5, 15], [0.5, 15], "k--", linewidth=0.6, alpha=0.4)
    ax.text(12.5, 11, "n=m", fontsize=7, color="gray", rotation=45)

    # Plot each optimal point
    markers = {
        ("Prefill", "perf"): ("*", 350),
        ("Prefill", "eff"):  ("D", 130),
        ("Decode", "perf"):  ("*", 350),
        ("Decode", "eff"):   ("D", 130),
    }
    colors = {
        ("Prefill", 4):    "#E53935",   # red
        ("Prefill", 1024): "#FF8A65",   # light red/orange
        ("Decode", 4):     "#1E88E5",   # blue
        ("Decode", 1024):  "#64B5F6",   # light blue
    }

    offsets = {}  # to slightly offset overlapping points
    offset_counter = {}

    points_info = []

    for phase, df in [("Prefill", pf), ("Decode", dc)]:
        for bs in bs_list:
            for obj, metric in [("perf", "effective_stps"), ("eff", "tokens_per_joule")]:
                _, bm, bn, bv = get_best_mn(df, bs, metric)
                marker, ms = markers[(phase, obj)]
                c = colors[(phase, bs)]

                # Offset for overlap
                key = (bm, bn)
                if key not in offset_counter:
                    offset_counter[key] = 0
                oc = offset_counter[key]
                offset_counter[key] += 1
                dx = (oc % 2) * 0.35 - 0.15
                dy = (oc // 2) * 0.35 - 0.15

                label = f"{phase} bs={bs} {'★Perf' if obj == 'perf' else '◆Eff'}"
                ax.scatter(bn + dx, bm + dy, marker=marker, s=ms, c=c,
                           edgecolors="black", linewidths=0.8, zorder=10,
                           label=label)
                points_info.append((phase, bs, obj, bm, bn))

    ax.set_xticks(even)
    ax.set_yticks(even)
    ax.set_xlabel("Active DRAM Layers (n)", fontsize=11)
    ax.set_ylabel("Total DRAM Layers (m)", fontsize=11)
    ax.set_xlim(0.5, 15)
    ax.set_ylim(0.5, 15)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # Legend — group by color and shape
    handles = []
    for phase, bs in [("Prefill", 4), ("Prefill", 1024), ("Decode", 4), ("Decode", 1024)]:
        c = colors[(phase, bs)]
        handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor=c,
                              markeredgecolor="black", markersize=12,
                              label=f"{phase} bs={bs} Perf"))
        handles.append(Line2D([0], [0], marker="D", color="w", markerfacecolor=c,
                              markeredgecolor="black", markersize=7,
                              label=f"{phase} bs={bs} Eff"))
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=2,
              framealpha=0.9, columnspacing=0.8)

    ax.set_title("Optimal (m, n) Varies by Phase, Batch Size, and Objective",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    _save(fig, "planD_single_grid_all_optima")


# ======================================================================
# PLAN E: Side-by-side radar/comparison — grouped bar chart style
#   x = 4 scenarios, y = optimal m and n, colored by perf vs eff
# ======================================================================
def plan_e_grouped_comparison(pf, dc):
    """Grouped bar chart comparing optimal m and n across scenarios."""
    bs_list = [4, 1024]

    scenarios = []
    opt_m_perf = []
    opt_n_perf = []
    opt_m_eff = []
    opt_n_eff = []

    for phase, df in [("Prefill", pf), ("Decode", dc)]:
        for bs in bs_list:
            _, bm_s, bn_s, _ = get_best_mn(df, bs, "effective_stps")
            _, bm_j, bn_j, _ = get_best_mn(df, bs, "tokens_per_joule")
            scenarios.append(f"{phase}\nbs={bs}")
            opt_m_perf.append(bm_s)
            opt_n_perf.append(bn_s)
            opt_m_eff.append(bm_j)
            opt_n_eff.append(bn_j)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    x = np.arange(len(scenarios))
    w = 0.35

    # Left: optimal m
    ax = axes[0]
    bars1 = ax.bar(x - w/2, opt_m_perf, w, label="Best Throughput",
                   color="#2196F3", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, opt_m_eff, w, label="Best Tokens/J",
                   color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("Optimal Total Layers (m)", fontsize=10)
    ax.set_title("(a) Optimal Stacking Depth", fontweight="bold")
    ax.set_ylim(0, 16)
    ax.legend(fontsize=8)
    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{int(bar.get_height())}", ha="center", fontsize=8, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{int(bar.get_height())}", ha="center", fontsize=8, fontweight="bold")

    # Right: optimal n
    ax = axes[1]
    bars1 = ax.bar(x - w/2, opt_n_perf, w, label="Best Throughput",
                   color="#2196F3", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, opt_n_eff, w, label="Best Tokens/J",
                   color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("Optimal Active Layers (n)", fontsize=10)
    ax.set_title("(b) Optimal Active Connections", fontweight="bold")
    ax.set_ylim(0, 16)
    ax.legend(fontsize=8)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{int(bar.get_height())}", ha="center", fontsize=8, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{int(bar.get_height())}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Optimal DRAM Configuration Depends on Workload and Objective",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "planE_grouped_comparison")


# ======================================================================
# PLAN F: The "money figure" — heatmap + annotation + takeaway
#   2×2 heatmap with connected arrows showing how optimal shifts.
#   Annotations that directly spell out the conclusions.
# ======================================================================
def plan_f_money_figure(pf, dc):
    """The polished paper figure combining both conclusions."""
    even = EVEN_M_VALUES
    bs_list = [4, 1024]
    phase_data = [("Prefill", pf), ("Decode", dc)]

    fig = plt.figure(figsize=(7.5, 7.0))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.08], hspace=0.3, wspace=0.3)

    all_optima = {}

    for row, bs in enumerate(bs_list):
        for col, (phase, df) in enumerate(phase_data):
            ax = fig.add_subplot(gs[row, col])

            pt_stps, bm_s, bn_s, bv_s = get_best_mn(df, bs, "effective_stps")
            pt_tokj, bm_j, bn_j, bv_j = get_best_mn(df, bs, "tokens_per_joule")

            all_optima[(phase, bs)] = {
                "perf": (bm_s, bn_s, bv_s),
                "eff": (bm_j, bn_j, bv_j),
            }

            # Compute speedup relative to (2,2) for normalized comparison
            vals = pt_stps.values
            baseline = vals[0, 0] if not np.isnan(vals[0, 0]) else np.nanmin(vals)
            speedup = vals / baseline

            im = ax.imshow(speedup, aspect="auto", cmap="YlOrRd", origin="lower",
                           vmin=np.nanmin(speedup), vmax=np.nanmax(speedup))

            ax.set_xticks(range(len(even)))
            ax.set_xticklabels(even)
            ax.set_yticks(range(len(even)))
            ax.set_yticklabels(even)

            # Annotate speedup values
            smax, smin = np.nanmax(speedup), np.nanmin(speedup)
            sthresh = (smax + smin) / 2
            for yi in range(len(even)):
                for xi in range(len(even)):
                    v = speedup[yi, xi]
                    if not np.isnan(v):
                        color = "white" if v > sthresh else "black"
                        ax.text(xi, yi, f"{v:.1f}×", ha="center", va="center",
                                fontsize=5.5, color=color)

            # Mark best perf: blue star
            bm_s_idx, bn_s_idx = even.index(bm_s), even.index(bn_s)
            ax.scatter(bn_s_idx, bm_s_idx, marker="*", s=300, c="#2196F3",
                       edgecolors="black", linewidths=1, zorder=10)
            ax.annotate(f"Perf\n({bm_s},{bn_s})",
                        xy=(bn_s_idx, bm_s_idx),
                        xytext=(bn_s_idx + 1.5, bm_s_idx + 1.2),
                        fontsize=6.5, fontweight="bold", color="#1565C0",
                        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1),
                        zorder=11)

            # Mark best eff: green diamond
            bm_j_idx, bn_j_idx = even.index(bm_j), even.index(bn_j)
            ax.scatter(bn_j_idx, bm_j_idx, marker="D", s=120, c="#4CAF50",
                       edgecolors="black", linewidths=1, zorder=10)
            # Offset annotation to avoid overlap
            ann_dx = -1.5 if bn_j_idx > 2 else 1.5
            ann_dy = -1.2 if bm_j_idx > 2 else 1.2
            ax.annotate(f"Eff\n({bm_j},{bn_j})",
                        xy=(bn_j_idx, bm_j_idx),
                        xytext=(bn_j_idx + ann_dx, bm_j_idx + ann_dy),
                        fontsize=6.5, fontweight="bold", color="#2E7D32",
                        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1),
                        zorder=11)

            # Panel label
            ax.text(0.02, 0.96, f"({chr(97 + row*2 + col)})",
                    transform=ax.transAxes, fontsize=10, fontweight="bold",
                    va="top", ha="left")

            if row == 1:
                ax.set_xlabel("Active layers (n)")
            if col == 0:
                ax.set_ylabel("Total layers (m)")

            # Combined title
            ax.set_title(f"{phase}, bs={bs}", fontweight="bold", fontsize=11)

    # Shared colorbar at bottom
    cbar_ax = fig.add_subplot(gs[2, :])
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0.5, 2.5))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Speedup over (m=2, n=2)", fontsize=10)

    # Legend
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2196F3",
               markeredgecolor="black", markersize=14,
               label="★ Best Throughput"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#4CAF50",
               markeredgecolor="black", markersize=8,
               label="◆ Best Tokens/J"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), frameon=True, fontsize=9)

    _save(fig, "planF_money_figure")

    # Print the conclusions
    print("\n  === Conclusion Summary ===")
    for (phase, bs), opts in all_optima.items():
        pm, pn, pv = opts["perf"]
        em, en, ev = opts["eff"]
        print(f"  {phase:>7s} bs={bs:>5d}: "
              f"Best Perf → (m={pm}, n={pn}), "
              f"Best Eff → (m={em}, n={en})  "
              f"{'✓ SAME' if pm==em and pn==en else '✗ DIFFERENT'}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    pf = load_and_process(
        "dram_layer_prefill_result_dpsk_mn1to12_seq1024.csv",
        "scaled_stps", "raw_stps")
    dc = load_and_process(
        "dram_layer_decode_result_dpsk_mn1to12.csv",
        "scaled_stps_avg", "raw_stps_avg")
    print(f"Prefill: {len(pf)} rows | Decode: {len(dc)} rows")

    print("\n=== Generating conclusion figures ===")

    print("\nPlan A: 2×2 heatmap with dual markers")
    plan_a_heatmap_2x2(pf, dc)

    print("\nPlan B: 2×2 3D surface with dual markers")
    plan_b_3d_surface_2x2(pf, dc)

    print("\nPlan C: Heatmaps + summary table")
    plan_c_summary_with_table(pf, dc)

    print("\nPlan D: Single grid with all 8 optima")
    plan_d_single_grid_all_optima(pf, dc)

    print("\nPlan E: Grouped bar comparison")
    plan_e_grouped_comparison(pf, dc)

    print("\nPlan F: Money figure (polished heatmap + annotations)")
    plan_f_money_figure(pf, dc)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
