#!/usr/bin/env python3
"""
New visualization ideas for DRAM layer DSE — targeting MICRO/ASPLOS.
Core question: for prefill/decode at each batch size, how many layers
to stack (m) and how many to connect (n)?

Even m,n only: 2, 4, 6, 8, 10, 12 (m up to 14).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
import os

# ── Constants ──────────────────────────────────────────────────────────────
POWER_CAP = 100.0
NUM_DEVICES = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures_ideas")
DATA_DIR = os.path.join(BASE_DIR, "lfs")
os.makedirs(FIG_DIR, exist_ok=True)

EVEN_M_VALUES = [2, 4, 6, 8, 10, 12, 14]

# ── MICRO/ASPLOS style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# Color palettes
BS_COLORS = {
    1: "#1f77b4", 4: "#ff7f0e", 16: "#2ca02c", 64: "#d62728",
    256: "#9467bd", 1024: "#8c564b", 4096: "#e377c2", 16384: "#7f7f7f",
}
# For n values
N_COLORS = {
    2: "#e41a1c", 4: "#377eb8", 6: "#4daf4a",
    8: "#984ea3", 10: "#ff7f00", 12: "#a65628",
}
N_MARKERS = {2: "o", 4: "s", 6: "^", 8: "D", 10: "v", 12: "P"}


# ── Data loading ──────────────────────────────────────────────────────────
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


# ======================================================================
# IDEA A: Line plots — Throughput vs m, lines colored by n, faceted by bs
#   Directly shows: "as I stack more layers, how does perf change?"
#   and "how much does connecting more layers help?"
# ======================================================================
def idea_a_lines_by_n(pf, dc):
    """
    Small multiples: one subplot per bs.
    x = m (total layers), y = best throughput across all other params.
    Each line = a different n (active layers), colored distinctly.
    Prefill and Decode as separate figures.
    """
    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        bs_vals = sorted(df["bs"].unique())
        ncols = min(len(bs_vals), 4)
        nrows = (len(bs_vals) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                                 sharey=False, squeeze=False)

        for i, bs in enumerate(bs_vals):
            ax = axes[i // ncols][i % ncols]
            sub = df[df["bs"] == bs]
            # Best throughput per (m, n)
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            n_vals = sorted(best["dram_active_layers"].unique())
            for n_val in n_vals:
                line_data = best[best["dram_active_layers"] == n_val].sort_values("dram_total_layers")
                ax.plot(line_data["dram_total_layers"], line_data["effective_stps"],
                        marker=N_MARKERS.get(n_val, "o"), markersize=4, linewidth=1.2,
                        color=N_COLORS.get(n_val, "#333"), label=f"n={n_val}")

            # Highlight the overall best point
            best_row = best.loc[best["effective_stps"].idxmax()]
            ax.scatter(best_row["dram_total_layers"], best_row["effective_stps"],
                       s=120, facecolors="none", edgecolors="red", linewidths=2, zorder=10)

            ax.set_xlabel("Total layers (m)")
            ax.set_ylabel("Throughput (tok/s)")
            ax.set_title(f"bs={bs}", fontweight="bold")
            ax.set_xticks([2, 4, 6, 8, 10, 12, 14])

        # Hide unused subplots
        for i in range(len(bs_vals), nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        # Shared legend
        handles = [Line2D([0], [0], marker=N_MARKERS.get(n, "o"), color=N_COLORS.get(n, "#333"),
                          markersize=5, linewidth=1.2, label=f"n={n}")
                   for n in sorted(N_COLORS.keys())]
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                              markeredgecolor="red", markeredgewidth=2, markersize=8, label="Best"))
        fig.legend(handles=handles, loc="lower center", ncol=7,
                   bbox_to_anchor=(0.5, -0.06), frameon=True)
        fig.suptitle(f"{phase_name}: Throughput vs. Total DRAM Layers",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, f"ideaA_{prefix}_lines_by_n")


# ======================================================================
# IDEA B: Speedup heatmap relative to (m=2, n=2) baseline
#   Shows diminishing returns clearly. Per-bs facets.
# ======================================================================
def idea_b_speedup_heatmap(pf, dc):
    """
    Heatmap of speedup over m=2,n=2 baseline.
    This normalizes across bs so reviewers can compare shapes.
    """
    even_vals = EVEN_M_VALUES
    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        bs_vals = sorted(df["bs"].unique())
        ncols = min(len(bs_vals), 4)
        nrows = (len(bs_vals) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.2 * nrows), squeeze=False)

        for i, bs in enumerate(bs_vals):
            ax = axes[i // ncols][i % ncols]
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()
            pt = best.pivot(index="dram_total_layers", columns="dram_active_layers", values="effective_stps")
            pt = pt.reindex(index=even_vals, columns=even_vals)

            # Baseline: m=2, n=2
            baseline = pt.loc[2, 2] if 2 in pt.index and 2 in pt.columns and not np.isnan(pt.loc[2, 2]) else np.nanmin(pt.values)
            speedup = pt / baseline

            im = ax.imshow(speedup.values, aspect="auto", cmap="RdYlGn", origin="lower",
                           vmin=0.5, vmax=np.nanmax(speedup.values))
            ax.set_xticks(range(len(even_vals)))
            ax.set_xticklabels(even_vals)
            ax.set_yticks(range(len(even_vals)))
            ax.set_yticklabels(even_vals)
            ax.set_xlabel("Active layers (n)")
            ax.set_ylabel("Total layers (m)")
            ax.set_title(f"bs={bs}", fontweight="bold")

            # Annotate
            for yi in range(len(even_vals)):
                for xi in range(len(even_vals)):
                    val = speedup.values[yi, xi]
                    if not np.isnan(val):
                        color = "white" if val > np.nanmean(speedup.values) * 1.2 else "black"
                        ax.text(xi, yi, f"{val:.1f}×", ha="center", va="center",
                                fontsize=6, fontweight="bold", color=color)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Speedup")

        for i in range(len(bs_vals), nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        fig.suptitle(f"{phase_name}: Speedup over (m=2, n=2) Baseline",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, f"ideaB_{prefix}_speedup_heatmap")


# ======================================================================
# IDEA C: Optimal frontier — best throughput as m increases (with best n)
#   Shows marginal benefit of stacking more layers.
#   One plot per phase, lines colored by bs.
# ======================================================================
def idea_c_optimal_frontier(pf, dc):
    """
    For each bs: as m increases, what is the best achievable throughput
    (optimizing over n and all other params)?
    Shows diminishing returns of stacking more layers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    for ax, df, title in [(axes[0], pf, "Prefill"), (axes[1], dc, "Decode")]:
        bs_vals = sorted(df["bs"].unique())
        for bs in bs_vals:
            sub = df[df["bs"] == bs]
            # Best throughput per m (optimizing over n and all other params)
            best_per_m = sub.groupby("dram_total_layers")["effective_stps"].max().reset_index()
            best_per_m = best_per_m.sort_values("dram_total_layers")
            # Normalize to m=2 for that bs
            baseline = best_per_m[best_per_m["dram_total_layers"] == 2]["effective_stps"].values
            if len(baseline) > 0:
                best_per_m["normalized"] = best_per_m["effective_stps"] / baseline[0]
            else:
                best_per_m["normalized"] = best_per_m["effective_stps"] / best_per_m["effective_stps"].iloc[0]

            ax.plot(best_per_m["dram_total_layers"], best_per_m["normalized"],
                    marker="o", markersize=4, linewidth=1.5,
                    color=BS_COLORS.get(bs, "#333"), label=f"bs={bs}")

        ax.set_xlabel("Total DRAM Layers (m)")
        ax.set_ylabel("Normalized Throughput\n(vs. m=2)")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=6, loc="best", ncol=2)

    fig.suptitle("Throughput Scaling with DRAM Layer Count (best n per m)",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "ideaC_optimal_frontier")


# ======================================================================
# IDEA D: Best (m,n) config summary — compact bar+annotation chart
#   For each bs, show the optimal m and n as grouped bars.
#   Prefill vs Decode side by side.
# ======================================================================
def idea_d_best_mn_summary(pf, dc):
    """
    For each bs: show the best (m, n) pair for throughput and efficiency.
    Compact grouped bar chart with (m, n) annotation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0))

    for row, (df, phase) in enumerate([(pf, "Prefill"), (dc, "Decode")]):
        for col, (metric, ylabel) in enumerate([
            ("effective_stps", "Best Throughput (tok/s)"),
            ("tokens_per_joule", "Best Efficiency (tok/J)"),
        ]):
            ax = axes[row, col]
            bs_vals = sorted(df["bs"].unique())
            x = np.arange(len(bs_vals))

            # Find best (m,n) per bs
            best_vals = []
            best_labels = []
            for bs in bs_vals:
                sub = df[df["bs"] == bs]
                best_idx = sub[metric].idxmax()
                best_row = sub.loc[best_idx]
                best_vals.append(best_row[metric])
                best_labels.append(f"({int(best_row['dram_total_layers'])},{int(best_row['dram_active_layers'])})")

            # Color bars by m value
            m_vals = [int(l.split(",")[0][1:]) for l in best_labels]
            norm = plt.Normalize(2, 14)
            colors = [plt.cm.viridis(norm(m)) for m in m_vals]

            bars = ax.bar(x, best_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.7)

            # Annotate with (m,n)
            for xi, (bar, label) in enumerate(zip(bars, best_labels)):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        label, ha="center", va="bottom", fontsize=7, fontweight="bold",
                        rotation=45)

            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in bs_vals], rotation=45, ha="right")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(ylabel, fontweight="bold")
            if col == 0:
                ax.annotate(phase, xy=(-0.3, 0.5), xycoords="axes fraction",
                            fontsize=11, fontweight="bold", rotation=90, va="center")

    # Colorbar for m
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(2, 14))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Total layers (m)")

    fig.suptitle("Optimal DRAM Configuration (m, n) per Batch Size",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "ideaD_best_mn_summary")


# ======================================================================
# IDEA E: Contour plot (smooth) — more elegant than discrete heatmap
#   Uses interpolation for smooth contours on (m, n) space.
# ======================================================================
def idea_e_contour(pf, dc):
    """
    Contour plot on (m, n) grid for throughput. One per bs per phase.
    More visually appealing than heatmap for architecture papers.
    """
    from scipy.interpolate import griddata

    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        bs_vals = sorted(df["bs"].unique())
        ncols = min(len(bs_vals), 4)
        nrows = (len(bs_vals) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows), squeeze=False)

        for i, bs in enumerate(bs_vals):
            ax = axes[i // ncols][i % ncols]
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            # Only valid points where n <= m
            m_pts = best["dram_total_layers"].values
            n_pts = best["dram_active_layers"].values
            z_pts = best["effective_stps"].values

            # Create fine grid
            m_fine = np.linspace(2, 14, 50)
            n_fine = np.linspace(2, 12, 50)
            M, N = np.meshgrid(m_fine, n_fine)

            try:
                Z = griddata((m_pts, n_pts), z_pts, (M, N), method="cubic")
                # Mask out n > m region
                Z[N > M] = np.nan

                cs = ax.contourf(M, N, Z, levels=12, cmap="YlOrRd", alpha=0.85)
                ax.contour(M, N, Z, levels=12, colors="black", linewidths=0.3, alpha=0.4)
                plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
            except Exception:
                # Fallback: scatter
                sc = ax.scatter(m_pts, n_pts, c=z_pts, cmap="YlOrRd", s=50, edgecolors="black", linewidths=0.5)
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

            # Mark optimal
            best_row = best.loc[best["effective_stps"].idxmax()]
            ax.scatter(best_row["dram_total_layers"], best_row["dram_active_layers"],
                       s=100, marker="*", c="blue", edgecolors="white", linewidths=1, zorder=10)
            ax.text(best_row["dram_total_layers"] + 0.3, best_row["dram_active_layers"] + 0.3,
                    f"({int(best_row['dram_total_layers'])},{int(best_row['dram_active_layers'])})",
                    fontsize=7, fontweight="bold", color="blue")

            # n <= m diagonal
            ax.plot([2, 14], [2, 14], "k--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Total layers (m)")
            ax.set_ylabel("Active layers (n)")
            ax.set_title(f"bs={bs}", fontweight="bold")

        for i in range(len(bs_vals), nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        fig.suptitle(f"{phase_name}: Throughput Landscape over (m, n)",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, f"ideaE_{prefix}_contour")


# ======================================================================
# IDEA F: Compact 2-row figure for paper — Prefill/Decode, selected bs
#   One figure with 2 rows (Prefill/Decode), columns = representative bs.
#   Heatmap with best config star. Paper-ready single figure.
# ======================================================================
def idea_f_compact_paper_figure(pf, dc):
    """
    A single compact figure suitable for a half-page in a paper.
    2 rows (Prefill, Decode) × 3-4 cols (representative bs values).
    Heatmap with optimal point marked.
    """
    even_vals = EVEN_M_VALUES

    # Pick representative bs values
    pf_bs = sorted(pf["bs"].unique())
    dc_bs = sorted(dc["bs"].unique())
    # Pick: small, medium, large bs for each
    def pick_representative(bs_list):
        if len(bs_list) <= 4:
            return bs_list
        return [bs_list[0], bs_list[len(bs_list)//3], bs_list[2*len(bs_list)//3], bs_list[-1]]

    pf_rep = pick_representative(pf_bs)
    dc_rep = pick_representative(dc_bs)
    # Unify: use the intersection or a smart selection
    all_rep = sorted(set(pf_rep) | set(dc_rep))
    if len(all_rep) > 5:
        all_rep = [1, 16, 256, 1024, 4096, 16384]
        all_rep = [b for b in all_rep if b in pf_bs or b in dc_bs]

    ncols = len(all_rep)
    fig, axes = plt.subplots(2, ncols, figsize=(2.8 * ncols, 5.0), squeeze=False)

    for row, (df, phase, valid_bs) in enumerate([
        (pf, "Prefill", pf_bs), (dc, "Decode", dc_bs)
    ]):
        for col, bs in enumerate(all_rep):
            ax = axes[row][col]
            if bs not in valid_bs:
                ax.set_visible(False)
                continue

            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()
            pt = best.pivot(index="dram_total_layers", columns="dram_active_layers", values="effective_stps")
            pt = pt.reindex(index=even_vals, columns=even_vals)

            # Normalize to max for visual comparison across bs
            vmax = np.nanmax(pt.values)
            vmin = np.nanmin(pt.values)

            im = ax.imshow(pt.values, aspect="auto", cmap="YlOrRd", origin="lower",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(even_vals)))
            ax.set_xticklabels(even_vals, fontsize=7)
            ax.set_yticks(range(len(even_vals)))
            ax.set_yticklabels(even_vals, fontsize=7)

            if row == 1:
                ax.set_xlabel("Active (n)", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{phase}\nTotal (m)", fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(f"bs={bs}", fontweight="bold", fontsize=10)

            # Mark best cell with a star
            best_overall = best.loc[best["effective_stps"].idxmax()]
            bm = int(best_overall["dram_total_layers"])
            bn = int(best_overall["dram_active_layers"])
            bm_idx = even_vals.index(bm) if bm in even_vals else -1
            bn_idx = even_vals.index(bn) if bn in even_vals else -1
            if bm_idx >= 0 and bn_idx >= 0:
                ax.scatter(bn_idx, bm_idx, marker="*", s=150, c="blue",
                           edgecolors="white", linewidths=0.8, zorder=10)

            # Annotate cells with values
            for yi in range(len(even_vals)):
                for xi in range(len(even_vals)):
                    val = pt.values[yi, xi]
                    if not np.isnan(val):
                        # Compact format
                        if val >= 1e6:
                            txt = f"{val/1e6:.1f}M"
                        elif val >= 1e3:
                            txt = f"{val/1e3:.0f}K"
                        else:
                            txt = f"{val:.0f}"
                        thresh = (vmax + vmin) / 2
                        color = "white" if val > thresh else "black"
                        ax.text(xi, yi, txt, ha="center", va="center",
                                fontsize=5.5, color=color)

    fig.tight_layout(h_pad=1.5, w_pad=0.8)
    _save(fig, "ideaF_compact_paper_heatmap")


# ======================================================================
# IDEA G: Marginal gain analysis — bar chart of delta throughput
#   For each bs, show: gain from m+2 layers, gain from n+2 active layers.
#   Helps readers understand: is it better to stack more or connect more?
# ======================================================================
def idea_g_marginal_gain(pf, dc):
    """
    For each bs, compare:
    - Marginal throughput gain from increasing m by 2 (more stacking, fixed n)
    - Marginal throughput gain from increasing n by 2 (more connecting, fixed m)
    Averaged over feasible (m,n) configurations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    for ax, df, title in [(axes[0], pf, "Prefill"), (axes[1], dc, "Decode")]:
        bs_vals = sorted(df["bs"].unique())
        gain_m_list = []  # gain from increasing m
        gain_n_list = []  # gain from increasing n

        for bs in bs_vals:
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()
            pt = best.pivot(index="dram_total_layers", columns="dram_active_layers", values="effective_stps")

            # Gain from +2 m (fix n)
            gains_m = []
            for n_val in pt.columns:
                col = pt[n_val].dropna().sort_index()
                for i in range(len(col) - 1):
                    m1, m2 = col.index[i], col.index[i + 1]
                    if m2 - m1 == 2:
                        gains_m.append((col.iloc[i + 1] - col.iloc[i]) / col.iloc[i] * 100)

            # Gain from +2 n (fix m)
            gains_n = []
            for m_val in pt.index:
                row = pt.loc[m_val].dropna().sort_index()
                for i in range(len(row) - 1):
                    n1, n2 = row.index[i], row.index[i + 1]
                    if n2 - n1 == 2:
                        gains_n.append((row.iloc[i + 1] - row.iloc[i]) / row.iloc[i] * 100)

            gain_m_list.append(np.mean(gains_m) if gains_m else 0)
            gain_n_list.append(np.mean(gains_n) if gains_n else 0)

        x = np.arange(len(bs_vals))
        w = 0.35
        ax.bar(x - w/2, gain_m_list, w, label="Stack +2 layers (↑m)",
               color="#2196F3", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, gain_n_list, w, label="Connect +2 layers (↑n)",
               color="#FF9800", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bs_vals], rotation=45, ha="right")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Avg. Marginal Gain (%)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.axhline(y=0, color="gray", linewidth=0.5)

    fig.suptitle("Marginal Benefit: Stacking (+m) vs. Connecting (+n) More Layers",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "ideaG_marginal_gain")


# ======================================================================
# IDEA H: Combined Prefill+Decode view — scatter plot where each point
#   is an (m,n) config, x = prefill throughput, y = decode throughput.
#   Shows trade-off between the two phases.
# ======================================================================
def idea_h_prefill_vs_decode(pf, dc):
    """
    For selected bs values: scatter (m,n) configs with
    x = best prefill throughput, y = best decode throughput.
    Reveals whether the same (m,n) is good for both phases.
    """
    # Find common bs values
    common_bs = sorted(set(pf["bs"].unique()) & set(dc["bs"].unique()))
    ncols = min(len(common_bs), 3)
    nrows = (len(common_bs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False)

    for i, bs in enumerate(common_bs):
        ax = axes[i // ncols][i % ncols]

        pf_best = pf[pf["bs"] == bs].groupby(
            ["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()
        dc_best = dc[dc["bs"] == bs].groupby(
            ["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

        merged = pf_best.merge(dc_best, on=["dram_total_layers", "dram_active_layers"],
                               suffixes=("_pf", "_dc"))

        # Color by n/m ratio
        merged["ratio"] = merged["dram_active_layers"] / merged["dram_total_layers"]

        sc = ax.scatter(merged["effective_stps_pf"], merged["effective_stps_dc"],
                        c=merged["ratio"], cmap="coolwarm", s=40,
                        edgecolors="black", linewidths=0.3, vmin=0, vmax=1)

        # Annotate extreme points
        for _, r in merged.iterrows():
            ax.annotate(f"({int(r['dram_total_layers'])},{int(r['dram_active_layers'])})",
                        (r["effective_stps_pf"], r["effective_stps_dc"]),
                        fontsize=5, alpha=0.7)

        ax.set_xlabel("Prefill Throughput")
        ax.set_ylabel("Decode Throughput")
        ax.set_title(f"bs={bs}", fontweight="bold")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="n/m ratio")

    for i in range(len(common_bs), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("Prefill vs. Decode Throughput by (m, n) Configuration",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "ideaH_prefill_vs_decode")


# ======================================================================
# IDEA I: Stacked area showing optimal n as fraction of m across bs
#   Shows the trend: for large bs, do you want n≈m or n<<m?
# ======================================================================
def idea_i_optimal_ratio_trend(pf, dc):
    """
    For each bs, find the best (m, n) config.
    Plot how optimal n/m ratio and absolute m, n change with bs.
    """
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))

    for row, (df, phase) in enumerate([(pf, "Prefill"), (dc, "Decode")]):
        bs_vals = sorted(df["bs"].unique())
        best_m = []
        best_n = []
        best_ratio = []

        for bs in bs_vals:
            sub = df[df["bs"] == bs]
            best_idx = sub["effective_stps"].idxmax()
            best_row = sub.loc[best_idx]
            best_m.append(int(best_row["dram_total_layers"]))
            best_n.append(int(best_row["dram_active_layers"]))
            best_ratio.append(best_row["dram_active_layers"] / best_row["dram_total_layers"])

        x = np.arange(len(bs_vals))

        # Col 0: optimal m and n
        ax = axes[row, 0]
        ax.plot(x, best_m, "o-", color="#e41a1c", linewidth=2, markersize=6, label="m (total)")
        ax.plot(x, best_n, "s-", color="#377eb8", linewidth=2, markersize=6, label="n (active)")
        ax.fill_between(x, best_n, best_m, alpha=0.15, color="gray", label="idle layers")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bs_vals], rotation=45, ha="right")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Layer Count")
        ax.set_title(f"{phase}: Optimal m, n")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 16)

        # Col 1: optimal n/m ratio
        ax = axes[row, 1]
        ax.bar(x, best_ratio, color="#4daf4a", edgecolor="black", linewidth=0.5, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bs_vals], rotation=45, ha="right")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("n/m Ratio")
        ax.set_title(f"{phase}: Optimal n/m Ratio")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, label="n=m")
        ax.legend(fontsize=7)

        # Col 2: idle layers = m - n
        ax = axes[row, 2]
        idle = [m - n for m, n in zip(best_m, best_n)]
        ax.bar(x, idle, color="#984ea3", edgecolor="black", linewidth=0.5, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bs_vals], rotation=45, ha="right")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Idle Layers (m−n)")
        ax.set_title(f"{phase}: Capacity-only Layers")
        ax.set_ylim(0, max(idle) + 2 if idle else 4)

    fig.suptitle("How Optimal DRAM Architecture Changes with Batch Size",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "ideaI_optimal_ratio_trend")


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

    print("\n=== Generating visualization ideas ===")
    print("\nIdea A: Line plots — throughput vs m, lines by n, faceted by bs")
    idea_a_lines_by_n(pf, dc)

    print("\nIdea B: Speedup heatmap relative to (m=2,n=2) baseline")
    idea_b_speedup_heatmap(pf, dc)

    print("\nIdea C: Optimal frontier — best throughput as m increases")
    idea_c_optimal_frontier(pf, dc)

    print("\nIdea D: Best (m,n) summary bar chart with annotations")
    idea_d_best_mn_summary(pf, dc)

    print("\nIdea E: Contour plots — smooth throughput landscape")
    idea_e_contour(pf, dc)

    print("\nIdea F: Compact 2-row paper figure (Prefill/Decode × selected bs)")
    idea_f_compact_paper_figure(pf, dc)

    print("\nIdea G: Marginal gain — stacking vs connecting")
    idea_g_marginal_gain(pf, dc)

    print("\nIdea H: Prefill vs Decode trade-off scatter")
    idea_h_prefill_vs_decode(pf, dc)

    print("\nIdea I: Optimal ratio trend across batch sizes")
    idea_i_optimal_ratio_trend(pf, dc)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
