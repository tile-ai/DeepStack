#!/usr/bin/env python3
"""
Paper-quality figures and tables for DRAM layer DSE analysis.
DeepSeekV3 model, prefill (seq=1024) and decode phases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import os

# ── Constants ────────────────────────────────────────────────────────────────
POWER_CAP = 100.0
NUM_DEVICES = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "paper_figures")
DATA_DIR = os.path.join(BASE_DIR, "lfs")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
})

# ── Data loading ─────────────────────────────────────────────────────────────
def load_and_prepare(csv_path, stps_col, raw_stps_col):
    df = pd.read_csv(csv_path)
    over = df["total_power_per_device_W"] > POWER_CAP
    df["eff_power"] = df["total_power_per_device_W"].copy()
    df.loc[over, "eff_power"] = POWER_CAP
    df["eff_stps"] = df[raw_stps_col].copy()
    df.loc[over, "eff_stps"] = df.loc[over, stps_col]
    df["sys_power"] = df["eff_power"] * NUM_DEVICES
    df["tokens_per_joule"] = df["eff_stps"] / df["sys_power"]
    df["m"] = df["dram_total_layers"]
    df["n"] = df["dram_active_layers"]
    return df

pf = load_and_prepare(
    os.path.join(DATA_DIR, "dram_layer_prefill_result_dpsk_mn1to12_seq1024.csv"),
    "scaled_stps", "raw_stps"
)
dc = load_and_prepare(
    os.path.join(DATA_DIR, "dram_layer_decode_result_dpsk_mn1to12.csv"),
    "scaled_stps_avg", "raw_stps_avg"
)


def savefig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: (m,n) Heatmap — Best stps by DRAM layer config
#   2 rows (prefill / decode) × 3 cols (representative bs values)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_mn_heatmap():
    prefill_bs = [1, 64, 1024]
    decode_bs = [1, 256, 16384]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))

    for col, bs in enumerate(prefill_bs):
        _draw_mn_heatmap(axes[0, col], pf, bs, "eff_stps",
                         title=f"Prefill bs={bs}", cmap="YlOrRd", fmt=".0f",
                         divide=1000, cbar_label="stps (×10³)")

    for col, bs in enumerate(decode_bs):
        _draw_mn_heatmap(axes[1, col], dc, bs, "eff_stps",
                         title=f"Decode bs={bs}", cmap="YlGnBu", fmt=".0f",
                         divide=1000, cbar_label="stps (×10³)")

    fig.suptitle("Best System Throughput (stps) by DRAM Layer Config (m, n)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig1_mn_heatmap_stps")


def fig1b_mn_heatmap_tokj():
    prefill_bs = [1, 64, 1024]
    decode_bs = [1, 256, 16384]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))

    for col, bs in enumerate(prefill_bs):
        _draw_mn_heatmap(axes[0, col], pf, bs, "tokens_per_joule",
                         title=f"Prefill bs={bs}", cmap="YlOrRd", fmt=".2f",
                         cbar_label="tokens/J")

    for col, bs in enumerate(decode_bs):
        _draw_mn_heatmap(axes[1, col], dc, bs, "tokens_per_joule",
                         title=f"Decode bs={bs}", cmap="YlGnBu", fmt=".2f",
                         cbar_label="tokens/J")

    fig.suptitle("Best Energy Efficiency (tokens/J) by DRAM Layer Config (m, n)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig1b_mn_heatmap_tokj")


def _draw_mn_heatmap(ax, df, bs, metric, title, cmap, fmt, divide=1, cbar_label=None):
    sub = df[df["bs"] == bs]
    pivot = sub.groupby(["m", "n"])[metric].max().reset_index()
    if divide != 1:
        pivot[metric] = pivot[metric] / divide
    pt = pivot.pivot(index="m", columns="n", values=metric)
    pt = pt.sort_index(ascending=True)
    pt = pt[sorted(pt.columns)]

    im = ax.imshow(pt.values, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(pt.columns)))
    ax.set_xticklabels(pt.columns.astype(int), fontsize=6)
    ax.set_yticks(range(len(pt.index)))
    ax.set_yticklabels(pt.index.astype(int), fontsize=6)
    ax.set_xlabel("n (active layers)")
    ax.set_ylabel("m (total layers)")
    ax.set_title(title, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Annotate cells
    vmax = np.nanmax(pt.values)
    vmin = np.nanmin(pt.values)
    mid = (vmax + vmin) / 2
    for yi in range(len(pt.index)):
        for xi in range(len(pt.columns)):
            val = pt.values[yi, xi]
            if not np.isnan(val):
                color = "white" if val > mid else "black"
                ax.text(xi, yi, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=5, color=color)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Pareto Front — Performance vs Energy Efficiency
#   Side-by-side prefill and decode, multiple bs curves
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # Prefill
    _draw_pareto(axes[0], pf, "eff_stps", "tokens_per_joule",
                 bs_list=[1, 16, 64, 256, 1024],
                 title="Prefill (seq=1024)",
                 xlabel="tokens/J", ylabel="System throughput (stps)")

    # Decode
    _draw_pareto(axes[1], dc, "eff_stps", "tokens_per_joule",
                 bs_list=[1, 64, 256, 1024, 4096, 16384],
                 title="Decode",
                 xlabel="tokens/J", ylabel="System throughput (stps)")

    fig.tight_layout()
    savefig(fig, "fig2_pareto_front")


def _draw_pareto(ax, df, perf_col, eff_col, bs_list, title, xlabel, ylabel):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(bs_list)))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    for i, bs in enumerate(bs_list):
        sub = df[df["bs"] == bs].copy()
        if len(sub) == 0:
            continue

        # Compute Pareto front
        sub = sub.sort_values(eff_col, ascending=False)
        pareto_pts = []
        best_perf = -1
        for _, row in sub.iterrows():
            if row[perf_col] > best_perf:
                pareto_pts.append(row)
                best_perf = row[perf_col]

        pareto_df = pd.DataFrame(pareto_pts)
        pareto_df = pareto_df.sort_values(eff_col)

        # Plot all points as faint background
        ax.scatter(sub[eff_col], sub[perf_col], alpha=0.03, s=2,
                   color=colors[i], rasterized=True)
        # Plot Pareto front
        ax.plot(pareto_df[eff_col], pareto_df[perf_col], '-',
                color=colors[i], linewidth=1.5, alpha=0.8)
        ax.scatter(pareto_df[eff_col], pareto_df[perf_col],
                   color=colors[i], s=25, marker=markers[i % len(markers)],
                   edgecolors='black', linewidths=0.3, zorder=5,
                   label=f"bs={bs}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, markerscale=0.8, framealpha=0.7)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Parallelism Strategy Shift — Grouped bar for best-perf configs
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_parallelism():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    _draw_parallelism_bars(axes[0], pf, "eff_stps",
                           bs_list=[1, 4, 16, 64, 256, 1024],
                           title="Prefill — Best Perf Config Parallelism")

    _draw_parallelism_bars(axes[1], dc, "eff_stps",
                           bs_list=[1, 4, 16, 64, 256, 1024, 4096, 16384],
                           title="Decode — Best Perf Config Parallelism")

    fig.tight_layout()
    savefig(fig, "fig3_parallelism_shift")


def _draw_parallelism_bars(ax, df, stps_col, bs_list, title):
    """Stacked bar showing tp, ep, dp, pp for best-perf config at each bs."""
    strategies = ["tp", "ep", "dp", "pp"]
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    data = {s: [] for s in strategies}
    labels = []

    for bs in bs_list:
        grp = df[df["bs"] == bs]
        if len(grp) == 0:
            continue
        best = grp.loc[grp[stps_col].idxmax()]
        total = best["tp"] * best["ep"] * best["dp"] * best["pp"]
        for s in strategies:
            data[s].append(best[s] / total * 100)  # percentage
        labels.append(str(bs))

    x = np.arange(len(labels))
    width = 0.6
    bottom = np.zeros(len(labels))

    for s, c in zip(strategies, colors):
        vals = np.array(data[s])
        ax.bar(x, vals, width, bottom=bottom, color=c, label=s.upper(),
               edgecolor="white", linewidth=0.3)
        # Label if > 15%
        for i, v in enumerate(vals):
            if v > 15:
                ax.text(x[i], bottom[i] + v / 2, f"{int(v)}%",
                        ha="center", va="center", fontsize=6, color="white",
                        fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Parallelism share (%)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
    ax.set_ylim(0, 105)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Best stps vs m (n=m, full connection) across bs — line plot
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_stps_vs_m():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Prefill
    _draw_stps_vs_m(axes[0], pf, "eff_stps",
                    bs_list=[1, 16, 64, 256, 1024],
                    title="Prefill — Best stps vs DRAM Layers (n=m)",
                    ylabel="Best stps (×10³)", divide=1000)

    # Decode
    _draw_stps_vs_m(axes[1], dc, "eff_stps",
                    bs_list=[1, 64, 256, 1024, 4096, 16384],
                    title="Decode — Best stps vs DRAM Layers (n=m)",
                    ylabel="Best stps (×10³)", divide=1000)

    fig.tight_layout()
    savefig(fig, "fig4_stps_vs_m")


def _draw_stps_vs_m(ax, df, stps_col, bs_list, title, ylabel, divide=1):
    colors = plt.cm.tab10(np.linspace(0, 1, len(bs_list)))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    # Only n=m configs
    sub = df[df["m"] == df["n"]]
    m_vals = sorted(sub["m"].unique())

    for i, bs in enumerate(bs_list):
        bs_sub = sub[sub["bs"] == bs]
        if len(bs_sub) == 0:
            continue
        best_per_m = bs_sub.groupby("m")[stps_col].max().reindex(m_vals)
        ax.plot(m_vals, best_per_m.values / divide, '-',
                marker=markers[i % len(markers)], markersize=4,
                color=colors[i], linewidth=1.2, label=f"bs={bs}")

    ax.set_xlabel("DRAM layers (m = n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_xticks(m_vals)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Partial activation — best stps at fixed m, varying n
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_partial_activation():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Fix bs=1024 for prefill, show stps vs n for m=8, 10, 12
    _draw_partial(axes[0], pf, "eff_stps", bs=1024,
                  m_list=[6, 8, 10, 12],
                  title="Prefill bs=1024 — stps vs Active Layers",
                  ylabel="Best stps (×10³)", divide=1000)

    # Fix bs=4096 for decode
    _draw_partial(axes[1], dc, "eff_stps", bs=4096,
                  m_list=[6, 8, 10, 12],
                  title="Decode bs=4096 — stps vs Active Layers",
                  ylabel="Best stps (×10³)", divide=1000)

    fig.tight_layout()
    savefig(fig, "fig5_partial_activation")


def _draw_partial(ax, df, stps_col, bs, m_list, title, ylabel, divide=1):
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(m_list)))
    markers = ['o', 's', '^', 'D', 'v']

    bs_sub = df[df["bs"] == bs]

    for i, m in enumerate(m_list):
        m_sub = bs_sub[bs_sub["m"] == m]
        n_vals = sorted(m_sub["n"].unique())
        best_per_n = m_sub.groupby("n")[stps_col].max().reindex(n_vals)
        ax.plot(n_vals, best_per_n.values / divide, '-',
                marker=markers[i % len(markers)], markersize=5,
                color=colors[i], linewidth=1.3, label=f"m={m}")

    ax.set_xlabel("Active layers (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Partial activation — tokens/J at fixed m, varying n
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_partial_tokj():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    _draw_partial(axes[0], pf, "tokens_per_joule", bs=1024,
                  m_list=[6, 8, 10, 12],
                  title="Prefill bs=1024 — tokens/J vs Active Layers",
                  ylabel="Best tokens/J", divide=1)

    _draw_partial(axes[1], dc, "tokens_per_joule", bs=4096,
                  m_list=[6, 8, 10, 12],
                  title="Decode bs=4096 — tokens/J vs Active Layers",
                  ylabel="Best tokens/J", divide=1)

    fig.tight_layout()
    savefig(fig, "fig6_partial_tokj")


# ═══════════════════════════════════════════════════════════════════════════════
# Table 1: Best configs summary
# ═══════════════════════════════════════════════════════════════════════════════
def table1_best_configs():
    print("\n" + "=" * 120)
    print("TABLE 1: Best Performance and Best Efficiency Configs")
    print("=" * 120)

    header = (f"{'Phase':<10} {'bs':>5} | {'Metric':<8} | {'stps':>10} {'tok/J':>8} "
              f"| {'m':>2} {'n':>2} {'SM':>3} {'L1bw':>5} {'DDRbw':>7} "
              f"| {'tp':>3} {'ep':>4} {'dp':>3} {'pp':>3} | {'Pwr(W)':>7}")
    print(header)
    print("-" * len(header))

    for phase_name, df, bs_list in [
        ("Prefill", pf, [1, 64, 256, 1024]),
        ("Decode", dc, [1, 64, 256, 1024, 4096, 16384]),
    ]:
        for bs in bs_list:
            grp = df[df["bs"] == bs]
            # Best perf
            bp = grp.loc[grp["eff_stps"].idxmax()]
            print(f"{phase_name:<10} {bs:>5} | {'Perf':<8} | {bp['eff_stps']:>10.1f} {bp['tokens_per_joule']:>8.4f} "
                  f"| {int(bp['m']):>2} {int(bp['n']):>2} {int(bp['sm_count']):>3} {int(bp['l1_throughput_Bpc']):>5} {bp['ddr_peak_bw_TBs']:>7.3f} "
                  f"| {int(bp['tp']):>3} {int(bp['ep']):>4} {int(bp['dp']):>3} {int(bp['pp']):>3} | {bp['total_power_per_device_W']:>7.1f}")
            # Best eff
            be = grp.loc[grp["tokens_per_joule"].idxmax()]
            print(f"{'':>10} {'':>5} | {'Eff':<8} | {be['eff_stps']:>10.1f} {be['tokens_per_joule']:>8.4f} "
                  f"| {int(be['m']):>2} {int(be['n']):>2} {int(be['sm_count']):>3} {int(be['l1_throughput_Bpc']):>5} {be['ddr_peak_bw_TBs']:>7.3f} "
                  f"| {int(be['tp']):>3} {int(be['ep']):>4} {int(be['dp']):>3} {int(be['pp']):>3} | {be['total_power_per_device_W']:>7.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Table 2: Pareto-optimal configs for paper
# ═══════════════════════════════════════════════════════════════════════════════
def table2_pareto():
    print("\n" + "=" * 120)
    print("TABLE 2: Pareto-Optimal Configs (perf vs tokens/J)")
    print("=" * 120)

    for phase_name, df, bs_list in [
        ("Prefill", pf, [64, 1024]),
        ("Decode", dc, [256, 4096, 16384]),
    ]:
        for bs in bs_list:
            grp = df[df["bs"] == bs].copy()
            grp = grp.sort_values("tokens_per_joule", ascending=False)
            pareto = []
            best_perf = -1
            for _, row in grp.iterrows():
                if row["eff_stps"] > best_perf:
                    pareto.append(row)
                    best_perf = row["eff_stps"]

            print(f"\n  {phase_name} bs={bs}: {len(pareto)} Pareto configs")
            for p in pareto:
                print(f"    stps={p['eff_stps']:>10.1f}  tok/J={p['tokens_per_joule']:.4f}  "
                      f"m={int(p['m'])} n={int(p['n'])}  sm={int(p['sm_count'])}  "
                      f"l1_bw={int(p['l1_throughput_Bpc'])}  ddr_bw={p['ddr_peak_bw_TBs']:.3f}  "
                      f"tp={int(p['tp'])} ep={int(p['ep'])} dp={int(p['dp'])} pp={int(p['pp'])}  "
                      f"pwr={p['total_power_per_device_W']:.1f}W")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Generating paper figures...")

    fig1_mn_heatmap()
    fig1b_mn_heatmap_tokj()
    fig2_pareto()
    fig3_parallelism()
    fig4_stps_vs_m()
    fig5_partial_activation()
    fig6_partial_tokj()

    table1_best_configs()
    table2_pareto()

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
