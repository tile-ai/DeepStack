#!/usr/bin/env python3
"""
Paper figure: Hardware config sweep across DRAM layers (m=2..14)
for DeepSeekV3, bs=4 and bs=1024, prefill and decode.

Shows how optimal hardware (SM, n, DDR BW, L1 BW) shifts with m,
and the resulting best stps and tokens/J.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import os

POWER_CAP = 100.0
NUM_DEVICES = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "paper_figures")
DATA_DIR = os.path.join(BASE_DIR, "lfs")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


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

# Filter m <= 14
pf = pf[pf["m"] <= 14]
dc = dc[dc["m"] <= 14]


def get_best_per_m(df, bs, metric="eff_stps"):
    """For each m, find the config with the best metric value."""
    grp = df[df["bs"] == bs]
    results = []
    for m_val in sorted(grp["m"].unique()):
        sub = grp[grp["m"] == m_val]
        best = sub.loc[sub[metric].idxmax()]
        results.append(best)
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# Main figure: 5 rows × 2 cols
#   Col 1: bs=4,  Col 2: bs=1024
#   Row 1: Best stps vs m
#   Row 2: Best tokens/J vs m
#   Row 3: SM count of best config vs m
#   Row 4: Active layers (n) vs m
#   Row 5: DDR BW (TB/s) vs m
# Each cell: 4 lines = prefill best-perf, prefill best-eff,
#                       decode best-perf, decode best-eff
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure():
    bs_list = [4, 1024]

    # Colors and styles
    c_pf = "#d62728"   # red for prefill
    c_dc = "#1f77b4"   # blue for decode
    ls_perf = "-"       # solid for best-perf
    ls_eff  = "--"      # dashed for best-eff
    mk_perf = "o"
    mk_eff  = "s"
    ms = 4.5  # marker size

    fig, axes = plt.subplots(5, 2, figsize=(7.0, 9.5),
                             gridspec_kw={"height_ratios": [1.2, 1.2, 1, 1, 1]})

    row_labels = [
        "Best System\nThroughput (stps)",
        "Best Energy Eff.\n(tokens/J)",
        "SM Count",
        "Active DRAM\nLayers (n)",
        "DDR BW\n(TB/s)",
    ]

    for col, bs in enumerate(bs_list):
        # Get best-perf and best-eff data for each phase
        pf_perf = get_best_per_m(pf, bs, "eff_stps")
        pf_eff  = get_best_per_m(pf, bs, "tokens_per_joule")
        dc_perf = get_best_per_m(dc, bs, "eff_stps")
        dc_eff  = get_best_per_m(dc, bs, "tokens_per_joule")

        datasets = [
            # (data, color, linestyle, marker, label_prefix)
            (pf_perf, c_pf, ls_perf, mk_perf, "Prefill best-perf"),
            (pf_eff,  c_pf, ls_eff,  mk_eff,  "Prefill best-eff"),
            (dc_perf, c_dc, ls_perf, mk_perf, "Decode best-perf"),
            (dc_eff,  c_dc, ls_eff,  mk_eff,  "Decode best-eff"),
        ]

        # Row 0: stps
        ax = axes[0, col]
        for data, c, ls, mk, lbl in datasets:
            ax.plot(data["m"], data["eff_stps"] / 1000, ls, color=c,
                    marker=mk, markersize=ms, linewidth=1.3, label=lbl)
        ax.set_ylabel("stps (×10³)")
        ax.set_title(f"bs = {bs}", fontsize=10, fontweight="bold")

        # Row 1: tokens/J
        ax = axes[1, col]
        for data, c, ls, mk, lbl in datasets:
            ax.plot(data["m"], data["tokens_per_joule"], ls, color=c,
                    marker=mk, markersize=ms, linewidth=1.3)
        ax.set_ylabel("tokens/J")

        # Row 2: SM count
        ax = axes[2, col]
        for data, c, ls, mk, lbl in datasets:
            ax.plot(data["m"], data["sm_count"], ls, color=c,
                    marker=mk, markersize=ms, linewidth=1.3)
        ax.set_ylabel("SM count")
        ax.set_yticks(range(1, 12))

        # Row 3: Active layers (n)
        ax = axes[3, col]
        for data, c, ls, mk, lbl in datasets:
            ax.plot(data["m"], data["n"], ls, color=c,
                    marker=mk, markersize=ms, linewidth=1.3)
        ax.set_ylabel("Active layers (n)")
        # Reference line n=m
        m_range = sorted(pf_perf["m"].unique())
        ax.plot(m_range, m_range, ":", color="gray", linewidth=0.8, alpha=0.5)
        ax.text(max(m_range) - 0.5, max(m_range) - 0.8, "n=m", fontsize=6,
                color="gray", ha="right")

        # Row 4: DDR BW
        ax = axes[4, col]
        for data, c, ls, mk, lbl in datasets:
            ax.plot(data["m"], data["ddr_peak_bw_TBs"], ls, color=c,
                    marker=mk, markersize=ms, linewidth=1.3)
        ax.set_ylabel("DDR BW (TB/s)")
        ax.set_xlabel("Total DRAM Layers (m)")

    # Set x-axis for all
    m_ticks = sorted(pf[pf["bs"] == 4]["m"].unique())
    for row in range(5):
        for col in range(2):
            ax = axes[row, col]
            ax.set_xticks(m_ticks)
            ax.set_xlim(m_ticks[0] - 0.5, m_ticks[-1] + 0.5)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            if row < 4:
                ax.set_xticklabels([])

    # Row labels on left
    for row, lbl in enumerate(row_labels):
        axes[row, 0].annotate(lbl, xy=(-0.35, 0.5), xycoords="axes fraction",
                              fontsize=7.5, ha="center", va="center",
                              rotation=90, fontweight="bold",
                              annotation_clip=False)

    # Legend at top
    legend_elements = [
        Line2D([0], [0], color=c_pf, linestyle=ls_perf, marker=mk_perf,
               markersize=5, label="Prefill best-perf"),
        Line2D([0], [0], color=c_pf, linestyle=ls_eff, marker=mk_eff,
               markersize=5, label="Prefill best-eff"),
        Line2D([0], [0], color=c_dc, linestyle=ls_perf, marker=mk_perf,
               markersize=5, label="Decode best-perf"),
        Line2D([0], [0], color=c_dc, linestyle=ls_eff, marker=mk_eff,
               markersize=5, label="Decode best-eff"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0.06, 0, 1, 0.97])
    fig.subplots_adjust(hspace=0.15)

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig_hw_sweep_bs4_bs1024.{ext}"))
    plt.close(fig)
    print("Saved fig_hw_sweep_bs4_bs1024")


# ═══════════════════════════════════════════════════════════════════════════════
# Companion figure: L1 BW and power
# ═══════════════════════════════════════════════════════════════════════════════
def make_figure_l1bw_power():
    bs_list = [4, 1024]
    c_pf = "#d62728"
    c_dc = "#1f77b4"
    ls_perf = "-"
    ls_eff  = "--"
    mk_perf = "o"
    mk_eff  = "s"
    ms = 4.5

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0))

    for col, bs in enumerate(bs_list):
        pf_perf = get_best_per_m(pf, bs, "eff_stps")
        pf_eff  = get_best_per_m(pf, bs, "tokens_per_joule")
        dc_perf = get_best_per_m(dc, bs, "eff_stps")
        dc_eff  = get_best_per_m(dc, bs, "tokens_per_joule")

        datasets = [
            (pf_perf, c_pf, ls_perf, mk_perf),
            (pf_eff,  c_pf, ls_eff,  mk_eff),
            (dc_perf, c_dc, ls_perf, mk_perf),
            (dc_eff,  c_dc, ls_eff,  mk_eff),
        ]

        # Row 0: L1 BW
        ax = axes[0, col]
        for data, c, ls, mk in datasets:
            ax.plot(data["m"], data["l1_throughput_Bpc"], ls, color=c,
                    marker=mk, markersize=ms, linewidth=1.3)
        ax.set_ylabel("L1 Throughput (B/cycle)")
        ax.set_title(f"bs = {bs}", fontsize=10, fontweight="bold")

        # Row 1: Power
        ax = axes[1, col]
        for data, c, ls, mk in datasets:
            ax.plot(data["m"], data["total_power_per_device_W"], ls, color=c,
                    marker=mk, markersize=ms, linewidth=1.3)
        ax.axhline(POWER_CAP, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylabel("Per-device Power (W)")
        ax.set_xlabel("Total DRAM Layers (m)")

    m_ticks = sorted(pf[pf["bs"] == 4]["m"].unique())
    for row in range(2):
        for col in range(2):
            axes[row, col].set_xticks(m_ticks)
            axes[row, col].set_xlim(m_ticks[0] - 0.5, m_ticks[-1] + 0.5)
            axes[row, col].grid(True, alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig_hw_sweep_l1bw_power.{ext}"))
    plt.close(fig)
    print("Saved fig_hw_sweep_l1bw_power")


# ═══════════════════════════════════════════════════════════════════════════════
# Print detailed table for paper
# ═══════════════════════════════════════════════════════════════════════════════
def print_table():
    print("\n" + "=" * 140)
    print("  TABLE: Best Perf & Best Eff configs per m, bs=4 and bs=1024")
    print("=" * 140)

    for phase_name, df in [("Prefill", pf), ("Decode", dc)]:
        for bs in [4, 1024]:
            grp = df[df["bs"] == bs]
            if len(grp) == 0:
                continue

            for metric, metric_label in [("eff_stps", "Best-Perf"), ("tokens_per_joule", "Best-Eff")]:
                print(f"\n  {phase_name} bs={bs} — {metric_label}")
                print(f"  {'m':>3} {'n':>3} | {'stps':>10} {'tok/J':>8} {'pwr':>6} | "
                      f"{'SM':>3} {'L1bw':>5} {'DDRbw':>7} | "
                      f"{'tp':>3} {'ep':>4} {'dp':>3} {'pp':>3}")
                print(f"  " + "-" * 85)

                for m_val in sorted(grp["m"].unique()):
                    sub = grp[grp["m"] == m_val]
                    best = sub.loc[sub[metric].idxmax()]
                    print(f"  {int(best['m']):>3} {int(best['n']):>3} | "
                          f"{best['eff_stps']:>10.1f} {best['tokens_per_joule']:>8.4f} {best['total_power_per_device_W']:>6.1f} | "
                          f"{int(best['sm_count']):>3} {int(best['l1_throughput_Bpc']):>5} {best['ddr_peak_bw_TBs']:>7.3f} | "
                          f"{int(best['tp']):>3} {int(best['ep']):>4} {int(best['dp']):>3} {int(best['pp']):>3}")


def main():
    make_figure()
    make_figure_l1bw_power()
    print_table()
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
