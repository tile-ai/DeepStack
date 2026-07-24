"""
bw_analysis_connected.py — BW analysis with stack == connected layers (m=n).

Generates a single figure with subplots for each (smem, l1) combination.
Each subplot shows BW breakdown bars + SM count line on secondary y-axis.
"""

import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
    find_max_sm_count,
    _make_raw_arch,
    get_actual_bw,
    is_l1_bound,
    SMEM_CAPACITIES,
    L1_THROUGHPUTS,
)


MAX_LAYERS = 16


def analyze_bw_connected(n, smem_cap, l1_tp):
    """Compute BW tiers for m=n config (no thermal scaling)."""
    m = n
    sm_count = find_max_sm_count(m, n, smem_cap, l1_tp)

    arch = _make_raw_arch(
        m, n, smem_cap, l1_tp, sm_count=sm_count
    )

    # 1) DDR peak / effective BW
    ddr_peak_bw = arch.ddr_peak_bandwidth
    ddr_eff_bw = arch.ddr_bandwidth

    # 2) Little's law cap
    arch, _, _ = arch.with_littles_law()
    littles_law_bw = arch.ddr_bandwidth
    l1_bound = is_l1_bound(arch)
    actual_bw = get_actual_bw(arch)

    return {
        "n": n,
        "sm_count": sm_count,
        "smem_cap_KiB": smem_cap // 1024,
        "l1_tp_Bpc": l1_tp,
        "ddr_peak_bw_TBs": ddr_peak_bw / 1e12,
        "ddr_eff_bw_TBs": ddr_eff_bw / 1e12,
        "littles_law_bw_TBs": littles_law_bw / 1e12,
        "actual_bw_TBs": actual_bw / 1e12,
        "is_l1_bound": l1_bound,
    }


def run_analysis(out_dir="runs/bw_analysis_connected"):
    os.makedirs(out_dir, exist_ok=True)

    # Compute all results (m=n, sweep n=1..MAX_LAYERS)
    results = []
    total = MAX_LAYERS * len(SMEM_CAPACITIES) * len(L1_THROUGHPUTS)
    for i, n in enumerate(range(1, MAX_LAYERS + 1)):
        for smem_cap in SMEM_CAPACITIES:
            for l1_tp in L1_THROUGHPUTS:
                row = analyze_bw_connected(n, smem_cap, l1_tp)
                results.append(row)
    print(f"Computed {len(results)} configs (m=n, n=1..{MAX_LAYERS})")

    # Write CSV
    csv_path = os.path.join(out_dir, "bw_analysis_connected.csv")
    header = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV written to {csv_path}")

    # ── Big subplot figure ───────────────────────────────────────────────
    n_rows = len(SMEM_CAPACITIES)
    n_cols = len(L1_THROUGHPUTS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 2.4 * n_rows),
                             sharex=True, sharey="row")
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for ri, smem_cap in enumerate(SMEM_CAPACITIES):
        smem_kib = smem_cap // 1024
        for ci, l1_tp in enumerate(L1_THROUGHPUTS):
            ax = axes[ri, ci]
            subset = [r for r in results
                      if r["smem_cap_KiB"] == smem_kib and r["l1_tp_Bpc"] == l1_tp]
            subset.sort(key=lambda r: r["n"])

            labels = [str(r["n"]) for r in subset]
            peak = np.array([r["ddr_peak_bw_TBs"] for r in subset])
            eff = np.array([r["ddr_eff_bw_TBs"] for r in subset])
            ll_bw = np.array([r["littles_law_bw_TBs"] for r in subset])
            actual = np.array([r["actual_bw_TBs"] for r in subset])
            sm_counts = np.array([r["sm_count"] for r in subset])

            x = np.arange(len(labels))
            width = 0.65

            # Stacked bars (bottom → top: L1 effective, Little's gap,
            #   charging gap, peak gap)
            ax.bar(x, actual, width, color="#2196F3", zorder=3)
            ax.bar(x, ll_bw - actual, width, bottom=actual,
                   color="#9E9E9E", alpha=0.6, zorder=2)
            ax.bar(x, eff - ll_bw, width, bottom=ll_bw,
                   color="#4CAF50", alpha=0.5, zorder=2)
            ax.bar(x, peak - eff, width, bottom=eff,
                   color="#E8EAF6", edgecolor="#BDBDBD",
                   linewidth=0.5, zorder=1)

            ax.set_title(f"smem={smem_kib}KiB, L1={l1_tp}B/cyc", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", alpha=0.3)

            # Only leftmost column gets y-label
            if ci == 0:
                ax.set_ylabel("BW (TB/s)", fontsize=11)

            # Secondary y-axis: SM count line
            ax2 = ax.twinx()
            ax2.plot(x, sm_counts, color="#FF9800", marker="o", markersize=3,
                     linewidth=1.5, zorder=5)
            ax2.tick_params(axis="y", labelcolor="#FF9800", labelsize=9)
            # Only rightmost column gets SM y-label
            if ci == n_cols - 1:
                ax2.set_ylabel("SM Count", fontsize=11, color="#FF9800")

    # Shared x-label on bottom row
    for ci in range(n_cols):
        axes[-1, ci].set_xlabel("Layers", fontsize=11)

    # ── Unified legend at top ────────────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    bw_handles = [
        Patch(facecolor="#E8EAF6", edgecolor="#BDBDBD", label="3D DRAM Peak BW"),
        Patch(facecolor="#4CAF50", alpha=0.5, label="After Charging Limits"),
        Patch(facecolor="#9E9E9E", alpha=0.6, label="Little's Law Limit"),
        Patch(facecolor="#2196F3", label="L1 Limit (Effective BW)"),
    ]
    sm_handle = [
        Line2D([0], [0], color="#FF9800", marker="o", markersize=4,
               linewidth=1.5, label="SM Count"),
    ]
    fig.legend(handles=bw_handles + sm_handle, loc="upper center",
               ncol=5, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("BW Breakdown & SM Count  (stack = connected layers)",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = "bw_connected_subplots.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {os.path.join(out_dir, fname)}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/bw_analysis_connected"
    run_analysis(out_dir)
