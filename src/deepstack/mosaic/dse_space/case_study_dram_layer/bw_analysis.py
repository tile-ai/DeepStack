"""
bw_analysis.py — Bandwidth analysis across DRAM layer configs (no thermal).

Generates:
  1. CSV with all (m, n, smem, l1) configs and their DDR peak / Little's law / L1-bound BW
  2. Bar chart visualization
"""

import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
    generate_dram_layer_configs,
    find_max_sm_count,
    _make_raw_arch,
    get_actual_bw,
    is_l1_bound,
    reference_thermal_resistance,
    SMEM_CAPACITIES,
    L1_THROUGHPUTS,
)


def analyze_bw_for_config(m, n, smem_cap, l1_tp):
    """Compute three BW tiers for a single config (no thermal scaling)."""
    # Resolve the sealed reference SM capacity.
    sm_count = find_max_sm_count(m, n, smem_cap, l1_tp)

    arch = _make_raw_arch(
        m, n, smem_cap, l1_tp, sm_count=sm_count
    )

    thermal_r = reference_thermal_resistance(m)

    # 1) DDR peak BW (after interleave efficiency)
    ddr_peak_bw = arch.ddr_peak_bandwidth
    ddr_eff_bw = arch.ddr_bandwidth   # interleave-adjusted

    # 2) Little's law cap
    arch, littles_law_limited, required_buf = arch.with_littles_law()
    littles_law_bw = arch.ddr_bandwidth
    l1_bound = is_l1_bound(arch)
    actual_bw = get_actual_bw(arch)

    return {
        "m": m, "n": n,
        "sm_count": sm_count,
        "smem_cap_KiB": smem_cap // 1024,
        "l1_tp_Bpc": l1_tp,
        "ddr_peak_bw_TBs": ddr_peak_bw / 1e12,
        "ddr_eff_bw_TBs": ddr_eff_bw / 1e12,
        "littles_law_bw_TBs": littles_law_bw / 1e12,
        "littles_law_limited": littles_law_limited,
        "littles_law_required_buf_KiB": required_buf / 1024,
        "actual_bw_TBs": actual_bw / 1e12,
        "is_l1_bound": l1_bound,
        "thermal_resistance_CpW": thermal_r,
    }


def run_analysis(out_dir="runs/bw_analysis"):
    os.makedirs(out_dir, exist_ok=True)

    configs = generate_dram_layer_configs()
    print(f"Total configs: {len(configs)}")

    csv_path = os.path.join(out_dir, "bw_analysis.csv")
    header = [
        "m", "n", "sm_count", "smem_cap_KiB", "l1_tp_Bpc",
        "ddr_peak_bw_TBs", "ddr_eff_bw_TBs",
        "littles_law_bw_TBs", "littles_law_limited", "littles_law_required_buf_KiB",
        "actual_bw_TBs", "is_l1_bound",
        "thermal_resistance_CpW",
    ]

    results = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for i, (m, n, smem_cap, l1_tp) in enumerate(configs):
            row = analyze_bw_for_config(m, n, smem_cap, l1_tp)
            writer.writerow(row)
            results.append(row)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(configs)} done", flush=True)

    print(f"CSV written to {csv_path}")

    # ── Visualization ────────────────────────────────────────────────────
    # For each (smem, l1) config, plot BW vs (m, n) as grouped bars
    for smem_cap in SMEM_CAPACITIES:
        for l1_tp in L1_THROUGHPUTS:
            _plot_bw_bars(results, smem_cap // 1024, l1_tp, out_dir)

    print(f"Plots written to {out_dir}")


def _plot_bw_bars(results, smem_kib, l1_tp, out_dir):
    """Bar chart for a fixed (smem, l1) showing BW breakdown across (m, n)."""
    subset = [r for r in results if r["smem_cap_KiB"] == smem_kib and r["l1_tp_Bpc"] == l1_tp]
    if not subset:
        return

    labels = [f"m{r['m']}n{r['n']}" for r in subset]
    peak = np.array([r["ddr_peak_bw_TBs"] for r in subset])
    eff = np.array([r["ddr_eff_bw_TBs"] for r in subset])
    ll_bw = np.array([r["littles_law_bw_TBs"] for r in subset])
    actual = np.array([r["actual_bw_TBs"] for r in subset])

    x = np.arange(len(labels))
    width = 0.8

    fig, ax = plt.subplots(figsize=(max(16, len(labels) * 0.3), 6))

    # Stack: actual (colored) on bottom, then LL gap (gray), then peak gap (white)
    ax.bar(x, actual, width, label="Actual BW (after L1 bound)", color="#2196F3", zorder=3)
    ax.bar(x, ll_bw - actual, width, bottom=actual, label="Little's law limited gap", color="#9E9E9E", alpha=0.6, zorder=2)
    ax.bar(x, peak - ll_bw, width, bottom=ll_bw, label="DDR peak gap (interleave)", color="white", edgecolor="#BDBDBD", linewidth=0.5, zorder=1)

    # Mark interleave-adjusted effective BW line
    ax.step(x, eff, where="mid", color="red", linewidth=1.0, linestyle="--", label="DDR eff BW (interleave)", zorder=4)

    ax.set_xlabel("DRAM config (m=total layers, n=active layers)")
    ax.set_ylabel("Bandwidth (TB/s)")
    ax.set_title(f"BW Breakdown — smem={smem_kib}KiB, L1={l1_tp}B/cyc")

    # Reduce tick density for readability
    if len(labels) > 40:
        step = max(1, len(labels) // 40)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=90, fontsize=6)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)

    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fname = f"bw_smem{smem_kib}KiB_l1_{l1_tp}Bpc.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/bw_analysis"
    run_analysis(out_dir)
