"""
best_stps_per_bs.py — Normalized best STPS vs BW multiplier, one curve per BS (per layer)

Usage:
    python3 -m mosaic.dse_space.case_study_which_layer_noc_matter.best_stps_per_bs <run_dir> [--seq SEQ]

    Or pass two run dirs (one decode, one prefill):
    python3 -m mosaic.dse_space.case_study_which_layer_noc_matter.best_stps_per_bs <decode_run_dir> <prefill_run_dir> [--seq SEQ]

    --seq SEQ   Filter prefill data to this seq length (default: 1024)

Reads:
    <run_dir>/noc_bw_decode_result.csv
    <run_dir>/noc_bw_prefill_result.csv

Outputs plots into <run_dir>/analysis/  (single dir mode)
                 or <first_run_dir>/analysis/  (two dir mode)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


def get_metric(phase):
    if phase == "decode":
        return "scaled_stps_avg", "Scaled STPS (decode)"
    return "scaled_stps", "Scaled STPS (prefill)"


def load_data(*run_dirs):
    data = {}
    for run_dir in run_dirs:
        for phase in ("decode", "prefill"):
            if phase in data:
                continue
            result_path = os.path.join(run_dir, f"noc_bw_{phase}_result.csv")
            if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                df = pd.read_csv(result_path)
                if len(df) > 0:
                    data[phase] = df
                    print(f"  Loaded {phase}: {len(df)} rows from {result_path}")
    return data


def _save(fig, out_dir, name):
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {name}.{{png,pdf}}")


def plot_best_stps_per_bs(df, phase, out_dir):
    """For each (baseline, layer), plot normalized best STPS vs bw_multiplier with one curve per BS.

    Performance is normalized to bw_multiplier=1.0 (i.e. value at 1x = 1.0).
    """
    metric, metric_label = get_metric(phase)
    baselines = sorted(df["baseline_noc"].unique())
    layers = sorted(df["scaled_layer"].unique())

    for bl in baselines:
        ncols = len(layers)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)

        for li, layer in enumerate(layers):
            ax = axes[0, li]
            sub = df[(df["baseline_noc"] == bl) & (df["scaled_layer"] == layer)]
            if sub.empty:
                ax.set_visible(False)
                continue

            bs_vals = sorted(sub["bs"].unique())
            cmap = plt.cm.viridis(np.linspace(0, 1, len(bs_vals)))

            for wi, bs in enumerate(bs_vals):
                sl = sub[sub["bs"] == bs]
                best = sl.groupby("bw_multiplier")[metric].max().sort_index()
                base_val = best.get(1.0, None)
                if base_val is None or base_val <= 0:
                    continue
                normalized = best / base_val
                ax.plot(normalized.index, normalized.values, marker="o", markersize=4,
                        linewidth=1.5, color=cmap[wi], label=f"BS={bs}")

            ax.axhline(y=1.0, color="#9E9E9E", linestyle=":", linewidth=0.8)
            ax.set_xlabel("BW Multiplier")
            ax.set_ylabel("Normalized STPS (vs 1x)")
            ax.set_title(f"Scale {layer}", fontsize=10)
            ax.legend(fontsize=7, ncol=1)
            ax.grid(alpha=0.3)
            ax.set_xscale("log", base=2)

        fig.suptitle(f"{phase.capitalize()} — Normalized STPS per BS vs BW [{bl}]",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, out_dir, f"{phase}_best_stps_per_bs_{bl}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Best STPS per BS vs BW analysis")
    parser.add_argument("run_dirs", nargs="+", help="Run directory(ies)")
    parser.add_argument("--seq", type=int, default=1024,
                        help="Filter prefill to this seq length (default: 1024)")
    args = parser.parse_args()

    out_dir = os.path.join(args.run_dirs[0], "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Normalized STPS per BS — Analysis ===")
    data = load_data(*args.run_dirs)
    if not data:
        print("No noc_bw_*_result.csv found. Exiting.")
        return

    for phase in ("decode", "prefill"):
        if phase not in data:
            print(f"\n  Skipping {phase} (no data)")
            continue

        df = data[phase]

        # Filter prefill to specified seq
        if phase == "prefill":
            avail_seq = sorted(df["seq"].unique())
            if args.seq not in avail_seq:
                print(f"\n  WARNING: seq={args.seq} not found, available: {avail_seq}")
                continue
            df = df[df["seq"] == args.seq]
            print(f"\n--- {phase.upper()} (seq={args.seq}) ---")
        else:
            print(f"\n--- {phase.upper()} ---")

        print(f"  Baselines: {sorted(df['baseline_noc'].unique())}")
        print(f"  Layers: {sorted(df['scaled_layer'].unique())}")
        print(f"  BW multipliers: {sorted(df['bw_multiplier'].unique())}")
        print(f"  BS values: {sorted(df['bs'].unique())}")

        plot_best_stps_per_bs(df, phase, out_dir)

    print(f"\n=== Done. Results in {out_dir}/ ===")


if __name__ == "__main__":
    main()
