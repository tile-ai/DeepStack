"""
parallel_analysis.py — Analyze how optimal parallelism changes across DRAM layers.

Usage:
    python3 -m mosaic.dse_space.case_study_dram_layer.parallel_analysis \
        mosaic/dse_space/case_study_dram_layer/runs/your-run-id

For each batch size (16, 256, 1024), finds the config with best scaled_stps_avg
per dram_total_layers, then visualizes how tp/ep/dp/pp change.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

PARALLEL_COLS = ["tp", "ep", "cp", "dp", "pp"]  # degree columns (integer)
PARALLEL_COLORS = {
    "tp": "#2196F3",   # blue
    "ep": "#4CAF50",   # green
    "cp": "#009688",   # teal
    "dp": "#FF9800",   # orange (dp without fsdp)
    "fsdp": "#7B1FA2", # purple (dp with fsdp)
    "pp": "#D32F2F",   # red
}
MAX_LAYERS = 12
BS_LIST = [16, 256, 1024]


def load_and_find_best(csv_path: str):
    """Load CSV and find best config per (bs, dram_total_layers)."""
    df = pd.read_csv(csv_path)
    df = df[df["dram_total_layers"] <= MAX_LAYERS]
    df = df[df["dram_total_layers"] == df["dram_active_layers"]]
    results = {}
    for bs in BS_LIST:
        sub = df[df["bs"] == bs]
        if len(sub) == 0:
            print(f"  WARNING: no data for bs={bs}")
            continue
        best_idx = sub.groupby("dram_total_layers")["scaled_stps_avg"].idxmax()
        best = sub.loc[best_idx].sort_values("dram_total_layers").reset_index(drop=True)
        results[bs] = best
        print(f"  bs={bs}: {len(best)} layer configs")
    return results


def plot_parallel_stacked_bar(results: dict, out_dir: str):
    """Stacked bar chart: for each layer, show the composition of tp*ep*dp*pp."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, (bs, best) in zip(axes, sorted(results.items())):
        layers = best["dram_total_layers"].values
        x = np.arange(len(layers))

        # Log2 of each parallel dimension for stacking
        log_vals = {}
        for col in PARALLEL_COLS:
            log_vals[col] = np.log2(best[col].values).astype(float)
        fsdp_flags = best["fsdp"].values

        bottom = np.zeros(len(layers))
        added_labels = set()
        for col in PARALLEL_COLS:
            if col == "dp":
                # Draw DP bars one by one: purple(FSDP) vs orange(DP)
                for i in range(len(layers)):
                    is_fsdp = bool(fsdp_flags[i])
                    color = PARALLEL_COLORS["fsdp"] if is_fsdp else PARALLEL_COLORS["dp"]
                    lbl_key = "FSDP" if is_fsdp else "DP"
                    label = lbl_key if lbl_key not in added_labels else None
                    added_labels.add(lbl_key)
                    ax.bar(x[i], log_vals[col][i], bottom=bottom[i], color=color,
                           label=label, edgecolor="white", linewidth=0.5,
                           hatch="//" if is_fsdp else None)
                    v = best[col].values[i]
                    if v > 1:
                        ax.text(x[i], bottom[i] + log_vals[col][i] / 2, str(v),
                                ha="center", va="center", fontsize=7, fontweight="bold",
                                color="white")
            else:
                ax.bar(x, log_vals[col], bottom=bottom, color=PARALLEL_COLORS[col],
                       label=col.upper(), edgecolor="white", linewidth=0.5)
                for i, v in enumerate(best[col].values):
                    if v > 1:
                        ax.text(x[i], bottom[i] + log_vals[col][i] / 2, str(v),
                                ha="center", va="center", fontsize=7, fontweight="bold",
                                color="white")
            bottom += log_vals[col]

        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_xlabel("DRAM Total Layers")
        ax.set_ylabel("log2(parallelism)")
        ax.set_title(f"BS={bs}")
        ax.legend(loc="upper left", framealpha=0.9)

        # Annotate total device count on top
        for i in range(len(layers)):
            total = int(best.iloc[i]["tp"] * best.iloc[i]["ep"]
                        * best.iloc[i]["cp"] * best.iloc[i]["dp"]
                        * best.iloc[i]["pp"])
            ax.text(x[i], bottom[i] + 0.2, f"{total}",
                    ha="center", va="bottom", fontsize=7, color="#333")

    fig.suptitle("Optimal Parallelism Composition vs DRAM Layers (Decode)", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "parallel_stacked_bar.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_parallel_lines(results: dict, out_dir: str):
    """Line chart: each parallel dimension as a separate line across layers."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, (bs, best) in zip(axes, sorted(results.items())):
        layers = best["dram_total_layers"].values
        fsdp_flags = best["fsdp"].values

        for col in PARALLEL_COLS:
            vals = best[col].values
            if col == "dp":
                # Plot DP line, then overlay FSDP markers
                ax.plot(layers, vals, marker="o", markersize=5, linewidth=2,
                        color=PARALLEL_COLORS["dp"], label="DP")
                fsdp_mask = np.array([bool(f) for f in fsdp_flags])
                if fsdp_mask.any():
                    ax.scatter(layers[fsdp_mask], vals[fsdp_mask], marker="D",
                               s=80, color=PARALLEL_COLORS["fsdp"], zorder=6,
                               label="FSDP", edgecolors="white", linewidth=0.8)
            else:
                ax.plot(layers, vals, marker="o", markersize=5, linewidth=2,
                        color=PARALLEL_COLORS[col], label=col.upper())

        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlabel("DRAM Total Layers")
        ax.set_ylabel("Parallelism Degree")
        ax.set_title(f"BS={bs}")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)

    fig.suptitle("Optimal Parallelism Dimensions vs DRAM Layers (Decode)", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "parallel_lines.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_throughput_with_parallel_annotation(results: dict, out_dir: str):
    """Throughput line with parallel config annotated at each point."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, (bs, best) in zip(axes, sorted(results.items())):
        layers = best["dram_total_layers"].values
        stps = best["scaled_stps_avg"].values

        ax.plot(layers, stps, marker="s", markersize=6, linewidth=2,
                color="#2196F3", zorder=5)
        ax.fill_between(layers, stps, alpha=0.1, color="#2196F3")

        # Annotate parallel config at each point
        for i in range(len(layers)):
            row = best.iloc[i]
            fsdp_str = "F" if row["fsdp"] else "D"
            cp_str = f"C{int(row['cp'])}" if row["cp"] > 1 else ""
            label = f"T{int(row['tp'])}E{int(row['ep'])}\n{cp_str}{fsdp_str}{int(row['dp'])}P{int(row['pp'])}"
            offset = 10 if i % 2 == 0 else -15
            ax.annotate(label, (layers[i], stps[i]),
                        textcoords="offset points", xytext=(0, offset),
                        fontsize=6, ha="center", va="bottom" if offset > 0 else "top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                                  ec="gray", alpha=0.8))

        ax.set_xlabel("DRAM Total Layers")
        ax.set_ylabel("Scaled STPS (avg)")
        ax.set_title(f"BS={bs}")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)

    fig.suptitle("Best Throughput & Parallel Config vs DRAM Layers (Decode)", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "throughput_with_parallel.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_throughput_comparison(results: dict, out_dir: str):
    """All BS on one plot for direct throughput comparison, with dual y-axes."""
    BS_COLORS = {16: "#2196F3", 256: "#4CAF50", 1024: "#FF9800"}
    BS_MARKERS = {16: "o", 256: "s", 1024: "D"}

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for bs, best in sorted(results.items()):
        layers = best["dram_total_layers"].values
        stps = best["scaled_stps_avg"].values
        ax1.plot(layers, stps, marker=BS_MARKERS[bs], markersize=6, linewidth=2,
                 color=BS_COLORS[bs], label=f"BS={bs}")

    ax1.set_xlabel("DRAM Total Layers")
    ax1.set_ylabel("Scaled STPS (avg)")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=11)
    ax1.grid(True, alpha=0.3)
    all_layers = sorted(set().union(*(r["dram_total_layers"].values for r in results.values())))
    ax1.set_xticks(all_layers)
    ax1.set_title("Best Throughput Comparison Across Batch Sizes (Decode)", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "throughput_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── Normalized version (relative to peak per BS) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for bs, best in sorted(results.items()):
        layers = best["dram_total_layers"].values
        stps = best["scaled_stps_avg"].values
        normed = stps / stps.max()
        ax.plot(layers, normed, marker=BS_MARKERS[bs], markersize=6, linewidth=2,
                color=BS_COLORS[bs], label=f"BS={bs} (peak={stps.max():.0f})")
        # Mark the peak
        peak_idx = np.argmax(stps)
        ax.annotate(f"L={layers[peak_idx]}", (layers[peak_idx], 1.0),
                    textcoords="offset points", xytext=(8, -5), fontsize=9,
                    color=BS_COLORS[bs], fontweight="bold")

    ax.set_xlabel("DRAM Total Layers")
    ax.set_ylabel("Normalized STPS (fraction of peak)")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_layers)
    ax.set_title("Normalized Throughput vs DRAM Layers (Decode)", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "throughput_normalized.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_summary_csv(results: dict, out_dir: str):
    """Save summary table of best configs."""
    all_rows = []
    for bs, best in sorted(results.items()):
        for _, row in best.iterrows():
            all_rows.append({
                "bs": int(bs),
                "dram_total_layers": int(row["dram_total_layers"]),
                "dram_active_layers": int(row["dram_active_layers"]),
                "tp": int(row["tp"]),
                "ep": int(row["ep"]),
                "dp": int(row["dp"]),
                "pp": int(row["pp"]),
                "fsdp": bool(row["fsdp"]),
                "cp": int(row["cp"]),
                "minibatch": int(row["minibatch"]),
                "total_devices": int(row["tp"] * row["ep"] * row["cp"] * row["dp"] * row["pp"]),
                "scaled_stps_avg": row["scaled_stps_avg"],
                "scaled_utps_avg": row["scaled_utps_avg"],
            })
    summary = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, "best_parallel_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(summary.to_string(index=False))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m mosaic.dse_space.case_study_dram_layer.parallel_analysis <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    csv_path = os.path.join(run_dir, "dram_layer_decode_result.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    results = load_and_find_best(csv_path)

    print("Plotting stacked bar...")
    plot_parallel_stacked_bar(results, out_dir)

    print("Plotting parallel lines...")
    plot_parallel_lines(results, out_dir)

    print("Plotting throughput with annotations...")
    plot_throughput_with_parallel_annotation(results, out_dir)

    print("Plotting throughput comparison...")
    plot_throughput_comparison(results, out_dir)

    print("Saving summary...")
    save_summary_csv(results, out_dir)

    print("Done!")


if __name__ == "__main__":
    main()
