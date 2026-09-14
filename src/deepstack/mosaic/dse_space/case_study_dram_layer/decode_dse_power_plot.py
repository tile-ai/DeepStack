"""
decode_dse_power_plot.py — Plot power distribution for decode DSE results.

Usage:
    python3 decode_dse_power_plot.py <run_dir>

    e.g.:
    python3 decode_dse_power_plot.py runs/your-run-id

Outputs figures into <run_dir>/analysis/:
    - power_breakdown_vs_m.{png,pdf}        : Chip vs NoC power stacked bar per m
    - power_vs_stps_scatter.{png,pdf}        : Power vs scaled STPS scatter (color=m)
    - power_distribution_by_m.{png,pdf}      : Box plot of total power per m
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────
MAX_M = 12
TDP_W = 100.0
MAX_BS = 4096
T_AMBIENT_C = 35.0      # ambient temperature (°C)
T_THROTTLE_C = 95.0     # thermal throttle threshold (°C)


def load_data(run_dir):
    """Load CSV, filter m==n & BS<=MAX_BS, return filtered DataFrame."""
    csv_path = os.path.join(run_dir, "dram_layer_decode_result.csv")
    df = pd.read_csv(csv_path)
    df_mn = df[(df["dram_total_layers"] == df["dram_active_layers"]) & (df["bs"] <= MAX_BS)]
    return df_mn


def plot_power_breakdown_bar(df_mn, out_dir):
    """Figure 1: Average chip power vs NoC power stacked bar per m (best-STPS configs)."""
    # For each (m, bs), pick the config with best scaled_stps_avg
    best_idx = df_mn.groupby(["dram_total_layers", "bs"])["scaled_stps_avg"].idxmax()
    best = df_mn.loc[best_idx]

    # Average across BS for each m
    avg_power = best.groupby("dram_total_layers").agg(
        chip_power=("chip_power_W", "mean"),
        noc_power=("noc_power_per_device_W", "mean"),
    ).reset_index()
    avg_power.columns = ["m", "chip_power", "noc_power"]
    avg_power = avg_power.sort_values("m")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(avg_power))
    w = 0.6

    ax.bar(x, avg_power["chip_power"], w, label="Chip Compute Power",
           color="#2196F3", edgecolor="black", linewidth=0.6)
    ax.bar(x, avg_power["noc_power"], w, bottom=avg_power["chip_power"],
           label="NoC Power", color="#FF9800", edgecolor="black", linewidth=0.6)

    ax.axhline(y=TDP_W, color="red", linestyle="--", linewidth=1.5, label=f"TDP = {TDP_W:.0f} W")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(m)}" for m in avg_power["m"]], fontsize=10)
    ax.set_xlabel("Stacked DRAM Layers (m = n, All Connected)", fontsize=12)
    ax.set_ylabel("Power per Device (W)", fontsize=12)
    ax.set_title("Power Breakdown vs DRAM Layers (Best-STPS Configs, Avg over BS)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Label total power
    for i, row in avg_power.reset_index(drop=True).iterrows():
        total = row["chip_power"] + row["noc_power"]
        ax.text(i, total + 1, f"{total:.1f}W", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"power_breakdown_vs_m.{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved power_breakdown_vs_m.{{png,pdf}}")


def plot_power_vs_stps_scatter(df_mn, out_dir):
    """Figure 2: Total power vs scaled STPS scatter, colored by m, one subplot per BS."""
    bs_vals = sorted(df_mn["bs"].unique())
    n_bs = len(bs_vals)
    ncols = min(4, n_bs)
    nrows = (n_bs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    m_vals = sorted(df_mn["dram_total_layers"].unique())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(m_vals)))
    m_to_color = {m: cmap[i] for i, m in enumerate(m_vals)}

    for idx, bs in enumerate(bs_vals):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub = df_mn[df_mn["bs"] == bs]

        for m in m_vals:
            ms = sub[sub["dram_total_layers"] == m]
            ax.scatter(ms["scaled_stps_avg"], ms["total_power_per_device_W"],
                       s=12, alpha=0.5, color=m_to_color[m], label=f"m={m}")

        ax.axhline(y=TDP_W, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"BS={bs}", fontsize=11)
        ax.set_xlabel("Scaled STPS (tokens/s)", fontsize=9)
        ax.set_ylabel("Total Power/Device (W)", fontsize=9)
        ax.grid(alpha=0.3)

    # Remove unused subplots
    for idx in range(n_bs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(m_vals), 6),
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Power vs Scaled STPS by Batch Size", fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"power_vs_stps_scatter.{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved power_vs_stps_scatter.{{png,pdf}}")


def plot_temp_vs_stps_scatter(df_mn, out_dir):
    """Figure 2b: Junction temperature vs scaled STPS scatter, one subplot per BS.

    T_junction = T_ambient + R_thermal(m) × P_total_per_device
    R_thermal comes from the CSV column 'thermal_resistance_CpW'.
    """
    df = df_mn.copy()
    df["temp_C"] = T_AMBIENT_C + df["thermal_resistance_CpW"] * df["total_power_per_device_W"]

    bs_vals = sorted(df["bs"].unique())
    n_bs = len(bs_vals)
    ncols = min(4, n_bs)
    nrows = (n_bs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    m_vals = sorted(df["dram_total_layers"].unique())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(m_vals)))
    m_to_color = {m: cmap[i] for i, m in enumerate(m_vals)}

    for idx, bs in enumerate(bs_vals):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub = df[df["bs"] == bs]

        for m in m_vals:
            ms = sub[sub["dram_total_layers"] == m]
            ax.scatter(ms["scaled_stps_avg"], ms["temp_C"],
                       s=12, alpha=0.5, color=m_to_color[m], label=f"m={m}")

        ax.axhline(y=T_THROTTLE_C, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"Throttle = {T_THROTTLE_C:.0f} °C")
        ax.set_title(f"BS={bs}", fontsize=11)
        ax.set_xlabel("Scaled STPS (tokens/s)", fontsize=9)
        ax.set_ylabel("Junction Temperature (°C)", fontsize=9)
        ax.grid(alpha=0.3)

    for idx in range(n_bs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(m_vals) + 1, 7),
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Junction Temperature vs Scaled STPS by Batch Size (T_amb={T_AMBIENT_C:.0f} °C)",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"temp_vs_stps_scatter.{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved temp_vs_stps_scatter.{{png,pdf}}")


def plot_power_boxplot(df_mn, out_dir):
    """Figure 3: Box plot of total power per device, grouped by m."""
    m_vals = sorted(df_mn["dram_total_layers"].unique())
    data = [df_mn[df_mn["dram_total_layers"] == m]["total_power_per_device_W"].values for m in m_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, positions=range(len(m_vals)), widths=0.5, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='.', markersize=2, alpha=0.3))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(m_vals)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=TDP_W, color="red", linestyle="--", linewidth=1.5, label=f"TDP = {TDP_W:.0f} W")
    ax.set_xticks(range(len(m_vals)))
    ax.set_xticklabels([f"m={m}" for m in m_vals], fontsize=10)
    ax.set_xlabel("Stacked DRAM Layers (m = n)", fontsize=12)
    ax.set_ylabel("Total Power per Device (W)", fontsize=12)
    ax.set_title("Power Distribution vs DRAM Layers (All Configs)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"power_distribution_by_m.{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved power_distribution_by_m.{{png,pdf}}")


def plot_power_wall_heatmap(df_mn, out_dir):
    """Figure 4: Heatmap of power wall hit rate (m vs BS)."""
    pivot = df_mn.groupby(["dram_total_layers", "bs"])["hit_power_wall"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"BS={int(b)}" for b in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"m={int(m)}" for m in pivot.index], fontsize=10)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Stacked DRAM Layers", fontsize=12)
    ax.set_title("Power Wall Hit Rate (m vs BS)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Fraction Hitting Power Wall")

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"power_wall_heatmap.{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved power_wall_heatmap.{{png,pdf}}")


def main(run_dir):
    run_dir = os.path.abspath(run_dir)
    out_dir = os.path.join(run_dir, "power_analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Decode Power Distribution Plot for {run_dir} ===")
    df_mn = load_data(run_dir)
    print(f"  Loaded {len(df_mn)} rows (m==n, BS<={MAX_BS})")

    plot_power_breakdown_bar(df_mn, out_dir)
    plot_power_vs_stps_scatter(df_mn, out_dir)
    plot_temp_vs_stps_scatter(df_mn, out_dir)
    plot_power_boxplot(df_mn, out_dir)
    plot_power_wall_heatmap(df_mn, out_dir)

    print(f"=== Done. Results in {out_dir}/ ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 decode_dse_power_plot.py <run_dir>")
        print("  e.g.: python3 decode_dse_power_plot.py runs/your-run-id")
        sys.exit(1)
    main(sys.argv[1])
