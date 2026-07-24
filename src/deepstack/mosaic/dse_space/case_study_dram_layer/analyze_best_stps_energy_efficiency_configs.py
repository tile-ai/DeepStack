#!/usr/bin/env python3
"""
Analyze DRAM layer DSE results:
1. Best performance configs per bs (prefill & decode)
2. Most power-efficient configs per bs (tokens/J with 100W cap)
3. Tokens/J distribution plots

Note: tokens/J = stps (tokens/s) / system_power (per_device_W * 256 devices = J/s)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

POWER_CAP = 100.0  # W per device
NUM_DEVICES = 256  # total devices in the system
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures_tokens_per_joule")
DATA_DIR = os.path.join(BASE_DIR, "lfs")
os.makedirs(FIG_DIR, exist_ok=True)

# Config columns to display
ID_COLS = [
    "dram_total_layers", "dram_active_layers", "sm_count", "smem_capacity_KiB",
    "l1_throughput_Bpc", "ddr_peak_bw_TBs", "ddr_capacity_GB",
    "arch", "noc", "bs", "tp", "ep", "dp", "pp",
]


def load_prefill():
    return pd.read_csv(os.path.join(DATA_DIR, "dram_layer_prefill_result_dpsk_mn1to12_seq1024.csv"))


def load_decode():
    return pd.read_csv(os.path.join(DATA_DIR, "dram_layer_decode_result_dpsk_mn1to12.csv"))


def compute_effective_power_and_efficiency(df, stps_col, raw_stps_col):
    """
    Power cap logic:
    - If total_power_per_device_W > 100W: freq scaled down → power becomes 100W,
      performance = scaled_stps (already reflects freq scaling)
    - If total_power_per_device_W <= 100W: power as-is, performance = raw_stps
    tokens_per_joule = effective_stps / (effective_power_per_device * NUM_DEVICES)
    """
    df = df.copy()
    over = df["total_power_per_device_W"] > POWER_CAP

    df["effective_power_W"] = df["total_power_per_device_W"].copy()
    df.loc[over, "effective_power_W"] = POWER_CAP

    df["effective_stps"] = df[raw_stps_col].copy()
    df.loc[over, "effective_stps"] = df.loc[over, stps_col]

    # System-level: total power = per-device power * NUM_DEVICES
    df["effective_system_power_W"] = df["effective_power_W"] * NUM_DEVICES
    df["tokens_per_joule"] = df["effective_stps"] / df["effective_system_power_W"]
    return df


def print_config(row, extra_cols):
    id_vals = {c: row[c] for c in ID_COLS if c in row.index}
    print(f"  Config: {id_vals}")
    for c in extra_cols:
        if c in row.index:
            val = row[c]
            print(f"  {c}: {val:.4f}" if isinstance(val, float) else f"  {c}: {val}")
    print()


def find_best_perf(df, stps_col, label):
    print(f"\n{'='*90}")
    print(f"  BEST PERFORMANCE — {label} (by {stps_col})")
    print(f"{'='*90}")
    for bs in sorted(df["bs"].unique()):
        grp = df[df["bs"] == bs]
        idx = grp[stps_col].idxmax()
        row = grp.loc[idx]
        print(f"\n[BS={bs}]  {stps_col} = {row[stps_col]:.2f}")
        print_config(row, [stps_col, "total_power_per_device_W", "effective_stps",
                           "effective_power_W", "tokens_per_joule", "total_freq_scale"])


def find_best_efficiency(df, label):
    print(f"\n{'='*90}")
    print(f"  BEST POWER EFFICIENCY — {label} (tokens/J, power cap={POWER_CAP}W)")
    print(f"{'='*90}")
    for bs in sorted(df["bs"].unique()):
        grp = df[df["bs"] == bs]
        idx = grp["tokens_per_joule"].idxmax()
        row = grp.loc[idx]
        throttled = " [THROTTLED]" if row["total_power_per_device_W"] > POWER_CAP else ""
        print(f"\n[BS={bs}]  tokens/J = {row['tokens_per_joule']:.4f}{throttled}")
        print_config(row, ["effective_stps", "total_power_per_device_W", "effective_power_W",
                           "tokens_per_joule", "total_freq_scale"])


def find_best_effective_perf(df, label):
    """Best performance under 100W cap."""
    print(f"\n{'='*90}")
    print(f"  BEST EFFECTIVE PERFORMANCE — {label} (highest effective_stps under {POWER_CAP}W cap)")
    print(f"{'='*90}")
    for bs in sorted(df["bs"].unique()):
        grp = df[df["bs"] == bs]
        idx = grp["effective_stps"].idxmax()
        row = grp.loc[idx]
        throttled = " [THROTTLED]" if row["total_power_per_device_W"] > POWER_CAP else ""
        print(f"\n[BS={bs}]  effective_stps = {row['effective_stps']:.2f}{throttled}")
        print_config(row, ["effective_stps", "total_power_per_device_W", "effective_power_W",
                           "tokens_per_joule", "total_freq_scale"])


def plot_tokens_per_joule_distribution(df, label, filename_prefix):
    """Histogram of tokens/J per bs."""
    bs_values = sorted(df["bs"].unique())
    n = len(bs_values)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    for i, bs in enumerate(bs_values):
        ax = axes[i // ncols][i % ncols]
        data = df[df["bs"] == bs]["tokens_per_joule"].dropna()
        ax.hist(data, bins=60, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_title(f"bs={bs}  (n={len(data)})")
        ax.set_xlabel("tokens/J")
        ax.set_ylabel("count")
        med = data.median()
        mx = data.max()
        ax.axvline(med, color="red", linestyle="--", linewidth=1.2, label=f"median={med:.3f}")
        ax.axvline(mx, color="green", linestyle="--", linewidth=1.2, label=f"max={mx:.3f}")
        ax.legend(fontsize=7)

    # hide unused axes
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(f"{label} — tokens/J distribution (power cap={POWER_CAP}W)", fontsize=14, y=1.01)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{filename_prefix}_tokens_per_joule_dist.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename_prefix}_tokens_per_joule_dist.png/pdf")


def plot_perf_vs_efficiency(df, stps_col, label, filename_prefix):
    """Scatter: effective_stps vs tokens/J, colored by bs."""
    bs_values = sorted(df["bs"].unique())
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.tab10
    for i, bs in enumerate(bs_values):
        sub = df[df["bs"] == bs]
        ax.scatter(sub["tokens_per_joule"], sub["effective_stps"],
                   alpha=0.25, s=8, color=cmap(i % 10), label=f"bs={bs}")
    ax.set_xlabel("tokens/J")
    ax.set_ylabel("effective stps")
    ax.set_title(f"{label} — Performance vs Power Efficiency")
    ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{filename_prefix}_perf_vs_eff.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename_prefix}_perf_vs_eff.png/pdf")


def plot_power_distribution(df, label, filename_prefix):
    """Histogram of original per-device power with 100W line."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["total_power_per_device_W"], bins=80, edgecolor="black", alpha=0.7, color="coral")
    ax.axvline(POWER_CAP, color="red", linewidth=2, linestyle="--", label=f"Power cap={POWER_CAP}W")
    pct_over = (df["total_power_per_device_W"] > POWER_CAP).mean() * 100
    ax.set_title(f"{label} — Per-device power distribution ({pct_over:.1f}% configs over {POWER_CAP}W)")
    ax.set_xlabel("total_power_per_device_W")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{filename_prefix}_power_dist.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename_prefix}_power_dist.png/pdf")


def plot_combined_tokens_per_joule_violin(df, label, filename_prefix):
    """Violin plot of tokens/J across all bs values."""
    bs_values = sorted(df["bs"].unique())
    data_list = [df[df["bs"] == bs]["tokens_per_joule"].dropna().values for bs in bs_values]

    fig, ax = plt.subplots(figsize=(max(8, len(bs_values) * 1.2), 6))
    parts = ax.violinplot(data_list, positions=range(len(bs_values)), showmedians=True, showextrema=True)
    ax.set_xticks(range(len(bs_values)))
    ax.set_xticklabels([f"bs={bs}" for bs in bs_values], rotation=45, ha="right")
    ax.set_ylabel("tokens/J")
    ax.set_title(f"{label} — tokens/J violin plot (power cap={POWER_CAP}W)")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{filename_prefix}_tokens_per_joule_violin.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename_prefix}_tokens_per_joule_violin.png/pdf")


def analyze_by_mn(df, stps_col, label, prefix):
    """Analyze best perf and efficiency broken down by (m, n) = (dram_total_layers, dram_active_layers)."""
    m_col = "dram_total_layers"
    n_col = "dram_active_layers"

    print(f"\n{'='*90}")
    print(f"  ANALYSIS BY (m, n) = (dram_total_layers, dram_active_layers) — {label}")
    print(f"{'='*90}")

    mn_pairs = sorted(df[[m_col, n_col]].drop_duplicates().values.tolist())
    print(f"  Total (m, n) combos: {len(mn_pairs)}")

    bs_values = sorted(df["bs"].unique())

    # --- Per (m,n) best perf and efficiency across all bs ---
    print(f"\n--- Best perf & efficiency per (m, n), aggregated across all bs ---")
    print(f"{'m':>3s} {'n':>3s} | {'best_stps':>12s} {'@bs':>5s} | {'best_tok/J':>12s} {'@bs':>5s} | {'mean_tok/J':>12s} {'count':>7s}")
    print("-" * 80)
    for m, n in mn_pairs:
        sub = df[(df[m_col] == m) & (df[n_col] == n)]
        best_perf_idx = sub[stps_col].idxmax()
        best_eff_idx = sub["tokens_per_joule"].idxmax()
        best_perf_row = sub.loc[best_perf_idx]
        best_eff_row = sub.loc[best_eff_idx]
        mean_tpw = sub["tokens_per_joule"].mean()
        print(f"{m:>3d} {n:>3d} | {best_perf_row[stps_col]:>12.2f} {int(best_perf_row['bs']):>5d} | "
              f"{best_eff_row['tokens_per_joule']:>12.4f} {int(best_eff_row['bs']):>5d} | "
              f"{mean_tpw:>12.4f} {len(sub):>7d}")

    # --- Per (m,n) per bs ---
    print(f"\n--- Best perf per (m, n) per bs ---")
    for bs in bs_values:
        print(f"\n  [BS={bs}]")
        print(f"  {'m':>3s} {'n':>3s} | {'best_stps':>12s} | {'best_tok/J':>12s} | {'mean_tok/J':>12s} {'count':>6s}")
        print(f"  " + "-" * 70)
        bs_sub = df[df["bs"] == bs]
        for m, n in mn_pairs:
            sub = bs_sub[(bs_sub[m_col] == m) & (bs_sub[n_col] == n)]
            if len(sub) == 0:
                continue
            best_stps = sub[stps_col].max()
            best_tpw = sub["tokens_per_joule"].max()
            mean_tpw = sub["tokens_per_joule"].mean()
            print(f"  {m:>3d} {n:>3d} | {best_stps:>12.2f} | {best_tpw:>12.4f} | {mean_tpw:>12.4f} {len(sub):>6d}")

    # --- Heatmap: best tokens/J per (m, n) for each bs ---
    plot_mn_heatmaps(df, stps_col, "tokens_per_joule", label, prefix, metric_label="best tokens/J")
    plot_mn_heatmaps(df, stps_col, stps_col, label, prefix, metric_label="best stps")


def plot_mn_heatmaps(df, stps_col, metric_col, label, prefix, metric_label="best tokens/J"):
    """Heatmap of best metric per (m, n) for each bs."""
    m_col = "dram_total_layers"
    n_col = "dram_active_layers"
    bs_values = sorted(df["bs"].unique())
    n_bs = len(bs_values)
    ncols = min(n_bs, 4)
    nrows = (n_bs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)

    for i, bs in enumerate(bs_values):
        ax = axes[i // ncols][i % ncols]
        bs_sub = df[df["bs"] == bs]
        pivot = bs_sub.groupby([m_col, n_col])[metric_col].max().reset_index()
        pivot_table = pivot.pivot(index=m_col, columns=n_col, values=metric_col)
        # Sort axes
        pivot_table = pivot_table.sort_index(ascending=True)
        pivot_table = pivot_table[sorted(pivot_table.columns)]

        im = ax.imshow(pivot_table.values, aspect="auto", cmap="YlOrRd",
                       origin="lower")
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns.astype(int), fontsize=7)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index.astype(int), fontsize=7)
        ax.set_xlabel("n (active layers)")
        ax.set_ylabel("m (total layers)")
        ax.set_title(f"bs={bs}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells
        for yi in range(len(pivot_table.index)):
            for xi in range(len(pivot_table.columns)):
                val = pivot_table.values[yi, xi]
                if not np.isnan(val):
                    ax.text(xi, yi, f"{val:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if val > pivot_table.values[~np.isnan(pivot_table.values)].mean() else "black")

    for i in range(n_bs, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    safe_label = metric_label.replace("/", "_per_").replace(" ", "_")
    fig.suptitle(f"{label} — {metric_label} by (m, n)", fontsize=14, y=1.01)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{prefix}_mn_heatmap_{safe_label}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {prefix}_mn_heatmap_{safe_label}.png/pdf")


def analyze_phase(df, phase_name, stps_col, raw_stps_col, prefix):
    print(f"\n{'#'*90}")
    print(f"  {phase_name.upper()}")
    print(f"{'#'*90}")
    print(f"Total rows: {len(df)}, unique bs values: {sorted(df['bs'].unique())}")

    df = compute_effective_power_and_efficiency(df, stps_col, raw_stps_col)

    # Print summary stats per bs
    print(f"\n--- Summary stats per bs ---")
    for bs in sorted(df["bs"].unique()):
        grp = df[df["bs"] == bs]
        tpw = grp["tokens_per_joule"]
        print(f"  bs={bs:>3d}: n={len(grp):>6d}  "
              f"tokens/J  min={tpw.min():.4f}  median={tpw.median():.4f}  "
              f"mean={tpw.mean():.4f}  max={tpw.max():.4f}  "
              f"pct_throttled={100*(grp['total_power_per_device_W']>POWER_CAP).mean():.1f}%")

    find_best_perf(df, stps_col, phase_name)
    find_best_efficiency(df, phase_name)
    find_best_effective_perf(df, phase_name)

    # (m, n) analysis
    analyze_by_mn(df, stps_col, phase_name, prefix)

    # Plots
    print(f"\n--- Generating plots ---")
    plot_tokens_per_joule_distribution(df, phase_name, prefix)
    plot_perf_vs_efficiency(df, stps_col, phase_name, prefix)
    plot_power_distribution(df, phase_name, prefix)
    plot_combined_tokens_per_joule_violin(df, phase_name, prefix)

    return df


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    pf = load_prefill()
    analyze_phase(pf, "Prefill (seq=1024)", "scaled_stps", "raw_stps", "dram_layer_prefill")

    print("\n\n")

    dc = load_decode()
    analyze_phase(dc, "Decode", "scaled_stps_avg", "raw_stps_avg", "dram_layer_decode")

    print(f"\n\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
