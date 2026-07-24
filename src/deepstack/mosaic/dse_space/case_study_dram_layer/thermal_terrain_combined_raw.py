"""
thermal_terrain_combined_raw.py — Same as thermal_terrain_combined.py but uses raw STPS
(before power-wall frequency scaling).

Usage:
    python3 thermal_terrain_combined_raw.py

Reads from lfs/:
    - dram_layer_prefill_result_dpsk_mn1to12_seq1024.csv
    - dram_layer_decode_result_dpsk_mn1to12.csv

Outputs into lfs/:
    - thermal_terrain_combined_raw.{png,pdf}
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata

# ── Config ────────────────────────────────────────────────────────────────
T_AMBIENT_C = 35.0
T_WARNING_C = 85.0
T_THROTTLE_C = 95.0
BS_LIST = [4, 1024]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LFS_DIR = os.path.join(BASE_DIR, "lfs")
PREFILL_CSV = os.path.join(LFS_DIR, "dram_layer_prefill_result_dpsk_mn1to12_seq1024.csv")
DECODE_CSV = os.path.join(LFS_DIR, "dram_layer_decode_result_dpsk_mn1to12.csv")

# ── Publication style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
})


def _style_3d_ax(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgrey")
    ax.yaxis.pane.set_edgecolor("lightgrey")
    ax.zaxis.pane.set_edgecolor("lightgrey")
    ax.grid(True, alpha=0.2, linewidth=0.3)


def load_prefill():
    df = pd.read_csv(PREFILL_CSV)
    df = df[df["dram_total_layers"] == df["dram_active_layers"]].copy()
    df["temp_C"] = T_AMBIENT_C + df["thermal_resistance_CpW"] * df["total_power_per_device_W"]
    df["stps"] = df["raw_stps"]
    return df


def load_decode():
    df = pd.read_csv(DECODE_CSV)
    df = df[df["dram_total_layers"] == df["dram_active_layers"]].copy()
    df["temp_C"] = T_AMBIENT_C + df["thermal_resistance_CpW"] * df["total_power_per_device_W"]
    df["stps"] = df["raw_stps_avg"]
    return df


def _plot_one_panel(ax, sub, title, zlim=None):
    """Draw a single 3D terrain panel."""
    if sub.empty:
        ax.set_title(title)
        return

    x = sub["stps"].values
    y = sub["dram_total_layers"].values.astype(float)
    z = sub["temp_C"].values

    norm = plt.Normalize(35, 130)

    # Grid interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")

    facecolors = cm.coolwarm(norm(np.nan_to_num(Zi, nan=35)))

    ax.plot_surface(Xi, Yi, Zi, facecolors=facecolors, alpha=0.85,
                    rstride=2, cstride=2, linewidth=0, antialiased=True,
                    shade=True)

    # 95°C contour
    try:
        ax.contour(Xi, Yi, Zi, levels=[T_THROTTLE_C], colors=["red"], linewidths=1.5)
    except Exception:
        pass

    # Scatter data points
    ax.scatter(x, y, z, c=cm.coolwarm(norm(z)), s=3, alpha=0.3,
               edgecolors="none", depthshade=True)

    # 85°C warning plane
    x_r = [x.min(), x.max()]
    y_r = [y.min() - 0.5, y.max() + 0.5]
    verts_85 = [[(x_r[0], y_r[0], T_WARNING_C), (x_r[1], y_r[0], T_WARNING_C),
                 (x_r[1], y_r[1], T_WARNING_C), (x_r[0], y_r[1], T_WARNING_C)]]
    plane_85 = Poly3DCollection(verts_85, alpha=0.18, facecolor="orange",
                                edgecolor="darkorange", linewidth=1.0)
    ax.add_collection3d(plane_85)
    ax.text(x_r[0] + (x_r[1] - x_r[0]) * 0.55, y_r[1], T_WARNING_C + 1.2,
            f"{T_WARNING_C:.0f} °C", color="darkorange", fontsize=9, fontstyle="italic",
            ha="center")

    ax.set_xlabel("STPS (tok/s)", labelpad=2)
    ax.set_ylabel("DRAM Layers", labelpad=2)
    ax.set_zlabel("Temp (°C)", labelpad=2)
    m_vals = sorted(sub["dram_total_layers"].unique())
    ax.set_yticks(m_vals)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, p: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))

    if zlim is not None:
        ax.set_zlim(zlim)
    ax.set_title(title, fontweight="bold", y=0.95)
    ax.view_init(elev=28, azim=-50)
    ax.set_box_aspect((1.35, 1.0, 0.8), zoom=1.15)
    _style_3d_ax(ax)


def _make_figure(panels_spec, out_dir, filename):
    """Create a 1×2 terrain figure.

    panels_spec: list of (df, bs, title)
    """
    fig = plt.figure(figsize=(10.8, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.0)

    # Compute shared z-axis range
    all_temps = [df[df["bs"] == bs]["temp_C"] for df, bs, _ in panels_spec]
    z_min = min(t.min() for t in all_temps) - 2
    z_max = max(t.max() for t in all_temps) + 2
    shared_zlim = (z_min, z_max)

    for ax, (df, bs, title) in zip([
        fig.add_subplot(gs[0, 0], projection="3d"),
        fig.add_subplot(gs[0, 1], projection="3d"),
    ], panels_spec):
        sub = df[df["bs"] == bs]
        _plot_one_panel(ax, sub, title, zlim=shared_zlim)

    # Shared colorbar
    norm = plt.Normalize(35, 130)
    sm = cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Temperature (°C)")

    fig.subplots_adjust(left=0.01, right=0.97, top=0.94, bottom=0.03)
    cbar_pos = cbar_ax.get_position()
    cbar_ax.set_position([cbar_pos.x0 + 0.03, cbar_pos.y0, cbar_pos.width, cbar_pos.height])

    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{filename}.{fmt}"),
                    dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {filename}.{{png,pdf}}")


def main():
    out_dir = LFS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("=== Combined Thermal Terrain (Raw STPS) ===")
    df_prefill = load_prefill()
    df_decode = load_decode()
    print(f"  Prefill: {len(df_prefill)} rows, Decode: {len(df_decode)} rows")

    # Figure 1: Prefill BS=1024 + Decode BS=1024
    _make_figure([
        (df_prefill, 1024, "Prefill, BS = 1024"),
        (df_decode,  1024, "Decode, BS = 1024"),
    ], out_dir, "thermal_terrain_raw_pf_dc_bs1024")

    # Figure 2: Decode BS=4 + Decode BS=1024
    _make_figure([
        (df_decode, 4,    "Decode, BS = 4"),
        (df_decode, 1024, "Decode, BS = 1024"),
    ], out_dir, "thermal_terrain_raw_dc_bs4_1024")

    print(f"=== Done. Results in {out_dir}/ ===")


if __name__ == "__main__":
    main()
