#!/usr/bin/env python3
"""
3D visualization ideas for DRAM layer DSE — targeting MICRO/ASPLOS.
x = m (total layers), y = n (active layers), z = throughput or efficiency.
Even m,n only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
import os

# ── Constants ──────────────────────────────────────────────────────────────
POWER_CAP = 100.0
NUM_DEVICES = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures_3d")
DATA_DIR = os.path.join(BASE_DIR, "lfs")
os.makedirs(FIG_DIR, exist_ok=True)

EVEN_M_VALUES = [2, 4, 6, 8, 10, 12, 14]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BS_COLORS = {
    1: "#1f77b4", 4: "#ff7f0e", 16: "#2ca02c", 64: "#d62728",
    256: "#9467bd", 1024: "#8c564b", 4096: "#e377c2", 16384: "#7f7f7f",
}


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
# 3D-A: Surface plot — smooth surface on (m, n) grid, z = throughput
#   One subplot per bs, with wireframe + color surface.
# ======================================================================
def fig3d_surface(pf, dc):
    """Smooth 3D surface plot per bs, faceted."""
    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        bs_vals = sorted(df["bs"].unique())
        ncols = min(len(bs_vals), 4)
        nrows = (len(bs_vals) + ncols - 1) // ncols
        fig = plt.figure(figsize=(5.0 * ncols, 4.5 * nrows))

        for i, bs in enumerate(bs_vals):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            m_pts = best["dram_total_layers"].values.astype(float)
            n_pts = best["dram_active_layers"].values.astype(float)
            z_pts = best["effective_stps"].values

            # Fine grid for interpolation
            m_fine = np.linspace(2, 14, 40)
            n_fine = np.linspace(2, 12, 35)
            M, N = np.meshgrid(m_fine, n_fine)

            try:
                Z = griddata((m_pts, n_pts), z_pts, (M, N), method="cubic")
                # Mask n > m
                Z[N > M] = np.nan

                surf = ax.plot_surface(M, N, Z, cmap="YlOrRd", alpha=0.85,
                                       edgecolor="gray", linewidth=0.2,
                                       rstride=2, cstride=2, antialiased=True)
            except Exception:
                ax.scatter(m_pts, n_pts, z_pts, c=z_pts, cmap="YlOrRd", s=30)

            # Mark the best point
            best_row = best.loc[best["effective_stps"].idxmax()]
            ax.scatter([best_row["dram_total_layers"]], [best_row["dram_active_layers"]],
                       [best_row["effective_stps"]], s=120, c="blue", marker="*",
                       edgecolors="white", linewidths=1, zorder=10, depthshade=False)

            # Also plot raw data points
            ax.scatter(m_pts, n_pts, z_pts, c="black", s=8, alpha=0.5, depthshade=False)

            ax.set_xlabel("m (total)", fontsize=8, labelpad=2)
            ax.set_ylabel("n (active)", fontsize=8, labelpad=2)
            ax.set_zlabel("Throughput", fontsize=8, labelpad=2)
            ax.set_title(f"bs={bs}", fontweight="bold", fontsize=10)
            ax.view_init(elev=25, azim=-135)
            ax.tick_params(labelsize=6)

        # Hide unused
        fig.suptitle(f"{phase_name}: Throughput Surface over (m, n)",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save(fig, f"3dA_{prefix}_surface")


# ======================================================================
# 3D-B: Bar3D plot — 3D bar chart on discrete (m, n) grid
#   Bars colored by throughput, easy to read exact values.
# ======================================================================
def fig3d_bar(pf, dc):
    """3D bar chart on discrete (m, n) grid."""
    even_vals = EVEN_M_VALUES

    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        bs_vals = sorted(df["bs"].unique())
        ncols = min(len(bs_vals), 4)
        nrows = (len(bs_vals) + ncols - 1) // ncols
        fig = plt.figure(figsize=(5.0 * ncols, 4.5 * nrows))

        for i, bs in enumerate(bs_vals):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            m_vals = best["dram_total_layers"].values
            n_vals = best["dram_active_layers"].values
            z_vals = best["effective_stps"].values

            # Normalize colors
            z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-10)
            colors = cm.YlOrRd(z_norm)

            bar_width = 1.5
            ax.bar3d(m_vals - bar_width/2, n_vals - bar_width/2,
                     np.zeros_like(z_vals), bar_width, bar_width, z_vals,
                     color=colors, edgecolor="black", linewidth=0.3, alpha=0.85)

            # Mark best
            best_idx = z_vals.argmax()
            ax.scatter([m_vals[best_idx]], [n_vals[best_idx]], [z_vals[best_idx] * 1.05],
                       s=150, c="blue", marker="*", edgecolors="white",
                       linewidths=1, zorder=10, depthshade=False)

            ax.set_xlabel("m (total)", fontsize=8, labelpad=2)
            ax.set_ylabel("n (active)", fontsize=8, labelpad=2)
            ax.set_zlabel("Throughput", fontsize=8, labelpad=2)
            ax.set_title(f"bs={bs}", fontweight="bold", fontsize=10)
            ax.view_init(elev=30, azim=-130)
            ax.tick_params(labelsize=6)

        fig.suptitle(f"{phase_name}: 3D Bar Chart — Throughput by (m, n)",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save(fig, f"3dB_{prefix}_bar3d")


# ======================================================================
# 3D-C: Wireframe comparison — overlay multiple bs on same 3D plot
#   Different colored wireframes for each bs to show how the landscape
#   changes with batch size.
# ======================================================================
def fig3d_wireframe_overlay(pf, dc):
    """Overlay wireframes for different bs on same axes."""
    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        bs_vals = sorted(df["bs"].unique())
        # Pick representative bs values to avoid clutter
        if len(bs_vals) > 4:
            rep_bs = [bs_vals[0], bs_vals[len(bs_vals)//3],
                      bs_vals[2*len(bs_vals)//3], bs_vals[-1]]
        else:
            rep_bs = bs_vals

        m_fine = np.linspace(2, 14, 30)
        n_fine = np.linspace(2, 12, 25)
        M, N = np.meshgrid(m_fine, n_fine)

        for bs in rep_bs:
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            m_pts = best["dram_total_layers"].values.astype(float)
            n_pts = best["dram_active_layers"].values.astype(float)
            z_pts = best["effective_stps"].values

            # Normalize z for comparison
            z_max = z_pts.max()
            z_pts_norm = z_pts / z_max if z_max > 0 else z_pts

            try:
                Z = griddata((m_pts, n_pts), z_pts_norm, (M, N), method="cubic")
                Z[N > M] = np.nan
                c = BS_COLORS.get(bs, "#333")
                ax.plot_wireframe(M, N, Z, color=c, linewidth=0.8, alpha=0.7,
                                  rstride=3, cstride=3, label=f"bs={bs}")
            except Exception:
                pass

        ax.set_xlabel("m (total layers)", fontsize=10, labelpad=5)
        ax.set_ylabel("n (active layers)", fontsize=10, labelpad=5)
        ax.set_zlabel("Normalized Throughput", fontsize=10, labelpad=5)
        ax.set_title(f"{phase_name}: Throughput Landscape Shape Across Batch Sizes",
                     fontweight="bold", fontsize=12)
        ax.view_init(elev=25, azim=-135)
        ax.legend(loc="upper left", fontsize=8)
        _save(fig, f"3dC_{prefix}_wireframe_overlay")


# ======================================================================
# 3D-D: Scatter3D — all (m, n, throughput) points, colored by n/m ratio
#   With projection shadows on the walls for readability.
# ======================================================================
def fig3d_scatter_projections(pf, dc):
    """3D scatter with projections on xy, xz, yz planes."""
    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        bs_vals = sorted(df["bs"].unique())
        ncols = min(len(bs_vals), 4)
        nrows = (len(bs_vals) + ncols - 1) // ncols
        fig = plt.figure(figsize=(5.5 * ncols, 5.0 * nrows))

        for i, bs in enumerate(bs_vals):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            m = best["dram_total_layers"].values.astype(float)
            n = best["dram_active_layers"].values.astype(float)
            z = best["effective_stps"].values
            ratio = n / m

            # Main scatter
            sc = ax.scatter(m, n, z, c=ratio, cmap="coolwarm", s=60,
                            edgecolors="black", linewidths=0.5, vmin=0, vmax=1,
                            depthshade=False, zorder=5)

            # Project shadows onto walls
            z_floor = ax.get_zlim()[0] if len(z) > 0 else 0
            # Shadow on floor (xy plane at z=0)
            ax.scatter(m, n, np.full_like(z, z.min() * 0.9), c="gray", s=15, alpha=0.3, depthshade=False)
            # Shadow on back wall (xz plane at n=max)
            ax.scatter(m, np.full_like(n, 13), z, c="gray", s=15, alpha=0.3, depthshade=False)
            # Shadow on side wall (yz plane at m=min)
            ax.scatter(np.full_like(m, 1), n, z, c="gray", s=15, alpha=0.3, depthshade=False)

            # Connect vertical lines from floor
            for mi, ni, zi in zip(m, n, z):
                ax.plot([mi, mi], [ni, ni], [z.min() * 0.9, zi],
                        color="gray", linewidth=0.3, alpha=0.4)

            # Mark best
            best_idx = z.argmax()
            ax.scatter([m[best_idx]], [n[best_idx]], [z[best_idx]],
                       s=200, c="red", marker="*", edgecolors="black",
                       linewidths=1, zorder=10, depthshade=False)
            ax.text(m[best_idx], n[best_idx], z[best_idx] * 1.05,
                    f"({int(m[best_idx])},{int(n[best_idx])})",
                    fontsize=7, fontweight="bold", color="red")

            ax.set_xlabel("m", fontsize=8, labelpad=1)
            ax.set_ylabel("n", fontsize=8, labelpad=1)
            ax.set_zlabel("tok/s", fontsize=8, labelpad=1)
            ax.set_title(f"bs={bs}", fontweight="bold", fontsize=10)
            ax.view_init(elev=25, azim=-130)
            ax.tick_params(labelsize=6)

        fig.suptitle(f"{phase_name}: 3D Scatter with Projections",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save(fig, f"3dD_{prefix}_scatter_proj")


# ======================================================================
# 3D-E: Terrain / surface with gradient arrows pointing to optimal
#   Shows the optimization landscape with gradient flow.
# ======================================================================
def fig3d_terrain_gradient(pf, dc):
    """Surface plot with gradient arrows showing direction of improvement."""
    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        bs_vals = sorted(df["bs"].unique())
        # Pick 4 representative bs
        if len(bs_vals) > 4:
            rep_bs = [bs_vals[0], bs_vals[len(bs_vals)//3],
                      bs_vals[2*len(bs_vals)//3], bs_vals[-1]]
        else:
            rep_bs = bs_vals

        fig = plt.figure(figsize=(5.5 * min(len(rep_bs), 2), 5.0 * ((len(rep_bs) + 1) // 2)))
        ncols = min(len(rep_bs), 2)
        nrows = (len(rep_bs) + ncols - 1) // ncols

        for i, bs in enumerate(rep_bs):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            m_pts = best["dram_total_layers"].values.astype(float)
            n_pts = best["dram_active_layers"].values.astype(float)
            z_pts = best["effective_stps"].values

            m_fine = np.linspace(2, 14, 40)
            n_fine = np.linspace(2, 12, 35)
            M, N = np.meshgrid(m_fine, n_fine)

            try:
                Z = griddata((m_pts, n_pts), z_pts, (M, N), method="cubic")
                Z[N > M] = np.nan

                # Surface
                surf = ax.plot_surface(M, N, Z, cmap="terrain", alpha=0.75,
                                       edgecolor="none", rstride=2, cstride=2,
                                       antialiased=True)

                # Compute gradient on the discrete data for arrows
                # Use coarser grid for arrows
                m_arrow = np.linspace(3, 13, 8)
                n_arrow = np.linspace(3, 11, 7)
                Ma, Na = np.meshgrid(m_arrow, n_arrow)
                Za = griddata((m_pts, n_pts), z_pts, (Ma, Na), method="cubic")

                dm = m_arrow[1] - m_arrow[0]
                dn = n_arrow[1] - n_arrow[0]
                grad_m, grad_n = np.gradient(Za, dm, dn)

                # Plot gradient arrows on the surface
                for yi in range(len(n_arrow)):
                    for xi in range(len(m_arrow)):
                        if Na[yi, xi] <= Ma[yi, xi] and not np.isnan(Za[yi, xi]):
                            gm = grad_m[yi, xi] if not np.isnan(grad_m[yi, xi]) else 0
                            gn = grad_n[yi, xi] if not np.isnan(grad_n[yi, xi]) else 0
                            mag = np.sqrt(gm**2 + gn**2)
                            if mag > 0:
                                # Normalize arrow length
                                scale = min(1.5, mag / (np.nanmax(np.abs(Za)) * 0.1 + 1e-10)) * 0.8
                                ax.quiver(Ma[yi, xi], Na[yi, xi], Za[yi, xi],
                                          gm / mag * scale, gn / mag * scale, 0,
                                          color="black", alpha=0.5, arrow_length_ratio=0.3,
                                          linewidth=0.6)
            except Exception:
                ax.scatter(m_pts, n_pts, z_pts, c=z_pts, cmap="terrain", s=30)

            # Mark best
            best_row = best.loc[best["effective_stps"].idxmax()]
            ax.scatter([best_row["dram_total_layers"]], [best_row["dram_active_layers"]],
                       [best_row["effective_stps"] * 1.02],
                       s=200, c="red", marker="*", edgecolors="white",
                       linewidths=1, zorder=10, depthshade=False)

            ax.set_xlabel("m", fontsize=9, labelpad=3)
            ax.set_ylabel("n", fontsize=9, labelpad=3)
            ax.set_zlabel("tok/s", fontsize=9, labelpad=3)
            ax.set_title(f"bs={bs}", fontweight="bold", fontsize=11)
            ax.view_init(elev=30, azim=-140)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"{phase_name}: Throughput Terrain with Gradient Flow",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save(fig, f"3dE_{prefix}_terrain_gradient")


# ======================================================================
# 3D-F: Stacked surface — multiple bs as stacked transparent surfaces
#   All on absolute throughput scale to show how landscape shifts.
# ======================================================================
def fig3d_stacked_surfaces(pf, dc):
    """Multiple bs surfaces stacked in one 3D plot with transparency."""
    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        bs_vals = sorted(df["bs"].unique())
        # Pick a subset to avoid too much clutter
        if len(bs_vals) > 5:
            rep_bs = [bs_vals[0], bs_vals[len(bs_vals)//4],
                      bs_vals[len(bs_vals)//2], bs_vals[3*len(bs_vals)//4],
                      bs_vals[-1]]
        else:
            rep_bs = bs_vals

        cmap_list = ["Blues", "Oranges", "Greens", "Reds", "Purples"]

        m_fine = np.linspace(2, 14, 30)
        n_fine = np.linspace(2, 12, 25)
        M, N = np.meshgrid(m_fine, n_fine)

        for j, bs in enumerate(rep_bs):
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            m_pts = best["dram_total_layers"].values.astype(float)
            n_pts = best["dram_active_layers"].values.astype(float)
            z_pts = best["effective_stps"].values

            try:
                Z = griddata((m_pts, n_pts), z_pts, (M, N), method="cubic")
                Z[N > M] = np.nan
                cmap_name = cmap_list[j % len(cmap_list)]
                ax.plot_surface(M, N, Z, cmap=cmap_name, alpha=0.4,
                                edgecolor="none", rstride=3, cstride=3, antialiased=True)
                # Add wireframe for the same surface for clarity
                ax.plot_wireframe(M, N, Z, color=BS_COLORS.get(bs, "#333"),
                                  linewidth=0.4, alpha=0.5, rstride=5, cstride=5)
            except Exception:
                pass

        # Legend
        handles = [Line2D([0], [0], color=BS_COLORS.get(bs, "#333"),
                          linewidth=2, label=f"bs={bs}") for bs in rep_bs]
        ax.legend(handles=handles, loc="upper left", fontsize=9)

        ax.set_xlabel("m (total layers)", fontsize=10, labelpad=5)
        ax.set_ylabel("n (active layers)", fontsize=10, labelpad=5)
        ax.set_zlabel("Throughput (tok/s)", fontsize=10, labelpad=5)
        ax.set_title(f"{phase_name}: Throughput Surfaces Across Batch Sizes",
                     fontweight="bold", fontsize=12)
        ax.view_init(elev=25, azim=-135)
        _save(fig, f"3dF_{prefix}_stacked_surfaces")


# ======================================================================
# 3D-G: Side-by-side prefill vs decode on same scale
#   Two 3D surfaces at a specific bs for direct comparison.
# ======================================================================
def fig3d_prefill_vs_decode(pf, dc):
    """Side-by-side 3D surface: prefill vs decode at same bs."""
    common_bs = sorted(set(pf["bs"].unique()) & set(dc["bs"].unique()))
    # Pick 2-3 representative
    if len(common_bs) > 3:
        rep_bs = [common_bs[0], common_bs[len(common_bs)//2], common_bs[-1]]
    else:
        rep_bs = common_bs

    fig = plt.figure(figsize=(10, 4.5 * len(rep_bs)))

    for row, bs in enumerate(rep_bs):
        for col, (df, phase, cmap_name) in enumerate([
            (pf, "Prefill", "YlOrRd"), (dc, "Decode", "YlGnBu")
        ]):
            ax = fig.add_subplot(len(rep_bs), 2, row * 2 + col + 1, projection="3d")
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            m_pts = best["dram_total_layers"].values.astype(float)
            n_pts = best["dram_active_layers"].values.astype(float)
            z_pts = best["effective_stps"].values

            m_fine = np.linspace(2, 14, 35)
            n_fine = np.linspace(2, 12, 30)
            M, N = np.meshgrid(m_fine, n_fine)

            try:
                Z = griddata((m_pts, n_pts), z_pts, (M, N), method="cubic")
                Z[N > M] = np.nan
                ax.plot_surface(M, N, Z, cmap=cmap_name, alpha=0.8,
                                edgecolor="gray", linewidth=0.15,
                                rstride=2, cstride=2, antialiased=True)
            except Exception:
                ax.scatter(m_pts, n_pts, z_pts, c=z_pts, cmap=cmap_name, s=30)

            # Mark best
            best_row = best.loc[best["effective_stps"].idxmax()]
            ax.scatter([best_row["dram_total_layers"]], [best_row["dram_active_layers"]],
                       [best_row["effective_stps"]],
                       s=180, c="red", marker="*", edgecolors="white",
                       linewidths=1, zorder=10, depthshade=False)
            ax.text(best_row["dram_total_layers"], best_row["dram_active_layers"],
                    best_row["effective_stps"] * 1.05,
                    f"({int(best_row['dram_total_layers'])},{int(best_row['dram_active_layers'])})",
                    fontsize=8, fontweight="bold", color="red")

            ax.set_xlabel("m", fontsize=9, labelpad=2)
            ax.set_ylabel("n", fontsize=9, labelpad=2)
            ax.set_zlabel("tok/s", fontsize=9, labelpad=2)
            ax.set_title(f"{phase} bs={bs}", fontweight="bold", fontsize=10)
            ax.view_init(elev=25, azim=-135)
            ax.tick_params(labelsize=6)

    fig.suptitle("Prefill vs. Decode: Throughput Landscape Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "3dG_prefill_vs_decode")


# ======================================================================
# 3D-H: 3D stem plot (lollipop) — clean discrete representation
#   Each (m,n) is a stem with a colored ball on top.
# ======================================================================
def fig3d_stem(pf, dc):
    """3D stem/lollipop plot for clean discrete data visualization."""
    for phase_name, df, prefix in [("Prefill", pf, "prefill"), ("Decode", dc, "decode")]:
        bs_vals = sorted(df["bs"].unique())
        ncols = min(len(bs_vals), 4)
        nrows = (len(bs_vals) + ncols - 1) // ncols
        fig = plt.figure(figsize=(5.0 * ncols, 4.5 * nrows))

        for i, bs in enumerate(bs_vals):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            sub = df[df["bs"] == bs]
            best = sub.groupby(["dram_total_layers", "dram_active_layers"])["effective_stps"].max().reset_index()

            m = best["dram_total_layers"].values.astype(float)
            n = best["dram_active_layers"].values.astype(float)
            z = best["effective_stps"].values

            # Normalize for color
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
            colors = cm.RdYlGn(z_norm)  # Green=high, Red=low

            # Draw stems
            for mi, ni, zi, ci in zip(m, n, z, colors):
                ax.plot([mi, mi], [ni, ni], [0, zi], color="gray",
                        linewidth=0.8, alpha=0.5)
                ax.scatter([mi], [ni], [zi], c=[ci], s=50,
                           edgecolors="black", linewidths=0.4, depthshade=False)

            # Mark best
            best_idx = z.argmax()
            ax.scatter([m[best_idx]], [n[best_idx]], [z[best_idx]],
                       s=180, c="gold", marker="*", edgecolors="black",
                       linewidths=1, zorder=10, depthshade=False)

            ax.set_xlabel("m", fontsize=8, labelpad=1)
            ax.set_ylabel("n", fontsize=8, labelpad=1)
            ax.set_zlabel("tok/s", fontsize=8, labelpad=1)
            ax.set_title(f"bs={bs}", fontweight="bold", fontsize=10)
            ax.view_init(elev=25, azim=-130)
            ax.tick_params(labelsize=6)

        fig.suptitle(f"{phase_name}: Throughput Stem Plot by (m, n)",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save(fig, f"3dH_{prefix}_stem")


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

    print("\n=== 3D Visualization Ideas ===")

    print("\n3D-A: Smooth surface plots per bs")
    fig3d_surface(pf, dc)

    print("\n3D-B: 3D bar charts per bs")
    fig3d_bar(pf, dc)

    print("\n3D-C: Wireframe overlay (multiple bs)")
    fig3d_wireframe_overlay(pf, dc)

    print("\n3D-D: Scatter with projections")
    fig3d_scatter_projections(pf, dc)

    print("\n3D-E: Terrain with gradient arrows")
    fig3d_terrain_gradient(pf, dc)

    print("\n3D-F: Stacked transparent surfaces")
    fig3d_stacked_surfaces(pf, dc)

    print("\n3D-G: Prefill vs Decode side-by-side")
    fig3d_prefill_vs_decode(pf, dc)

    print("\n3D-H: Stem/lollipop plots")
    fig3d_stem(pf, dc)

    print(f"\nAll 3D figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
