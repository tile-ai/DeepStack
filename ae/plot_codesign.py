"""Paper-style plots for the bounded co-design reproduction (Figs. 15--23).

The plotting code deliberately consumes only outputs produced by the artifact.
For figures whose paper version came from the exhaustive DSE (Figs. 18--20),
the rendered panels use compact, repository-local projections of the archived
search.  The result manifest distinguishes those plot inputs from the fixed
configurations and deterministic samples that the workflow actually reruns.
Fig. 17 additionally reconstructs its nested display bands in memory from the
published panel selectors, so calibration intermediates are not persisted in
the generated result CSV.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .paths import RESULTS_DIR, ROOT

# Direct module use falls back to a repository-local cache.  The supported
# top-level plot workflow sets a selected-scope cache before importing us.
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

MODEL_TITLES = {
    "DeepSeekV3": "DeepSeek-V3",
    "Qwen3_235b_a22b": "Qwen3-235B-A22B",
    "Llama3_405b": "Llama3 405B",
    "Llama3_70b": "Llama3 70B",
}
MODEL_ORDER = tuple(MODEL_TITLES)
ARCH_COLORS = {
    "stacked_gpu_large_vector": "#1f77b4",
    "stacked_gpu_large_matrix": "#ff7f0e",
    "stacked_gpu_reduced_sm": "#98df8a",
    "stacked_gpu_base": "#ff9896",
    "stacked_gpu_high_l1": "#8c564b",
    "stacked_gpu_high_l2": "#e377c2",
    "stacked_gpu_low_noc": "#c7c7c7",
    "stacked_gpu_wgmma": "#dbdb8d",
}
ARCH_LABELS = {
    "stacked_gpu_large_vector": "Large Vector",
    "stacked_gpu_large_matrix": "Large Matrix",
    "stacked_gpu_reduced_sm": "Less SM",
    "stacked_gpu_base": "Standard",
    "stacked_gpu_high_l1": "Strong L1",
    "stacked_gpu_high_l2": "Strong L2",
    "stacked_gpu_low_noc": "Weak NoC",
    "stacked_gpu_wgmma": "WGMMA",
}
NOC_MARKERS = {
    "strong_torus_mesh_switch_4": "o",
    "torus_mesh_mesh_3": "s",
    "torus_mesh_switch_1": "^",
    "torus_mesh_switch_2": "D",
    "torus_mesh_switch_7": "P",
    "torus_mesh_switch_8": "X",
    "torus_mesh_switch_9": "*",
    "weak_torus_mesh_switch_5": "v",
}
NOC_LABELS = {
    "strong_torus_mesh_switch_4": "TMS3",
    "torus_mesh_mesh_3": "TMM8",
    "torus_mesh_switch_1": "TMS1",
    "torus_mesh_switch_2": "TMS2",
    "torus_mesh_switch_7": "TMS5",
    "torus_mesh_switch_8": "TMS6",
    "torus_mesh_switch_9": "TMS7",
    "weak_torus_mesh_switch_5": "TMS4",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "CMU Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.35,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing plotting input {path}; run `./run.sh reproduce` first"
        )
    return path


def _stage_input(stage_root: Path, *parts: str) -> Path:
    """Require one input without allowing a nested symlink to escape scope."""

    path = stage_root.joinpath(*parts).resolve()
    try:
        path.relative_to(stage_root)
    except ValueError as exc:
        raise ValueError(f"plot input escapes scoped stage root: {path}") from exc
    return _require(path)


def _source_string(paths: Iterable[Path]) -> str:
    values: list[str] = []
    for path in paths:
        resolved = path.resolve()
        scoped = next(
            (parent for parent in resolved.parents if parent.name == "stages"),
            None,
        )
        if scoped is not None:
            values.append(resolved.relative_to(scoped.parent).as_posix())
            continue
        try:
            values.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            values.append(str(resolved))
    return ";".join(values)


def _save(
    fig: plt.Figure,
    output_dir: Path,
    *,
    figure: str,
    stem: str,
    sources: Iterable[Path],
    model_version: str,
    note: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor="white",
        metadata={"Creator": "DeepStack AE", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return {
        "figure": figure,
        "stem": stem,
        "model_version": model_version,
        "source_files": _source_string(sources),
        "png": png.name,
        "pdf": pdf.name,
        "png_bytes": png.stat().st_size,
        "pdf_bytes": pdf.stat().st_size,
        "status": "PASS",
        "note": note,
    }


def _style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, linestyle="--", alpha=0.28, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.65)


def _plot_fig15(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(
        stage_root, "fig15", model_version, "fig15_pareto_candidates.csv"
    )
    frame = pd.read_csv(source)
    paper_titles = {
        "DeepSeekV3": "DeepSeek V3",
        "Qwen3_235b_a22b": "Qwen3 235B A22B",
        "Llama3_405b": "Llama3 405B",
        "Llama3_70b": "Llama3 70B",
    }
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        fig, axes = plt.subplots(1, 4, figsize=(13.3, 2.62))
        for ax, model in zip(axes, MODEL_ORDER, strict=True):
            subset = frame[frame.model == model]
            for (arch, noc), group in subset.groupby(["arch", "noc"], sort=True):
                group = group.sort_values("bs")
                ax.plot(
                    group.current_utps_avg,
                    group.current_stps_avg / 1000.0,
                    color=ARCH_COLORS.get(str(arch), "#555555"),
                    marker=NOC_MARKERS.get(str(noc), "o"),
                    linewidth=0.85,
                    markersize=4.6,
                    markeredgecolor="#222222",
                    markeredgewidth=0.55,
                    alpha=0.78,
                )
            ax.set_title(paper_titles[model], pad=4)
            ax.set_xlabel("UTPS (token/s)", labelpad=1)
            if ax is axes[0]:
                ax.set_ylabel("STPS", fontweight="bold", fontstyle="italic")
            _style_axis(ax)

        # The final paper composition labels representative DeepSeek batch
        # envelopes rather than explaining the direction in prose.
        deepseek = frame[frame.model == "DeepSeekV3"]
        envelope = deepseek.loc[
            deepseek.groupby("bs")["current_stps_avg"].idxmax()
        ].set_index("bs")
        callouts = (
            (1024, "BS=1024", (5, -8)),
            (256, "BS=256", (12, 3)),
            (128, "BS=128", (10, -2)),
            (64, "BS=\n64/16", (18, 1)),
            (4, "BS=\n4/1", (19, -7)),
        )
        for bs, label, offset in callouts:
            point = envelope.loc[bs]
            axes[0].annotate(
                label,
                (point.current_utps_avg, point.current_stps_avg / 1000.0),
                xytext=offset,
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.8,
            )

        arch_handles = [
            Line2D(
                [0], [0], color=color, lw=2.2, label=ARCH_LABELS.get(name, name)
            )
            for name, color in ARCH_COLORS.items()
            if name in set(frame.arch.astype(str))
        ]
        noc_handles = [
            Line2D(
                [0],
                [0],
                color="#222222",
                marker=marker,
                linestyle="None",
                markersize=5,
                label=NOC_LABELS.get(name, name),
            )
            for name, marker in NOC_MARKERS.items()
            if name in set(frame.noc.astype(str))
        ]
        fig.legend(
            noc_handles,
            [handle.get_label() for handle in noc_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.17),
            ncol=8,
            frameon=False,
            handlelength=1.3,
            columnspacing=1.25,
        )
        fig.legend(
            arch_handles,
            [handle.get_label() for handle in arch_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.075),
            ncol=8,
            frameon=False,
            handlelength=2.2,
            columnspacing=1.05,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.25)
    return _save(
        fig,
        output_dir,
        figure="Fig. 15",
        stem="fig15_decode_pareto",
        sources=(source,),
        model_version=model_version,
        note="784 re-evaluated local winners/competitors; no exhaustive DSE",
    )


def _plot_fig16(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(stage_root, "fig16", model_version, "fixed_winners.csv")
    frame = pd.read_csv(source)
    platform_style = {
        "DeepStack": ("#1f77b4", "*", "-", "DeepStack DSE"),
        "H100_SCALED": ("#ff7f0e", "o", "--", "H100 (scaled)"),
        "H200_SCALED": ("#2ca02c", "s", "-.", "H200 (scaled)"),
    }
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 2.45))
    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        subset = frame[frame.model == model]
        for platform, (color, marker, line, label) in platform_style.items():
            group = subset[subset.platform == platform].sort_values("bs")
            if group.empty:
                continue
            ax.plot(
                group.current_utps_avg,
                group.current_stps_avg / 1000.0,
                color=color,
                marker=marker,
                linestyle=line,
                markersize=5.7 if platform == "DeepStack" else 3.8,
                label=label,
            )
        ax.set_title(f"{MODEL_TITLES[model]} Decode")
        ax.set_xlabel("User TPS (tokens/s)")
        if ax is axes[0]:
            ax.set_ylabel("System TPS (K tokens/s)")
        _style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(
        fig,
        output_dir,
        figure="Fig. 16",
        stem="fig16_3d_vs_2_5d",
        sources=(source,),
        model_version=model_version,
        note="80 fixed platform/model/batch winners",
    )


def _plot_fig17(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(
        stage_root, "fig17", "bw_analysis_connected_8x_2x3.csv"
    )
    frame = _fig17_plot_frame(pd.read_csv(source))
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.0), sharex=True, sharey=True)
    bw_columns = (
        ("ddr_peak_bw_TBs", "3D DRAM peak", "#ececf4", 0.82),
        ("littles_law_bw_TBs", "Little's-law limit", "#bdbdbd", 0.64),
        ("ddr_eff_bw_TBs", "After charging limits", "#98d594", 0.47),
        ("actual_bw_TBs", "L1-limited effective BW", "#3298dc", 0.30),
    )
    for row, smem in enumerate((128, 512)):
        for col, l1 in enumerate((128, 256, 1024)):
            ax = axes[row, col]
            subset = frame[
                (frame.smem_cap_KiB == smem) & (frame.l1_tp_Bpc == l1)
            ].sort_values("n")
            x = subset.n.to_numpy()
            for column, label, color, width in bw_columns:
                ax.bar(
                    x,
                    subset[column] * 8.0,
                    width=width,
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.35,
                    label=label,
                )
            ax.axhline(4.8, color="#d62728", ls="--", lw=0.85)
            ax.axhline(2.0, color="#9467bd", ls="--", lw=0.85)
            twin = ax.twinx()
            twin.plot(x, subset.sm_count * 8.0, color="#ff9800", marker="o", ms=2.5)
            twin.tick_params(axis="y", colors="#d97706", labelsize=6.8)
            if col != 2:
                twin.set_yticklabels([])
            else:
                twin.set_ylabel("SM count", color="#d97706")
            ax.set_title(f"L1 cap. {smem} KiB, BW {l1} B/c")
            ax.set_xticks(range(2, 17, 2))
            if row == 1:
                ax.set_xlabel("Stacked 3D DRAM layers")
            if col == 0:
                ax.set_ylabel("BW (TB/s)")
            _style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra = [
        Line2D([0], [0], color="#d62728", ls="--", label="H200 (4.8 TB/s)"),
        Line2D([0], [0], color="#9467bd", ls="--", label="A100 (2.0 TB/s)"),
        Line2D([0], [0], color="#ff9800", marker="o", label="SM count"),
    ]
    fig.legend(
        handles + extra,
        labels + [item.get_label() for item in extra],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(
        fig,
        output_dir,
        figure="Fig. 17",
        stem="fig17_dram_bandwidth",
        sources=(source,),
        model_version=model_version,
        note="complete 96-point analytical bandwidth sweep; paper 8x area scale",
    )


def _fig17_plot_frame(published: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct Fig. 17 display-only bands without persisting them."""

    from .fig17 import PUBLISHED_FIELDS, analyze_configuration

    columns = set(published.columns)
    expected = set(PUBLISHED_FIELDS)
    missing = sorted(expected - columns)
    unexpected = sorted(columns - expected)
    if missing or unexpected:
        raise ValueError(
            "Fig. 17 publish-safe CSV schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    rows: list[dict[str, object]] = []
    for public_row in published.to_dict("records"):
        full_row = analyze_configuration(
            int(public_row["n"]),
            int(public_row["smem_cap_KiB"]) * 1024,
            int(public_row["l1_tp_Bpc"]),
        )
        if not np.isclose(
            float(full_row["actual_bw_TBs"]),
            float(public_row["actual_bw_TBs"]),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Fig. 17 persisted effective bandwidth differs from recomputation"
            )
        rows.append(
            {
                **public_row,
                "ddr_peak_bw_TBs": full_row["ddr_peak_bw_TBs"],
                "littles_law_bw_TBs": full_row["littles_law_bw_TBs"],
                "ddr_eff_bw_TBs": full_row["ddr_eff_bw_TBs"],
                "sm_count": full_row["sm_count"],
            }
        )
    return pd.DataFrame(rows)


def _roles(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    mask = frame.selection_roles.astype(str).str.split(";").map(
        lambda values: role in values
    )
    return frame[mask].copy()


def _plot_fig18(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    model_source = _stage_input(
        stage_root, "fig18_20", model_version, "model_points.csv"
    )
    curve_source = _stage_input(
        stage_root, "fig18_20", "archive", "fig18_throughput_curves.csv"
    )
    curves = pd.read_csv(curve_source)
    # The freshly evaluated endpoint rows must point to the same archived
    # winners used for the BS=4/1024 paper curves.
    endpoints = _roles(pd.read_csv(model_source), "fig18_curve_best")
    if len(endpoints) != 44:
        raise AssertionError(f"Fig. 18 expected 44 rerun endpoints, got {len(endpoints)}")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        top=0.75,
        bottom=0.22,
        wspace=0.20,
    )
    colors = {
        4: "#ff7f0e",
        16: "#2ca02c",
        64: "#d62728",
        256: "#9467bd",
        1024: "#8c564b",
    }
    markers = {4: "s", 16: "^", 64: "D", 256: "v", 1024: "P"}
    for ax, phase in zip(axes, ("decode", "prefill"), strict=True):
        for bs in (4, 16, 64, 256, 1024):
            group = curves[(curves.phase == phase) & (curves.bs == bs)].sort_values("m")
            ax.plot(
                group.m,
                group.best_scaled_stps / 1000.0,
                color=colors[bs],
                marker=markers[bs],
                label=f"BS={bs}",
                markeredgecolor="black",
                markeredgewidth=0.45,
            )
        panel = "a" if phase == "decode" else "b"
        ax.set_title(
            f"({panel}) {phase.capitalize()} Throughput",
            fontweight="bold",
            pad=5,
        )
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks(range(1, 13, 2))
        ax.set_xticks(range(1, 13), minor=True)
        ax.grid(True, alpha=0.25, linewidth=0.45)
        if phase == "decode":
            ax.set_ylabel("Throughput (K tok/s)")
        ax.annotate(
            "OOM",
            xy=(1, ax.get_ylim()[0]),
            fontsize=8,
            ha="center",
            va="bottom",
            color="red",
            fontweight="bold",
        )
        _style_axis(ax, grid=False)

    line_handles = [
        Line2D(
            [0],
            [0],
            color=colors[bs],
            marker=markers[bs],
            markeredgecolor="black",
            label=f"BS={bs}",
        )
        for bs in (4, 16, 64, 256, 1024)
    ]
    fig.legend(
        handles=line_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.supxlabel("Stacked DRAM Layers", y=0.035)
    return _save(
        fig,
        output_dir,
        figure="Fig. 18",
        stem="fig18_dram_layer_throughput",
        sources=(curve_source, model_source),
        model_version=model_version,
        note=(
            "paper-style 110-point archived DSE throughput curves; "
            "44 BS=4/1024 winners re-evaluated by the model"
        ),
    )


def _plot_fig19(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    model_source = _stage_input(
        stage_root, "fig18_20", model_version, "model_points.csv"
    )
    grid_source = _stage_input(
        stage_root, "fig18_20", "archive", "fig19_metric_grid.csv"
    )
    frame = pd.read_csv(model_source)
    grid_frame = pd.read_csv(grid_source)
    throughput = _roles(frame, "fig19_throughput_winner")
    efficiency = _roles(frame, "fig19_efficiency_winner")
    even_m = [2, 4, 6, 8, 10, 12, 14]
    even_n = [2, 4, 6, 8, 10, 12]
    cmap_stps = LinearSegmentedColormap.from_list(
        "fig19_stps", ["#FFF8F0", "#FFE0B2", "#FFB74D", "#F4511E", "#B71C1C"]
    )
    cmap_tokj = LinearSegmentedColormap.from_list(
        "fig19_tokj", ["#F0F9F4", "#B2DFDB", "#4DB6AC", "#00897B", "#004D40"]
    )

    def format_value(value: float, metric: str) -> str:
        if metric == "best_effective_stps":
            if value >= 1e6:
                return f"{value / 1e6:.1f}M"
            if value >= 1e4:
                return f"{value / 1e3:.0f}K"
            if value >= 1e3:
                return f"{value / 1e3:.1f}K"
            return f"{value:.0f}"
        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}"
        if value >= 1:
            return f"{value:.2f}"
        return f"{value:.3f}"

    def text_color(cmap: LinearSegmentedColormap, norm: Normalize, value: float) -> str:
        red, green, blue, _ = cmap(norm(value))
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        return "white" if luminance < 0.5 else "black"

    # The paper's final Fig. 19 is method2 from the original DSE plotting
    # script: rows are batch sizes and columns are phase/metric pairs.  Keep
    # the archived grid as the rendered surface while checking every marked
    # optimum against a freshly re-evaluated model point.
    fig, axes = plt.subplots(2, 4, figsize=(15, 5))
    panels = (
        ("prefill", "Prefill", "best_effective_stps", "System Throughput", cmap_stps),
        ("prefill", "Prefill", "best_tokens_per_joule", "Tokens/J", cmap_tokj),
        ("decode", "Decode", "best_effective_stps", "System Throughput", cmap_stps),
        ("decode", "Decode", "best_tokens_per_joule", "Tokens/J", cmap_tokj),
    )
    rerun_by_metric = {
        "best_effective_stps": throughput,
        "best_tokens_per_joule": efficiency,
    }

    for row_index, bs in enumerate((4, 1024)):
        for column_index, (phase, phase_label, metric, metric_label, cmap) in enumerate(panels):
            ax = axes[row_index, column_index]
            subset = grid_frame[
                (grid_frame.phase == phase) & (grid_frame.bs == bs)
            ]
            values = subset.pivot(index="m", columns="n", values=metric).reindex(
                index=even_m, columns=even_n
            )
            value_array = values.to_numpy()
            norm = Normalize(
                vmin=float(np.nanmin(value_array)),
                vmax=float(np.nanmax(value_array)),
            )
            image = ax.imshow(
                value_array,
                aspect="auto",
                cmap=cmap,
                origin="lower",
                norm=norm,
            )

            archived_winner = subset.loc[subset[metric].idxmax()]
            rerun_winner = rerun_by_metric[metric]
            rerun_winner = rerun_winner[
                (rerun_winner.phase == phase) & (rerun_winner.bs == bs)
            ]
            metric_name = "throughput" if metric == "best_effective_stps" else "tokens/J"
            if len(rerun_winner) != 1:
                raise AssertionError(
                    f"Fig. 19 expected one {phase}/{bs}/{metric_name} winner, "
                    f"got {len(rerun_winner)}"
                )
            rerun_row = rerun_winner.iloc[0]
            if (int(rerun_row.dram_total_layers), int(rerun_row.dram_active_layers)) != (
                int(archived_winner.m),
                int(archived_winner.n),
            ):
                raise AssertionError(
                    "Fig. 19 archived and rerun winner coordinates disagree for "
                    f"{phase}/BS={bs}/{metric_name}"
                )

            winner_y = even_m.index(int(archived_winner.m))
            winner_x = even_n.index(int(archived_winner.n))
            for y_index in range(len(even_m)):
                for x_index in range(len(even_n)):
                    value = value_array[y_index, x_index]
                    if np.isnan(value):
                        continue
                    is_winner = y_index == winner_y and x_index == winner_x
                    ax.text(
                        x_index,
                        y_index,
                        format_value(float(value), metric),
                        ha="center",
                        va="center",
                        fontsize=8 if is_winner else 6.5,
                        color=text_color(cmap, norm, float(value)),
                        fontweight="bold" if is_winner else "normal",
                        zorder=6,
                    )

            winner_color = "#D32F2F" if metric == "best_effective_stps" else "#1565C0"
            winner_label = (
                "Best Throughput"
                if metric == "best_effective_stps"
                else "Best Power Efficiency"
            )
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (winner_x - 0.49, winner_y - 0.49),
                    0.98,
                    0.98,
                    boxstyle="round,pad=0.03",
                    facecolor="none",
                    edgecolor="white",
                    linewidth=4.5,
                    zorder=7,
                )
            )
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (winner_x - 0.47, winner_y - 0.47),
                    0.94,
                    0.94,
                    boxstyle="round,pad=0.03",
                    facecolor="none",
                    edgecolor=winner_color,
                    linewidth=2.5,
                    zorder=8,
                )
            )
            ax.text(
                0.97,
                0.03,
                winner_label,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                fontweight="bold",
                color=winner_color,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": winner_color,
                    "alpha": 0.9,
                    "linewidth": 0.8,
                },
                zorder=9,
            )
            ax.set_xticks(range(len(even_n)))
            ax.set_xticklabels(even_n, fontsize=8)
            ax.set_yticks(range(len(even_m)))
            ax.set_yticklabels(even_m, fontsize=8)
            if row_index == 1:
                ax.set_xlabel("Connected DRAM Layers", fontsize=10)
            if column_index == 0:
                ax.set_ylabel(
                    rf"$\bf{{BS={bs}}}$" + "\nStacked DRAM Layers",
                    fontsize=10,
                )
            if row_index == 0:
                ax.set_title(
                    rf"$\bf{{{phase_label}}}$" + f"\n{metric_label}",
                    fontsize=10,
                )
            fig.colorbar(
                image,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                format=mticker.FuncFormatter(
                    lambda value, _position, selected_metric=metric: format_value(
                        value, selected_metric
                    )
                ),
            )

    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    return _save(
        fig,
        output_dir,
        figure="Fig. 19",
        stem="fig19_prefill_decode_dse_heatmaps",
        sources=(grid_source, model_source),
        model_version=model_version,
        note=(
            "paper-matched 2x4 heatmaps from the 108-cell archived DSE grid; eight displayed "
            "global winners re-evaluated by the model"
        ),
    )


def _plot_fig20(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    terrain_source = _stage_input(
        stage_root,
        "fig18_20",
        "archive",
        "fig20_decode_thermal_terrain.csv",
    )
    sample_source = _stage_input(
        stage_root,
        "fig18_20",
        model_version,
        "terrain_sample_validation.csv",
    )
    terrain = pd.read_csv(terrain_source)
    sample = pd.read_csv(sample_source)
    if len(sample) != 22 or not set(sample.terrain_reference_id).issubset(
        set(terrain.terrain_reference_id)
    ):
        raise AssertionError("Fig. 20 terrain sample coverage mismatch")

    from scipy.interpolate import griddata

    fig = plt.figure(figsize=(10.8, 4.1))
    layout = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.03), wspace=0.0)
    axes = [
        fig.add_subplot(layout[0, 0], projection="3d"),
        fig.add_subplot(layout[0, 1], projection="3d"),
    ]
    z_limits = (
        float(terrain.temperature_C.min()) - 2.0,
        float(terrain.temperature_C.max()) + 2.0,
    )
    temperature_norm = Normalize(35, 130)
    for ax, bs in zip(axes, (4, 1024), strict=True):
        subset = terrain[terrain.bs == bs]
        x = subset.raw_stps.to_numpy()
        y = subset.dram_total_layers.to_numpy(dtype=float)
        z = subset.temperature_C.to_numpy()
        xi = np.linspace(float(x.min()), float(x.max()), 100)
        yi = np.linspace(float(y.min()), float(y.max()), 100)
        xx, yy = np.meshgrid(xi, yi)
        zz = griddata((x, y), z, (xx, yy), method="linear")
        facecolors = cm.coolwarm(temperature_norm(np.nan_to_num(zz, nan=35.0)))
        ax.plot_surface(
            xx,
            yy,
            zz,
            facecolors=facecolors,
            alpha=0.85,
            rstride=2,
            cstride=2,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        try:
            ax.contour(xx, yy, zz, levels=[95.0], colors=["red"], linewidths=1.5)
        except ValueError:
            pass
        ax.scatter(
            x,
            y,
            z,
            c=cm.coolwarm(temperature_norm(z)),
            s=3,
            alpha=0.3,
            edgecolors="none",
            depthshade=True,
        )
        x_range = (float(x.min()), float(x.max()))
        y_range = (float(y.min()) - 0.5, float(y.max()) + 0.5)
        warning_vertices = [
            [
                (x_range[0], y_range[0], 85.0),
                (x_range[1], y_range[0], 85.0),
                (x_range[1], y_range[1], 85.0),
                (x_range[0], y_range[1], 85.0),
            ]
        ]
        ax.add_collection3d(
            Poly3DCollection(
                warning_vertices,
                alpha=0.18,
                facecolor="orange",
                edgecolor="darkorange",
                linewidth=1.0,
            )
        )
        ax.text(
            x_range[0] + (x_range[1] - x_range[0]) * 0.55,
            y_range[1],
            86.2,
            "85 °C",
            color="darkorange",
            fontsize=9,
            fontstyle="italic",
            ha="center",
        )
        ax.set_xlabel("STPS (tok/s)", labelpad=2)
        ax.set_ylabel("DRAM Layers", labelpad=2)
        ax.set_zlabel("Temp (°C)", labelpad=2)
        ax.set_yticks(sorted(subset.dram_total_layers.unique()))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _position: (
                    f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}"
                )
            )
        )
        ax.set_zlim(z_limits)
        ax.set_title(f"Decode, BS = {bs}", fontweight="bold", y=0.95)
        ax.view_init(elev=28, azim=-50)
        ax.set_box_aspect((1.35, 1.0, 0.8), zoom=1.15)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("lightgrey")
        ax.grid(True, alpha=0.2, linewidth=0.3)

    color_axis = fig.add_subplot(layout[0, 2])
    scalar = cm.ScalarMappable(cmap="coolwarm", norm=temperature_norm)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=color_axis)
    colorbar.set_label("Temperature (°C)")
    fig.subplots_adjust(left=0.01, right=0.97, top=0.94, bottom=0.03)
    position = color_axis.get_position()
    color_axis.set_position(
        [position.x0 + 0.03, position.y0, position.width, position.height]
    )
    return _save(
        fig,
        output_dir,
        figure="Fig. 20",
        stem="fig20_dram_thermal_terrain",
        sources=(terrain_source, sample_source),
        model_version=model_version,
        note=(
            "paper-style 19,233-point archived raw-STPS terrain; 22 stratified "
            "hash samples re-evaluated by the model"
        ),
    )


def _plot_fig21(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(stage_root, "fig21", model_version, "raw.csv")
    frame = pd.read_csv(source)
    reduced = (
        frame.groupby(["bs", "latency_multiplier", "bw_multiplier"], as_index=False)
        .reproduced_stps_avg.max()
        .sort_values(["bs", "latency_multiplier", "bw_multiplier"])
    )
    batch_sizes = (4, 64, 1024)
    bandwidths = np.array((0.5, 0.75, 1.0, 1.25, 1.5, 2.0))
    latencies = np.array((0.25, 0.5, 1.0, 2.0, 4.0))
    xx, yy = np.meshgrid(bandwidths, latencies)
    matrices: dict[int, np.ndarray] = {}
    for bs in batch_sizes:
        group = reduced[reduced.bs == bs]
        baseline = float(
            group[
                np.isclose(group.latency_multiplier, 1.0)
                & np.isclose(group.bw_multiplier, 1.0)
            ].reproduced_stps_avg.iloc[0]
        )
        pivot = group.pivot(
            index="latency_multiplier",
            columns="bw_multiplier",
            values="reproduced_stps_avg",
        ).reindex(index=latencies, columns=bandwidths)
        matrices[bs] = pivot.to_numpy() / baseline

    all_values = np.concatenate([matrix.ravel() for matrix in matrices.values()])
    all_values = all_values[~np.isnan(all_values)]
    value_min = float(all_values.min())
    value_max = float(all_values.max())
    norm = TwoSlopeNorm(vmin=value_min, vcenter=1.0, vmax=value_max)
    cmap = plt.get_cmap("RdYlGn")

    # Match the source plot used by the paper.  The final paper asset is this
    # 5.5x2-inch canvas with a narrow PDF crop around the visible content.
    fig = plt.figure(figsize=(5.5, 2.0))
    for column_index, bs in enumerate(batch_sizes):
        ax = fig.add_subplot(1, 3, column_index + 1, projection="3d")
        matrix = matrices[bs]
        ax.plot_surface(
            xx,
            yy,
            matrix,
            cmap=cmap,
            norm=norm,
            alpha=0.85,
            edgecolor="grey",
            linewidth=0.2,
            rstride=1,
            cstride=1,
            antialiased=True,
        )
        ax.plot_surface(
            xx,
            yy,
            np.ones_like(matrix),
            color="#cccccc",
            alpha=0.25,
            edgecolor="none",
        )
        ax.set_xlabel("BW", labelpad=1, fontsize=6.5)
        ax.set_ylabel("Latency", labelpad=1, fontsize=6.5)
        ax.set_zlabel("Normalized STPS", labelpad=-4, fontsize=6.5)
        ax.set_title(f"BS = {bs}", fontweight="bold", pad=-2, fontsize=7.5)
        ax.set_xticks((0.5, 1.0, 1.5, 2.0))
        ax.set_xticklabels(("0.5×", "1×", "1.5×", "2×"), fontsize=5.5)
        ax.set_yticks((0.25, 1.0, 2.0, 4.0))
        ax.set_yticklabels(("0.25×", "1×", "2×", "4×"), fontsize=5.5)
        ax.tick_params(axis="z", labelsize=5.5)
        ax.view_init(elev=25, azim=-50)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("lightgrey")
            axis._axinfo["grid"]["linewidth"] = 0.2
        ax.set_zlim(value_min - 0.02, value_max + 0.02)

    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.05, top=0.92, wspace=0.08)
    color_axis = fig.add_axes((1.01, 0.12, 0.015, 0.72))
    scalar = cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=color_axis)
    colorbar.ax.tick_params(labelsize=5.5)
    return _save(
        fig,
        output_dir,
        figure="Fig. 21",
        stem="fig21_noc_bw_latency_sensitivity",
        sources=(source,),
        model_version=model_version,
        note=(
            "paper-matched three-panel surface; best-of-two MoE mappings over "
            "the complete 3x5x6 fixed grid"
        ),
    )


def _normalized_fig22_curves(
    frame: pd.DataFrame,
    *,
    phase: str,
    layer: str,
) -> dict[int, pd.DataFrame]:
    """Return per-BS paper curves normalized to each workload's 1x point."""

    subset = frame[
        (frame.phase == phase)
        & (frame.baseline_noc == "torus_mesh_switch_1")
        & (frame.scaled_layer == layer)
    ].copy()
    if phase == "prefill":
        subset = subset[subset.seq == 1024]
    curves: dict[int, pd.DataFrame] = {}
    for bs, group in subset.groupby("bs", sort=True):
        reduced = (
            group.groupby("bw_multiplier", as_index=False)
            .reproduced_logic_stps.max()
            .sort_values("bw_multiplier")
        )
        baseline = reduced[np.isclose(reduced.bw_multiplier, 1.0)]
        if len(baseline) != 1 or float(baseline.reproduced_logic_stps.iloc[0]) <= 0:
            raise AssertionError(
                f"Fig. 22 missing unique positive 1x baseline for {phase}/{layer}/BS={bs}"
            )
        reduced["normalized_stps"] = (
            reduced.reproduced_logic_stps
            / float(baseline.reproduced_logic_stps.iloc[0])
        )
        curves[int(bs)] = reduced
    return curves


def _plot_fig22_phase(
    output_dir: Path,
    source: Path,
    frame: pd.DataFrame,
    phase: str,
    model_version: str,
) -> dict[str, object]:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    batches = sorted(
        set(
            frame[
                (frame.phase == phase)
                & (frame.baseline_noc == "torus_mesh_switch_1")
            ].bs.astype(int)
        )
    )
    cmap = plt.get_cmap("viridis")
    colors = {
        bs: cmap(index / max(1, len(batches) - 1))
        for index, bs in enumerate(batches)
    }
    for ax, layer in zip(axes, ("L1", "L2", "L3"), strict=True):
        curves = _normalized_fig22_curves(frame, phase=phase, layer=layer)
        for bs in batches:
            group = curves.get(bs)
            if group is None:
                continue
            ax.plot(
                group.bw_multiplier,
                group.normalized_stps,
                color=colors[bs],
                marker="o",
                ms=2.8,
                label=f"BS={bs}",
            )
        ax.axhline(1.0, color="#777777", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks((0.25, 0.5, 1.0, 2.0, 4.0))
        ax.set_xticklabels(("0.25×", "0.5×", "1×", "2×", "4×"))
        ax.set_xlabel("BW multiplier")
        ax.set_title(f"Scale {layer}", fontweight="bold")
        _style_axis(ax)
    axes[0].set_ylabel("Average normalized STPS")
    batch_handles = [
        Line2D([0], [0], color=colors[bs], marker="o", label=f"BS={bs}")
        for bs in batches
    ]
    fig.legend(
        batch_handles,
        [item.get_label() for item in batch_handles],
        loc="upper center",
        ncol=len(batch_handles),
        frameon=False,
    )
    fig.suptitle(
        f"{phase.capitalize()} per-layer NoC bandwidth sensitivity",
        fontweight="bold",
        y=0.91,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    return _save(
        fig,
        output_dir,
        figure="Fig. 22",
        stem=f"fig22_noc_layer_{phase}",
        sources=(source,),
        model_version=model_version,
        note=f"{phase} per-batch curves normalized to each fixed workload's 1x point",
    )


def _plot_fig22(
    output_dir: Path, stage_root: Path, model_version: str
) -> list[dict[str, object]]:
    source = _stage_input(
        stage_root, "fig22", model_version, "model_points.csv"
    )
    frame = pd.read_csv(source)
    return [
        _plot_fig22_phase(output_dir, source, frame, phase, model_version)
        for phase in ("decode", "prefill")
    ]


def _plot_fig23(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(stage_root, "fig23", model_version, "raw.csv")
    frame = pd.read_csv(source)
    tier_styles = {
        "astra_space": ("#666666", "o", "ASTRA-sim*"),
        "expanded_parallel": ("#268bd2", "s", "+ More parallel dims."),
        "module_flexible": ("#d73027", "^", "+ Flexible module parallelism (DeepStack)"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), sharex="col")
    for row_index, phase in enumerate(("decode", "prefill")):
        for col_index, model in enumerate(("DeepSeekV3", "Qwen3_235b_a22b")):
            ax = axes[row_index, col_index]
            subset = frame[(frame.phase == phase) & (frame.model == model)]
            plotted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for tier, (color, marker, label) in tier_styles.items():
                group = subset[subset.tier == tier].sort_values("bs")
                x = np.arange(len(group))
                y = group.reproduced_stps.to_numpy() / 1000.0
                plotted[tier] = (x, y)
                ax.plot(x, y, color=color, marker=marker, label=label)
            if set(plotted) == set(tier_styles):
                x = plotted["astra_space"][0]
                ax.fill_between(
                    x,
                    0.0,
                    plotted["astra_space"][1],
                    color="#bdbdbd",
                    alpha=0.30,
                )
                ax.fill_between(
                    x,
                    plotted["astra_space"][1],
                    plotted["expanded_parallel"][1],
                    color="#9ecae1",
                    alpha=0.45,
                )
                ax.fill_between(
                    x,
                    plotted["expanded_parallel"][1],
                    plotted["module_flexible"][1],
                    color="#f4a6a6",
                    alpha=0.35,
                )
            batches = subset[subset.tier == "astra_space"].sort_values("bs").bs
            ax.set_xticks(np.arange(len(batches)))
            ax.set_xticklabels([str(int(value)) for value in batches])
            if row_index == 0:
                ax.set_title(MODEL_TITLES[model], fontweight="bold")
            if row_index == 1:
                ax.set_xlabel("Batch size")
            if col_index == 0:
                ax.set_ylabel(f"{phase.capitalize()} STPS\n(K tokens/s)")
            _style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(
        fig,
        output_dir,
        figure="Fig. 23",
        stem="fig23_parallelism_search_space",
        sources=(source,),
        model_version=model_version,
        note="84 re-evaluated curve winners across three nested parallelism spaces",
    )


def _plot_table4(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(stage_root, "table4", "model_points.csv")
    frame = pd.read_csv(source)
    frame = frame[frame.model_version == model_version].copy()
    pivot = frame.pivot(index="step", columns="bs", values="recomputed_stps_avg")
    names = frame.drop_duplicates("step").set_index("step").step_name
    rows: list[list[str]] = []
    previous = {4: None, 1024: None}
    for step in range(1, 8):
        values: list[str] = [str(step), str(names.loc[step])]
        for bs in (4, 1024):
            value = float(pivot.loc[step, bs])
            if previous[bs] is None:
                text = f"{value:,.1f}"
            else:
                gain = 100.0 * (value / float(previous[bs]) - 1.0)
                text = f"{value:,.1f} ({gain:+.0f}%)"
            previous[bs] = value
            values.append(text)
        rows.append(values)
    speedups = {
        bs: float(pivot.loc[7, bs] / pivot.loc[1, bs]) for bs in (4, 1024)
    }
    rows.append(["", "Ablation gain", f"{speedups[4]:.2f}×", f"{speedups[1024]:.2f}×"])
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=("Step", "Technique/search-space addition", "BS=4 STPS", "BS=1024 STPS"),
        colWidths=(0.07, 0.49, 0.2, 0.24),
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.42)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#d9eaf7")
            cell.set_text_props(weight="bold")
        elif row == len(rows):
            cell.set_facecolor("#edf7ed")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    ax.set_title("Table 4 — DeepStack search-space ablation", fontweight="bold", pad=7)
    return _save(
        fig,
        output_dir,
        figure="Table 4",
        stem="table04_ablation",
        sources=(source,),
        model_version=model_version,
        note="14 selected configurations rendered as the paper-style ablation table",
    )


def render_codesign_figures(
    output_dir: Path,
    *,
    result_root: Path,
    model_version: str = "paper_legacy",
) -> list[dict[str, object]]:
    """Render scoped Figs. 15--23 plus a graphical Table 4 summary.

    All inputs come from ``result_root/stages`` and all rendered files stay
    under ``result_root/figures``.
    """

    if model_version not in {"paper_legacy", "current_corrected"}:
        raise ValueError(f"unsupported model version: {model_version}")
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_dir = output_dir.resolve()
    result_root = result_root if result_root.is_absolute() else ROOT / result_root
    result_root = result_root.resolve()
    figure_root = (result_root / "figures").resolve()
    try:
        figure_root.relative_to(result_root)
    except ValueError as exc:
        raise ValueError(
            f"figure root must stay inside {result_root}: {figure_root}"
        ) from exc
    try:
        output_dir.relative_to(figure_root)
    except ValueError as exc:
        raise ValueError(
            f"figure output must stay inside {figure_root}: {output_dir}"
        ) from exc
    stage_root = (result_root / "stages").resolve()
    try:
        stage_root.relative_to(result_root)
    except ValueError as exc:
        raise ValueError(
            f"stage inputs must stay inside {result_root}: {stage_root}"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    rows: list[dict[str, object]] = [
        _plot_fig15(output_dir, stage_root, model_version),
        _plot_fig16(output_dir, stage_root, model_version),
        _plot_fig17(output_dir, stage_root, model_version),
        _plot_fig18(output_dir, stage_root, model_version),
        _plot_fig19(output_dir, stage_root, model_version),
        _plot_fig20(output_dir, stage_root, model_version),
        _plot_fig21(output_dir, stage_root, model_version),
    ]
    rows.extend(_plot_fig22(output_dir, stage_root, model_version))
    rows.extend(
        (
            _plot_fig23(output_dir, stage_root, model_version),
            _plot_table4(output_dir, stage_root, model_version),
        )
    )
    return rows
