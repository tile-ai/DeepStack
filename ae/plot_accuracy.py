"""Render the paper-style accuracy figures (Fig. 8--14).

The plotting pass is deliberately separate from experiment execution: it only
reads normalized tables already present under the selected workflow's
``stages/`` directory.  Relative paths are anchored at the artifact root;
absolute paths support copied self-contained result scopes.  The requested
output directory must stay under that workflow's ``figures/`` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# Direct module use falls back to a repository-local cache.  The supported
# top-level plot workflow sets a selected-scope cache before importing us.
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib-cache"))

import matplotlib  # noqa: E402  (backend must be selected before pyplot)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402


MODEL_VERSIONS = ("paper_legacy", "current_corrected")
MODEL_COLOR = "#a6cee3"
REFERENCE_COLOR = "#1f78b4"
EDGE_COLOR = "#222222"
ANALYTICAL_COLOR = "#33a02c"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"plot input is missing: {_display(path)}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_columns(
    rows: Sequence[dict[str, str]], columns: Iterable[str], path: Path
) -> None:
    if not rows:
        raise ValueError(f"plot input is empty: {_display(path)}")
    missing = set(columns).difference(rows[0])
    if missing:
        raise ValueError(f"{_display(path)}: missing columns {sorted(missing)}")


def _display(path: Path) -> str:
    resolved = path.resolve()
    for parent in resolved.parents:
        if parent.name == "stages":
            return str(resolved.relative_to(parent.parent))
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_path(path: Path) -> Path:
    """Resolve relative paths at the artifact root while allowing copied scopes."""

    return (path if path.is_absolute() else ROOT / path).resolve()


def _scoped_paths(result_root: Path, output_dir: Path) -> tuple[Path, Path]:
    """Resolve the selected workflow's stage-input and figure-output roots."""

    resolved_result = _resolve_path(result_root)
    stage_root = (resolved_result / "stages").resolve()
    try:
        stage_root.relative_to(resolved_result)
    except ValueError as exc:
        raise ValueError(
            f"stage inputs must stay inside {resolved_result}: {stage_root}"
        ) from exc
    figure_root = (resolved_result / "figures").resolve()
    try:
        figure_root.relative_to(resolved_result)
    except ValueError as exc:
        raise ValueError(
            f"figure root must stay inside {resolved_result}: {figure_root}"
        ) from exc
    destination = _resolve_path(output_dir)
    try:
        destination.relative_to(figure_root)
    except ValueError as exc:
        raise ValueError(
            f"figure output must stay inside {figure_root}: {output_dir}"
        ) from exc
    return stage_root, destination


def _stage_input(stage_root: Path, *parts: str) -> Path:
    """Resolve one stage CSV without allowing nested symlinks to escape."""

    path = stage_root.joinpath(*parts).resolve()
    try:
        path.relative_to(stage_root)
    except ValueError as exc:
        raise ValueError(f"plot input escapes scoped stage root: {path}") from exc
    return path


def _float(value: str, *, path: Path, column: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{_display(path)}: invalid {column} value {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{_display(path)}: non-finite {column} value {value!r}")
    return result


def _finite_or_none(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "CMU Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.15,
            "patch.linewidth": 0.65,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axis(ax: Any, *, grid_axis: str = "y") -> None:
    ax.grid(True, which="major", axis=grid_axis, linestyle="--", alpha=0.28, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=2.8, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def _paired_bars(
    ax: Any,
    labels: Sequence[str],
    model: Sequence[float],
    reference: Sequence[float],
    *,
    model_label: str,
    reference_label: str,
    show_ticks: bool = True,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.39
    ax.bar(
        x - width / 2,
        model,
        width,
        color=MODEL_COLOR,
        edgecolor=EDGE_COLOR,
        label=model_label,
    )
    ax.bar(
        x + width / 2,
        reference,
        width,
        color=REFERENCE_COLOR,
        edgecolor=EDGE_COLOR,
        label=reference_label,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels if show_ticks else [""] * len(labels))
    if not show_ticks:
        ax.tick_params(axis="x", length=0)
    _style_axis(ax)


def _save(fig: Any, output_dir: Path, stem: str) -> list[str]:
    outputs: list[str] = []
    output_root = output_dir.resolve()
    for extension in ("png", "pdf"):
        destination = (output_root / f"{stem}.{extension}").resolve()
        try:
            destination.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(
                f"refusing to write outside figure output: {destination}"
            ) from exc
        temporary = destination.with_name(f".{destination.name}.tmp")
        fig.savefig(
            temporary,
            format=extension,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.025,
        )
        temporary.replace(destination)
        outputs.append(_display(destination))
    plt.close(fig)
    return outputs


def _record(
    *,
    figure: str,
    stem: str,
    sources: Sequence[Path],
    outputs: Sequence[str],
    model_version: str,
) -> dict[str, object]:
    return {
        "stem": stem,
        "figure": figure,
        "source": [_display(path) for path in sources],
        "outputs": list(outputs),
        "model_version": model_version,
    }


def _fig08(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(stage_root, "fig08", "model_points.csv")
    rows = _read_csv(source)
    model_column = (
        "paper_model_ms" if model_version == "paper_legacy" else "current_model_ms"
    )
    _require_columns(
        rows,
        ("category", "point_index", model_column, "gpu_reference_ms"),
        source,
    )
    specs = (
        ("all_reduce_gemm", "(a) All-Reduce GEMM"),
        ("moe_ep_all_to_all", "(b) MoE EP All-To-All"),
        ("ulysses_attention", "(c) Ulysses Attention"),
        ("all_gather_gemm", "(d) All-Gather GEMM"),
        ("reduce_scatter_gemm", "(e) Reduce-Scatter GEMM"),
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    for category in grouped:
        grouped[category].sort(key=lambda row: int(row["point_index"]))
    if set(grouped) != {category for category, _ in specs}:
        raise ValueError(f"{_display(source)}: unexpected Fig. 8 category closure")

    fig = plt.figure(figsize=(10.0, 5.5))
    top = fig.add_gridspec(
        1,
        3,
        left=0.07,
        right=0.98,
        top=0.84,
        bottom=0.54,
        wspace=0.34,
        width_ratios=[8, 8, 12],
    )
    bottom = fig.add_gridspec(
        1,
        2,
        left=0.07,
        right=0.98,
        top=0.44,
        bottom=0.09,
        wspace=0.25,
        width_ratios=[12, 12],
    )
    axes = [fig.add_subplot(top[0, i]) for i in range(3)]
    axes.extend(fig.add_subplot(bottom[0, i]) for i in range(2))
    for ax, (category, title) in zip(axes, specs):
        panel = grouped[category]
        _paired_bars(
            ax,
            [str(index + 1) for index in range(len(panel))],
            [_float(row[model_column], path=source, column=model_column) for row in panel],
            [_float(row["gpu_reference_ms"], path=source, column="gpu_reference_ms") for row in panel],
            model_label="DeepStack (modeling)",
            reference_label="Ground truth (8×H100)",
            show_ticks=False,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.985),
    )
    stem = "fig08_modeling_accuracy_h100"
    return _record(
        figure="Fig. 8",
        stem=stem,
        sources=[source],
        outputs=_save(fig, output_dir, stem),
        model_version=model_version,
    )


def _fig09(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(stage_root, "fig09", "model_points.csv")
    rows = _read_csv(source)
    model_column = (
        "paper_legacy_stps"
        if model_version == "paper_legacy"
        else "current_corrected_stps"
    )
    _require_columns(rows, ("display_model", "bs", model_column, "vllm_stps"), source)
    order = ("Qwen3 235B", "Llama3 70B", "Llama3 405B", "DeepSeek V3")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["display_model"]].append(row)
    if set(grouped) != set(order):
        raise ValueError(f"{_display(source)}: unexpected Fig. 9 model closure")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 5.5))
    flat = list(axes.flat)
    for index, (ax, name) in enumerate(zip(flat, order)):
        panel = sorted(grouped[name], key=lambda row: int(row["bs"]))
        _paired_bars(
            ax,
            [row["bs"] for row in panel],
            [_float(row[model_column], path=source, column=model_column) for row in panel],
            [_float(row["vllm_stps"], path=source, column="vllm_stps") for row in panel],
            model_label="DeepStack (modeling for 8×B200)",
            reference_label="Ground truth (vLLM on 8×B200)",
        )
        ax.set_title(f"({chr(ord('a') + index)}) {name}", fontweight="bold")
        ax.set_xlabel("Batch size")
        if index % 2 == 0:
            ax.set_ylabel("Decode TPS")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")
    for row_index in (0, 1):
        ymax = max(flat[row_index * 2 + offset].get_ylim()[1] for offset in (0, 1))
        for offset in (0, 1):
            flat[row_index * 2 + offset].set_ylim(0, ymax)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.10, wspace=0.12, hspace=0.54)
    stem = "fig09_b200_decode_validation"
    return _record(
        figure="Fig. 9",
        stem=stem,
        sources=[source],
        outputs=_save(fig, output_dir, stem),
        model_version=model_version,
    )


def _fig10(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(stage_root, "fig10", "model_points.csv")
    rows = _read_csv(source)
    model_column = (
        "paper_legacy_recomputed_tps"
        if model_version == "paper_legacy"
        else "current_corrected_recomputed_tps"
    )
    _require_columns(rows, ("config_key", "bs", model_column, "gpu_vllm_tps"), source)
    specs = (
        ("llama31_405b", "Llama3.1-405B"),
        ("llama33_70b", "Llama3.3-70B"),
        ("qwen3_235b", "Qwen3-235B"),
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["config_key"]].append(row)
    if set(grouped) != {key for key, _ in specs}:
        raise ValueError(f"{_display(source)}: unexpected Fig. 10 model closure")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.45))
    for index, (ax, (key, title)) in enumerate(zip(axes, specs)):
        panel = sorted(grouped[key], key=lambda row: int(row["bs"]))
        _paired_bars(
            ax,
            [row["bs"] for row in panel],
            [_float(row[model_column], path=source, column=model_column) for row in panel],
            [_float(row["gpu_vllm_tps"], path=source, column="gpu_vllm_tps") for row in panel],
            model_label="DeepStack Modeling",
            reference_label="Ground truth (vLLM)",
        )
        ax.set_title(f"({chr(ord('a') + index)}) {title}", fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Decode TPS" if index == 0 else "")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        edgecolor="black",
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.72, bottom=0.25, wspace=0.24)
    stem = "fig10_mi325x_decode_validation"
    return _record(
        figure="Fig. 10",
        stem=stem,
        sources=[source],
        outputs=_save(fig, output_dir, stem),
        model_version=model_version,
    )


def _context_label(kv_mid: int) -> str:
    return f"{int(round(kv_mid / 1024.0))}K"


def _fig12(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    source = _stage_input(
        stage_root, "fig12", model_version, "model_points.csv"
    )
    rows = _read_csv(source)
    _require_columns(
        rows,
        ("prompt_tokens", "kv_mid", "bs", "model_stps", "vllm_stps", "error_pct"),
        source,
    )
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_tokens"])].append(row)
    expected_prompts = (1024, 3072, 7168, 15360, 31744)
    if set(grouped) != set(expected_prompts):
        raise ValueError(f"{_display(source)}: unexpected Fig. 12 context closure")
    colors = {
        1024: "#4c78a8",
        3072: "#f58518",
        7168: "#54a24b",
        15360: "#b279a2",
        31744: "#e45756",
    }

    fig, (ax, error_ax) = plt.subplots(
        2,
        1,
        figsize=(4.65, 2.35),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 0.85], "hspace": 0.07},
    )
    all_errors: list[float] = []
    for prompt in expected_prompts:
        panel = sorted(grouped[prompt], key=lambda row: int(row["bs"]))
        batch = [int(row["bs"]) for row in panel]
        model = [_float(row["model_stps"], path=source, column="model_stps") for row in panel]
        measured = [_float(row["vllm_stps"], path=source, column="vllm_stps") for row in panel]
        errors = [_float(row["error_pct"], path=source, column="error_pct") for row in panel]
        all_errors.extend(errors)
        color = colors[prompt]
        ax.plot(batch, model, color=color, marker="s", ms=3.0, lw=1.0)
        ax.plot(
            batch,
            measured,
            color=color,
            marker="o",
            mfc="white",
            mec=color,
            mew=0.65,
            ms=3.1,
            lw=0.9,
            ls="--",
        )
        error_ax.plot(batch, errors, color=color, marker="o", ms=2.8, lw=0.95)
        ax.text(
            batch[-1] * 1.08,
            model[-1],
            _context_label(int(panel[0]["kv_mid"])),
            color=color,
            fontsize=6.4,
            va="center",
            ha="left",
        )
    ax.set_xlim(0.85, 850)
    ax.set_ylim(0, 1.18 * max(ax.get_ylim()[1], max(line.get_ydata().max() for line in ax.lines)))
    ax.set_ylabel("Decode TPS")
    _style_axis(ax, grid_axis="both")
    ax.tick_params(axis="x", which="both", labelbottom=False)

    error_limit = max(11.5, math.ceil(max(abs(value) for value in all_errors) * 1.12 / 5.0) * 5.0)
    error_ax.axhspan(-10, 10, color="#d9d9d9", alpha=0.4, lw=0)
    error_ax.axhline(0, color="black", lw=0.65)
    error_ax.axhline(-10, color="#777777", lw=0.5, ls=":")
    error_ax.axhline(10, color="#777777", lw=0.5, ls=":")
    error_ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    error_ax.set_xticks(ticks)
    error_ax.set_xticklabels([str(value) for value in ticks])
    error_ax.set_ylim(-error_limit, error_limit)
    error_ax.set_ylabel("Error (%)")
    error_ax.set_xlabel("Batch size")
    _style_axis(error_ax, grid_axis="both")

    style_handles = [
        Line2D([], [], color=EDGE_COLOR, marker="s", lw=1.0, ms=3.0, label="DeepStack Modeling"),
        Line2D(
            [],
            [],
            color=EDGE_COLOR,
            marker="o",
            mfc="white",
            mec=EDGE_COLOR,
            lw=0.9,
            ls="--",
            ms=3.1,
            label="Ground truth (vLLM)",
        ),
    ]
    first_legend = ax.legend(handles=style_handles, loc="upper left", frameon=True, framealpha=0.95, fontsize=6.3)
    ax.add_artist(first_legend)
    wmape = 100.0 * sum(
        abs(_float(row["model_stps"], path=source, column="model_stps") - _float(row["vllm_stps"], path=source, column="vllm_stps"))
        for row in rows
    ) / sum(_float(row["vllm_stps"], path=source, column="vllm_stps") for row in rows)
    error_ax.text(
        0.985,
        0.93,
        f"n={len(rows)}, WMAPE={wmape:.1f}%",
        transform=error_ax.transAxes,
        fontsize=6.3,
        ha="right",
        va="top",
    )
    stem = "fig12_deepseek_v32_dsa"
    return _record(
        figure="Fig. 12",
        stem=stem,
        sources=[source],
        outputs=_save(fig, output_dir, stem),
        model_version=model_version,
    )


FIG13_SPECS = (
    ("torus", "all_gather", "all_gather_rs_ag_ag_3stage_test.csv", "All-Gather"),
    ("torus", "all_to_all", "all_to_all_test.csv", "All-To-All"),
    ("torus", "all_reduce", "all_reduce_ring_test.csv", "All-Reduce"),
    ("switch", "all_gather", "all_gather_halving_doubling_test.csv", "All-Gather"),
    ("switch", "all_to_all", "all_to_all_test.csv", "All-To-All"),
    ("switch", "all_reduce", "all_reduce_rabenseifner_test.csv", "All-Reduce"),
)


def _fig13(
    output_dir: Path, stage_root: Path, model_version: str
) -> dict[str, object]:
    model_source = _stage_input(stage_root, "fig13", "model_points.csv")
    model_rows = _read_csv(model_source)
    _require_columns(
        model_rows,
        (
            "topology",
            "collective",
            "source_file",
            "row_index",
            "recomputed_model_time_ns",
            "ns3_time_ns",
            "analytical_time_ns",
        ),
        model_source,
    )
    panel_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in model_rows:
        key = (row["topology"], row["collective"], int(row["row_index"]))
        panel_key = key[:2]
        if any(int(existing["row_index"]) == key[2] for existing in panel_rows[panel_key]):
            raise ValueError(f"{_display(model_source)}: duplicate model-point key {key}")
        expected_file = next(
            (
                filename
                for topology, collective, filename, _ in FIG13_SPECS
                if (topology, collective) == key[:2]
            ),
            None,
        )
        if expected_file is None or row["source_file"] != expected_file:
            raise ValueError(
                f"{_display(model_source)}: source mismatch for {key}: {row['source_file']}"
            )
        panel_rows[panel_key].append(row)

    panels: list[tuple[list[float], list[float], list[float]]] = []
    for topology, collective, filename, _ in FIG13_SPECS:
        rows = sorted(
            panel_rows.get((topology, collective), []),
            key=lambda row: int(row["row_index"]),
        )
        if not rows:
            raise ValueError(
                f"{_display(model_source)}: no rows for {(topology, collective)}"
            )
        joined: list[tuple[float, float, float]] = []
        for row in rows:
            row_index = int(row["row_index"])
            ns3 = _finite_or_none(row["ns3_time_ns"])
            analytical = _finite_or_none(row["analytical_time_ns"])
            model = _float(
                row["recomputed_model_time_ns"],
                path=model_source,
                column="recomputed_model_time_ns",
            )
            if ns3 is None:
                continue
            if analytical is None:
                raise ValueError(
                    f"{_display(model_source)}: row {row_index} lacks analytical time"
                )
            if min(ns3, analytical, model) <= 0:
                raise ValueError(
                    f"{_display(model_source)}: row {row_index} has non-positive log data"
                )
            joined.append((ns3, model, analytical))
        if not joined:
            raise ValueError(
                f"{_display(model_source)}: no NS-3 points for {(topology, collective)}"
            )
        joined.sort(key=lambda point: point[0])
        panels.append(
            (
                [point[0] for point in joined],
                [point[1] for point in joined],
                [point[2] for point in joined],
            )
        )
    expected_panels = {(topology, collective) for topology, collective, _, _ in FIG13_SPECS}
    if set(panel_rows) != expected_panels:
        extra = sorted(set(panel_rows).difference(expected_panels))
        raise ValueError(f"{_display(model_source)}: unexpected panels {extra}")

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0))
    for index, (ax, panel, spec) in enumerate(zip(axes.flat, panels, FIG13_SPECS)):
        ns3, model, analytical = panel
        ax.plot(ns3, ns3, color=REFERENCE_COLOR, marker="o", ms=2.7, label="ASTRA-sim NS-3")
        ax.plot(ns3, model, color="#ff7f0e", marker="s", ms=2.7, ls="--", label="DeepStack")
        ax.plot(ns3, analytical, color=ANALYTICAL_COLOR, marker="^", ms=2.7, ls="-.", label="ASTRA-sim analytical")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if index < 3:
            ax.set_title(spec[3])
        _style_axis(ax, grid_axis="both")
    axes[1, 1].set_xlabel("ASTRA-sim NS-3 time (ns)")
    handles = [
        Line2D([], [], color=REFERENCE_COLOR, linestyle="-", marker="o", markersize=5, label="ASTRA-sim NS-3"),
        Line2D([], [], color="#ff7f0e", linestyle="--", marker="s", markersize=5, label="DeepStack"),
        Line2D([], [], color=ANALYTICAL_COLOR, linestyle="-.", marker="^", markersize=5, label="ASTRA-sim analytical"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), frameon=True)
    fig.tight_layout(rect=(0.06, 0.0, 1.0, 0.95))
    for row_index, name in ((0, "Torus"), (1, "Switch")):
        bbox = axes[row_index, 0].get_position()
        center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.008, center, name, fontweight="bold", ha="left", va="center", rotation=90, fontsize=12)
        fig.text(0.028, center, "Time (ns)", ha="left", va="center", rotation=90, fontsize=10)
    stem = "fig13_noc_model_validation"
    return _record(
        figure="Fig. 13",
        stem=stem,
        sources=[model_source],
        outputs=_save(fig, output_dir, stem),
        model_version=model_version,
    )


def render_accuracy_figures(
    output_dir: Path,
    model_version: str = "paper_legacy",
    *,
    result_root: Path = RESULTS_DIR / "reproduce",
) -> list[dict[str, object]]:
    """Render retained validation figures as PNG/PDF manifest records.

    Inputs are read only from ``result_root/stages``.  ``output_dir`` must be
    inside ``result_root/figures``; nested symlinks cannot escape either scoped
    directory.
    """

    if model_version not in MODEL_VERSIONS:
        raise ValueError(
            f"model_version must be one of {MODEL_VERSIONS}, got {model_version!r}"
        )
    stage_root, destination = _scoped_paths(Path(result_root), Path(output_dir))
    destination.mkdir(parents=True, exist_ok=True)
    _style()
    renderers = (_fig08, _fig09, _fig10, _fig12, _fig13)
    return [
        renderer(destination, stage_root, model_version) for renderer in renderers
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULTS_DIR / "reproduce",
        help="workflow result directory (default: results/reproduce)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="optional directory under RESULT_ROOT/figures",
    )
    parser.add_argument("--model-version", choices=MODEL_VERSIONS, default="paper_legacy")
    args = parser.parse_args(argv)
    result_root = _resolve_path(args.result_root)
    output_dir = args.output_dir or result_root / "figures"
    records = render_accuracy_figures(
        output_dir,
        args.model_version,
        result_root=result_root,
    )
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
