"""Reproduce Fig. 8's 8xH100 kernel-model validation on CPU.

The 8xH100 measurements are immutable reference data.  This module reruns
only DeepStack/TileSight's analytical model for all 52 published problem
points, computes the five per-kernel MAPEs and the paper's sample-weighted
MAPE, and writes a normalized comparison table.

Run from any directory with::

    python -m ae.fig08

All default paths are anchored at the artifact root.  No GPU is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .paths import display_path


ROOT = Path(__file__).resolve().parents[1]
DEEPSTACK_SRC = ROOT / "src" / "deepstack"
TILESIGHT_SRC = ROOT / "src" / "tilesight"
REFERENCE_DIR = ROOT / "data" / "fig08" / "reference" / "paper_csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reproduce" / "stages" / "fig08"

MODEL_COLUMN = "overall_time/ms(modeling, overlap)"
GT_COLUMN = "ref_time/ms(gpu)"
EXPECTED_POINTS = 52
EXPECTED_PAPER_WEIGHTED_MAPE = 8.366124199417992
EXPECTED_PAPER_AG_MAPE = 3.9717596932736123
DETERMINISTIC_SEED = 0

# The exact ordering and semantics used by the published subplot script.
CATEGORIES = (
    ("all_reduce_gemm", "moe_reduce_ar.csv", MODEL_COLUMN, GT_COLUMN, 8),
    ("moe_ep_all_to_all", "ep_a2a.csv", "modeling_dispatch_time/ms", "dispatch_ms", 8),
    ("ulysses_attention", "ulysses.csv", MODEL_COLUMN, GT_COLUMN, 12),
    ("all_gather_gemm", "ag_gemm.csv", MODEL_COLUMN, GT_COLUMN, 12),
    ("reduce_scatter_gemm", "gemm_rs.csv", MODEL_COLUMN, GT_COLUMN, 12),
)

# Original MoE AR inputs are needed because routed-expert count is not present
# in the published result CSV.  Tuple fields are
# (tokens, intermediate, hidden, routed experts, activated experts).
MOE_AR_INPUTS = (
    (1024, 4096, 4096, 128, 32),
    (1024, 8192, 8192, 128, 32),
    (1024, 4096, 4096, 256, 32),
    (1024, 8192, 8192, 256, 32),
    (1024, 4096, 4096, 128, 64),
    (1024, 8192, 8192, 128, 64),
    (1024, 4096, 4096, 256, 64),
    (1024, 8192, 8192, 256, 64),
)

TILING_AR = (128, 128, 64, 64, 64, 64, 3, 12)
TILING_DEFAULT = (128, 256, 64, 64, 128, 64, 3, 12)


def _activate_vendored_sources() -> None:
    for source_root in (TILESIGHT_SRC, DEEPSTACK_SRC):
        source = str(source_root)
        if source not in sys.path:
            sys.path.insert(0, source)


def _artifact_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Use artifact-relative paths when possible, absolute paths otherwise."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


class H100KernelModel:
    """Small, faithful extraction of the five original Fig. 8 model scripts."""

    def __init__(self) -> None:
        _activate_vendored_sources()
        import torch
        from mosaic.noc.noc_config_set import h100x8
        from mosaic.utils import OpBytes, Tensor_Loc
        from tilesight.arch import H100_SXM

        self.torch = torch
        self.single_chip = H100_SXM().set_to_microbench()
        self.noc8 = h100x8()
        self.gemm_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.float16, loc="ddr"),
            input2=Tensor_Loc(dtype=torch.float16, loc="ddr"),
            output=Tensor_Loc(dtype=torch.float16, loc="ddr"),
        )

    def _gemm_time(
        self,
        m: float,
        n: float,
        k: float,
        tiling: tuple[int, ...],
        *,
        batch: int = 1,
    ) -> float:
        from tilesight.fused_op_dtype_wave.matmul_fused_op_new_api_wave import (
            calculate_matmul_resource_utilization_new,
        )
        from tilesight.fusion_support.hete_post_process_single_op import (
            hete_post_process_tensor_core_op,
        )
        from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
        from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion

        tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, row_panel = tiling
        input_bytes, weight_bytes, _ = self.gemm_bytes.get_dtype_bytes()
        del input_bytes
        mma_type = "utcmma_cta1" if self.single_chip.support_utcmma else (
            "wgmma" if self.single_chip.support_wgmma else "wmma"
        )
        raw = calculate_matmul_resource_utilization_new(
            [m, n, k],
            [tb_m, tb_n, tb_k],
            [wp_m, wp_n, wp_k],
            int(stages),
            self.single_chip,
            self.gemm_bytes.to_mem_levels(),
            row_panel=row_panel,
            batch=batch,
            mma_type=mma_type,
        )
        post = hete_post_process_tensor_core_op(raw, self.single_chip, weight_bytes)
        fused_register = hete_reg_fusion([post], self.single_chip)
        fused_smem = hete_smem_fusion(
            [fused_register], [m / tb_m, n / tb_n, batch], self.single_chip
        )
        return float(fused_smem[0])

    @staticmethod
    def _e2e(compute: float, hop: float, link: float, waves: int) -> float:
        from mosaic.utils.get_comp_comm_e2e_time import get_comp_comm_e2e_time

        return float(
            get_comp_comm_e2e_time(
                compute_time=compute,
                network_hop_latency=hop,
                network_link_time=link,
                waves=waves,
                overlap=True,
            )
        )

    @staticmethod
    def _all_to_all(nodes: int, noc: Any, num_bytes: float) -> tuple[float, float]:
        from mosaic.noc.noc_topo import get_extend_max_routes
        from mosaic.noc.traffic_matrix import TrafficMatrix

        traffic = TrafficMatrix(nodes)
        traffic.add_intra_group_traffic(
            "tp", num_bytes, tp=nodes, ep=1, sp=1, cp=1, dp=1, pp=1
        )
        hop, link, _ = get_extend_max_routes(traffic, noc)
        return float(hop), float(link)

    @staticmethod
    def _all_gather_recursive(
        nodes: int, noc: Any, num_bytes: float
    ) -> tuple[float, float]:
        from mosaic.noc.noc_topo import get_extend_max_routes
        from mosaic.noc.traffic_matrix import TrafficMatrix

        if nodes & (nodes - 1):
            raise ValueError("recursive-doubling all-gather requires a power of two")
        traffic = TrafficMatrix(nodes)
        stages = int(math.log2(nodes))
        max_stride = 1 << (stages - 1)
        total_hop = 0.0
        total_link = 0.0
        for stage in range(stages):
            stride = max_stride >> stage
            bytes_this_stage = num_bytes * (1 << stage)
            pairs = [[bytes_this_stage, rank, rank ^ stride] for rank in range(nodes)]
            traffic.add_intra_group_traffic_pair_bulk(
                "tp", pairs, tp=nodes, ep=1, sp=1, cp=1, dp=1, pp=1
            )
            hop, link, _ = get_extend_max_routes(traffic, noc)
            total_hop += float(hop)
            total_link += float(link)
            traffic.reset()
        return total_hop, total_link

    def all_reduce_gemm(self) -> list[float]:
        input_bytes, _, output_bytes = self.gemm_bytes.get_dtype_bytes()
        results: list[float] = []
        for tokens, intermediate, hidden, routed, activated in MOE_AR_INPUTS:
            m = tokens * activated
            n = hidden
            k = math.ceil(intermediate / 8)
            num_bytes = m * n * output_bytes / 8
            hop, link = self._all_to_all(8, self.noc8, num_bytes / 4)
            hop *= 4
            link *= 4
            compute = self._gemm_time(
                m / (routed / 8), n, k / 4, TILING_AR
            ) * 4 * (routed / 8)
            overall = self._e2e(compute, hop, link, waves=4)
            tb_m, tb_n, tb_k, *_ = TILING_AR
            additional = (
                (tb_m * tb_k + tb_n * tb_k)
                * input_bytes
                * self.single_chip.sm_count
                / self.single_chip.l2_bandwidth
            )
            additional += (
                tb_m
                * tb_n
                * output_bytes
                * self.single_chip.sm_count
                / self.single_chip.ddr_bandwidth
            )
            additional += (
                (m / 8 * n * output_bytes) * 2 / self.single_chip.ddr_bandwidth
            )
            results.append((overall + additional) * 1000)
        return results

    def ep_all_to_all(self, reference: Sequence[dict[str, str]]) -> list[float]:
        from mosaic.collectives import ep_all_to_all_wrapper
        from mosaic.noc.noc_config_set import h100x8
        from mosaic.parallelism import ParallelScheme
        from mosaic.utils import Modeling_Granularity

        parallel = ParallelScheme(
            tp=1, ep=8, sp=1, cp=1, dp=1, pp=1, fsdp=False, ep1=1, ep2=8
        )
        granularity = Modeling_Granularity(
            mode="coarse", comp_comm_overlap=True, auto_tune=False
        )
        noc = h100x8()
        results: list[float] = []
        for row in reference:
            m = int(row["M"])
            hidden = int(row["N"])
            routed = int(row["G"])
            activated = int(row["topk"])
            seq = m * 8
            # The current wrapper selects its deterministic analytical path for
            # these seq lengths.  A valid-shaped routing array is nevertheless
            # supplied to preserve the original call contract.
            routing = np.zeros((seq, activated), dtype=np.int64)
            hop, link, _ = ep_all_to_all_wrapper(
                parallel,
                noc,
                granularity,
                2 * hidden,
                routing,
                1,
                seq,
                routed,
                activated,
                1.4,
            )
            results.append((float(hop) + float(link)) * 1000)
        return results

    def ulysses(self, reference: Sequence[dict[str, str]]) -> list[float]:
        from mosaic.noc.noc_config_set import h100x

        input_bytes, _, output_bytes = self.gemm_bytes.get_dtype_bytes()
        results: list[float] = []
        for row in reference:
            bs = int(row["bs"])
            heads = int(row["nh"])
            seq = int(row["seqlen"])
            head_dim = int(row["hd"])
            out_features = int(row["out_features"])
            comm_sms = int(row["num_comm_sm"])
            sp = int(row["sp_size"])
            noc = h100x(sp)
            num_bytes = bs * seq * heads * head_dim * input_bytes / sp
            hop, link = self._all_to_all(sp, noc, num_bytes)
            compute = self._gemm_time(
                seq / sp,
                out_features,
                head_dim * heads,
                TILING_DEFAULT,
                batch=bs,
            )
            # This value is reported as time_single_chip in the original CSV;
            # the overlap equation intentionally uses the unscaled compute
            # value, matching the original Fig. 8 script exactly.
            _reported_compute = compute * self.single_chip.sm_count / (
                self.single_chip.sm_count - comm_sms
            )
            del _reported_compute
            overall = self._e2e(compute, hop, link, waves=3)
            tb_m, tb_n, tb_k, *_ = TILING_DEFAULT
            additional = (
                (tb_m * tb_k + tb_n * tb_k)
                * input_bytes
                * self.single_chip.sm_count
                / self.single_chip.l2_bandwidth
            )
            additional += (
                tb_m
                * tb_n
                * output_bytes
                * self.single_chip.sm_count
                / self.single_chip.ddr_bandwidth
            )
            additional += (
                (bs * seq * heads * head_dim / sp * output_bytes)
                * 2
                / self.single_chip.ddr_bandwidth
            )
            results.append((overall + additional) * 1000)
        return results

    def all_gather_gemm(self, reference: Sequence[dict[str, str]]) -> list[float]:
        input_bytes, _, output_bytes = self.gemm_bytes.get_dtype_bytes()
        results: list[float] = []
        for row in reference:
            # N is already divided by 8 in the published CSV.
            m, n, k = (int(row[name]) for name in ("M", "N", "K"))
            num_bytes = m * k * input_bytes / 8
            hop, link = self._all_gather_recursive(8, self.noc8, num_bytes)
            compute = self._gemm_time(m, n, k, TILING_DEFAULT)
            tb_m, tb_n, tb_k, *_ = TILING_DEFAULT
            compute += (
                (tb_m * tb_k + tb_n * tb_k)
                * input_bytes
                * self.single_chip.sm_count
                / self.single_chip.l2_bandwidth
            )
            compute += (
                tb_m
                * tb_n
                * output_bytes
                * self.single_chip.sm_count
                / self.single_chip.ddr_bandwidth
            )
            waves = math.ceil((m / tb_m) * (n / tb_n) / self.single_chip.sm_count)
            results.append(self._e2e(compute, hop, link, waves) * 1000)
        return results

    def reduce_scatter_gemm(
        self, reference: Sequence[dict[str, str]]
    ) -> list[float]:
        input_bytes, _, output_bytes = self.gemm_bytes.get_dtype_bytes()
        results: list[float] = []
        for row in reference:
            # K is already divided by 8 in the published CSV.
            m, n, k = (int(row[name]) for name in ("M", "N", "K"))
            num_bytes = m * n * output_bytes / 8
            hop, link = self._all_to_all(8, self.noc8, num_bytes)
            compute = self._gemm_time(m, n, k, TILING_DEFAULT)
            tb_m, tb_n, tb_k, *_ = TILING_DEFAULT
            waves = math.ceil((m / tb_m) * (n / tb_n) / self.single_chip.sm_count)
            overall = self._e2e(compute, hop, link, waves)
            additional = (
                (tb_m * tb_k + tb_n * tb_k)
                * input_bytes
                * self.single_chip.sm_count
                / self.single_chip.l2_bandwidth
            )
            additional += (
                tb_m
                * tb_n
                * output_bytes
                * self.single_chip.sm_count
                / self.single_chip.ddr_bandwidth
            )
            additional += m * n * output_bytes * 2 / self.single_chip.ddr_bandwidth
            results.append((overall + additional) * 1000)
        return results


def reproduce_points(reference_dir: Path = REFERENCE_DIR) -> list[dict[str, Any]]:
    """Rerun and normalize all 52 model points."""

    # TileSight's legacy L2 reuse-distance estimator randomizes each wave's
    # block order.  The paper scripts did not seed it, which causes tiny
    # run-to-run changes (~1e-4%).  Pinning the seed makes the artifact CSVs
    # byte-reproducible while retaining the estimator's exact algorithm.
    np.random.seed(DETERMINISTIC_SEED)
    references = {
        category: _read_csv(reference_dir / filename)
        for category, filename, _, _, _ in CATEGORIES
    }
    model = H100KernelModel()
    current = {
        "all_reduce_gemm": model.all_reduce_gemm(),
        "moe_ep_all_to_all": model.ep_all_to_all(references["moe_ep_all_to_all"]),
        "ulysses_attention": model.ulysses(references["ulysses_attention"]),
        "all_gather_gemm": model.all_gather_gemm(references["all_gather_gemm"]),
        "reduce_scatter_gemm": model.reduce_scatter_gemm(
            references["reduce_scatter_gemm"]
        ),
    }

    normalized: list[dict[str, Any]] = []
    for category, filename, model_column, gt_column, expected_count in CATEGORIES:
        rows = references[category]
        values = current[category]
        if len(rows) != expected_count or len(values) != expected_count:
            raise AssertionError(
                f"{category}: expected {expected_count} points, "
                f"got {len(rows)} reference/{len(values)} model"
            )
        for index, (row, current_ms) in enumerate(zip(rows, values, strict=True)):
            paper_ms = float(row[model_column])
            gpu_ms = float(row[gt_column])
            normalized.append(
                {
                    "category": category,
                    "point_index": index,
                    "paper_source_csv": filename,
                    "paper_model_ms": paper_ms,
                    "current_model_ms": current_ms,
                    "gpu_reference_ms": gpu_ms,
                    "paper_abs_pct_error": abs(paper_ms - gpu_ms) / gpu_ms * 100,
                    "current_abs_pct_error": abs(current_ms - gpu_ms) / gpu_ms * 100,
                    "current_vs_paper_pct": (current_ms - paper_ms) / paper_ms * 100,
                }
            )
    return normalized


def summarize(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return five category summaries plus the 52-point weighted row."""

    summaries: list[dict[str, Any]] = []
    for category, _, _, _, expected_count in CATEGORIES:
        selected = [row for row in points if row["category"] == category]
        if len(selected) != expected_count:
            raise AssertionError(
                f"{category}: expected {expected_count} normalized points, got {len(selected)}"
            )
        summaries.append(_summary_row(category, selected))
    summaries.append(_summary_row("all_52_sample_weighted", points))
    return summaries


def _summary_row(scope: str, points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paper_errors = [float(row["paper_abs_pct_error"]) for row in points]
    current_errors = [float(row["current_abs_pct_error"]) for row in points]
    drifts = [abs(float(row["current_vs_paper_pct"])) for row in points]
    return {
        "scope": scope,
        "points": len(points),
        "paper_mape_pct": float(np.mean(paper_errors)),
        "current_mape_pct": float(np.mean(current_errors)),
        "current_vs_paper_mean_abs_pct": float(np.mean(drifts)),
        "current_vs_paper_max_abs_pct": float(np.max(drifts)),
    }


def verify(points: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]]) -> None:
    """Verify data closure and the two numerical claims printed in the paper."""

    if len(points) != EXPECTED_POINTS:
        raise AssertionError(f"expected {EXPECTED_POINTS} points, got {len(points)}")
    unique = {(row["category"], int(row["point_index"])) for row in points}
    if len(unique) != EXPECTED_POINTS:
        raise AssertionError("Fig. 8 category/index keys are not unique")

    by_scope = {str(row["scope"]): row for row in summary}
    weighted = float(by_scope["all_52_sample_weighted"]["paper_mape_pct"])
    ag = float(by_scope["all_gather_gemm"]["paper_mape_pct"])
    if not math.isclose(weighted, EXPECTED_PAPER_WEIGHTED_MAPE, abs_tol=1e-12):
        raise AssertionError(
            f"paper 52-point MAPE {weighted:.15g} != {EXPECTED_PAPER_WEIGHTED_MAPE:.15g}"
        )
    if not math.isclose(ag, EXPECTED_PAPER_AG_MAPE, abs_tol=1e-12):
        raise AssertionError(
            f"paper AG-GEMM MAPE {ag:.15g} != {EXPECTED_PAPER_AG_MAPE:.15g}"
        )
    if round(weighted, 1) != 8.4:
        raise AssertionError(f"52-point MAPE does not round to 8.4%: {weighted}")
    if round(ag, 2) != 3.97:
        raise AssertionError(f"AG-GEMM MAPE does not round to 3.97%: {ag}")
    for category, filename, _, _, expected_count in CATEGORIES:
        path = REFERENCE_DIR / filename
        if not path.is_file():
            raise AssertionError(f"missing GPU reference CSV: {path}")
        if len(_read_csv(path)) != expected_count:
            raise AssertionError(f"unexpected row count in {path}")


def _plot(points: Sequence[dict[str, Any]], output: Path) -> None:
    """Render a compact, deterministic equivalent of the five Fig. 8 panels."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    widths = (1, 1, 1.5)
    figure = plt.figure(figsize=(10, 5.5))
    top = figure.add_gridspec(
        1, 3, left=0.06, right=0.95, top=0.85, bottom=0.53,
        wspace=0.40, width_ratios=widths,
    )
    bottom = figure.add_gridspec(
        1, 2, left=0.06, right=0.95, top=0.45, bottom=0.05, wspace=0.30
    )
    axes = [
        figure.add_subplot(top[0, 0]),
        figure.add_subplot(top[0, 1]),
        figure.add_subplot(top[0, 2]),
        figure.add_subplot(bottom[0, 0]),
        figure.add_subplot(bottom[0, 1]),
    ]
    titles = (
        "(a) All-Reduce GEMM",
        "(b) MoE EP All-To-All",
        "(c) Ulysses Attention",
        "(d) All-Gather GEMM",
        "(e) Reduce-Scatter GEMM",
    )
    for axis, (category, *_), title in zip(axes, CATEGORIES, titles, strict=True):
        selected = [row for row in points if row["category"] == category]
        x = np.arange(len(selected))
        width = 0.38
        axis.bar(
            x - width / 2,
            [float(row["current_model_ms"]) for row in selected],
            width,
            label="DeepStack (Modeling)",
            edgecolor="black",
        )
        axis.bar(
            x + width / 2,
            [float(row["gpu_reference_ms"]) for row in selected],
            width,
            label="Ground Truth (8xH100)",
            edgecolor="black",
        )
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_ylabel("Time (ms)", fontsize=12)
        axis.set_xticks([])
        axis.tick_params(axis="y", labelsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=13,
        frameon=True,
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)),
        help="result directory, relative to the artifact root by default",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip rendering the comparison PNG"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="write outputs without checking the bundled paper statistics",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = _artifact_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    points = reproduce_points()
    summary = summarize(points)
    points_path = output_dir / "model_points.csv"
    summary_path = output_dir / "summary.csv"
    _write_csv(points_path, points)
    _write_csv(summary_path, summary)
    if not args.no_verify:
        verify(points, summary)
    if not args.no_plot:
        _plot(points, output_dir / "modeling_accuracy.png")

    elapsed = time.perf_counter() - started
    by_scope = {str(row["scope"]): row for row in summary}
    weighted = by_scope["all_52_sample_weighted"]
    ag = by_scope["all_gather_gemm"]
    hashed_files = [
        *sorted(REFERENCE_DIR.glob("*.csv")),
        ROOT / "ae" / "fig08.py",
        ROOT / "docs" / "result_matrix.md",
        points_path,
        summary_path,
    ]
    plot_path = output_dir / "modeling_accuracy.png"
    if plot_path.is_file():
        hashed_files.append(plot_path)
    manifest = {
        "status": "PASS" if not args.no_verify else "SKIPPED",
        "gpu_execution": False,
        "numpy_seed": DETERMINISTIC_SEED,
        "points": len(points),
        "wall_time_seconds": elapsed,
        "paper_weighted_mape_pct": weighted["paper_mape_pct"],
        "current_weighted_mape_pct": weighted["current_mape_pct"],
        "paper_ag_gemm_mape_pct": ag["paper_mape_pct"],
        "current_ag_gemm_mape_pct": ag["current_mape_pct"],
        "files": {_display_path(path): _sha256(path) for path in hashed_files},
    }
    manifest_path = output_dir / "verification.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Fig. 8 verification: {manifest['status']}")
    print(f"Model points recomputed: {len(points)} (CPU only; GPU runs: 0)")
    print(
        "52-point sample-weighted MAPE: "
        f"paper {float(weighted['paper_mape_pct']):.6f}% -> "
        f"current {float(weighted['current_mape_pct']):.6f}%"
    )
    print(
        "AG-GEMM MAPE: "
        f"paper {float(ag['paper_mape_pct']):.6f}% -> "
        f"current {float(ag['current_mape_pct']):.6f}%"
    )
    print(f"Points CSV: {display_path(points_path)}")
    print(f"Summary CSV: {display_path(summary_path)}")
    print(f"Verification manifest: {display_path(manifest_path)}")
    print(f"Wall time: {elapsed:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
