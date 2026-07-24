"""CPU-only reproduction of Fig. 10's 4xMI325X validation.

The vLLM measurements under ``data/fig10/reference`` are immutable inputs;
this module never launches a GPU workload.  It reruns all 28 fixed
DeepStack model points twice:

* ``paper_legacy`` preserves the submitted model's activated-expert binning;
* ``current_corrected`` uses the post-submission bug fix.

Run from the artifact root with::

    python -m ae.fig10 --workers 16

Default paths are anchored at the artifact root, not the current directory.
Only the 4xMI325X / TP4 subset used by Fig. 10 is in the default experiment.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ROOT, DEEPSTACK_SRC, activate_vendored_sources, display_path


DATA_DIR = ROOT / "data" / "fig10"
REFERENCE_DIR = DATA_DIR / "reference"
PAPER_LEGACY_DIR = DATA_DIR / "paper_legacy"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reproduce" / "stages" / "fig10"

EXPECTED_POINTS = 28
EXPECTED_PAPER_CLAIM_WMAPE_PCT = 13.095363138127121
EXPECTED_PAPER_AGGREGATE_WMAPE_PCT = 16.27604525871703
VERSIONS = ("paper_legacy", "current_corrected")
DETERMINISTIC_SEED = 0
LEGACY_REPLAY_REL_TOL = 2e-5
LEGACY_METRIC_ABS_TOL_PCT = 2e-4

# These are the exact files copied from the paper-figure snapshot.  The three
# measurement CSVs are the only GPU-produced inputs needed for Fig. 10.
EXPECTED_SHA256 = {
    "reference/llama31_405b_mi325x4_tp4_io1k1k.csv":
        "b2fc473b21ff32ac6146929335a2bac5cdfeead2618fcd9a9f499ac0be63a532",
    "reference/llama33_70b_mi325x4_tp4_io1k1k.csv":
        "165e8f5bdc71b8224779a5488fe9c8fcbae6f4f690ee0b29eebe306c3429dbfd",
    "reference/qwen3_235b_mi325x4_tp4_io1k1k.csv":
        "cbd8a9aa26411edd96efc3e9948d6a3f71b774e9c2d53773ea08831ed06cd151",
    "paper_legacy/llama31_405b_mi325x4_tp4_io1k1k_all2all_spec_merged.csv":
        "e15e28e896e2f62d3dfed9c1cefe27efa6a0b0ab33fc675a769ec0d2145d4e02",
    "paper_legacy/llama33_70b_mi325x4_tp4_io1k1k_all2all_spec_merged.csv":
        "6a45802160aa88e75400557b135686cbd972d1399fc8cff2e0bc8d7bfb80bac1",
    "paper_legacy/qwen3_235b_mi325x4_tp4_io1k1k_all2all_spec_merged.csv":
        "e04aca749bf35a2824cb4e17d5ae488ba7fe46af534454cd226c345e85cb8cc0",
}

CONFIGS = (
    {
        "key": "llama31_405b",
        "label": "Llama-3.1-405B / MI325x4 (TP4) io1k1k",
        "model": "Llama3_405b",
        "gt_file": "llama31_405b_mi325x4_tp4_io1k1k.csv",
        "model_file": "llama31_405b_mi325x4_tp4_io1k1k_all2all_spec_merged.csv",
        "expected_points": 8,
    },
    {
        "key": "llama33_70b",
        "label": "Llama-3.3-70B / MI325x4 (TP4) io1k1k",
        "model": "Llama3_70b",
        "gt_file": "llama33_70b_mi325x4_tp4_io1k1k.csv",
        "model_file": "llama33_70b_mi325x4_tp4_io1k1k_all2all_spec_merged.csv",
        "expected_points": 10,
    },
    {
        "key": "qwen3_235b",
        "label": "Qwen3-235B / MI325x4 (TP4) io1k1k",
        "model": "Qwen3_235b_a22b",
        "gt_file": "qwen3_235b_mi325x4_tp4_io1k1k.csv",
        "model_file": "qwen3_235b_mi325x4_tp4_io1k1k_all2all_spec_merged.csv",
        "expected_points": 10,
    },
)

REQUIRED_MODEL_COLUMNS = (
    "arch", "noc", "model", "bs", "minibatch", "seq", "tp", "ep",
    "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
    "tp_transform_moe", "kv_len_1", "kv_len_2", "kv_len_3", "kv_len_4",
    "utps_avg", "stps_avg", "stats_run_id",
)

_TRACE_CACHE: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_input_hashes() -> dict[str, str]:
    """Verify all bundled inputs before using either model path."""

    actual: dict[str, str] = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = DATA_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing Fig. 10 input: {path}")
        digest = _sha256(path)
        actual[relative] = digest
        if digest != expected:
            raise AssertionError(
                f"Fig. 10 input hash mismatch for {relative}: "
                f"got {digest}, expected {expected}"
            )
    return actual


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def load_fixed_points() -> pd.DataFrame:
    """Load and validate the exact submitted configurations and GPU values."""

    frames: list[pd.DataFrame] = []
    for config in CONFIGS:
        model_path = PAPER_LEGACY_DIR / str(config["model_file"])
        gt_path = REFERENCE_DIR / str(config["gt_file"])
        model = pd.read_csv(model_path)
        gt = pd.read_csv(gt_path)

        missing = [column for column in REQUIRED_MODEL_COLUMNS if column not in model.columns]
        if missing:
            raise ValueError(f"{model_path.name} is missing columns: {missing}")
        if not {"BS", "system_decode_tps"}.issubset(gt.columns):
            raise ValueError(f"{gt_path.name} lacks BS/system_decode_tps")
        if len(model) != int(config["expected_points"]):
            raise AssertionError(
                f"{model_path.name}: expected {config['expected_points']} rows, got {len(model)}"
            )
        if model["bs"].duplicated().any() or gt["BS"].duplicated().any():
            raise AssertionError(f"duplicate batch size in {config['key']} inputs")

        merged = model.merge(
            gt[["BS", "system_decode_tps"]],
            left_on="bs",
            right_on="BS",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != int(config["expected_points"]):
            raise AssertionError(f"GPU/model BS mismatch for {config['key']}")
        if set(merged["arch"]) != {"MI325XSpec"}:
            raise AssertionError(f"unexpected arch in {config['key']}: {set(merged['arch'])}")
        if set(merged["noc"]) != {"mi325x4_all2all"}:
            raise AssertionError(f"unexpected NoC in {config['key']}: {set(merged['noc'])}")
        if set(merged["model"]) != {config["model"]}:
            raise AssertionError(f"unexpected model class in {config['key']}")
        if set(merged["tp"]) != {4} or set(merged["ep"]) != {1}:
            raise AssertionError(f"Fig. 10 expects fixed TP4/EP1 for {config['key']}")

        merged.insert(0, "point_id", [f"{config['key']}:bs{int(bs)}" for bs in merged["bs"]])
        merged.insert(1, "config", str(config["label"]))
        merged.insert(2, "config_key", str(config["key"]))
        merged.rename(
            columns={
                "utps_avg": "paper_legacy_archived_utps",
                "stps_avg": "paper_legacy_archived_tps",
                "system_decode_tps": "gpu_vllm_tps",
            },
            inplace=True,
        )
        frames.append(merged)

    points = pd.concat(frames, ignore_index=True, sort=False)
    if len(points) != EXPECTED_POINTS or points["point_id"].nunique() != EXPECTED_POINTS:
        raise AssertionError(
            f"expected {EXPECTED_POINTS} unique Fig. 10 points, got {len(points)}"
        )
    return points.sort_values(["config_key", "bs"]).reset_index(drop=True)


def _trace_for_model(model_name: str) -> Any:
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape

    trace_name = "qwen" if model_name == "Qwen3_235b_a22b" else "deepseek"
    if trace_name not in _TRACE_CACHE:
        if trace_name == "qwen":
            path = (
                DEEPSTACK_SRC / "mosaic" / "data" / "aime_qwen_235b"
                / "qwen3_moe_activations_batch0.npz"
            )
        else:
            path = (
                DEEPSTACK_SRC / "mosaic" / "data" / "aime_ds_r1"
                / "moe_activations_batch0.npz"
            )
        if not path.is_file():
            raise FileNotFoundError(f"missing routed-expert trace: {path}")
        _, _TRACE_CACHE[trace_name] = load_npz_routing_keep_shape(str(path))
    return _TRACE_CACHE[trace_name]


def _model_worker(row: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one fixed config/version with the vendored CPU model."""

    activate_vendored_sources()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    version = str(row["model_version"])
    if version not in VERSIONS:
        raise ValueError(f"unsupported model version: {version}")
    os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = version

    import importlib
    import numpy as np
    import torch

    from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import modeling_decode
    from mosaic.parallelism import ParallelScheme
    from mosaic.utils import Modeling_Granularity

    torch.set_num_threads(1)
    # TileSight's cache-reuse estimator samples a deterministic SM schedule.
    # Every task gets a fresh process (see ``recompute``), so this seed also
    # prevents a prior shape's simulation cache from influencing a point.
    random.seed(DETERMINISTIC_SEED)
    np.random.seed(DETERMINISTIC_SEED)
    torch.manual_seed(DETERMINISTIC_SEED)
    logging.getLogger("mosaic.parallelism.parallel").setLevel(logging.ERROR)
    arch_module = importlib.import_module("mosaic.arch")
    llm_module = importlib.import_module("mosaic.llm_arch")
    noc_module = importlib.import_module("mosaic.noc.noc_config_set")

    scheme = ParallelScheme(
        tp=int(row["tp"]), ep=int(row["ep"]), ep1=int(row["ep1"]),
        ep2=int(row["ep2"]), sp=int(row["sp"]), cp=int(row["cp"]),
        dp=int(row["dp"]), fsdp=_as_bool(row["fsdp"]), pp=int(row["pp"]),
    )
    non_moe = dataclasses.replace(
        scheme, ep=1, ep1=1, ep2=1, dp=scheme.dp * scheme.ep
    )
    kv_lengths = [int(row[f"kv_len_{index}"]) for index in range(1, 5)]
    model_name = str(row["model"])
    started = time.perf_counter()
    times, _, _ = modeling_decode(
        model_arch=getattr(llm_module, model_name)(),
        bs=int(row["minibatch"]),
        seq=int(row["seq"]),
        cached_kv_list=kv_lengths,
        moe_parallel=scheme,
        non_moe_parallel=non_moe,
        single_chip=getattr(arch_module, str(row["arch"]))(),
        noc_hierarchy=getattr(noc_module, str(row["noc"]))(),
        granularity=Modeling_Granularity("coarse", True, False, True),
        routing_array=_trace_for_model(model_name),
        tp_transform_moe=str(row["tp_transform_moe"]),
    )
    utps = sum(1.0 / values[-1] / scheme.pp for values in times) / len(times)
    stps = sum(int(row["minibatch"]) / values[-1] for values in times) / len(times)
    return {
        "point_id": str(row["point_id"]),
        "model_version": version,
        "recomputed_utps": utps,
        "recomputed_tps": stps,
        "model_runtime_s": time.perf_counter() - started,
    }


def recompute(points: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Rerun all 28 model points on both legacy and corrected paths."""

    activate_vendored_sources()
    tasks: list[dict[str, Any]] = []
    for row in points.to_dict(orient="records"):
        for version in VERSIONS:
            task = dict(row)
            task["model_version"] = version
            tasks.append(task)

    computed: list[dict[str, Any]] = []
    # A few TileSight reuse-distance routines retain shape-specific simulation
    # state.  One task per spawned child makes the 56 evaluations independent
    # of worker scheduling and reproducible across repeated artifact runs.
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        mp_context=mp.get_context("spawn"),
        max_tasks_per_child=1,
    ) as executor:
        futures = {executor.submit(_model_worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                computed.append(future.result())
            except Exception as error:
                raise RuntimeError(
                    f"Fig. 10 model failed for {task['point_id']} / {task['model_version']}"
                ) from error

    model = pd.DataFrame(computed)
    if len(model) != EXPECTED_POINTS * len(VERSIONS):
        raise AssertionError(f"expected 56 model evaluations, got {len(model)}")
    if model.duplicated(["point_id", "model_version"]).any():
        raise AssertionError("duplicate Fig. 10 model result")

    wide = model.pivot(index="point_id", columns="model_version")
    wide.columns = [f"{version}_{metric}" for metric, version in wide.columns]
    wide.reset_index(inplace=True)
    return wide


def build_model_points(points: pd.DataFrame, computed: pd.DataFrame) -> pd.DataFrame:
    """Build a compact, auditable table with both model paths."""

    keep = [
        "point_id", "config", "config_key", "model", "bs", "minibatch",
        "seq", "arch", "noc", "tp", "ep", "ep1", "ep2", "sp", "cp",
        "dp", "fsdp", "pp", "tp_transform_moe", "kv_len_1", "kv_len_2",
        "kv_len_3", "kv_len_4", "gpu_vllm_tps",
        "paper_legacy_archived_utps", "paper_legacy_archived_tps", "stats_run_id",
    ]
    result = points[keep].merge(computed, on="point_id", validate="one_to_one")
    for version in VERSIONS:
        tps_column = f"{version}_recomputed_tps"
        result[f"{version}_abs_error_tps"] = (
            result[tps_column] - result["gpu_vllm_tps"]
        ).abs()
        result[f"{version}_abs_pct_error"] = (
            result[f"{version}_abs_error_tps"] / result["gpu_vllm_tps"] * 100.0
        )
    result["legacy_recompute_delta_pct"] = 100.0 * (
        result["paper_legacy_recomputed_tps"]
        / result["paper_legacy_archived_tps"] - 1.0
    )
    result["corrected_vs_legacy_pct"] = 100.0 * (
        result["current_corrected_recomputed_tps"]
        / result["paper_legacy_recomputed_tps"] - 1.0
    )
    return result.sort_values(["config_key", "bs"]).reset_index(drop=True)


def _wmape(frame: pd.DataFrame, model_column: str) -> float:
    return float(
        (frame[model_column] - frame["gpu_vllm_tps"]).abs().sum()
        / frame["gpu_vllm_tps"].sum() * 100.0
    )


def build_summary(model_points: pd.DataFrame, wall_time_s: float) -> pd.DataFrame:
    """Compute per-model, paper-claim, and all-point WMAPE definitions."""

    rows: list[dict[str, Any]] = []
    versions_and_columns = (
        ("paper_legacy_archived", "paper_legacy_archived_tps"),
        ("paper_legacy_recomputed", "paper_legacy_recomputed_tps"),
        ("current_corrected", "current_corrected_recomputed_tps"),
    )
    for version, column in versions_and_columns:
        per_model: list[float] = []
        for config in CONFIGS:
            group = model_points[model_points["config_key"] == config["key"]]
            wmape = _wmape(group, column)
            per_model.append(wmape)
            rows.append(
                {
                    "model_version": version,
                    "scope": str(config["key"]),
                    "metric_definition": "sum_abs_error_over_sum_gpu",
                    "n_points": len(group),
                    "gpu_tps_sum": group["gpu_vllm_tps"].sum(),
                    "abs_error_tps_sum": (group[column] - group["gpu_vllm_tps"]).abs().sum(),
                    "wmape_pct": wmape,
                    "wall_time_s": wall_time_s,
                    "gpu_runs_executed": 0,
                }
            )
        rows.append(
            {
                "model_version": version,
                "scope": "PAPER_CLAIM_EQUAL_MODEL_MEAN",
                "metric_definition": "arithmetic_mean_of_three_per_model_wmapes",
                "n_points": len(model_points),
                "gpu_tps_sum": model_points["gpu_vllm_tps"].sum(),
                "abs_error_tps_sum": math.nan,
                "wmape_pct": sum(per_model) / len(per_model),
                "wall_time_s": wall_time_s,
                "gpu_runs_executed": 0,
            }
        )
        rows.append(
            {
                "model_version": version,
                "scope": "ALL_POINTS_AGGREGATE",
                "metric_definition": "sum_abs_error_over_sum_gpu_all_28_points",
                "n_points": len(model_points),
                "gpu_tps_sum": model_points["gpu_vllm_tps"].sum(),
                "abs_error_tps_sum": (
                    model_points[column] - model_points["gpu_vllm_tps"]
                ).abs().sum(),
                "wmape_pct": _wmape(model_points, column),
                "wall_time_s": wall_time_s,
                "gpu_runs_executed": 0,
            }
        )
    return pd.DataFrame(rows)


def verify_results(
    model_points: pd.DataFrame,
    summary: pd.DataFrame,
    hashes: dict[str, str],
    wall_time_s: float,
) -> dict[str, Any]:
    """Check provenance, exact legacy replay, and the submitted metric."""

    legacy_rel = (
        model_points["paper_legacy_recomputed_tps"]
        / model_points["paper_legacy_archived_tps"] - 1.0
    ).abs()
    max_legacy_rel = float(legacy_rel.max())
    archived_claim_row = summary[
        (summary["model_version"] == "paper_legacy_archived")
        & (summary["scope"] == "PAPER_CLAIM_EQUAL_MODEL_MEAN")
    ]
    archived_aggregate_row = summary[
        (summary["model_version"] == "paper_legacy_archived")
        & (summary["scope"] == "ALL_POINTS_AGGREGATE")
    ]
    replay_claim_row = summary[
        (summary["model_version"] == "paper_legacy_recomputed")
        & (summary["scope"] == "PAPER_CLAIM_EQUAL_MODEL_MEAN")
    ]
    replay_aggregate_row = summary[
        (summary["model_version"] == "paper_legacy_recomputed")
        & (summary["scope"] == "ALL_POINTS_AGGREGATE")
    ]
    archived_claim = float(archived_claim_row.iloc[0]["wmape_pct"])
    archived_aggregate = float(archived_aggregate_row.iloc[0]["wmape_pct"])
    replay_claim = float(replay_claim_row.iloc[0]["wmape_pct"])
    replay_aggregate = float(replay_aggregate_row.iloc[0]["wmape_pct"])
    checks = {
        "input_hashes_match": hashes == EXPECTED_SHA256,
        "x4_points": len(model_points) == EXPECTED_POINTS,
        "legacy_recompute_matches_archived_within_tolerance": (
            max_legacy_rel <= LEGACY_REPLAY_REL_TOL
        ),
        "archived_paper_claim_metric_matches": math.isclose(
            archived_claim, EXPECTED_PAPER_CLAIM_WMAPE_PCT,
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "archived_aggregate_metric_matches": math.isclose(
            archived_aggregate, EXPECTED_PAPER_AGGREGATE_WMAPE_PCT,
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "legacy_replay_claim_within_tolerance": math.isclose(
            replay_claim, EXPECTED_PAPER_CLAIM_WMAPE_PCT,
            rel_tol=0.0, abs_tol=LEGACY_METRIC_ABS_TOL_PCT,
        ),
        "legacy_replay_aggregate_within_tolerance": math.isclose(
            replay_aggregate, EXPECTED_PAPER_AGGREGATE_WMAPE_PCT,
            rel_tol=0.0, abs_tol=LEGACY_METRIC_ABS_TOL_PCT,
        ),
        "all_outputs_finite": bool(
            model_points[
                ["paper_legacy_recomputed_tps", "current_corrected_recomputed_tps"]
            ].map(math.isfinite).all().all()
        ),
        "gpu_runs_zero": bool((summary["gpu_runs_executed"] == 0).all()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "points": len(model_points),
        "cpu_model_evaluations": EXPECTED_POINTS * len(VERSIONS),
        "gpu_runs_executed": 0,
        "wall_time_s": wall_time_s,
        "paper_claim_wmape_pct": archived_claim,
        "paper_all_points_aggregate_wmape_pct": archived_aggregate,
        "legacy_replay_claim_wmape_pct": replay_claim,
        "legacy_replay_all_points_aggregate_wmape_pct": replay_aggregate,
        "max_legacy_recompute_relative_error": max_legacy_rel,
        "legacy_replay_relative_tolerance": LEGACY_REPLAY_REL_TOL,
        "legacy_metric_absolute_tolerance_pct": LEGACY_METRIC_ABS_TOL_PCT,
        "input_sha256": hashes,
    }


def run(output_dir: Path, workers: int) -> dict[str, Any]:
    started = time.perf_counter()
    hashes = verify_input_hashes()
    points = load_fixed_points()
    computed = recompute(points, workers=workers)
    model_points = build_model_points(points, computed)
    wall_time_s = time.perf_counter() - started
    summary = build_summary(model_points, wall_time_s)
    verification = verify_results(model_points, summary, hashes, wall_time_s)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_points.to_csv(
        output_dir / "model_points.csv", index=False, float_format="%.15g"
    )
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.15g")
    with (output_dir / "verification.json").open("w", encoding="utf-8") as handle:
        json.dump(verification, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "PASS.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{verification['status']}\n")
        handle.write(f"4xMI325X model points: {len(model_points)}\n")
        handle.write(f"CPU model evaluations: {verification['cpu_model_evaluations']}\n")
        handle.write("GPU runs executed: 0\n")
        handle.write(f"Wall time: {wall_time_s:.3f} s\n")
        handle.write(f"Paper metric (legacy): {verification['paper_claim_wmape_pct']:.6f}%\n")

    if verification["status"] != "PASS":
        failed = [name for name, passed in verification["checks"].items() if not passed]
        raise AssertionError(f"Fig. 10 verification failed: {failed}")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 1),
        help="CPU worker processes (default: min(16, CPU count))",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="output directory (default: results/reproduce/stages/fig10)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    verification = run(output_dir, workers=max(1, args.workers))
    print(
        f"[PASS] Fig. 10: {verification['points']} x4 points, "
        f"paper-legacy WMAPE={verification['paper_claim_wmape_pct']:.3f}%, "
        f"GPU runs=0, wall={verification['wall_time_s']:.3f}s"
    )
    print(f"[ok] wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
