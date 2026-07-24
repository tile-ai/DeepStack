"""CPU-only reproduction of the model side of Figure 9.

The four vLLM CSVs under :mod:`data/fig09/reference` are recorded 8xB200
measurements.  They are checksum-verified but never rerun.  Every plotted
DeepStack point is instead recomputed from the fixed configuration manifest
with the vendored DeepStack and TileSight sources.

Two model paths are evaluated:

``paper_legacy``
    Retains the historical MoE activated-expert counting behavior.

``current_corrected``
    Uses the corrected activated-expert count (the default in the released
    model).  Dense models are identical between the two paths.

Run from any directory with::

    python -m ae.fig09 --workers 32
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import DATA_DIR, RESULTS_DIR, DEEPSTACK_SRC, activate_vendored_sources


INPUT_DIR = DATA_DIR / "fig09"
REFERENCE_DIR = INPUT_DIR / "reference"
MANIFEST = INPUT_DIR / "fixed_configs.csv"
CLAIM_AUDIT = INPUT_DIR / "paper_claim_12_18_audit.csv"
DUAL_PATH_EXPECTED = INPUT_DIR / "dual_path_expected.csv"
SOURCE_VERSION_AUDIT = INPUT_DIR / "legacy_source_version_audit.csv"
OUTPUT_DIR = RESULTS_DIR / "reproduce" / "stages" / "fig09"
MODEL_VERSIONS = ("paper_legacy", "current_corrected")
BS_ORDER = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

MODEL_ORDER = (
    "Qwen3_235b_a22b",
    "Llama3_70b",
    "Llama3_405b",
    "DeepSeekV3",
)

# Values obtained by executing the exact filtering/MAPE code in the final
# four-panel plotting script against its four named modeling CSVs.  These
# constants make changes in extraction semantics visible to the verifier.
PAPER_FOUR_PANEL_MAPE = {
    "Qwen3_235b_a22b": 7.553438776010900,
    "Llama3_70b": 10.775446478567979,
    "Llama3_405b": 20.815325152372700,
    "DeepSeekV3": 12.531692561441355,
}
PAPER_FOUR_PANEL_OVERALL_MAPE = 12.918975742098231
PAPER_THREE_PANEL_MACRO_MAPE = 12.177422248325889
PAPER_REPORTED_ROUNDED_MAPE = 12.18

_TRACE_CACHE: dict[str, Any] = {}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _trace_for_model(model_name: str) -> Any:
    """Load a routed-expert trace once in each worker process."""

    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape

    trace_name = "qwen" if model_name == "Qwen3_235b_a22b" else "deepseek"
    if trace_name not in _TRACE_CACHE:
        if trace_name == "qwen":
            path = (
                DEEPSTACK_SRC
                / "mosaic/data/aime_qwen_235b/qwen3_moe_activations_batch0.npz"
            )
        else:
            path = DEEPSTACK_SRC / "mosaic/data/aime_ds_r1/moe_activations_batch0.npz"
        _, _TRACE_CACHE[trace_name] = load_npz_routing_keep_shape(str(path))
    return _TRACE_CACHE[trace_name]


def _model_worker(row: dict[str, Any], model_version: str) -> dict[str, Any]:
    """Evaluate one fixed Figure 9 configuration on one model path."""

    activate_vendored_sources()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = model_version

    import importlib

    import torch

    from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
        modeling_decode,
    )
    from mosaic.parallelism import ParallelScheme
    from mosaic.utils import Modeling_Granularity

    torch.set_num_threads(1)
    logging.getLogger("mosaic.parallelism.parallel").setLevel(logging.ERROR)

    arch_module = importlib.import_module("mosaic.arch")
    llm_module = importlib.import_module("mosaic.llm_arch")
    noc_module = importlib.import_module("mosaic.noc.noc_config_set")

    parallel = ParallelScheme(
        tp=int(row["tp"]),
        ep=int(row["ep"]),
        ep1=int(row["ep1"]),
        ep2=int(row["ep2"]),
        sp=int(row["sp"]),
        cp=int(row["cp"]),
        dp=int(row["dp"]),
        fsdp=_as_bool(row["fsdp"]),
        pp=int(row["pp"]),
    )
    non_moe = dataclasses.replace(
        parallel,
        ep=1,
        ep1=1,
        ep2=1,
        dp=parallel.dp * parallel.ep,
    )
    kv_lengths = [int(row[f"kv_len_{index}"]) for index in range(1, 5)]
    model_name = str(row["model"])
    started = time.perf_counter()
    times, _, _ = modeling_decode(
        model_arch=getattr(llm_module, model_name)(),
        bs=int(row["minibatch"]),
        seq=int(row["seq"]),
        cached_kv_list=kv_lengths,
        moe_parallel=parallel,
        non_moe_parallel=non_moe,
        single_chip=getattr(arch_module, str(row["arch"]))(),
        noc_hierarchy=getattr(noc_module, str(row["noc"]))(),
        # dump_perf_log=True is required by the current stats-aware decode
        # entry point; it does not write a log in this direct-call path.
        granularity=Modeling_Granularity("coarse", True, False, True),
        routing_array=_trace_for_model(model_name),
        tp_transform_moe=str(row["tp_transform_moe"]),
    )
    stps = sum(int(row["minibatch"]) / values[-1] for values in times) / len(times)
    utps = sum(1.0 / values[-1] / parallel.pp for values in times) / len(times)
    return {
        "point_id": str(row["point_id"]),
        "model_version": model_version,
        "rerun_stps": stps,
        "rerun_utps": utps,
        "model_runtime_s": time.perf_counter() - started,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hashes() -> tuple[int, int]:
    checked = 0
    mismatches = 0
    for line in (INPUT_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip()
        checked += 1
        mismatches += _sha256(INPUT_DIR / relative) != expected
    return checked, mismatches


def _load_ground_truth(manifest: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for reference_file, selected in manifest.groupby("reference_file", sort=False):
        source = pd.read_csv(REFERENCE_DIR / reference_file)
        eager_disabled = source[
            source["Eager_Mode"].astype(str).str.lower().eq("false")
        ][["BS", "System Decode TPS"]].copy()
        eager_disabled.rename(columns={"System Decode TPS": "vllm_stps"}, inplace=True)
        eager_disabled["BS"] = eager_disabled["BS"].astype(int)
        points = selected[["point_id", "minibatch"]].rename(
            columns={"minibatch": "BS"}
        )
        merged = points.merge(eager_disabled, on="BS", how="inner", validate="one_to_one")
        if len(merged) != len(points):
            missing = sorted(set(points.BS) - set(merged.BS))
            raise ValueError(f"{reference_file}: missing eager-disabled BS values {missing}")
        frames.append(merged[["point_id", "vllm_stps"]])
    return pd.concat(frames, ignore_index=True)


def _run_model_points(manifest: pd.DataFrame, workers: int) -> pd.DataFrame:
    tasks = [
        (row, version)
        for row in manifest.to_dict(orient="records")
        for version in MODEL_VERSIONS
    ]
    computed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_model_worker, row, version): (row["point_id"], version)
            for row, version in tasks
        }
        for future in as_completed(futures):
            point_id, version = futures[future]
            try:
                computed.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Figure 9 point {point_id} ({version}) failed") from exc

    long = pd.DataFrame(computed)
    if len(long) != 2 * len(manifest):
        raise AssertionError("not all dual-path model points completed")
    values = long.pivot(index="point_id", columns="model_version", values="rerun_stps")
    values.rename(
        columns={version: f"{version}_stps" for version in MODEL_VERSIONS},
        inplace=True,
    )
    runtimes = long.pivot(
        index="point_id", columns="model_version", values="model_runtime_s"
    )
    runtimes.rename(
        columns={version: f"{version}_runtime_s" for version in MODEL_VERSIONS},
        inplace=True,
    )
    return values.join(runtimes).reset_index()


def _mape(frame: pd.DataFrame, prediction: str) -> float:
    return float(((frame[prediction] - frame.vllm_stps).abs() / frame.vllm_stps).mean() * 100)


def _wmape(frame: pd.DataFrame, prediction: str) -> float:
    return float((frame[prediction] - frame.vllm_stps).abs().sum() / frame.vllm_stps.sum() * 100)


def _claim(
    metric: str,
    value: float,
    unit: str,
    expected: float | None,
    tolerance: float | None,
    scope: str,
    evidence: str,
) -> dict[str, object]:
    if expected is None or tolerance is None:
        status = "INFO"
    else:
        status = "PASS" if abs(value - expected) <= tolerance else "FAIL"
    return {
        "metric": metric,
        "value": f"{value:.15g}",
        "unit": unit,
        "expected": "" if expected is None else f"{expected:.15g}",
        "tolerance": "" if tolerance is None else f"{tolerance:.15g}",
        "status": status,
        "scope": scope,
        "evidence": evidence,
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.15g")


def run(output_dir: Path = OUTPUT_DIR, workers: int = 32) -> int:
    wall_started = time.perf_counter()
    activate_vendored_sources()
    manifest = pd.read_csv(MANIFEST)
    if len(manifest) != 40:
        raise ValueError(f"expected 40 fixed Figure 9 points, found {len(manifest)}")
    expected_pairs = {(model, bs) for model in MODEL_ORDER for bs in BS_ORDER}
    actual_pairs = set(zip(manifest.model, manifest.minibatch.astype(int)))
    if actual_pairs != expected_pairs:
        raise ValueError("fixed manifest does not contain the expected 4 models x 10 BS")

    ground_truth = _load_ground_truth(manifest)
    rerun = _run_model_points(manifest, workers)
    points = manifest.merge(ground_truth, on="point_id", validate="one_to_one")
    points = points.merge(rerun, on="point_id", validate="one_to_one")
    expected = pd.read_csv(DUAL_PATH_EXPECTED).rename(
        columns={
            "paper_legacy_stps": "expected_paper_legacy_stps",
            "current_corrected_stps": "expected_current_corrected_stps",
        }
    )
    points = points.merge(expected, on="point_id", validate="one_to_one")

    for version in MODEL_VERSIONS:
        points[f"{version}_ape_pct"] = (
            (points[f"{version}_stps"] - points.vllm_stps).abs()
            / points.vllm_stps
            * 100
        )
    points["paper_csv_ape_pct"] = (
        (points.paper_stps_avg - points.vllm_stps).abs() / points.vllm_stps * 100
    )
    points["paper_legacy_vs_paper_csv_pct"] = (
        (points.paper_legacy_stps / points.paper_stps_avg - 1.0) * 100
    )
    points["current_corrected_vs_paper_csv_pct"] = (
        (points.current_corrected_stps / points.paper_stps_avg - 1.0) * 100
    )
    points["corrected_vs_legacy_pct"] = (
        (points.current_corrected_stps / points.paper_legacy_stps - 1.0) * 100
    )
    points["gpu_runs_performed"] = 0
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    points["_model_order"] = points.model.map(order)
    points.sort_values(["_model_order", "minibatch"], inplace=True)
    points.drop(columns="_model_order", inplace=True)
    _write_csv(output_dir / "model_points.csv", points)

    hashes_checked, hash_mismatches = _check_hashes()
    claims: list[dict[str, object]] = [
        _claim(
            "reference_checksum_mismatches",
            float(hash_mismatches),
            "files",
            0.0,
            0.0,
            f"{hashes_checked} self-contained Figure 9 inputs",
            "data/fig09/SHA256SUMS",
        ),
        _claim(
            "model_points_recomputed",
            float(len(points)),
            "points",
            40.0,
            0.0,
            "4 models x BS 1..512 intersection",
            "model_points.csv",
        ),
        _claim(
            "gpu_runs_performed",
            0.0,
            "runs",
            0.0,
            0.0,
            "recorded vLLM reference data only",
            "data/fig09/SOURCE_PROVENANCE.csv",
        ),
    ]

    for model in MODEL_ORDER:
        selected = points[points.model == model]
        paper_mape = _mape(selected, "paper_stps_avg")
        claims.extend(
            [
                _claim(
                    f"paper_csv_mape_{model}",
                    paper_mape,
                    "percent",
                    PAPER_FOUR_PANEL_MAPE[model],
                    1e-9,
                    "10 plotted batch sizes",
                    "fixed_configs.csv + recorded vLLM CSV",
                ),
                _claim(
                    f"paper_legacy_rerun_mape_{model}",
                    _mape(selected, "paper_legacy_stps"),
                    "percent",
                    None,
                    None,
                    "10 CPU-rerun model points",
                    "model_points.csv",
                ),
                _claim(
                    f"current_corrected_rerun_mape_{model}",
                    _mape(selected, "current_corrected_stps"),
                    "percent",
                    None,
                    None,
                    "10 CPU-rerun model points",
                    "model_points.csv",
                ),
            ]
        )

    final_four_mape = _mape(points, "paper_stps_avg")
    audit = pd.read_csv(CLAIM_AUDIT)
    historical_macro = float(audit.mape_pct.mean())
    rounded_historical = round(historical_macro, 2)
    max_legacy_delta = float(points.paper_legacy_vs_paper_csv_pct.abs().max())
    source_audit = pd.read_csv(SOURCE_VERSION_AUDIT)
    source_check = source_audit[
        ["point_id", "model", "version_matched_updated_stps"]
    ].merge(
        points[["point_id", "paper_legacy_stps"]],
        on="point_id",
        validate="one_to_one",
    )
    source_check["legacy_vs_updated_pct"] = (
        source_check.paper_legacy_stps
        / source_check.version_matched_updated_stps
        - 1.0
    ) * 100
    small_bs = points[points.minibatch <= 8]
    large_bs = points[points.minibatch >= 64]
    claims.extend(
        [
            _claim(
                "final_four_panel_paper_csv_mape",
                final_four_mape,
                "percent",
                PAPER_FOUR_PANEL_OVERALL_MAPE,
                1e-9,
                "40 equally weighted plotted points",
                "plot_modeling_ref_vllm_0313_2x2.py semantics",
            ),
            _claim(
                "reported_12_18_three_panel_macro_mape_unrounded",
                historical_macro,
                "percent",
                PAPER_THREE_PANEL_MACRO_MAPE,
                1e-9,
                "Qwen3 235B + legacy DeepSeek V3 + Llama3 70B",
                "data/fig09/paper_claim_12_18_audit.csv",
            ),
            _claim(
                "reported_12_18_three_panel_macro_mape_rounded",
                rounded_historical,
                "percent",
                PAPER_REPORTED_ROUNDED_MAPE,
                0.0,
                "macro-average rounded to two decimals",
                "plot_modeling_ref_vllm_0224_1x3.py comments",
            ),
            _claim(
                "paper_legacy_max_same_config_drift",
                max_legacy_delta,
                "percent",
                None,
                None,
                "all 40 points; final plot pins older Llama70/DeepSeek CSVs",
                "model_points.csv + data/fig09/legacy_source_version_audit.csv",
            ),
            _claim(
                "paper_legacy_version_matched_csv_points",
                float(len(source_check)),
                "points",
                20.0,
                0.0,
                "Llama70 0223 and DeepSeek 0313 updated model CSVs",
                "data/fig09/legacy_source_version_audit.csv",
            ),
            _claim(
                "paper_legacy_version_matched_csv_max_error",
                float(source_check.legacy_vs_updated_pct.abs().max()),
                "percent",
                0.0,
                1e-8,
                "20 points whose final plot selected an older CSV",
                "data/fig09/legacy_source_version_audit.csv",
            ),
            _claim(
                "paper_legacy_dual_path_regression_max_error",
                float(
                    (
                        points.paper_legacy_stps
                        / points.expected_paper_legacy_stps
                        - 1.0
                    ).abs().max()
                    * 100
                ),
                "percent",
                0.0,
                1e-8,
                "40 deterministic CPU model outputs",
                "data/fig09/dual_path_expected.csv",
            ),
            _claim(
                "current_corrected_dual_path_regression_max_error",
                float(
                    (
                        points.current_corrected_stps
                        / points.expected_current_corrected_stps
                        - 1.0
                    ).abs().max()
                    * 100
                ),
                "percent",
                0.0,
                1e-8,
                "40 deterministic CPU model outputs",
                "data/fig09/dual_path_expected.csv",
            ),
            _claim(
                "current_corrected_small_bs_mean_change_vs_legacy",
                float(small_bs.corrected_vs_legacy_pct.mean()),
                "percent",
                None,
                None,
                "BS <= 8, all four models",
                "model_points.csv",
            ),
            _claim(
                "current_corrected_large_bs_mean_change_vs_legacy",
                float(large_bs.corrected_vs_legacy_pct.mean()),
                "percent",
                None,
                None,
                "BS >= 64, all four models",
                "model_points.csv",
            ),
            _claim(
                "paper_csv_four_panel_wmape",
                _wmape(points, "paper_stps_avg"),
                "percent",
                None,
                None,
                "sum absolute error / sum recorded vLLM TPS",
                "model_points.csv",
            ),
            _claim(
                "paper_legacy_rerun_four_panel_mape",
                _mape(points, "paper_legacy_stps"),
                "percent",
                None,
                None,
                "40 CPU-rerun model points",
                "model_points.csv",
            ),
            _claim(
                "current_corrected_rerun_four_panel_mape",
                _mape(points, "current_corrected_stps"),
                "percent",
                None,
                None,
                "40 CPU-rerun model points",
                "model_points.csv",
            ),
        ]
    )

    runtime_s = time.perf_counter() - wall_started
    claims.append(
        _claim(
            "wall_runtime",
            runtime_s,
            "seconds",
            None,
            None,
            f"{workers} CPU workers; 80 analytical evaluations",
            "this run",
        )
    )
    summary = pd.DataFrame(claims)
    _write_csv(output_dir / "summary.csv", summary)

    failures = summary[summary.status == "FAIL"]
    report = {
        "status": "PASS" if failures.empty else "FAIL",
        "wall_runtime_s": runtime_s,
        "cpu_workers": workers,
        "model_evaluations": 2 * len(points),
        "model_points": len(points),
        "gpu_runs_performed": 0,
        "failed_checks": failures.metric.tolist(),
        "paper_claim_note": (
            "12.18% is the rounded macro-MAPE of the earlier three-panel "
            "validation; applying the final four-panel script to its four "
            "named paper model CSVs gives 12.9189757421%."
        ),
    }
    (output_dir / report["status"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    opposite = output_dir / ("FAIL" if report["status"] == "PASS" else "PASS")
    if opposite.exists():
        opposite.unlink()

    print(f"Figure 9: {report['status']}")
    print(f"  model points: {len(points)} (80 dual-path CPU evaluations)")
    print(f"  final four-panel paper-CSV MAPE: {final_four_mape:.6f}%")
    print(f"  historical reported metric: {historical_macro:.6f}% -> {rounded_historical:.2f}%")
    print(f"  paper_legacy rerun MAPE: {_mape(points, 'paper_legacy_stps'):.6f}%")
    print(f"  current_corrected rerun MAPE: {_mape(points, 'current_corrected_stps'):.6f}%")
    print(f"  GPU runs: 0; wall runtime: {runtime_s:.3f}s")
    return 0 if failures.empty else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="CPU worker processes (default: min(32, CPU count))",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="result directory (default: results/reproduce/stages/fig09)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(run(args.output_dir, args.workers))
