"""CPU-only fixed-point reproduction for paper Figure 12.

Figure 12 compares DeepStack's DeepSeek-V3.2 DSA decode model with a
recorded vLLM run on eight B200 GPUs.  The GPU measurements are immutable
inputs under ``data/fig12/reference``; this module performs no GPU work.  It
re-evaluates all 40 DeepStack points on the CPU with the exact deployment
configuration used by the paper.

Two model paths are deliberately kept separate:

``paper_legacy``
    Reproduces the pre-July-2026 activated-expert counting behavior used to
    generate the submitted figure.

``current_corrected``
    Uses the corrected activated-expert count.  It is a same-configuration
    version-drift diagnostic, not a new DSE or a replacement GPU experiment.

Run from any directory with::

    python -m ae.fig12
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paths import DATA_DIR, RESULTS_DIR, DEEPSTACK_SRC, activate_vendored_sources


INPUT_DIR = DATA_DIR / "fig12"
REFERENCE_DIR = INPUT_DIR / "reference"
MANIFEST = INPUT_DIR / "fixed_configs.csv"
MEASURED = REFERENCE_DIR / "vllm_measured_decode.csv"
PAPER_EXPECTED = INPUT_DIR / "paper_legacy_expected.csv"
CORRECTED_EXPECTED = INPUT_DIR / "current_corrected_expected.csv"
HASH_MANIFEST = INPUT_DIR / "SHA256SUMS"
OUTPUT_DIR = RESULTS_DIR / "reproduce" / "stages" / "fig12"

MODEL_VERSIONS = ("paper_legacy", "current_corrected")
EXPECTED_POINTS = 40
EXPECTED_CONTEXTS = (1024, 3072, 7168, 15360, 31744)
EXPECTED_GRID_COUNTS = {1024: 10, 3072: 9, 7168: 8, 15360: 7, 31744: 6}
PAPER_WMAPE = 2.6120978817229794
PAPER_ROUNDED_WMAPE = 2.6
PAPER_MAX_ABS_ERROR_PCT = 8.29025430626853
REGRESSION_TOLERANCE_PCT = 1.0e-9
METRIC_TOLERANCE = 1.0e-10


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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hashes() -> tuple[int, int]:
    checked = 0
    mismatches = 0
    for line in HASH_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        path = INPUT_DIR / relative.strip()
        checked += 1
        if not path.is_file() or _sha256(path) != expected:
            mismatches += 1
    return checked, mismatches


def _validate_inputs(
    manifest: Sequence[dict[str, str]], measured: Sequence[dict[str, str]]
) -> None:
    if len(manifest) != EXPECTED_POINTS:
        raise AssertionError(f"expected {EXPECTED_POINTS} configs, got {len(manifest)}")
    if len(measured) != EXPECTED_POINTS:
        raise AssertionError(
            f"expected {EXPECTED_POINTS} vLLM measurements, got {len(measured)}"
        )

    config_ids = [row["point_id"] for row in manifest]
    measured_ids = [row["point_id"] for row in measured]
    if len(set(config_ids)) != EXPECTED_POINTS or set(config_ids) != set(measured_ids):
        raise AssertionError("Fig. 12 config/measurement point IDs are not a 1:1 set")

    context_counts: dict[int, int] = {}
    for row in manifest:
        prompt = int(row["prompt_tokens"])
        context_counts[prompt] = context_counts.get(prompt, 0) + 1
        if int(row["max_new_tokens"]) != 1024:
            raise AssertionError("Fig. 12 requires max_new_tokens=1024")
        if int(row["kv_mid"]) != prompt + 512:
            raise AssertionError("representative KV must be prompt + max_new_tokens/2")
        if int(row["max_kv"]) != prompt + 1024:
            raise AssertionError("max_kv does not match prompt + max_new_tokens")
        if row["arch_factory"] != "B200" or row["noc_factory"] != "b200_8x1_8":
            raise AssertionError("unexpected Fig. 12 architecture/NoC")
        if row["model_class"] != "DeepSeekV3_2":
            raise AssertionError("unexpected Fig. 12 model class")
        if int(row["dsa_index_topk"]) != 2048:
            raise AssertionError("unexpected DSA index_topk")

    if tuple(sorted(context_counts)) != EXPECTED_CONTEXTS:
        raise AssertionError(f"unexpected context grid: {sorted(context_counts)}")
    if context_counts != EXPECTED_GRID_COUNTS:
        raise AssertionError(f"unexpected per-context grid sizes: {context_counts}")
    if any(row["status"] != "ok" for row in measured):
        raise AssertionError("one or more archived vLLM points is not status=ok")


def _relative_error_pct(value: float, expected: float) -> float:
    return abs(value / expected - 1.0) * 100.0


def _evaluate_all(
    manifest: Sequence[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate all fixed configurations for both historical model paths."""

    activate_vendored_sources()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from mosaic.dse_space.arch_noc_combinations import (
        b200_8x1_8_arch_noc_combinations,
    )
    from mosaic.llm.dsa.bench_dsv32_decode_stps import (
        footprint_per_gpu,
        full_model_decode_time,
    )
    from mosaic.llm_arch.deepseek_v3_2 import DeepSeekV3_2
    from mosaic.parallelism import ParallelScheme
    from mosaic.utils import Modeling_Granularity
    from mosaic.utils.allocate_ep import allocate_ep
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape

    first = manifest[0]
    arch, noc = b200_8x1_8_arch_noc_combinations()[0]
    model = DeepSeekV3_2()
    if model.dsa_arch.index_topk != int(first["dsa_index_topk"]):
        raise AssertionError("vendored model's DSA top-k differs from the manifest")
    granularity = Modeling_Granularity(
        mode=first["granularity_mode"],
        comp_comm_overlap=_as_bool(first["comp_comm_overlap"]),
        auto_tune=_as_bool(first["auto_tune"]),
        dump_perf_log=_as_bool(first["dump_perf_log"]),
    )
    trace_path = DEEPSTACK_SRC / first["trace_file"]
    routing = load_npz_routing_keep_shape(str(trace_path), as_list=False)[1]
    gib = 1024**3

    outputs: dict[str, list[dict[str, Any]]] = {
        version: [] for version in MODEL_VERSIONS
    }
    previous_mode = os.environ.get("DEEPSTACK_SWIGLU_COUNTS_MODE")
    try:
        for config in manifest:
            declared = ParallelScheme(
                tp=int(config["declared_moe_tp"]),
                ep=int(config["declared_moe_ep"]),
                ep1=int(config["declared_moe_ep1"]),
                ep2=int(config["declared_moe_ep2"]),
                sp=int(config["declared_moe_sp"]),
                cp=int(config["declared_moe_cp"]),
                dp=int(config["declared_moe_dp"]),
                pp=int(config["declared_moe_pp"]),
                fsdp=_as_bool(config["declared_moe_fsdp"]),
            )
            bs = int(config["bs"])
            seq = int(config["seq"])
            allocate_ep(parallel=declared, bs=bs, seq=seq)
            non_moe = dataclasses.replace(
                declared,
                ep=1,
                ep1=1,
                ep2=1,
                dp=declared.dp * declared.ep,
            )
            effective_moe = dataclasses.replace(
                declared,
                tp=1,
                ep=declared.ep * declared.tp,
                ep1=declared.ep * declared.tp,
                ep2=1,
            )
            if (
                declared.tp != int(config["declared_moe_tp"])
                or declared.ep != int(config["declared_moe_ep"])
                or declared.ep1 != int(config["declared_moe_ep1"])
                or declared.ep2 != int(config["declared_moe_ep2"])
                or non_moe.tp != int(config["non_moe_tp"])
                or non_moe.ep != int(config["non_moe_ep"])
                or non_moe.dp != int(config["non_moe_dp"])
                or non_moe.pp != int(config["non_moe_pp"])
                or effective_moe.tp != int(config["effective_moe_tp"])
                or effective_moe.ep != int(config["effective_moe_ep"])
                or effective_moe.ep1 != int(config["effective_moe_ep1"])
                or effective_moe.ep2 != int(config["effective_moe_ep2"])
            ):
                raise AssertionError(f"parallelism drift at {config['point_id']}")

            activation, weights, kv_cache = footprint_per_gpu(
                model,
                bs,
                int(config["max_kv"]),
                declared,
                non_moe,
            )
            memory_gib = (activation + weights + kv_cache) / gib
            fits = activation + weights + kv_cache <= arch.ddr_capacity

            for version in MODEL_VERSIONS:
                os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = version
                started = time.perf_counter()
                total_s, parts = full_model_decode_time(
                    model,
                    bs,
                    int(config["kv_mid"]),
                    declared,
                    non_moe,
                    arch,
                    noc,
                    granularity,
                    routing,
                )
                runtime_s = time.perf_counter() - started
                outputs[version].append(
                    {
                        "point_id": config["point_id"],
                        "model_version": version,
                        "scenario": config["scenario"],
                        "prompt_tokens": int(config["prompt_tokens"]),
                        "max_new_tokens": int(config["max_new_tokens"]),
                        "kv_mid": int(config["kv_mid"]),
                        "max_kv": int(config["max_kv"]),
                        "bs": bs,
                        "model_stps": bs / total_s,
                        "per_token_time_ms": total_s * 1e3,
                        "attention_time_ms": parts["attn"] * 1e3,
                        "dense_ffn_time_ms": parts["dense"] * 1e3,
                        "moe_time_ms": parts["moe"] * 1e3,
                        "rms_norm_time_ms": parts["rms"] * 1e3,
                        "residual_time_ms": parts["res"] * 1e3,
                        "memory_per_gpu_gib": memory_gib,
                        "fits_b200_192gib": fits,
                        "model_runtime_s": runtime_s,
                        "gpu_runs": 0,
                    }
                )
    finally:
        if previous_mode is None:
            os.environ.pop("DEEPSTACK_SWIGLU_COUNTS_MODE", None)
        else:
            os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = previous_mode
    return outputs


def _join_measurements(
    model_rows: Sequence[dict[str, Any]],
    measured: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    by_id = {row["point_id"]: row for row in measured}
    joined: list[dict[str, Any]] = []
    for model_row in model_rows:
        reference = by_id[model_row["point_id"]]
        vllm_stps = float(reference["vllm_stps"])
        model_stps = float(model_row["model_stps"])
        joined.append(
            {
                **model_row,
                "vllm_stps": vllm_stps,
                "error_pct": (model_stps / vllm_stps - 1.0) * 100.0,
                "absolute_error_pct": abs(model_stps / vllm_stps - 1.0) * 100.0,
                "measurement_status": reference["status"],
                "measurement_field": reference["measurement_field"],
                "measurement_source_run_id": reference["source_run_id"],
            }
        )
    return joined


def _wmape(rows: Iterable[dict[str, Any]]) -> float:
    rows = list(rows)
    numerator = sum(abs(float(row["model_stps"]) - float(row["vllm_stps"])) for row in rows)
    denominator = sum(float(row["vllm_stps"]) for row in rows)
    return numerator / denominator * 100.0


def _mape(rows: Iterable[dict[str, Any]]) -> float:
    rows = list(rows)
    return sum(float(row["absolute_error_pct"]) for row in rows) / len(rows)


def _expected_values(path: Path, value_column: str) -> dict[str, float]:
    return {row["point_id"]: float(row[value_column]) for row in _read_csv(path)}


def _summary_row(
    metric: str,
    scope: str,
    value: Any,
    expected: Any,
    tolerance: Any,
    status: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "scope": scope,
        "value": value,
        "expected": expected,
        "tolerance": tolerance,
        "status": status,
        "notes": notes,
    }


def reproduce(
    *,
    output_dir: Path = OUTPUT_DIR,
    skip_corrected_regression: bool = False,
) -> None:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    for marker in (output_dir / "PASS", output_dir / "FAIL"):
        marker.unlink(missing_ok=True)

    manifest = _read_csv(MANIFEST)
    measured = _read_csv(MEASURED)
    _validate_inputs(manifest, measured)
    hash_count, hash_mismatches = _check_hashes()
    evaluated = _evaluate_all(manifest)
    points = {
        version: _join_measurements(evaluated[version], measured)
        for version in MODEL_VERSIONS
    }
    for version in MODEL_VERSIONS:
        _write_csv(output_dir / version / "model_points.csv", points[version])

    paper_expected = _expected_values(PAPER_EXPECTED, "paper_legacy_stps")
    if len(paper_expected) != EXPECTED_POINTS:
        raise AssertionError("paper legacy baseline is incomplete")
    legacy_regression_max = max(
        _relative_error_pct(float(row["model_stps"]), paper_expected[row["point_id"]])
        for row in points["paper_legacy"]
    )

    corrected_regression_max = float("nan")
    if not skip_corrected_regression:
        corrected_expected = _expected_values(CORRECTED_EXPECTED, "current_corrected_stps")
        if len(corrected_expected) != EXPECTED_POINTS:
            raise AssertionError("current corrected baseline is incomplete")
        corrected_regression_max = max(
            _relative_error_pct(
                float(row["model_stps"]), corrected_expected[row["point_id"]]
            )
            for row in points["current_corrected"]
        )

    legacy_wmape = _wmape(points["paper_legacy"])
    corrected_wmape = _wmape(points["current_corrected"])
    legacy_max_abs = max(float(row["absolute_error_pct"]) for row in points["paper_legacy"])
    corrected_max_abs = max(
        float(row["absolute_error_pct"]) for row in points["current_corrected"]
    )
    all_fit = all(
        _as_bool(row["fits_b200_192gib"])
        for version in MODEL_VERSIONS
        for row in points[version]
    )

    legacy_by_id = {row["point_id"]: row for row in points["paper_legacy"]}
    drift: list[dict[str, Any]] = []
    for corrected in points["current_corrected"]:
        legacy = legacy_by_id[corrected["point_id"]]
        drift.append(
            {
                "point_id": corrected["point_id"],
                "scenario": corrected["scenario"],
                "prompt_tokens": corrected["prompt_tokens"],
                "bs": corrected["bs"],
                "paper_legacy_stps": legacy["model_stps"],
                "current_corrected_stps": corrected["model_stps"],
                "corrected_vs_legacy_pct": (
                    float(corrected["model_stps"]) / float(legacy["model_stps"]) - 1.0
                )
                * 100.0,
                "vllm_stps": corrected["vllm_stps"],
                "paper_legacy_error_pct": legacy["error_pct"],
                "current_corrected_error_pct": corrected["error_pct"],
            }
        )
    _write_csv(output_dir / "version_drift.csv", drift)

    max_drift = max(abs(float(row["corrected_vs_legacy_pct"])) for row in drift)
    small_drift = [
        abs(float(row["corrected_vs_legacy_pct"]))
        for row in drift
        if int(row["bs"]) <= 8
    ]
    large_drift = [
        abs(float(row["corrected_vs_legacy_pct"]))
        for row in drift
        if int(row["bs"]) >= 64
    ]

    checks: list[dict[str, Any]] = [
        _summary_row(
            "point_closure",
            "both_paths",
            sum(len(rows) for rows in points.values()),
            2 * EXPECTED_POINTS,
            0,
            "PASS" if all(len(rows) == EXPECTED_POINTS for rows in points.values()) else "FAIL",
            "40 fixed configs are evaluated independently on each model path",
        ),
        _summary_row(
            "input_sha256_mismatches",
            "all_inputs",
            hash_mismatches,
            0,
            0,
            "PASS" if hash_mismatches == 0 else "FAIL",
            f"checked {hash_count} files listed in data/fig12/SHA256SUMS",
        ),
        _summary_row(
            "gpu_runs",
            "fig12",
            0,
            0,
            0,
            "PASS",
            "archived vLLM CSV is read-only; all reruns are CPU model calls",
        ),
        _summary_row(
            "memory_fit_points",
            "both_paths",
            sum(
                _as_bool(row["fits_b200_192gib"])
                for version in MODEL_VERSIONS
                for row in points[version]
            ),
            2 * EXPECTED_POINTS,
            0,
            "PASS" if all_fit else "FAIL",
            "B200 capacity is 192 GiB per GPU",
        ),
        _summary_row(
            "paper_model_regression_max_error_pct",
            "paper_legacy",
            legacy_regression_max,
            0,
            REGRESSION_TOLERANCE_PCT,
            "PASS" if legacy_regression_max <= REGRESSION_TOLERANCE_PCT else "FAIL",
            "same fixed configs versus submitted model column",
        ),
        _summary_row(
            "wmape_pct",
            "paper_legacy",
            legacy_wmape,
            PAPER_WMAPE,
            METRIC_TOLERANCE,
            "PASS" if abs(legacy_wmape - PAPER_WMAPE) <= METRIC_TOLERANCE else "FAIL",
            "sum(abs(model-vLLM))/sum(vLLM)*100; rounds to 2.6%",
        ),
        _summary_row(
            "rounded_wmape_pct",
            "paper_legacy",
            round(legacy_wmape, 1),
            PAPER_ROUNDED_WMAPE,
            0,
            "PASS" if round(legacy_wmape, 1) == PAPER_ROUNDED_WMAPE else "FAIL",
            "paper prose precision",
        ),
        _summary_row(
            "mape_pct",
            "paper_legacy",
            _mape(points["paper_legacy"]),
            "",
            "",
            "INFO",
            "unweighted point MAPE; not the paper's 2.6% metric",
        ),
        _summary_row(
            "maximum_absolute_point_error_pct",
            "paper_legacy",
            legacy_max_abs,
            PAPER_MAX_ABS_ERROR_PCT,
            METRIC_TOLERANCE,
            "PASS"
            if abs(legacy_max_abs - PAPER_MAX_ABS_ERROR_PCT) <= METRIC_TOLERANCE
            and legacy_max_abs < 10.0
            else "FAIL",
            "all 40 submitted points remain within +/-10%",
        ),
        _summary_row(
            "wmape_pct",
            "current_corrected",
            corrected_wmape,
            "",
            "",
            "INFO",
            "fixed-config diagnostic after the activated-expert bug fix",
        ),
        _summary_row(
            "maximum_absolute_point_error_pct",
            "current_corrected",
            corrected_max_abs,
            "",
            "",
            "INFO",
            "diagnostic only; the archived GPU reference is unchanged",
        ),
        _summary_row(
            "maximum_version_drift_pct",
            "current_corrected_vs_paper_legacy",
            max_drift,
            "",
            "",
            "INFO",
            "same configs, model code path only",
        ),
        _summary_row(
            "mean_absolute_version_drift_pct",
            "bs_le_8",
            sum(small_drift) / len(small_drift),
            "",
            "",
            "INFO",
            "small-batch diagnostic",
        ),
        _summary_row(
            "mean_absolute_version_drift_pct",
            "bs_ge_64",
            sum(large_drift) / len(large_drift),
            "",
            "",
            "INFO",
            "large-batch diagnostic",
        ),
    ]
    if skip_corrected_regression:
        checks.append(
            _summary_row(
                "corrected_model_regression_max_error_pct",
                "current_corrected",
                "",
                "",
                "",
                "SKIP",
                "development-only baseline bootstrap requested",
            )
        )
    else:
        checks.append(
            _summary_row(
                "corrected_model_regression_max_error_pct",
                "current_corrected",
                corrected_regression_max,
                0,
                REGRESSION_TOLERANCE_PCT,
                "PASS"
                if corrected_regression_max <= REGRESSION_TOLERANCE_PCT
                else "FAIL",
                "same fixed configs versus the CPU-generated release baseline",
            )
        )
    checks.append(
        _summary_row(
            "wall_runtime_s",
            "fig12",
            time.perf_counter() - started,
            "",
            "",
            "INFO",
            "includes both 40-point CPU paths and verification",
        )
    )
    _write_csv(output_dir / "summary.csv", checks)

    failed = [row for row in checks if row["status"] == "FAIL"]
    if failed:
        names = ", ".join(f"{row['scope']}:{row['metric']}" for row in failed)
        (output_dir / "FAIL").write_text(f"FAIL\n{names}\n", encoding="utf-8")
        raise AssertionError(f"Figure 12 verification failed: {names}")

    (output_dir / "PASS").write_text(
        "PASS\n"
        f"points_per_path={EXPECTED_POINTS}\n"
        f"paper_legacy_wmape_pct={legacy_wmape:.12f}\n"
        f"current_corrected_wmape_pct={corrected_wmape:.12f}\n"
        "gpu_runs=0\n",
        encoding="utf-8",
    )
    print("Figure 12: PASS")
    print(f"  model points: {EXPECTED_POINTS} x {len(MODEL_VERSIONS)} paths")
    print(f"  paper_legacy wMAPE: {legacy_wmape:.12f}% (paper: 2.6%)")
    print(f"  current_corrected wMAPE: {corrected_wmape:.12f}%")
    print(f"  maximum version drift: {max_drift:.6f}%")
    print(f"  wall runtime: {time.perf_counter() - started:.3f}s")
    print("  GPU runs: 0")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for all generated Figure 12 outputs",
    )
    parser.add_argument(
        "--skip-corrected-regression",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    reproduce(
        output_dir=args.output_dir,
        skip_corrected_regression=args.skip_corrected_regression,
    )


if __name__ == "__main__":
    main()
