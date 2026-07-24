"""Selected-configuration verifier for the Table 4 ablation study.

The exhaustive architecture, DRAM, and NoC searches are intentionally not
part of the AE package.  Instead, this module re-evaluates the fourteen
published winners (seven steps at BS=4 and BS=1024) from their complete fixed
configurations.

Two model paths are exposed:

``paper_legacy``
    Preserves the historical MoE grouped-GEMM bin-counting behavior and
    reproduces the submitted Table 4 values.

``current_corrected``
    Uses the 2026-07-13 activated-expert bug fix.  These are same-configuration
    diagnostics, not a new DSE or a claim that the old winners remain optimal.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import DATA_DIR, RESULTS_DIR, activate_vendored_sources


MODEL_VERSIONS = ("paper_legacy", "current_corrected")
MODEL_VERSION_CHOICES = (*MODEL_VERSIONS, "both")
DEFAULT_MANIFEST = DATA_DIR / "table4" / "fixed_configs.csv"
DEFAULT_CURRENT_EXPECTED = DATA_DIR / "table4" / "current_corrected_expected.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "reproduce" / "stages" / "table4"
EXPECTED_CONFIGS = 14
EXPECTED_WORLD_SIZE = 256
PUBLISHED_MODEL_POINT_COLUMNS = (
    "candidate_id",
    "model_version",
    "step",
    "step_name",
    "bs",
    "recomputed_stps_avg",
    "delta_from_paper_stps_pct",
)

_ROUTING_ARRAY: Any | None = None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"expected a Boolean value, got {value!r}")
    return normalized == "true"


def _routing_array() -> Any:
    global _ROUTING_ARRAY
    if _ROUTING_ARRAY is None:
        from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
        from .paths import DEEPSTACK_SRC

        trace = (
            DEEPSTACK_SRC
            / "mosaic"
            / "data"
            / "aime_ds_r1"
            / "moe_activations_batch0.npz"
        )
        _, _ROUTING_ARRAY = load_npz_routing_keep_shape(str(trace), as_list=False)
    return _ROUTING_ARRAY


def _build_hardware(row: dict[str, Any]) -> tuple[Any, Any]:
    """Construct the exact architecture and NoC named by one manifest row."""

    step = int(row["step"])
    l1_mult = float(row["L1_mult"])
    l2_mult = float(row["L2_mult"])
    l3_mult = float(row["L3_mult"])

    if step == 7:
        from mosaic.dse_space.abaltion_study_e2e.noc_multi_layer_bw import (
            make_arch_for_noc,
            make_noc_multi,
        )

        noc = make_noc_multi(
            str(row["noc"]),
            {"L1": l1_mult, "L2": l2_mult, "L3": l3_mult},
        )
        arch, sm_count = make_arch_for_noc(
            noc,
            m=int(row["dram_total_layers"]),
            n=int(row["dram_active_layers"]),
            smem_cap=int(row["smem_capacity_KiB"]) * 1024,
            l1_throughput=int(row["l1_throughput_Bpc"]),
        )
    else:
        noc_module = importlib.import_module("mosaic.noc.noc_config_set")
        noc = getattr(noc_module, str(row["noc"]))()
        if step == 6:
            from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
                make_arch_for_config,
            )

            arch, _, _, _, sm_count = make_arch_for_config(
                int(row["dram_total_layers"]),
                int(row["dram_active_layers"]),
                int(row["smem_capacity_KiB"]) * 1024,
                int(row["l1_throughput_Bpc"]),
            )
        else:
            arch_module = importlib.import_module("mosaic.arch")
            arch = getattr(arch_module, str(row["arch"]))()
            sm_count = int(arch.sm_count)

    expected_hardware = {
        "dram_total_layers": int(row["dram_total_layers"]),
        "dram_active_layers": int(row["dram_active_layers"]),
        "sm_count": int(row["sm_count"]),
        "smem_capacity_KiB": int(row["smem_capacity_KiB"]),
    }
    actual_hardware = {
        "dram_total_layers": int(arch.dram_layers_per_cluster),
        "dram_active_layers": int(arch.dram_active_layers),
        "sm_count": int(sm_count),
        "smem_capacity_KiB": int(arch.configurable_smem_capacity) // 1024,
    }
    if actual_hardware != expected_hardware:
        raise AssertionError(
            f"{row['candidate_id']}: constructed hardware {actual_hardware} "
            f"does not match manifest {expected_hardware}"
        )
    return arch, noc


def _evaluate_one(row: dict[str, Any], model_version: str) -> dict[str, Any]:
    activate_vendored_sources()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = model_version

    import numpy as np
    import torch

    from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
        compute_power_wall,
    )
    from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
        modeling_decode,
    )
    from mosaic.llm_arch import DeepSeekV3
    from mosaic.parallelism import ParallelScheme
    from mosaic.utils import Modeling_Granularity

    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"unknown model version: {model_version!r}")

    # Reset the legacy cache estimator's RNG for scheduling-independent output.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    logging.getLogger("mosaic.parallelism.parallel").setLevel(logging.ERROR)

    arch, noc = _build_hardware(row)
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
    minibatch = int(row["minibatch"])

    started = time.perf_counter()
    times, model_stats, _ = modeling_decode(
        model_arch=DeepSeekV3(),
        bs=minibatch,
        seq=int(row["seq"]),
        cached_kv_list=kv_lengths,
        moe_parallel=parallel,
        non_moe_parallel=non_moe,
        single_chip=arch,
        noc_hierarchy=noc,
        granularity=Modeling_Granularity(
            mode="coarse",
            comp_comm_overlap=_as_bool(row["comp_comm_overlap"]),
            auto_tune=False,
            dump_perf_log=True,
        ),
        routing_array=_routing_array(),
        tp_transform_moe=str(row["tp_transform_moe"]),
    )
    runtime_s = time.perf_counter() - started

    raw_stps = sum(minibatch / point[-1] for point in times) / len(times)
    raw_utps = sum(1.0 / point[-1] / parallel.pp for point in times) / len(times)

    representative_time = times[len(times) // 2][-1]
    chip_power = model_stats.chip_total_energy_j / representative_time / noc.num_devices
    noc_power = model_stats.noc_total_energy_j / representative_time / noc.num_devices
    total_power = chip_power + noc_power
    hit_power_wall, power_freq_scale = compute_power_wall(chip_power, noc_power)

    # Steps 6/7 were selected by scaled STPS.  Earlier steps report raw STPS.
    apply_power_scale = int(row["step"]) >= 6 and hit_power_wall
    reported_stps = raw_stps * power_freq_scale if apply_power_scale else raw_stps
    reported_utps = raw_utps * power_freq_scale if apply_power_scale else raw_utps

    return {
        **row,
        "model_version": model_version,
        "raw_utps_avg": raw_utps,
        "raw_stps_avg": raw_stps,
        "recomputed_utps_avg": reported_utps,
        "recomputed_stps_avg": reported_stps,
        "delta_from_paper_stps_pct": 100.0
        * (reported_stps / float(row["paper_stps_avg"]) - 1.0),
        "chip_power_W": chip_power,
        "noc_power_per_device_W": noc_power,
        "total_power_per_device_W": total_power,
        "hit_power_wall": hit_power_wall,
        "freq_scale_power": power_freq_scale,
        "model_runtime_s": runtime_s,
    }


def _validate_manifest(frame: pd.DataFrame) -> None:
    required = {
        "candidate_id",
        "step",
        "bs",
        "minibatch",
        "comp_comm_overlap",
        "arch",
        "noc",
        "tp",
        "ep",
        "ep1",
        "ep2",
        "sp",
        "cp",
        "dp",
        "fsdp",
        "pp",
        "tp_transform_moe",
        "dram_total_layers",
        "dram_active_layers",
        "sm_count",
        "smem_capacity_KiB",
        "l1_throughput_Bpc",
        "L1_mult",
        "L2_mult",
        "L3_mult",
        "paper_stps_avg",
        "selection_source",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"fixed-config manifest is missing columns: {missing}")
    if len(frame) != EXPECTED_CONFIGS or frame.candidate_id.nunique() != EXPECTED_CONFIGS:
        raise AssertionError(
            f"expected {EXPECTED_CONFIGS} unique Table 4 configurations, got {len(frame)}"
        )
    expected_keys = {(step, bs) for step in range(1, 8) for bs in (4, 1024)}
    actual_keys = {(int(row.step), int(row.bs)) for row in frame.itertuples()}
    if actual_keys != expected_keys:
        raise AssertionError(f"manifest step/batch closure mismatch: {actual_keys}")

    for row in frame.to_dict(orient="records"):
        world_size = math.prod(
            int(row[field]) for field in ("tp", "ep", "sp", "cp", "dp", "pp")
        )
        if world_size != EXPECTED_WORLD_SIZE:
            raise AssertionError(
                f"{row['candidate_id']}: world size {world_size} != {EXPECTED_WORLD_SIZE}"
            )
        if int(row["minibatch"]) != math.ceil(int(row["bs"]) / int(row["pp"])):
            raise AssertionError(f"{row['candidate_id']}: minibatch/PP mismatch")
        expected_overlap = int(row["step"]) >= 5
        if _as_bool(row["comp_comm_overlap"]) != expected_overlap:
            raise AssertionError(f"{row['candidate_id']}: overlap does not match step semantics")
        if int(row["step"]) < 7 and any(
            not math.isclose(float(row[field]), 1.0)
            for field in ("L1_mult", "L2_mult", "L3_mult")
        ):
            raise AssertionError(f"{row['candidate_id']}: premature NoC scaling")


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError(
            f"{label}: recomputed STPS {actual:.15g} != expected {expected:.15g}"
        )


def _verify_results(results: pd.DataFrame, current_expected_path: Path) -> None:
    paper = results[results.model_version == "paper_legacy"]
    for row in paper.itertuples():
        _assert_close(
            float(row.recomputed_stps_avg),
            float(row.paper_stps_avg),
            f"paper_legacy/{row.candidate_id}",
        )

    current = results[results.model_version == "current_corrected"]
    if not current.empty:
        if not current_expected_path.is_file():
            raise FileNotFoundError(
                f"missing corrected-model reference: {current_expected_path}"
            )
        expected = pd.read_csv(current_expected_path)
        expected_by_id = dict(
            zip(expected.candidate_id, expected.expected_stps_avg, strict=True)
        )
        if set(current.candidate_id) != set(expected_by_id):
            raise AssertionError("current-corrected expected set does not match manifest")
        for row in current.itertuples():
            _assert_close(
                float(row.recomputed_stps_avg),
                float(expected_by_id[row.candidate_id]),
                f"current_corrected/{row.candidate_id}",
            )

    for version in results.model_version.unique():
        subset = results[results.model_version == version]
        for bs, paper_claim in ((4, 2.8), (1024, 9.5)):
            batch = subset[subset.bs == bs].set_index("step")
            speedup = float(batch.loc[7, "recomputed_stps_avg"]) / float(
                batch.loc[1, "recomputed_stps_avg"]
            )
            if version == "paper_legacy" and round(speedup, 1) != paper_claim:
                raise AssertionError(
                    f"BS={bs}: paper speedup rounds to {speedup:.1f}x, not {paper_claim:.1f}x"
                )


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for version in results.model_version.unique():
        subset = results[results.model_version == version]
        for bs, claim in ((4, 2.8), (1024, 9.5)):
            batch = subset[subset.bs == bs].set_index("step")
            step1 = float(batch.loc[1, "recomputed_stps_avg"])
            step7 = float(batch.loc[7, "recomputed_stps_avg"])
            rows.append(
                {
                    "model_version": version,
                    "bs": bs,
                    "step1_stps_avg": step1,
                    "step7_stps_avg": step7,
                    "speedup_x": step7 / step1,
                    "rounded_speedup_x": round(step7 / step1, 1),
                    "paper_claim_rounded_speedup_x": claim,
                    "interpretation": (
                        "submitted Table 4 reproduction"
                        if version == "paper_legacy"
                        else "same fixed configs after activated-expert bug fix; no re-search"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _published_model_points(results: pd.DataFrame) -> pd.DataFrame:
    """Project full verifier results onto the paper-facing Table 4 fields."""

    return results.loc[:, PUBLISHED_MODEL_POINT_COLUMNS].copy()


def run(
    manifest_path: Path,
    output_dir: Path,
    *,
    model_version: str,
    workers: int,
    current_expected_path: Path = DEFAULT_CURRENT_EXPECTED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    activate_vendored_sources()
    manifest = pd.read_csv(manifest_path)
    _validate_manifest(manifest)
    versions = MODEL_VERSIONS if model_version == "both" else (model_version,)
    tasks = [
        (row, version)
        for version in versions
        for row in manifest.to_dict(orient="records")
    ]

    started = time.perf_counter()
    evaluated: list[dict[str, Any]] = []
    worker_count = max(1, min(int(workers), len(tasks)))
    if worker_count == 1:
        evaluated = [_evaluate_one(row, version) for row, version in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_evaluate_one, row, version): (row["candidate_id"], version)
                for row, version in tasks
            }
            for future in as_completed(futures):
                candidate_id, version = futures[future]
                try:
                    evaluated.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"{version}/{candidate_id} failed") from exc

    elapsed = time.perf_counter() - started
    results = pd.DataFrame(evaluated).sort_values(["model_version", "step", "bs"])
    _verify_results(results, current_expected_path)
    summary = _summarize(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    _published_model_points(results).to_csv(
        output_dir / "model_points.csv", index=False, float_format="%.15g"
    )
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.15g")
    with (output_dir / "PASS").open("w", encoding="utf-8") as handle:
        handle.write("Table 4 selected-configuration verification: PASS\n")
        handle.write(f"model_version={model_version}\n")
        handle.write(f"fixed_configs={len(manifest)}\n")
        handle.write(f"evaluated_points={len(results)}\n")
        handle.write(f"workers={worker_count}\n")
        handle.write(f"wall_time_s={elapsed:.6f}\n")

    print(f"Table 4: PASS ({len(results)} model points, {elapsed:.3f} s)")
    print(f"wrote {output_dir / 'model_points.csv'}")
    print(f"wrote {output_dir / 'summary.csv'}")
    return results, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version",
        choices=MODEL_VERSION_CHOICES,
        default="both",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--current-expected",
        type=Path,
        default=DEFAULT_CURRENT_EXPECTED,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    run(
        args.manifest,
        args.output_dir,
        model_version=args.model_version,
        workers=args.workers,
        current_expected_path=args.current_expected,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
