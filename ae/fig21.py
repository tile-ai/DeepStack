"""Re-evaluate the fixed-configuration sweep used by paper Fig. 21.

The paper searched the parallel configurations separately.  The compact AE
manifest records the two fixed MoE mappings selected for every point on the
published NoC latency/bandwidth surface.  This module only re-evaluates those
recorded configurations; it does not perform a new DSE.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import DATA_DIR, RESULTS_DIR, DEEPSTACK_SRC, activate_vendored_sources


MODEL_VERSIONS = ("paper_legacy", "current_corrected")
EXPECTED_BATCH_SIZES = (4, 64, 1024)
EXPECTED_LATENCY_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
EXPECTED_BW_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
EXPECTED_TP_TRANSFORM_MODES = ("none", "replace_only")
PAPER_BASELINE = "torus_mesh_switch_1"
LEGACY_MAX_ERROR_PCT = 1.0e-8

# Keep the full fixed-configuration result in memory for regression and claim
# evaluation.  The paper-facing CSV contains only the identities and values
# needed to redraw the surface and inspect fixed-point closure.
PUBLISHED_RAW_COLUMNS = (
    "candidate_id",
    "model_version",
    "bs",
    "latency_multiplier",
    "bw_multiplier",
    "tp_transform_moe",
    "reproduced_stps_avg",
    "reproduced_utps_avg",
    "delta_from_paper_stps_pct",
    "delta_from_paper_utps_pct",
)

_TRACE: Any | None = None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _load_trace() -> Any:
    global _TRACE
    if _TRACE is None:
        from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape

        path = (
            DEEPSTACK_SRC
            / "mosaic"
            / "data"
            / "aime_ds_r1"
            / "moe_activations_batch0.npz"
        )
        _, _TRACE = load_npz_routing_keep_shape(str(path), as_list=False)
    return _TRACE


def _validate_manifest(frame: pd.DataFrame) -> None:
    required = {
        "candidate_id",
        "baseline_noc",
        "latency_multiplier",
        "bw_multiplier",
        "bs",
        "minibatch",
        "seq",
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
        "kv_len_1",
        "kv_len_2",
        "kv_len_3",
        "kv_len_4",
        "paper_utps_avg",
        "paper_stps_avg",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fig. 21 manifest is missing columns: {missing}")
    if len(frame) != 180:
        raise ValueError(f"expected 180 Fig. 21 candidates, found {len(frame)}")
    if frame.candidate_id.duplicated().any():
        raise ValueError("Fig. 21 candidate IDs must be unique")
    if set(frame.baseline_noc) != {PAPER_BASELINE}:
        raise ValueError("Fig. 21 manifest must contain only the TMS1 baseline")

    expected_grid = {
        (bs, latency, bw, mode)
        for bs in EXPECTED_BATCH_SIZES
        for latency in EXPECTED_LATENCY_MULTIPLIERS
        for bw in EXPECTED_BW_MULTIPLIERS
        for mode in EXPECTED_TP_TRANSFORM_MODES
    }
    actual_grid = set(
        frame[
            ["bs", "latency_multiplier", "bw_multiplier", "tp_transform_moe"]
        ].itertuples(index=False, name=None)
    )
    if actual_grid != expected_grid:
        missing_grid = sorted(expected_grid - actual_grid)
        extra_grid = sorted(actual_grid - expected_grid)
        raise ValueError(
            f"Fig. 21 grid mismatch: missing={missing_grid}, extra={extra_grid}"
        )


def _worker(row: dict[str, Any]) -> dict[str, Any]:
    activate_vendored_sources()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # Explicit on every task so that a reused worker cannot inherit the other
    # branch from an earlier invocation.
    os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = str(row["_model_version"])

    import torch

    from mosaic.dse_space.case_study_latency_vs_bandwidth.lat_bw_config import (
        compute_power_wall,
        make_arch_for_noc_config,
        make_noc_for_config,
    )
    from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
        modeling_decode,
    )
    from mosaic.llm_arch import DeepSeekV3
    from mosaic.parallelism import ParallelScheme
    from mosaic.utils import Modeling_Granularity

    torch.set_num_threads(1)
    logging.getLogger("mosaic.parallelism.parallel").setLevel(logging.ERROR)

    scheme = ParallelScheme(
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
        scheme,
        ep=1,
        ep1=1,
        ep2=1,
        dp=scheme.dp * scheme.ep,
    )
    noc = make_noc_for_config(
        str(row["baseline_noc"]),
        float(row["latency_multiplier"]),
        float(row["bw_multiplier"]),
    )
    arch, sm_count = make_arch_for_noc_config(noc)
    kv_lengths = [int(row[f"kv_len_{index}"]) for index in range(1, 5)]
    transform_mode = str(row["tp_transform_moe"])
    started = time.perf_counter()
    times, model_stats, _ = modeling_decode(
        model_arch=DeepSeekV3(),
        bs=int(row["minibatch"]),
        seq=int(row["seq"]),
        cached_kv_list=kv_lengths,
        moe_parallel=scheme,
        non_moe_parallel=non_moe,
        single_chip=arch,
        noc_hierarchy=noc,
        granularity=Modeling_Granularity("coarse", True, False, True),
        routing_array=_load_trace(),
        tp_transform_moe=transform_mode,
    )

    raw_utps = sum(1.0 / values[-1] / scheme.pp for values in times) / len(times)
    raw_stps = (
        sum(int(row["minibatch"]) / values[-1] for values in times) / len(times)
    )
    middle_time = times[len(times) // 2][-1]
    devices = scheme.world_size()
    chip_power = model_stats.chip_total_energy_j / middle_time / devices
    noc_power = model_stats.noc_total_energy_j / middle_time / devices
    hit_power_wall, frequency_scale = compute_power_wall(chip_power, noc_power)

    return {
        "candidate_id": str(row["candidate_id"]),
        "model_version": str(row["_model_version"]),
        "sm_count": int(sm_count),
        "raw_utps_avg": raw_utps,
        "raw_stps_avg": raw_stps,
        "chip_power_per_device_w": chip_power,
        "noc_power_per_device_w": noc_power,
        "hit_power_wall": bool(hit_power_wall),
        "freq_scale_power": frequency_scale,
        "reproduced_utps_avg": raw_utps * frequency_scale,
        "reproduced_stps_avg": raw_stps * frequency_scale,
        "model_runtime_s": time.perf_counter() - started,
    }


def _best_surface(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Apply the paper plot's best-of-two-MoE-mappings reduction."""

    keys = ["bs", "latency_multiplier", "bw_multiplier"]
    return frame.loc[frame.groupby(keys)[metric].idxmax()].copy()


def _point(surface: pd.DataFrame, metric: str, bs: int, latency: float, bw: float) -> float:
    row = surface[
        (surface.bs == bs)
        & (surface.latency_multiplier == latency)
        & (surface.bw_multiplier == bw)
    ]
    if len(row) != 1:
        raise ValueError(
            f"expected one surface point for BS={bs}, latency={latency}, BW={bw}; "
            f"found {len(row)}"
        )
    return float(row.iloc[0][metric])


def _claim_values(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    surface = _best_surface(frame, metric)
    baseline_4 = _point(surface, metric, 4, 1.0, 1.0)
    baseline_1024 = _point(surface, metric, 1024, 1.0, 1.0)

    latency_1024 = [
        _point(surface, metric, 1024, latency, 1.0)
        for latency in EXPECTED_LATENCY_MULTIPLIERS
    ]

    # The paper's "up to 31.3%" uses the largest endpoint range across
    # bandwidths and normalizes it to the (BW=1, latency=1) BS=4 baseline.
    bs4_ranges = []
    for bw in EXPECTED_BW_MULTIPLIERS:
        low_latency = _point(surface, metric, 4, 0.25, bw)
        high_latency = _point(surface, metric, 4, 4.0, bw)
        bs4_ranges.append((low_latency - high_latency, bw, low_latency, high_latency))
    maximum_range, maximum_range_bw, low_endpoint, high_endpoint = max(bs4_ranges)

    return {
        "bs1024_latency_variation_pct_at_bw1":
            100.0 * (max(latency_1024) - min(latency_1024)) / baseline_1024,
        "bs4_max_latency_variation_pct": 100.0 * maximum_range / baseline_4,
        "bs4_max_latency_variation_bw": maximum_range_bw,
        "bs4_low_high_latency_endpoint_ratio": low_endpoint / high_endpoint,
        "bs4_bw075_lat1_gain_pct":
            100.0 * (_point(surface, metric, 4, 1.0, 0.75) / baseline_4 - 1.0),
        "bs4_bw075_lat025_gain_pct":
            100.0 * (_point(surface, metric, 4, 0.25, 0.75) / baseline_4 - 1.0),
        "bs1024_bw075_lat1_gain_pct":
            100.0
            * (_point(surface, metric, 1024, 1.0, 0.75) / baseline_1024 - 1.0),
        "bs1024_bw2_lat1_change_pct":
            100.0 * (_point(surface, metric, 1024, 1.0, 2.0) / baseline_1024 - 1.0),
    }


def _build_summary(result: pd.DataFrame) -> pd.DataFrame:
    paper = _claim_values(result, "paper_stps_avg")
    reproduced = _claim_values(result, "reproduced_stps_avg")
    definitions = {
        "bs1024_latency_variation_pct_at_bw1": (
            "(max_latency_STPS-min_latency_STPS)/(BW1,lat1 baseline), BS1024 and BW1"
        ),
        "bs4_max_latency_variation_pct": (
            "max_BW(STPS_lat0.25-STPS_lat4)/(BW1,lat1 baseline), BS4"
        ),
        "bs4_max_latency_variation_bw": "BW multiplier attaining the BS4 maximum latency range",
        "bs4_low_high_latency_endpoint_ratio": (
            "STPS_lat0.25/STPS_lat4 at the BW attaining the BS4 maximum latency range"
        ),
        "bs4_bw075_lat1_gain_pct": "STPS(BW0.75,lat1)/STPS(BW1,lat1)-1, BS4",
        "bs4_bw075_lat025_gain_pct": "STPS(BW0.75,lat0.25)/STPS(BW1,lat1)-1, BS4",
        "bs1024_bw075_lat1_gain_pct": "STPS(BW0.75,lat1)/STPS(BW1,lat1)-1, BS1024",
        "bs1024_bw2_lat1_change_pct": "STPS(BW2,lat1)/STPS(BW1,lat1)-1, BS1024",
    }
    paper_reported = {
        "bs1024_latency_variation_pct_at_bw1": 3.6,
        "bs4_max_latency_variation_pct": 31.3,
        "bs4_max_latency_variation_bw": 0.75,
        "bs4_low_high_latency_endpoint_ratio": 1.36,
        "bs4_bw075_lat1_gain_pct": 8.8,
        "bs4_bw075_lat025_gain_pct": 17.3,
        "bs1024_bw075_lat1_gain_pct": 2.4,
        "bs1024_bw2_lat1_change_pct": -27.2,
    }
    rows = []
    for name, paper_value in paper.items():
        reproduced_value = reproduced[name]
        rows.append(
            {
                "metric": name,
                "definition": definitions[name],
                "paper_reported": paper_reported[name],
                "paper_recomputed": paper_value,
                "reproduced": reproduced_value,
                "delta_from_paper_recomputed_pct": (
                    100.0 * (reproduced_value / paper_value - 1.0)
                    if paper_value != 0
                    else reproduced_value - paper_value
                ),
            }
        )
    return pd.DataFrame(rows)


def _published_raw(result: pd.DataFrame) -> pd.DataFrame:
    """Project full Fig. 21 results onto the paper-facing surface fields."""

    return result.loc[:, PUBLISHED_RAW_COLUMNS].copy()


def run(
    candidate_csv: Path,
    output_root: Path,
    *,
    model_version: str,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"unknown model version: {model_version}")
    candidates = pd.read_csv(candidate_csv)
    candidates.tp_transform_moe = candidates.tp_transform_moe.fillna("none")
    _validate_manifest(candidates)
    candidates["_model_version"] = model_version
    rows = candidates.to_dict(orient="records")

    started = time.perf_counter()
    computed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_worker, row): row["candidate_id"] for row in rows}
        for future in as_completed(futures):
            candidate_id = futures[future]
            try:
                computed.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Fig. 21 candidate {candidate_id} failed") from exc
    wall_time = time.perf_counter() - started

    result = candidates.drop(columns="_model_version").merge(
        pd.DataFrame(computed), on="candidate_id", validate="one_to_one"
    )
    result["delta_from_paper_stps_pct"] = 100.0 * (
        result.reproduced_stps_avg / result.paper_stps_avg - 1.0
    )
    result["delta_from_paper_utps_pct"] = 100.0 * (
        result.reproduced_utps_avg / result.paper_utps_avg - 1.0
    )
    result.sort_values(
        ["bs", "latency_multiplier", "bw_multiplier", "tp_transform_moe"],
        inplace=True,
    )
    summary = _build_summary(result)

    output_dir = output_root / model_version
    output_dir.mkdir(parents=True, exist_ok=True)
    _published_raw(result).to_csv(
        output_dir / "raw.csv", index=False, float_format="%.15g"
    )
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.15g")

    if result.hit_power_wall.any():
        raise AssertionError("Fig. 21 unexpectedly hit the 100 W power wall")
    if model_version == "paper_legacy":
        maximum_error = max(
            result.delta_from_paper_stps_pct.abs().max(),
            result.delta_from_paper_utps_pct.abs().max(),
        )
        if maximum_error > LEGACY_MAX_ERROR_PCT:
            raise AssertionError(
                f"paper_legacy maximum error {maximum_error:.12g}% exceeds "
                f"{LEGACY_MAX_ERROR_PCT:.12g}%"
            )
    return result, summary, wall_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=DATA_DIR / "fig21" / "candidates_march2026.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_DIR / "reproduce" / "stages" / "fig21",
    )
    parser.add_argument("--model-version", choices=MODEL_VERSIONS, required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()

    result, summary, wall_time = run(
        args.candidate_csv,
        args.output_root,
        model_version=args.model_version,
        workers=args.workers,
    )
    maximum_error = max(
        result.delta_from_paper_stps_pct.abs().max(),
        result.delta_from_paper_utps_pct.abs().max(),
    )
    print(f"Fig. 21 {args.model_version}: {len(result)} fixed configurations")
    print(f"wall time: {wall_time:.3f} s")
    print(f"maximum fixed-configuration error vs paper: {maximum_error:.12g}%")
    print(f"power-wall cases: {int(result.hit_power_wall.sum())}")
    print(summary[["metric", "reproduced"]].to_string(index=False))
    print(f"wrote {args.output_root / args.model_version / 'raw.csv'}")
    print(f"wrote {args.output_root / args.model_version / 'summary.csv'}")


if __name__ == "__main__":
    main()
