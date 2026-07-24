"""CPU-only fixed-configuration reproduction for paper Figure 22.

Figure 22 scales one level of the three-level NoC at a time for DeepSeek-V3
on 256 devices.  The expensive DSE is not repeated.  This runner re-evaluates
all 756 recorded winners (294 decode and 462 prefill points), including the
paper's comparison between charging NoC area to the logic die and placing that
area in the 3D DRAM stack while holding the SM count fixed.

The submitted model path and the corrected activated-expert path are explicit:
``paper_legacy`` verifies the archived figure, while ``current_corrected`` is
only a same-candidate-set diagnostic after the July 2026 bug fix.
"""

from __future__ import annotations

import os

# Pin numerical libraries before importing pandas/numpy in this process or in
# spawned workers.  Parallelism comes only from the requested worker count.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse
import dataclasses
import hashlib
import logging
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .paths import DATA_DIR, RESULTS_DIR, DEEPSTACK_SRC, activate_vendored_sources


INPUT_DIR = DATA_DIR / "fig22"
REFERENCE_DIR = INPUT_DIR / "reference"
MANIFEST = INPUT_DIR / "fixed_configs.csv"
PAPER_SUMMARIES = {
    "decode": REFERENCE_DIR / "paper_decode_summary.csv",
    "prefill": REFERENCE_DIR / "paper_prefill_summary.csv",
}
EXPECTED = {
    "paper_legacy": INPUT_DIR / "paper_legacy_expected.csv",
    "current_corrected": INPUT_DIR / "current_corrected_expected.csv",
}
HASH_MANIFEST = INPUT_DIR / "SHA256SUMS"
OUTPUT_ROOT = RESULTS_DIR / "reproduce" / "stages" / "fig22"

MODEL_VERSIONS = ("paper_legacy", "current_corrected")
PHASE_COUNTS = {"decode": 294, "prefill": 462}
EXPECTED_POINTS = sum(PHASE_COUNTS.values())
BASELINES = ("torus_mesh_mesh_3", "torus_mesh_switch_1")
LAYERS = ("L1", "L2", "L3")
BW_MULTIPLIERS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
DECODE_BATCHES = (1, 4, 16, 64, 256, 1024, 4096)
PREFILL_WORKLOADS = (
    (1, 1024),
    (1, 8192),
    (4, 1024),
    (4, 8192),
    (16, 1024),
    (16, 8192),
    (64, 1024),
    (64, 8192),
    (256, 1024),
    (256, 8192),
    (1024, 1024),
)
MAX_WORKERS = 64
TP_TRANSFORM_MOE = "replace_only"

# Floating-point-only regression when model/source inputs are unchanged.
EXPECTED_TOLERANCE_PCT = 1.0e-9
# A clean replay is bitwise-equivalent up to CSV floating-point formatting.
PAPER_POINT_TOLERANCE_PCT = 1.0e-9
PAPER_SUMMARY_TOLERANCE = 1.0e-12

# Full model frames retain power, frequency, raw-throughput, candidate-search,
# and runtime fields for regression and summary checks.  Only this minimal
# plot/closure projection is written to the paper-facing model-points CSV.
PUBLISHED_MODEL_POINT_COLUMNS = (
    "point_id",
    "model_version",
    "phase",
    "baseline_noc",
    "scaled_layer",
    "bw_multiplier",
    "bs",
    "seq",
    "reproduced_logic_stps",
    "reproduced_dram_stps",
)
PUBLISHED_VERSION_DRIFT_COLUMNS = (
    "point_id",
    "phase",
    "baseline_noc",
    "scaled_layer",
    "bw_multiplier",
    "bs",
    "seq",
    "logic_corrected_vs_legacy_pct",
    "dram_corrected_vs_legacy_pct",
)

_ROUTING_ARRAY: Any | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hashes() -> tuple[int, int]:
    if not HASH_MANIFEST.is_file():
        return 0, 0
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


def _validate_manifest(frame: pd.DataFrame) -> None:
    required = {
        "point_id",
        "phase",
        "model",
        "baseline_noc",
        "scaled_layer",
        "bw_multiplier",
        "bs",
        "minibatch",
        "seq",
        "orig_sm_count",
        "baseline_sm_count",
        "num_devices",
        "tp",
        "ep",
        "dp",
        "pp",
        "sp",
        "cp",
        "onex_tp",
        "onex_ep",
        "onex_dp",
        "onex_pp",
        "kv_len_1",
        "kv_len_2",
        "kv_len_3",
        "kv_len_4",
        "tp_transform_moe",
        "paper_logic_stps",
        "paper_dram_stps",
        "paper_dse_stps",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"Fig. 22 manifest is missing columns: {missing}")
    if len(frame) != EXPECTED_POINTS or frame.point_id.nunique() != EXPECTED_POINTS:
        raise AssertionError(
            f"expected {EXPECTED_POINTS} unique fixed points, got "
            f"{len(frame)}/{frame.point_id.nunique()}"
        )
    if frame.groupby("phase").size().to_dict() != PHASE_COUNTS:
        raise AssertionError("unexpected decode/prefill point counts")
    if set(frame.model) != {"DeepSeekV3"}:
        raise AssertionError("Fig. 22 must contain only DeepSeek-V3")
    if set(frame.baseline_noc) != set(BASELINES):
        raise AssertionError("unexpected Fig. 22 NoC baseline")
    if set(frame.scaled_layer) != set(LAYERS):
        raise AssertionError("unexpected Fig. 22 scaled layer")
    if set(frame.bw_multiplier) != set(BW_MULTIPLIERS):
        raise AssertionError("unexpected Fig. 22 bandwidth grid")
    if set(frame.tp_transform_moe) != {TP_TRANSFORM_MOE}:
        raise AssertionError("unexpected MoE TP transform mode")
    if set(frame.num_devices) != {256} or set(frame.baseline_sm_count) != {8}:
        raise AssertionError("Fig. 22 expects 256 devices and eight baseline SMs")
    if set(frame[frame.phase == "decode"].bs) != set(DECODE_BATCHES):
        raise AssertionError("unexpected decode batch grid")
    prefill_workloads = set(
        frame[frame.phase == "prefill"][["bs", "seq"]].itertuples(
            index=False, name=None
        )
    )
    if prefill_workloads != set(PREFILL_WORKLOADS):
        raise AssertionError("unexpected prefill workload grid")
    if not (
        frame.minibatch.astype(int)
        == frame.apply(lambda row: math.ceil(int(row.bs) / int(row.pp)), axis=1)
    ).all():
        raise AssertionError("manifest minibatch does not match ceil(BS/PP)")

    key_columns = ["baseline_noc", "scaled_layer", "bw_multiplier"]
    decode_sizes = (
        frame[frame.phase == "decode"].groupby(key_columns).size().unique().tolist()
    )
    prefill_sizes = (
        frame[frame.phase == "prefill"].groupby(key_columns).size().unique().tolist()
    )
    if decode_sizes != [len(DECODE_BATCHES)] or prefill_sizes != [len(PREFILL_WORKLOADS)]:
        raise AssertionError("Fig. 22 grid is not rectangular")


def _init_worker(trace_path: str) -> None:
    activate_vendored_sources()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"

    import torch
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape

    torch.set_num_threads(1)
    logging.getLogger("mosaic.parallelism.parallel").setLevel(logging.ERROR)
    global _ROUTING_ARRAY
    _, _ROUTING_ARRAY = load_npz_routing_keep_shape(trace_path, as_list=False)


def _seed_model() -> None:
    import torch

    np.random.seed(0)
    torch.manual_seed(0)


def _make_scheme(
    *, tp: int, ep: int, dp: int, pp: int, sp: int = 1, cp: int = 1
) -> Any:
    from mosaic.parallelism import ParallelScheme

    return ParallelScheme(
        tp=int(tp),
        ep=int(ep),
        ep1=1,
        ep2=int(ep),
        sp=int(sp),
        cp=int(cp),
        dp=int(dp),
        fsdp=False,
        pp=int(pp),
    )


def _make_arch(sm_count: int) -> Any:
    from mosaic.arch.stacked_gpu_wgmma import stacked_gpu_wgmma

    return stacked_gpu_wgmma().with_sm_count(int(sm_count))


def _decode_value(
    *,
    model: Any,
    noc: Any,
    granularity: Any,
    bs: int,
    sm_count: int,
    tp: int,
    ep: int,
    dp: int,
    pp: int,
    kv_lengths: Sequence[int],
) -> dict[str, Any]:
    from mosaic.dse_space.case_study_which_layer_noc_matter.noc_bw_config import (
        compute_power_wall,
    )
    from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
        modeling_decode,
    )
    from mosaic.utils.allocate_ep import allocate_ep

    _seed_model()
    scheme = _make_scheme(tp=tp, ep=ep, dp=dp, pp=pp)
    minibatch = math.ceil(int(bs) / scheme.pp)
    allocate_ep(parallel=scheme, bs=minibatch, seq=1)
    non_moe = dataclasses.replace(
        scheme, ep=1, ep1=1, ep2=1, dp=scheme.dp * scheme.ep
    )
    times, model_stats, _ = modeling_decode(
        model_arch=model,
        bs=minibatch,
        seq=1,
        cached_kv_list=list(kv_lengths),
        moe_parallel=scheme,
        non_moe_parallel=non_moe,
        single_chip=_make_arch(sm_count),
        noc_hierarchy=noc,
        granularity=granularity,
        routing_array=_ROUTING_ARRAY,
        tp_transform_moe=TP_TRANSFORM_MOE,
    )
    stps = sum(minibatch / values[-1] for values in times) / len(times)
    representative_time = times[len(times) // 2][-1]
    devices = scheme.world_size()
    chip_power = model_stats.chip_total_energy_j / representative_time / devices
    noc_power = model_stats.noc_total_energy_j / representative_time / devices
    hit_power_wall, frequency_scale = compute_power_wall(chip_power, noc_power)
    return {
        "stps": stps * frequency_scale,
        "raw_stps": stps,
        "chip_power_w": chip_power,
        "noc_power_w": noc_power,
        "hit_power_wall": bool(hit_power_wall),
        "frequency_scale": frequency_scale,
        "tp": scheme.tp,
        "ep": scheme.ep,
        "ep1": scheme.ep1,
        "ep2": scheme.ep2,
        "sp": scheme.sp,
        "cp": scheme.cp,
        "dp": scheme.dp,
        "pp": scheme.pp,
        "minibatch": minibatch,
    }


def _sp_cp_candidates(tp: int, ep: int, dp: int, pp: int, devices: int) -> list[tuple[int, int]]:
    product = int(tp) * int(ep) * int(dp) * int(pp)
    missing = max(1, round(int(devices) / product)) if product else 1
    return [(sp, missing // sp) for sp in range(1, missing + 1) if missing % sp == 0]


def _prefill_value(
    *,
    model: Any,
    noc: Any,
    granularity: Any,
    bs: int,
    seq: int,
    sm_count: int,
    tp: int,
    ep: int,
    dp: int,
    pp: int,
    sp: int,
    cp: int,
) -> dict[str, Any]:
    from mosaic.dse_space.case_study_which_layer_noc_matter.noc_bw_config import (
        compute_power_wall,
    )
    from mosaic.dse_space.dse_framework_multi_process_v4_prefill_dump_stats import (
        modeling_prefill,
    )
    from mosaic.utils.allocate_ep import allocate_ep

    _seed_model()
    scheme = _make_scheme(tp=tp, ep=ep, dp=dp, pp=pp, sp=sp, cp=cp)
    minibatch = math.ceil(int(bs) / scheme.pp)
    allocate_ep(parallel=scheme, bs=minibatch, seq=int(seq))
    attention = dataclasses.replace(
        scheme, ep=1, ep1=1, ep2=1, dp=scheme.dp * scheme.ep
    )
    moe = dataclasses.replace(scheme, cp=1, sp=scheme.cp * scheme.sp)
    non_attention_non_moe = dataclasses.replace(
        attention, cp=1, sp=scheme.cp * scheme.sp
    )
    values = modeling_prefill(
        model_arch=model,
        bs=minibatch,
        seq=int(seq),
        cached_kv=1,
        parallel=scheme,
        atten_parallel=attention,
        moe_parallel=moe,
        non_atten_non_moe_parallel=non_attention_non_moe,
        single_chip=_make_arch(sm_count),
        noc_hierarchy=noc,
        granularity=granularity,
        routing_array=_ROUTING_ARRAY,
        tp_transform_moe=TP_TRANSFORM_MOE,
    )
    total_time = values[7]
    model_stats = values[8]
    stps = minibatch * int(seq) / total_time
    devices = scheme.world_size()
    chip_power = model_stats.chip_total_energy_j / total_time / devices
    noc_power = model_stats.noc_total_energy_j / total_time / devices
    hit_power_wall, frequency_scale = compute_power_wall(chip_power, noc_power)
    return {
        "stps": stps * frequency_scale,
        "raw_stps": stps,
        "chip_power_w": chip_power,
        "noc_power_w": noc_power,
        "hit_power_wall": bool(hit_power_wall),
        "frequency_scale": frequency_scale,
        "tp": scheme.tp,
        "ep": scheme.ep,
        "ep1": scheme.ep1,
        "ep2": scheme.ep2,
        "sp": scheme.sp,
        "cp": scheme.cp,
        "dp": scheme.dp,
        "pp": scheme.pp,
        "minibatch": minibatch,
    }


def _best(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    # Candidate order is sorted before evaluation; Python's max retains the
    # first exact tie, making winner metadata independent of process schedule.
    return max(values, key=lambda value: float(value["stps"]))


def _worker(row: dict[str, Any]) -> dict[str, Any]:
    activate_vendored_sources()
    os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = str(row["_model_version"])

    from mosaic.dse_space.case_study_which_layer_noc_matter.noc_bw_config import (
        make_noc_for_config,
    )
    from mosaic.llm_arch import DeepSeekV3
    from mosaic.utils import Modeling_Granularity

    if _ROUTING_ARRAY is None:
        raise RuntimeError("routing trace was not initialized")
    model = DeepSeekV3()
    noc = make_noc_for_config(
        str(row["baseline_noc"]),
        str(row["scaled_layer"]),
        float(row["bw_multiplier"]),
    )
    granularity = Modeling_Granularity("coarse", True, False, True)
    common = {
        "model": model,
        "noc": noc,
        "granularity": granularity,
        "bs": int(row["bs"]),
    }
    started = time.perf_counter()

    if row["phase"] == "decode":
        kv_lengths = tuple(int(row[f"kv_len_{index}"]) for index in range(1, 5))
        logic = _decode_value(
            **common,
            sm_count=int(row["orig_sm_count"]),
            tp=int(row["tp"]),
            ep=int(row["ep"]),
            dp=int(row["dp"]),
            pp=int(row["pp"]),
            kv_lengths=kv_lengths,
        )
        candidate_tuples = sorted(
            {
                (int(row["tp"]), int(row["ep"]), int(row["dp"]), int(row["pp"])),
                (
                    int(row["onex_tp"]),
                    int(row["onex_ep"]),
                    int(row["onex_dp"]),
                    int(row["onex_pp"]),
                ),
            }
        )
        dram_values = [
            _decode_value(
                **common,
                sm_count=int(row["baseline_sm_count"]),
                tp=tp,
                ep=ep,
                dp=dp,
                pp=pp,
                kv_lengths=kv_lengths,
            )
            for tp, ep, dp, pp in candidate_tuples
        ]
    elif row["phase"] == "prefill":
        common["seq"] = int(row["seq"])
        logic = _prefill_value(
            **common,
            sm_count=int(row["orig_sm_count"]),
            tp=int(row["tp"]),
            ep=int(row["ep"]),
            dp=int(row["dp"]),
            pp=int(row["pp"]),
            sp=int(row["sp"]),
            cp=int(row["cp"]),
        )
        base_candidates = {
            (int(row["tp"]), int(row["ep"]), int(row["dp"]), int(row["pp"])),
            (
                int(row["onex_tp"]),
                int(row["onex_ep"]),
                int(row["onex_dp"]),
                int(row["onex_pp"]),
            ),
        }
        candidate_tuples = sorted(
            (tp, ep, dp, pp, sp, cp)
            for tp, ep, dp, pp in base_candidates
            for sp, cp in _sp_cp_candidates(
                tp, ep, dp, pp, int(row["num_devices"])
            )
        )
        dram_values = [
            _prefill_value(
                **common,
                sm_count=int(row["baseline_sm_count"]),
                tp=tp,
                ep=ep,
                dp=dp,
                pp=pp,
                sp=sp,
                cp=cp,
            )
            for tp, ep, dp, pp, sp, cp in candidate_tuples
        ]
    else:
        raise ValueError(f"unknown phase: {row['phase']}")

    dram = _best(dram_values)
    output: dict[str, Any] = {
        "point_id": str(row["point_id"]),
        "model_version": str(row["_model_version"]),
        "reproduced_logic_stps": logic["stps"],
        "reproduced_dram_stps": dram["stps"],
        "logic_raw_stps": logic["raw_stps"],
        "dram_raw_stps": dram["raw_stps"],
        "logic_chip_power_w": logic["chip_power_w"],
        "logic_noc_power_w": logic["noc_power_w"],
        "dram_chip_power_w": dram["chip_power_w"],
        "dram_noc_power_w": dram["noc_power_w"],
        "logic_hit_power_wall": logic["hit_power_wall"],
        "dram_hit_power_wall": dram["hit_power_wall"],
        "logic_frequency_scale": logic["frequency_scale"],
        "dram_frequency_scale": dram["frequency_scale"],
        "dram_tp": dram["tp"],
        "dram_ep": dram["ep"],
        "dram_ep1": dram["ep1"],
        "dram_ep2": dram["ep2"],
        "dram_sp": dram["sp"],
        "dram_cp": dram["cp"],
        "dram_dp": dram["dp"],
        "dram_pp": dram["pp"],
        "dram_minibatch": dram["minibatch"],
        "model_calls": 1 + len(dram_values),
        "model_runtime_s": time.perf_counter() - started,
        "gpu_runs": 0,
    }
    return output


def _normalized_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase in ("decode", "prefill"):
        phase_frame = frame[frame.phase == phase]
        workloads = ["bs"] if phase == "decode" else ["bs", "seq"]
        for baseline in BASELINES:
            for layer in LAYERS:
                subset = phase_frame[
                    (phase_frame.baseline_noc == baseline)
                    & (phase_frame.scaled_layer == layer)
                ]
                normalized: dict[str, list[pd.Series]] = {
                    "reproduced_logic_stps": [],
                    "reproduced_dram_stps": [],
                }
                for _, group in subset.groupby(workloads):
                    group = group.sort_values("bw_multiplier")
                    base = group[group.bw_multiplier == 1.0]
                    if len(base) != 1:
                        raise AssertionError("missing or duplicate 1x normalization point")
                    for metric in normalized:
                        normalized[metric].append(
                            pd.Series(
                                group[metric].to_numpy() / float(base.iloc[0][metric]),
                                index=group.bw_multiplier.to_numpy(),
                            )
                        )
                logic = pd.concat(normalized["reproduced_logic_stps"], axis=1).mean(axis=1)
                dram = pd.concat(normalized["reproduced_dram_stps"], axis=1).mean(axis=1)
                for bandwidth in BW_MULTIPLIERS:
                    rows.append(
                        {
                            "phase": phase,
                            "baseline_noc": baseline,
                            "scaled_layer": layer,
                            "bw_multiplier": bandwidth,
                            "logic_norm_stps": float(logic.loc[bandwidth]),
                            "dram_norm_stps": float(dram.loc[bandwidth]),
                        }
                    )
    return pd.DataFrame(rows)


def _published_model_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Project full Fig. 22 results onto plot and fixed-point closure fields."""

    return frame.loc[:, PUBLISHED_MODEL_POINT_COLUMNS].copy()


def _published_version_drift(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop per-version absolute STPS from the public drift diagnostic."""

    return frame.loc[:, PUBLISHED_VERSION_DRIFT_COLUMNS].copy()


def _paper_summary() -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(PAPER_SUMMARIES[phase]).assign(phase=phase)
            for phase in ("decode", "prefill")
        ],
        ignore_index=True,
    )[
        [
            "phase",
            "baseline_noc",
            "scaled_layer",
            "bw_multiplier",
            "logic_norm_stps",
            "dram_norm_stps",
        ]
    ]


def _summary_error(reproduced: pd.DataFrame) -> float:
    keys = ["phase", "baseline_noc", "scaled_layer", "bw_multiplier"]
    joined = reproduced.merge(
        _paper_summary(),
        on=keys,
        suffixes=("_reproduced", "_paper"),
        validate="one_to_one",
    )
    return float(
        max(
            (joined.logic_norm_stps_reproduced - joined.logic_norm_stps_paper)
            .abs()
            .max(),
            (joined.dram_norm_stps_reproduced - joined.dram_norm_stps_paper)
            .abs()
            .max(),
        )
    )


def _expected_error(frame: pd.DataFrame, version: str) -> float:
    expected = pd.read_csv(EXPECTED[version])
    joined = frame.merge(expected, on="point_id", validate="one_to_one")
    logic = (joined.reproduced_logic_stps / joined.expected_logic_stps - 1.0).abs()
    dram = (joined.reproduced_dram_stps / joined.expected_dram_stps - 1.0).abs()
    return float(max(logic.max(), dram.max()) * 100.0)


def _trend_values(summary: pd.DataFrame) -> dict[str, float]:
    return {
        "l2_half_bw_min_logic_norm": float(
            summary[
                (summary.scaled_layer == "L2") & (summary.bw_multiplier == 0.5)
            ].logic_norm_stps.min()
        ),
        "l2_half_bw_min_dram_norm": float(
            summary[
                (summary.scaled_layer == "L2") & (summary.bw_multiplier == 0.5)
            ].dram_norm_stps.min()
        ),
        "l3_1p5_bw_min_dram_norm": float(
            summary[
                (summary.scaled_layer == "L3") & (summary.bw_multiplier == 1.5)
            ].dram_norm_stps.min()
        ),
        "l3_4x_bw_max_dram_gain_pct": float(
            100.0
            * (
                summary[
                    (summary.scaled_layer == "L3")
                    & (summary.bw_multiplier == 4.0)
                ].dram_norm_stps.max()
                - 1.0
            )
        ),
        "l2_4x_bw_min_logic_norm": float(
            summary[
                (summary.scaled_layer == "L2") & (summary.bw_multiplier == 4.0)
            ].logic_norm_stps.min()
        ),
        "l2_4x_bw_min_dram_norm": float(
            summary[
                (summary.scaled_layer == "L2") & (summary.bw_multiplier == 4.0)
            ].dram_norm_stps.min()
        ),
    }


def _check_row(
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


def _write_checks(
    *,
    frame: pd.DataFrame,
    normalized: pd.DataFrame,
    version: str,
    wall_time: float,
    hash_count: int,
    hash_mismatches: int,
    skip_expected_regression: bool,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    paper_logic_error = float(frame.delta_from_paper_logic_pct.abs().max())
    paper_dram_error = float(frame.delta_from_paper_dram_pct.abs().max())
    paper_point_error = max(paper_logic_error, paper_dram_error)
    paper_summary_error = _summary_error(normalized)
    trends = _trend_values(normalized)

    if skip_expected_regression:
        expected_error = float("nan")
        expected_status = "SKIP"
    else:
        expected_error = _expected_error(frame, version)
        expected_status = (
            "PASS" if expected_error <= EXPECTED_TOLERANCE_PCT else "FAIL"
        )

    legacy = version == "paper_legacy"
    checks = [
        _check_row(
            "fixed_point_closure",
            version,
            len(frame),
            EXPECTED_POINTS,
            0,
            "PASS" if len(frame) == EXPECTED_POINTS else "FAIL",
            "294 decode plus 462 prefill DSE winners",
        ),
        _check_row(
            "input_sha256_mismatches",
            "fig22_inputs",
            hash_mismatches,
            0,
            0,
            "PASS" if hash_count > 0 and hash_mismatches == 0 else "FAIL",
            f"checked {hash_count} files from data/fig22/SHA256SUMS",
        ),
        _check_row(
            "gpu_runs",
            version,
            int(frame.gpu_runs.sum()),
            0,
            0,
            "PASS" if int(frame.gpu_runs.sum()) == 0 else "FAIL",
            "all DeepStack values are CPU analytical-model reruns",
        ),
        _check_row(
            "deterministic_expected_max_error_pct",
            version,
            expected_error,
            0,
            EXPECTED_TOLERANCE_PCT,
            expected_status,
            "model outputs only; runtime and process ordering are excluded",
        ),
        _check_row(
            "paper_fixed_point_max_error_pct",
            version,
            paper_point_error,
            0 if legacy else "",
            PAPER_POINT_TOLERANCE_PCT if legacy else "",
            (
                "PASS"
                if legacy and paper_point_error <= PAPER_POINT_TOLERANCE_PCT
                else ("FAIL" if legacy else "INFO")
            ),
            "maximum across logic-die and NoC-in-DRAM model values",
        ),
        _check_row(
            "paper_normalized_summary_max_abs_error",
            version,
            paper_summary_error,
            0 if legacy else "",
            PAPER_SUMMARY_TOLERANCE if legacy else "",
            (
                "PASS"
                if legacy and paper_summary_error <= PAPER_SUMMARY_TOLERANCE
                else ("FAIL" if legacy else "INFO")
            ),
            "84 normalized curve points across decode and prefill",
        ),
        _check_row(
            "l2_half_bw_min_logic_norm",
            version,
            trends["l2_half_bw_min_logic_norm"],
            ">=0.95" if legacy else "",
            "",
            (
                "PASS"
                if legacy and trends["l2_half_bw_min_logic_norm"] >= 0.95
                else ("FAIL" if legacy else "INFO")
            ),
            "paper trend: L2 can be reduced modestly without throughput loss",
        ),
        _check_row(
            "l2_half_bw_min_dram_norm",
            version,
            trends["l2_half_bw_min_dram_norm"],
            ">=0.95" if legacy else "",
            "",
            (
                "PASS"
                if legacy and trends["l2_half_bw_min_dram_norm"] >= 0.95
                else ("FAIL" if legacy else "INFO")
            ),
            "same L2 trend with NoC area moved into the DRAM stack",
        ),
        _check_row(
            "l3_1p5_bw_min_dram_norm",
            version,
            trends["l3_1p5_bw_min_dram_norm"],
            ">1" if legacy else "",
            "",
            (
                "PASS"
                if legacy and trends["l3_1p5_bw_min_dram_norm"] > 1.0
                else ("FAIL" if legacy else "INFO")
            ),
            "paper trend: every phase/topology benefits from a modest L3 increase when SM count is fixed",
        ),
        _check_row(
            "l2_4x_bw_min_logic_norm",
            version,
            trends["l2_4x_bw_min_logic_norm"],
            "<=0.55" if legacy else "",
            "",
            (
                "PASS"
                if legacy and trends["l2_4x_bw_min_logic_norm"] <= 0.55
                else ("FAIL" if legacy else "INFO")
            ),
            "higher NoC area reduces SM count on the logic-die path",
        ),
        _check_row(
            "l2_4x_bw_min_dram_norm",
            version,
            trends["l2_4x_bw_min_dram_norm"],
            ">=0.99" if legacy else "",
            "",
            (
                "PASS"
                if legacy and trends["l2_4x_bw_min_dram_norm"] >= 0.99
                else ("FAIL" if legacy else "INFO")
            ),
            "the high-BW collapse disappears with baseline SM count fixed",
        ),
        _check_row(
            "l3_4x_bw_max_dram_gain_pct",
            version,
            trends["l3_4x_bw_max_dram_gain_pct"],
            19.47924826276175 if legacy else "",
            PAPER_SUMMARY_TOLERANCE * 100.0 if legacy else "",
            (
                "PASS"
                if legacy
                and abs(trends["l3_4x_bw_max_dram_gain_pct"] - 19.47924826276175)
                <= PAPER_SUMMARY_TOLERANCE * 100.0
                else ("FAIL" if legacy else "INFO")
            ),
            "maximum archived L3 gain: prefill on torus-mesh-switch",
        ),
        _check_row(
            "power_wall_points",
            version,
            int(frame.logic_hit_power_wall.sum() + frame.dram_hit_power_wall.sum()),
            "",
            "",
            "INFO",
            "frequency scaling is already included in reproduced STPS",
        ),
        _check_row(
            "analytical_model_calls",
            version,
            int(frame.model_calls.sum()),
            "",
            "",
            "INFO",
            "fixed winners plus the paper's tiny 1x-scheme ambiguity set",
        ),
        _check_row(
            "wall_runtime_s",
            version,
            wall_time,
            "",
            "",
            "INFO",
            "includes CPU model calls and verification for this path",
        ),
    ]
    pd.DataFrame(checks).to_csv(
        output_dir / "summary.csv", index=False, float_format="%.15g"
    )
    return checks, trends


def run_version(
    candidates: pd.DataFrame,
    *,
    model_version: str,
    workers: int,
    hash_count: int,
    hash_mismatches: int,
    skip_expected_regression: bool,
    output_root: Path = OUTPUT_ROOT,
) -> tuple[pd.DataFrame, float, dict[str, float]]:
    rows = candidates.copy()
    rows["_model_version"] = model_version
    records = rows.to_dict(orient="records")
    trace_path = (
        DEEPSTACK_SRC
        / "mosaic"
        / "data"
        / "aime_ds_r1"
        / "moe_activations_batch0.npz"
    )
    started = time.perf_counter()
    computed: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(str(trace_path),),
    ) as executor:
        futures = {executor.submit(_worker, row): row["point_id"] for row in records}
        completed = 0
        for future in as_completed(futures):
            point_id = futures[future]
            try:
                computed.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Fig. 22 point {point_id} failed") from exc
            completed += 1
            if completed % 100 == 0 or completed == len(records):
                elapsed = time.perf_counter() - started
                print(
                    f"  {model_version}: {completed}/{len(records)} points, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    wall_time = time.perf_counter() - started
    frame = candidates.merge(pd.DataFrame(computed), on="point_id", validate="one_to_one")
    frame["delta_from_paper_logic_pct"] = (
        frame.reproduced_logic_stps / frame.paper_logic_stps - 1.0
    ) * 100.0
    frame["delta_from_paper_dram_pct"] = (
        frame.reproduced_dram_stps / frame.paper_dram_stps - 1.0
    ) * 100.0
    frame.sort_values(
        ["phase", "baseline_noc", "scaled_layer", "bw_multiplier", "bs", "seq"],
        inplace=True,
    )
    normalized = _normalized_summary(frame)
    normalized.insert(0, "model_version", model_version)

    output_dir = output_root / model_version
    output_dir.mkdir(parents=True, exist_ok=True)
    for marker in (output_dir / "PASS", output_dir / "FAIL"):
        marker.unlink(missing_ok=True)
    _published_model_points(frame).to_csv(
        output_dir / "model_points.csv", index=False, float_format="%.15g"
    )
    normalized.to_csv(
        output_dir / "normalized_summary.csv", index=False, float_format="%.15g"
    )
    checks, trends = _write_checks(
        frame=frame,
        normalized=normalized,
        version=model_version,
        wall_time=wall_time,
        hash_count=hash_count,
        hash_mismatches=hash_mismatches,
        skip_expected_regression=skip_expected_regression,
        output_dir=output_dir,
    )
    failures = [row for row in checks if row["status"] == "FAIL"]
    if failures:
        names = ", ".join(str(row["metric"]) for row in failures)
        (output_dir / "FAIL").write_text(f"FAIL\n{names}\n", encoding="utf-8")
        raise AssertionError(f"Figure 22 {model_version} checks failed: {names}")

    maximum_paper_error = max(
        float(frame.delta_from_paper_logic_pct.abs().max()),
        float(frame.delta_from_paper_dram_pct.abs().max()),
    )
    (output_dir / "PASS").write_text(
        "PASS\n"
        f"model_version={model_version}\n"
        f"fixed_points={len(frame)}\n"
        f"model_calls={int(frame.model_calls.sum())}\n"
        f"max_error_vs_paper_pct={maximum_paper_error:.12g}\n"
        f"wall_runtime_s={wall_time:.6f}\n"
        "gpu_runs=0\n",
        encoding="utf-8",
    )
    print(
        f"Fig. 22 {model_version}: PASS; points={len(frame)}; "
        f"model_calls={int(frame.model_calls.sum())}; "
        f"max_paper_error_pct={maximum_paper_error:.6g}; "
        f"wall_time_s={wall_time:.3f}"
    )
    return frame, wall_time, trends


def _write_version_drift(
    frames: dict[str, pd.DataFrame], *, output_root: Path = OUTPUT_ROOT
) -> None:
    legacy = frames["paper_legacy"]
    corrected = frames["current_corrected"]
    columns = ["point_id", "phase", "baseline_noc", "scaled_layer", "bw_multiplier", "bs", "seq"]
    joined = legacy[columns + ["reproduced_logic_stps", "reproduced_dram_stps"]].merge(
        corrected[columns + ["reproduced_logic_stps", "reproduced_dram_stps"]],
        on=columns,
        suffixes=("_paper_legacy", "_current_corrected"),
        validate="one_to_one",
    )
    joined["logic_corrected_vs_legacy_pct"] = (
        joined.reproduced_logic_stps_current_corrected
        / joined.reproduced_logic_stps_paper_legacy
        - 1.0
    ) * 100.0
    joined["dram_corrected_vs_legacy_pct"] = (
        joined.reproduced_dram_stps_current_corrected
        / joined.reproduced_dram_stps_paper_legacy
        - 1.0
    ) * 100.0
    _published_version_drift(joined).to_csv(
        output_root / "version_drift.csv", index=False, float_format="%.15g"
    )

    drift_rows: list[dict[str, Any]] = []
    for phase in ("decode", "prefill", "all"):
        subset = joined if phase == "all" else joined[joined.phase == phase]
        for batch_scope, selected in (
            ("bs_le_16", subset[subset.bs <= 16]),
            ("bs_ge_256", subset[subset.bs >= 256]),
            ("all_bs", subset),
        ):
            values = np.concatenate(
                [
                    selected.logic_corrected_vs_legacy_pct.abs().to_numpy(),
                    selected.dram_corrected_vs_legacy_pct.abs().to_numpy(),
                ]
            )
            drift_rows.append(
                {
                    "phase": phase,
                    "batch_scope": batch_scope,
                    "model_values": len(values),
                    "mean_absolute_drift_pct": float(values.mean()),
                    "maximum_absolute_drift_pct": float(values.max()),
                }
            )
    pd.DataFrame(drift_rows).to_csv(
        output_root / "version_drift_summary.csv", index=False, float_format="%.15g"
    )


def reproduce(
    *,
    model_version: str,
    workers: int,
    output_root: Path = OUTPUT_ROOT,
    skip_expected_regression: bool = False,
) -> dict[str, pd.DataFrame]:
    activate_vendored_sources()
    candidates = pd.read_csv(MANIFEST)
    _validate_manifest(candidates)
    hash_count, hash_mismatches = _check_hashes()
    versions = MODEL_VERSIONS if model_version == "both" else (model_version,)
    frames: dict[str, pd.DataFrame] = {}
    for version in versions:
        frame, _, _ = run_version(
            candidates,
            model_version=version,
            workers=workers,
            hash_count=hash_count,
            hash_mismatches=hash_mismatches,
            skip_expected_regression=skip_expected_regression,
            output_root=output_root,
        )
        frames[version] = frame
    if set(frames) == set(MODEL_VERSIONS):
        _write_version_drift(frames, output_root=output_root)
        (output_root / "PASS").write_text(
            "PASS\nmodel_paths=paper_legacy,current_corrected\n"
            f"fixed_points_per_path={EXPECTED_POINTS}\n"
            "gpu_runs=0\n",
            encoding="utf-8",
        )
    return frames


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version",
        choices=(*MODEL_VERSIONS, "both"),
        default="both",
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="directory containing all generated Figure 22 outputs",
    )
    parser.add_argument(
        "--skip-expected-regression",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    available = max(1, os.cpu_count() or 1)
    workers = min(max(1, int(args.workers)), MAX_WORKERS, available)
    if workers != args.workers:
        print(f"Fig. 22 worker count adjusted from {args.workers} to {workers}")
    reproduce(
        model_version=args.model_version,
        workers=workers,
        output_root=args.output_dir,
        skip_expected_regression=args.skip_expected_regression,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
