"""CPU-only selected-configuration reproduction for paper Figs. 18--20.

The original DRAM-layer searches contain hundreds of thousands of rows.  The
artifact re-evaluates 114 complete configurations selected from those searches:
the endpoint throughput curves, every throughput/efficiency winner,
representative thermal extrema, and 22 deterministic terrain samples.  Compact
full-DSE projections used to redraw the paper figures are copied into the result
scope and explicitly labelled as archived rather than freshly searched.

``paper_legacy`` restores the submitted grouped-MoE bin-counting behavior.
``current_corrected`` evaluates the *same historical configurations* with the
activated-expert bug fix; it is a regression diagnostic, not a new DSE.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import DATA_DIR, RESULTS_DIR, DEEPSTACK_SRC, activate_vendored_sources


MODEL_VERSIONS = ("paper_legacy", "current_corrected")
MODEL_VERSION_CHOICES = (*MODEL_VERSIONS, "both")
DEFAULT_MANIFEST = DATA_DIR / "fig18_20" / "fixed_configs.csv"
DEFAULT_REGRESSION_DIGEST_DIR = DATA_DIR / "fig18_20"
DEFAULT_CHECKSUMS = DATA_DIR / "fig18_20" / "SHA256SUMS"
DEFAULT_ARCHIVE_DIR = DATA_DIR / "fig18_20" / "archive"
DEFAULT_OUTPUT = RESULTS_DIR / "reproduce" / "stages" / "fig18_20"
EXPECTED_CONFIGS = 114
EXPECTED_ROLE_COUNTS = {
    "fig18_curve_best": 44,
    "fig19_efficiency_winner": 14,
    "fig19_throughput_winner": 14,
    "fig20_archive_sample": 22,
    "fig20_hottest_per_layer": 22,
    "fig20_raw_throughput_best_per_layer": 22,
    "fig20_safe_throughput_best_per_layer": 22,
}
ARCHIVE_FILES = {
    "fig18_throughput_curves.csv": 110,
    "fig19_metric_grid.csv": 108,
    "fig20_decode_thermal_terrain.csv": 19233,
}
NUM_DEVICES = 256
POWER_CAP_W = 100.0
THERMAL_LIMIT_C = 85.0
# The historical DSE was produced by many long-lived workers whose cache-model
# RNG state was not serialized.  The selected-config rerun resets seed 0, so a
# small numerical tolerance is appropriate while the architectural selections
# and paper conclusions remain exact.
LEGACY_MAX_ERROR_PCT = 0.10
LEGACY_MAX_TEMPERATURE_ERROR_C = 0.01
REGRESSION_DIGEST_FORMAT = "deepstack-fig18-20-regression-v1"
REGRESSION_SIGNIFICANT_DIGITS = 10
REGRESSION_COLUMNS = (
    "candidate_id",
    "recomputed_raw_utps",
    "recomputed_raw_stps",
    "recomputed_scaled_utps",
    "recomputed_scaled_stps",
    "chip_power_W",
    "noc_power_per_device_W",
    "total_power_per_device_W",
    "hit_power_wall",
    "freq_scale_power",
    "effective_power_W",
    "effective_stps",
    "tokens_per_joule",
    "temperature_C",
)

# The full values above remain in memory for regression and claim evaluation.
# Result CSVs deliberately expose only configuration identities needed for
# reviewer closure.  Numeric model observables stay in memory for claims and
# the non-reversible regression digest.
PUBLISHED_MODEL_COLUMNS = (
    "candidate_id",
    "terrain_reference_id",
    "phase",
    "selection_roles",
    "model_version",
    "bs",
    "dram_total_layers",
    "dram_active_layers",
    "sm_count",
    "smem_capacity_KiB",
    "l1_throughput_Bpc",
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
    "model_runtime_s",
)

_ROUTING_ARRAY: Any | None = None


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"expected Boolean, got {value!r}")
    return normalized == "true"


def _routing_array() -> Any:
    global _ROUTING_ARRAY
    if _ROUTING_ARRAY is None:
        from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape

        trace = (
            DEEPSTACK_SRC
            / "mosaic"
            / "data"
            / "aime_ds_r1"
            / "moe_activations_batch0.npz"
        )
        _, _ROUTING_ARRAY = load_npz_routing_keep_shape(str(trace), as_list=False)
    return _ROUTING_ARRAY


def _roles(value: object) -> set[str]:
    return {item for item in str(value).split(";") if item}


def _validate_manifest(frame: pd.DataFrame) -> None:
    required = {
        "candidate_id",
        "terrain_reference_id",
        "phase",
        "selection_roles",
        "dram_total_layers",
        "dram_active_layers",
        "sm_count",
        "smem_capacity_KiB",
        "l1_throughput_Bpc",
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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Figs. 18--20 manifest is missing columns: {missing}")
    if len(frame) != EXPECTED_CONFIGS:
        raise ValueError(f"expected {EXPECTED_CONFIGS} configs, found {len(frame)}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate IDs must be unique")
    if "stats_run_id" in frame.columns:
        raise ValueError("manifest must not expose source-run IDs")
    if set(frame["phase"]) != {"decode", "prefill"}:
        raise ValueError("manifest must contain decode and prefill rows")
    if set(frame["model"]) != {"DeepSeekV3"}:
        raise ValueError("manifest must contain only DeepSeekV3")
    if set(frame["arch"]) != {"stacked_gpu_wgmma"}:
        raise ValueError("unexpected architecture in manifest")
    if set(frame["noc"]) != {"torus_mesh_switch_1"}:
        raise ValueError("unexpected NoC in manifest")

    role_counts = (
        frame["selection_roles"].str.split(";").explode().value_counts().to_dict()
    )
    if role_counts != EXPECTED_ROLE_COUNTS:
        raise ValueError(
            f"selection-role coverage mismatch: expected={EXPECTED_ROLE_COUNTS}, "
            f"actual={role_counts}"
        )
    sample_mask = frame["selection_roles"].map(
        lambda value: "fig20_archive_sample" in _roles(value)
    )
    reference_mask = frame["terrain_reference_id"].notna() & frame[
        "terrain_reference_id"
    ].astype(str).ne("")
    if not reference_mask.equals(sample_mask):
        raise ValueError(
            "terrain reference IDs must appear only on the 22 Fig. 20 samples"
        )
    if frame.loc[reference_mask, "terrain_reference_id"].duplicated().any():
        raise ValueError("terrain reference IDs must be unique")


def _verify_checksums(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        target = path.parent / filename.strip()
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            raise AssertionError(
                f"checksum mismatch for {target}: expected {digest}, got {actual}"
            )


def _build_hardware(row: dict[str, Any]) -> tuple[Any, Any, float]:
    from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
        make_arch_for_config,
        reference_thermal_resistance,
    )
    from mosaic.noc.noc_config_set import torus_mesh_switch_1

    arch, _, _, _, sm_count = make_arch_for_config(
        int(row["dram_total_layers"]),
        int(row["dram_active_layers"]),
        int(round(float(row["smem_capacity_KiB"]))) * 1024,
        int(row["l1_throughput_Bpc"]),
    )
    noc = torus_mesh_switch_1()
    actual = {
        "sm_count": int(sm_count),
        "dram_total_layers": int(arch.dram_layers_per_cluster),
        "dram_active_layers": int(arch.dram_active_layers),
        "smem_capacity_KiB": int(arch.configurable_smem_capacity) // 1024,
    }
    expected = {
        "sm_count": int(row["sm_count"]),
        "dram_total_layers": int(row["dram_total_layers"]),
        "dram_active_layers": int(row["dram_active_layers"]),
        "smem_capacity_KiB": int(round(float(row["smem_capacity_KiB"]))),
    }
    if actual != expected:
        raise AssertionError(
            f"{row['candidate_id']}: reconstructed hardware {actual} != {expected}"
        )
    thermal_resistance = reference_thermal_resistance(
        int(row["dram_total_layers"])
    )
    return arch, noc, thermal_resistance


def _evaluate_one(row: dict[str, Any], model_version: str) -> dict[str, Any]:
    activate_vendored_sources()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = model_version

    import torch

    from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
        compute_power_wall,
    )
    from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
        modeling_decode,
    )
    from mosaic.dse_space.dse_framework_multi_process_v4_prefill_dump_stats import (
        modeling_prefill,
    )
    from mosaic.llm_arch import DeepSeekV3
    from mosaic.parallelism import ParallelScheme
    from mosaic.utils import Modeling_Granularity

    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"unknown model version {model_version!r}")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    logging.getLogger("mosaic.parallelism.parallel").setLevel(logging.ERROR)

    arch, noc, thermal_resistance = _build_hardware(row)
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
    minibatch = int(row["minibatch"])
    granularity = Modeling_Granularity(
        mode="coarse",
        comp_comm_overlap=True,
        auto_tune=False,
        dump_perf_log=True,
    )

    started = time.perf_counter()
    if row["phase"] == "decode":
        non_moe = dataclasses.replace(
            parallel,
            ep=1,
            ep1=1,
            ep2=1,
            dp=parallel.dp * parallel.ep,
        )
        kv_lengths = [int(row[f"kv_len_{index}"]) for index in range(1, 5)]
        times, model_stats, _ = modeling_decode(
            model_arch=DeepSeekV3(),
            bs=minibatch,
            seq=int(row["seq"]),
            cached_kv_list=kv_lengths,
            moe_parallel=parallel,
            non_moe_parallel=non_moe,
            single_chip=arch,
            noc_hierarchy=noc,
            granularity=granularity,
            routing_array=_routing_array(),
            tp_transform_moe=str(row["tp_transform_moe"]),
        )
        raw_utps = sum(1.0 / values[-1] / parallel.pp for values in times) / len(
            times
        )
        raw_stps = sum(minibatch / values[-1] for values in times) / len(times)
        representative_time = times[len(times) // 2][-1]
    elif row["phase"] == "prefill":
        attention_parallel = dataclasses.replace(
            parallel,
            ep=1,
            ep1=1,
            ep2=1,
            dp=parallel.dp * parallel.ep,
        )
        moe_parallel = dataclasses.replace(
            parallel,
            cp=1,
            sp=parallel.cp * parallel.sp,
        )
        non_attention_non_moe = dataclasses.replace(
            attention_parallel,
            cp=1,
            sp=parallel.cp * parallel.sp,
        )
        values = modeling_prefill(
            model_arch=DeepSeekV3(),
            bs=minibatch,
            seq=int(row["seq"]),
            cached_kv=int(row["seq"]),
            parallel=parallel,
            atten_parallel=attention_parallel,
            moe_parallel=moe_parallel,
            non_atten_non_moe_parallel=non_attention_non_moe,
            single_chip=arch,
            noc_hierarchy=noc,
            granularity=granularity,
            routing_array=_routing_array(),
            tp_transform_moe=str(row["tp_transform_moe"]),
        )
        time_total = values[-2]
        model_stats = values[-1]
        raw_utps = int(row["seq"]) / (time_total * parallel.pp)
        raw_stps = minibatch * int(row["seq"]) / time_total
        representative_time = time_total
    else:  # pragma: no cover - validated before dispatch
        raise ValueError(f"unknown phase {row['phase']!r}")

    model_runtime_s = time.perf_counter() - started
    devices = parallel.world_size()
    if devices != NUM_DEVICES:
        raise AssertionError(
            f"{row['candidate_id']}: expected {NUM_DEVICES} devices, got {devices}"
        )
    chip_power = model_stats.chip_total_energy_j / representative_time / devices
    noc_power = model_stats.noc_total_energy_j / representative_time / devices
    total_power = chip_power + noc_power
    hit_power_wall, power_frequency_scale = compute_power_wall(chip_power, noc_power)
    scaled_utps = raw_utps * power_frequency_scale
    scaled_stps = raw_stps * power_frequency_scale
    effective_power = min(total_power, POWER_CAP_W)
    effective_stps = scaled_stps if hit_power_wall else raw_stps
    tokens_per_joule = effective_stps / (effective_power * devices)
    temperature = 35.0 + thermal_resistance * total_power

    return {
        "candidate_id": str(row["candidate_id"]),
        "terrain_reference_id": (
            ""
            if pd.isna(row.get("terrain_reference_id"))
            else str(row["terrain_reference_id"])
        ),
        "phase": str(row["phase"]),
        "selection_roles": str(row["selection_roles"]),
        "model_version": model_version,
        "bs": int(row["bs"]),
        "dram_total_layers": int(row["dram_total_layers"]),
        "dram_active_layers": int(row["dram_active_layers"]),
        "sm_count": int(row["sm_count"]),
        "smem_capacity_KiB": int(round(float(row["smem_capacity_KiB"]))),
        "l1_throughput_Bpc": int(row["l1_throughput_Bpc"]),
        "tp": parallel.tp,
        "ep": parallel.ep,
        "ep1": parallel.ep1,
        "ep2": parallel.ep2,
        "sp": parallel.sp,
        "cp": parallel.cp,
        "dp": parallel.dp,
        "fsdp": parallel.fsdp,
        "pp": parallel.pp,
        "tp_transform_moe": str(row["tp_transform_moe"]),
        "recomputed_raw_utps": raw_utps,
        "recomputed_raw_stps": raw_stps,
        "recomputed_scaled_utps": scaled_utps,
        "recomputed_scaled_stps": scaled_stps,
        "chip_power_W": chip_power,
        "noc_power_per_device_W": noc_power,
        "total_power_per_device_W": total_power,
        "hit_power_wall": bool(hit_power_wall),
        "freq_scale_power": power_frequency_scale,
        "effective_power_W": effective_power,
        "effective_stps": effective_stps,
        "tokens_per_joule": tokens_per_joule,
        "temperature_C": temperature,
        "model_runtime_s": model_runtime_s,
    }


def _evaluate_version(
    manifest: pd.DataFrame, model_version: str, workers: int
) -> pd.DataFrame:
    rows = manifest.to_dict(orient="records")
    outputs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_evaluate_one, row, model_version) for row in rows]
        for future in as_completed(futures):
            outputs.append(future.result())
    return pd.DataFrame(outputs).sort_values("candidate_id").reset_index(drop=True)


def _selected(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    return frame[frame["selection_roles"].map(lambda value: role in _roles(value))]


def _publish_archives(output_dir: Path) -> pd.DataFrame:
    """Copy checked, repository-local DSE projections into this result scope."""

    archive_output = output_dir / "archive"
    archive_output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for filename, expected_rows in ARCHIVE_FILES.items():
        source = DEFAULT_ARCHIVE_DIR / filename
        frame = pd.read_csv(source)
        if len(frame) != expected_rows:
            raise AssertionError(
                f"{source} has {len(frame)} rows; expected {expected_rows}"
            )
        target = archive_output / filename
        shutil.copyfile(source, target)
        records.append(
            {
                "dataset": filename.removesuffix(".csv"),
                "file": filename,
                "rows": len(frame),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "exhaustive_search_rerun": False,
                "status": "ARCHIVED_DSE_PROJECTION",
            }
        )
    manifest = pd.DataFrame(records)
    manifest.to_csv(archive_output / "archive_manifest.csv", index=False)
    return manifest


def _write_terrain_sample_validation(
    frame: pd.DataFrame, version_dir: Path
) -> pd.DataFrame:
    sample = _selected(frame, "fig20_archive_sample").copy()
    if len(sample) != EXPECTED_ROLE_COUNTS["fig20_archive_sample"]:
        raise AssertionError(f"expected 22 terrain samples, found {len(sample)}")
    archive = pd.read_csv(
        DEFAULT_ARCHIVE_DIR / "fig20_decode_thermal_terrain.csv"
    )[
        ["terrain_reference_id", "raw_stps", "temperature_C"]
    ].rename(
        columns={
            "raw_stps": "archived_raw_stps",
            "temperature_C": "archived_temperature_C",
        }
    )
    if archive["terrain_reference_id"].duplicated().any():
        raise AssertionError("Fig. 20 terrain reference IDs must be unique")
    sample = sample.merge(
        archive,
        on="terrain_reference_id",
        how="left",
        validate="one_to_one",
    )
    if sample[["archived_raw_stps", "archived_temperature_C"]].isna().any().any():
        raise AssertionError("Fig. 20 terrain sample is absent from the archive")
    sample["raw_stps_error_pct"] = 100.0 * (
        sample["recomputed_raw_stps"] / sample["archived_raw_stps"] - 1.0
    )
    sample["temperature_error_C"] = (
        sample["temperature_C"] - sample["archived_temperature_C"]
    )
    columns = [
        "candidate_id",
        "terrain_reference_id",
        "model_version",
        "bs",
        "dram_total_layers",
        "dram_active_layers",
        "archived_raw_stps",
        "recomputed_raw_stps",
        "raw_stps_error_pct",
        "archived_temperature_C",
        "temperature_C",
        "temperature_error_C",
    ]
    sample.rename(
        columns={"temperature_C": "recomputed_temperature_C"},
        inplace=True,
    )
    columns[columns.index("temperature_C")] = "recomputed_temperature_C"
    sample = sample[columns].sort_values(["bs", "dram_total_layers"])
    sample.to_csv(
        version_dir / "terrain_sample_validation.csv",
        index=False,
        float_format="%.15g",
    )
    return sample


def _add_claim(
    claims: list[dict[str, Any]],
    figure: str,
    metric: str,
    value: object,
    status: str,
    note: str,
) -> None:
    claims.append(
        {
            "figure": figure,
            "metric": metric,
            "value": value,
            "status": status,
            "note": note,
        }
    )


def _claim_summary(
    frame: pd.DataFrame,
    model_version: str,
    terrain_samples: pd.DataFrame,
) -> pd.DataFrame:
    claims: list[dict[str, Any]] = []
    is_legacy = model_version == "paper_legacy"
    claim_status = "PASS" if is_legacy else "INFO_FIXED_CONFIG_DIAGNOSTIC"

    curve = _selected(frame, "fig18_curve_best")
    peaks: dict[tuple[str, int], pd.Series] = {}
    for (phase, bs), group in curve.groupby(["phase", "bs"]):
        peak = group.loc[group["recomputed_scaled_stps"].idxmax()]
        peaks[(str(phase), int(bs))] = peak
        _add_claim(
            claims,
            "Fig18",
            f"{phase}_bs{int(bs)}_fixed_curve_peak_layer",
            int(peak["dram_total_layers"]),
            claim_status,
            "Best among the archived per-layer winners; no new DSE.",
        )
        _add_claim(
            claims,
            "Fig18",
            f"{phase}_bs{int(bs)}_fixed_curve_peak_stps",
            float(peak["recomputed_scaled_stps"]),
            claim_status,
            "Scaled system tokens/s.",
        )

    expected_peak_layers = {
        ("decode", 4): {9, 10},
        ("decode", 1024): {6, 7},
        ("prefill", 4): {7},
        ("prefill", 1024): {2},
    }
    if is_legacy:
        peak_ok = all(
            int(peaks[key]["dram_total_layers"]) in allowed
            for key, allowed in expected_peak_layers.items()
        )
        _add_claim(
            claims,
            "Fig18",
            "paper_peak_layer_pattern",
            peak_ok,
            "PASS" if peak_ok else "FAIL",
            "Small decode deep (~9--10), large decode 6--7, small prefill 7, large prefill 2.",
        )
    decode4 = curve[(curve["phase"] == "decode") & (curve["bs"] == 4)]
    layer9 = decode4[decode4["dram_total_layers"] == 9].iloc[0]
    peak4 = peaks[("decode", 4)]
    gap9 = 100.0 * (
        1.0
        - float(layer9["recomputed_scaled_stps"])
        / float(peak4["recomputed_scaled_stps"])
    )
    _add_claim(
        claims,
        "Fig18",
        "decode_bs4_layer9_gap_from_fixed_curve_peak_pct",
        gap9,
        claim_status,
        "Quantifies the paper's approximate '~9 layers' wording.",
    )

    throughput = _selected(frame, "fig19_throughput_winner")
    efficiency = _selected(frame, "fig19_efficiency_winner")
    comparisons: list[dict[str, Any]] = []
    for phase, bs in sorted(
        set(zip(throughput["phase"], throughput["bs"], strict=False))
    ):
        t = throughput[(throughput["phase"] == phase) & (throughput["bs"] == bs)]
        e = efficiency[(efficiency["phase"] == phase) & (efficiency["bs"] == bs)]
        if len(t) != 1 or len(e) != 1:
            raise AssertionError(f"missing Fig. 19 pair for {phase}, BS={bs}")
        tr = t.iloc[0]
        er = e.iloc[0]
        comparisons.append(
            {
                "phase": phase,
                "bs": int(bs),
                "tokens_j_gain_pct": 100.0
                * (float(er["tokens_per_joule"]) / float(tr["tokens_per_joule"]) - 1.0),
                "power_reduction_pct": 100.0
                * (
                    1.0
                    - float(er["total_power_per_device_W"])
                    / float(tr["total_power_per_device_W"])
                ),
                "throughput_idle_layers": int(tr["dram_total_layers"])
                - int(tr["dram_active_layers"]),
                "efficiency_idle_layers": int(er["dram_total_layers"])
                - int(er["dram_active_layers"]),
            }
        )
    pair_frame = pd.DataFrame(comparisons)
    fig19_values = {
        "tokens_j_gain_min_pct": pair_frame["tokens_j_gain_pct"].min(),
        "tokens_j_gain_max_pct": pair_frame["tokens_j_gain_pct"].max(),
        "power_reduction_min_pct": pair_frame["power_reduction_pct"].min(),
        "power_reduction_max_pct": pair_frame["power_reduction_pct"].max(),
        "scenarios_efficiency_has_at_least_as_many_idle_layers": int(
            (
                pair_frame["efficiency_idle_layers"]
                >= pair_frame["throughput_idle_layers"]
            ).sum()
        ),
    }
    for metric, value in fig19_values.items():
        _add_claim(
            claims,
            "Fig19",
            metric,
            value,
            claim_status,
            "Computed across all 14 archived batch-size/phase pairs.",
        )
    if is_legacy:
        gain_ok = (
            fig19_values["tokens_j_gain_min_pct"] >= 3.0
            and fig19_values["tokens_j_gain_max_pct"] >= 23.0
            and fig19_values[
                "scenarios_efficiency_has_at_least_as_many_idle_layers"
            ]
            == 14
        )
        power_ok = (
            fig19_values["power_reduction_min_pct"] > 0.0
            and fig19_values["power_reduction_max_pct"] >= 48.0
        )
        _add_claim(
            claims,
            "Fig19",
            "paper_energy_architecture_pattern",
            gain_ok and power_ok,
            "PASS" if gain_ok and power_ok else "FAIL",
            "3--24% tokens/J gain; all efficiency winners lower power and use no fewer idle layers.",
        )

    hottest = _selected(frame, "fig20_hottest_per_layer")
    raw_best = _selected(frame, "fig20_raw_throughput_best_per_layer")
    safe_best = _selected(frame, "fig20_safe_throughput_best_per_layer")
    sample_raw_error = float(
        terrain_samples["raw_stps_error_pct"].abs().max()
    )
    sample_temperature_error = float(
        terrain_samples["temperature_error_C"].abs().max()
    )
    sample_ok = (
        sample_raw_error <= LEGACY_MAX_ERROR_PCT
        and sample_temperature_error <= LEGACY_MAX_TEMPERATURE_ERROR_C
    )
    sample_status = (
        ("PASS" if sample_ok else "FAIL") if is_legacy else claim_status
    )
    _add_claim(
        claims,
        "Fig20",
        "archived_terrain_stratified_hash_samples",
        len(terrain_samples),
        sample_status,
        "One non-extremal sample for every displayed (BS, layer) stratum.",
    )
    _add_claim(
        claims,
        "Fig20",
        "terrain_sample_max_abs_raw_stps_error_pct",
        sample_raw_error,
        sample_status,
        "Direct check of the raw-STPS coordinate used by the terrain plot.",
    )
    _add_claim(
        claims,
        "Fig20",
        "terrain_sample_max_abs_temperature_error_C",
        sample_temperature_error,
        sample_status,
        "Temperature is recomputed from model power and analytical resistance.",
    )
    hot_by_bs = hottest.loc[hottest.groupby("bs")["temperature_C"].idxmax()]
    for _, row in hot_by_bs.sort_values("bs").iterrows():
        _add_claim(
            claims,
            "Fig20",
            f"decode_bs{int(row['bs'])}_selected_max_temperature_C",
            float(row["temperature_C"]),
            claim_status,
            f"m={int(row['dram_total_layers'])}, raw STPS={float(row['recomputed_raw_stps']):.3f}.",
        )
    large_peak = raw_best[raw_best["bs"] == 1024].loc[
        raw_best[raw_best["bs"] == 1024]["recomputed_raw_stps"].idxmax()
    ]
    large_safe = safe_best[safe_best["bs"] == 1024].loc[
        safe_best[safe_best["bs"] == 1024]["recomputed_raw_stps"].idxmax()
    ]
    _add_claim(
        claims,
        "Fig20",
        "decode_bs1024_fixed_curve_raw_peak_layer",
        int(large_peak["dram_total_layers"]),
        claim_status,
        f"{float(large_peak['recomputed_raw_stps']):.3f} tok/s at {float(large_peak['temperature_C']):.3f} C.",
    )
    _add_claim(
        claims,
        "Fig20",
        "decode_bs1024_best_selected_safe_raw_stps",
        float(large_safe["recomputed_raw_stps"]),
        claim_status,
        f"m={int(large_safe['dram_total_layers'])}, T={float(large_safe['temperature_C']):.3f} C.",
    )
    if is_legacy:
        small_hot = float(hot_by_bs[hot_by_bs["bs"] == 4].iloc[0]["temperature_C"])
        large_hot = float(
            hot_by_bs[hot_by_bs["bs"] == 1024].iloc[0]["temperature_C"]
        )
        thermal_ok = (
            large_hot > small_hot
            and large_hot > 130.0
            and int(large_peak["dram_total_layers"]) == 7
            and float(large_safe["temperature_C"]) <= THERMAL_LIMIT_C + 1.0e-9
        )
        _add_claim(
            claims,
            "Fig20",
            "paper_thermal_feasibility_pattern",
            thermal_ok,
            "PASS" if thermal_ok else "FAIL",
            "Large-batch decode is hotter; hot/slow and safe/high-throughput selected points coexist.",
        )

    return pd.DataFrame(claims)


def _legacy_regression(
    terrain_samples: pd.DataFrame,
) -> tuple[bool, float, float]:
    maximum = float(terrain_samples["raw_stps_error_pct"].abs().max())
    maximum_temperature = float(
        terrain_samples["temperature_error_C"].abs().max()
    )
    return (
        maximum <= LEGACY_MAX_ERROR_PCT
        and maximum_temperature <= LEGACY_MAX_TEMPERATURE_ERROR_C,
        maximum,
        maximum_temperature,
    )


def _regression_digest_path(directory: Path, model_version: str) -> Path:
    return directory / f"{model_version}_regression.sha256"


def _canonical_regression_digest(
    frame: pd.DataFrame, model_version: str
) -> str:
    """Hash a stable, high-precision projection without publishing its values."""

    missing = sorted(set(REGRESSION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"regression frame is missing columns: {missing}")
    ordered = frame[list(REGRESSION_COLUMNS)].sort_values(
        "candidate_id"
    )
    if len(ordered) != EXPECTED_CONFIGS or ordered["candidate_id"].duplicated().any():
        raise ValueError("regression digest requires 114 unique candidate IDs")

    lines = [
        REGRESSION_DIGEST_FORMAT,
        model_version,
        str(REGRESSION_SIGNIFICANT_DIGITS),
        ",".join(REGRESSION_COLUMNS),
    ]
    precision = REGRESSION_SIGNIFICANT_DIGITS - 1
    for row in ordered.itertuples(index=False, name=None):
        values: list[str] = []
        for column, value in zip(REGRESSION_COLUMNS, row, strict=True):
            if column == "candidate_id":
                values.append(str(value))
            elif column == "hit_power_wall":
                values.append("1" if _as_bool(value) else "0")
            else:
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError(
                        f"non-finite regression value for {column}: {value!r}"
                    )
                values.append(f"{number:.{precision}e}")
        lines.append(",".join(values))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_regression_digest(
    frame: pd.DataFrame, path: Path, model_version: str
) -> str:
    digest = _canonical_regression_digest(frame, model_version)
    record = {
        "columns": list(REGRESSION_COLUMNS),
        "format": REGRESSION_DIGEST_FORMAT,
        "model_version": model_version,
        "rows": EXPECTED_CONFIGS,
        "sha256": digest,
        "significant_digits": REGRESSION_SIGNIFICANT_DIGITS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return digest


def _check_regression_digest(
    frame: pd.DataFrame, path: Path, model_version: str
) -> tuple[bool, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {model_version} regression digest {path}; maintainers may "
            "create it with --refresh-regression-digests"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    expected_metadata = {
        "format": REGRESSION_DIGEST_FORMAT,
        "model_version": model_version,
        "rows": EXPECTED_CONFIGS,
        "significant_digits": REGRESSION_SIGNIFICANT_DIGITS,
        "columns": list(REGRESSION_COLUMNS),
    }
    for key, expected in expected_metadata.items():
        if record.get(key) != expected:
            raise AssertionError(
                f"{path}: regression digest metadata {key!r} is "
                f"{record.get(key)!r}, expected {expected!r}"
            )
    expected_digest = str(record.get("sha256", ""))
    if len(expected_digest) != 64:
        raise AssertionError(f"{path}: invalid SHA-256 digest")
    actual_digest = _canonical_regression_digest(frame, model_version)
    return actual_digest == expected_digest, actual_digest


def _write_dual_drift(outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    if set(outputs) != set(MODEL_VERSIONS):
        return
    legacy = outputs["paper_legacy"]
    corrected = outputs["current_corrected"]
    identity = [
        "candidate_id",
        "phase",
        "selection_roles",
        "bs",
        "dram_total_layers",
        "dram_active_layers",
    ]
    value_columns = [
        "recomputed_raw_stps",
        "recomputed_scaled_stps",
        "total_power_per_device_W",
        "tokens_per_joule",
        "temperature_C",
    ]
    merged = legacy[identity + value_columns].merge(
        corrected[identity + value_columns],
        on=identity,
        suffixes=("_paper_legacy", "_current_corrected"),
        validate="one_to_one",
    )
    for column in value_columns:
        merged[f"{column}_drift_pct"] = 100.0 * (
            merged[f"{column}_current_corrected"]
            / merged[f"{column}_paper_legacy"]
            - 1.0
        )
    drift_columns = identity + [
        f"{column}_drift_pct" for column in value_columns
    ]
    merged[drift_columns].to_csv(
        output_dir / "dual_path_drift.csv", index=False, float_format="%.15g"
    )
    drift = merged.groupby(["phase", "bs"]).agg(
        configs=("candidate_id", "count"),
        mean_abs_scaled_stps_drift_pct=(
            "recomputed_scaled_stps_drift_pct",
            lambda values: values.abs().mean(),
        ),
        max_abs_scaled_stps_drift_pct=(
            "recomputed_scaled_stps_drift_pct",
            lambda values: values.abs().max(),
        ),
    )
    drift.reset_index().to_csv(
        output_dir / "dual_path_drift_summary.csv",
        index=False,
        float_format="%.15g",
    )


def run(
    manifest_path: Path,
    output_dir: Path,
    model_version: str,
    workers: int,
    regression_digest_dir: Path,
    refresh_regression_digests: bool,
) -> bool:
    activate_vendored_sources()
    _verify_checksums(DEFAULT_CHECKSUMS)
    manifest = pd.read_csv(manifest_path)
    _validate_manifest(manifest)
    versions = MODEL_VERSIONS if model_version == "both" else (model_version,)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_manifest = _publish_archives(output_dir)
    outputs: dict[str, pd.DataFrame] = {}
    all_pass = True

    for version in versions:
        started = time.perf_counter()
        frame = _evaluate_version(manifest, version, workers)
        elapsed = time.perf_counter() - started
        version_dir = output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        frame[list(PUBLISHED_MODEL_COLUMNS)].to_csv(
            version_dir / "model_points.csv", index=False, float_format="%.15g"
        )
        terrain_samples = _write_terrain_sample_validation(frame, version_dir)
        claims = _claim_summary(frame, version, terrain_samples)

        digest_path = _regression_digest_path(regression_digest_dir, version)
        if refresh_regression_digests:
            _write_regression_digest(frame, digest_path, version)
        digest_ok, actual_digest = _check_regression_digest(
            frame, digest_path, version
        )
        if version == "paper_legacy":
            historical_ok, maximum_error, maximum_temperature_error = (
                _legacy_regression(terrain_samples)
            )
            regression_ok = historical_ok and digest_ok
            regression_note = (
                f"canonical SHA-256={'match' if digest_ok else 'mismatch'} "
                f"({actual_digest[:12]}...); max archived sample raw-STPS error="
                f"{maximum_error:.9g}% and temperature error="
                f"{maximum_temperature_error:.9g} C"
            )
        else:
            regression_ok = digest_ok
            regression_note = (
                f"canonical SHA-256={'match' if digest_ok else 'mismatch'} "
                f"at {REGRESSION_SIGNIFICANT_DIGITS} significant digits "
                f"({actual_digest[:12]}...)"
            )

        claims = pd.concat(
            [
                claims,
                pd.DataFrame(
                    [
                        {
                            "figure": "Figs18-20",
                            "metric": "model_regression",
                            "value": regression_ok,
                            "status": "PASS" if regression_ok else "FAIL",
                            "note": regression_note,
                        },
                        {
                            "figure": "Figs18-20",
                            "metric": "wall_runtime_s",
                            "value": elapsed,
                            "status": "INFO",
                            "note": f"{len(frame)} CPU-only fixed configurations, workers={workers}.",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        claims.to_csv(
            version_dir / "claim_summary.csv", index=False, float_format="%.15g"
        )
        claim_fail = bool((claims["status"] == "FAIL").any())
        version_pass = regression_ok and not claim_fail
        status = {
            "status": "PASS" if version_pass else "FAIL",
            "model_version": version,
            "fixed_configurations": len(frame),
            "workers": workers,
            "wall_runtime_s": elapsed,
            "gpu_required": False,
            "search_performed": False,
            "archived_projection_rows": int(archive_manifest["rows"].sum()),
            "terrain_samples_recomputed": len(terrain_samples),
            "current_path_is_same_config_diagnostic": version
            == "current_corrected",
            "regression_note": regression_note,
        }
        (version_dir / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        marker_name = "PASS" if version_pass else "FAIL"
        opposite_name = "FAIL" if version_pass else "PASS"
        (version_dir / opposite_name).unlink(missing_ok=True)
        (version_dir / marker_name).write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[{status['status']}] Figs. 18--20 {version}: "
            f"{len(frame)} fixed configs in {elapsed:.3f}s; {regression_note}"
        )
        outputs[version] = frame
        all_pass = all_pass and version_pass

    _write_dual_drift(outputs, output_dir)
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version", choices=MODEL_VERSION_CHOICES, default="both"
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--regression-digest-dir",
        type=Path,
        default=DEFAULT_REGRESSION_DIGEST_DIR,
    )
    parser.add_argument(
        "--refresh-regression-digests",
        action="store_true",
        help="maintainer-only: replace the non-reversible regression baselines",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    passed = run(
        manifest_path=args.manifest,
        output_dir=args.output,
        model_version=args.model_version,
        workers=args.workers,
        regression_digest_dir=args.regression_digest_dir,
        refresh_regression_digests=args.refresh_regression_digests,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
