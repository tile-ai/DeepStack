"""Re-evaluate the fixed winners plotted in paper Figure 23.

Figure 23 was produced by an exhaustive search over three nested parallelism
spaces.  The artifact bundles only the canonical winner for every plotted
curve point and re-evaluates those fixed configurations; it never repeats the
exhaustive search.  ``current_corrected`` is therefore a fixed-configuration
diagnostic, not a corrected-model DSE.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import DATA_DIR, RESULTS_DIR, DEEPSTACK_SRC, activate_vendored_sources


MODEL_VERSIONS = ("paper_legacy", "current_corrected")
TIERS = ("astra_space", "expanded_parallel", "module_flexible")
MODELS = ("DeepSeekV3", "Qwen3_235b_a22b")
EXPECTED_BATCH_SIZES = {
    "decode": (1, 4, 16, 64, 128, 512, 1024),
    "prefill": (1, 4, 8, 16, 64, 128, 1024),
}
LEGACY_MAX_ERROR_PCT = 1.0e-8
CURRENT_MAX_ERROR_PCT = 1.0e-8

_TRACE_CACHE: dict[str, Any] = {}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _seed_model() -> None:
    import numpy as np
    import torch

    # TileSight's cache estimator uses NumPy permutations.  A per-point seed
    # makes the result independent of process scheduling.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)


def _trace_for_model(model_name: str) -> Any:
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape

    if model_name not in _TRACE_CACHE:
        if model_name == "Qwen3_235b_a22b":
            path = (
                DEEPSTACK_SRC
                / "mosaic"
                / "data"
                / "aime_qwen_235b"
                / "qwen3_moe_activations_batch0.npz"
            )
        else:
            path = (
                DEEPSTACK_SRC
                / "mosaic"
                / "data"
                / "aime_ds_r1"
                / "moe_activations_batch0.npz"
            )
        _, _TRACE_CACHE[model_name] = load_npz_routing_keep_shape(
            str(path), as_list=False
        )
    return _TRACE_CACHE[model_name]


def _scheme(row: dict[str, Any]) -> Any:
    from mosaic.parallelism import ParallelScheme

    return ParallelScheme(
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


def _worker(row: dict[str, Any]) -> dict[str, Any]:
    activate_vendored_sources()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = str(row["_model_version"])

    import importlib

    from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
        modeling_decode,
    )
    from mosaic.dse_space.dse_framework_multi_process_v4_prefill_dump_stats import (
        modeling_prefill,
    )
    from mosaic.utils import Modeling_Granularity

    _seed_model()
    logging.getLogger("mosaic.parallelism.parallel").setLevel(logging.ERROR)

    arch_module = importlib.import_module("mosaic.arch")
    model_module = importlib.import_module("mosaic.llm_arch")
    noc_module = importlib.import_module("mosaic.noc.noc_config_set")
    scheme = _scheme(row)
    arch = getattr(arch_module, str(row["arch"]))()
    noc = getattr(noc_module, str(row["noc"]))()
    model_name = str(row["model"])
    model = getattr(model_module, model_name)()
    granularity = Modeling_Granularity("coarse", True, False, True)
    trace = _trace_for_model(model_name)

    started = time.perf_counter()
    if str(row["phase"]) == "decode":
        non_moe = dataclasses.replace(
            scheme,
            ep=1,
            ep1=1,
            ep2=1,
            dp=scheme.dp * scheme.ep,
        )
        kv_lengths = [int(row[f"kv_len_{index}"]) for index in range(1, 5)]
        values, _, _ = modeling_decode(
            model_arch=model,
            bs=int(row["minibatch"]),
            seq=int(row["seq"]),
            cached_kv_list=kv_lengths,
            moe_parallel=scheme,
            non_moe_parallel=non_moe,
            single_chip=arch,
            noc_hierarchy=noc,
            granularity=granularity,
            routing_array=trace,
            tp_transform_moe=str(row["tp_transform_moe"]),
        )
        reproduced_utps = (
            sum(1.0 / value[-1] / scheme.pp for value in values) / len(values)
        )
        reproduced_stps = (
            sum(int(row["minibatch"]) / value[-1] for value in values)
            / len(values)
        )
    else:
        attention = dataclasses.replace(
            scheme,
            ep=1,
            ep1=1,
            ep2=1,
            dp=scheme.dp * scheme.ep,
        )
        moe = dataclasses.replace(scheme, cp=1, sp=scheme.cp * scheme.sp)
        non_attention_non_moe = dataclasses.replace(
            attention, cp=1, sp=scheme.cp * scheme.sp
        )
        values = modeling_prefill(
            model_arch=model,
            bs=int(row["minibatch"]),
            seq=int(row["seq"]),
            cached_kv=int(row["cached_kv"]),
            parallel=scheme,
            atten_parallel=attention,
            moe_parallel=moe,
            non_atten_non_moe_parallel=non_attention_non_moe,
            single_chip=arch,
            noc_hierarchy=noc,
            granularity=granularity,
            routing_array=trace,
            tp_transform_moe=str(row["tp_transform_moe"]),
        )
        total_time = values[7]
        reproduced_utps = int(row["seq"]) / total_time / scheme.pp
        reproduced_stps = int(row["minibatch"]) * int(row["seq"]) / total_time

    return {
        "evaluation_key": str(row["evaluation_key"]),
        "model_version": str(row["_model_version"]),
        "reproduced_utps": reproduced_utps,
        "reproduced_stps": reproduced_stps,
        "model_runtime_s": time.perf_counter() - started,
    }


def _evaluation_key(row: pd.Series) -> str:
    fields = [
        "phase",
        "model",
        "arch",
        "noc",
        "minibatch",
        "seq",
        "cached_kv",
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
    ]
    payload = "|".join(str(row[field]) for field in fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validate_manifest(frame: pd.DataFrame) -> None:
    required = {
        "candidate_id",
        "phase",
        "model",
        "bs",
        "tier",
        "arch",
        "noc",
        "minibatch",
        "seq",
        "cached_kv",
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
        "paper_utps",
        "paper_stps",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fig. 23 manifest is missing columns: {missing}")
    if len(frame) != 84 or frame.candidate_id.duplicated().any():
        raise ValueError("Fig. 23 manifest must contain 84 unique curve points")
    if set(frame.model) != set(MODELS) or set(frame.tier) != set(TIERS):
        raise ValueError("Fig. 23 model or tier coverage is incomplete")
    if set(frame.arch) != {"stacked_gpu_wgmma"} or set(frame.noc) != {
        "torus_mesh_switch_1"
    }:
        raise ValueError("Fig. 23 winners must use the paper's fixed architecture")

    for phase, batches in EXPECTED_BATCH_SIZES.items():
        subset = frame[frame.phase == phase]
        if len(subset) != 42 or tuple(sorted(subset.bs.unique())) != batches:
            raise ValueError(f"Fig. 23 {phase} coverage mismatch")
        counts = subset.groupby(["model", "bs"]).size()
        if not (counts == 3).all():
            raise ValueError(f"Fig. 23 {phase} requires three tiers per point")

    for row in frame.to_dict(orient="records"):
        world_size = math.prod(
            int(row[name]) for name in ("tp", "ep", "sp", "cp", "dp", "pp")
        )
        if world_size != 256:
            raise ValueError(f"{row['candidate_id']} uses {world_size}, not 256 devices")
        if int(row["minibatch"]) != math.ceil(int(row["bs"]) / int(row["pp"])):
            raise ValueError(f"{row['candidate_id']} has an invalid minibatch")
        if row["tier"] == "astra_space":
            if not (
                int(row["ep"]) == 1
                and int(row["sp"]) == 1
                and int(row["cp"]) == 1
                and not _as_bool(row["fsdp"])
                and row["tp_transform_moe"] == "none"
            ):
                raise ValueError(f"{row['candidate_id']} violates ASTRA-space")
        if row["tier"] == "expanded_parallel" and row["tp_transform_moe"] != "none":
            raise ValueError(f"{row['candidate_id']} is not module-uniform")


def _summary(result: pd.DataFrame) -> pd.DataFrame:
    paper = result.pivot(
        index=["phase", "model", "bs"], columns="tier", values="paper_stps"
    )
    reproduced = result.pivot(
        index=["phase", "model", "bs"],
        columns="tier",
        values="reproduced_stps",
    )
    rows: list[dict[str, object]] = []
    for key in paper.index:
        phase, model, bs = key
        row: dict[str, object] = {"phase": phase, "model": model, "bs": bs}
        for tier in TIERS:
            row[f"paper_{tier}_stps"] = float(paper.loc[key, tier])
            row[f"reproduced_{tier}_stps"] = float(reproduced.loc[key, tier])
        for prefix, values in (("paper", paper), ("reproduced", reproduced)):
            row[f"{prefix}_expanded_over_astra"] = float(
                values.loc[key, "expanded_parallel"]
                / values.loc[key, "astra_space"]
            )
            row[f"{prefix}_flexible_over_expanded"] = float(
                values.loc[key, "module_flexible"]
                / values.loc[key, "expanded_parallel"]
            )
            row[f"{prefix}_flexible_over_astra"] = float(
                values.loc[key, "module_flexible"]
                / values.loc[key, "astra_space"]
            )
        reported = float("nan")
        if phase == "decode" and bs == 1024:
            reported = 5.03 if model == "DeepSeekV3" else 2.31
        row["paper_reported_flexible_over_astra"] = reported
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["phase", "model", "bs"])


def _headline_claims(summary: pd.DataFrame, model_version: str) -> pd.DataFrame:
    reported = {
        "DeepSeekV3": {
            "expanded_over_astra": 4.16,
            "flexible_over_expanded": 1.21,
            "flexible_over_astra": 5.03,
        },
        "Qwen3_235b_a22b": {
            "expanded_over_astra": 1.90,
            "flexible_over_expanded": 1.21,
            "flexible_over_astra": 2.31,
        },
    }
    headline = summary[(summary.phase == "decode") & (summary.bs == 1024)]
    rows: list[dict[str, object]] = []
    for source in headline.to_dict(orient="records"):
        for metric, paper_reported in reported[str(source["model"])].items():
            paper_recomputed = float(source[f"paper_{metric}"])
            reproduced = float(source[f"reproduced_{metric}"])
            if round(paper_recomputed, 2) != paper_reported:
                raise AssertionError(
                    f"Fig. 23 source no longer rounds to the reported {paper_reported} "
                    f"for {source['model']} {metric}: {paper_recomputed}"
                )
            rows.append(
                {
                    "model_version": model_version,
                    "model": source["model"],
                    "phase": "decode",
                    "bs": 1024,
                    "metric": metric,
                    "paper_reported": paper_reported,
                    "paper_recomputed": paper_recomputed,
                    "reproduced": reproduced,
                    "reproduced_delta_from_paper_recomputed_pct": 100.0
                    * (reproduced / paper_recomputed - 1.0),
                }
            )
    return pd.DataFrame(rows)


def _validate_nested_paper_trend(summary: pd.DataFrame, model_version: str) -> None:
    paper_nested = (
        (summary.paper_expanded_over_astra >= 1.0)
        & (summary.paper_flexible_over_expanded >= 1.0)
    )
    if not paper_nested.all():
        raise AssertionError("paper Figure 23 violates its nested-search-space trend")
    if model_version == "paper_legacy":
        reproduced_nested = (
            (summary.reproduced_expanded_over_astra >= 1.0 - 1.0e-10)
            & (summary.reproduced_flexible_over_expanded >= 1.0 - 1.0e-10)
        )
        if not reproduced_nested.all():
            raise AssertionError("legacy rerun does not preserve the Figure 23 trend")


def _validate_current_expected(result: pd.DataFrame, expected_path: Path) -> float:
    if not expected_path.exists():
        raise FileNotFoundError(
            f"missing Fig. 23 current-corrected regression table: {expected_path}"
        )
    expected = pd.read_csv(expected_path)
    required = {"candidate_id", "expected_utps", "expected_stps"}
    if set(expected.columns) != required or len(expected) != 84:
        raise ValueError("invalid Fig. 23 current-corrected regression table")
    if set(expected.candidate_id) != set(result.candidate_id):
        raise ValueError("Fig. 23 regression IDs do not match the winner manifest")
    checked = result.merge(expected, on="candidate_id", validate="one_to_one")
    errors = pd.concat(
        [
            100.0 * (checked.reproduced_utps / checked.expected_utps - 1.0),
            100.0 * (checked.reproduced_stps / checked.expected_stps - 1.0),
        ],
        ignore_index=True,
    ).abs()
    maximum = float(errors.max())
    if maximum > CURRENT_MAX_ERROR_PCT:
        raise AssertionError(
            f"current-corrected regression error {maximum:.12g}% exceeds "
            f"{CURRENT_MAX_ERROR_PCT:.12g}%"
        )
    return maximum


def run(
    manifest_path: Path,
    output_root: Path,
    *,
    model_version: str,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"unknown model version: {model_version}")
    manifest = pd.read_csv(manifest_path)
    _validate_manifest(manifest)
    manifest["evaluation_key"] = manifest.apply(_evaluation_key, axis=1)
    unique = manifest.drop_duplicates("evaluation_key").copy()
    unique["_model_version"] = model_version

    started = time.perf_counter()
    computed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_worker, row): row["evaluation_key"]
            for row in unique.to_dict(orient="records")
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                computed.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Fig. 23 evaluation {key} failed") from exc
    wall_time = time.perf_counter() - started

    result = manifest.merge(
        pd.DataFrame(computed), on="evaluation_key", validate="many_to_one"
    )
    result["delta_from_paper_utps_pct"] = 100.0 * (
        result.reproduced_utps / result.paper_utps - 1.0
    )
    result["delta_from_paper_stps_pct"] = 100.0 * (
        result.reproduced_stps / result.paper_stps - 1.0
    )
    result.sort_values(["phase", "model", "bs", "tier_order"], inplace=True)
    summary = _summary(result)
    _validate_nested_paper_trend(summary, model_version)
    claims = _headline_claims(summary, model_version)

    maximum_paper_error = float(
        max(
            result.delta_from_paper_utps_pct.abs().max(),
            result.delta_from_paper_stps_pct.abs().max(),
        )
    )
    if model_version == "paper_legacy" and maximum_paper_error > LEGACY_MAX_ERROR_PCT:
        raise AssertionError(
            f"paper-legacy maximum error {maximum_paper_error:.12g}% exceeds "
            f"{LEGACY_MAX_ERROR_PCT:.12g}%"
        )

    current_error = float("nan")
    if model_version == "current_corrected":
        current_error = _validate_current_expected(
            result, DATA_DIR / "fig23" / "current_corrected_expected.csv"
        )

    output_dir = output_root / model_version
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "raw.csv", index=False, float_format="%.15g")
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.15g")
    claims.to_csv(
        output_dir / "headline_claims.csv", index=False, float_format="%.15g"
    )
    headline = summary[(summary.phase == "decode") & (summary.bs == 1024)]
    regression_value = (
        "not_applicable"
        if math.isnan(current_error)
        else f"{current_error:.15g}"
    )
    scope = (
        "paper_legacy replays the submitted fixed winners"
        if model_version == "paper_legacy"
        else "current_corrected is a fixed-config diagnostic, not a new DSE"
    )
    lines = [
        "PASS",
        f"model_version={model_version}",
        f"curve_points={len(result)}",
        f"unique_fixed_configurations={len(unique)}",
        f"wall_time_s={wall_time:.6f}",
        f"maximum_error_vs_paper_pct={maximum_paper_error:.15g}",
        f"maximum_current_regression_error_pct={regression_value}",
        "paper_headline_claims=PASS",
        "paper_nested_search_space_trend=PASS",
        f"scope={scope}",
    ]
    for row in headline.itertuples(index=False):
        lines.append(
            f"decode_bs1024_{row.model}_flexible_over_astra="
            f"{row.reproduced_flexible_over_astra:.15g}"
        )
    (output_dir / "PASS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result, summary, wall_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA_DIR / "fig23" / "selected_winners_march2026.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_DIR / "reproduce" / "stages" / "fig23",
    )
    parser.add_argument(
        "--model-version",
        choices=(*MODEL_VERSIONS, "both"),
        default="both",
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()

    versions = MODEL_VERSIONS if args.model_version == "both" else (args.model_version,)
    for version in versions:
        result, summary, wall_time = run(
            args.manifest,
            args.output_root,
            model_version=version,
            workers=args.workers,
        )
        maximum_error = max(
            result.delta_from_paper_utps_pct.abs().max(),
            result.delta_from_paper_stps_pct.abs().max(),
        )
        headline = summary[(summary.phase == "decode") & (summary.bs == 1024)]
        print(f"Fig. 23 {version}: {len(result)} curve points")
        print(f"unique fixed configurations: {result.evaluation_key.nunique()}")
        print(f"wall time: {wall_time:.3f} s")
        print(f"maximum fixed-config error vs paper: {maximum_error:.12g}%")
        print(
            headline[
                [
                    "model",
                    "paper_flexible_over_astra",
                    "reproduced_flexible_over_astra",
                ]
            ].to_string(index=False)
        )
        print(f"wrote {args.output_root / version / 'raw.csv'}")
        print(f"wrote {args.output_root / version / 'summary.csv'}")
        print(f"wrote {args.output_root / version / 'headline_claims.csv'}")
        print(f"wrote {args.output_root / version / 'PASS'}")


if __name__ == "__main__":
    main()
