"""Select and re-evaluate fixed decode configurations."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .paths import activate_vendored_sources


TRACE_CACHE: dict[str, Any] = {}

FIG15_MODELS = ("DeepSeekV3", "Qwen3_235b_a22b", "Llama3_405b", "Llama3_70b")
FIG15_BATCH_SIZES = (1, 4, 16, 64, 128, 256, 1024)
FIG15_LOCAL_BEST_ROWS = 784
FIG15_QUICK_ROWS = 28
FIG15_QUICK_SALT = "fig15-dse-replay-v1"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _trace_for_model(model_name: str) -> Any:
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
    from .paths import DEEPSTACK_SRC

    trace_name = "qwen" if model_name == "Qwen3_235b_a22b" else "deepseek"
    if trace_name not in TRACE_CACHE:
        if trace_name == "qwen":
            path = DEEPSTACK_SRC / "mosaic" / "data" / "aime_qwen_235b" / "qwen3_moe_activations_batch0.npz"
        else:
            path = DEEPSTACK_SRC / "mosaic" / "data" / "aime_ds_r1" / "moe_activations_batch0.npz"
        _, TRACE_CACHE[trace_name] = load_npz_routing_keep_shape(str(path))
    return TRACE_CACHE[trace_name]


def _worker(row: dict[str, Any]) -> dict[str, Any]:
    activate_vendored_sources()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["DEEPSTACK_SWIGLU_COUNTS_MODE"] = str(row["_model_version"])

    import importlib

    import numpy as np
    import torch

    from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import modeling_decode
    from mosaic.parallelism import ParallelScheme
    from mosaic.utils import Modeling_Granularity

    # TileSight's legacy cache estimator permutes blocks. Resetting the RNG
    # for each fixed configuration makes process scheduling irrelevant.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    # The framework explicitly allows ep1*ep2 < ep when the local token grid is
    # smaller than EP, so suppress that expected per-candidate warning here.
    logging.getLogger("mosaic.parallelism.parallel").setLevel(logging.ERROR)
    arch_module = importlib.import_module("mosaic.arch")
    llm_module = importlib.import_module("mosaic.llm_arch")
    noc_module = importlib.import_module("mosaic.noc.noc_config_set")

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
    current_utps = sum(1.0 / values[-1] / scheme.pp for values in times) / len(times)
    current_stps = sum(int(row["minibatch"]) / values[-1] for values in times) / len(times)
    return {
        "candidate_id": str(row["candidate_id"]),
        "model_version": str(row["_model_version"]),
        "current_utps_avg": current_utps,
        "current_stps_avg": current_stps,
        "model_runtime_s": time.perf_counter() - started,
    }


def _with_candidate_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the source run identifier as the stable replay identifier."""

    result = frame.copy()
    if "candidate_id" not in result:
        if "stats_run_id" not in result:
            raise ValueError("candidate table needs candidate_id or stats_run_id")
        result.insert(0, "candidate_id", result.stats_run_id.astype(str))
    if result.candidate_id.isna().any() or result.candidate_id.duplicated().any():
        raise ValueError("candidate_id values must be present and unique")
    return result


def load_candidate_table(path: Path) -> pd.DataFrame:
    """Load either a standalone manifest or the canonical DSE population."""

    return _with_candidate_id(pd.read_csv(path))


def _fig15_population(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id",
        "arch",
        "noc",
        "model",
        "bs",
        "paper_stps_avg",
        "paper_utps_avg",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fig. 15 population is missing columns: {missing}")
    subset = frame[
        frame.model.isin(FIG15_MODELS) & frame.bs.isin(FIG15_BATCH_SIZES)
    ].copy()
    if subset.empty or not subset.arch.astype(str).str.startswith("stacked_gpu_").all():
        raise ValueError("Fig. 15 population must contain only stacked-GPU architectures")
    return subset


def select_fig15_local_best(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the paper local best for every model/BS/architecture/NoC bucket."""

    candidates = _fig15_population(_with_candidate_id(frame))
    # pandas idxmax keeps the first exact maximum, reproducing the historical
    # extractor's canonical-source-order tie break.
    indices = candidates.groupby(
        ["model", "bs", "arch", "noc"], sort=True
    ).paper_stps_avg.idxmax()
    selected = candidates.loc[indices].sort_values(
        ["model", "bs", "paper_stps_avg", "candidate_id"],
        ascending=[True, True, False, True],
    )
    if len(selected) != FIG15_LOCAL_BEST_ROWS:
        raise ValueError(
            f"expected {FIG15_LOCAL_BEST_ROWS} Fig. 15 local-best rows, "
            f"found {len(selected)}"
        )
    if selected.groupby(["model", "bs"]).ngroups != 28:
        raise ValueError("expected 28 Fig. 15 model/batch-size strata")
    return selected.reset_index(drop=True)


def _sample_hash(candidate_id: object) -> str:
    value = f"{FIG15_QUICK_SALT}|{candidate_id}".encode()
    return hashlib.sha256(value).hexdigest()


def select_fig15_quick(frame: pd.DataFrame) -> pd.DataFrame:
    """Hash-sample one non-winner from each model/target-BS stratum."""

    population = _fig15_population(_with_candidate_id(frame))
    winner_ids = set(select_fig15_local_best(population).candidate_id.astype(str))
    eligible = population[
        ~population.candidate_id.astype(str).isin(winner_ids)
    ].copy()
    eligible["_sample_hash"] = eligible.candidate_id.map(_sample_hash)
    selected = (
        eligible.sort_values(["model", "bs", "_sample_hash", "candidate_id"])
        .groupby(["model", "bs"], sort=True, as_index=False)
        .head(1)
        .drop(columns="_sample_hash")
        .sort_values(["model", "bs"])
        .reset_index(drop=True)
    )
    if len(selected) != FIG15_QUICK_ROWS:
        raise ValueError(
            f"expected {FIG15_QUICK_ROWS} Fig. 15 quick rows, found {len(selected)}"
        )
    if set(selected.candidate_id.astype(str)) & winner_ids:
        raise ValueError("Fig. 15 quick sample overlaps the local-best winners")
    return selected


def run_candidates(
    candidate_csv: Path,
    output_csv: Path,
    *,
    workers: int,
    model_version: str,
    fig15_scope: str | None = None,
) -> pd.DataFrame:
    activate_vendored_sources()
    candidates = load_candidate_table(candidate_csv)
    if fig15_scope == "quick":
        candidates = select_fig15_quick(candidates)
    elif fig15_scope == "reproduce":
        candidates = select_fig15_local_best(candidates)
    elif fig15_scope is not None:
        raise ValueError(f"unsupported Fig. 15 scope: {fig15_scope}")
    candidates["_model_version"] = model_version
    rows = candidates.to_dict(orient="records")

    computed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_worker, row): row["candidate_id"] for row in rows}
        for future in as_completed(futures):
            candidate_id = futures[future]
            try:
                computed.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"candidate {candidate_id} failed") from exc

    result = candidates.drop(columns="_model_version").merge(
        pd.DataFrame(computed), on="candidate_id", validate="one_to_one"
    )
    result["delta_from_paper_stps_pct"] = 100.0 * (
        result.current_stps_avg / result.paper_stps_avg - 1.0
    )
    result["delta_from_paper_utps_pct"] = 100.0 * (
        result.current_utps_avg / result.paper_utps_avg - 1.0
    )
    result["current_stps_rank"] = result.groupby(["model", "bs"])["current_stps_avg"].rank(
        method="min", ascending=False
    )
    result["current_utps_rank"] = result.groupby(["model", "bs"])["current_utps_avg"].rank(
        method="min", ascending=False
    )
    result.sort_values(["model", "bs", "current_stps_rank", "candidate_id"], inplace=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, float_format="%.15g")
    return result
