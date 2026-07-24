"""
lat_bw_dse_prefill.py — NoC latency vs bandwidth DSE for prefill (DeepSeek R1 only)

Outer loop: generate_latency_bw_configs()  (baseline, lat_mult, bw_mult)
Inner loop: prefill tasks                  (model, BS, SEQ, SEQ, SEQ)

Global sweep: all layers scaled by same latency/BW multipliers simultaneously.
"""

from typing import Any
from mosaic.dse_space.parallel_schemes import (
    all_parallel_schemes, filter_parallel_schemes_by_max,
    filter_illegal_fsdp, filter_parallel_schemes_by_product_max,
)
from mosaic.llm_arch import DeepSeekV3, LLM_Arch
from mosaic.parallelism import ParallelScheme
from mosaic.utils.allocate_ep import allocate_ep
import logging
log = logging.getLogger(__name__)
import math
import dataclasses
import hashlib
import json
import re
from mosaic.utils import Modeling_Granularity
from mosaic.noc.noc_topo import Hierarchy
from tilesight.arch.arch_base import Arch
import numpy as np
import csv
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
from mosaic.cost.op_perf_stats import OpPerfStats
from mosaic.cost.model_perf_stats import ModelPerfStats

from .lat_bw_config import (
    generate_latency_bw_configs, make_noc_for_config, make_arch_for_noc_config,
    compute_power_wall, TDP_W,
)

from mosaic.dse_space.dse_framework_multi_process_v4_prefill_dump_stats import (
    get_max_footprint_prefill, modeling_prefill,
    _get_effective_moe_parallel, _format_duration, _dump_stats_bundle,
    MODEL_REG,
)

TP_TRANSFORM_MOE_MODES: list[str] = ["replace_only"]

_ROUTING_ARRAY = None


def _init_worker(npz_trace_file: str):
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
    global _ROUTING_ARRAY
    _, decode_array = load_npz_routing_keep_shape(npz_trace_file, as_list=False)
    _ROUTING_ARRAY = decode_array


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return slug.strip("-") or "na"


# ── Task combinations ─────────────────────────────────────────────────────

def lat_bw_prefill_tasks():
    # tasks = lat_bw_prefill_tasks_full()
    tasks = lat_bw_prefill_tasks_short_kv_len()
    return tasks

def lat_bw_prefill_tasks_full():
    tasks = []
    model = DeepSeekV3()
    for BS in [1, 4, 16, 64, 256, 1024]:
        for SEQ in [1024, 8192, 65536]:
            tasks.append((model, BS, SEQ, SEQ, SEQ))
    return tasks

def lat_bw_prefill_tasks_short_kv_len():
    tasks = []
    model = DeepSeekV3()
    for BS in [1, 4, 16, 64, 256, 1024]:
        for SEQ in [1024, 8192]:
            tasks.append((model, BS, SEQ, SEQ, SEQ))
    return tasks


def _make_stats_run_id(
    baseline_name: str, lat_mult: float, bw_mult: float,
    arch_name: str, noc_name: str, model_name: str,
    bs: int, minibatch: int, seq: int,
    parallel_scheme: ParallelScheme, tp_transform_moe: str,
) -> str:
    key = (
        f"baseline={baseline_name}|lat_mult={lat_mult}|bw_mult={bw_mult}|"
        f"arch={arch_name}|noc={noc_name}|model={model_name}|bs={bs}|minibatch={minibatch}|seq={seq}|"
        f"tp={parallel_scheme.tp}|ep={parallel_scheme.ep}|ep1={parallel_scheme.ep1}|ep2={parallel_scheme.ep2}|"
        f"sp={parallel_scheme.sp}|cp={parallel_scheme.cp}|dp={parallel_scheme.dp}|fsdp={parallel_scheme.fsdp}|"
        f"pp={parallel_scheme.pp}|tp_transform_moe={tp_transform_moe}"
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{_slugify(model_name)}_{_slugify(arch_name)}_{_slugify(noc_name)}_{digest}"


# ── Worker ────────────────────────────────────────────────────────────────

def _compute_time_row(args):
    (
        baseline_name, lat_mult, bw_mult,
        model_key,
        BS, minibatch, cached_kv, seq,
        parallel_scheme,
        max_activation, global_total_weight, global_kv_cache,
        granularity_tuple,
        atten_parallel, moe_parallel, non_atten_non_moe_parallel,
        proc_log_dir,
        run_dir,
        stats_bundle_root,
        tp_transform_moe,
    ) = args

    proc_log_file = None
    if proc_log_dir is not None:
        import multiprocessing
        proc_name = multiprocessing.current_process().name
        os.makedirs(proc_log_dir, exist_ok=True)
        proc_log_file = os.path.join(proc_log_dir, f"{proc_name}.log")
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if not root_logger.handlers:
            handler = logging.FileHandler(proc_log_file, encoding="utf-8")
            handler.setFormatter(logging.Formatter('%(asctime)s %(processName)s: %(message)s', datefmt='%H:%M:%S'))
            root_logger.addHandler(handler)
        logging.info(f"Worker {os.getpid()} is running")

    noc = make_noc_for_config(baseline_name, lat_mult, bw_mult)
    arch, _ = make_arch_for_noc_config(noc)
    model_arch = MODEL_REG[model_key]()

    granularity = Modeling_Granularity(
        mode=granularity_tuple[0],
        comp_comm_overlap=granularity_tuple[1],
        auto_tune=granularity_tuple[2],
        dump_perf_log=granularity_tuple[3],
    )

    global _ROUTING_ARRAY
    routing_array = _ROUTING_ARRAY
    if routing_array is None:
        raise RuntimeError("routing_array not initialized in worker")

    pp_p2p_time, time_gqa, time_mla, time_dense_ffn, time_moe, time_rms_norm, time_add_residual, time_total, model_stats = modeling_prefill(
        model_arch=model_arch, bs=minibatch, seq=seq, cached_kv=cached_kv,
        parallel=parallel_scheme, atten_parallel=atten_parallel,
        moe_parallel=moe_parallel, non_atten_non_moe_parallel=non_atten_non_moe_parallel,
        single_chip=arch, noc_hierarchy=noc, granularity=granularity,
        routing_array=routing_array, tp_transform_moe=tp_transform_moe,
    )

    ttft_per_pp = time_total
    ttft = ttft_per_pp * parallel_scheme.pp
    utps = seq / ttft if ttft > 0 else 0
    stps = minibatch * seq / ttft_per_pp if ttft_per_pp > 0 else 0

    # ---- Power wall ----
    e2e_time = time_total
    num_devices = parallel_scheme.world_size()
    chip_power_per_device_w = model_stats.chip_total_energy_j / e2e_time / num_devices if e2e_time > 0 else 0
    noc_power_per_device_w = model_stats.noc_total_energy_j / e2e_time / num_devices if e2e_time > 0 else 0
    total_power_per_device_w = chip_power_per_device_w + noc_power_per_device_w
    hit_power_wall, freq_scale_power = compute_power_wall(chip_power_per_device_w, noc_power_per_device_w)
    scaled_utps = utps * freq_scale_power
    scaled_stps = stps * freq_scale_power

    arch_name = arch.__class__.__name__
    noc_name = noc.name
    model_name = model_arch.__class__.__name__
    stats_run_id = _make_stats_run_id(
        baseline_name=baseline_name, lat_mult=lat_mult, bw_mult=bw_mult,
        arch_name=arch_name, noc_name=noc_name, model_name=model_name,
        bs=BS, minibatch=minibatch, seq=seq,
        parallel_scheme=parallel_scheme, tp_transform_moe=tp_transform_moe,
    )
    proc_log_relpath = ""
    if proc_log_file:
        proc_log_relpath = os.path.relpath(proc_log_file, run_dir)
    _dump_stats_bundle(
        stats_bundle_root=stats_bundle_root,
        run_dir=run_dir,
        stats_run_id=stats_run_id,
        representative_kv_len=cached_kv,
        model_stats=model_stats,
        metadata={
            "baseline_noc": baseline_name, "latency_multiplier": lat_mult, "bw_multiplier": bw_mult,
            "arch": arch_name, "noc": noc_name, "model": model_name,
            "bs": BS, "minibatch": minibatch, "seq": seq, "cached_kv": cached_kv,
            "parallel_scheme": dataclasses.asdict(parallel_scheme),
            "tp_transform_moe": tp_transform_moe,
            "proc_log_relpath": proc_log_relpath,
        },
    )

    return [
        baseline_name, lat_mult, bw_mult,
        arch_name, noc_name, model_name,
        BS, minibatch, seq,
        parallel_scheme,                      # [9]
        max_activation / (1024**3),           # [10]
        global_total_weight / (1024**3),      # [11]
        global_kv_cache / (1024**3),          # [12]
        pp_p2p_time, time_gqa, time_mla, time_dense_ffn, time_moe, time_rms_norm, time_add_residual,  # [13-19]
        ttft_per_pp, ttft,                    # [20-21]
        tp_transform_moe,                     # [22]
        utps, stps,                           # [23-24]
        chip_power_per_device_w, noc_power_per_device_w, total_power_per_device_w,  # [25-27]
        hit_power_wall, freq_scale_power,     # [28-29]
        scaled_utps, scaled_stps,             # [30-31]
        stats_run_id,                         # [32]
    ]


# ── Main DSE loop ─────────────────────────────────────────────────────────

def dse_1(run_dir, num_workers, enable_proc_log=False, start_cfg_idx=0):
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False, dump_perf_log=True)
    gran_tuple = (granularity.mode, granularity.comp_comm_overlap, granularity.auto_tune, granularity.dump_perf_log)
    run_dir = os.path.abspath(run_dir)

    _progress_log_path = os.path.join(run_dir, "progress.log")
    def _log(msg):
        print(msg, flush=True)
        with open(_progress_log_path, "a") as _f:
            _f.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")

    proc_log_dir = os.path.join(run_dir, "proc_logs") if enable_proc_log else None
    stats_bundle_root = os.path.join(run_dir, "stats_bundles")
    os.makedirs(stats_bundle_root, exist_ok=True)

    task_combinations = lat_bw_prefill_tasks()
    noc_configs = generate_latency_bw_configs()

    csv_path = os.path.join(run_dir, "lat_bw_prefill_result.csv")
    invalid_csv_path = os.path.join(run_dir, "lat_bw_prefill_invalid.csv")

    header = [
        "baseline_noc", "latency_multiplier", "bw_multiplier",
        "sm_count",
        "l1_link_bw_GBs", "l2_link_bw_GBs", "l3_link_bw_GBs",
        "arch", "noc", "model", "bs", "minibatch", "seq",
        "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
        "tp_transform_moe",
        "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB",
        "pp_p2p_time_ms", "time_gqa_ms", "time_mla_ms", "time_dense_ffn_ms",
        "time_moe_ms", "time_rms_norm_ms", "time_add_residual_ms",
        "ttft_per_pp_ms", "ttft_ms",
        "raw_utps", "raw_stps",
        "chip_power_W", "noc_power_per_device_W", "total_power_per_device_W",
        "hit_power_wall", "freq_scale_power",
        "scaled_utps", "scaled_stps",
        "stats_run_id",
    ]

    if start_cfg_idx == 0:
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(header)
        with open(invalid_csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow([
                "baseline_noc", "latency_multiplier", "bw_multiplier", "sm_count",
                "arch", "noc", "model", "bs", "minibatch", "seq", "cached_kv",
                "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
                "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB",
            ])

    total_combo_cnt = len(noc_configs) * len(task_combinations)
    finished_combo_cnt = start_cfg_idx * len(task_combinations)
    combo_time_sum_sec = 0.0
    overall_start_ts = time.time()

    _log(f"[PREFILL DSE] noc_configs={len(noc_configs)}, tasks={len(task_combinations)}, "
         f"total_combos={total_combo_cnt}, workers={num_workers}, start_cfg_idx={start_cfg_idx}")

    for cfg_idx, (baseline_name, lat_mult, bw_mult) in enumerate(noc_configs):
        if cfg_idx < start_cfg_idx:
            continue
        noc = make_noc_for_config(baseline_name, lat_mult, bw_mult)
        arch, sm_count = make_arch_for_noc_config(noc)

        l1_link_bw_gbs = noc.layers[-1].link_bandwidth / 1e9
        l2_link_bw_gbs = noc.layers[-2].link_bandwidth / 1e9
        l3_link_bw_gbs = noc.layers[-3].link_bandwidth / 1e9

        meta_prefix = [
            baseline_name, lat_mult, bw_mult,
            sm_count,
            l1_link_bw_gbs, l2_link_bw_gbs, l3_link_bw_gbs,
        ]

        num_nodes = int(noc.num_devices)

        for task_idx, task_combination in enumerate(task_combinations):
            combo_start_ts = time.time()

            model_arch = task_combination[0]
            BS = task_combination[1]
            INPUT_SEQ = task_combination[2]
            TASK_MAX_SEQ = task_combination[3]
            PARALLEL_SEQ = task_combination[4]
            assert INPUT_SEQ == PARALLEL_SEQ

            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent.parent
            if model_arch.__class__.__name__ in ("DeepSeekV3", "DeepSeekV3_A8W8"):
                npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")
            elif model_arch.__class__.__name__ == "Qwen3_235b_a22b":
                npz_trace_file = str(project_root / "data" / "aime_qwen_235b" / "qwen3_moe_activations_batch0.npz")
            else:
                npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")

            MAX_KV_LEN = min(TASK_MAX_SEQ, model_arch.max_seq_len)

            parallel_schemes = all_parallel_schemes(num_nodes)
            parallel_schemes = filter_illegal_fsdp(parallel_schemes)

            filtered_parallel_schemes = filter_parallel_schemes_by_product_max(parallel_schemes, "dp", "pp", BS)
            filtered_parallel_schemes = filter_illegal_fsdp(filtered_parallel_schemes)
            filtered_parallel_schemes = filter_parallel_schemes_by_max(filtered_parallel_schemes, "sp", PARALLEL_SEQ)

            if model_arch.moe_arch is None:
                filtered_parallel_schemes = filter_parallel_schemes_by_max(filtered_parallel_schemes, "ep", 1)
            else:
                filtered_parallel_schemes = filter_parallel_schemes_by_max(filtered_parallel_schemes, "ep", model_arch.moe_arch.num_routed_experts)

            valid_parallel_schemes: list[tuple[ParallelScheme, str]] = []
            scheme_footprints: dict[tuple, tuple] = {}

            filter_start_ts = time.time()
            for scheme in filtered_parallel_schemes:
                minibatch = math.ceil(BS / scheme.pp)
                allocate_ep(parallel=scheme, bs=minibatch, seq=PARALLEL_SEQ)

                atten_parallel = dataclasses.replace(
                    scheme, ep=1, ep1=1, ep2=1, dp=scheme.dp * scheme.ep
                )
                moe_parallel = dataclasses.replace(
                    scheme, cp=1, sp=scheme.cp * scheme.sp
                )
                non_atten_non_moe_parallel = dataclasses.replace(
                    atten_parallel, cp=1, sp=scheme.cp * scheme.sp
                )

                for tp_transform_moe in TP_TRANSFORM_MOE_MODES:
                    max_activation, global_total_weight, global_kv_cache = get_max_footprint_prefill(
                        model_arch=model_arch, bs=minibatch, seq=PARALLEL_SEQ, cached_kv=1,
                        parallel=scheme, atten_parallel=atten_parallel, moe_parallel=moe_parallel,
                        non_atten_non_moe_parallel=non_atten_non_moe_parallel,
                        tp_transform_moe=tp_transform_moe,
                    )

                    if (max_activation + global_total_weight + global_kv_cache) <= 1.0 * arch.ddr_capacity:
                        valid_parallel_schemes.append((scheme, tp_transform_moe))
                        scheme_footprints[(scheme, tp_transform_moe)] = (
                            minibatch, max_activation, global_total_weight, global_kv_cache,
                            atten_parallel, moe_parallel, non_atten_non_moe_parallel,
                        )
                    else:
                        with open(invalid_csv_path, mode="a", newline="") as f:
                            csv.writer(f).writerow([
                                baseline_name, lat_mult, bw_mult, sm_count,
                                arch.__class__.__name__, noc.name, model_arch.__class__.__name__,
                                BS, minibatch, 1, MAX_KV_LEN,
                                scheme.tp, scheme.ep, scheme.ep1, scheme.ep2, scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                                max_activation / (1024**3), global_total_weight / (1024**3), global_kv_cache / (1024**3),
                            ])
            filter_elapsed_sec = time.time() - filter_start_ts

            if not valid_parallel_schemes:
                finished_combo_cnt += 1
                combo_elapsed_sec = time.time() - combo_start_ts
                combo_time_sum_sec += combo_elapsed_sec
                overall_elapsed_sec = time.time() - overall_start_ts
                avg_combo_sec = combo_time_sum_sec / max(finished_combo_cnt, 1)
                remain_combo = max(total_combo_cnt - finished_combo_cnt, 0)
                eta_sec = avg_combo_sec * remain_combo
                _log(
                    f"[Progress] combo {finished_combo_cnt}/{total_combo_cnt} "
                    f"(cfg {cfg_idx+1}/{len(noc_configs)}, task {task_idx+1}/{len(task_combinations)}) "
                    f"{baseline_name} lat*{lat_mult} bw*{bw_mult}: valid=0, "
                    f"filter={_format_duration(filter_elapsed_sec)}, combo={_format_duration(combo_elapsed_sec)}, "
                    f"elapsed={_format_duration(overall_elapsed_sec)}, eta={_format_duration(eta_sec)}"
                )
                continue

            tasks = []
            model_key = model_arch.__class__.__name__
            for scheme, tp_transform_moe in valid_parallel_schemes:
                minibatch, max_activation, global_total_weight, global_kv_cache, atten_parallel, moe_parallel, non_atten_non_moe_parallel = scheme_footprints[(scheme, tp_transform_moe)]
                tasks.append((
                    baseline_name, lat_mult, bw_mult,
                    model_key,
                    BS, minibatch, MAX_KV_LEN, PARALLEL_SEQ,
                    scheme,
                    max_activation, global_total_weight, global_kv_cache,
                    gran_tuple,
                    atten_parallel, moe_parallel, non_atten_non_moe_parallel,
                    proc_log_dir,
                    run_dir,
                    stats_bundle_root,
                    tp_transform_moe,
                ))

            rows = []
            pool_start_ts = time.time()
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(npz_trace_file,),
            ) as executor:
                futures = [executor.submit(_compute_time_row, t) for t in tasks]
                for fut in as_completed(futures):
                    try:
                        row = fut.result()
                        rows.append(row)
                    except Exception as e:
                        log.exception("Parallel task failed: %s", e)
            pool_elapsed_sec = time.time() - pool_start_ts

            if rows:
                rows.sort(key=lambda r: (r[3], r[4], str(r[9]), r[22]))
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    for r in rows:
                        scheme = r[9]
                        flat = list(meta_prefix) + [
                            r[3], r[4], r[5], r[6], r[7], r[8],
                            scheme.tp, scheme.ep, scheme.ep1, scheme.ep2, scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                            r[22],
                            r[10], r[11], r[12],
                            r[13] * 1000, r[14] * 1000, r[15] * 1000, r[16] * 1000,
                            r[17] * 1000, r[18] * 1000, r[19] * 1000,
                            r[20] * 1000, r[21] * 1000,
                            r[23], r[24],
                            r[25], r[26], r[27],
                            r[28], r[29],
                            r[30], r[31],
                            r[32],
                        ]
                        writer.writerow(flat)

            finished_combo_cnt += 1
            combo_elapsed_sec = time.time() - combo_start_ts
            combo_time_sum_sec += combo_elapsed_sec
            overall_elapsed_sec = time.time() - overall_start_ts
            avg_combo_sec = combo_time_sum_sec / max(finished_combo_cnt, 1)
            remain_combo = max(total_combo_cnt - finished_combo_cnt, 0)
            eta_sec = avg_combo_sec * remain_combo
            _log(
                f"[Progress] combo {finished_combo_cnt}/{total_combo_cnt} "
                f"(cfg {cfg_idx+1}/{len(noc_configs)}, task {task_idx+1}/{len(task_combinations)}) "
                f"{baseline_name} lat*{lat_mult} bw*{bw_mult}: "
                f"valid={len(valid_parallel_schemes)}, tasks={len(tasks)}, done={len(rows)}, "
                f"filter={_format_duration(filter_elapsed_sec)}, compute={_format_duration(pool_elapsed_sec)}, "
                f"combo={_format_duration(combo_elapsed_sec)}, elapsed={_format_duration(overall_elapsed_sec)}, "
                f"eta={_format_duration(eta_sec)}"
            )


if __name__ == "__main__":
    import logging, os, argparse
    from datetime import datetime
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an existing run directory to resume")
    parser.add_argument("--start-cfg-idx", type=int, default=0,
                        help="0-based config index to resume from (skip configs before this)")
    args = parser.parse_args()

    if args.resume:
        run_dir = args.resume
        start_cfg_idx = args.start_cfg_idx
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", run_tag)
        start_cfg_idx = 0

    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "dse.log")
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    mp.set_start_method("spawn", force=True)

    num_workers = os.cpu_count() // 2

    import time
    start_time = time.time()
    dse_1(run_dir, num_workers, enable_proc_log=False, start_cfg_idx=start_cfg_idx)
    print("dse_1 finished")
    end_time = time.time()
    print("dse_1 time: %s seconds" % (end_time - start_time))
