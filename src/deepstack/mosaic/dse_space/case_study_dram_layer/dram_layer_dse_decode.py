"""
dram_layer_dse_decode.py — DRAM-layer DSE for decode (DeepSeek R1 only)

Outer loop: generate_dram_layer_configs()  (m, n, smem_cap, l1_tp)
Inner loop: dram_layer_decode_tasks()      (model, BS, INPUT_SEQ, MAX_SEQ, PARALLEL_DECODING)

Reuses modeling_decode / get_max_footprint_decode / helpers from the original
decode dump-stats framework; adds per-config DRAM columns and power-wall columns.
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
import numpy as np
import csv
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time

from mosaic.utils import Modeling_Granularity
from mosaic.noc.noc_config_set import torus_mesh_switch_1
from .dram_layer_config import (
    generate_dram_layer_configs, make_arch_for_config, compute_thermal_freq_scale,
    apply_littles_law, is_l1_bound, get_actual_bw, compute_power_wall,
    reference_thermal_resistance,
)

from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
    get_max_footprint_decode, modeling_decode, _get_effective_moe_parallel,
    _format_duration, _dump_stats_bundle,
    MODEL_REG,
)

# tp_transform_moe: whether to convert MoE TP into EP before modeling
#   "none"         – keep original parallel scheme for MoE
#   "replace_only" – replace: tp=1, ep=ep*tp (better in practice)
TP_TRANSFORM_MOE_MODES: list[str] = ["replace_only"]

# ── sub-process global cache (avoid serialising numpy across processes) ──
_ROUTING_ARRAY = None


def _init_worker(npz_trace_file: str):
    """Each worker loads the routing array once at start-up."""
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
    global _ROUTING_ARRAY
    _, decode_array = load_npz_routing_keep_shape(npz_trace_file, as_list=False)
    _ROUTING_ARRAY = decode_array


# ── Task combinations (DeepSeek R1 only) ─────────────────────────────────

def dram_layer_decode_tasks():

    # tasks = dram_layer_decode_tasks_full()
    tasks = dram_layer_decode_tasks_short_kv_len()
    return tasks

def dram_layer_decode_tasks_full():
    tasks = []
    model = DeepSeekV3()
    PARALLEL_DECODING = 1
    for BS in [1, 4, 16, 64, 256, 1024, 4096, 8192]:
        for INPUT_SEQ, MAX_SEQ in [(1024, 2048), (8192, 16384), (65536, 131072)]:
            tasks.append((model, BS, INPUT_SEQ, MAX_SEQ, PARALLEL_DECODING))
    return tasks

def dram_layer_decode_tasks_short_kv_len():
    tasks = []
    model = DeepSeekV3()
    PARALLEL_DECODING = 1
    for BS in [1, 4, 16, 64, 256, 1024, 4096, 16384]:
        for INPUT_SEQ, MAX_SEQ in [(1024, 2048)]:
            tasks.append((model, BS, INPUT_SEQ, MAX_SEQ, PARALLEL_DECODING))
    return tasks


# ── Helpers ───────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return slug.strip("-") or "na"


def _make_stats_run_id(
    arch_name: str,
    noc_name: str,
    model_name: str,
    bs: int,
    minibatch: int,
    seq: int,
    parallel_scheme: ParallelScheme,
    tp_transform_moe: str,
    m: int,
    n: int,
    smem: int,
    l1: int,
) -> str:
    key = (
        f"arch={arch_name}|noc={noc_name}|model={model_name}|bs={bs}|minibatch={minibatch}|seq={seq}|"
        f"tp={parallel_scheme.tp}|ep={parallel_scheme.ep}|ep1={parallel_scheme.ep1}|ep2={parallel_scheme.ep2}|"
        f"sp={parallel_scheme.sp}|cp={parallel_scheme.cp}|dp={parallel_scheme.dp}|fsdp={parallel_scheme.fsdp}|"
        f"pp={parallel_scheme.pp}|tp_transform_moe={tp_transform_moe}|"
        f"m={m}|n={n}|smem={smem}|l1={l1}"
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{_slugify(model_name)}_{_slugify(arch_name)}_{_slugify(noc_name)}_{digest}"


# ── Worker ────────────────────────────────────────────────────────────────

def _compute_time_row(args):
    (
        m, n, smem_cap, l1_tp, freq_scale_thermal,
        model_key,
        BS, minibatch, sampled_kv_len, seq,
        parallel_scheme,
        max_activation, global_total_weight, global_kv_cache,
        granularity_tuple,
        non_moe_parallel,
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

    # Rebuild arch/noc in the worker
    arch, _, _, _, _ = make_arch_for_config(m, n, smem_cap, l1_tp)
    noc = torus_mesh_switch_1()
    model_arch = MODEL_REG[model_key]()

    # Rebuild granularity
    granularity = Modeling_Granularity(
        mode=granularity_tuple[0],
        comp_comm_overlap=granularity_tuple[1],
        auto_tune=granularity_tuple[2],
        dump_perf_log=granularity_tuple[3],
    )

    # Routing array from sub-process global
    global _ROUTING_ARRAY
    routing_array = _ROUTING_ARRAY
    if routing_array is None:
        raise RuntimeError("routing_array not initialized in worker")

    time_list = []
    sampled_kv_len_utps_stps_list = []
    utps_all = 0
    stps_all = 0

    time_list, model_stats, representative_kv_len = modeling_decode(
        model_arch=model_arch, bs=minibatch, seq=seq, cached_kv_list=sampled_kv_len,
        moe_parallel=parallel_scheme, non_moe_parallel=non_moe_parallel,
        single_chip=arch, noc_hierarchy=noc, granularity=granularity,
        routing_array=routing_array, tp_transform_moe=tp_transform_moe,
    )

    for i in range(len(sampled_kv_len)):
        pp_p2p_time, time_gqa, time_mla, time_dense_ffn, time_moe, time_rms_norm, time_add_residual, time_total = time_list[i]
        kv_len = sampled_kv_len[i]
        stps = minibatch / time_total
        utps = 1 / time_total / parallel_scheme.pp
        utps_all += utps
        stps_all += stps
        sampled_kv_len_utps_stps_list.append((kv_len, time_gqa, time_mla, utps, stps))

    utps_average = utps_all / len(sampled_kv_len)
    stps_average = stps_all / len(sampled_kv_len)

    # ---- Power wall computation ----
    # chip_total_energy_j and noc_total_energy_j are totals across ALL devices
    # → must divide by num_devices to get per-device power
    middle_idx = len(sampled_kv_len) // 2
    _, _, _, _, _, _, _, e2e_time = time_list[middle_idx]
    num_devices = parallel_scheme.world_size()
    chip_power_per_device_w = model_stats.chip_total_energy_j / e2e_time / num_devices if e2e_time > 0 else 0
    noc_power_per_device_w = model_stats.noc_total_energy_j / e2e_time / num_devices if e2e_time > 0 else 0
    total_power_per_device_w = chip_power_per_device_w + noc_power_per_device_w
    hit_power_wall, freq_scale_power = compute_power_wall(chip_power_per_device_w, noc_power_per_device_w)
    total_freq_scale = freq_scale_thermal * freq_scale_power
    scaled_utps_avg = utps_average * freq_scale_power
    scaled_stps_avg = stps_average * freq_scale_power

    arch_name = arch.__class__.__name__
    noc_name = noc.name
    model_name = model_arch.__class__.__name__
    stats_run_id = _make_stats_run_id(
        arch_name=arch_name,
        noc_name=noc_name,
        model_name=model_name,
        bs=BS,
        minibatch=minibatch,
        seq=seq,
        parallel_scheme=parallel_scheme,
        tp_transform_moe=tp_transform_moe,
        m=m, n=n, smem=smem_cap, l1=l1_tp,
    )
    proc_log_relpath = ""
    if proc_log_file:
        proc_log_relpath = os.path.relpath(proc_log_file, run_dir)
    stats_bundle_relpath, manifest_relpath = _dump_stats_bundle(
        stats_bundle_root=stats_bundle_root,
        run_dir=run_dir,
        stats_run_id=stats_run_id,
        representative_kv_len=representative_kv_len,
        model_stats=model_stats,
        metadata={
            "arch": arch_name,
            "noc": noc_name,
            "model": model_name,
            "bs": BS,
            "minibatch": minibatch,
            "seq": seq,
            "parallel_scheme": dataclasses.asdict(parallel_scheme),
            "tp_transform_moe": tp_transform_moe,
            "dram_total_layers": m,
            "dram_active_layers": n,
            "smem_capacity": smem_cap,
            "l1_throughput": l1_tp,
            "proc_log_relpath": proc_log_relpath,
        },
    )

    return [
        # DRAM config
        m, n, smem_cap, l1_tp, freq_scale_thermal,
        # original fields
        arch_name, noc_name, model_name,
        BS, minibatch, seq,
        parallel_scheme,
        max_activation / (1024**3),
        global_total_weight / (1024**3),
        global_kv_cache / (1024**3),
        sampled_kv_len_utps_stps_list,
        pp_p2p_time, time_dense_ffn, time_moe, time_rms_norm, time_add_residual,
        utps_average, stps_average,
        tp_transform_moe,
        # power/perf
        chip_power_per_device_w, noc_power_per_device_w, total_power_per_device_w,
        hit_power_wall, freq_scale_power, total_freq_scale,
        scaled_utps_avg, scaled_stps_avg,
        stats_run_id,
    ]


# ── Main DSE loop ─────────────────────────────────────────────────────────

def dse_1(run_dir, num_workers, enable_proc_log=False, start_dram_cfg_idx=0):
    """
    Outer loop: DRAM configs  (m, n, smem_cap, l1_tp)
    Inner loop: task combos   (model, BS, INPUT_SEQ, MAX_SEQ, PARALLEL_DECODING)

    For each (config, task), enumerate parallel schemes, filter by footprint,
    dispatch to worker pool, collect results and write a single CSV.

    start_dram_cfg_idx: skip dram configs before this index (0-based) for resuming.
    """
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False, dump_perf_log=True)
    gran_tuple = (granularity.mode, granularity.comp_comm_overlap, granularity.auto_tune, granularity.dump_perf_log)
    run_dir = os.path.abspath(run_dir)

    # ---- progress log: print + write to file ----
    _progress_log_path = os.path.join(run_dir, "progress.log")
    def _log(msg):
        print(msg, flush=True)
        with open(_progress_log_path, "a") as _f:
            _f.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")

    proc_log_dir = os.path.join(run_dir, "proc_logs") if enable_proc_log else None
    stats_bundle_root = os.path.join(run_dir, "stats_bundles")
    os.makedirs(stats_bundle_root, exist_ok=True)

    NUM_KV_POINT = 4

    task_combinations = dram_layer_decode_tasks()
    dram_configs = generate_dram_layer_configs()

    # ---- CSV paths ----
    csv_path = os.path.join(run_dir, "dram_layer_decode_result.csv")
    invalid_csv_path = os.path.join(run_dir, "dram_layer_decode_invalid.csv")

    # ---- CSV header ----
    header = [
        # DRAM config
        "dram_total_layers", "dram_active_layers",
        "sm_count",
        "smem_capacity_KiB", "l1_throughput_Bpc",
        "ddr_peak_bw_TBs", "ddr_eff_bw_TBs", "ddr_capacity_GB",
        "littles_law_required_buf_KiB", "littles_law_limited", "is_l1_bound",
        "thermal_resistance_CpW",
        "freq_scale_thermal",
        # original columns
        "arch", "noc", "model", "bs", "minibatch", "seq",
        "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
        "tp_transform_moe",
        "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB",
    ]
    for i in range(NUM_KV_POINT):
        idx = i + 1
        header.extend([f"kv_len_{idx}", f"time_gqa_{idx}ms", f"time_mla_{idx}ms", f"utps_{idx}", f"stps_{idx}"])
    header.extend([
        "pp_p2p_time_ms", "time_dense_ffn_ms", "time_moe_ms", "time_rms_norm_ms", "time_add_residual_ms",
        "raw_utps_avg", "raw_stps_avg",
        "chip_power_W", "noc_power_per_device_W", "total_power_per_device_W",
        "hit_power_wall", "freq_scale_power", "total_freq_scale",
        "scaled_utps_avg", "scaled_stps_avg",
        "stats_run_id",
    ])

    if start_dram_cfg_idx == 0:
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(header)
        with open(invalid_csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow([
                "dram_total_layers", "dram_active_layers",
                "sm_count", "smem_capacity_KiB", "l1_throughput_Bpc",
                "arch", "noc", "model", "bs", "minibatch", "seq", "cached_kv",
                "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
                "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB",
            ])
    else:
        if (not os.path.exists(csv_path)) or os.stat(csv_path).st_size == 0:
            with open(csv_path, mode="w", newline="") as f:
                csv.writer(f).writerow(header)
        if (not os.path.exists(invalid_csv_path)) or os.stat(invalid_csv_path).st_size == 0:
            with open(invalid_csv_path, mode="w", newline="") as f:
                csv.writer(f).writerow([
                    "dram_total_layers", "dram_active_layers",
                    "sm_count", "smem_capacity_KiB", "l1_throughput_Bpc",
                    "arch", "noc", "model", "bs", "minibatch", "seq", "cached_kv",
                    "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
                    "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB",
                ])

    total_combo_cnt = (len(dram_configs) - start_dram_cfg_idx) * len(task_combinations)
    finished_combo_cnt = 0
    combo_time_sum_sec = 0.0
    overall_start_ts = time.time()

    _log(f"[DECODE DSE] dram_configs={len(dram_configs)}, tasks={len(task_combinations)}, "
         f"total_combos={total_combo_cnt}, workers={num_workers}")

    # ---- Outer loop: DRAM configs ----
    for cfg_idx, (m, n, smem_cap, l1_tp) in enumerate(dram_configs):
        if cfg_idx < start_dram_cfg_idx:
            continue
        # Build arch once per config (for footprint filtering)
        arch, freq_scale_thermal, littles_law_limited, required_buf, sm_count = make_arch_for_config(m, n, smem_cap, l1_tp)
        noc = torus_mesh_switch_1()

        # Compute aggregate thermal information without exposing die area.
        thermal_resistance = reference_thermal_resistance(m)
        l1_bound = is_l1_bound(arch)

        ddr_peak_bw_tbs = arch.ddr_peak_bandwidth / 1e12 if hasattr(arch, 'ddr_peak_bandwidth') else arch.ddr_bandwidth / 1e12
        ddr_eff_bw_tbs = arch.ddr_bandwidth / 1e12
        ddr_capacity_gb = arch.ddr_capacity / (1024**3)

        num_nodes = int(noc.num_devices)
        parallel_schemes = all_parallel_schemes(num_nodes)
        parallel_schemes = filter_illegal_fsdp(parallel_schemes)

        # ---- Inner loop: task combinations ----
        for task_idx, task_combination in enumerate(task_combinations):
            combo_start_ts = time.time()

            model_arch = task_combination[0]
            BS = task_combination[1]
            INPUT_SEQ = task_combination[2]
            TASK_MAX_SEQ = task_combination[3]
            PARALLEL_SEQ = task_combination[4]

            # ---- routing trace npz ----
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent.parent  # .../mosaic
            if model_arch.__class__.__name__ in ("DeepSeekV3", "DeepSeekV3_A8W8"):
                npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")
            elif model_arch.__class__.__name__ == "Qwen3_235b_a22b":
                npz_trace_file = str(project_root / "data" / "aime_qwen_235b" / "qwen3_moe_activations_batch0.npz")
            else:
                npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")

            MAX_KV_LEN = min(TASK_MAX_SEQ, model_arch.max_seq_len)

            assert NUM_KV_POINT >= 2
            if NUM_KV_POINT == 2:
                sampled_kv_len = (INPUT_SEQ, MAX_KV_LEN)
            else:
                step = (MAX_KV_LEN - INPUT_SEQ) / (NUM_KV_POINT - 1)
                sampled_kv_len = tuple(int(round(INPUT_SEQ + i * step)) for i in range(NUM_KV_POINT))

            # Filter parallel schemes for this task
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
            _total_schemes = len(filtered_parallel_schemes)
            for _si, scheme in enumerate(filtered_parallel_schemes):
                if _si % 200 == 0 or (time.time() - filter_start_ts) > 60:
                    print(f"  [filter] scheme {_si}/{_total_schemes}, "
                          f"elapsed={_format_duration(time.time()-filter_start_ts)}, "
                          f"tp={scheme.tp} ep={scheme.ep} sp={scheme.sp} cp={scheme.cp} dp={scheme.dp} pp={scheme.pp}",
                          flush=True)
                minibatch = math.ceil(BS / scheme.pp)
                allocate_ep(parallel=scheme, bs=minibatch, seq=PARALLEL_SEQ)

                non_moe_parallel = dataclasses.replace(
                    scheme, ep=1, ep1=1, ep2=1, dp=scheme.dp * scheme.ep
                )

                for tp_transform_moe in TP_TRANSFORM_MOE_MODES:
                    max_activation, global_total_weight, global_kv_cache = get_max_footprint_decode(
                        model_arch=model_arch, bs=minibatch, seq=PARALLEL_SEQ, cached_kv=MAX_KV_LEN,
                        moe_parallel=scheme, non_moe_parallel=non_moe_parallel,
                        tp_transform_moe=tp_transform_moe,
                    )

                    if (max_activation + global_total_weight + global_kv_cache) <= 1.0 * arch.ddr_capacity:
                        log.info("parallel scheme: %s, tp_transform_moe: %s", scheme, tp_transform_moe)
                        log.info("valid")
                        valid_parallel_schemes.append((scheme, tp_transform_moe))
                        scheme_footprints[(scheme, tp_transform_moe)] = (minibatch, max_activation, global_total_weight, global_kv_cache, non_moe_parallel)
                    else:
                        log.info("parallel scheme: %s, tp_transform_moe: %s — invalid", scheme, tp_transform_moe)
                        with open(invalid_csv_path, mode="a", newline="") as f:
                            csv.writer(f).writerow([
                                m, n, sm_count,
                                smem_cap // 1024, l1_tp,
                                arch.__class__.__name__, noc.name, model_arch.__class__.__name__,
                                BS, minibatch, 1, MAX_KV_LEN,
                                scheme.tp, scheme.ep, scheme.ep1, scheme.ep2, scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                                max_activation / (1024**3), global_total_weight / (1024**3), global_kv_cache / (1024**3),
                            ])
            filter_elapsed_sec = time.time() - filter_start_ts

            log.info("cfg (%d,%d,%d,%d) model_arch: %s, bs: %s — valid schemes: %d",
                     m, n, smem_cap, l1_tp, model_arch.__class__.__name__, BS, len(valid_parallel_schemes))

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
                    f"(cfg {cfg_idx+1}/{len(dram_configs)}, task {task_idx+1}/{len(task_combinations)}) "
                    f"m={m},n={n}: valid=0, "
                    f"filter={_format_duration(filter_elapsed_sec)}, combo={_format_duration(combo_elapsed_sec)}, "
                    f"elapsed={_format_duration(overall_elapsed_sec)}, eta={_format_duration(eta_sec)}"
                )
                continue

            # ---- Build worker tasks ----
            tasks = []
            model_key = model_arch.__class__.__name__
            for scheme, tp_transform_moe in valid_parallel_schemes:
                minibatch, max_activation, global_total_weight, global_kv_cache, non_moe_parallel = scheme_footprints[(scheme, tp_transform_moe)]
                tasks.append((
                    m, n, smem_cap, l1_tp, freq_scale_thermal,
                    model_key,
                    BS, minibatch, sampled_kv_len, PARALLEL_SEQ,
                    scheme,
                    max_activation, global_total_weight, global_kv_cache,
                    gran_tuple,
                    non_moe_parallel,
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

            # ---- Write CSV rows ----
            if rows:
                # Sort by arch, noc, scheme, tp_transform_moe
                rows.sort(key=lambda r: (r[5], r[6], str(r[11]), r[22]))
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    flat_rows = []
                    for r in rows:
                        # r layout:
                        #  0: m, 1: n, 2: smem_cap, 3: l1_tp, 4: freq_scale_thermal,
                        #  5: arch_name, 6: noc_name, 7: model_name,
                        #  8: BS, 9: minibatch, 10: seq,
                        # 11: parallel_scheme,
                        # 12: max_act/GiB, 13: weight/GiB, 14: kv/GiB,
                        # 15: sampled_kv_list,
                        # 16: pp_p2p_time, 17: time_dense_ffn, 18: time_moe, 19: time_rms_norm, 20: time_add_residual,
                        # 21: utps_avg, 22: stps_avg, 23: tp_transform_moe,
                        # 24: chip_power_per_device_w, 25: noc_power_per_device_w, 26: total_power_per_device_w,
                        # 27: hit_power_wall, 28: freq_scale_power, 29: total_freq_scale,
                        # 30: scaled_utps_avg, 31: scaled_stps_avg,
                        # 32: stats_run_id
                        scheme = r[11]
                        base = [
                            # DRAM config columns
                            r[0], r[1],                                     # m, n
                            sm_count,                                       # sm_count
                            r[2] // 1024, r[3],                             # smem KiB, l1 Bpc
                            ddr_peak_bw_tbs, ddr_eff_bw_tbs, ddr_capacity_gb,
                            required_buf / 1024, littles_law_limited, l1_bound,
                            thermal_resistance,
                            r[4],
                            # original columns
                            r[5], r[6], r[7], r[8], r[9], r[10],
                            scheme.tp, scheme.ep, scheme.ep1, scheme.ep2, scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                            r[23],                                          # tp_transform_moe
                            r[12], r[13], r[14],                            # activation, weight, kv (GiB)
                        ]
                        # KV point columns
                        for kv_len, time_gqa, time_mla, utps, stps in r[15]:
                            base.extend([kv_len, time_gqa * 1000, time_mla * 1000, utps, stps])
                        # Timing columns (ms)
                        base.extend([
                            r[16] * 1000, r[17] * 1000, r[18] * 1000, r[19] * 1000, r[20] * 1000,
                            r[21], r[22],                                   # raw utps/stps avg
                        ])
                        # Power/perf columns
                        base.extend([
                            r[24], r[25], r[26],                            # chip/noc/total power W
                            r[27], r[28], r[29],                            # hit_wall, freq_scale_power, total_freq_scale
                            r[30], r[31],                                   # scaled utps/stps avg
                            r[32],                                          # stats_run_id
                        ])
                        flat_rows.append(base)
                    writer.writerows(flat_rows)

            finished_combo_cnt += 1
            combo_elapsed_sec = time.time() - combo_start_ts
            combo_time_sum_sec += combo_elapsed_sec
            overall_elapsed_sec = time.time() - overall_start_ts
            avg_combo_sec = combo_time_sum_sec / max(finished_combo_cnt, 1)
            remain_combo = max(total_combo_cnt - finished_combo_cnt, 0)
            eta_sec = avg_combo_sec * remain_combo
            _log(
                f"[Progress] combo {finished_combo_cnt}/{total_combo_cnt} "
                f"(cfg {cfg_idx+1}/{len(dram_configs)}, task {task_idx+1}/{len(task_combinations)}) "
                f"m={m},n={n},smem={smem_cap//1024}K,l1={l1_tp}: "
                f"valid={len(valid_parallel_schemes)}, tasks={len(tasks)}, done={len(rows)}, "
                f"filter={_format_duration(filter_elapsed_sec)}, "
                f"compute={_format_duration(pool_elapsed_sec)}, combo={_format_duration(combo_elapsed_sec)}, "
                f"elapsed={_format_duration(overall_elapsed_sec)}, eta={_format_duration(eta_sec)}"
            )


if __name__ == "__main__":
    import argparse
    import logging
    import os
    from datetime import datetime
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing run directory to resume",
    )
    parser.add_argument(
        "--start-dram-cfg-idx",
        type=int,
        default=0,
        help="0-based DRAM configuration index to resume from",
    )
    args = parser.parse_args()

    if args.resume is not None:
        run_dir = args.resume
        start_dram_cfg_idx = args.start_dram_cfg_idx
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", run_tag)
        start_dram_cfg_idx = 0
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
    log.info("Using %d workers (cpu_count=%s)", num_workers, os.cpu_count())

    import time
    start_time = time.time()
    dse_1(
        run_dir,
        num_workers,
        enable_proc_log=False,
        start_dram_cfg_idx=start_dram_cfg_idx,
    )
    print("dse_1 finished")
    end_time = time.time()
    print("dse_1 time: %s seconds" % (end_time - start_time))
