"""
dram_layer_dse_decode_fixed_parallel.py — Fixed-parallel DSE for decode

Fixed parallel configs (from bw=1x, lat=1x best):
  BS=4:    TP=32  EP=8   DP=1 PP=1
  BS=64:   TP=16  EP=16  DP=1 PP=1
  BS=1024: TP=4   EP=32  DP=1 PP=2

Sweep: bw × latency (no parallel search)
  BW:      0.5, 0.75, 1.0, 1.25, 1.5, 2.0
  Latency: 0.25, 0.5, 1.0, 2.0, 4.0
  BS:      4, 64, 1024
"""

from typing import Any
from mosaic.llm_arch import DeepSeekV3, LLM_Arch
from mosaic.parallelism import ParallelScheme
from mosaic.utils.allocate_ep import allocate_ep
import logging
log = logging.getLogger(__name__)
import math
import dataclasses
import hashlib
import re
import numpy as np
import csv
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time

from mosaic.utils import Modeling_Granularity
from .lat_bw_config import (
    make_noc_for_config, make_arch_for_noc_config,
    compute_power_wall, TDP_W,
    BASELINES,
)

from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
    get_max_footprint_decode, modeling_decode, _get_effective_moe_parallel,
    _format_duration, _dump_stats_bundle,
    MODEL_REG,
)

# ── Fixed configs ─────────────────────────────────────────────────────────

FIXED_PARALLEL = {
    4:    ParallelScheme(tp=32,  ep=8,  ep1=1, ep2=8,  sp=1, cp=1, dp=1, fsdp=False, pp=1),
    64:   ParallelScheme(tp=16,  ep=16, ep1=1, ep2=16, sp=1, cp=1, dp=1, fsdp=False, pp=1),
    1024: ParallelScheme(tp=4,   ep=32, ep1=1, ep2=32, sp=1, cp=1, dp=1, fsdp=False, pp=2),
}

TARGET_BS = [4, 64, 1024]
BW_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
LATENCY_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]

TP_TRANSFORM_MOE_MODES: list[str | None] = [None, "replace_only"]

_ROUTING_ARRAY = None


def _init_worker(npz_trace_file: str):
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
    global _ROUTING_ARRAY
    _, decode_array = load_npz_routing_keep_shape(npz_trace_file, as_list=False)
    _ROUTING_ARRAY = decode_array


# ── Task combinations ─────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return slug.strip("-") or "na"


def _make_stats_run_id(
    arch_name, noc_name, model_name,
    bs, minibatch, seq,
    parallel_scheme, tp_transform_moe,
    baseline_name, lat_mult, bw_mult,
):
    key = (
        f"arch={arch_name}|noc={noc_name}|model={model_name}|bs={bs}|minibatch={minibatch}|seq={seq}|"
        f"tp={parallel_scheme.tp}|ep={parallel_scheme.ep}|ep1={parallel_scheme.ep1}|ep2={parallel_scheme.ep2}|"
        f"sp={parallel_scheme.sp}|cp={parallel_scheme.cp}|dp={parallel_scheme.dp}|fsdp={parallel_scheme.fsdp}|"
        f"pp={parallel_scheme.pp}|tp_transform_moe={tp_transform_moe}|"
        f"baseline={baseline_name}|lat_mult={lat_mult}|bw_mult={bw_mult}"
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{_slugify(model_name)}_{_slugify(arch_name)}_{_slugify(noc_name)}_{digest}"


# ── Worker ────────────────────────────────────────────────────────────────

def _compute_time_row(args):
    (
        baseline_name, lat_mult, bw_mult,
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

    middle_idx = len(sampled_kv_len) // 2
    _, _, _, _, _, _, _, e2e_time = time_list[middle_idx]
    num_devices = parallel_scheme.world_size()
    chip_power_per_device_w = model_stats.chip_total_energy_j / e2e_time / num_devices if e2e_time > 0 else 0
    noc_power_per_device_w = model_stats.noc_total_energy_j / e2e_time / num_devices if e2e_time > 0 else 0
    total_power_per_device_w = chip_power_per_device_w + noc_power_per_device_w
    hit_power_wall, freq_scale_power = compute_power_wall(chip_power_per_device_w, noc_power_per_device_w)
    scaled_utps_avg = utps_average * freq_scale_power
    scaled_stps_avg = stps_average * freq_scale_power

    arch_name = arch.__class__.__name__
    noc_name = noc.name
    model_name = model_arch.__class__.__name__
    stats_run_id = _make_stats_run_id(
        arch_name=arch_name, noc_name=noc_name, model_name=model_name,
        bs=BS, minibatch=minibatch, seq=seq,
        parallel_scheme=parallel_scheme, tp_transform_moe=tp_transform_moe,
        baseline_name=baseline_name, lat_mult=lat_mult, bw_mult=bw_mult,
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
            "arch": arch_name, "noc": noc_name, "model": model_name,
            "bs": BS, "minibatch": minibatch, "seq": seq,
            "parallel_scheme": dataclasses.asdict(parallel_scheme),
            "tp_transform_moe": tp_transform_moe,
            "baseline_noc": baseline_name,
            "latency_multiplier": lat_mult,
            "bw_multiplier": bw_mult,
            "proc_log_relpath": proc_log_relpath,
        },
    )

    return [
        baseline_name, lat_mult, bw_mult,
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
        chip_power_per_device_w, noc_power_per_device_w, total_power_per_device_w,
        hit_power_wall, freq_scale_power,
        scaled_utps_avg, scaled_stps_avg,
        stats_run_id,
    ]


# ── Main DSE loop ─────────────────────────────────────────────────────────

def dse_1(run_dir, num_workers, enable_proc_log=False):
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

    NUM_KV_POINT = 4

    # Build (baseline, lat, bw) configs
    noc_configs = []
    for baseline_name in BASELINES:
        for lat_mult in LATENCY_MULTIPLIERS:
            for bw_mult in BW_MULTIPLIERS:
                noc_configs.append((baseline_name, lat_mult, bw_mult))

    # Build task combos: (model, BS, INPUT_SEQ, MAX_SEQ, PARALLEL_DECODING)
    model = DeepSeekV3()
    task_combinations = []
    for BS in TARGET_BS:
        task_combinations.append((model, BS, 1024, 2048, 1))

    csv_path = os.path.join(run_dir, "lat_bw_decode_result.csv")
    invalid_csv_path = os.path.join(run_dir, "lat_bw_decode_invalid.csv")

    header = [
        "baseline_noc", "latency_multiplier", "bw_multiplier",
        "sm_count",
        "l1_link_bw_GBs", "l2_link_bw_GBs", "l3_link_bw_GBs",
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
        "hit_power_wall", "freq_scale_power",
        "scaled_utps_avg", "scaled_stps_avg",
        "stats_run_id",
    ])

    if (not os.path.exists(csv_path)) or os.stat(csv_path).st_size == 0:
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(header)
    if (not os.path.exists(invalid_csv_path)) or os.stat(invalid_csv_path).st_size == 0:
        with open(invalid_csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow([
                "baseline_noc", "latency_multiplier", "bw_multiplier", "sm_count",
                "arch", "noc", "model", "bs", "minibatch", "seq", "cached_kv",
                "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
                "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB",
            ])

    _log(f"[FIXED-PARALLEL DECODE DSE] noc_configs={len(noc_configs)}, tasks={len(task_combinations)}, "
         f"workers={num_workers}")
    _log(f"  BW mults: {BW_MULTIPLIERS}")
    _log(f"  Lat mults: {LATENCY_MULTIPLIERS}")
    _log(f"  BS: {TARGET_BS}")
    _log(f"  tp_transform_moe modes: {TP_TRANSFORM_MOE_MODES}")
    _log(f"  Fixed parallel: { {bs: f'TP={p.tp} EP={p.ep} DP={p.dp} PP={p.pp}' for bs, p in FIXED_PARALLEL.items()} }")

    # ── Phase 1: collect all tasks upfront ────────────────────────────────
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")

    all_tasks = []        # list of worker arg tuples
    task_meta = []        # parallel metadata for CSV writing (SM count and per-layer BW/latency)

    _log("Collecting tasks ...")
    for cfg_idx, (baseline_name, lat_mult, bw_mult) in enumerate(noc_configs):
        noc = make_noc_for_config(baseline_name, lat_mult, bw_mult)
        arch, sm_count = make_arch_for_noc_config(noc)

        l1_link_bw_gbs = noc.layers[-1].link_bandwidth / 1e9
        l2_link_bw_gbs = noc.layers[-2].link_bandwidth / 1e9
        l3_link_bw_gbs = noc.layers[-3].link_bandwidth / 1e9

        meta = (sm_count,
                l1_link_bw_gbs, l2_link_bw_gbs, l3_link_bw_gbs)

        for task_combination in task_combinations:
            model_arch = task_combination[0]
            BS = task_combination[1]
            INPUT_SEQ = task_combination[2]
            TASK_MAX_SEQ = task_combination[3]
            PARALLEL_SEQ = task_combination[4]

            MAX_KV_LEN = min(TASK_MAX_SEQ, model_arch.max_seq_len)

            assert NUM_KV_POINT >= 2
            if NUM_KV_POINT == 2:
                sampled_kv_len = (INPUT_SEQ, MAX_KV_LEN)
            else:
                step = (MAX_KV_LEN - INPUT_SEQ) / (NUM_KV_POINT - 1)
                sampled_kv_len = tuple(int(round(INPUT_SEQ + i * step)) for i in range(NUM_KV_POINT))

            scheme = FIXED_PARALLEL[BS]
            minibatch = math.ceil(BS / scheme.pp)
            allocate_ep(parallel=scheme, bs=minibatch, seq=PARALLEL_SEQ)

            non_moe_parallel = dataclasses.replace(
                scheme, ep=1, ep1=1, ep2=1, dp=scheme.dp * scheme.ep
            )

            model_key = model_arch.__class__.__name__
            for tp_transform_moe in TP_TRANSFORM_MOE_MODES:
                max_activation, global_total_weight, global_kv_cache = get_max_footprint_decode(
                    model_arch=model_arch, bs=minibatch, seq=PARALLEL_SEQ, cached_kv=MAX_KV_LEN,
                    moe_parallel=scheme, non_moe_parallel=non_moe_parallel,
                    tp_transform_moe=tp_transform_moe,
                )

                if (max_activation + global_total_weight + global_kv_cache) > 1.0 * arch.ddr_capacity:
                    with open(invalid_csv_path, mode="a", newline="") as f:
                        csv.writer(f).writerow([
                            baseline_name, lat_mult, bw_mult, sm_count,
                            arch.__class__.__name__, noc.name, model_arch.__class__.__name__,
                            BS, minibatch, 1, MAX_KV_LEN,
                            scheme.tp, scheme.ep, scheme.ep1, scheme.ep2, scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                            max_activation / (1024**3), global_total_weight / (1024**3), global_kv_cache / (1024**3),
                        ])
                    continue

                all_tasks.append((
                    baseline_name, lat_mult, bw_mult,
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
                task_meta.append(meta)

    _log(f"Collected {len(all_tasks)} valid tasks. Launching pool with {num_workers} workers ...")

    # ── Phase 2: run all tasks in one big pool ────────────────────────────
    overall_start_ts = time.time()
    done_cnt = 0
    total_cnt = len(all_tasks)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(npz_trace_file,),
    ) as executor:
        future_to_idx = {executor.submit(_compute_time_row, t): i
                         for i, t in enumerate(all_tasks)}

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            done_cnt += 1
            try:
                r = fut.result()
            except Exception as e:
                log.exception("Task %d failed: %s", idx, e)
                continue

            # Write row to CSV immediately
            (sm_count,
             l1_link_bw_gbs, l2_link_bw_gbs, l3_link_bw_gbs) = task_meta[idx]

            scheme_r = r[9]
            base = [
                r[0], r[1], r[2],
                sm_count,
                l1_link_bw_gbs, l2_link_bw_gbs, l3_link_bw_gbs,
                r[3], r[4], r[5], r[6], r[7], r[8],
                scheme_r.tp, scheme_r.ep, scheme_r.ep1, scheme_r.ep2, scheme_r.sp, scheme_r.cp, scheme_r.dp, scheme_r.fsdp, scheme_r.pp,
                r[21],
                r[10], r[11], r[12],
            ]
            for kv_len, time_gqa, time_mla, utps, stps in r[13]:
                base.extend([kv_len, time_gqa * 1000, time_mla * 1000, utps, stps])
            base.extend([
                r[14] * 1000, r[15] * 1000, r[16] * 1000, r[17] * 1000, r[18] * 1000,
                r[19], r[20],
            ])
            base.extend([
                r[22], r[23], r[24],
                r[25], r[26],
                r[27], r[28],
                r[29],
            ])
            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow(base)

            # Progress log every 10 tasks or at the end
            if done_cnt % 10 == 0 or done_cnt == total_cnt:
                elapsed = time.time() - overall_start_ts
                avg = elapsed / done_cnt
                eta = avg * (total_cnt - done_cnt)
                _log(f"[Progress] {done_cnt}/{total_cnt} done, "
                     f"elapsed={_format_duration(elapsed)}, eta={_format_duration(eta)}")

    total_elapsed = time.time() - overall_start_ts
    _log(f"All {total_cnt} tasks done in {_format_duration(total_elapsed)}")


if __name__ == "__main__":
    import logging, os
    from datetime import datetime
    import multiprocessing as mp

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_fixed_parallel"
    run_dir = os.path.join("runs", run_tag)
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
    dse_1(run_dir, num_workers, enable_proc_log=False)
    print("dse_1 finished")
    end_time = time.time()
    print("dse_1 time: %s seconds" % (end_time - start_time))
