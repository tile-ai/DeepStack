"""
Single-task version of dse_framework_multi_process_v4_decode_dump_stats.py

直接在 __main__ 里修改参数即可运行：
  python -m mosaic.dse_space.dse_single_task_decode
"""

from mosaic.dse_space.parallel_schemes import (
    all_parallel_schemes,
    filter_parallel_schemes_by_max,
    filter_illegal_fsdp,
    filter_parallel_schemes_by_product_max,
)
from mosaic.llm_arch import (
    DeepSeekV3, LLM_Arch, Qwen3_235b_a22b, Qwen3_480b_a35b,
    Llama3_70b, Llama3_405b, DeepSeekV3_A8W8,
)
from mosaic.parallelism import ParallelScheme
from mosaic.utils.allocate_ep import allocate_ep
import logging
log = logging.getLogger(__name__)
import math
import dataclasses
import numpy as np
import csv
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
from mosaic.utils import Modeling_Granularity
from mosaic.noc.noc_topo import Hierarchy
from tilesight.arch.arch_base import Arch

# Reuse core functions from the original module
from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
    MODEL_REG,
    GET_ARCH_NOC_COMBINATIONS,
    TP_TRANSFORM_MOE_MODES,
    _format_duration,
    _init_worker,
    _make_stats_run_id,
    _dump_stats_bundle,
    _get_effective_moe_parallel,
    get_max_footprint_decode,
    modeling_decode,
    _compute_time_row,
)

# arch classes
from mosaic.arch import (
    stacked_gpu_base, stacked_gpu_large_matrix, stacked_gpu_large_vector, stacked_gpu_high_l1,
    stacked_gpu_high_l2, stacked_gpu_high_noc, stacked_gpu_low_noc, stacked_gpu_reduced_sm,
    stacked_gpu_wgmma, H100_SCALED, H200_SCALED, H100, H200, B200,
)
# noc configs
from mosaic.noc.noc_config_set import (
    torus_mesh_switch_1, torus_mesh_switch_2, torus_mesh_mesh_3,
    strong_torus_mesh_switch_4, weak_torus_mesh_switch_5,
    torus_mesh_switch_7, torus_mesh_switch_8, torus_mesh_switch_9,
    h200x8, h100x8, h100x32_strong, h100x32_medium,
    h100_8x1_8, h100_8x2_16, h100_8x4_32, h100_8x8_64,
    stacked_gpu_4x4, stacked_gpu_4x8, stacked_gpu_4x16, stacked_gpu_8x8, stacked_gpu_8x16,
    b200_8x1_8,
)


NUM_KV_POINT = 4


def _find_combo_idx(arch: Arch, noc: Hierarchy) -> int:
    """在 GET_ARCH_NOC_COMBINATIONS() 中查找匹配的 (arch_class, noc_name) 索引。

    _compute_time_row 在子进程中通过 GET_ARCH_NOC_COMBINATIONS()[combo_idx]
    重建 arch/noc 对象，所以必须找到对应的索引。
    """
    all_combos = GET_ARCH_NOC_COMBINATIONS()
    arch_cls_name = arch.__class__.__name__
    noc_name = noc.name
    for i, (a, n) in enumerate(all_combos):
        if a.__class__.__name__ == arch_cls_name and n.name == noc_name:
            return i
    raise ValueError(
        f"arch={arch_cls_name}, noc={noc_name} 不在 GET_ARCH_NOC_COMBINATIONS() 中。"
        f"请先将该组合添加到 arch_noc_combinations 里，或修改模块顶部的 GET_ARCH_NOC_COMBINATIONS。"
    )


def dse_single_task(
    run_dir: str,
    num_workers: int,
    model_key: str,
    bs: int,
    input_seq: int,
    max_seq: int,
    parallel_seq: int = 1,
    arch_noc_list: list[list] | None = None,
    tp: int | None = None,
    ep: int | None = None,
    sp: int | None = None,
    cp: int | None = None,
    dp: int | None = None,
    pp: int | None = None,
    fsdp: bool = False,
    tp_transform_moe_modes: list[str] | None = None,
    enable_proc_log: bool = False,
):
    """
    Run DSE for a single workload (model, bs, seq) with optional arch/noc and parallel constraints.

    Parameters:
        model_key: Model name, one of MODEL_REG keys
        bs: Batch size
        input_seq: Input sequence length
        max_seq: Maximum sequence length (task max seq)
        parallel_seq: Parallel decoding seq (default 1)
        arch_noc_list: List of [arch, noc] pairs to iterate.
                       None = use GET_ARCH_NOC_COMBINATIONS() (search all).
        tp/ep/sp/cp/dp/pp/fsdp: If any parallel param is set, use that fixed scheme
                                 (unset params default to 1). None = search all.
        tp_transform_moe_modes: List of modes. Default: TP_TRANSFORM_MOE_MODES.
        enable_proc_log: Whether to write per-process log files.
    """
    if tp_transform_moe_modes is None:
        tp_transform_moe_modes = TP_TRANSFORM_MOE_MODES

    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False, dump_perf_log=True)
    gran_tuple = (granularity.mode, granularity.comp_comm_overlap, granularity.auto_tune, granularity.dump_perf_log)
    run_dir = os.path.abspath(run_dir)

    proc_log_dir = os.path.join(run_dir, "proc_logs") if enable_proc_log else None
    stats_bundle_root = os.path.join(run_dir, "stats_bundles")
    os.makedirs(stats_bundle_root, exist_ok=True)

    # Build model
    if model_key not in MODEL_REG:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODEL_REG.keys())}")
    model_arch = MODEL_REG[model_key]()

    BS = bs
    INPUT_SEQ = input_seq
    TASK_MAX_SEQ = max_seq
    PARALLEL_SEQ = parallel_seq
    MAX_KV_LEN = min(TASK_MAX_SEQ, model_arch.max_seq_len)

    # Sample KV lengths
    assert NUM_KV_POINT >= 2
    if NUM_KV_POINT == 2:
        sampled_kv_len = (INPUT_SEQ, MAX_KV_LEN)
    else:
        step = (MAX_KV_LEN - INPUT_SEQ) / (NUM_KV_POINT - 1)
        sampled_kv_len = tuple(int(round(INPUT_SEQ + i * step)) for i in range(NUM_KV_POINT))

    # Routing trace file
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    if model_arch.__class__.__name__ in ("DeepSeekV3", "DeepSeekV3_A8W8"):
        npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")
    elif model_arch.__class__.__name__ == "Qwen3_235b_a22b":
        npz_trace_file = str(project_root / "data" / "aime_qwen_235b" / "qwen3_moe_activations_batch0.npz")
    elif model_arch.__class__.__name__ == "Qwen3_30b_a3b":
        # Same 128 routed experts as Qwen3-235B → default to the recorded 235B
        # trace. Override with QWEN3_30B_A3B_TRACE_NPZ to supply another
        # compatible recorded or synthetic trace.
        default_trace = str(project_root / "data" / "aime_qwen_235b" / "qwen3_moe_activations_batch0.npz")
        npz_trace_file = os.environ.get("QWEN3_30B_A3B_TRACE_NPZ", default_trace)
    elif model_arch.__class__.__name__ == "Qwen3_480b_a35b":
        # SIMULATED trace.  Default = linear remap from Qwen3-235B (128→160).
        # Override with QWEN3_480B_TRACE_NPZ to supply another compatible
        # recorded or synthetic trace.
        default_trace = str(
            project_root / "data" / "aime_qwen3coder_480b_simulated_from_235b" / "qwen3coder_moe_activations_batch0.npz"
        )
        npz_trace_file = os.environ.get("QWEN3_480B_TRACE_NPZ", default_trace)
    else:
        npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")

    # Determine arch/noc combinations
    if arch_noc_list is not None:
        combinations = arch_noc_list
    else:
        combinations = GET_ARCH_NOC_COMBINATIONS()

    # Check if user specified a fixed parallel scheme
    parallel_params = [tp, ep, sp, cp, dp, pp]
    user_specified_parallel = any(p is not None for p in parallel_params)
    if user_specified_parallel:
        fixed_scheme = ParallelScheme(
            tp=tp if tp is not None else 1,
            ep=ep if ep is not None else 1,
            sp=sp if sp is not None else 1,
            cp=cp if cp is not None else 1,
            dp=dp if dp is not None else 1,
            pp=pp if pp is not None else 1,
            fsdp=fsdp,
        )

    overall_start_ts = time.time()

    for combo_i, combination in enumerate(combinations):
        combo_start_ts = time.time()
        arch, noc = combination[0], combination[1]

        # 查找子进程需要的 combo_idx（对应 GET_ARCH_NOC_COMBINATIONS 的下标）
        try:
            combo_idx = _find_combo_idx(arch, noc)
        except ValueError as e:
            print(f"[ERROR] {e}")
            continue

        num_nodes = int(noc.num_devices)

        if user_specified_parallel:
            if fixed_scheme.world_size() != num_nodes:
                print(
                    f"[SKIP] {arch.__class__.__name__}/{noc.name}: "
                    f"num_devices={num_nodes} != parallel world_size={fixed_scheme.world_size()}"
                )
                continue
            filtered_parallel_schemes = [fixed_scheme]
        else:
            parallel_schemes = all_parallel_schemes(num_nodes)
            parallel_schemes = filter_illegal_fsdp(parallel_schemes)
            filtered_parallel_schemes = filter_parallel_schemes_by_product_max(parallel_schemes, "dp", "pp", BS)
            filtered_parallel_schemes = filter_illegal_fsdp(filtered_parallel_schemes)
            filtered_parallel_schemes = filter_parallel_schemes_by_max(filtered_parallel_schemes, "sp", PARALLEL_SEQ)

            if model_arch.moe_arch is None:
                filtered_parallel_schemes = filter_parallel_schemes_by_max(filtered_parallel_schemes, "ep", 1)
            else:
                filtered_parallel_schemes = filter_parallel_schemes_by_max(
                    filtered_parallel_schemes, "ep", model_arch.moe_arch.num_routed_experts
                )

        # CSV paths
        csv_path = os.path.join(
            run_dir,
            f"{model_arch.__class__.__name__}_BS{BS}_INPUT_SEQ{INPUT_SEQ}_MAX_SEQ{TASK_MAX_SEQ}_result.csv",
        )
        invalid_csv_path = os.path.join(
            run_dir,
            f"{model_arch.__class__.__name__}_BS{BS}_INPUT_SEQ{INPUT_SEQ}_MAX_SEQ{TASK_MAX_SEQ}_invalid_config.csv",
        )

        # Write CSV headers if needed
        if (not os.path.exists(csv_path)) or os.stat(csv_path).st_size == 0:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = [
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
                    "utps_avg", "stps_avg",
                    "stats_run_id",
                ])
                writer.writerow(header)

        if (not os.path.exists(invalid_csv_path)) or os.stat(invalid_csv_path).st_size == 0:
            with open(invalid_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "arch", "noc", "model", "bs", "minibatch", "seq", "cached_kv",
                    "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
                    "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB",
                ])

        # Filter valid parallel schemes (footprint check)
        valid_parallel_schemes: list[tuple[ParallelScheme, str]] = []
        scheme_footprints: dict[tuple, tuple] = {}

        filter_start_ts = time.time()
        for scheme in filtered_parallel_schemes:
            minibatch = math.ceil(BS / scheme.pp)
            allocate_ep(parallel=scheme, bs=minibatch, seq=PARALLEL_SEQ)

            non_moe_parallel = dataclasses.replace(
                scheme, ep=1, ep1=1, ep2=1, dp=scheme.dp * scheme.ep
            )

            for tp_transform_moe in tp_transform_moe_modes:
                max_activation, global_total_weight, global_kv_cache = get_max_footprint_decode(
                    model_arch=model_arch, bs=minibatch, seq=PARALLEL_SEQ, cached_kv=MAX_KV_LEN,
                    moe_parallel=scheme, non_moe_parallel=non_moe_parallel,
                    tp_transform_moe=tp_transform_moe,
                )

                if (max_activation + global_total_weight + global_kv_cache) <= 1.0 * arch.ddr_capacity:
                    valid_parallel_schemes.append((scheme, tp_transform_moe))
                    scheme_footprints[(scheme, tp_transform_moe)] = (
                        minibatch, max_activation, global_total_weight, global_kv_cache, non_moe_parallel
                    )
                else:
                    with open(invalid_csv_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            arch.__class__.__name__, noc.name, model_arch.__class__.__name__,
                            BS, minibatch, 1, MAX_KV_LEN,
                            scheme.tp, scheme.ep, scheme.ep1, scheme.ep2,
                            scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                            max_activation / (1024**3), global_total_weight / (1024**3), global_kv_cache / (1024**3),
                        ])

        filter_elapsed_sec = time.time() - filter_start_ts

        print(
            f"[Info] {arch.__class__.__name__}/{noc.name}: "
            f"valid={len(valid_parallel_schemes)}, filter={_format_duration(filter_elapsed_sec)}"
        )

        if not valid_parallel_schemes:
            print(f"[SKIP] No valid parallel schemes for {arch.__class__.__name__}/{noc.name}")
            continue

        # Build tasks
        tasks = []
        for scheme, tp_transform_moe in valid_parallel_schemes:
            minibatch, max_activation, global_total_weight, global_kv_cache, non_moe_parallel = scheme_footprints[
                (scheme, tp_transform_moe)
            ]
            tasks.append((
                combo_idx,
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

        # Execute in parallel
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

        # Write results to CSV
        if rows:
            rows.sort(key=lambda r: (r[0], r[1], str(r[6]), r[18]))
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                flat_rows = []
                for r in rows:
                    scheme = r[6]
                    tp_transform_moe = r[18]
                    base = [
                        r[0], r[1], r[2], r[3], r[4], r[5],
                        scheme.tp, scheme.ep, scheme.ep1, scheme.ep2,
                        scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                        tp_transform_moe,
                        r[7], r[8], r[9],
                    ]
                    for kv_len, time_gqa, time_mla, utps, stps in r[10]:
                        base.extend([kv_len, time_gqa * 1000, time_mla * 1000, utps, stps])
                    base.extend([
                        r[11] * 1000, r[12] * 1000, r[13] * 1000, r[14] * 1000, r[15] * 1000,
                        r[16], r[17],
                        r[19],
                    ])
                    flat_rows.append(base)
                writer.writerows(flat_rows)

        combo_elapsed_sec = time.time() - combo_start_ts
        overall_elapsed_sec = time.time() - overall_start_ts
        print(
            f"[Done] {arch.__class__.__name__}/{noc.name}: "
            f"valid={len(valid_parallel_schemes)}, tasks={len(tasks)}, done={len(rows)}, "
            f"filter={_format_duration(filter_elapsed_sec)}, compute={_format_duration(pool_elapsed_sec)}, "
            f"combo={_format_duration(combo_elapsed_sec)}, elapsed={_format_duration(overall_elapsed_sec)}"
        )

    total_elapsed = time.time() - overall_start_ts
    print(f"\n[Finished] Total time: {_format_duration(total_elapsed)}")


if __name__ == "__main__":

    # ========================================================================
    # ==================== 在这里直接修改参数即可运行 ==========================
    # ========================================================================

    # ---- Workload: 模型, bs, seq ----
    # 可用模型: "DeepSeekV3", "DeepSeekV3_A8W8", "Qwen3_235b_a22b", "Qwen3_480b_a35b", "Llama3_70b", "Llama3_405b"
    MODEL        = "DeepSeekV3"
    BS           = 1024
    INPUT_SEQ    = 1024
    MAX_SEQ      = 2048
    PARALLEL_SEQ = 1

    # ---- Arch / NoC (可选) ----
    # 直接用 arch class 和 noc 生成函数构造，设为 None 则搜索 GET_ARCH_NOC_COMBINATIONS() 里的所有组合
    # arch 可选: stacked_gpu_base(), stacked_gpu_large_matrix(), stacked_gpu_large_vector(), stacked_gpu_high_l1(),
    #            stacked_gpu_high_l2(), stacked_gpu_high_noc(), stacked_gpu_low_noc(), stacked_gpu_reduced_sm(),
    #            stacked_gpu_wgmma(), H100_SCALED(), H200_SCALED(), H100(), H200(), B200()
    # noc  可选: torus_mesh_switch_1(), torus_mesh_switch_2(), torus_mesh_mesh_3(),
    #            strong_torus_mesh_switch_4(), weak_torus_mesh_switch_5(),
    #            torus_mesh_switch_7(), torus_mesh_switch_8(), torus_mesh_switch_9(),
    #            h200x8(), h100x8(), h100x32_strong(), h100x32_medium(),
    #            h100_8x1_8(), h100_8x2_16(), h100_8x4_32(), h100_8x8_64(),
    #            stacked_gpu_4x4(), stacked_gpu_4x8(), stacked_gpu_4x16(), stacked_gpu_8x8(), stacked_gpu_8x16(),
    #            b200_8x1_8()
    ARCH_NOC_LIST = [
        [stacked_gpu_wgmma(), torus_mesh_switch_1()],
    ]
    # ARCH_NOC_LIST = None              # None = 搜索所有 GET_ARCH_NOC_COMBINATIONS() 组合

    # ---- Parallel scheme (可选) ----
    # 指定具体的并行方案。任一参数不为 None 就视为指定了固定方案（未设的默认为 1）
    # 全部为 None 则搜索所有合法并行方案
    # TP   = None                     # tensor parallelism,   e.g. 8
    # EP   = None                     # expert parallelism,   e.g. 4
    # SP   = None                     # sequence parallelism, e.g. 1
    # CP   = None                     # context parallelism,  e.g. 1
    # DP   = None                     # data parallelism,     e.g. 2
    # PP   = None                     # pipeline parallelism, e.g. 1
    # FSDP = False                    # fully sharded data parallel

    TP   = 8
    EP   = 16
    SP   = 1
    CP   = 1
    DP   = 1
    PP   = 2
    FSDP = False
    # ---- tp_transform_moe (可选) ----
    # None = 默认 TP_TRANSFORM_MOE_MODES (["replace_only", "none"])
    # 或: ["replace_only"], ["none"], ["replace_only", "none"]
    TP_TRANSFORM_MOE = ["replace_only"]

    # ---- 其他 ----
    NUM_WORKERS     = None          # None = cpu_count // 2
    RUN_DIR         = None          # None = "runs/single_<timestamp>"
    ENABLE_PROC_LOG = False

    # ========================================================================
    # ========================= 以下不需要改动 ================================
    # ========================================================================

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_DIR or os.path.join("runs", f"single_{run_tag}")
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

    num_workers = NUM_WORKERS or (os.cpu_count() // 2)
    print(f"Using {num_workers} workers (cpu_count={os.cpu_count()})")
    print(f"Output dir: {run_dir}")
    print(f"Workload: model={MODEL}, bs={BS}, input_seq={INPUT_SEQ}, max_seq={MAX_SEQ}, parallel_seq={PARALLEL_SEQ}")
    if ARCH_NOC_LIST is not None:
        for a, n in ARCH_NOC_LIST:
            print(f"Arch/NoC: arch={a.__class__.__name__}, noc={n.name}")
    else:
        print("Arch/NoC: searching all combinations")
    if any(p is not None for p in [TP, EP, SP, CP, DP, PP]):
        print(f"Parallel: tp={TP}, ep={EP}, sp={SP}, cp={CP}, dp={DP}, pp={PP}, fsdp={FSDP}")
    else:
        print("Parallel: searching all valid schemes")
    print()

    start_time = time.time()
    dse_single_task(
        run_dir=run_dir,
        num_workers=num_workers,
        model_key=MODEL,
        bs=BS,
        input_seq=INPUT_SEQ,
        max_seq=MAX_SEQ,
        parallel_seq=PARALLEL_SEQ,
        arch_noc_list=ARCH_NOC_LIST,
        tp=TP,
        ep=EP,
        sp=SP,
        cp=CP,
        dp=DP,
        pp=PP,
        fsdp=FSDP,
        tp_transform_moe_modes=TP_TRANSFORM_MOE,
        enable_proc_log=ENABLE_PROC_LOG,
    )
    print(f"Total time: {time.time() - start_time:.1f} seconds")
