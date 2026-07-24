# from mosaic.dse_space.arch_noc_combinations import arch_noc_combinations1
from typing import Any
from mosaic.dse_space.arch_noc_combinations import arch_noc_combinations1110,arch_noc_combinations_scale_64_512_nodes,h100_8_to_64_arch_noc_combinations
from mosaic.dse_space.task_combination import task_combination_4, scale_64_512_nodes_task_combination_prefill
from mosaic.dse_space.parallel_schemes import all_parallel_schemes, filter_parallel_schemes_by_max, filter_illegal_fsdp, filter_parallel_schemes_by_product_max
from mosaic.llm_arch import DeepSeekV3, LLM_Arch, Qwen3_235b_a22b, Qwen3_480b_a35b, Llama3_70b, Llama3_405b, DeepSeekV3_A8W8
from mosaic.parallelism import ParallelScheme
from mosaic.utils.allocate_ep import allocate_ep
import logging
log = logging.getLogger(__name__)
import math
import dataclasses
import hashlib
import json
import re
from mosaic.llm.rms_norm.rms_norm_top import rms_norm_top, get_rms_norm_footprint
from mosaic.llm.swiglu.swiglu_top import swiglu_top, get_swiglu_footprint
from mosaic.llm.add_residual.add_residual_top import add_residual_top, get_add_residual_footprint
from mosaic.llm.gqa.gqa_decode_top import get_gqa_decode_footprint
from mosaic.llm.gqa.gqa_prefill_top import gqa_prefill_top
from mosaic.llm.rope.get_rope_global_footprint import get_rope_global_footprint
from mosaic.llm.moe.moe_top import moe_top, get_moe_footprint
from mosaic.llm.mla.mla_footprint import get_mla_absorb_and_no_absorb_footprint
from mosaic.llm.mla.mla_prefill_top import mla_prefill_top
from mosaic.utils import Modeling_Granularity
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import get_extend_max_routes, get_extend_max_routes_with_traffic, Hierarchy
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


MODEL_REG = {
    "DeepSeekV3": DeepSeekV3,
    "Qwen3_235b_a22b": Qwen3_235b_a22b,
    "Qwen3_480b_a35b": Qwen3_480b_a35b,
    "Llama3_70b": Llama3_70b,
    "Llama3_405b": Llama3_405b,
    "DeepSeekV3_A8W8": DeepSeekV3_A8W8,
}

# 全局指定使用的 arch_noc 组合函数
# GET_ARCH_NOC_COMBINATIONS = arch_noc_combinations1110
# GET_ARCH_NOC_COMBINATIONS = h100_8_to_64_arch_noc_combinations
# GET_ARCH_NOC_COMBINATIONS = arch_noc_combinations_scale_64_512_nodes
from mosaic.dse_space.arch_noc_combinations import b200_8x1_8_arch_noc_combinations, stacked_gpu_dse_vs_h200_scaled_0309
GET_ARCH_NOC_COMBINATIONS = stacked_gpu_dse_vs_h200_scaled_0309
# GET_ARCH_NOC_COMBINATIONS = b200_8x1_8_arch_noc_combinations

# 全局指定使用的 task combination 函数
# GET_TASK_COMBINATIONS = task_combination_4
from mosaic.dse_space.task_combination import hybrid_prefill_task_combination_0301, customized_prefill_task_combination
# GET_TASK_COMBINATIONS = scale_64_512_nodes_task_combination_prefill
GET_TASK_COMBINATIONS = customized_prefill_task_combination
# GET_TASK_COMBINATIONS = hybrid_prefill_task_combination_0301

# tp_transform_moe: whether to convert MoE TP into EP before modeling
#   "none"         – keep original parallel scheme for MoE
#   "replace_only" – replace: tp=1, ep=ep*tp (current hardcoded behaviour, better in practice)
# Set to a list to enumerate multiple modes (each becomes its own CSV row).
TP_TRANSFORM_MOE_MODES: list[str] = ["replace_only", "none"]          # change to ["none", "replace_only"] for both
# TP_TRANSFORM_MOE_MODES: list[str] = ["replace_only"]  

# 子进程内的全局缓存，避免把 numpy 数组跨进程序列化
_ROUTING_ARRAY = None


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _init_worker(npz_trace_file: str):
    """每个子进程启动时执行：只在子进程里加载 routing array，一次到位"""
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
    global _ROUTING_ARRAY
    _, decode_array = load_npz_routing_keep_shape(npz_trace_file, as_list=False)
    _ROUTING_ARRAY = decode_array


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
) -> str:
    key = (
        f"arch={arch_name}|noc={noc_name}|model={model_name}|bs={bs}|minibatch={minibatch}|seq={seq}|"
        f"tp={parallel_scheme.tp}|ep={parallel_scheme.ep}|ep1={parallel_scheme.ep1}|ep2={parallel_scheme.ep2}|"
        f"sp={parallel_scheme.sp}|cp={parallel_scheme.cp}|dp={parallel_scheme.dp}|fsdp={parallel_scheme.fsdp}|"
        f"pp={parallel_scheme.pp}|tp_transform_moe={tp_transform_moe}"
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{_slugify(model_name)}_{_slugify(arch_name)}_{_slugify(noc_name)}_{digest}"


def _dump_stats_bundle(
    stats_bundle_root: str,
    run_dir: str,
    stats_run_id: str,
    representative_kv_len: int,
    model_stats: ModelPerfStats,
    metadata: dict,
) -> tuple[str, str]:
    bundle_dir = os.path.join(stats_bundle_root, stats_run_id)
    os.makedirs(bundle_dir, exist_ok=True)

    model_json = os.path.join(bundle_dir, "model_stats.json")
    model_txt = os.path.join(bundle_dir, "model_stats.txt")
    model_md = os.path.join(bundle_dir, "model_stats.md")
    manifest = os.path.join(bundle_dir, "manifest.json")

    model_stats.dump_json(model_json)
    model_stats.dump_human(model_txt, markdown=False)
    model_stats.dump_human(model_md, markdown=True)

    manifest_payload = {
        "stats_run_id": stats_run_id,
        "representative_kv_len": int(representative_kv_len),
        "model_stats_files": {
            "json": os.path.relpath(model_json, run_dir),
            "txt": os.path.relpath(model_txt, run_dir),
            "md": os.path.relpath(model_md, run_dir),
        },
        "metadata": metadata,
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    return os.path.relpath(bundle_dir, run_dir), os.path.relpath(manifest, run_dir)




def _get_effective_moe_parallel(moe_parallel: ParallelScheme, tp_transform_moe: str, num_routed_experts: int = 0) -> ParallelScheme:
    """Return the actual ParallelScheme used for MoE ops based on tp_transform_moe mode."""
    if tp_transform_moe == "replace_only":
        combined = moe_parallel.ep * moe_parallel.tp
        if num_routed_experts > 0 and combined > num_routed_experts:
            ep = num_routed_experts
            tp = combined // num_routed_experts
        else:
            ep = combined
            tp = 1
        return dataclasses.replace(
            moe_parallel, tp=tp,
            ep=ep,
            ep1=ep,
            ep2=1,
        )
    return moe_parallel  # "none" – keep as-is


def get_max_footprint_prefill(model_arch:LLM_Arch, bs:int, seq:int, cached_kv:int,
    parallel:ParallelScheme, atten_parallel:ParallelScheme, moe_parallel:ParallelScheme, non_atten_non_moe_parallel:ParallelScheme, tp_transform_moe:str="replace_only"):
    hidden = model_arch.hidden_size



    shard_layer = math.ceil(model_arch.num_layer/parallel.pp)
    if model_arch.gqa_arch is not None:
        wq_hidden = model_arch.gqa_arch.num_head * model_arch.gqa_arch.head_dim
        head_dim = model_arch.gqa_arch.head_dim
    elif model_arch.mla_arch is not None:
        wq_hidden = model_arch.mla_arch.num_head * model_arch.mla_arch.head_dim
        head_dim = model_arch.mla_arch.head_dim
    else:
        raise ValueError("model_arch must have one of gqa_arch or mla_arch")

    def get_gqa_footprint_wrapper(model_arch:LLM_Arch, cached_kv:int, atten_parallel:ParallelScheme, non_atten_non_moe_parallel:ParallelScheme):
        from mosaic.llm.gqa.gqa_prefill_top import get_gqa_prefill_footprint
        num_head = model_arch.gqa_arch.num_head
        num_kv_head = model_arch.gqa_arch.num_kv_head
        head_dim = model_arch.gqa_arch.head_dim
        atten_parallel = parallel
        next_parallel = parallel
        atten_bytes = model_arch.gqa_arch.atten_bytes


        # def get_gqa_prefill_footprint(bs:int, seq:int, cached_kv:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme, atten_bytes:OpBytes):


        max_activation, mem_weight, mem_kv_cache = get_gqa_prefill_footprint(
            bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, 
            num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim,
            parallel=non_atten_non_moe_parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes
        )

        shard_layer = math.ceil(model_arch.num_layer/parallel.pp)
        return max_activation, mem_weight*shard_layer, mem_kv_cache*model_arch.num_layer

    def get_mla_footprint_wrapper(model_arch:LLM_Arch, cached_kv:int, atten_parallel:ParallelScheme, non_atten_non_moe_parallel:ParallelScheme):
        max_activation, mem_weight, mem_kv_cache = get_mla_absorb_and_no_absorb_footprint(
            bs=bs, seq=seq, cached_kv=cached_kv, model_arch=model_arch, parallel=non_atten_non_moe_parallel, atten_parallel=atten_parallel
        )
        shard_layer = math.ceil(model_arch.num_layer/parallel.pp)
        return max_activation, mem_weight*shard_layer, mem_kv_cache*model_arch.num_layer

    def get_dense_ffn_footprint_wrapper(model_arch:LLM_Arch, cached_kv:int, non_atten_non_moe_parallel:ParallelScheme):
        from mosaic.llm.swiglu.swiglu_top import get_swiglu_footprint

        up_hidden = model_arch.dense_ffn_arch.up_hidden
        swiglu_bytes = model_arch.dense_ffn_arch.swiglu_bytes

        max_activation, mem_weight = get_swiglu_footprint(bs, seq, hidden, up_hidden, non_atten_non_moe_parallel, swiglu_bytes)

        shard_layer = math.ceil(model_arch.num_dense_layer/non_atten_non_moe_parallel.pp)
        return max_activation, mem_weight*shard_layer

    def get_moe_footprint_wrapper(model_arch:LLM_Arch, cached_kv:int, moe_parallel:ParallelScheme):
        num_shared_experts=model_arch.moe_arch.num_shared_experts
        num_routed_experts=model_arch.moe_arch.num_routed_experts
        num_activated_experts=model_arch.moe_arch.num_activated_experts
        moe_down_hidden=model_arch.moe_arch.moe_down_hidden
        expert_bytes=model_arch.moe_arch.expert_bytes
        gate_bytes=model_arch.moe_arch.gate_bytes

        max_activation, mem_weight = get_moe_footprint(
            bs, seq, hidden, moe_down_hidden, moe_parallel, expert_bytes, gate_bytes,
            num_shared_experts, num_routed_experts
        )

        shard_layer = math.ceil(model_arch.num_moe_layer/moe_parallel.pp)
        return max_activation, mem_weight*shard_layer

    if model_arch.gqa_arch is not None:
        gqa_max_activation, gqa_mem_weight, gqa_mem_kv_cache = get_gqa_footprint_wrapper(model_arch=model_arch, cached_kv=cached_kv, atten_parallel=atten_parallel, non_atten_non_moe_parallel=non_atten_non_moe_parallel)
        log.info("gqa_max_activation: %s GiB, gqa_mem_weight: %s GiB, gqa_mem_kv_cache: %s GiB", gqa_max_activation/(1024**3), gqa_mem_weight/(1024**3), gqa_mem_kv_cache/(1024**3))
    else:
        gqa_max_activation = 0
        gqa_mem_weight = 0
        gqa_mem_kv_cache = 0

    if model_arch.mla_arch is not None:
        mla_max_activation, mla_mem_weight, mla_mem_kv_cache = get_mla_footprint_wrapper(model_arch=model_arch, cached_kv=cached_kv, atten_parallel=atten_parallel, non_atten_non_moe_parallel=non_atten_non_moe_parallel)
        log.info("mla_max_activation: %s GiB, mla_mem_weight: %s GiB, mla_mem_kv_cache: %s GiB", mla_max_activation/(1024**3), mla_mem_weight/(1024**3), mla_mem_kv_cache/(1024**3))
    else:
        mla_max_activation = 0
        mla_mem_weight = 0
        mla_mem_kv_cache = 0

    if model_arch.dense_ffn_arch is not None:
        dense_ffn_max_activation, dense_ffn_mem_weight = get_dense_ffn_footprint_wrapper(model_arch=model_arch, cached_kv=cached_kv, non_atten_non_moe_parallel=non_atten_non_moe_parallel)
        log.info("dense_ffn_max_activation: %s GiB, dense_ffn_mem_weight: %s GiB", dense_ffn_max_activation/(1024**3), dense_ffn_mem_weight/(1024**3))
    else:
        dense_ffn_max_activation = 0
        dense_ffn_mem_weight = 0

    if model_arch.moe_arch is not None:
        effective_moe_parallel = _get_effective_moe_parallel(moe_parallel, tp_transform_moe, num_routed_experts=model_arch.moe_arch.num_routed_experts)
        moe_max_activation, moe_mem_weight = get_moe_footprint_wrapper(model_arch=model_arch, cached_kv=cached_kv, moe_parallel=effective_moe_parallel)
        log.info("moe_max_activation: %s GiB, moe_mem_weight: %s GiB", moe_max_activation/(1024**3), moe_mem_weight/(1024**3))
    else:
        moe_max_activation = 0
        moe_mem_weight = 0

    # rms_norm, 2blocks per layer
    act_rms, mem_weight_rms = get_rms_norm_footprint(bs=bs, seq=seq, hidden=hidden, parallel=non_atten_non_moe_parallel, rms_norm_bytes=model_arch.rms_norm_bytes)
    mem_weight_rms = mem_weight_rms * shard_layer * 2
    log.info("act_rms: %s GiB, mem_weight_rms: %s GiB", act_rms/(1024**3), mem_weight_rms/(1024**3))

    # rope global
    mem_weight_rope = get_rope_global_footprint(bs=bs, head_dim=head_dim, parallel=non_atten_non_moe_parallel, rope_bytes=model_arch.rope_bytes, MAX_ROPE_SEQ = model_arch.max_seq_len)
    log.info("mem_weight_rope: %s GiB", mem_weight_rope/(1024**3))

    # add_residual, 2blocks per layer
    act_add_residual, mem_weight_add_residual = get_add_residual_footprint(bs=bs, seq=seq, hidden=hidden, parallel=non_atten_non_moe_parallel, add_residual_bytes=model_arch.add_residual_bytes)
    mem_weight_add_residual = mem_weight_add_residual * shard_layer * 2
    log.info("act_add_residual: %s GiB, mem_weight_add_residual: %s GiB", act_add_residual/(1024**3), mem_weight_add_residual/(1024**3))

    max_activation = max(act_rms, act_add_residual, gqa_max_activation, mla_max_activation, dense_ffn_max_activation, moe_max_activation)
    global_total_weight = mem_weight_rms + mem_weight_rope + mem_weight_add_residual + gqa_mem_weight + mla_mem_weight + dense_ffn_mem_weight + moe_mem_weight
    global_kv_cache = gqa_mem_kv_cache + mla_mem_kv_cache

    log.info("act_rms: %s GiB, act_add_residual: %s GiB, gqa_max_activation: %s GiB, mla_max_activation: %s GiB, dense_ffn_max_activation: %s GiB, moe_max_activation: %s GiB", act_rms/(1024**3), act_add_residual/(1024**3), gqa_max_activation/(1024**3), mla_max_activation/(1024**3), dense_ffn_max_activation/(1024**3), moe_max_activation/(1024**3))
    log.info("max activation: %s GiB", max_activation/(1024**3))
    log.info("mem_weight_rms: %s GiB, mem_weight_rope: %s GiB, mem_weight_add_residual: %s GiB, gqa_mem_weight: %s GiB, mla_mem_weight: %s GiB, dense_ffn_mem_weight: %s GiB, moe_mem_weight: %s GiB", mem_weight_rms/(1024**3), mem_weight_rope/(1024**3), mem_weight_add_residual/(1024**3), gqa_mem_weight/(1024**3), mla_mem_weight/(1024**3), dense_ffn_mem_weight/(1024**3), moe_mem_weight/(1024**3))
    log.info("global total weight: %s GiB", global_total_weight/(1024**3))
    log.info("global kv cache: %s GiB", global_kv_cache/(1024**3))
    return max_activation, global_total_weight, global_kv_cache

def modeling_prefill(model_arch:LLM_Arch, bs:int, seq:int, cached_kv:int, parallel:ParallelScheme, atten_parallel:ParallelScheme, moe_parallel:ParallelScheme, non_atten_non_moe_parallel:ParallelScheme, single_chip:Arch, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, routing_array: np.ndarray, tp_transform_moe:str="replace_only"):

    # parallel = non_moe_parallel
    hidden = model_arch.hidden_size
    shard_layer = math.ceil(model_arch.num_layer/non_atten_non_moe_parallel.pp)

    num_routed = model_arch.moe_arch.num_routed_experts if model_arch.moe_arch is not None else 0
    effective_moe_parallel = _get_effective_moe_parallel(moe_parallel, tp_transform_moe, num_routed_experts=num_routed)

    model_stats = ModelPerfStats(
        name=(
            f"model={model_arch.name}-prefill-"
            f"bs{bs}-"
            f"seq{seq}-"
            f"tp{moe_parallel.tp}-"
            f"ep{moe_parallel.ep}-"
            f"sp{moe_parallel.sp}-"
            f"cp{moe_parallel.cp}-"
            f"dp{moe_parallel.dp}-"
            f"pp{moe_parallel.pp}-"
            f"fsdp{moe_parallel.fsdp}-"
        ),
        dump_perf_log=granularity.dump_perf_log,
    )

    def get_pipeline_time():

        pp_op_stats = OpPerfStats(op_name="pp_overhead", dump_perf_log=granularity.dump_perf_log)

        if non_atten_non_moe_parallel.pp == 1:
            return 0, None

        shard_bs = math.ceil(bs / non_atten_non_moe_parallel.dp)
        shard_seq = math.ceil(seq / non_atten_non_moe_parallel.sp)
        hidden = model_arch.hidden_size
        activation_bytes = model_arch.add_residual_bytes.input1.num_bytes

        pipeline_activation_bytes = shard_bs * shard_seq * hidden * activation_bytes

        pp_time_list = []

        for i in range(non_atten_non_moe_parallel.pp - 1):
            traffic_pair = [[pipeline_activation_bytes, i, (i+1)%non_atten_non_moe_parallel.pp]]
            tm = TrafficMatrix(non_atten_non_moe_parallel.world_size())
            tm.add_intra_group_traffic_pair_bulk("pp", traffic_pair, tp=non_atten_non_moe_parallel.tp, ep=non_atten_non_moe_parallel.ep, sp=non_atten_non_moe_parallel.sp, cp=non_atten_non_moe_parallel.cp, dp=non_atten_non_moe_parallel.dp, pp=non_atten_non_moe_parallel.pp)
            pp_hop_latency, pp_ext_max, pp_overall_time, traffic = get_extend_max_routes_with_traffic(tm, noc_hierarchy, True)
            pp_time_list.append(pp_overall_time)
            pp_op_stats.append_traffic(traffic, hop_time_s=pp_hop_latency, link_time_s=pp_ext_max, comm_time_s=pp_overall_time)

        # 取流水段之间的最大时延；若为空则为 0
        return (max(pp_time_list), pp_op_stats) if pp_time_list else (0.0, None)

    pp_p2p_time, pp_op_stats = get_pipeline_time()

    if pp_op_stats is not None:
        pp_op_stats.finalize(total_time_s=pp_p2p_time, h=noc_hierarchy)
        model_stats.add(pp_op_stats, n=1)

    if model_arch.gqa_arch is not None:
        time_gqa, gqa_stats = gqa_prefill_top(bs=bs, seq=seq, hidden=hidden, num_head=model_arch.gqa_arch.num_head, num_kv_head=model_arch.gqa_arch.num_kv_head, head_dim=model_arch.gqa_arch.head_dim, parallel=non_atten_non_moe_parallel, atten_parallel=atten_parallel, next_parallel=non_atten_non_moe_parallel, atten_bytes=model_arch.gqa_arch.atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        model_stats.add(gqa_stats, n=shard_layer)
        time_gqa = time_gqa * shard_layer
    else:
        time_gqa = 0

    if model_arch.mla_arch is not None:
        time_mla, mla_stats = mla_prefill_top(bs=bs, seq=seq, model_arch=model_arch, parallel=non_atten_non_moe_parallel, atten_parallel=atten_parallel, next_parallel=non_atten_non_moe_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        time_mla = time_mla * shard_layer
        model_stats.add(mla_stats, n=shard_layer)
    else:
        time_mla = 0

    if model_arch.dense_ffn_arch is not None:
        time_dense_ffn, dense_ffn_stats = swiglu_top(bs=bs, seq=seq, hidden=hidden, up_hidden=model_arch.dense_ffn_arch.up_hidden, parallel=non_atten_non_moe_parallel, next_parallel=non_atten_non_moe_parallel, swiglu_bytes=model_arch.dense_ffn_arch.swiglu_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        shard_dense_layer = math.ceil(model_arch.num_dense_layer/non_atten_non_moe_parallel.pp)
        time_dense_ffn = time_dense_ffn * shard_dense_layer
        model_stats.add(dense_ffn_stats, n=shard_dense_layer)
    else:
        time_dense_ffn = 0

    if model_arch.moe_arch is not None:
        time_moe, moe_stats = moe_top(bs=bs, seq=seq, hidden=hidden, moe_down_hidden=model_arch.moe_arch.moe_down_hidden, parallel=effective_moe_parallel, next_parallel=non_atten_non_moe_parallel, expert_bytes=model_arch.moe_arch.expert_bytes, gate_bytes=model_arch.moe_arch.gate_bytes, num_shared_experts=model_arch.moe_arch.num_shared_experts, num_routed_experts=model_arch.moe_arch.num_routed_experts, num_activated_experts=model_arch.moe_arch.num_activated_experts, routing_array=routing_array, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        shard_moe_layer = math.ceil(model_arch.num_moe_layer/non_atten_non_moe_parallel.pp)
        time_moe = time_moe * shard_moe_layer
        model_stats.add(moe_stats, n=shard_moe_layer)
    else:
        time_moe = 0

    time_rms_norm, rms_norm_stats = rms_norm_top(bs=bs, seq=seq, hidden=hidden, parallel=non_atten_non_moe_parallel, next_parallel=non_atten_non_moe_parallel, rms_norm_bytes=model_arch.rms_norm_bytes,
        granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    time_rms_norm = time_rms_norm * shard_layer * 2
    model_stats.add(rms_norm_stats, n=2*shard_layer)

    time_add_residual, add_residual_stats = add_residual_top(bs=bs, seq=seq, hidden=hidden, parallel=non_atten_non_moe_parallel, next_parallel=non_atten_non_moe_parallel, add_residual_bytes=model_arch.add_residual_bytes,
        granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    time_add_residual = time_add_residual * shard_layer * 2
    model_stats.add(add_residual_stats, n=2*shard_layer)

    time_total = pp_p2p_time + time_gqa + time_mla + time_dense_ffn + time_moe + time_rms_norm + time_add_residual
    log.info("p2p time: %s s, gqa time: %s s, mla time: %s s, dense_ffn time: %s s, moe time: %s s, rms_norm time: %s s, add_residual time: %s s", pp_p2p_time, time_gqa, time_mla, time_dense_ffn, time_moe, time_rms_norm, time_add_residual)
    log.info("total time: %s s", time_total)

    model_stats.finalize(h=noc_hierarchy)

    return pp_p2p_time, time_gqa, time_mla, time_dense_ffn, time_moe, time_rms_norm, time_add_residual, time_total, model_stats

# ---------- 多进程 worker：单个 (scheme, kv_len) 任务 ----------
def _compute_time_row(args):
    (
        combo_idx,              # int -> arch_noc_combinations1()[combo_idx]
        model_key,              # str -> MODEL_REG[model_key]()
        BS, minibatch, cached_kv, seq,
        parallel_scheme,        # ParallelScheme (dataclass，顶层定义，可 pickle)
        max_activation, global_total_weight, global_kv_cache,
        granularity_tuple,      # (mode, comp_comm_overlap, auto_tune, dump_perf_log) 纯标量
        atten_parallel, moe_parallel, non_atten_non_moe_parallel,        # ParallelScheme
        proc_log_dir,           # str or None，None 表示不写日志
        run_dir,                # run 根目录
        stats_bundle_root,      # stats bundle 根目录
        tp_transform_moe,       # str: "none" | "replace_only"
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

    # 在子进程里重建 heavy 对象
    arch, noc = GET_ARCH_NOC_COMBINATIONS()[combo_idx]
    model_arch = MODEL_REG[model_key]()

    # 重建 granularity
    granularity = Modeling_Granularity(
        mode=granularity_tuple[0],
        comp_comm_overlap=granularity_tuple[1],
        auto_tune=granularity_tuple[2],
        dump_perf_log=granularity_tuple[3],
    )

    # 拿到子进程全局 routing_array
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
    utps = seq / ttft
    stps = minibatch * seq / ttft_per_pp

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
            "arch": arch_name,
            "noc": noc_name,
            "model": model_name,
            "bs": BS,
            "minibatch": minibatch,
            "seq": seq,
            "cached_kv": cached_kv,
            "parallel_scheme": dataclasses.asdict(parallel_scheme),
            "tp_transform_moe": tp_transform_moe,
            "proc_log_relpath": proc_log_relpath,
        },
    )

    return [
        arch_name, noc_name, model_name,
        BS, minibatch, seq,
        parallel_scheme,                  # [6]
        max_activation/(1024**3),         # [7]
        global_total_weight/(1024**3),    # [8]
        global_kv_cache/(1024**3),        # [9]
        pp_p2p_time, time_gqa, time_mla, time_dense_ffn, time_moe, time_rms_norm, time_add_residual,  # [10-16]
        ttft_per_pp, ttft, utps, stps,    # [17-20]
        tp_transform_moe,                 # [21]
        stats_run_id,                     # [22]
    ]

def dse_1(run_dir, num_workers, enable_proc_log=False):
    """
    1) 先在主进程过滤 valid parallel schemes，并把 footprint 记下来；
    2) 然后把 (combo_idx, model_key, kv_len, scheme, ...) 这些可序列化的轻量键组装成任务；
    3) 用多进程并行计算 time/utps/stps（num_workers）；
    4) 主进程统一写入 CSV，避免多进程文件竞争。
    """
    # ---- 全局建模粒度（传入子进程用纯标量三元组）----
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False, dump_perf_log=True)
    gran_tuple = (granularity.mode, granularity.comp_comm_overlap, granularity.auto_tune, granularity.dump_perf_log)
    run_dir = os.path.abspath(run_dir)

    # ---- 每进程日志目录，None 表示不写日志 ----
    proc_log_dir = os.path.join(run_dir, "proc_logs") if enable_proc_log else None
    stats_bundle_root = os.path.join(run_dir, "stats_bundles")
    os.makedirs(stats_bundle_root, exist_ok=True)


    # ---- 任务组合（可以自行增删）----
    # task_combinations = [
    #     # [DeepSeekV3(), 16, 1024, 16384,1],
    #     [Qwen3_235b_a22b(), 16, 8192, 8192, 8192],
    #     # [Llama3_70b(), 16, 1024, 64*1024, 1]
    # ]
    # from mosaic.dse_space import task_combination_4
    # task_combinations = task_combination_4()
    # from mosaic.dse_space.task_combination import scale_64_512_nodes_task_combination_prefill
    # task_combinations = scale_64_512_nodes_task_combination_prefill()
    task_combinations = GET_TASK_COMBINATIONS()

    # ---- 枚举 arch/noc 组合 与 并行方案集合 ----
    # combinations = arch_noc_combinations1110()
    # from mosaic.dse_space.arch_noc_combinations import arch_noc_combinations_scale_64_512_nodes
    combinations = GET_ARCH_NOC_COMBINATIONS()
    # combinations = h100_8_to_64_arch_noc_combinations()
    # from mosaic.arch import stacked_gpu_base
    # from mosaic.noc.noc_config_set import torus_mesh_switch_1
    # combinations = [[stacked_gpu_base(), torus_mesh_switch_1()]]


    total_combo_cnt = len(task_combinations) * len(combinations)
    finished_combo_cnt = 0
    combo_time_sum_sec = 0.0
    overall_start_ts = time.time()

    for task_idx, task_combination in enumerate(task_combinations):
        model_arch = task_combination[0]
        BS = task_combination[1]
        INPUT_SEQ = task_combination[2]
        TASK_MAX_SEQ = task_combination[3]
        # for parallel decoding
        PARALLEL_SEQ = task_combination[4]
        assert INPUT_SEQ == PARALLEL_SEQ
        # PARALLEL_SEQ = INPUT_SEQ
        
        # ---- routing trace 的 npz 路径：由进程池 initializer 在子进程里加载 ----
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent  # .../mosaic
        if model_arch.__class__.__name__ == "DeepSeekV3" or model_arch.__class__.__name__ == "DeepSeekV3_A8W8":
            npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")
        elif model_arch.__class__.__name__ == "Qwen3_235b_a22b":
            npz_trace_file = str(project_root / "data" / "aime_qwen_235b" / "qwen3_moe_activations_batch0.npz")
        else:
            # won't be used, just for compatibility
            npz_trace_file = npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")





        MAX_KV_LEN = min(TASK_MAX_SEQ, model_arch.max_seq_len)


        # ---- CSV 路径与表头 ----
        # out_dir = os.path.dirname(os.path.abspath(__file__))
        # csv_path = os.path.join(out_dir, f"dse_{model_arch.__class__.__name__}_BS{BS}_INPUT_SEQ{INPUT_SEQ}_TASK_MAX_SEQ{TASK_MAX_SEQ}.csv")
        # invalid_csv_path = csv_path.replace(".csv", "_invalid_config.csv")
        csv_path = os.path.join(run_dir, f"{model_arch.__class__.__name__}_BS{BS}_INPUT_SEQ{INPUT_SEQ}_MAX_SEQ{TASK_MAX_SEQ}_result.csv")
        invalid_csv_path = os.path.join(run_dir, f"{model_arch.__class__.__name__}_BS{BS}_INPUT_SEQ{INPUT_SEQ}_MAX_SEQ{TASK_MAX_SEQ}_invalid_config.csv")


        # if (not os.path.exists(csv_path)) or os.stat(csv_path).st_size == 0:
        #     with open(csv_path, mode="w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["arch", "noc", "model", "bs", "minibatch", "seq", "cached_kv",
        #                          "parallel_scheme", "max_activation/GiB", "mem_weight/GiB",
        #                          "kv_cache/GiB", "User Token Per Second", "System Token Per Second"])
        if (not os.path.exists(csv_path)) or os.stat(csv_path).st_size == 0:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = [
                    "arch", "noc", "model", "bs", "minibatch", "seq",
                    "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
                    "tp_transform_moe",
                    "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB",
                    "pp_p2p_time_ms", "time_gqa_ms", "time_mla_ms", "time_dense_ffn_ms",
                    "time_moe_ms", "time_rms_norm_ms", "time_add_residual_ms",
                    "ttft_per_pp_ms", "ttft_ms", "utps", "stps",
                    "stats_run_id",
                ]
                writer.writerow(header)
        if (not os.path.exists(invalid_csv_path)) or os.stat(invalid_csv_path).st_size == 0:
            with open(invalid_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["arch", "noc", "model", "bs", "minibatch", "seq", "cached_kv",
                                 "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
                                 "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB"])

        # ---- 遍历所有 (arch, noc) 组合 ----
        for combo_idx, combination in enumerate(combinations):
            combo_start_ts = time.time()
            arch = combination[0]
            noc = combination[1]

            num_nodes = int(noc.num_devices)
            parallel_schemes = all_parallel_schemes(num_nodes)

            # parallel_schemes = all_parallel_schemes(256)
            parallel_schemes = filter_illegal_fsdp(parallel_schemes)
            # parallel_schemes = [ParallelScheme(tp=32, ep=2, sp=1, cp=1, dp=1, pp=4, fsdp=False)]


            # bs_parallel_schemes = filter_parallel_schemes_by_max(parallel_schemes, "pp", BS)
            filtered_parallel_schemes = filter_parallel_schemes_by_product_max(parallel_schemes, "dp", "pp", BS)
            filtered_parallel_schemes = filter_illegal_fsdp(filtered_parallel_schemes)
            filtered_parallel_schemes = filter_parallel_schemes_by_max(filtered_parallel_schemes, "sp", PARALLEL_SEQ)

            if model_arch.moe_arch is None:
                filtered_parallel_schemes = filter_parallel_schemes_by_max(filtered_parallel_schemes, "ep", 1)
            else:
                filtered_parallel_schemes = filter_parallel_schemes_by_max(filtered_parallel_schemes, "ep", model_arch.moe_arch.num_routed_experts)



            # valid_parallel_schemes: list of (scheme, tp_transform_moe) pairs
            valid_parallel_schemes: list[tuple[ParallelScheme, str]] = []
            # key: (scheme, tp_transform_moe); value: (minibatch, max_activation, global_total_weight, global_kv_cache, atten_parallel, moe_parallel, non_atten_non_moe_parallel)
            scheme_footprints: dict[tuple, tuple] = {}

            # ---- 先筛有效方案（主进程内进行，不涉及多进程）----
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

                    if ((max_activation + global_total_weight + global_kv_cache) <= 1.0 * arch.ddr_capacity):
                        log.info("parallel scheme: %s, tp_transform_moe: %s", scheme, tp_transform_moe)
                        log.info("max_activation: %s GiB, global_total_weight: %s GiB, global_kv_cache: %s GiB",
                                 max_activation/(1024**3), global_total_weight/(1024**3), global_kv_cache/(1024**3))
                        log.info("valid")

                        valid_parallel_schemes.append((scheme, tp_transform_moe))
                        scheme_footprints[(scheme, tp_transform_moe)] = (minibatch, max_activation, global_total_weight, global_kv_cache, atten_parallel, moe_parallel, non_atten_non_moe_parallel)
                    else:
                        log.info("parallel scheme: %s, tp_transform_moe: %s", scheme, tp_transform_moe)
                        log.info("max_activation: %s GiB, global_total_weight: %s GiB, global_kv_cache: %s GiB",
                                 max_activation/(1024**3), global_total_weight/(1024**3), global_kv_cache/(1024**3))
                        log.info("invalid")

                        with open(invalid_csv_path, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                arch.__class__.__name__, noc.name, model_arch.__class__.__name__,
                                BS, minibatch, 1, MAX_KV_LEN,
                                scheme.tp, scheme.ep, scheme.ep1, scheme.ep2, scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                                max_activation/(1024**3), global_total_weight/(1024**3), global_kv_cache/(1024**3),
                            ])
            filter_elapsed_sec = time.time() - filter_start_ts

            log.info("model_arch: %s, bs: %s, input_seq: %s, task_max_seq: %s",
                     model_arch.__class__.__name__, BS, INPUT_SEQ, TASK_MAX_SEQ)
            log.info("valid parallel schemes length: %s", len(valid_parallel_schemes))
            if not valid_parallel_schemes:
                finished_combo_cnt += 1
                combo_elapsed_sec = time.time() - combo_start_ts
                combo_time_sum_sec += combo_elapsed_sec
                overall_elapsed_sec = time.time() - overall_start_ts
                avg_combo_sec = combo_time_sum_sec / max(finished_combo_cnt, 1)
                remain_combo = max(total_combo_cnt - finished_combo_cnt, 0)
                eta_sec = avg_combo_sec * remain_combo
                print(
                    f"[Progress] combo {finished_combo_cnt}/{total_combo_cnt} "
                    f"(task {task_idx+1}/{len(task_combinations)}, arch_noc {combo_idx+1}/{len(combinations)}) "
                    f"{model_arch.__class__.__name__}/{noc.name}: valid=0, "
                    f"filter={_format_duration(filter_elapsed_sec)}, combo={_format_duration(combo_elapsed_sec)}, "
                    f"elapsed={_format_duration(overall_elapsed_sec)}, eta={_format_duration(eta_sec)}",
                    flush=True,
                )
                continue

            # ---- 组装并行任务（只传键/标量/小对象）----
            tasks = []
            model_key = model_arch.__class__.__name__
            for scheme, tp_transform_moe in valid_parallel_schemes:
                minibatch, max_activation, global_total_weight, global_kv_cache, atten_parallel, moe_parallel, non_atten_non_moe_parallel = scheme_footprints[(scheme, tp_transform_moe)]
                tasks.append((
                    combo_idx,              # arch/noc 的索引，子进程里重建
                    model_key,              # 模型名，子进程里重建
                    BS, minibatch, MAX_KV_LEN, PARALLEL_SEQ,
                    scheme,                 # 顶层 dataclass，可 pickle
                    max_activation, global_total_weight, global_kv_cache,
                    gran_tuple,             # 纯标量四元组
                    atten_parallel, moe_parallel, non_atten_non_moe_parallel,        # 顶层 dataclass，可 pickle
                    proc_log_dir,           # str or None
                    run_dir,                # run 根目录
                    stats_bundle_root,      # stats bundle 根目录
                    tp_transform_moe,       # str: "none" | "replace_only"
                ))

            

            rows = []
            pool_start_ts = time.time()
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,          # 在子进程内加载 routing_array
                initargs=(npz_trace_file,),
            ) as executor:
                futures = [executor.submit(_compute_time_row, t) for t in tasks]
                from concurrent.futures import as_completed
                for fut in as_completed(futures):
                    try:
                        row = fut.result()
                        rows.append(row)
                    except Exception as e:
                        log.exception("Parallel task failed: %s", e)
            pool_elapsed_sec = time.time() - pool_start_ts

            # ---- 主进程统一写 CSV ----
            if rows:
                # 排序：arch, noc, scheme_str, tp_transform_moe
                rows.sort(key=lambda r: (r[0], r[1], str(r[6]), r[21]))
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    flat_rows = []
                    for r in rows:
                        scheme = r[6]
                        tp_transform_moe = r[21]
                        base = [
                            r[0], r[1], r[2], r[3], r[4], r[5],
                            scheme.tp, scheme.ep, scheme.ep1, scheme.ep2, scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                            tp_transform_moe,
                            r[7], r[8], r[9],
                            r[10] * 1000, r[11] * 1000, r[12] * 1000, r[13] * 1000,
                            r[14] * 1000, r[15] * 1000, r[16] * 1000,
                            r[17] * 1000, r[18] * 1000, r[19], r[20],
                            r[22],
                        ]
                        flat_rows.append(base)
                    writer.writerows(flat_rows)

            finished_combo_cnt += 1
            combo_elapsed_sec = time.time() - combo_start_ts
            combo_time_sum_sec += combo_elapsed_sec
            overall_elapsed_sec = time.time() - overall_start_ts
            avg_combo_sec = combo_time_sum_sec / max(finished_combo_cnt, 1)
            remain_combo = max(total_combo_cnt - finished_combo_cnt, 0)
            eta_sec = avg_combo_sec * remain_combo
            print(
                f"[Progress] combo {finished_combo_cnt}/{total_combo_cnt} "
                f"(task {task_idx+1}/{len(task_combinations)}, arch_noc {combo_idx+1}/{len(combinations)}) "
                f"{model_arch.__class__.__name__}/{noc.name}: valid={len(valid_parallel_schemes)}, "
                f"tasks={len(tasks)}, done={len(rows)}, filter={_format_duration(filter_elapsed_sec)}, "
                f"compute={_format_duration(pool_elapsed_sec)}, combo={_format_duration(combo_elapsed_sec)}, "
                f"elapsed={_format_duration(overall_elapsed_sec)}, eta={_format_duration(eta_sec)}",
                flush=True,
            )



if __name__ == "__main__":
    import logging, os
    from datetime import datetime
    import multiprocessing as mp

    # === 每次运行创建独立目录 ===
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_tag)
    os.makedirs(run_dir, exist_ok=True)

    # # === 日志输出到屏幕 + 文件 ===
    # log_file = os.path.join(run_dir, "dse.log")
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    #     datefmt="%H:%M:%S",
    #     handlers=[
    #         logging.StreamHandler(),
    #         logging.FileHandler(log_file, encoding="utf-8")
    #     ]
    # )
    # === 日志输出到屏幕 + 文件 ===
    # log_file = os.path.join(run_dir, "dse.log")
    # logging.basicConfig(
    #     level=logging.WARNING,
    #     format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    #     datefmt="%H:%M:%S",
    #     handlers=[
    #         logging.StreamHandler(),
    #         logging.FileHandler(log_file, encoding="utf-8")
    #     ]
    # )
    log_file = os.path.join(run_dir, "dse.log")
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )

    mp.set_start_method("spawn", force=True)

    # num_workers = min(128, os.cpu_count() )
    # num_workers = os.cpu_count()
    # num_workers = 64
    # num_workers = os.cpu_count()//3
    # num_workers = min(128, os.cpu_count()//3)
    num_workers = os.cpu_count()//2
    log.info("Using %d workers (cpu_count=%s)", num_workers, os.cpu_count())

    # 把 run_dir 传给 dse_1()
    # 测量时间
    import time
    start_time = time.time()
    # 先关闭 proc_logs，仅保留 stats_bundles 下的新 dump（json/txt/md + manifest）
    dse_1(run_dir, num_workers, enable_proc_log=False)
    print("dse_1 finished")
    end_time = time.time()
    print("dse_1 time: %s seconds" % (end_time - start_time))
