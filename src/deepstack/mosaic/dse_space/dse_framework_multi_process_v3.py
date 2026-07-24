# from mosaic.dse_space.arch_noc_combinations import arch_noc_combinations1
from typing import Any
from mosaic.dse_space.arch_noc_combinations import arch_noc_combinations1110, arch_noc_combinations_scale_64_512_nodes,h100_8_to_64_arch_noc_combinations
from mosaic.dse_space.task_combination import task_combination_2, customized_decoding_task_combination, scale_64_512_nodes_task_combination_decode
from mosaic.dse_space.parallel_schemes import all_parallel_schemes, filter_parallel_schemes_by_max, filter_illegal_fsdp, filter_parallel_schemes_by_product_max
from mosaic.llm_arch import DeepSeekV3, LLM_Arch, Qwen3_235b_a22b, Qwen3_480b_a35b, Llama3_70b, Llama3_405b
from mosaic.parallelism import ParallelScheme
from mosaic.utils.allocate_ep import allocate_ep
import logging
log = logging.getLogger(__name__) 
import math
import dataclasses
from mosaic.llm.rms_norm.rms_norm_top import rms_norm_top, get_rms_norm_footprint
from mosaic.llm.swiglu.swiglu_top import swiglu_top, get_swiglu_footprint
from mosaic.llm.add_residual.add_residual_top import add_residual_top, get_add_residual_footprint
from mosaic.llm.gqa.gqa_decode_top import get_gqa_decode_footprint, gqa_decode_top, gqa_decode_kv_list_top
from mosaic.llm.rope.get_rope_global_footprint import get_rope_global_footprint
from mosaic.llm.moe.moe_top import moe_top, get_moe_footprint
from mosaic.llm.mla.mla_footprint import get_mla_absorb_and_no_absorb_footprint
from mosaic.llm.mla.mla_decode_top import mla_decode_top, mla_decode_kv_list_top
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
from mosaic.cost.energy_record import EnergyRecord


MODEL_REG = {
    "DeepSeekV3": DeepSeekV3,
    "Qwen3_235b_a22b": Qwen3_235b_a22b,
    "Qwen3_480b_a35b": Qwen3_480b_a35b,
    "Llama3_70b": Llama3_70b,
    "Llama3_405b": Llama3_405b,
}

# 全局指定使用的 arch_noc 组合函数
# GET_ARCH_NOC_COMBINATIONS = arch_noc_combinations1110
# GET_ARCH_NOC_COMBINATIONS = h100_8_to_64_arch_noc_combinations
# GET_ARCH_NOC_COMBINATIONS = arch_noc_combinations_scale_64_512_nodes
from mosaic.dse_space.arch_noc_combinations import b200_8x1_8_arch_noc_combinations
GET_ARCH_NOC_COMBINATIONS = b200_8x1_8_arch_noc_combinations

# 全局指定使用的 task combination 函数
# GET_TASK_COMBINATIONS = task_combination_2
# GET_TASK_COMBINATIONS = customized_decoding_task_combination
# GET_TASK_COMBINATIONS = scale_64_512_nodes_task_combination_decode
from mosaic.dse_space.task_combination import dpsk_decode_task_combination
GET_TASK_COMBINATIONS = dpsk_decode_task_combination

# 子进程内的全局缓存，避免把 numpy 数组跨进程序列化
_ROUTING_ARRAY = None

def _init_worker(npz_trace_file: str):
    """每个子进程启动时执行：只在子进程里加载 routing array，一次到位"""
    from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
    global _ROUTING_ARRAY
    _, decode_array = load_npz_routing_keep_shape(npz_trace_file, as_list=False)
    _ROUTING_ARRAY = decode_array



def get_max_footprint_decode(model_arch:LLM_Arch, bs:int, seq:int, cached_kv:int, moe_parallel:ParallelScheme, non_moe_parallel:ParallelScheme):

    hidden = model_arch.hidden_size

    parallel = non_moe_parallel

    shard_layer = math.ceil(model_arch.num_layer/parallel.pp)
    if model_arch.gqa_arch is not None:
        wq_hidden = model_arch.gqa_arch.num_head * model_arch.gqa_arch.head_dim
        head_dim = model_arch.gqa_arch.head_dim
    elif model_arch.mla_arch is not None:
        wq_hidden = model_arch.mla_arch.num_head * model_arch.mla_arch.head_dim
        head_dim = model_arch.mla_arch.head_dim
    else:
        raise ValueError("model_arch must have one of gqa_arch or mla_arch")

    def get_gqa_footprint_wrapper(model_arch:LLM_Arch, cached_kv:int, parallel:ParallelScheme):
        from mosaic.llm.gqa.gqa_decode_top import get_gqa_decode_footprint
        num_head = model_arch.gqa_arch.num_head
        num_kv_head = model_arch.gqa_arch.num_kv_head
        head_dim = model_arch.gqa_arch.head_dim
        atten_parallel = parallel
        next_parallel = parallel
        atten_bytes = model_arch.gqa_arch.atten_bytes

        max_activation, mem_weight, mem_kv_cache = get_gqa_decode_footprint(
            bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, 
            num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim,
            parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, atten_bytes=atten_bytes
        )

        shard_layer = math.ceil(model_arch.num_layer/parallel.pp)
        return max_activation, mem_weight*shard_layer, mem_kv_cache*model_arch.num_layer

    def get_mla_footprint_wrapper(model_arch:LLM_Arch, cached_kv:int, parallel:ParallelScheme):
        max_activation, mem_weight, mem_kv_cache = get_mla_absorb_and_no_absorb_footprint(
            bs, seq, cached_kv, model_arch, parallel, parallel
        )
        shard_layer = math.ceil(model_arch.num_layer/parallel.pp)
        return max_activation, mem_weight*shard_layer, mem_kv_cache*model_arch.num_layer

    def get_dense_ffn_footprint_wrapper(model_arch:LLM_Arch, cached_kv:int, parallel:ParallelScheme):
        from mosaic.llm.swiglu.swiglu_top import get_swiglu_footprint

        up_hidden = model_arch.dense_ffn_arch.up_hidden
        swiglu_bytes = model_arch.dense_ffn_arch.swiglu_bytes

        max_activation, mem_weight = get_swiglu_footprint(bs, seq, hidden, up_hidden, parallel, swiglu_bytes)

        shard_layer = math.ceil(model_arch.num_dense_layer/parallel.pp)
        return max_activation, mem_weight*shard_layer

    def get_moe_footprint_wrapper(model_arch:LLM_Arch, cached_kv:int, parallel:ParallelScheme):
        num_shared_experts=model_arch.moe_arch.num_shared_experts
        num_routed_experts=model_arch.moe_arch.num_routed_experts
        num_activated_experts=model_arch.moe_arch.num_activated_experts
        moe_down_hidden=model_arch.moe_arch.moe_down_hidden
        expert_bytes=model_arch.moe_arch.expert_bytes
        gate_bytes=model_arch.moe_arch.gate_bytes

        max_activation, mem_weight = get_moe_footprint(
            bs, seq, hidden, moe_down_hidden, parallel, expert_bytes, gate_bytes,
            num_shared_experts, num_routed_experts
        )

        shard_layer = math.ceil(model_arch.num_moe_layer/parallel.pp)
        return max_activation, mem_weight*shard_layer

    if model_arch.gqa_arch is not None:
        gqa_max_activation, gqa_mem_weight, gqa_mem_kv_cache = get_gqa_footprint_wrapper(model_arch=model_arch, cached_kv=cached_kv, parallel=parallel)
        log.info("gqa_max_activation: %s GiB, gqa_mem_weight: %s GiB, gqa_mem_kv_cache: %s GiB", gqa_max_activation/(1024**3), gqa_mem_weight/(1024**3), gqa_mem_kv_cache/(1024**3))
    else:
        gqa_max_activation = 0
        gqa_mem_weight = 0
        gqa_mem_kv_cache = 0

    if model_arch.mla_arch is not None:
        mla_max_activation, mla_mem_weight, mla_mem_kv_cache = get_mla_footprint_wrapper(model_arch=model_arch, cached_kv=cached_kv, parallel=parallel)
        log.info("mla_max_activation: %s GiB, mla_mem_weight: %s GiB, mla_mem_kv_cache: %s GiB", mla_max_activation/(1024**3), mla_mem_weight/(1024**3), mla_mem_kv_cache/(1024**3))
    else:
        mla_max_activation = 0
        mla_mem_weight = 0
        mla_mem_kv_cache = 0

    if model_arch.dense_ffn_arch is not None:
        dense_ffn_max_activation, dense_ffn_mem_weight = get_dense_ffn_footprint_wrapper(model_arch=model_arch, cached_kv=cached_kv, parallel=parallel)
        log.info("dense_ffn_max_activation: %s GiB, dense_ffn_mem_weight: %s GiB", dense_ffn_max_activation/(1024**3), dense_ffn_mem_weight/(1024**3))
    else:
        dense_ffn_max_activation = 0
        dense_ffn_mem_weight = 0

    if model_arch.moe_arch is not None:
        moe_max_activation, moe_mem_weight = get_moe_footprint_wrapper(model_arch=model_arch, cached_kv=cached_kv, parallel=moe_parallel)
        log.info("moe_max_activation: %s GiB, moe_mem_weight: %s GiB", moe_max_activation/(1024**3), moe_mem_weight/(1024**3))
    else:
        moe_max_activation = 0
        moe_mem_weight = 0

    # rms_norm, 2blocks per layer
    act_rms, mem_weight_rms = get_rms_norm_footprint(bs=bs, seq=seq, hidden=hidden, parallel=parallel, rms_norm_bytes=model_arch.rms_norm_bytes)
    mem_weight_rms = mem_weight_rms * shard_layer * 2
    log.info("act_rms: %s GiB, mem_weight_rms: %s GiB", act_rms/(1024**3), mem_weight_rms/(1024**3))

    # rope global
    mem_weight_rope = get_rope_global_footprint(bs=bs, head_dim=head_dim, parallel=parallel, rope_bytes=model_arch.rope_bytes, MAX_ROPE_SEQ = model_arch.max_seq_len)
    log.info("mem_weight_rope: %s GiB", mem_weight_rope/(1024**3))

    # add_residual, 2blocks per layer
    act_add_residual, mem_weight_add_residual = get_add_residual_footprint(bs=bs, seq=seq, hidden=hidden, parallel=parallel, add_residual_bytes=model_arch.add_residual_bytes)
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

def modeling_decode(model_arch:LLM_Arch, bs:int, seq:int, cached_kv_list:list[int], moe_parallel:ParallelScheme, non_moe_parallel:ParallelScheme, single_chip:Arch, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, routing_array: np.ndarray):

    parallel = non_moe_parallel
    hidden = model_arch.hidden_size
    shard_layer = math.ceil(model_arch.num_layer/non_moe_parallel.pp)

    energy_record = EnergyRecord(parallel.world_size())

    # for energy calculation: 
    # we need a global record for all the tm_demo, reg_rw, smem_rw, l2_rw, dram_rw, sfu_ops, cuda_ops, tensor_ops

    

    def get_pipeline_time():

        if parallel.pp == 1:
            return 0

        shard_bs = math.ceil(bs / non_moe_parallel.dp)
        shard_seq = math.ceil(seq / non_moe_parallel.sp)
        hidden = model_arch.hidden_size
        activation_bytes = model_arch.add_residual_bytes.input1.num_bytes

        pipeline_activation_bytes = shard_bs * shard_seq * hidden * activation_bytes

        pp_time_list = []
        
        for i in range(parallel.pp - 1):
            traffic_pair = [[pipeline_activation_bytes, i, (i+1)%parallel.pp]]
            tm = TrafficMatrix(parallel.world_size())
            tm.add_intra_group_traffic_pair_bulk("pp", traffic_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
            pp_hop_latency, pp_ext_max, pp_overall_time, pp_traffic = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
            pp_time_list.append(pp_overall_time)

        energy_record.add_tm(tm)
        # 取流水段之间的最大时延；若为空则为 0
        return max(pp_time_list) if pp_time_list else 0.0

    pp_p2p_time = get_pipeline_time()

    time_gqa_list = []
    time_mla_list = []

    # for cached_kv in cached_kv_list:

    #     if model_arch.gqa_arch is not None:
    #         time_gqa = gqa_decode_top(bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, num_head=model_arch.gqa_arch.num_head, num_kv_head=model_arch.gqa_arch.num_kv_head, head_dim=model_arch.gqa_arch.head_dim, parallel=parallel, atten_parallel=parallel, next_parallel=parallel, atten_bytes=model_arch.gqa_arch.atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    #         time_gqa = time_gqa * shard_layer
    #     else:
    #         time_gqa = 0

    #     if model_arch.mla_arch is not None:
    #         time_mla = mla_decode_top(bs=bs, seq=seq, cached_kv=cached_kv, model_arch=model_arch, parallel=parallel, atten_parallel=parallel, next_parallel=parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    #         time_mla = time_mla * shard_layer
    #     else:
    #         time_mla = 0

    #     time_gqa_list.append(time_gqa)
    #     time_mla_list.append(time_mla)

    if model_arch.gqa_arch is not None:
        time_gqa_list = gqa_decode_kv_list_top(bs=bs, seq=seq, cached_kv_list=cached_kv_list, hidden=hidden, num_head=model_arch.gqa_arch.num_head, num_kv_head=model_arch.gqa_arch.num_kv_head, head_dim=model_arch.gqa_arch.head_dim, parallel=parallel, atten_parallel=parallel, next_parallel=parallel, atten_bytes=model_arch.gqa_arch.atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        time_gqa_list = [x * shard_layer for x in time_gqa_list]
    else:
        time_gqa_list = [0] * len(cached_kv_list)

    if model_arch.mla_arch is not None:
        time_mla_list = mla_decode_kv_list_top(bs=bs, seq=seq, cached_kv_list=cached_kv_list, model_arch=model_arch, parallel=parallel, atten_parallel=parallel, next_parallel=parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        time_mla_list = [x * shard_layer for x in time_mla_list]
    else:
        time_mla_list = [0] * len(cached_kv_list)

    if model_arch.dense_ffn_arch is not None:
        time_dense_ffn, _ = swiglu_top(bs=bs, seq=seq, hidden=hidden, up_hidden=model_arch.dense_ffn_arch.up_hidden, parallel=parallel, next_parallel=parallel, swiglu_bytes=model_arch.dense_ffn_arch.swiglu_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        shard_dense_layer = math.ceil(model_arch.num_dense_layer/parallel.pp)
        time_dense_ffn = time_dense_ffn * shard_dense_layer
    else:
        time_dense_ffn = 0

    if model_arch.moe_arch is not None:
        time_moe, _ = moe_top(bs=bs, seq=seq, hidden=hidden, moe_down_hidden=model_arch.moe_arch.moe_down_hidden, parallel=moe_parallel, next_parallel=moe_parallel, expert_bytes=model_arch.moe_arch.expert_bytes, gate_bytes=model_arch.moe_arch.gate_bytes, num_shared_experts=model_arch.moe_arch.num_shared_experts, num_routed_experts=model_arch.moe_arch.num_routed_experts, num_activated_experts=model_arch.moe_arch.num_activated_experts, routing_array=routing_array, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        shard_moe_layer = math.ceil(model_arch.num_moe_layer/parallel.pp)
        time_moe = time_moe * shard_moe_layer
    else:
        time_moe = 0   

    time_rms_norm, _ = rms_norm_top(bs=bs, seq=seq, hidden=hidden, parallel=parallel, next_parallel=parallel, rms_norm_bytes=model_arch.rms_norm_bytes, 
        granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    time_rms_norm = time_rms_norm * shard_layer * 2

    time_add_residual=add_residual_top(bs=bs, seq=seq, hidden=hidden, parallel=parallel, next_parallel=parallel, add_residual_bytes=model_arch.add_residual_bytes, 
        granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    time_add_residual = time_add_residual * shard_layer * 2

    # time_total = pp_p2p_time + time_gqa + time_mla + time_dense_ffn + time_moe + time_rms_norm + time_add_residual
    # log.info("p2p time: %s s, gqa time: %s s, mla time: %s s, dense_ffn time: %s s, moe time: %s s, rms_norm time: %s s, add_residual time: %s s", pp_p2p_time, time_gqa, time_mla, time_dense_ffn, time_moe, time_rms_norm, time_add_residual)
    # log.info("total time: %s s", time_total)

    # here we got tm_energy_record

    time_total_list = []
    for i in range(len(cached_kv_list)):
        time_total = pp_p2p_time + time_gqa_list[i] + time_mla_list[i] + time_dense_ffn + time_moe + time_rms_norm + time_add_residual
        # time_total_list.append(time_total)
        time_total_list.append((pp_p2p_time, time_gqa_list[i], time_mla_list[i], time_dense_ffn, time_moe, time_rms_norm, time_add_residual, time_total))
        log.info("p2p time: %s s, gqa time: %s s, mla time: %s s, dense_ffn time: %s s, moe time: %s s, rms_norm time: %s s, add_residual time: %s s", pp_p2p_time, time_gqa_list[i], time_mla_list[i], time_dense_ffn, time_moe, time_rms_norm, time_add_residual)
        log.info("total time: %s s", time_total)

    return time_total_list

# ---------- 多进程 worker：单个 (scheme, kv_len) 任务 ----------
def _compute_time_row(args):
    (
        combo_idx,              # int -> arch_noc_combinations1()[combo_idx]
        model_key,              # str -> MODEL_REG[model_key]()
        BS, minibatch, sampled_kv_len, seq,
        parallel_scheme,        # ParallelScheme (dataclass，顶层定义，可 pickle)
        max_activation, global_total_weight, global_kv_cache,
        granularity_tuple,      # (mode, comp_comm_overlap, auto_tune) 纯标量
        non_moe_parallel        # ParallelScheme
    ) = args

    # 在子进程里重建 heavy 对象
    # arch, noc = arch_noc_combinations1110()[combo_idx]
    arch, noc = GET_ARCH_NOC_COMBINATIONS()[combo_idx]
    # arch, noc = h100_8_to_64_arch_noc_combinations()[combo_idx]
    model_arch = MODEL_REG[model_key]()

    # 重建 granularity
    granularity = Modeling_Granularity(
        mode=granularity_tuple[0],
        comp_comm_overlap=granularity_tuple[1],
        auto_tune=granularity_tuple[2],
    )

    # 拿到子进程全局 routing_array
    global _ROUTING_ARRAY
    routing_array = _ROUTING_ARRAY
    if routing_array is None:
        raise RuntimeError("routing_array not initialized in worker")

    time_list = []
    sampled_kv_len_utps_stps_list = []
    utps_all = 0
    stps_all = 0
    # for kv_len in sampled_kv_len:
    #     # 计算 time/utps/stps
    #     time = modeling_decode(
    #         model_arch=model_arch, bs=minibatch, seq=seq, cached_kv_list=sampled_kv_len,
    #         moe_parallel=parallel_scheme, non_moe_parallel=non_moe_parallel,
    #         single_chip=arch, noc_hierarchy=noc, granularity=granularity,
    #         routing_array=routing_array
    #     )
    #     stps = minibatch / time
    #     utps =  1 / time / parallel_scheme.pp
    #     time_list.append(time)
    #     stps_list.append(stps)
    #     utps_list.append(utps)
    #     utps_all += utps
    #     stps_all += stps
    #     sampled_kv_len_utps_stps_list.append((kv_len, utps, stps))
    time_list = modeling_decode(
        model_arch=model_arch, bs=minibatch, seq=seq, cached_kv_list=sampled_kv_len,
        moe_parallel=parallel_scheme, non_moe_parallel=non_moe_parallel,
        single_chip=arch, noc_hierarchy=noc, granularity=granularity,
        routing_array=routing_array
    )

    for i in range(len(sampled_kv_len)):
        pp_p2p_time, time_gqa, time_mla, time_dense_ffn, time_moe, time_rms_norm, time_add_residual, time_total = time_list[i]

        time = time_total
        kv_len = sampled_kv_len[i]
        stps = minibatch / time
        utps =  1 / time / parallel_scheme.pp
        utps_all += utps
        stps_all += stps
        sampled_kv_len_utps_stps_list.append((kv_len, time_gqa, time_mla, utps, stps))
    

    # 时间随着kv len 增长是线性的
    # time_all = 

    # utps = minibatch / time
    # stps = 1 / time

    utps_average = utps_all / len(sampled_kv_len)
    stps_average = stps_all / len(sampled_kv_len)

    # 打包 sampled_kv_len, 

    

    # 返回轻量结果（字符串/数值/小对象）
    # return [
    #     arch.__class__.__name__, noc.name, model_arch.__class__.__name__,
    #     BS, minibatch, 1, kv_len,
    #     str(parallel_scheme),                  # 返回字符串更稳妥
    #     max_activation/(1024**3),
    #     global_total_weight/(1024**3),
    #     global_kv_cache/(1024**3),
    #     utps, stps
    # ]
    return [
        arch.__class__.__name__, noc.name, model_arch.__class__.__name__,
        BS, minibatch, seq,
        parallel_scheme,                  # 返回字符串更稳妥
        max_activation/(1024**3),
        global_total_weight/(1024**3),
        global_kv_cache/(1024**3),
        sampled_kv_len_utps_stps_list,
        pp_p2p_time, time_dense_ffn, time_moe, time_rms_norm, time_add_residual,
        utps_average, stps_average
    ]

def dse_1(run_dir, num_workers):
    """
    1) 先在主进程过滤 valid parallel schemes，并把 footprint 记下来；
    2) 然后把 (combo_idx, model_key, kv_len, scheme, ...) 这些可序列化的轻量键组装成任务；
    3) 用多进程并行计算 time/utps/stps（num_workers）；
    4) 主进程统一写入 CSV，避免多进程文件竞争。
    """
    # ---- 全局建模粒度（传入子进程用纯标量三元组）----
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False)
    gran_tuple = (granularity.mode, granularity.comp_comm_overlap, granularity.auto_tune)

    NUM_KV_POINT = 4



    # ---- 任务组合（可以自行增删）----
    # task_combinations = [
    #     # [DeepSeekV3(), 16, 1024, 16384,1],
    #     [Qwen3_235b_a22b(), 16, 1024, 64 * 1024,1],
    #     # [Llama3_70b(), 16, 1024, 64*1024, 1]
    # ]
    # from mosaic.dse_space import task_combination_2, customized_decoding_task_combination
    # task_combinations = task_combination_2()
    # task_combinations = customized_decoding_task_combination()
    # from mosaic.dse_space.task_combination import scale_64_512_nodes_task_combination_decode
    # task_combinations = scale_64_512_nodes_task_combination_decode()
    task_combinations = GET_TASK_COMBINATIONS()

    # ---- 枚举 arch/noc 组合 与 并行方案集合 ----
    # combinations = arch_noc_combinations1110()
    # from mosaic.dse_space.arch_noc_combinations import arch_noc_combinations_scale_64_512_nodes
    combinations = GET_ARCH_NOC_COMBINATIONS()
    # combinations = h100_8_to_64_arch_noc_combinations()
    # from mosaic.arch import stacked_gpu_base
    # from mosaic.noc.noc_config_set import torus_mesh_switch_1
    # combinations = [[stacked_gpu_base(), torus_mesh_switch_1()]]


    for task_combination in task_combinations:
        model_arch = task_combination[0]
        BS = task_combination[1]
        INPUT_SEQ = task_combination[2]
        TASK_MAX_SEQ = task_combination[3]
        # for parallel decoding
        PARALLEL_SEQ = task_combination[4]
        
        # ---- routing trace 的 npz 路径：由进程池 initializer 在子进程里加载 ----
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent  # .../mosaic
        if model_arch.__class__.__name__ == "DeepSeekV3":
            npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")
        elif model_arch.__class__.__name__ == "Qwen3_235b_a22b":
            npz_trace_file = str(project_root / "data" / "aime_qwen_235b" / "qwen3_moe_activations_batch0.npz")
        else:
            # won't be used, just for compatibility
            npz_trace_file = npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")





        MAX_KV_LEN = min(TASK_MAX_SEQ, model_arch.max_seq_len)

        # 采样若干 KV 长度点（含起点和终点）
        assert NUM_KV_POINT >= 2
        if NUM_KV_POINT == 2:
            sampled_kv_len = (INPUT_SEQ, MAX_KV_LEN)
        else:
            step = (MAX_KV_LEN - INPUT_SEQ) / (NUM_KV_POINT - 1)
            sampled_kv_len = tuple(int(round(INPUT_SEQ + i * step)) for i in range(NUM_KV_POINT))

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
                    "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB"
                ]
                for i in range(NUM_KV_POINT):
                    idx = i + 1
                    header.extend([f"kv_len_{idx}", f"time_gqa_{idx}ms", f"time_mla_{idx}ms", f"utps_{idx}", f"stps_{idx}"])
                header.extend(["pp_p2p_time_ms", "time_dense_ffn_ms", "time_moe_ms", "time_rms_norm_ms", "time_add_residual_ms", "utps_avg", "stps_avg"])
                writer.writerow(header)
        if (not os.path.exists(invalid_csv_path)) or os.stat(invalid_csv_path).st_size == 0:
            with open(invalid_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["arch", "noc", "model", "bs", "minibatch", "seq", "cached_kv",
                                 "tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp",
                                 "max_activation/GiB", "mem_weight/GiB", "kv_cache/GiB"])

        # ---- 遍历所有 (arch, noc) 组合 ----
        for combo_idx, combination in enumerate[Any](combinations):
            arch = combination[0]
            noc = combination[1]

            num_nodes = int(noc.name.rsplit("_", 1)[-1])
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


            valid_parallel_schemes: list[ParallelScheme] = []
            # 记录 footprint，避免重复计算；key 为 ParallelScheme（你已实现 __hash__）
            # value: (minibatch, max_activation, global_total_weight, global_kv_cache, non_moe_parallel)
            scheme_footprints: dict[ParallelScheme, tuple] = {}

            # ---- 先筛有效方案（主进程内进行，不涉及多进程）----
            for scheme in filtered_parallel_schemes:
                minibatch = math.ceil(BS / scheme.pp)
                allocate_ep(parallel=scheme, bs=minibatch, seq=PARALLEL_SEQ)

                non_moe_parallel = dataclasses.replace(
                    scheme, ep=1, ep1=1, ep2=1, dp=scheme.dp * scheme.ep
                )
                max_activation, global_total_weight, global_kv_cache = get_max_footprint_decode(
                    model_arch=model_arch, bs=minibatch, seq=PARALLEL_SEQ, cached_kv=MAX_KV_LEN,
                    moe_parallel=scheme, non_moe_parallel=non_moe_parallel
                )

                if ((max_activation + global_total_weight + global_kv_cache) <= 1.0 * arch.ddr_capacity):
                    log.info("parallel scheme: %s", scheme)
                    log.info("max_activation: %s GiB, global_total_weight: %s GiB, global_kv_cache: %s GiB",
                             max_activation/(1024**3), global_total_weight/(1024**3), global_kv_cache/(1024**3))
                    log.info("valid")

                    valid_parallel_schemes.append(scheme)
                    scheme_footprints[scheme] = (minibatch, max_activation, global_total_weight, global_kv_cache, non_moe_parallel)
                else:
                    log.info("parallel scheme: %s", scheme)
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

            log.info("model_arch: %s, bs: %s, input_seq: %s, task_max_seq: %s",
                     model_arch.__class__.__name__, BS, INPUT_SEQ, TASK_MAX_SEQ)
            log.info("valid parallel schemes length: %s", len(valid_parallel_schemes))
            if not valid_parallel_schemes:
                continue

            # ---- 组装并行任务（只传键/标量/小对象）----
            tasks = []
            model_key = model_arch.__class__.__name__
            for scheme in valid_parallel_schemes:
                minibatch, max_activation, global_total_weight, global_kv_cache, non_moe_parallel = scheme_footprints[scheme]
                # for kv_len in sampled_kv_len:
                #     tasks.append((
                #         combo_idx,              # arch/noc 的索引，子进程里重建
                #         model_key,              # 模型名，子进程里重建
                #         BS, minibatch, kv_len,PARALLEL_SEQ,
                #         scheme,                 # 顶层 dataclass，可 pickle
                #         max_activation, global_total_weight, global_kv_cache,
                #         gran_tuple,             # 纯标量三元组
                #         non_moe_parallel        # 顶层 dataclass，可 pickle
                #     ))
                tasks.append((
                    combo_idx,              # arch/noc 的索引，子进程里重建
                    model_key,              # 模型名，子进程里重建
                    BS, minibatch, sampled_kv_len, PARALLEL_SEQ,
                    scheme,                 # 顶层 dataclass，可 pickle
                    max_activation, global_total_weight, global_kv_cache,
                    gran_tuple,             # 纯标量三元组
                    non_moe_parallel        # 顶层 dataclass，可 pickle
                ))

            

            rows = []
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

            # ---- 主进程统一写 CSV ----
            if rows:
                # 排序：arch, noc, scheme_str
                rows.sort(key=lambda r: (r[0], r[1], str(r[6])))
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    # 将 (kv_len, time_gqa, time_mla, utps, stps) 列表展平到列，匹配表头
                    flat_rows = []
                    for r in rows:
                        scheme = r[6]
                        base = [
                            r[0], r[1], r[2], r[3], r[4], r[5],
                            scheme.tp, scheme.ep, scheme.ep1, scheme.ep2, scheme.sp, scheme.cp, scheme.dp, scheme.fsdp, scheme.pp,
                            r[7], r[8], r[9]
                        ]
                        for kv_len, time_gqa, time_mla, utps, stps in r[10]:
                            base.extend([kv_len, time_gqa * 1000, time_mla * 1000, utps, stps])
                        base.extend([r[11] * 1000, r[12] * 1000, r[13] * 1000, r[14] * 1000, r[15] * 1000, r[16], r[17]])
                        flat_rows.append(base)
                    writer.writerows(flat_rows)



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
    num_workers = min(128, os.cpu_count()//2)
    log.info("Using %d workers (cpu_count=%s)", num_workers, os.cpu_count())

    # 把 run_dir 传给 dse_1()
    # 测量时间
    import time
    start_time = time.time()
    dse_1(run_dir, num_workers)
    print("dse_1 finished")
    end_time = time.time()
    print("dse_1 time: %s seconds" % (end_time - start_time))