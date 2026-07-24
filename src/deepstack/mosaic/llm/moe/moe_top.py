"""Parallel partition for MoE inference."""

import torch
import math
import numpy as np
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, make_mesh_or_torus, make_switch, TopoKind, PortSpread, get_extend_max_routes, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity, allocate_ep
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
import cProfile
import logging
from tilesight.arch import *
# from mosaic.arch.arch_base import Arch
from mosaic.arch import *
from .moe_coarse import moe_coarse
# from mosaic.utils import allocate_ep, import_results_as_expert_id_lists
from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)
from mosaic.cost.energy_record import EnergyRecord

def moe_top(bs:int, seq:int, hidden:int, moe_down_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, expert_bytes:OpBytes, gate_bytes:OpBytes, num_shared_experts:int, num_routed_experts:int, num_activated_experts:int, routing_array:np.ndarray, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    """
    MoE's EP means dp/sp in other modules without experts
    """

    # function-level call counter
    # moe_top.call_count = getattr(moe_top, "call_count", 0) + 1
    # log.debug("moe_top call #%s", moe_top.call_count)

    # ( (x @ W1) ⊗ silu(x @ W2) @ W3
    # assert parallel.cp==1
    # assert parallel.ep==1
    if parallel.ep != 1 and (parallel.ep1 is None or parallel.ep2 is None):
        log.warning("parallel.ep1 is None or parallel.ep2 is None, allocate ep")
        parallel = allocate_ep(parallel, bs, seq)

    if seq == 1:
        assert parallel.sp == 1
        assert parallel.ep2 == 1



    # max_activation, mem_weight = get_moe_footprint(bs, seq, hidden, moe_down_hidden, parallel, expert_bytes)
    # log.info("max_activation: %s GiB, mem_weight: %s GiB", max_activation/(1024**3), mem_weight/(1024**3))

    if granularity.get_mode() == "coarse":
        # return max_activation, mem_weight, moe_coarse(bs, seq, hidden, moe_down_hidden, parallel, next_parallel, expert_bytes, granularity, single_chip, noc_hierarchy)
        # return moe_coarse(bs, seq, hidden, moe_down_hidden, parallel, next_parallel, expert_bytes, granularity, single_chip, noc_hierarchy)
        total_time, stats = moe_coarse(bs, seq, hidden, moe_down_hidden, parallel, next_parallel, expert_bytes, gate_bytes, num_shared_experts, num_routed_experts, num_activated_experts, routing_array, granularity, single_chip, noc_hierarchy)
        return total_time, stats
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass

def get_moe_footprint(bs: int, seq: int, hidden: int, moe_down_hidden: int, parallel: ParallelScheme, expert_bytes: OpBytes, gate_bytes: OpBytes, num_shared_experts: int, num_routed_experts: int):
    """
    估算 moe 在单设备上的内存足迹。

    返回一个二元组:
    - max_activation: 峰值激活内存 (bytes)
    - mem_weight: 权重内存 (bytes)
    """

    # we assume each device hold the complete shared experts, gating & routing matrix
    # for non-shared experts, they are distributed across devices with EP-level parallelism
    # input activations (bs, seq) are distributed across devices with EP-level parallelism (extra additional)

    in_bytes, expert_weight_bytes, out_bytes = expert_bytes.get_dtype_bytes()
    _, gate_weight_bytes, routing_output_bytes = gate_bytes.get_dtype_bytes()

    # 1 for gating & dispatch
    # x:[bs/dp/ep1, seq/sp/ep2, hidden]
    # gate: [hidden, num_routed_experts]
    # gated scores: [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    # do some normalization by scores for weights


    # 2.1 for shared experts
    # x:[bs/dp/ep1, seq/sp/ep2, hidden]
    # shared experts: W1,W2 num_shared_experts * [hidden, moe_down_hidden/tp], W3 num_shared_experts * [moe_down_hidden/tp, hidden]
    # swiglu
    # the device who's reponsible for this token will do the shared experts computation

    # 2.2 for non-shared experts
    # all-to-all routing
    # x: [bs/dp/ep1, seq/sp/ep2, hidden]
    # W1,W2 num_routed_experts / ep * [hidden, moe_down_hidden/tp]
    # W3 num_routed_experts / ep * [moe_down_hidden/tp, hidden]
    # swiglu
    # [bs/dp/ep1] * scores/norm_scores  (element-wise multiplication) -> [bs/dp/ep1, seq/sp/ep2, num_routed_experts]


    # all the experts doing:
    # x@W1 -> [bs/dp/ep1, seq/sp/ep2, moe_down_hidden/tp]
    # x@W2 -> [bs/dp/ep1, seq/sp/ep2, moe_down_hidden/tp]
    # silu-> [bs/dp/ep1, seq/sp/ep2, moe_down_hidden/tp]
    # (x @ W1) ⊗ silu(x @ W2) -> [bs/dp/ep1, seq/sp/ep2, moe_down_hidden/tp]
    # @ W3 -> [bs/dp/ep1, seq/sp/ep2, hidden]
    # tp-level all-reduce -> [bs/dp/ep1, seq/sp/ep2, hidden]

    # 3. accumulation
    # all-to-all routing back the experts
    # the device who's reponsible for this token will do the accumulation: num_shared_experts + num_routed_experts, reduce-sum

    # let's model it one-by-one
    # 1. x@W1 -> [bs/dp, seq/sp, hidden] @ [hidden, moe_down_hidden/tp] -> [bs/dp, seq/sp, moe_down_hidden/tp]
    # x: [shard_bs, shard_seq, shard_hidden]
    # W1: [shard_hidden, shard_moe_down_hidden]
    # x@W1: [shard_bs, shard_seq, shard_moe_down_hidden]

    shard_bs = math.ceil(bs / parallel.dp / parallel.ep1)
    shard_seq = math.ceil(seq / parallel.sp / parallel.ep2)
    # shard_hidden = math.ceil(hidden / parallel.tp)
    shard_moe_down_hidden = math.ceil(moe_down_hidden / parallel.tp)
    num_shard_routed_experts = math.ceil(num_routed_experts / parallel.ep)


    # weight memory
    expert_weight1 = hidden * shard_moe_down_hidden * expert_weight_bytes
    expert_weight2 = hidden * shard_moe_down_hidden * expert_weight_bytes
    expert_weight3 = shard_moe_down_hidden * hidden * expert_weight_bytes


    shared_experts_weight = num_shared_experts * (expert_weight1 + expert_weight2 + expert_weight3)
    routed_experts_weight = num_shard_routed_experts * (expert_weight1 + expert_weight2 + expert_weight3)
    gate_weight = hidden * num_routed_experts * gate_weight_bytes

    # 权重内存；FSDP 时按数据并行维度均分
    dp_divisor = np.uint64(parallel.dp) if parallel.fsdp else np.uint64(1)
    mem_weight = (shared_experts_weight + routed_experts_weight) // dp_divisor + gate_weight

    # activation memory
    input_act_bytes = shard_bs * shard_seq * hidden * in_bytes
    shared_experts_act_bytes = shard_bs * shard_seq * shard_moe_down_hidden * in_bytes * num_shared_experts
    routed_experts_act_bytes = shard_bs * shard_seq * shard_moe_down_hidden * in_bytes * num_shard_routed_experts * min (2, parallel.ep)
    gate_act_bytes = shard_bs * shard_seq * num_routed_experts * in_bytes

    all_reduce_bytes = shard_bs * shard_seq * hidden * out_bytes * 2

    # 峰值激活估计（覆盖四个关键阶段）
    max_activation = max(
        input_act_bytes + shared_experts_act_bytes + routed_experts_act_bytes + gate_act_bytes,
        shared_experts_act_bytes + routed_experts_act_bytes + gate_act_bytes + all_reduce_bytes
    )

    return max_activation, mem_weight
