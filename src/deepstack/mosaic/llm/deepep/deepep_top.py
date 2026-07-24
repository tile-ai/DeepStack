# deepep 里 low latency 是direct

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
from .deepep_coarse import deepep_coarse
from mosaic.arch.h200 import H200
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
        # return max_activation, mem_weight, deepep_coarse(bs, seq, hidden, moe_down_hidden, parallel, next_parallel, expert_bytes, granularity, single_chip, noc_hierarchy)
        # return deepep_coarse(bs, seq, hidden, moe_down_hidden, parallel, next_parallel, expert_bytes, granularity, single_chip, noc_hierarchy)
        return deepep_coarse(bs, seq, hidden, moe_down_hidden, parallel, next_parallel, expert_bytes, gate_bytes, num_shared_experts, num_routed_experts, num_activated_experts, routing_array, granularity, single_chip, noc_hierarchy)
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



if __name__ == "__main__":

    logging.basicConfig(
    level=logging.INFO,                              # 全局日志级别
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    )

    parallel = ParallelScheme(tp=1, ep=32, sp=1, cp=1, dp=1, pp=1, fsdp=False, ep1=32, ep2=1)
    next_parallel = ParallelScheme(tp=1, ep=32, sp=1, cp=1, dp=1, pp=1, fsdp=False, ep1=32, ep2=1)

    log.info("parallel: %s", parallel)
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=True)

    nvlink_bw = 450e9 * 0.8
    ib_bw = 400e9 * 0.8
    L1 = make_switch(8, hop_latency=2.5/2/1e6, link_bandwidth=nvlink_bw, switch_center_in_bw=nvlink_bw*4, switch_center_out_bw=nvlink_bw*4)
    L2 = make_switch(4, hop_latency=5/1e6, link_bandwidth=ib_bw, switch_center_in_bw=ib_bw*2, switch_center_out_bw=ib_bw*2)
    L3 = make_switch(1, hop_latency=0, link_bandwidth=ib_bw*10, switch_center_in_bw=ib_bw*10*8, switch_center_out_bw=ib_bw*10*8)

    noc_hierarchy = Hierarchy(layers=[L3, L2, L1], port_spread=PortSpread.EVEN, node_mapper=None, name="h100x32_medium")
    

    bs=128*4
    seq=1
    hidden=7168
    moe_down_hidden=3072

    num_shared_experts=0
    num_routed_experts=160
    num_activated_experts=8

    expert_bytes=OpBytes(
        input1=Tensor_Loc(dtype=torch.float16, loc="ddr"),
        input2=Tensor_Loc(dtype=torch.float8_e5m2, loc="ddr"),
        output=Tensor_Loc(dtype=torch.float16, loc="smem"),
    )

    gate_bytes=OpBytes(
        input1=Tensor_Loc(dtype=torch.float32, loc="ddr"),
        input2=Tensor_Loc(dtype=torch.float32, loc="ddr"),
        output=Tensor_Loc(dtype=torch.float32, loc="ddr"),
    )

    granularity=Modeling_Granularity(mode="coarse", comp_comm_overlap=True,auto_tune=False)
    single_chip=H200().set_to_microbench()

    max_activation, mem_weight = get_moe_footprint(bs=bs, seq=seq, hidden=hidden, moe_down_hidden=moe_down_hidden, parallel=parallel, expert_bytes=expert_bytes, gate_bytes=gate_bytes, num_shared_experts=num_shared_experts, num_routed_experts=num_routed_experts)
    log.info("max_activation: %s GiB, mem_weight: %s GiB", max_activation/(1024**3), mem_weight/(1024**3))


    # Resolve JSON path relative to this file so it works regardless of CWD
    project_root = Path(__file__).resolve().parent.parent.parent  # .../mosaic
    # json_trace_file = str(project_root / "data" / "routing_outputs" / "routing_iter128_N256_G8_TG4_E8.json")
    # npz_trace_file = str(project_root / "data" / "aime_ds_r1" / "moe_activations_batch0.npz")
    npz_trace_file = str(project_root / "data" / "aime_qwen_235b" / "qwen3_moe_activations_batch0.npz")
    tokens_total = bs * seq 
    # routing_list= import_results_as_expert_id_lists(json_trace_file, n=8192)
    prefill_array, decode_array = load_npz_routing_keep_shape(npz_trace_file, as_list=False)
    routing_array = decode_array


    e2e_time=moe_top(bs=bs, seq=seq, hidden=hidden, moe_down_hidden=moe_down_hidden, parallel=parallel, next_parallel=next_parallel, 
        expert_bytes=expert_bytes, gate_bytes=gate_bytes, num_shared_experts=num_shared_experts, num_routed_experts=num_routed_experts, num_activated_experts = num_activated_experts,  
        routing_array=routing_array, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # do not use fstring
