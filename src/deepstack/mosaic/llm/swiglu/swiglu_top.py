import torch
import math
import numpy as np
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, make_mesh_or_torus, make_switch, TopoKind, PortSpread, get_extend_max_routes, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
import cProfile
import logging
from tilesight.arch import *
from .swiglu_coarse import swiglu_coarse

log = logging.getLogger(__name__) 
from mosaic.cost.energy_record import EnergyRecord

def swiglu_top(bs:int, seq:int, hidden:int, up_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, swiglu_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    # ( (x @ W1) ⊗ silu(x @ W2) @ W3
    # assert parallel.cp==1
    # assert parallel.ep==1

    if seq == 1:
        assert parallel.sp == 1

    # max_activation, mem_weight = get_swiglu_footprint(bs, seq, hidden, up_hidden, parallel, swiglu_bytes)
    # log.info("max_activation: %s GiB, mem_weight: %s GiB", max_activation/(1024**3), mem_weight/(1024**3))
    
    if granularity.get_mode() == "coarse":
        # return max_activation, mem_weight, swiglu_coarse(bs, seq, hidden, up_hidden, parallel, next_parallel, swiglu_bytes, granularity, single_chip, noc_hierarchy)
        total_time, stats = swiglu_coarse(bs, seq, hidden, up_hidden, parallel, next_parallel, swiglu_bytes, granularity, single_chip, noc_hierarchy)
        return total_time, stats
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass
    
def get_swiglu_footprint(bs: int, seq: int, hidden: int, up_hidden: int, parallel: ParallelScheme, swiglu_bytes: OpBytes):
    """
    估算 SWiGLU 在单设备上的内存足迹。

    返回一个二元组:
    - max_activation: 峰值激活内存 (bytes)
    - mem_weight: 权重内存 (bytes)
    """

    in_bytes, weight_bytes, out_bytes = swiglu_bytes.get_dtype_bytes()
    in_bytes_u = np.uint64(in_bytes)
    weight_bytes_u = np.uint64(weight_bytes)
    out_bytes_u = np.uint64(out_bytes)

    # x: [bs/dp, seq/sp, hidden]
    # W1,W2 [hidden, up_hidden/tp]
    # W3 [up_hidden/tp, hidden]

    # x@W1 -> [bs/dp, seq/sp, up_hidden/tp]
    # x@W2 -> [bs/dp, seq/sp, up_hidden/tp]
    # silu-> [bs/dp, seq/sp, up_hidden/tp]
    # (x @ W1) ⊗ silu(x @ W2) -> [bs/dp, seq/sp, up_hidden/tp]
    # @ W3 -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce -> [bs/dp, seq/sp, hidden]

    # let's model it one-by-one
    # 1. x@W1 -> [bs/dp, seq/sp, hidden] @ [hidden, up_hidden/tp] -> [bs/dp, seq/sp, up_hidden/tp]
    # x: [shard_bs, shard_seq, shard_hidden]
    # W1: [shard_hidden, shard_up_hidden]
    # x@W1: [shard_bs, shard_seq, shard_up_hidden]

    # 权重按张量并行的分片尺寸 (使用整型上取整)
    bs_u = np.uint64(bs)
    seq_u = np.uint64(seq)
    hidden_u = np.uint64(hidden)
    up_hidden_u = np.uint64(up_hidden)
    tp_u = np.uint64(parallel.tp)
    dp_u = np.uint64(parallel.dp)
    sp_u = np.uint64(parallel.sp)

    shard_hidden = (hidden_u + tp_u - np.uint64(1)) // tp_u
    # up_hidden 可能为浮点（如 8192*2.5），先按 tp 做浮点除法再 ceil，再转为 uint64
    # shard_up_hidden = np.uint64(math.ceil(float(up_hidden) / float(parallel.tp)))
    shard_up_hidden = (up_hidden_u + tp_u - np.uint64(1)) // tp_u
    shard_bs = (bs_u + dp_u - np.uint64(1)) // dp_u
    shard_seq = (seq_u + sp_u - np.uint64(1)) // sp_u

    # 公共系数与中间量
    tokens_per_shard = shard_bs * shard_seq
    up_act_bytes = tokens_per_shard * shard_up_hidden * in_bytes_u
    hidden_act_bytes = tokens_per_shard * shard_hidden * in_bytes_u
    input_act_bytes = tokens_per_shard * hidden_u * in_bytes_u
    all_reduce_bytes = tokens_per_shard * hidden_u * out_bytes_u * np.uint64(2)

    # 峰值激活估计（覆盖四个关键阶段）
    max_activation = max(
        input_act_bytes + np.uint64(2) * up_act_bytes,
        np.uint64(3) * up_act_bytes,
        np.uint64(2) * up_act_bytes,
        hidden_act_bytes + all_reduce_bytes
    )

    # 权重内存；FSDP 时按数据并行维度均分
    dp_divisor = np.uint64(parallel.dp) if parallel.fsdp else np.uint64(1)
    w1_total = hidden_u * shard_up_hidden * weight_bytes_u
    w2_total = hidden_u * shard_up_hidden * weight_bytes_u
    w3_total = shard_up_hidden * shard_hidden * weight_bytes_u
    w1_weight = w1_total // dp_divisor
    w2_weight = w2_total // dp_divisor
    w3_weight = w3_total // dp_divisor
    mem_weight = w1_weight + w2_weight + w3_weight

    return np.uint64(max_activation), np.uint64(mem_weight)
