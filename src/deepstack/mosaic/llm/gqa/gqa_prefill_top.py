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
from .gqa_prefill_coarse import gqa_prefill_coarse

log = logging.getLogger(__name__) 
from mosaic.cost.energy_record import EnergyRecord
from mosaic.cost.op_perf_stats import OpPerfStats

def get_gqa_prefill_footprint(bs:int, seq:int, cached_kv:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme, atten_bytes:OpBytes):
    
    group_size = math.ceil (num_head/num_kv_head)
    wq_hidden = num_head * head_dim

    # 其中x [bs/dp, seq/sp, hidden], Wq, [hidden, hidden/tp], Wk, Wv [hidden, hidden/g/tp], Wo [hidden/tp, hidden]
    # Attention 中Q.sp * KV.cp == Global.sp 

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_head = math.ceil(num_head / parallel.tp)
    shard_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_seq_kv = math.ceil(seq / atten_parallel.cp)

    shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)


    # max_activation
    q_weight = hidden * shard_hidden * weight_bytes
    k_weight = hidden * shard_hidden / group_size * weight_bytes
    v_weight = hidden * shard_hidden / group_size * weight_bytes
    o_weight = shard_hidden * hidden * weight_bytes
    
    
    
    input_act = in_bytes * shard_bs *shard_seq * hidden
    q_act = in_bytes * shard_bs *shard_seq_q * shard_hidden
    k_act = in_bytes * shard_bs *shard_seq_kv * shard_hidden / group_size
    v_act = in_bytes * shard_bs *shard_seq_kv * shard_hidden / group_size

    k_cache = in_bytes * shard_bs * shard_cached_kv * shard_hidden / group_size 
    v_cache = in_bytes * shard_bs * shard_cached_kv * shard_hidden / group_size 
    
    p_act = in_bytes * shard_bs * math.ceil(seq/atten_parallel.sp) * shard_hidden 
    p_reduce_act = in_bytes * shard_bs * math.ceil(seq/atten_parallel.sp) * shard_hidden * 3

    o_act = out_bytes * shard_bs * shard_seq * hidden 
    o_reduce_act = out_bytes * shard_bs * shard_seq * hidden * 3

    # mem_weight = q_weight + k_weight + v_weight + o_weight + k_act + v_act
    mem_weight = q_weight + k_weight + v_weight + o_weight
    mem_kv_cache = k_cache + v_cache
    max_activation = max(( q_act + k_act + v_act + max(p_act,input_act)), (p_act + p_reduce_act) , (p_reduce_act + o_act), (o_act + o_reduce_act))
    # log.info("max_activation candidates: %s, %s, %s, %s",( q_act + k_act + v_act + max(p_act,input_act)), (p_act + p_reduce_act) , (p_reduce_act + o_act), (o_act + o_reduce_act) )
    dp_divisor = np.uint64(parallel.dp) if parallel.fsdp else np.uint64(1)

    mem_weight = mem_weight // dp_divisor

    return max_activation, mem_weight, mem_kv_cache
    
    # shard_bs = math.ceil(bs / parallel.dp)
    # shard_seq = math.ceil(seq / parallel.sp)
    # shard_head = math.ceil(num_head / parallel.tp)
    # shard_kv_head = math.ceil(num_kv_head / parallel.tp)
    # return in_bytes * shard_bs * shard_seq * shard_head * shard_kv_head * out_bytes, weight_bytes * shard_bs * shard_seq * shard_head * shard_kv_head * out_bytes

def gqa_prefill_top(bs:int, seq:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    assert parallel.ep==1

    # max_activation, mem_weight, mem_kv_cache = get_gqa_prefill_footprint(bs, seq, hidden, num_head, num_kv_head, head_dim, parallel, atten_parallel, atten_bytes)
    # log.info("max_activation: %s GiB, mem_weight: %s GiB, mem_kv_cache: %s GiB", max_activation/(1024**3), mem_weight/(1024**3), mem_kv_cache/(1024**3))

    assert parallel.ep==1

    if seq == 1:
        assert parallel.sp == 1

    
    if granularity.get_mode() == "coarse":
        return gqa_prefill_coarse(bs, seq, hidden, num_head, num_kv_head, head_dim, parallel, atten_parallel, next_parallel, atten_bytes, granularity, single_chip, noc_hierarchy)
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass
