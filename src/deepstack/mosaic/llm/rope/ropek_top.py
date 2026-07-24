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
from .ropek_coarse import ropek_coarse


log = logging.getLogger(__name__) 
from mosaic.cost.energy_record import EnergyRecord

def ropek_top(bs:int, head:int, seq:int, head_dim:int, parallel:ParallelScheme, next_parallel:ParallelScheme, rope_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    # rope(x, cos, sin)=x⊗cos+rot(x)⊗sin  
    # 其中，x [bs, head, seq, hidden/head], cos 和sin都是[1,1,seq,hidden/head], rot算子就是在最后一维重排，例如x[x1, x2], rot(x)= [-x2, x1]
    # 数学上计算定义为：RoPE 对一个向量 [ 𝑎 , 𝑏 ] （即前半部分和后半部分）做二维旋转：
    # [ 𝑎 ′ , 𝑏 ′ ] = [ 𝑎 ⋅ cos ⁡ 𝜃 − 𝑏 ⋅ sin ⁡ 𝜃 , 𝑎 ⋅ sin ⁡ 𝜃 + 𝑏 ⋅ cos ⁡ 𝜃 ]

    # assert parallel.cp==1
    # assert parallel.ep==1

    if seq == 1:
        assert parallel.sp == 1

    max_activation, mem_weight = get_rope_footprint(bs, head, seq, head_dim, parallel, rope_bytes)
    log.info("max_activation: %s GiB, mem_weight: %s GiB", max_activation/(1024**3), mem_weight/(1024**3))
    
    if granularity.get_mode() == "coarse":
        return max_activation, mem_weight, ropek_coarse(bs, head, seq, head_dim, parallel, next_parallel, rope_bytes, granularity, single_chip, noc_hierarchy)
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass
        
    pass

def get_rope_footprint(bs: int, head: int, seq: int, head_dim: int, parallel: ParallelScheme, rope_bytes: OpBytes):
    '''
    To remember:
    Rope cos sin weights are not related to seq or layers!! 
    Only related to MAX_SEQ and head_dim (which is usually 128)!
    '''
    # rope(x, cos, sin)=x⊗cos+rot(x)⊗sin  
    # 其中，x [bs, head, seq, hidden/head], cos 和sin都是[1,1,seq,hidden/head], rot算子就是在最后一维重排，例如x[x1, x2], rot(x)= [-x2, x1]
    # 数学上计算定义为：RoPE 对一个向量 [ 𝑎 , 𝑏 ] （即前半部分和后半部分）做二维旋转：
    # [ 𝑎 ′ , 𝑏 ′ ] = [ 𝑎 ⋅ cos ⁡ 𝜃 − 𝑏 ⋅ sin ⁡ 𝜃 , 𝑎 ⋅ sin ⁡ 𝜃 + 𝑏 ⋅ cos ⁡ 𝜃 ]

    MAX_ROPE_SEQ = 128*1024
    # 128k context length


    in_bytes, weight_bytes, out_bytes = rope_bytes.get_dtype_bytes()
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_head = math.ceil(head / parallel.tp)

    # head_dim is not split
    
    mem_weight=weight_bytes * np.prod([1,1,MAX_ROPE_SEQ,head_dim]) * 2 # cos and sin

    max_activation=in_bytes * shard_bs * shard_seq * shard_head * head_dim *2

    return np.uint64(max_activation), np.uint64(mem_weight)
