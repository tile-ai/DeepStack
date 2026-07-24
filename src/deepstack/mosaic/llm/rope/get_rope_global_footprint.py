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
from .rope_coarse import rope_coarse


log = logging.getLogger(__name__) 

def get_rope_global_footprint(bs: int, head_dim: int, parallel: ParallelScheme, rope_bytes: OpBytes, MAX_ROPE_SEQ = 128*1024):
    '''
    To remember:
    Rope cos sin weights are not related to seq or layers!! 
    Only related to MAX_SEQ and head_dim (which is usually 128)!
    '''
    # rope(x, cos, sin)=x⊗cos+rot(x)⊗sin  
    # 其中，x [bs, head, seq, hidden/head], cos 和sin都是[1,1,seq,hidden/head], rot算子就是在最后一维重排，例如x[x1, x2], rot(x)= [-x2, x1]
    # 数学上计算定义为：RoPE 对一个向量 [ 𝑎 , 𝑏 ] （即前半部分和后半部分）做二维旋转：
    # [ 𝑎 ′ , 𝑏 ′ ] = [ 𝑎 ⋅ cos ⁡ 𝜃 − 𝑏 ⋅ sin ⁡ 𝜃 , 𝑎 ⋅ sin ⁡ 𝜃 + 𝑏 ⋅ cos ⁡ 𝜃 ]

    
    # 128k context length


    in_bytes, weight_bytes, out_bytes = rope_bytes.get_dtype_bytes()
    shard_bs = math.ceil(bs / parallel.dp)
    shard_max_seq = math.ceil(MAX_ROPE_SEQ / parallel.sp)

    # head_dim is not split
    
    mem_weight=weight_bytes * np.prod([1,1,MAX_ROPE_SEQ,head_dim]) * 2 # cos and sin


    return np.uint64(mem_weight)
