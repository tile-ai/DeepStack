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
from .rms_norm_coarse import rms_norm_coarse


log = logging.getLogger(__name__) 
from mosaic.cost.energy_record import EnergyRecord

def rms_norm_top(bs:int, seq:int, hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, rms_norm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    
    # x ⨂ (1/√(mean(x^2, last_dim)+ε)) ⨂ g
    # g is weight, while too small, [1,1,hidden] no split across tensor here
    # e.g. , llama2-70b, size of (g) = 2(rms norm blocks per layer)* 80 (layres) * 8192 (hidden) * 2 (float16) = 2621440 = 2.5MiB
    
    # Step 1 mean(x^2, last_dim), [bs/dp, seq/sp, hidden/tp]
    #   If with tp, then an additional all-reduce is needed
    #   Step 1.1[bs/dp, seq/sp, hidden/tp] -> [bs/dp, seq/sp, hidden/tp]
    #   
    # Step 2 mean+ epsilon -> [bs/dp, seq/sp, hidden/tp]   
    # sqrt,
    # 1/rep
    # 1 cuda op, 2 sfu op
    # 
    # Step 3 x ⨂ (1/√(mean(x^2, last_dim)+ε)) ⨂ g
    # element wise mul x 2, first mul rep, then mul g
    # [bs/dp, seq/sp, hidden/tp] -> [bs/dp, seq/sp, hidden/tp] -> [bs/dp, seq/sp, hidden/tp]
    #   Again, if with tp, then an additional all-gather is needed

    # finally, no matter for attention or mlp, we all need whole hidden/tp elements to do the post-attention/mlp



    # assert parallel.cp==1
    # assert parallel.ep==1

    if seq == 1:
        assert parallel.sp == 1

    max_activation, mem_weight = get_rms_norm_footprint(bs, seq, hidden, parallel, rms_norm_bytes)
    log.info("max_activation: %s GiB, mem_weight: %s GiB", max_activation/(1024**3), mem_weight/(1024**3))
    
    if granularity.get_mode() == "coarse":
        total_time, stats = rms_norm_coarse(bs, seq, hidden, parallel, next_parallel, rms_norm_bytes, granularity, single_chip, noc_hierarchy)
        return total_time, stats
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass
    
    
    pass

def get_rms_norm_footprint(bs: int, seq: int, hidden: int, parallel: ParallelScheme, rms_norm_bytes: OpBytes):
    # x ⨂ (1/√(mean(x^2, last_dim)+ε)) ⨂ g
    # g is weight, while too small, [1,1,hidden] no split across tensor here
    # e.g. , llama2-70b, size of (g) = 2(rms norm blocks per layer)* 80 (layres) * 8192 (hidden) * 2 (float16) = 2621440 = 2.5MiB
    
    in_bytes, weight_bytes, out_bytes = rms_norm_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    mem_weight=weight_bytes * shard_hidden

    max_activation=in_bytes * shard_bs * shard_seq * shard_hidden * 2

    return np.uint64(max_activation), np.uint64(mem_weight)
