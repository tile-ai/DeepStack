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
from .add_residual_coarse import add_residual_coarse
from mosaic.cost.energy_record import EnergyRecord
from mosaic.cost.op_perf_stats import OpPerfStats


log = logging.getLogger(__name__) 

def get_add_residual_footprint(bs: int, seq: int, hidden: int, parallel: ParallelScheme, add_residual_bytes: OpBytes):
    # y=x + x_residual
    # x from smem possibly, x_residual from ddr, y to ddr 
    in_bytes, weight_bytes, out_bytes = add_residual_bytes.get_dtype_bytes()

    mem_weight= 0 

    max_activation=in_bytes * bs/parallel.dp * seq/parallel.sp * hidden * 2

    return np.uint64(max_activation), np.uint64(mem_weight)


def add_residual_top(bs:int, seq:int, hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, add_residual_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy) -> tuple[float, OpPerfStats]:
    # y=x + x_residual
    # x from smem possibly, x_residual from ddr, y to smem possibly
    # but usually x needs collectives to reconstruct the tensor, so from ddr makes more sense
    # assert parallel.cp==1
    # assert parallel.ep==1

    if seq == 1:
        assert parallel.sp == 1

    # max_activation, mem_weight = get_add_residual_footprint(bs, seq, hidden, parallel, add_residual_bytes)
    # log.info("max_activation: %s GiB, mem_weight: %s GiB", max_activation/(1024**3), mem_weight/(1024**3))
    
    if granularity.get_mode() == "coarse":
        # return max_activation, mem_weight, add_residual_coarse(bs, seq, hidden, parallel, next_parallel, add_residual_bytes, granularity, single_chip, noc_hierarchy)
        return add_residual_coarse(bs, seq, hidden, parallel, next_parallel, add_residual_bytes, granularity, single_chip, noc_hierarchy)
    
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass
    
    
    pass
