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
from .mla_decode_coarse import mla_decode_coarse, mla_decode_kv_list_coarse
from mosaic.llm_arch import MLA_Arch, GQA_Arch, MoE_Arch, Dense_FFN_Arch, LLM_Arch

log = logging.getLogger(__name__) 
from mosaic.cost.energy_record import EnergyRecord

def mla_decode_top(bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    assert parallel.ep==1

    if seq == 1:
        assert parallel.sp == 1

    assert model_arch.mla_arch is not None

    
    if granularity.get_mode() == "coarse":
        
        return mla_decode_coarse(bs, seq, cached_kv, model_arch, parallel, atten_parallel, next_parallel, granularity, single_chip, noc_hierarchy)

    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass


def mla_decode_kv_list_top(bs:int, seq:int, cached_kv_list:list[int], model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    assert parallel.ep==1

    if seq == 1:
        assert parallel.sp == 1

    assert model_arch.mla_arch is not None

    
    if granularity.get_mode() == "coarse":
        
        return mla_decode_kv_list_coarse(bs, seq, cached_kv_list, model_arch, parallel, atten_parallel, next_parallel, granularity, single_chip, noc_hierarchy)

    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass
