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
from .mla_prefill_coarse import mla_prefill_coarse
from mosaic.llm_arch import MLA_Arch, GQA_Arch, MoE_Arch, Dense_FFN_Arch, LLM_Arch


log = logging.getLogger(__name__) 
from mosaic.cost.energy_record import EnergyRecord


def mla_prefill_top(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    assert parallel.ep==1


    if seq == 1:
        assert parallel.sp == 1

    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    num_kv_head = mla_arch.num_kv_head
    head_dim = mla_arch.head_dim
    q_down_hidden = mla_arch.q_down_hidden
    q_nope_head_dim = mla_arch.q_nope_head_dim
    q_rope_head_dim = mla_arch.q_rope_head_dim
    kv_rope_head_dim = mla_arch.kv_rope_head_dim
    kv_nope_head_dim = mla_arch.kv_nope_head_dim
    kv_nope_up_hidden = mla_arch.kv_nope_up_hidden
    atten_bytes = mla_arch.atten_bytes

    group_size = math.ceil (num_head/num_kv_head)
    wq_hidden = num_head * head_dim
    
    if granularity.get_mode() == "coarse":
        
        return mla_prefill_coarse(bs, seq, model_arch, parallel, atten_parallel, next_parallel, granularity, single_chip, noc_hierarchy)
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass
