
from mosaic.utils import OpBytes, Modeling_Granularity,Tensor_Loc
from tilesight.arch import Arch
import math
# from tilesight.fused_op_dtype.matmul_fused_op_new_api import calculate_matmul_resource_utilization_new
from tilesight.fused_op_dtype_wave.N_0_element_wise_fused_op_wave import calculate_N_0_elementwise_resource_utilization
from tilesight.fused_op_dtype_wave.N_1_element_wise_fused_op_wave import calculate_N_1_elementwise_resource_utilization
from tilesight.fused_op_dtype_wave.N_N_element_wise_fused_op_wave import calculate_N_N_elementwise_resource_utilization

from tilesight.fusion_support.hete_post_process_single_op import hete_post_process_tensor_core_op,hete_post_process_sfu_core_op,hete_post_process_cuda_core_op

from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
# tiling_configs
# tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages
from functools import lru_cache
import torch
import numpy as np
import logging
log = logging.getLogger(__name__)

from mosaic.parallelism import ParallelScheme
from numpy import uint64
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, make_mesh_or_torus, make_switch, TopoKind, PortSpread, get_extend_max_routes, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity

def all_gather_sp_fission(parallel:ParallelScheme, noc_hierarchy:Hierarchy, bytes_needed_per_device:int, next_virtual_sp:int):

    all_gather_bytes = bytes_needed_per_device
    log.info("all_gather_bytes: %s", all_gather_bytes)
    if parallel.sp == 1:
        all_gather_hop_latency, all_gather_ext_max = 0, 0
        all_gather_traffic = None
        return all_gather_hop_latency, all_gather_ext_max, all_gather_traffic
    all_gather_bytes_per_pair = all_gather_bytes / (parallel.sp -1)
    log.info("all_gather_bytes_per_pair: %s", all_gather_bytes_per_pair)

    new_virtual_parallel = ParallelScheme(tp=parallel.tp, ep=parallel.ep, sp=int(parallel.sp/next_virtual_sp), cp=1, dp=1, pp=int(parallel.world_size()/parallel.tp/parallel.ep/(parallel.sp/next_virtual_sp)))
    
    new_virtual_all_gather_byte_per_pair=all_gather_bytes/new_virtual_parallel.sp
    log.info("new_virtual_all_gather_byte_per_pair: %s", new_virtual_all_gather_byte_per_pair)
    # reconstruct the weight matrix
    # tm = TrafficMatrix(parallel.world_size())
    # tm.add_intra_group_traffic("sp", all_gather_bytes_per_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
    tm = TrafficMatrix(new_virtual_parallel.world_size())
    tm.add_intra_group_traffic("sp", new_virtual_all_gather_byte_per_pair, tp=new_virtual_parallel.tp, ep=new_virtual_parallel.ep, sp=new_virtual_parallel.sp, cp=new_virtual_parallel.cp, dp=new_virtual_parallel.dp, pp=new_virtual_parallel.pp)
    # tm.save_heatmap(path='./sp_all_gather.png')
    # tm.add_intra_group_traffic("dp", 1000/3, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
    # all_gather_hop_latency, all_gather_ext_max, all_gather_overall_time= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
    all_gather_hop_latency, all_gather_ext_max, all_gather_overall_time, all_gather_traffic= get_extend_max_routes_with_traffic(tm, noc_hierarchy)

    log.info("all_gather_sp_fission, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", all_gather_hop_latency, all_gather_ext_max, all_gather_overall_time)

    return all_gather_hop_latency, all_gather_ext_max, all_gather_traffic

def all_gather_wrapper(all_gather_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    # TODO: implement this
    # specially for flexible connector between different modules
    pass
    