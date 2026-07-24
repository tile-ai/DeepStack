import torch
import math
import numpy as np
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, make_mesh_or_torus, make_switch, TopoKind, PortSpread, get_extend_max_routes, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
from mosaic.op_dtype.element_wrapper import element_wrapper
from mosaic.op_dtype.reduce_wrapper import reduce_wrapper
import cProfile

from tilesight.arch import *
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
import logging
log = logging.getLogger(__name__) 
from mosaic.cost.energy_record import EnergyRecord
from mosaic.collectives import all_reduce_wrapper
from mosaic.utils import get_comp_comm_e2e_time
from mosaic.cost.op_perf_stats import OpPerfStats

def rope_coarse(bs:int, head:int, seq:int, head_dim:int, parallel:ParallelScheme, next_parallel:ParallelScheme, rope_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    # rope(x, cos, sin)=x⊗cos+rot(x)⊗sin  
    # 其中，x [bs, head, seq, hidden/head], cos 和sin都是[1,1,seq,hidden/head], rot算子就是在最后一维重排，例如x[x1, x2], rot(x)= [-x2, x1]
    # 数学上计算定义为：RoPE 对一个向量 [ 𝑎 , 𝑏 ] （即前半部分和后半部分）做二维旋转：
    # [ 𝑎 ′ , 𝑏 ′ ] = [ 𝑎 ⋅ cos ⁡ 𝜃 − 𝑏 ⋅ sin ⁡ 𝜃 , 𝑎 ⋅ sin ⁡ 𝜃 + 𝑏 ⋅ cos ⁡ 𝜃 ]

    in_bytes, weight_bytes, out_bytes = rope_bytes.get_dtype_bytes()
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_head = math.ceil(head / parallel.tp)

    smem_fusion_list=[]

    cos_bytes=OpBytes(
        input1=Tensor_Loc(rope_bytes.input1.dtype, rope_bytes.input1.loc,[shard_bs, shard_head, shard_seq, head_dim]),
        input2=Tensor_Loc(rope_bytes.input2.dtype, 'ddr',[1, 1, shard_seq, head_dim]),
        output=Tensor_Loc(rope_bytes.input2.dtype, 'reg',[shard_bs, shard_head, shard_seq, head_dim]),
    )

    hete_post_data_cos, tb_cos=element_wrapper(element_op_bytes=cos_bytes, granularity=granularity, single_chip=single_chip, batch=1, type="cuda_core", tb_tiling_config=None)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_cos], single_chip))


    sin_bytes=OpBytes(
        input1=Tensor_Loc(rope_bytes.input1.dtype, 'reg',[shard_bs, shard_head, shard_seq, head_dim]),
        input2=Tensor_Loc(rope_bytes.input2.dtype, 'ddr',[1, 1, shard_seq, head_dim]),
        output=Tensor_Loc(rope_bytes.input2.dtype, 'ddr',[shard_bs, shard_head, shard_seq, head_dim]),
    )

    hete_post_data_sin, _=element_wrapper(element_op_bytes=sin_bytes, granularity=granularity, single_chip=single_chip, batch=1, type="cuda_core", tb_tiling_config=tb_cos)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_sin], single_chip))

    grids = [shard_bs/tb_cos[0], shard_head/tb_cos[1], shard_seq/tb_cos[2], head_dim/tb_cos[3]]

    waves = np.prod(grids) / single_chip.sm_count

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)

    single_chip_time=smem_fusion_post_data[0]

    if parallel.sp == 1 and next_parallel.sp == 1:
        log.info("rope_coarse: parallel.sp == 1, next_parallel.sp == 1, no need to do all-gather")
        e2e_time = single_chip_time
    elif parallel == next_parallel:
        log.info("rope_coarse: parallel == next_parallel, no need to do all-gather")
        e2e_time = single_chip_time
    elif parallel.sp > 1 and next_parallel.sp == 1 and (next_parallel.tp == parallel.sp*parallel.tp):
        log.info("rope_coarse: parallel.sp > 1, next_parallel.sp == 1, need to do all-gather")

        # all_to_all_bytes = ?
        # we need to carefully think about this question, especially for cases like there are not enough heads to get split across
        
        # befor all_to_all, each unit hold math.ceil(head/tp)   math.ceil(seq/sp) data
        # after all_to_all, each unit hold math.ceil(head/tp/sp)  seq  data
        
        # log.info ("shard_bs: %s, head_dim: %s, math.ceil(head/parallel.sp/parallel.tp): %s, (seq - shard_seq): %s, out_bytes: %s", shard_bs, head_dim, math.ceil(head/parallel.sp/parallel.tp), (seq - shard_seq), out_bytes)
        all_to_all_bytes = shard_bs * head_dim * math.ceil(head/parallel.sp/parallel.tp) * (seq - shard_seq) * out_bytes

        all_to_all_bytes_per_pair = all_to_all_bytes / (parallel.sp -1)
        
        # reconstruct the weight matrix
        tm = TrafficMatrix(parallel.world_size())
        tm.add_intra_group_traffic("sp", all_to_all_bytes_per_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        # tm.save_heatmap(path='./sp_all_to_all.png')
        # tm.add_intra_group_traffic("dp", 1000/3, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        all_to_all_hop_latency, all_to_all_ext_max, all_to_all_overall_time, all_to_all_traffic= get_extend_max_routes_with_traffic(tm, noc_hierarchy)

        e2e_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=all_to_all_hop_latency, network_link_time=all_to_all_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
        # 
    else:
        raise NotImplementedError("rope_coarse: parallel.sp > 1 and next_parallel.sp == 1 and (next_parallel.tp == parallel.sp*parallel.tp), not implemented", parallel, next_parallel)

    return e2e_time
