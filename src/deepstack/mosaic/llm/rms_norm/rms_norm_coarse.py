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

def rms_norm_coarse_stage1(bs:int, seq:int, hidden:int, parallel:ParallelScheme, rms_norm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    # Step 1 mean(x^2, last_dim), [bs/dp, seq/sp, hidden/tp]
    #   If with tp, then an additional all-reduce is needed
    #   Step 1.1[bs/dp, seq/sp, hidden/tp] -> [bs/dp, seq/sp, hidden/tp]
    # 
        
    in_bytes, weight_bytes, out_bytes = rms_norm_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    if parallel.tp == 1:
        smem_fusion_list=[]

        square_op_bytes=OpBytes(
        input1=Tensor_Loc(rms_norm_bytes.input1.dtype, rms_norm_bytes.input1.loc,[shard_bs, shard_seq, shard_hidden]),
        # input2=Tensor_Loc(rms_norm_bytes.input2.dtype, rms_norm_bytes.input2.loc,[shard_bs, shard_seq, shard_hidden]),
        input2=None,
        output=Tensor_Loc(rms_norm_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, shard_hidden]),
        )
        # batch =2, 1 is square, 1 is convert to fp32
        hete_post_data_square, tb_residual_config=element_wrapper(element_op_bytes=square_op_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=None)
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_square], single_chip))

        mean_op_bytes=OpBytes(
        input1=Tensor_Loc(rms_norm_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, shard_hidden]),
        # input2=Tensor_Loc(rms_norm_bytes.input2.dtype, rms_norm_bytes.input2.loc,[shard_bs, shard_seq, shard_hidden]),
        input2=None,
        output=Tensor_Loc(rms_norm_bytes.input1.dtype, 'ddr',[shard_bs, shard_seq, 1]),
        )

        
        hete_post_data_mean, out_tb_shape = reduce_wrapper(mean_op_bytes, granularity, single_chip, tb_tiling_config=None)
        # hete_post_data_mean, out_tb_shape = reduce_wrapper(mean_op_bytes, granularity, single_chip, tb_tiling_config=[2,2,1])
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_mean], single_chip))
        grids = [shard_bs/out_tb_shape[0], shard_seq/out_tb_shape[1]]

        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
    
        single_chip_time=smem_fusion_post_data[0]
        log.info("rms_norm_coarse_stage1 single chip time: %s s", single_chip_time)

        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        return single_chip_time


    elif parallel.tp>1:

        smem_fusion_list=[]

        square_op_bytes=OpBytes(
        input1=Tensor_Loc(rms_norm_bytes.input1.dtype, rms_norm_bytes.input1.loc,[shard_bs, shard_seq, shard_hidden]),
        # input2=Tensor_Loc(rms_norm_bytes.input2.dtype, rms_norm_bytes.input2.loc,[shard_bs, shard_seq, shard_hidden]),
        input2=None,
        output=Tensor_Loc(rms_norm_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, shard_hidden]),
        )
        # batch =2, 1 is square, 1 is convert to fp32
        hete_post_data_square, tb_residual_config=element_wrapper(element_op_bytes=square_op_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=None)
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_square], single_chip))

        mean_op_bytes=OpBytes(
        input1=Tensor_Loc(rms_norm_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, shard_hidden]),
        # input2=Tensor_Loc(rms_norm_bytes.input2.dtype, rms_norm_bytes.input2.loc,[shard_bs, shard_seq, shard_hidden]),
        input2=None,
        output=Tensor_Loc(rms_norm_bytes.input1.dtype, 'ddr',[shard_bs, shard_seq, 1]),
        )

        
        hete_post_data_mean, out_tb_shape = reduce_wrapper(mean_op_bytes, granularity, single_chip, tb_tiling_config=None)
        # hete_post_data_mean, out_tb_shape = reduce_wrapper(mean_op_bytes, granularity, single_chip, tb_tiling_config=[2,2,1])
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_mean], single_chip))
        grids = [shard_bs/out_tb_shape[0], shard_seq/out_tb_shape[1]]

        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
    
        single_chip_time=smem_fusion_post_data[0]
        log.info("rms_norm_coarse_stage1 single chip time: %s s", single_chip_time)

        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = all_reduce_wrapper(
            all_reduce_op_bytes=mean_op_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy,
            granularity=granularity, dim_to_process="tp", bytes=shard_bs*shard_seq*1*out_bytes)

        if stats is not None:
            stats.append_traffic(all_reduce_traffic, hop_time_s=all_reduce_hop, link_time_s=all_reduce_ext_max)

        waves = np.prod(grids) / single_chip.sm_count
        e2e_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=all_reduce_hop, network_link_time=all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())

        log.info("rms_norm_coarse_stage1 e2e time: %s s", e2e_time)
        return e2e_time
        
def rms_norm_coarse_stage2(bs:int, seq:int, hidden:int, parallel:ParallelScheme, rms_norm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    # # ------------------------------------- stage 2 -------------------------------------
    # # fused:
    # +epsislon (1e-5), sqrt, 1/rep, [bs/dp, seq/sp, 1] - > [bs/dp, seq/sp, 1] - > [bs/dp, seq/sp, 1] -> [bs/dp, seq/sp, 1] , 一个cuda core，两个sfu
    # Step 2 mean+ epsilon -> [bs/dp, seq/sp, hidden/tp]   
    # sqrt,
    # 1/rep
    # 1 cuda op, 2 sfu op

        
    in_bytes, weight_bytes, out_bytes = rms_norm_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    smem_fusion_list=[]
    epsilon_op_bytes=OpBytes(
    input1=Tensor_Loc(rms_norm_bytes.input1.dtype, 'ddr',[shard_bs, shard_seq, 1]),
    input2=Tensor_Loc(rms_norm_bytes.input2.dtype, 'ddr',[1, 1, 1]),
    output=Tensor_Loc(rms_norm_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, 1]),
    )

    hete_post_data_epsilon, tb_epsilon=element_wrapper(element_op_bytes=epsilon_op_bytes, granularity=granularity, single_chip=single_chip, batch=1, type="cuda_core", tb_tiling_config=None)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_epsilon], single_chip))


    sqrt_rep_op_bytes=OpBytes(
    input1=Tensor_Loc(rms_norm_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, 1]),
    input2=None,
    output=Tensor_Loc(rms_norm_bytes.input1.dtype, 'ddr',[shard_bs, shard_seq, 1]),
    )

    hete_post_data_sqrt_rep, _=element_wrapper(element_op_bytes=sqrt_rep_op_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="sfu_core", tb_tiling_config=tb_epsilon)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_sqrt_rep], single_chip))

    grids = [shard_bs/tb_epsilon[0], shard_seq/tb_epsilon[1],1]
    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)

    single_chip_time=smem_fusion_post_data[0]
    log.info("rms_norm_coarse_stage2 single chip time: %s s", single_chip_time)

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    return single_chip_time

def rms_norm_coarse_stage3(bs:int, seq:int, hidden:int, parallel:ParallelScheme, rms_norm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    # Step 3 x ⨂ (1/√(mean(x^2, last_dim)+ε)) ⨂ g
    # element wise mul x 2, first mul rep, then mul g
    # [bs/dp, seq/sp, hidden/tp] -> [bs/dp, seq/sp, hidden/tp] -> [bs/dp, seq/sp, hidden/tp]
    #   Again, if with tp, then an additional all-gather is needed

    in_bytes, weight_bytes, out_bytes = rms_norm_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
 

    smem_fusion_list=[]
    element_op_bytes=OpBytes(
    input1=Tensor_Loc(rms_norm_bytes.input1.dtype, 'ddr',[shard_bs, shard_seq, shard_hidden]),
    input2=Tensor_Loc(rms_norm_bytes.input2.dtype, 'ddr',[shard_bs, shard_seq, 1]),
    output=Tensor_Loc(rms_norm_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, shard_hidden]),
    )

    hete_post_data_mul, tb_mul=element_wrapper(element_op_bytes=element_op_bytes, granularity=granularity, single_chip=single_chip, batch=1, type="cuda_core", tb_tiling_config=None)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_mul], single_chip))


    mul_g_op_bytes=OpBytes(
    input1=Tensor_Loc(rms_norm_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, shard_hidden]),
    input2=Tensor_Loc(rms_norm_bytes.input2.dtype, 'ddr',[1, 1, shard_hidden]),
    output=Tensor_Loc(rms_norm_bytes.output.dtype, 'ddr',[shard_bs, shard_seq, shard_hidden]),
    )
    # batch =2, 1 is mul, 1 is convert to fp16
    hete_post_data_mul_g, _=element_wrapper(element_op_bytes=mul_g_op_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=tb_mul)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_mul_g], single_chip))

    grids = [shard_bs/tb_mul[0], shard_seq/tb_mul[1],shard_hidden/tb_mul[2]]

    waves = np.prod(grids) / single_chip.sm_count

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)

    single_chip_time=smem_fusion_post_data[0]
    log.info("rms_norm_coarse_stage3 single chip time: %s s", single_chip_time)

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    if parallel.tp>1:
        all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = all_reduce_wrapper(
            all_reduce_op_bytes=rms_norm_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy,
            granularity=granularity, dim_to_process="tp", bytes=shard_bs*shard_seq*shard_hidden*out_bytes)

        if stats is not None:
            stats.append_traffic(all_reduce_traffic, hop_time_s=all_reduce_hop, link_time_s=all_reduce_ext_max)
        waves = np.prod(grids) / single_chip.sm_count
        e2e_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=all_reduce_hop, network_link_time=all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())

    else:
        e2e_time = single_chip_time

    log.info("rms_norm_coarse_stage3 e2e time: %s s", e2e_time)
    return e2e_time


def rms_norm_coarse(bs:int, seq:int, hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, rms_norm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    assert granularity.get_mode() == "coarse"
    # print(parallel.world_size())
    
    stats = OpPerfStats(op_name="rms_norm") if granularity.dump_perf_log else None
    
    in_bytes, weight_bytes, out_bytes = rms_norm_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

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


    # ------------------------------------- stage 1 -------------------------------------

    # let's model it one-by-one
    # Step 1 mean(x^2, last_dim), [bs/dp, seq/sp, hidden/tp]
    #   If with tp, then an additional all-reduce is needed
    #   Step 1.1[bs/dp, seq/sp, hidden/tp] -> [bs/dp, seq/sp, hidden/tp]
    time_stage1 = rms_norm_coarse_stage1(bs=bs, seq=seq, hidden=hidden,  parallel=parallel, rms_norm_bytes=rms_norm_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # # ------------------------------------- stage 2 -------------------------------------
    # # fused:
    # +epsislon (1e-5), sqrt, 1/rep, [bs/dp, seq/sp, 1] - > [bs/dp, seq/sp, 1] - > [bs/dp, seq/sp, 1] -> [bs/dp, seq/sp, 1] , 一个cuda core，两个sfu
    # Step 2 mean+ epsilon -> [bs/dp, seq/sp, hidden/tp]   
    # sqrt,
    # 1/rep
    # 1 cuda op, 2 sfu op

    time_stage2 = rms_norm_coarse_stage2(bs=bs, seq=seq, hidden=hidden,  parallel=parallel, rms_norm_bytes=rms_norm_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    # # ------------------------------------- stage 3 -------------------------------------
    # Step 3 x ⨂ (1/√(mean(x^2, last_dim)+ε)) ⨂ g
    # element wise mul x 2, first mul rep, then mul g
    # [bs/dp, seq/sp, hidden/tp] -> [bs/dp, seq/sp, hidden/tp] -> [bs/dp, seq/sp, hidden/tp]
    #   Again, if with tp, then an additional all-gather is needed
    time_stage3 = rms_norm_coarse_stage3(bs=bs, seq=seq, hidden=hidden,  parallel=parallel, rms_norm_bytes=rms_norm_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)


    # # ------------------------------------- stage 4 -------------------------------------

    # time_stage4 =rms_norm_coarse_stage4(bs=bs, seq=seq, hidden=hidden,  parallel=parallel, rms_norm_bytes=rms_norm_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    total_time = time_stage1 + time_stage2 + time_stage3
    log.info("Whole rms_norm coarse e2e time: %s s", total_time)
    if stats is not None:
        stats.finalize(total_time, h=noc_hierarchy, arch=single_chip)
    return total_time, stats