import torch
import math
import os
import numpy as np
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, make_mesh_or_torus, make_switch, TopoKind, PortSpread, get_extend_max_routes, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
from mosaic.op_dtype.group_gemm_wrapper import group_gemm_wrapper
from mosaic.op_dtype.element_wrapper import element_wrapper
import cProfile

from tilesight.arch import *
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
import logging
log = logging.getLogger(__name__) 
from mosaic.cost.energy_record import EnergyRecord
from mosaic.collectives import all_reduce_wrapper
from mosaic.collectives import reduce_scatter_wrapper
from mosaic.utils import get_comp_comm_e2e_time
from mosaic.cost.op_perf_stats import OpPerfStats


def swiglu_coarse_group_gemm_stage1(expert_row:np.ndarray, counts:np.ndarray, local_num_activated_experts:int, group_gemm_bs_m_shape_list:list, hidden:int, up_hidden:int, parallel:ParallelScheme, swiglu_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    assert granularity.get_mode() == "coarse"
    # print(parallel.world_size())
    
    in_bytes, weight_bytes, out_bytes = swiglu_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_up_hidden = math.ceil(up_hidden / parallel.tp)

    # x: [bs/dp, seq/sp, hidden]
    # W1,W2 [hidden, up_hidden/tp]
    # W3 [up_hidden/tp, hidden]

    # x@W1 -> [bs/dp, seq/sp, up_hidden/tp]
    # x@W2 -> [bs/dp, seq/sp, up_hidden/tp]
    # silu-> [bs/dp, seq/sp, up_hidden/tp]
    # (x @ W1) ⊗ silu(x @ W2) -> [bs/dp, seq/sp, up_hidden/tp]
    # @ W3 -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce -> [bs/dp, seq/sp, hidden]



    # ------------------------------------- stage 1 -------------------------------------

    # let's model it one-by-one
    # 1. x@W1 -> [bs/dp, seq/sp, hidden] @ [hidden, up_hidden/tp] -> [bs/dp, seq/sp, up_hidden/tp]
    # x: [shard_bs, shard_seq, shard_hidden]
    # W1: [shard_hidden, shard_up_hidden]
    # x@W1: [shard_bs, shard_seq, shard_up_hidden]

    

    
    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_1, ext_max_1, overall_time_1= 0, 0, 0
    elif (parallel.fsdp == True):
        # reconstruct the weight matrix
        tm = TrafficMatrix(parallel.world_size())
        tm.add_intra_group_traffic("dp", hidden*shard_up_hidden*weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1) * local_num_activated_experts, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        # tm.add_intra_group_traffic("dp", 1000/3, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_1, ext_max_1, overall_time_1, noc_traffic_1= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=hop_time_1, link_time_s=ext_max_1)
        log.info("swiglu stage 1 fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_1, ext_max_1, overall_time_1)
        # shard_in_bytes = shard_hidden*shard_up_hidden*weight_bytes * (parallel.dp-1)/parallel.dp
        # shard_out_bytes = shard_in_bytes

    # tm.save_heatmap("tm_swiglu.png")
    # single device memory footprint

    gemm1_bytes=OpBytes(
        input1=Tensor_Loc(swiglu_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(swiglu_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(swiglu_bytes.output.dtype, 'ddr'),
    )
    # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
    # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only

    # extract shape list from expert_row and counts

    # generate local group gemm shape list
    local_group_gemm_shape_list=[]
    for bs, m in group_gemm_bs_m_shape_list:
        local_group_gemm_shape_list.append((bs, m, shard_up_hidden, hidden))

    # hete_post_data, smem_fusion_post_data, tiling_config=gemm_wrapper(M=shard_tokens, N=shard_up_hidden, K=hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
    smem_fusion_list, smem_fusion_post_data, grids =group_gemm_wrapper(shape_list=local_group_gemm_shape_list, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip, stats=stats)
    smem_fusion_post_data[0] += 1e-06 * local_num_activated_experts
    
    # log.debug("smem_fusion_post_data: %s, tiling_config: %s", smem_fusion_post_data, tiling_config)
    log.info("swiglu stage 1 single device time: %s s", smem_fusion_post_data[0])
    if (granularity.get_comp_comm_overlap()==True):
        time_stage1=max(smem_fusion_post_data[0],overall_time_1)
    elif (granularity.get_comp_comm_overlap()==False):
        time_stage1=smem_fusion_post_data[0] + overall_time_1

    # grids = [shard_tokens / tiling_config[0], shard_up_hidden / tiling_config[1]]

    waves = np.prod(grids) / single_chip.sm_count

    # log.info("grids: %s, waves: %s", grids, waves)
    # log.info("time_stage1: %s s", time_stage1)
    return time_stage1, waves

def swiglu_coarse_group_gemm_stage2(expert_row:np.ndarray, counts:np.ndarray, local_num_activated_experts:int, group_gemm_bs_m_shape_list:list, hidden:int, up_hidden:int, parallel:ParallelScheme, swiglu_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    assert granularity.get_mode() == "coarse"
    # print(parallel.world_size())
    
    in_bytes, weight_bytes, out_bytes = swiglu_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_up_hidden = math.ceil(up_hidden / parallel.tp)

    # x: [bs/dp, seq/sp, hidden]
    # W1,W2 [hidden, up_hidden/tp]
    # W3 [up_hidden/tp, hidden]

    # x@W1 -> [bs/dp, seq/sp, up_hidden/tp]
    # x@W2 -> [bs/dp, seq/sp, up_hidden/tp]
    # silu-> [bs/dp, seq/sp, up_hidden/tp]
    # (x @ W1) ⊗ silu(x @ W2) -> [bs/dp, seq/sp, up_hidden/tp]
    # @ W3 -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce -> [bs/dp, seq/sp, hidden]



    # ------------------------------------- stage 2 -------------------------------------
    # fused:
    # x@W2 -> [bs/dp, seq/sp, up_hidden/tp] smem
    # silu (x@W2) -> [bs/dp, seq/sp, up_hidden/tp] smem
    # ⊗ element-wise multiplication, (x @ W1) ⊗ silu(x @ W2) -> [bs/dp, seq/sp, up_hidden/tp], element-wise multiplication
    

    # ------------------------------------- stage 2.1 x@W2 -> [bs/dp, seq/sp, up_hidden/tp] smem -------------------------------------
    if (parallel.fsdp == False or parallel.dp == 1):
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1= 0, 0, 0
    elif (parallel.fsdp == True):
        # reconstruct the weight matrix
        tm = TrafficMatrix(parallel.world_size())
        tm.add_intra_group_traffic("dp", hidden*shard_up_hidden*weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1) * local_num_activated_experts, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        # tm.add_intra_group_traffic("dp", 1000/3, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=noc_hop_time_1, link_time_s=noc_ext_max_1)
        log.info("swiglu stage 2 fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", noc_hop_time_1, noc_ext_max_1, noc_overall_time_1)
        # shard_in_bytes = shard_hidden*shard_up_hidden*weight_bytes * (parallel.dp-1)/parallel.dp
        # shard_out_bytes = shard_in_bytes

    # tm.save_heatmap("tm_swiglu.png")
    # single device memory footprint
    
    # smem_fusion_list=[]

    gemm2_bytes=OpBytes(
        input1=Tensor_Loc(swiglu_bytes.output.dtype, 'ddr'),
        input2=Tensor_Loc(swiglu_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(swiglu_bytes.output.dtype, 'smem'),
    )

    local_group_gemm_shape_list=[]
    tokens_sum = 0
    for bs, m in group_gemm_bs_m_shape_list:
        local_group_gemm_shape_list.append((bs, m, shard_up_hidden, hidden))
        tokens_sum += bs * m

    
    # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
    # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
    # hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_tokens, N=shard_up_hidden, K=hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
    # grids = [shard_tokens / gemm_tiling_config[0], shard_up_hidden / gemm_tiling_config[1]]
    # smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))
    smem_fusion_list, gemm_smem_fusion_post_data, grids =group_gemm_wrapper(shape_list=local_group_gemm_shape_list, gemm_bytes=gemm2_bytes, granularity=granularity, single_chip=single_chip)
    
    
    # log.debug("smem_fusion_post_data: %s, tiling_config: %s", smem_fusion_post_data, tiling_config)
    # log.info("swiglu stage 1 single device time: %s s", smem_fusion_post_data[0])
    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time = noc_overall_time_1

    # ------------------------------------- stage 2.2 silu-> [bs/dp, seq/sp, up_hidden/tp] smem -------------------------------------
    # silu-> [bs/dp, seq/sp, up_hidden/tp]silu-> [bs/dp, seq/sp, up_hidden/tp]
    # silu(x) = x/(1+e^(-x))
    # e^(-x), sfu; 1+e^(-x) cuda, 1/(1+e^(-x)) sfu, x* (1/..), cuda
    
    silu_bytes_stage1=OpBytes(
        input1=Tensor_Loc(swiglu_bytes.input1.dtype, 'smem',[tokens_sum, shard_up_hidden]),
        input2=None,
        output=Tensor_Loc(swiglu_bytes.output.dtype, 'reg',[tokens_sum, shard_up_hidden]),
    )
    
    # hete_post_data_silu=element_wrapper(element_op_bytes=silu_bytes, granularity=granularity, single_chip=single_chip)
    # silu_tb_tile= (gemm_tiling_config[0],gemm_tiling_config[1])
    hete_post_data_silu_1, tb_shape_silu_1=element_wrapper(element_op_bytes=silu_bytes_stage1, granularity=granularity, single_chip=single_chip, batch=2, type="sfu_core")
    

    silu_bytes_stage2=OpBytes(
        input1=Tensor_Loc(swiglu_bytes.input1.dtype, 'reg',[tokens_sum, shard_up_hidden]),
        input2=None,
        output=Tensor_Loc(swiglu_bytes.output.dtype, 'smem',[tokens_sum, shard_up_hidden]),
    )

    hete_post_data_silu_2, tb_shape_silu_2=element_wrapper(element_op_bytes=silu_bytes_stage2, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core")
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_silu_1,hete_post_data_silu_2], single_chip))

    # ------------------------------------- stage 2.3 ⊗  (x @ W1) ⊗ silu(x @ W2) -> [bs/dp, seq/sp, up_hidden/tp] smem -------------------------------------

    # ⊗ element-wise multiplication, (x @ W1) ⊗ silu(x @ W2) -> [bs/dp, seq/sp, up_hidden/tp], element-wise multiplication
    element_mul_bytes=OpBytes(
        input1=Tensor_Loc(swiglu_bytes.output.dtype, 'ddr',[tokens_sum, shard_up_hidden]),
        input2=Tensor_Loc(swiglu_bytes.input1.dtype, 'smem',[tokens_sum, shard_up_hidden]),
        output=Tensor_Loc(swiglu_bytes.output.dtype, 'ddr',[tokens_sum, shard_up_hidden]),
    )

    hete_post_data_mul, tb_shape_mul=element_wrapper(element_op_bytes=element_mul_bytes, granularity=granularity, single_chip=single_chip, batch=1, type="cuda_core")
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_mul], single_chip))

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    smem_fusion_post_data[0] += 1e-06 * local_num_activated_experts
    log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
    # log.info("hete modeling single chip metrics: %s, gemm_tiling_config: %s, silu_tiling_config: %s", get_hete_metrics(smem_fusion_post_data), gemm_tiling_config, silu_tb_tile)

    single_chip_time=smem_fusion_post_data[0]
    log.info("swiglu stage 2 single chip time: %s s", single_chip_time)
    log.info("noc additional time: %s s", additional_time)

    log.info("overall time: %s s", single_chip_time + additional_time)

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    return single_chip_time + additional_time

def swiglu_coarse_group_gemm_stage3(expert_row:np.ndarray, counts:np.ndarray, local_num_activated_experts:int, group_gemm_bs_m_shape_list:list, hidden:int, up_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, swiglu_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    assert granularity.get_mode() == "coarse"
    # print(parallel.world_size())
    
    in_bytes, weight_bytes, out_bytes = swiglu_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_up_hidden = math.ceil(up_hidden / parallel.tp)


    # x: [bs/dp, seq/sp, hidden]
    # W1,W2 [hidden, up_hidden/tp]
    # W3 [up_hidden/tp, hidden]

    # x@W1 -> [bs/dp, seq/sp, up_hidden/tp]
    # x@W2 -> [bs/dp, seq/sp, up_hidden/tp]
    # silu-> [bs/dp, seq/sp, up_hidden/tp]
    # (x @ W1) ⊗ silu(x @ W2) -> [bs/dp, seq/sp, up_hidden/tp]
    # @ W3 -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce -> [bs/dp, seq/sp, hidden]



    # ------------------------------------- stage 3.1 -------------------------------------
    # input[bs/dp, seq/sp, up_hidden/tp], W3[up_hidden/tp, hidden]
    # @ W3 -> [bs/dp, seq/sp, hidden]
    
    
    if (parallel.fsdp == False or parallel.dp == 1):
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1= 0, 0, 0
    elif (parallel.fsdp == True):
        # reconstruct the weight matrix
        tm = TrafficMatrix(parallel.world_size())
        tm.add_intra_group_traffic("dp", shard_up_hidden*hidden*weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1) * local_num_activated_experts, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        # tm.add_intra_group_traffic("dp", 1000/3, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=noc_hop_time_1, link_time_s=noc_ext_max_1)
        log.info("swiglu stage 3 fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", noc_hop_time_1, noc_ext_max_1, noc_overall_time_1)
        # shard_in_bytes = shard_hidden*shard_up_hidden*weight_bytes * (parallel.dp-1)/parallel.dp
        # shard_out_bytes = shard_in_bytes

    # tm.save_heatmap("tm_swiglu.png")
    # single device memory footprint

    gemm3_bytes=OpBytes(
        input1=Tensor_Loc(swiglu_bytes.output.dtype, 'ddr'),
        input2=Tensor_Loc(swiglu_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(swiglu_bytes.output.dtype, 'ddr'),
    )

    local_group_gemm_shape_list=[]
    tokens_sum = 0
    for bs, m in group_gemm_bs_m_shape_list:
        local_group_gemm_shape_list.append((bs, m, hidden, shard_up_hidden))
        tokens_sum += bs * m

    # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
    # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
    # hete_post_data, smem_fusion_post_data, tiling_config=gemm_wrapper(M=shard_tokens, N=hidden, K=shard_up_hidden, gemm_bytes=gemm3_bytes, granularity=granularity, single_chip=single_chip)
    # log.debug("smem_fusion_post_data: %s, tiling_config: %s", smem_fusion_post_data, tiling_config)
    smem_fusion_list, gemm_smem_fusion_post_data, grids =group_gemm_wrapper(shape_list=local_group_gemm_shape_list, gemm_bytes=gemm3_bytes, granularity=granularity, single_chip=single_chip, stats=stats)
    gemm_smem_fusion_post_data[0] += 1e-06 * local_num_activated_experts
    log.info("swiglu stage 3 single device time: %s s", gemm_smem_fusion_post_data[0])
    
    
    if (granularity.get_comp_comm_overlap()==True):
        time_stage3=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)
    elif (granularity.get_comp_comm_overlap()==False):
        time_stage3=gemm_smem_fusion_post_data[0] + noc_overall_time_1
 


    # ------------------------------------- stage 3.2 -------------------------------------
    # tp-level all-reduce -> [bs/dp, seq/sp, hidden]
    # Each tp nodes hold [bs/dp, seq/sp, hidden]
    # tp nodes needs to do add across nodes
    # if parallel.tp == 1:
    #     all_reduce_hop, all_reduce_ext_max=0,0
    # elif parallel.tp > 1 and (parallel == next_parallel) :
    #     all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = all_reduce_wrapper(
    #         all_reduce_op_bytes=swiglu_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy, 
    #         granularity=granularity, dim_to_process="tp", bytes=shard_bs*shard_seq*hidden*weight_bytes)
    # elif parallel.tp > 1 and (next_parallel!=parallel) and (next_parallel.tp == 1) and (parallel.world_size() == next_parallel.world_size()):
    #     all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = reduce_scatter_wrapper(
    #         all_reduce_op_bytes=swiglu_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy, 
    #         granularity=granularity, dim_to_process="tp", bytes=shard_bs*shard_seq*hidden*weight_bytes)
    # else:
    #     log.error("Invalid parallel scheme between current and next. Current Parallel: %s, Next Parallel: %s", parallel, next_parallel)
    #     raise ValueError("Invalid parallel scheme", parallel, next_parallel)                

    if parallel.tp == 1:
        all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = 0, 0, None
    elif parallel.tp > 1 and ((next_parallel == parallel) or (parallel.tp * parallel.dp * parallel.sp == next_parallel.tp * next_parallel.dp * next_parallel.sp) ) :
        # with same parallel world size on activation
        all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = reduce_scatter_wrapper(
            all_reduce_op_bytes=swiglu_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy,
            granularity=granularity, dim_to_process="tp", bytes=tokens_sum*hidden*weight_bytes)
    elif parallel.tp > 1 and (next_parallel.tp ==1) and parallel.dp == next_parallel.dp and parallel.sp == next_parallel.sp:
        all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = all_reduce_wrapper(
            all_reduce_op_bytes=swiglu_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy,
            granularity=granularity, dim_to_process="tp", bytes=tokens_sum*hidden*weight_bytes)
    else:
        log.error("Not implemented.")
        log.error("Invalid parallel scheme between current and next. Current Parallel: %s, Next Parallel: %s", parallel, next_parallel)
        raise ValueError("Invalid parallel scheme", parallel, next_parallel)

    if stats is not None and all_reduce_traffic is not None:
        stats.append_traffic(all_reduce_traffic, hop_time_s=all_reduce_hop, link_time_s=all_reduce_ext_max)

    # grids = [shard_tokens / tiling_config[0], hidden / tiling_config[1]]
    waves = np.prod(grids) / single_chip.sm_count
    e2e_time = get_comp_comm_e2e_time(compute_time=time_stage3, network_hop_latency=all_reduce_hop, network_link_time=all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
    log.info("swiglu stage 3 comp+comm e2e time: %s s", e2e_time)
    return e2e_time


def swiglu_coarse_group_gemm(expert_row:np.ndarray, counts:np.ndarray, hidden:int, up_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, swiglu_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    
    # expert_row 表示, 每个ep组(total_experts/EP个), 每个expert激活次数
    # counts则是bin count,第index个数的值num_expert表示, 有index个token的expert总共有num_expert个
    # e.g.,
    # expert_row: [23 18 16 15 15 13 20 19 16 15 16 30 12 16 21 24 19 16 19 20 17 12 15 16 13 15 16  7 14 18 20 20]
    # counts: [0 0 0 0 0 0 0 1 0 0 0 0 2 2 1 5 7 1 2 3 4 1 0 1 1 0 0 0 0 0 1]
    log.info("swiglu_coarse_group_gemm")
    log.info("expert_row: %s", expert_row)
    log.info("counts: %s", counts)

    local_num_activated_experts = expert_row.shape[0] - counts[0]
    log.info("local_num_activated_experts: %s", local_num_activated_experts)
    group_gemm_bs_m_shape_list=[]
    counts_mode = os.environ.get(
        "DEEPSTACK_SWIGLU_COUNTS_MODE", "current_corrected"
    )
    if counts_mode == "paper_legacy":
        # Compatibility mode for results produced before the 2026-07-13 fix.
        # This intentionally preserves the historical off-by-one behavior so
        # reviewers can reproduce the submitted paper values bit-for-bit.
        activated_tokens=0
        for activated_expert_count in counts:
            if activated_tokens == 0:
                activated_tokens += 1
            if activated_expert_count == 0:
                activated_tokens += 1
            else:
                batch = int(activated_expert_count)
                m = activated_tokens
                group_gemm_bs_m_shape_list.append((batch, m))
                activated_tokens += 1
    elif counts_mode == "current_corrected":
        # counts[m] is the number of experts receiving exactly m tokens;
        # zero-token experts do not perform GEMM work.
        for activated_tokens, activated_expert_count in enumerate(counts):
            if activated_tokens == 0 or activated_expert_count == 0:
                continue
            group_gemm_bs_m_shape_list.append((int(activated_expert_count), activated_tokens))
    else:
        raise ValueError(
            "DEEPSTACK_SWIGLU_COUNTS_MODE must be "
            f"'current_corrected' or 'paper_legacy', got {counts_mode!r}"
        )

    # log.info("group_gemm_bs_m_shape_list: %s", group_gemm_bs_m_shape_list)

    # print(f"local_num_activated_experts: {local_num_activated_experts}")
    
    assert granularity.get_mode() == "coarse"
    # print(parallel.world_size())
    
    
    in_bytes, weight_bytes, out_bytes = swiglu_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_up_hidden = math.ceil(up_hidden / parallel.tp)

    # x: [bs/dp, seq/sp, hidden]
    # W1,W2 [hidden, up_hidden/tp]
    # W3 [up_hidden/tp, hidden]

    # x@W1 -> [bs/dp, seq/sp, up_hidden/tp]
    # x@W2 -> [bs/dp, seq/sp, up_hidden/tp]
    # silu-> [bs/dp, seq/sp, up_hidden/tp]
    # (x @ W1) ⊗ silu(x @ W2) -> [bs/dp, seq/sp, up_hidden/tp]
    # @ W3 -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce -> [bs/dp, seq/sp, hidden]

    # ------------------------------------- stage 1 -------------------------------------

    # let's model it one-by-one
    # 1. x@W1 -> [bs/dp, seq/sp, hidden] @ [hidden, up_hidden/tp] -> [bs/dp, seq/sp, up_hidden/tp]
    # x: [shard_bs, shard_seq, shard_hidden]
    # W1: [shard_hidden, shard_up_hidden]
    # x@W1: [shard_bs, shard_seq, shard_up_hidden]
    # swiglu_coarse_group_gemm_stage1(bs:int, seq:int, hidden:int, up_hidden:int, parallel:ParallelScheme, swiglu_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    time_stage1, waves = swiglu_coarse_group_gemm_stage1(expert_row=expert_row, counts=counts, local_num_activated_experts=local_num_activated_experts, group_gemm_bs_m_shape_list=group_gemm_bs_m_shape_list,
     hidden=hidden, up_hidden=up_hidden, parallel=parallel, swiglu_bytes=swiglu_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # ------------------------------------- stage 2 -------------------------------------
    # fused:
    # x@W2 -> [bs/dp, seq/sp, up_hidden/tp]
    # silu (x@W2) -> [bs/dp, seq/sp, up_hidden/tp]
    # ⊗ element-wise multiplication, (x @ W1) ⊗ silu(x @ W2) -> [bs/dp, seq/sp, up_hidden/tp], element-wise multiplication
    time_stage2 = swiglu_coarse_group_gemm_stage2(expert_row=expert_row, counts=counts, local_num_activated_experts=local_num_activated_experts, group_gemm_bs_m_shape_list=group_gemm_bs_m_shape_list,
     hidden=hidden, up_hidden=up_hidden, parallel=parallel, swiglu_bytes=swiglu_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # ------------------------------------- stage 3 -------------------------------------
    # input[bs/dp, seq/sp, up_hidden/tp], W3[up_hidden/tp, hidden]
    # @ W3 -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce -> [bs/dp, seq/sp, hidden]
    time_stage3 = swiglu_coarse_group_gemm_stage3(expert_row=expert_row, counts=counts, local_num_activated_experts=local_num_activated_experts, group_gemm_bs_m_shape_list=group_gemm_bs_m_shape_list,
     hidden=hidden, up_hidden=up_hidden, parallel=parallel, next_parallel=next_parallel, swiglu_bytes=swiglu_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)


    # ------------------------------------- stage 4 -------------------------------------
    # add residual
    # time_stage4 =swiglu_coarse_group、_token_stage4(bs=bs, seq=seq, hidden=hidden, up_hidden=up_hidden, parallel=parallel, swiglu_bytes=swiglu_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # now we move add_residual into a seperate operator

    # log.info("Whole swiglu coarse e2e time: %s s", time_stage1 + time_stage2 + time_stage3 + time_stage4)
    # return time_stage1 + time_stage2 + time_stage3 + time_stage4

    # log.info("Whole swiglu coarse e2e time: %s s", time_stage1 + time_stage2 + time_stage3)
    log.info("Whole swiglu coarse e2e time: %s s, time_stage1: %s, time_stage2: %s, time_stage3: %s", time_stage1 + time_stage2 + time_stage3, time_stage1, time_stage2, time_stage3)
    return time_stage1 + time_stage2 + time_stage3, waves
