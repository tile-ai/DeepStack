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
from mosaic.op_dtype.fa_decode_wrapper import fa_decode_wrapper
# from mosaic.op_dtype.fa_prefill_wrapper import fa_prefill_wrapper
import cProfile

from tilesight.arch import *
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
import logging
log = logging.getLogger(__name__) 
from mosaic.collectives import all_reduce_wrapper
from mosaic.utils import get_comp_comm_e2e_time

# from mosaic.llm.rope.ropeq_coarse import ropeq_coarse
from mosaic.llm.rope.ropeq_decoding_coarse_raw import ropeq_decoding_coarse_raw
# from mosaic.llm.rope.ropek_coarse import ropek_coarse
from mosaic.llm.rope.ropek_decoding_coarse_raw import ropek_decoding_coarse_raw
from mosaic.collectives import all_gather_sp_fission, reduce_scatter_wrapper,reduce_partial_brodcast_wrapper
from mosaic.cost.energy_record import EnergyRecord
from mosaic.cost.op_perf_stats import OpPerfStats

def gqa_decode_coarse_stage1(bs:int, seq:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme,
    atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):

    group_size = math.ceil (num_head/num_kv_head)
    wq_hidden = num_head * head_dim

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    # shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)


    # gqa:
    # head_dim = hidden/num_head, g = num_head/num_kv_head

    # x[bs/dp,seq/sp,hidden]
    # wq[hidden,hidden/tp]
    # wk, wv, [hidden, hiddden/tp/g]
    # wo [hidden/tp, hidden]
    # parallel.cp == 1, atten_parallel.cp * atten_parallel.sp == parallel.sp 

    # ------------------------------------- stage 1 -------------------------------------

    # q = x@wq -> [bs/dp, seq/sp, hidden/tp] (if fsdp, then dp-level reconstruct wq)
    # smem fused reshape q + transpose q : reshape to [bs/dp, seq/sp, num_head/tp, head_dim], transpose to [bs/dp, num_head/tp, seq/sp, head_dim]
    # rope(q) -> [bs/dp, num_head/tp, seq/sp, head_dim]
    # all-gather to q_gathered [bs/dp, num_head/tp, seq/atten.sp, head_dim]  
    def q_and_rope():
        if (parallel.fsdp == False or parallel.dp == 1):
            hop_time_1, ext_max_1, noc_overall_time_1, ext_traffic_1 = 0, 0, 0, None
        elif parallel.fsdp==True:
            # reconstruct wq at dp level
            tm = TrafficMatrix(parallel.world_size())
            bytes_each_pair = hidden * shard_hidden * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
            tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
            hop_time_1, ext_max_1, noc_overall_time_1, ext_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
            if stats is not None:
                stats.append_traffic(ext_traffic_1, hop_time_s=hop_time_1, link_time_s=ext_max_1)

            log.info("gqa decode coarse stage 1 fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_1, ext_max_1, noc_overall_time_1)
        
        gemm1_bytes=OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
            input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
            output=Tensor_Loc(atten_bytes.output.dtype, 'smem'),
        )

        smem_fusion_list=[]

        # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
        # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
        
        hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_bs*shard_seq, N=shard_hidden, K=hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
        log.info("M,N,K,tiling config: %s, %s, %s, %s", shard_bs*shard_seq, shard_hidden, hidden, gemm_tiling_config)
        grids = [shard_bs * shard_seq / gemm_tiling_config[0], shard_hidden / gemm_tiling_config[1]]
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))
        

        if (granularity.get_comp_comm_overlap()==True):
            # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
            additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
            log.info("gqa decode coarse stage 1 fsdp additional time: %s s", additional_time)
        elif (granularity.get_comp_comm_overlap()==False):
            # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
            additional_time = noc_overall_time_1
        
        
        
        reshape_transpose_bytes=OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem',[shard_bs, shard_num_head, shard_seq, head_dim]),
            input2=None,
            output=Tensor_Loc(atten_bytes.output.dtype, 'smem',[shard_bs, shard_num_head, shard_seq, head_dim]),
        )

        reshape_transpose_tb_tile = (gemm_tiling_config[0], gemm_tiling_config[1])
        
        hete_post_data_reshape_transpose, tb_shape_reshape_transpose=element_wrapper(element_op_bytes=reshape_transpose_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=reshape_transpose_tb_tile)

        smem_fusion_list.append(hete_reg_fusion([hete_post_data_reshape_transpose], single_chip))

        # next we need to do a rope
        # ropeq_coarse(bs:int, head:int, seq:int, head_dim:int, parallel:ParallelScheme, next_parallel:ParallelScheme, rope_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy)

        rope_bytes = OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
            input2=Tensor_Loc(torch.float16, 'ddr'),
            output=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        )

        # ropeq_time = ropeq_coarse(bs=bs, head=num_head, seq=seq, head_dim=head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        
        rope_smem_fusion_list, rope_hop_latency, rope_ext_max = ropeq_decoding_coarse_raw(bs=bs, head=num_head, seq=seq, head_dim=head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
        # if stats is not None:
        #     stats.append_traffic(rope_ext_max, hop_time_s=rope_hop_latency, link_time_s=rope_ext_max)
        smem_fusion_list.extend(rope_smem_fusion_list)

        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling single chip metrics: %s, gemm_tiling_config: %s, tb_shape_reshape_tiling: %s", get_hete_metrics(smem_fusion_post_data), gemm_tiling_config, tb_shape_reshape_transpose)

        single_chip_time=smem_fusion_post_data[0] + additional_time

        overall_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=rope_hop_latency, network_link_time=rope_ext_max, waves=np.prod(grids)/single_chip.sm_count, overlap=granularity.get_comp_comm_overlap())

        log.info("gqa decode coarse stage 1 2 3 q + rope overall time: %s s", overall_time)
        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        return overall_time

    def k_and_rope():
        # ------------------------------------- stage 2 -------------------------------------
        # seq = 1
        # k = x@wk -> [bs/dp, seq, hidden/tp/g] (if fsdp, then dp-level reconstruct wk)
        # smem fused reshape k + transpose k : reshape to [bs/dp, seq/sp, num_head/tp/g, head_dim], transpose to [bs/dp, num_head/tp/g, head_dim, seq/sp]
        # rope(k) -> [bs/dp, num_head/tp/g, head_dim, seq/sp]
        # all-gather to k_gathered [bs/dp, num_head/tp/g, head_dim, seq/atten.cp]
        # reconstruct to the [bs/dp, num_head/tp/g, head_dim, shard_cached_kv/atten.cp]
        
        if (parallel.fsdp == False or parallel.dp == 1):
            hop_time_1, ext_max_1, noc_overall_time_1, ext_traffic_1 = 0, 0, 0, None
        elif parallel.fsdp==True:
            # reconstruct wq at dp level
            tm = TrafficMatrix(parallel.world_size())
            bytes_each_pair = hidden * shard_kv_hidden * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
            tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
            hop_time_1, ext_max_1, noc_overall_time_1, ext_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
            if stats is not None:
                stats.append_traffic(ext_traffic_1, hop_time_s=hop_time_1, link_time_s=ext_max_1)

            log.info("gqa decode coarse stage 2 fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_1, ext_max_1, noc_overall_time_1)
        
        gemm1_bytes=OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
            input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
            output=Tensor_Loc(atten_bytes.output.dtype, 'smem'),
        )

        smem_fusion_list=[]

        # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
        # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
        # print("GEMM, M, N, K: ", shard_bs*shard_seq, shard_kv_hidden, hidden)
        hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_bs*shard_seq, N=shard_kv_hidden, K=hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
        grids = [shard_bs * shard_seq / gemm_tiling_config[0], shard_kv_hidden / gemm_tiling_config[1]]
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

        # log.error("gemm_smem_fusion_post_data: %s", gemm_smem_fusion_post_data)
        

        if (granularity.get_comp_comm_overlap()==True):
            # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
            additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
            log.info("gqa decode coarse stage 2 fsdp additional time: %s s", additional_time)
        elif (granularity.get_comp_comm_overlap()==False):
            # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
            additional_time = noc_overall_time_1
        
        
        
        reshape_transpose_bytes=OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem',[shard_bs, shard_num_kv_head, shard_seq, head_dim]),
            input2=None,
            output=Tensor_Loc(atten_bytes.output.dtype, 'smem',[shard_bs, shard_num_kv_head, shard_seq, head_dim]),
        )

        reshape_transpose_tb_tile = (gemm_tiling_config[0], gemm_tiling_config[1])
        # print("reshape_transpose_tb_tile: ", reshape_transpose_tb_tile)
        
        hete_post_data_reshape_transpose, tb_shape_reshape_transpose=element_wrapper(element_op_bytes=reshape_transpose_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=reshape_transpose_tb_tile)

        smem_fusion_list.append(hete_reg_fusion([hete_post_data_reshape_transpose], single_chip))

        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling single chip metrics: %s, gemm_tiling_config: %s, tb_shape_reshape_tiling: %s", get_hete_metrics(smem_fusion_post_data), gemm_tiling_config, tb_shape_reshape_transpose)

        single_chip_time=smem_fusion_post_data[0] + additional_time

        

        # next we need to do a rope
        # ropeq_coarse(bs:int, head:int, seq:int, head_dim:int, parallel:ParallelScheme, next_parallel:ParallelScheme, rope_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy)

        rope_bytes = OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
            input2=Tensor_Loc(torch.float16, 'ddr'),
            output=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        )

        # ropek_time = ropek_decoding_coarse_raw(bs=bs, head=num_kv_head, seq=seq, head_dim=head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        rope_smem_fusion_list, rope_hop_latency, rope_ext_max = ropek_decoding_coarse_raw(bs=bs, head=num_kv_head, seq=seq, head_dim=head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
        smem_fusion_list.extend(rope_smem_fusion_list)

        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling single chip metrics: %s, gemm_tiling_config: %s, tb_shape_reshape_tiling: %s", get_hete_metrics(smem_fusion_post_data), gemm_tiling_config, tb_shape_reshape_transpose)

        single_chip_time=smem_fusion_post_data[0] + additional_time

        overall_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=rope_hop_latency, network_link_time=rope_ext_max, waves=np.prod(grids)/single_chip.sm_count, overlap=granularity.get_comp_comm_overlap())

        log.info("gqa decode coarse stage 2 k + rope overall time: %s s", overall_time)
        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        return overall_time

    def v_no_rope():
        # v = x@wv -> [bs/dp, seq/sp, hidden/tp/g] (if fsdp, then dp-level reconstruct wv)
        # smem fused reshape v + transpose v : reshape to [bs/dp, seq/sp, num_head/tp/g, head_dim], transpose to [bs/dp, num_head/tp/g, seq/sp, head_dim]
        # all-gather to v_gathered [bs/dp, num_head/tp/g, seq/atten.cp, head_dim]
        if (parallel.fsdp == False or parallel.dp == 1):
            hop_time_1, ext_max_1, noc_overall_time_1, ext_traffic_1 = 0, 0, 0, None
        elif parallel.fsdp==True:
            # reconstruct wq at dp level
            tm = TrafficMatrix(parallel.world_size())
            bytes_each_pair = hidden * shard_kv_hidden * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
            tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
            hop_time_1, ext_max_1, noc_overall_time_1, ext_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
            if stats is not None:
                stats.append_traffic(ext_traffic_1, hop_time_s=hop_time_1, link_time_s=ext_max_1)

            log.info("gqa decode coarse stage 3 fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_1, ext_max_1, noc_overall_time_1)
        
        gemm1_bytes=OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
            input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
            output=Tensor_Loc(atten_bytes.output.dtype, 'smem'),
        )

        smem_fusion_list=[]

        # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
        # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
        hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_bs*shard_seq, N=shard_kv_hidden, K=hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
        grids = [shard_bs * shard_seq / gemm_tiling_config[0], shard_kv_hidden / gemm_tiling_config[1]]
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

        # log.error("gemm_smem_fusion_post_data: %s", gemm_smem_fusion_post_data)
        

        if (granularity.get_comp_comm_overlap()==True):
            # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
            additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
            log.info("gqa decode coarse stage 3 fsdp additional time: %s s", additional_time)
        elif (granularity.get_comp_comm_overlap()==False):
            # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
            additional_time = noc_overall_time_1
        
        
        
        reshape_transpose_bytes=OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem',[shard_bs, shard_num_kv_head, shard_seq, head_dim]),
            input2=None,
            output=Tensor_Loc(atten_bytes.output.dtype, 'smem',[shard_bs, shard_num_kv_head, shard_seq, head_dim]),
        )

        reshape_transpose_tb_tile = (gemm_tiling_config[0], gemm_tiling_config[1])
        
        hete_post_data_reshape_transpose, tb_shape_reshape_transpose=element_wrapper(element_op_bytes=reshape_transpose_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=reshape_transpose_tb_tile)

        smem_fusion_list.append(hete_reg_fusion([hete_post_data_reshape_transpose], single_chip))

        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling single chip metrics: %s, gemm_tiling_config: %s, tb_shape_reshape_tiling: %s", get_hete_metrics(smem_fusion_post_data), gemm_tiling_config, tb_shape_reshape_transpose)

        single_chip_time=smem_fusion_post_data[0] + additional_time

        bytes_needed_per_device = in_bytes * shard_bs * head_dim * shard_num_kv_head *(math.ceil(seq/atten_parallel.cp) - shard_seq)

        if parallel.sp == 1 or parallel.sp <= atten_parallel.cp:
            gather_v_latency, gather_v_ext_max, gather_v_traffic = 0, 0, None
        else:
            gather_v_latency, gather_v_ext_max, gather_v_traffic = all_gather_sp_fission(parallel=parallel, noc_hierarchy=noc_hierarchy, bytes_needed_per_device=bytes_needed_per_device, next_virtual_sp=atten_parallel.cp)
            if stats is not None:
                stats.append_traffic(gather_v_traffic, hop_time_s=gather_v_latency, link_time_s=gather_v_ext_max)

        log.info("gather_v_latency: %s, gather_v_ext_max: %s", gather_v_latency, gather_v_ext_max)

        waves=np.prod(grids)/single_chip.sm_count

        overall_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=gather_v_latency, network_link_time=gather_v_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
        # next we need to do a rope
        # ropeq_coarse(bs:int, head:int, seq:int, head_dim:int, parallel:ParallelScheme, next_parallel:ParallelScheme, rope_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy)

        # log.info("gqa decode coarse stage 3 single chip time before gather v: %s s", single_chip_time)
        # log.info("gqa decode coarse stage 3 all gather time: %s s", gather_v_latency + gather_v_ext_max)
        # log.info("gqa decode coarse stage 3 overall time: %s s", e2e_time)
        log.info("gqa decode coarse stage 3 v without rope overall time: %s s", overall_time)
        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        return overall_time


    q_time = q_and_rope()

    k_time = k_and_rope()

    v_no_rope_time = v_no_rope()

    overall_time = q_time + k_time + v_no_rope_time
    log.info("gqa decode coarse q_and_rope time: %s s", q_time)
    log.info("gqa decode coarse k_and_rope time: %s s", k_time)
    log.info("gqa decode coarse v_no_rope time: %s s", v_no_rope_time)
    log.info("gqa decode coarse qkv projection + q k rope + potential all gather overall time: %s s", overall_time)
    return overall_time


def gqa_decode_coarse_stage4(bs:int, seq:int, cached_kv:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme,
    atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    

    # 在 GQA 里, g = K/V head 数，而 group size = Q head 数 ÷ K/V head 数。
    group_size = math.ceil (num_head/num_kv_head)

    wq_hidden = num_head * head_dim

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_seq_kv = math.ceil(seq / atten_parallel.cp)

    shard_grouped_seq_q = shard_seq_q * math.ceil(shard_num_head / shard_num_kv_head)
    shard_group_size = math.ceil(shard_num_head / shard_num_kv_head)


    # gqa:
    # head_dim = hidden/num_head, g = num_head/num_kv_head

    # x[bs/dp,seq/sp,hidden]
    # wq[hidden,hidden/tp]
    # wk, wv, [hidden, hiddden/tp/g]
    # wo [hidden/tp, hidden]
    # parallel.cp == 1, atten_parallel.cp * atten_parallel.sp == parallel.sp 

    # ------------------------------------- stage 4 -------------------------------------
    # flash attention (though need cp-level all-reduce + scatter -> reduce-scatter)
    # q_gathered [bs/dp, num_head/tp, seq/atten.sp, head_dim]
    # k_gathered [bs/dp, num_head/tp/g, head_dim, seq/atten.cp]
    # s = q @ k -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # # # # s = softmax(s), local softmax
    # s_max(s,last_dim) -> [bs/dp, num_head/tp, seq/atten.sp, 1]
    # s_exp = exp(s - s_max) -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # s_exp_sum(last_dim) -> [bs/dp, num_head/tp, seq/atten.sp, 1]
    # scaled_s = s_exp / s_exp_sum -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # s *= scaled_s -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # p = s @ v_gathered -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp] @ [bs/dp, num_head/tp/g, seq/atten.cp, head_dim] -> [bs/dp, num_head/tp, seq/atten.sp, head_dim]

    
    # for a single chip, it's computation is:

    # q [shard_bs, shard_num_head, shard_seq_q, head_dim]
    # k' [shard_bs, shard_num_head, head_dim, shard_seq_kv]
    # v [shard_bs, shard_num_head, shard_seq_kv, head_dim]
    # s = q @ k -> [shard_bs, shard_num_head, shard_seq_q, shard_seq_kv]
    # s = softmax(s), local softmax
    # s_max(s,last_dim) -> [shard_bs, shard_num_head, shard_seq_q, 1]
    # s_exp = exp(s - s_max) -> [shard_bs, shard_num_head, shard_seq_q, shard_seq_kv]
    # s_exp_sum(last_dim) -> [shard_bs, shard_num_head, shard_seq_q, 1]
    # scaled_s = s_exp / s_exp_sum -> [shard_bs, shard_num_head, shard_seq_q, shard_seq_kv]
    # s *= scaled_s -> [shard_bs, shard_num_head, shard_seq_q, shard_seq_kv]
    # p = s @ v -> [shard_bs, shard_num_head, shard_seq_q, shard_seq_kv] @ [shard_bs, shard_num_head, shard_seq_kv, head_dim] -> [shard_bs, shard_num_head, shard_seq_q, head_dim]

    # to summarize, the parameters are: 
    # shard_bs, shard_num_head, shard_seq_q, shard_seq_kv, head_dim 
    # atten_parallel.cp

    # then a cp-level reduce-scatter

    # then a reconstruct, a element-wise op, new [shard_bs, shard_num_head, shard_seq_q, shard_seq_kv] / or [shard_bs, shard_num_head, shard_seq, seq] do a [shard_bs, shard_num_head, shard_seq, 1] （rescale）

    grids, single_chip_time = fa_decode_wrapper(bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    # grids, single_chip_time = fa_prefill_wrapper(bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    
    
    waves= np.prod(grids)/single_chip.sm_count

    if (atten_parallel.cp == 1):
        log.info("gqa decode coarse stage 4 overall time: %s s", single_chip_time)
        return single_chip_time
    elif (atten_parallel.cp > 1 and atten_parallel.cp <= parallel.sp):
        # reduce-scatter along atten.cp
        # each device holding: [shard_bs, shard_num_head, shard_seq_q, head_dim]
        cp_reduce_latency, cp_reduce_ext_max, cp_reduce_traffic = reduce_scatter_wrapper(all_reduce_op_bytes=atten_bytes, parallel=atten_parallel, 
        noc_hierarchy=noc_hierarchy, granularity=granularity, dim_to_process="cp", bytes=shard_bs*shard_num_head*shard_seq_q*head_dim*in_bytes)

        log.info("cp_reduce_scatter_latency: %s, cp_reduce_ext_max: %s", cp_reduce_latency, cp_reduce_ext_max)

        overall_time =get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=cp_reduce_latency, network_link_time=cp_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
        log.info("gqa decode coarse stage 4 overall time: %s s", overall_time)
        if stats is not None:
            stats.append_traffic(cp_reduce_traffic, hop_time_s=cp_reduce_latency, link_time_s=cp_reduce_ext_max)
        
        return overall_time
    elif (atten_parallel.cp > parallel.sp):
        cp_all_reduce_latency, cp_all_reduce_ext_max, cp_all_reduce_traffic = all_reduce_wrapper(all_reduce_op_bytes=atten_bytes, parallel=atten_parallel,
        noc_hierarchy=noc_hierarchy, granularity=granularity, dim_to_process="cp", bytes=shard_bs*shard_num_head*shard_seq_q*head_dim*in_bytes)

        log.info("cp_all_reduce_latency: %s, cp_all_reduce_ext_max: %s", cp_all_reduce_latency, cp_all_reduce_ext_max)

        overall_time =get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=cp_all_reduce_latency, network_link_time=cp_all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
        log.info("gqa decode coarse stage 4 overall time: %s s", overall_time)
        if stats is not None:
            stats.append_traffic(cp_all_reduce_traffic, hop_time_s=cp_all_reduce_latency, link_time_s=cp_all_reduce_ext_max)
        
        return overall_time
    else:
        raise NotImplementedError("not desinged for this case", parallel, atten_parallel)


def gqa_decode_coarse_stage5(bs:int, seq:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme,
    atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):

    group_size = math.ceil (num_head/num_kv_head)

    wq_hidden = num_head * head_dim

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_seq_kv = math.ceil(seq / atten_parallel.cp)

    shard_grouped_seq_q = shard_seq_q * math.ceil(shard_num_head / shard_num_kv_head)
    shard_group_size = math.ceil(shard_num_head / shard_num_kv_head)   

    def rescale_transpose_reshape():

        # ------------------------------------- stage 5 -------------------------------------
        # stage 5
        # smem fused
        # then a reconstruct, a element-wise op, new [shard_bs, shard_num_head, shard_seq_q, head_dim] / or [shard_bs, shard_num_head, shard_seq, head_dim] do a [shard_bs, shard_num_head, shard_seq, 1x2] (rescale： sum and max)
        # transpose p + reshape p  : 
        # [bs/dp, num_head/tp, seq/(atten.sp*atten.cp), head_dim] == [bs/dp, num_head/tp, seq/sp, head_dim]
        # transpose to [bs/dp, seq/sp, num_head/tp, head_dim]
        # reshape to [bs/dp, seq/sp, hidden/tp]    
        smem_fusion_list=[]
        rescale_bytes=OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr',[shard_bs, shard_num_head, shard_seq, head_dim]),
            input2=Tensor_Loc(atten_bytes.input1.dtype, 'ddr',[shard_bs, shard_num_head, shard_seq, 1]),
            output=Tensor_Loc(atten_bytes.output.dtype, 'smem',[shard_bs, shard_num_head, shard_seq, head_dim]),
        )
        
        hete_post_data_rescale, tb_shape_rescale=element_wrapper(element_op_bytes=rescale_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=None)

        smem_fusion_list.append(hete_reg_fusion([hete_post_data_rescale], single_chip))

        transpose_reshape_bytes=OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, shard_hidden]),
            input2=None,
            output=Tensor_Loc(atten_bytes.output.dtype, 'ddr',[shard_bs, shard_seq, shard_hidden]),
        )
        
        hete_post_data_transpose_reshape, tb_shape_transpose_reshape=element_wrapper(element_op_bytes=transpose_reshape_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=None)

        smem_fusion_list.append(hete_reg_fusion([hete_post_data_transpose_reshape], single_chip))

        grids = [shard_bs/tb_shape_transpose_reshape[0], shard_seq/tb_shape_transpose_reshape[1], shard_hidden/tb_shape_transpose_reshape[2]]

        # smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        # log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        
        # time_stage5 = smem_fusion_post_data[0] 
        # log.info("gqa decode coarse stage 5 overall time: %s s", time_stage5)

        # return time_stage5
        
        return smem_fusion_list, grids
    
    def out_projection():
        # ------------------------------------- stage 6 -------------------------------------
        # o = p@wo  [bs/dp, seq/sp, hidden/tp] @ [tp/hidden, hidden] -> [bs/dp, seq/sp, hidden](if fsdp, then dp-level reconstruct wo)
        # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
        # output [bs/dp, seq/sp, hidden]    
        
        if (parallel.fsdp == False or parallel.dp == 1):
            hop_time_1, ext_max_1, noc_overall_time_1= 0, 0, 0
        elif parallel.fsdp==True:
            # reconstruct wq at dp level
            tm = TrafficMatrix(parallel.world_size())
            bytes_each_pair = hidden * shard_hidden * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
            tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
            hop_time_1, ext_max_1, noc_overall_time_1, noc_traffic_1= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
            if stats is not None:
                stats.append_traffic(noc_traffic_1, hop_time_s=hop_time_1, link_time_s=ext_max_1)
            
            log.info("gqa decode coarse stage 6 fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_1, ext_max_1, noc_overall_time_1)
        
        gemm1_bytes=OpBytes(
            input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
            input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
            output=Tensor_Loc(atten_bytes.output.dtype, 'smem'),
        )

        smem_fusion_list=[]

        # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
        # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
        hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_bs*shard_seq, N=hidden, K=shard_hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
        grids = [shard_bs * shard_seq / gemm_tiling_config[0], hidden / gemm_tiling_config[1]]
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)

        log.info("GEMM M: %s, N: %s, K: %s", shard_bs * shard_seq, hidden, shard_hidden)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling single chip metrics: %s, gemm_tiling_config: %s", get_hete_metrics(smem_fusion_post_data), gemm_tiling_config)

        # log.error("gemm_smem_fusion_post_data: %s", gemm_smem_fusion_post_data)
        

        if (granularity.get_comp_comm_overlap()==True):
            # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
            additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
            log.info("gqa decode coarse stage 6 fsdp additional time: %s s", additional_time)
        elif (granularity.get_comp_comm_overlap()==False):
            # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
            additional_time = noc_overall_time_1

        # single_chip_time=smem_fusion_post_data[0] + additional_time

        # waves=np.prod(grids)/single_chip.sm_count

        if (parallel.tp > 1 and next_parallel.tp == parallel.tp):
            # all_reduce_time=all_reduce_wrapper(parallel=parallel, noc_hierarchy=noc_hierarchy, bytes=shard_bs*shard_seq*shard_hidden*out_bytes, dim_to_process="tp")
            # each device holding [shard_bs, shard_seq, hidden], tp-level all-reduce
            all_reduce_latency, all_reduce_ext_max, all_reduce_traffic = all_reduce_wrapper(all_reduce_op_bytes=atten_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy, granularity=granularity, 
                dim_to_process="tp", bytes=shard_bs*shard_seq*shard_hidden*out_bytes)
        elif (parallel.tp > 1 and next_parallel.ep !=1 and (parallel.tp * parallel.sp *parallel.dp == next_parallel.tp * next_parallel.ep * next_parallel.sp * next_parallel.dp)):
            all_reduce_latency, all_reduce_ext_max, all_reduce_traffic = reduce_partial_brodcast_wrapper(all_reduce_op_bytes=atten_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy, granularity=granularity,
                dim_to_process="tp", broadcast_degree=next_parallel.tp, bytes=shard_bs*shard_seq*shard_hidden*out_bytes)

            # tmp fix
            all_reduce_latency_2, all_reduce_ext_max_2, all_reduce_traffic_2 = all_reduce_wrapper(all_reduce_op_bytes=atten_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy, granularity=granularity, 
                dim_to_process="tp", bytes=shard_bs*shard_seq*shard_hidden*out_bytes)

            all_reduce_latency = min(all_reduce_latency, all_reduce_latency_2)
            all_reduce_ext_max = min(all_reduce_ext_max, all_reduce_ext_max_2)

        elif (parallel.tp == 1):
            all_reduce_latency, all_reduce_ext_max, all_reduce_traffic = 0, 0, None
        else:
            all_reduce_latency, all_reduce_ext_max, all_reduce_traffic = 0, 0, None
            # raise NotImplementedError("not desinged for this case", parallel, next_parallel)
        
        if stats is not None and all_reduce_traffic is not None:
            stats.append_traffic(all_reduce_traffic, hop_time_s=all_reduce_latency, link_time_s=all_reduce_ext_max)
        # e2e_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=all_reduce_latency, network_link_time=all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
        # next we need to do a rope
        # ropeq_coarse(bs:int, head:int, seq:int, head_dim:int, parallel:ParallelScheme, next_parallel:ParallelScheme, rope_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy)

        # log.info("gqa decode coarse stage 6 single chip time before all reduce: %s s", single_chip_time)
        # log.info("gqa decode coarse stage 6 all reduce time: %s s", all_reduce_latency + all_reduce_ext_max)
        # log.info("gqa decode coarse stage 6 overall time: %s s", e2e_time)
        # return e2e_time
        return additional_time, smem_fusion_list, grids, all_reduce_latency, all_reduce_ext_max

    # processing compute-comm

    reshape_smem_fusion_list, reshape_grids = rescale_transpose_reshape()
    out_projection_additional_time, out_projection_smem_fusion_list, out_projection_grids, tp_all_reduce_latency, tp_all_reduce_ext_max = out_projection()

    # print("reshape_smem_fusion_list: ", reshape_smem_fusion_list)
    # print("out_projection_smem_fusion_list: ", out_projection_smem_fusion_list)
    smem_fusion_list = reshape_smem_fusion_list + out_projection_smem_fusion_list

    # print("smem_fusion_list: ", smem_fusion_list)
    grids = reshape_grids if np.prod(reshape_grids) < np.prod(out_projection_grids) else out_projection_grids

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    waves=np.prod(grids)/single_chip.sm_count

    single_chip_time=smem_fusion_post_data[0] + out_projection_additional_time
    # log.info("gqa decode coarse stage 6 single chip time before all reduce: %s s", single_chip_time)
    # log.info ("gqa decode coarse stage 6 all reduce time: %s s", tp_all_reduce_latency + tp_all_reduce_ext_max)
    e2e_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=tp_all_reduce_latency, network_link_time=tp_all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())

    log.info("gqa decode coarse stage 5 6 reshape + out projection overall time: %s s", e2e_time)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    return e2e_time


def gqa_decode_coarse(bs:int, seq:int, cached_kv:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme,
    atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    group_size = math.ceil (num_head/num_kv_head)
    wq_hidden = num_head * head_dim

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_head = math.ceil(num_head / parallel.tp)
    shard_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)
    gqa_decode_stats = OpPerfStats(op_name="gqa_decode", dump_perf_log=granularity.dump_perf_log)

    # gqa:
    # head_dim = hidden/num_head, g = num_head/num_kv_head

    # x[bs/dp,seq/sp,hidden]
    # wq[hidden,hidden/tp]
    # wk, wv, [hidden, hiddden/tp/g]
    # wo [hidden/tp, hidden]
    # parallel.cp == 1, atten_parallel.cp * atten_parallel.sp == parallel.sp 

    # stage 1 
    # q = x@wq -> [bs/dp, seq/sp, hidden/tp] (if fsdp, then dp-level reconstruct wq)
    # smem fused reshape q + transpose q : reshape to [bs/dp, seq/sp, num_head/tp, head_dim], transpose to [bs/dp, num_head/tp, seq/sp, head_dim]
    # rope(q) -> [bs/dp, num_head/tp, seq/sp, head_dim]
    # all-gather to q_gathered [bs/dp, num_head/tp, seq/atten.sp, head_dim]  

    # stage 2
    # k = x@wk -> [bs/dp, seq/sp, hidden/tp/g] (if fsdp, then dp-level reconstruct wk)
    # smem fused reshape k + transpose k : reshape to [bs/dp, seq/sp, num_head/tp/g, head_dim], transpose to [bs/dp, num_head/tp/g, head_dim, seq/sp]
    # rope(k) -> [bs/dp, num_head/tp/g, head_dim, seq/sp]
    # all-gather to k_gathered [bs/dp, num_head/tp/g, head_dim, seq/atten.cp]

    # stage 3
    # v = x@wv -> [bs/dp, seq/sp, hidden/tp/g] (if fsdp, then dp-level reconstruct wv)
    # smem fused reshape v + transpose v : reshape to [bs/dp, seq/sp, num_head/tp/g, head_dim], transpose to [bs/dp, num_head/tp/g, seq/sp, head_dim]
    # all-gather to v_gathered [bs/dp, num_head/tp/g, seq/atten.cp, head_dim]

    # now we fuse stage 1~3 into one big kernel

    # stage 4
    # flash attention (though need cp-level all-reduce + scatter -> reduce-scatter)
    # q_gathered [bs/dp, num_head/tp, seq/atten.sp, head_dim]
    # k_gathered [bs/dp, num_head/tp/g, head_dim, seq/atten.cp]
    # s = q @ k -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # # # # s = softmax(s), local softmax
    # s_max(s,last_dim) -> [bs/dp, num_head/tp, seq/atten.sp, 1]
    # s_exp = exp(s - s_max) -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # s_exp_sum(last_dim) -> [bs/dp, num_head/tp, seq/atten.sp, 1]
    # scaled_s = s_exp / s_exp_sum -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # s *= scaled_s -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # p = s @ v_gathered -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp] @ [bs/dp, num_head/tp/g, seq/atten.cp, head_dim] -> [bs/dp, num_head/tp, seq/atten.sp, head_dim]
    # atten.cp-level all-reduce + scatter p, which means reduce-scatter -> [bs/dp, num_head/tp, seq/(atten.sp*atten.cp), head_dim]

    # stage 5
    # smem fused
    # then a reconstruct, a element-wise op, new [shard_bs, shard_num_head, shard_seq_q, shard_seq_kv] / or [shard_bs, shard_num_head, shard_seq, seq] do a [shard_bs, shard_num_head, shard_seq, 1x2] (rescale： sum and max)
    # transpose p + reshape p  : 
    # [bs/dp, num_head/tp, seq/(atten.sp*atten.cp), head_dim] == [bs/dp, num_head/tp, seq/sp, head_dim]
    # transpose to [bs/dp, seq/sp, num_head/tp, head_dim]
    # reshape to [bs/dp, seq/sp, hidden/tp]    
    
    # stage 6 
    # o = p@wo -> [bs/dp, seq/sp, hidden/tp] @ [tp/hidden, hidden] -> [bs/dp, seq/sp, hidden](if fsdp, then dp-level reconstruct wo)
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output [bs/dp, seq/sp, hidden]

    # let's model it one by one 

    # ------------------------------------- stage 1 2 3 to one big kernel (smem fused) -------------------------------------

    # q = x@wq -> [bs/dp, seq/sp, hidden/tp] (if fsdp, then dp-level reconstruct wq)
    # smem fused reshape q + transpose q : reshape to [bs/dp, seq/sp, num_head/tp, head_dim], transpose to [bs/dp, num_head/tp, seq/sp, head_dim]
    # rope(q) -> [bs/dp, num_head/tp, seq/sp, head_dim]
    # all-gather to q_gathered [bs/dp, num_head/tp, seq/atten.sp, head_dim]  
    # ------------------------------------- stage 2 -------------------------------------
    # k = x@wk -> [bs/dp, seq/sp, hidden/tp/g] (if fsdp, then dp-level reconstruct wk)
    # smem fused reshape k + transpose k : reshape to [bs/dp, seq/sp, num_head/tp/g, head_dim], transpose to [bs/dp, num_head/tp/g, head_dim, seq/sp]
    # rope(k) -> [bs/dp, num_head/tp/g, head_dim, seq/sp]
    # all-gather to k_gathered [bs/dp, num_head/tp/g, head_dim, seq/atten.cp]
    # ------------------------------------- stage 3 -------------------------------------
    # v = x@wv -> [bs/dp, seq/sp, hidden/tp/g] (if fsdp, then dp-level reconstruct wv)
    # smem fused reshape v + transpose v : reshape to [bs/dp, seq/sp, num_head/tp/g, head_dim], transpose to [bs/dp, num_head/tp/g, seq/sp, head_dim]
    # all-gather to v_gathered [bs/dp, num_head/tp/g, seq/atten.cp, head_dim]
    time_stage123 = gqa_decode_coarse_stage1(bs=bs, seq=seq, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)


    # ------------------------------------- stage 4 -------------------------------------
    # flash attention (though need cp-level all-reduce + scatter -> reduce-scatter)
    # q_gathered [bs/dp, num_head/tp, seq/atten.sp, head_dim]
    # k_gathered [bs/dp, num_head/tp/g, head_dim, seq/atten.cp]
    # s = q @ k -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # # # # s = softmax(s), local softmax
    # s_max(s,last_dim) -> [bs/dp, num_head/tp, seq/atten.sp, 1]
    # s_exp = exp(s - s_max) -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # s_exp_sum(last_dim) -> [bs/dp, num_head/tp, seq/atten.sp, 1]
    # scaled_s = s_exp / s_exp_sum -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # s *= scaled_s -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
    # p = s @ v_gathered -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp] @ [bs/dp, num_head/tp/g, seq/atten.cp, head_dim] -> [bs/dp, num_head/tp, seq/atten.sp, head_dim]
    # atten.cp-level all-reduce + scatter p, which means reduce-scatter -> [bs/dp, num_head/tp, seq/(atten.sp*atten.cp), head_dim]

    time_stage4 = gqa_decode_coarse_stage4(bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    
    # ------------------------------------- stage 5 -------------------------------------
    # stage 5
    # smem fused
    # then a reconstruct, a element-wise op, new [shard_bs, shard_num_head, shard_seq_q, shard_seq_kv] / or [shard_bs, shard_num_head, shard_seq, seq] do a [shard_bs, shard_num_head, shard_seq, 1x2] (rescale： sum and max)
    # transpose p + reshape p  : 
    # [bs/dp, num_head/tp, seq/(atten.sp*atten.cp), head_dim] == [bs/dp, num_head/tp, seq/sp, head_dim]
    # transpose to [bs/dp, seq/sp, num_head/tp, head_dim]
    # reshape to [bs/dp, seq/sp, hidden/tp]    
    time_stage56 = gqa_decode_coarse_stage5(bs=bs, seq=seq, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # # ------------------------------------- stage 6 -------------------------------------
    # # o = p@wo -> [bs/dp, seq/sp, hidden/tp] @ [tp/hidden, hidden] -> [bs/dp, seq/sp, hidden](if fsdp, then dp-level reconstruct wo)
    # # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # # output [bs/dp, seq/sp, hidden]    
    # time_stage6 = gqa_decode_coarse_stage6(bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # log.info("gqa decode coarse time: stage1: %s, stage2: %s, stage3: %s, stage4: %s, stage5: %s, stage6: %s", time_stage1, time_stage2, time_stage3, time_stage4, time_stage5, time_stage6)
    log.info("gqa decode coarse time: stage123: %s, stage4: %s, stage56: %s", time_stage123, time_stage4, time_stage56)

    # overall_time = time_stage1 + time_stage2 + time_stage3 + time_stage4 + time_stage5 + time_stage6
    # log.info("gqa decode coarse overall time: %s", overall_time)
    # return overall_time

    overall_time = time_stage123 + time_stage4 + time_stage56
    log.info("gqa decode coarse overall time: %s", overall_time)
    return overall_time


# def gqa_decode_kv_list_coarse(bs:int, seq:int, cached_kv_list:list[int], hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme,
#     atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

#     energy_record = EnergyRecord(parallel.world_size())

#     group_size = math.ceil (num_head/num_kv_head)
#     wq_hidden = num_head * head_dim

#     in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

#     shard_bs = math.ceil(bs / parallel.dp)
#     shard_seq = math.ceil(seq / parallel.sp)
#     shard_hidden=math.ceil(wq_hidden / parallel.tp)
#     shard_head = math.ceil(num_head / parallel.tp)
#     shard_kv_head = math.ceil(num_kv_head / parallel.tp)

#     shard_seq_q = math.ceil(seq / atten_parallel.sp)
    
#     gqa_decode_op_stats_list = []

#     gqa_decode_op_stats = OpPerfStats(op_name="gqa_decode", dump_perf_log=granularity.dump_perf_log)

#     # ------------------------------------- stage 1 2 3 to one big kernel (smem fused) -------------------------------------

#     # q = x@wq -> [bs/dp, seq/sp, hidden/tp] (if fsdp, then dp-level reconstruct wq)
#     # smem fused reshape q + transpose q : reshape to [bs/dp, seq/sp, num_head/tp, head_dim], transpose to [bs/dp, num_head/tp, seq/sp, head_dim]
#     # rope(q) -> [bs/dp, num_head/tp, seq/sp, head_dim]
#     # all-gather to q_gathered [bs/dp, num_head/tp, seq/atten.sp, head_dim]  
#     # ------------------------------------- stage 2 -------------------------------------
#     # k = x@wk -> [bs/dp, seq/sp, hidden/tp/g] (if fsdp, then dp-level reconstruct wk)
#     # smem fused reshape k + transpose k : reshape to [bs/dp, seq/sp, num_head/tp/g, head_dim], transpose to [bs/dp, num_head/tp/g, head_dim, seq/sp]
#     # rope(k) -> [bs/dp, num_head/tp/g, head_dim, seq/sp]
#     # all-gather to k_gathered [bs/dp, num_head/tp/g, head_dim, seq/atten.cp]
#     # ------------------------------------- stage 3 -------------------------------------
#     # v = x@wv -> [bs/dp, seq/sp, hidden/tp/g] (if fsdp, then dp-level reconstruct wv)
#     # smem fused reshape v + transpose v : reshape to [bs/dp, seq/sp, num_head/tp/g, head_dim], transpose to [bs/dp, num_head/tp/g, seq/sp, head_dim]
#     # all-gather to v_gathered [bs/dp, num_head/tp/g, seq/atten.cp, head_dim]
#     time_stage123 = gqa_decode_coarse_stage1(bs=bs, seq=seq, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)


#     # ------------------------------------- stage 4 -------------------------------------
#     # flash attention (though need cp-level all-reduce + scatter -> reduce-scatter)
#     # q_gathered [bs/dp, num_head/tp, seq/atten.sp, head_dim]
#     # k_gathered [bs/dp, num_head/tp/g, head_dim, seq/atten.cp]
#     # s = q @ k -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
#     # # # # s = softmax(s), local softmax
#     # s_max(s,last_dim) -> [bs/dp, num_head/tp, seq/atten.sp, 1]
#     # s_exp = exp(s - s_max) -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
#     # s_exp_sum(last_dim) -> [bs/dp, num_head/tp, seq/atten.sp, 1]
#     # scaled_s = s_exp / s_exp_sum -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
#     # s *= scaled_s -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp]
#     # p = s @ v_gathered -> [bs/dp, num_head/tp, seq/atten.sp, seq/atten.cp] @ [bs/dp, num_head/tp/g, seq/atten.cp, head_dim] -> [bs/dp, num_head/tp, seq/atten.sp, head_dim]
#     # atten.cp-level all-reduce + scatter p, which means reduce-scatter -> [bs/dp, num_head/tp, seq/(atten.sp*atten.cp), head_dim]
#     time_stage4_list = []
#     for cached_kv in cached_kv_list:
#         shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)

#         time_stage4 = gqa_decode_coarse_stage4(bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    
#         time_stage4_list.append(time_stage4)
#     # ------------------------------------- stage 5 -------------------------------------
#     # stage 5
#     # smem fused
#     # then a reconstruct, a element-wise op, new [shard_bs, shard_num_head, shard_seq_q, shard_seq_kv] / or [shard_bs, shard_num_head, shard_seq, seq] do a [shard_bs, shard_num_head, shard_seq, 1x2] (rescale： sum and max)
#     # transpose p + reshape p  : 
#     # [bs/dp, num_head/tp, seq/(atten.sp*atten.cp), head_dim] == [bs/dp, num_head/tp, seq/sp, head_dim]
#     # transpose to [bs/dp, seq/sp, num_head/tp, head_dim]
#     # reshape to [bs/dp, seq/sp, hidden/tp]    
#     time_stage56 = gqa_decode_coarse_stage5(bs=bs, seq=seq, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

#     # # ------------------------------------- stage 6 -------------------------------------
#     # # o = p@wo -> [bs/dp, seq/sp, hidden/tp] @ [tp/hidden, hidden] -> [bs/dp, seq/sp, hidden](if fsdp, then dp-level reconstruct wo)
#     # # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
#     # # output [bs/dp, seq/sp, hidden]    
#     # time_stage6 = gqa_decode_coarse_stage6(bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

#     # log.info("gqa decode coarse time: stage1: %s, stage2: %s, stage3: %s, stage4: %s, stage5: %s, stage6: %s", time_stage1, time_stage2, time_stage3, time_stage4, time_stage5, time_stage6)
#     overall_time_list = []

#     for i in range(len(cached_kv_list)):
#         time_stage4 = time_stage4_list[i]
#         overall_time = time_stage123 + time_stage4_list[i] + time_stage56
#         log.info("kv length: %s, over all time: %s", cached_kv_list[i], overall_time)
#         log.info("stage123: %s, stage4: %s, stage56: %s", time_stage123, time_stage4, time_stage56)
#         overall_time_list.append(overall_time)

#     return overall_time_list

def gqa_decode_kv_list_coarse(bs:int, seq:int, cached_kv_list:list[int], hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme,
    atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy) -> list[tuple[float, OpPerfStats]]:

    # Stage 123 and 56 are kv-independent: run once and capture hete/traffic into
    # a base stats object, which is then copied into each per-kv stats.
    if granularity.dump_perf_log == True:
        _base_stats = OpPerfStats(op_name="_base")
    else:
        _base_stats = None
        
    time_stage123 = gqa_decode_coarse_stage1(bs=bs, seq=seq, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    time_stage56 = gqa_decode_coarse_stage5(bs=bs, seq=seq, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)

    # result_list: list[tuple[float, OpPerfStats]] = []
    overall_time_list: list[float] = []
    if granularity.dump_perf_log == True:
        stats_list: list[OpPerfStats] = []
    else:
        stats_list = None
    

    for cached_kv in cached_kv_list:
        # Seed each per-kv stats with the kv-independent hete/traffic from stage123+56
        if granularity.dump_perf_log == True:
            stats = OpPerfStats(op_name=f"gqa_decode_kv{cached_kv}", dump_perf_log=granularity.dump_perf_log)
            stats.absorb(_base_stats)
        else:
            stats = None

        # Stage 4 varies per cached_kv; appends its comm_time into stats
        time_stage4 = gqa_decode_coarse_stage4(bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

        overall_time = time_stage123 + time_stage4 + time_stage56
        log.info("kv length: %s, over all time: %s", cached_kv, overall_time)
        log.info("stage123: %s, stage4: %s, stage56: %s", time_stage123, time_stage4, time_stage56)

        if granularity.dump_perf_log == True:
            stats.finalize(total_time_s=overall_time, h=noc_hierarchy, arch=single_chip)
            stats_list.append(stats)
        overall_time_list.append(overall_time)

    return overall_time_list, stats_list
