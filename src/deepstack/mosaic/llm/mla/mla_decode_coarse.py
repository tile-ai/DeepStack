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
from mosaic.op_dtype.mla_fa_decode_wrapper import mla_fa_decode_wrapper
import cProfile
from mosaic.llm_arch import MLA_Arch, GQA_Arch, MoE_Arch, Dense_FFN_Arch, LLM_Arch
from tilesight.arch import *
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
import logging
log = logging.getLogger(__name__) 
from mosaic.cost.energy_record import EnergyRecord
from mosaic.collectives import all_reduce_wrapper
from mosaic.utils import get_comp_comm_e2e_time

from mosaic.llm.rope.ropeq_decoding_coarse_raw import ropeq_decoding_coarse_raw
# from mosaic.llm.rope.ropek_coarse import ropek_coarse
from mosaic.llm.rope.ropek_decoding_coarse_raw import ropek_decoding_coarse_raw
from mosaic.collectives import all_gather_sp_fission, reduce_scatter_wrapper,reduce_partial_brodcast_wrapper
from mosaic.cost.op_perf_stats import OpPerfStats

def mla_decode_coarse_stage1(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    # 1.1 q_a = x@Wq_a -> [bs/dp, seq/sp, q_down_hidden]
    # 1.2 kv_down = x@Wkv_down -> [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]
    # 1.3 rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]

    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    # num_kv_head = mla_arch.num_kv_head
    num_kv_head = 1
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    # shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)

    # 1.1 q_a = x@Wq_a -> [bs/dp, seq/sp, q_down_hidden]
    # 1.2 kv_down = x@Wkv_down -> [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]
    # 1.3 rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]

    smem_fusion_list=[] 

    # 1.1 first we do q_a

    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_q_a, ext_max_q_a, noc_overall_time_q_a= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = hidden * q_down_hidden * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_q_a, ext_max_q_a, noc_overall_time_q_a, noc_traffic_q_a= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_q_a, hop_time_s=hop_time_q_a, link_time_s=ext_max_q_a)

        log.info("mla decode coarse stage 1 q_a fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_q_a, ext_max_q_a, noc_overall_time_q_a)
    
    gemm_q_a_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    hete_post_data_gemm_q_a, gemm_smem_fusion_post_data_q_a, gemm_tiling_config_q_a=gemm_wrapper(M=shard_bs*shard_seq, N=q_down_hidden, K=hidden, gemm_bytes=gemm_q_a_bytes, granularity=granularity, single_chip=single_chip)
    log.info("M,N,K,tiling config: %s, %s, %s, %s", shard_bs*shard_seq, q_down_hidden, hidden, gemm_tiling_config_q_a)
    grids = [shard_bs * shard_seq / gemm_tiling_config_q_a[0], q_down_hidden / gemm_tiling_config_q_a[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm_q_a], single_chip))
    

    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time_q_a=max(gemm_smem_fusion_post_data_q_a[0],noc_overall_time_q_a)-gemm_smem_fusion_post_data_q_a[0]
        log.info("mla decode coarse stage 1 fsdp additional time: %s s", additional_time_q_a)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time_q_a = noc_overall_time_q_a

    # 1.2 then we do kv_down
    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_kv_down, ext_max_kv_down, noc_overall_time_kv_down= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = hidden * (kv_rope_head_dim+kv_nope_head_dim) * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_kv_down, ext_max_kv_down, noc_overall_time_kv_down, noc_traffic_kv_down= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_kv_down, hop_time_s=hop_time_kv_down, link_time_s=ext_max_kv_down)
        log.info("mla decode coarse stage 1 kv_down fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_kv_down, ext_max_kv_down, noc_overall_time_kv_down)
    
    gemm_kv_down_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    hete_post_data_gemm_kv_down, gemm_smem_fusion_post_data_kv_down, gemm_tiling_config_kv_down=gemm_wrapper(M=shard_bs*shard_seq, N=(kv_rope_head_dim+kv_nope_head_dim), K=hidden, gemm_bytes=gemm_kv_down_bytes, granularity=granularity, single_chip=single_chip)
    log.info("M,N,K,tiling config: %s, %s, %s, %s", shard_bs*shard_seq, (kv_rope_head_dim+kv_nope_head_dim), hidden, gemm_tiling_config_kv_down)

    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time_kv_down=max(gemm_smem_fusion_post_data_kv_down[0],noc_overall_time_kv_down)-gemm_smem_fusion_post_data_kv_down[0]
        log.info("mla decode coarse stage 1 fsdp additional time: %s s", additional_time_kv_down)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time_kv_down = noc_overall_time_kv_down

    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm_kv_down], single_chip))

    # 1.3 then we do rope(kv_rope)
    rope_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
        input2=Tensor_Loc(torch.float16, 'ddr'),
        output=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
    )

    # ropek_time = ropek_decoding_coarse_raw(bs=bs, head=num_kv_head, seq=seq, head_dim=head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    rope_smem_fusion_list, rope_hop_latency, rope_ext_max = ropek_decoding_coarse_raw(bs=bs, head=num_kv_head, seq=seq, head_dim=kv_rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    smem_fusion_list.extend(rope_smem_fusion_list)    

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    single_chip_time = smem_fusion_post_data[0] + additional_time_q_a + additional_time_kv_down

    overall_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=rope_hop_latency, network_link_time=rope_ext_max, waves=np.prod(grids)/single_chip.sm_count, overlap=granularity.get_comp_comm_overlap())

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_stage1 = overall_time
    return time_stage1


def mla_decode_coarse_stage2(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    # 2.1 q_b = q_a@Wq_b -> [bs/dp, seq/sp, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # 2.2 rope(q_rope) -> [bs/dp, seq/sp, num_head/tp*q_rope_head_dim]
    # q_a [bs/dp, seq/sp, q_down_hidden]
    # Wq_b [q_down_hidden, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # output: [bs/dp, seq/sp, num_head/tp, (q_nope_head_dim+q_rope_head_dim)]
    # rope_q [bs/dp, seq/sp, num_head/tp, q_rope_head_dim]
    
    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    # num_kv_head = mla_arch.num_kv_head
    num_kv_head = 1
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    # shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)

    # 2.1 q_b = q_a@Wq_b -> [bs/dp, seq/sp, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # 2.2 rope(q_rope) -> [bs/dp, seq/sp, num_head/tp*q_rope_head_dim]

    smem_fusion_list=[] 

    # 2.1 first we do q_b
    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_q_b, ext_max_q_b, noc_overall_time_q_b= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = q_down_hidden * shard_num_head * (q_nope_head_dim+q_rope_head_dim) * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_q_b, ext_max_q_b, noc_overall_time_q_b, noc_traffic_q_b= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_q_b, hop_time_s=hop_time_q_b, link_time_s=ext_max_q_b)
        log.info("mla decode coarse stage 2 q_b fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_q_b, ext_max_q_b, noc_overall_time_q_b)
    
    gemm_q_b_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    hete_post_data_gemm_q_b, gemm_smem_fusion_post_data_q_b, gemm_tiling_config_q_b=gemm_wrapper(M=shard_bs*shard_seq, N=shard_num_head * (q_nope_head_dim+q_rope_head_dim) , K=q_down_hidden, gemm_bytes=gemm_q_b_bytes, granularity=granularity, single_chip=single_chip)
    log.info("M,N,K,tiling config: %s, %s, %s, %s", shard_bs*shard_seq, shard_num_head * (q_nope_head_dim+q_rope_head_dim), q_down_hidden, gemm_tiling_config_q_b)
    grids = [shard_bs * shard_seq / gemm_tiling_config_q_b[0], shard_num_head * (q_nope_head_dim+q_rope_head_dim) / gemm_tiling_config_q_b[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm_q_b], single_chip))
    

    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time_q_b=max(gemm_smem_fusion_post_data_q_b[0],noc_overall_time_q_b)-gemm_smem_fusion_post_data_q_b[0]
        log.info("mla decode coarse stage 2 fsdp additional time: %s s", additional_time_q_b)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time_q_b = noc_overall_time_q_b

    
    # 2.2 then we do rope(q_rope)
    ropeq_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
        input2=Tensor_Loc(torch.float16, 'ddr'),
        output=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
    )

    ropeq_smem_fusion_list, ropeq_hop_latency, ropeq_ext_max = ropeq_decoding_coarse_raw(bs=bs, head=num_head, seq=seq, head_dim=q_rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=ropeq_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    smem_fusion_list.extend(ropeq_smem_fusion_list)   


    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    single_chip_time = smem_fusion_post_data[0] + additional_time_q_b

    overall_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=ropeq_hop_latency, network_link_time=ropeq_ext_max, waves=np.prod(grids)/single_chip.sm_count, overlap=granularity.get_comp_comm_overlap())
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_stage2 = overall_time
    return time_stage2

def mla_decode_coarse_stage3(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    # q_up_a_trans = q_nope@Wq_up_a_trans -> [bs/dp, seq/sp, num_head/tp,(q_nope_head_dim)] @ [num_head/tp, head_dim, kv_nope_head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,kv_nope_head_dim]
    # num_head/tp is batch dimension.
    # q_nope [bs/dp, seq/sp, num_head/tp, q_nope_head_dim]
    # Wq_up_a_trans [num_head/tp, head_dim, kv_nope_head_dim]
    # output: [bs/dp, seq/sp, num_head/tp, kv_nope_head_dim]
    
    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    # num_kv_head = mla_arch.num_kv_head
    num_kv_head = 1
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    # shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)

    # q_up_a_trans = q_nope@Wq_up_a_trans -> [bs/dp, seq/sp, num_head/tp,(q_nope_head_dim)] @ [num_head/tp, head_dim, kv_nope_head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,kv_nope_head_dim]
    # num_head/tp is batch dimension.
    # q_nope [bs/dp, seq/sp, num_head/tp, q_nope_head_dim]
    # Wq_up_a_trans [num_head/tp, head_dim, kv_nope_head_dim]
    # output: [bs/dp, seq/sp, num_head/tp, kv_nope_head_dim]

    smem_fusion_list=[] 

    # 2.1 first we do q_b
    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_q_up_a_trans, ext_max_q_up_a_trans, noc_overall_time_q_up_a_trans= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = shard_num_head * q_nope_head_dim * kv_nope_head_dim * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_q_up_a_trans, ext_max_q_up_a_trans, noc_overall_time_q_up_a_trans, noc_traffic_q_up_a_trans= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_q_up_a_trans, hop_time_s=hop_time_q_up_a_trans, link_time_s=ext_max_q_up_a_trans)
        log.info("mla decode coarse stage 3 q_up_a_trans fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_q_up_a_trans, ext_max_q_up_a_trans, noc_overall_time_q_up_a_trans)
    
    gemm_q_up_a_trans_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    hete_post_data_gemm_q_up_a_trans, gemm_smem_fusion_post_data_q_up_a_trans, gemm_tiling_config_q_up_a_trans=gemm_wrapper(M=shard_bs*shard_seq, N=kv_nope_head_dim , K=q_nope_head_dim, gemm_bytes=gemm_q_up_a_trans_bytes, granularity=granularity, single_chip=single_chip, batch=shard_num_head)
    log.info("M,N,K,batch,tiling config: %s, %s, %s, %s, %s", shard_bs*shard_seq, kv_nope_head_dim, q_nope_head_dim, shard_num_head, gemm_tiling_config_q_up_a_trans)
    grids = [shard_bs * shard_seq / gemm_tiling_config_q_up_a_trans[0], kv_nope_head_dim / gemm_tiling_config_q_up_a_trans[1], shard_num_head]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm_q_up_a_trans], single_chip))
    

    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time_q_up_a_trans=max(gemm_smem_fusion_post_data_q_up_a_trans[0],noc_overall_time_q_up_a_trans)-gemm_smem_fusion_post_data_q_up_a_trans[0]
        log.info("mla decode coarse stage 3 fsdp additional time: %s s", additional_time_q_up_a_trans)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time_q_up_a_trans = noc_overall_time_q_up_a_trans

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    single_chip_time = smem_fusion_post_data[0] + additional_time_q_up_a_trans

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_stage3 = single_chip_time
    return time_stage3

def mla_decode_coarse_stage4_1(bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    # 4.1 mqa to atten.cp-level all-reduce + scatter p
    # mla:
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


    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    # num_kv_head = mla_arch.num_kv_head
    num_kv_head = 1
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)

    grids, single_chip_time = mla_fa_decode_wrapper(bs=bs, seq=seq, cached_kv=cached_kv, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=kv_nope_head_dim, qk_rope_head_dim=kv_rope_head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    
    waves= np.prod(grids)/single_chip.sm_count

    if (atten_parallel.cp == 1):
        log.info("mla decode coarse stage 4 overall time: %s s", single_chip_time)
        return single_chip_time
    elif (atten_parallel.cp > 1 and atten_parallel.cp <= parallel.sp):
        # reduce-scatter along atten.cp
        # each device holding: [shard_bs, shard_num_head, shard_seq_q, head_dim]
        cp_reduce_latency, cp_reduce_ext_max, cp_reduce_traffic = reduce_scatter_wrapper(all_reduce_op_bytes=atten_bytes, parallel=atten_parallel, 
        # noc_hierarchy=noc_hierarchy, granularity=granularity, dim_to_process="cp", bytes=shard_bs*shard_num_head*shard_seq_q*head_dim*in_bytes)
        noc_hierarchy=noc_hierarchy, granularity=granularity, dim_to_process="cp", bytes=shard_bs*shard_num_head*shard_seq_q*kv_nope_head_dim*in_bytes)

        log.info("cp_reduce_scatter_latency: %s, cp_reduce_ext_max: %s", cp_reduce_latency, cp_reduce_ext_max)

        overall_time =get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=cp_reduce_latency, network_link_time=cp_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
        log.info("mla decode coarse stage 4 overall time: %s s", overall_time)
        if stats is not None:
            stats.append_traffic(cp_reduce_traffic, hop_time_s=cp_reduce_latency, link_time_s=cp_reduce_ext_max)
        return overall_time
    elif (atten_parallel.cp > parallel.sp):
        # cp_all_reduce_latency, cp_all_reduce_ext_max, cp_all_reduce_traffic = all_reduce_wrapper(all_reduce_op_bytes=atten_bytes, parallel=atten_parallel,
        # noc_hierarchy=noc_hierarchy, granularity=granularity, dim_to_process="cp", bytes=shard_bs*shard_num_head*shard_seq_q*head_dim*in_bytes)
        cp_all_reduce_latency, cp_all_reduce_ext_max, cp_all_reduce_traffic = all_reduce_wrapper(all_reduce_op_bytes=atten_bytes, parallel=atten_parallel,
            noc_hierarchy=noc_hierarchy, granularity=granularity, dim_to_process="cp", bytes=shard_bs*shard_num_head*shard_seq_q*kv_nope_head_dim*in_bytes)

        log.info("cp_all_reduce_latency: %s, cp_all_reduce_ext_max: %s", cp_all_reduce_latency, cp_all_reduce_ext_max)

        overall_time =get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=cp_all_reduce_latency, network_link_time=cp_all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
        log.info("mla decode coarse stage 4 overall time: %s s", overall_time)
        if stats is not None:
            stats.append_traffic(cp_all_reduce_traffic, hop_time_s=cp_all_reduce_latency, link_time_s=cp_all_reduce_ext_max)
        return overall_time
    else:
        raise NotImplementedError("not desinged for this case", parallel, atten_parallel)

def mla_decode_coarse_stage4_2(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    # 4.2 transpose + reshape to [bs/dp, seq/sp, num_head/tp, nope_head_dim]
    pass

    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    # num_kv_head = mla_arch.num_kv_head
    num_kv_head = 1
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    # shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)

    # ------------------------------------- stage 5 -------------------------------------
    # stage 5
    # smem fused
    # then a reconstruct, a element-wise op, new [shard_bs, shard_num_head, shard_seq_q, kv_nope_head_dim] / or [shard_bs, shard_num_head, shard_seq, kv_nope_head_dim] do a [shard_bs, shard_num_head, shard_seq, 1x2] (rescale： sum and max)
    # transpose p + reshape p  : 
    # [bs/dp, num_head/tp, seq/(atten.sp*atten.cp), kv_nope_head_dim] == [bs/dp, num_head/tp, seq/sp, kv_nope_head_dim]
    # transpose to [bs/dp, seq/sp, num_head/tp, kv_nope_head_dim]
    # reshape to [bs/dp, seq/sp, hidden/tp]  
    smem_fusion_list=[]
    rescale_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr',[shard_bs, shard_num_head, shard_seq, kv_nope_head_dim]),
        input2=Tensor_Loc(atten_bytes.input1.dtype, 'ddr',[shard_bs, shard_num_head, shard_seq, 1]),
        output=Tensor_Loc(atten_bytes.output.dtype, 'smem',[shard_bs, shard_num_head, shard_seq, kv_nope_head_dim]),
    )
    
    hete_post_data_rescale, tb_shape_rescale=element_wrapper(element_op_bytes=rescale_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=None)

    smem_fusion_list.append(hete_reg_fusion([hete_post_data_rescale], single_chip))

    transpose_reshape_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem',[shard_bs, shard_seq, shard_num_head * kv_nope_head_dim]),
        input2=None,
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr',[shard_bs, shard_seq, shard_num_head * kv_nope_head_dim]),
    )
    
    hete_post_data_transpose_reshape, tb_shape_transpose_reshape=element_wrapper(element_op_bytes=transpose_reshape_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=None)

    smem_fusion_list.append(hete_reg_fusion([hete_post_data_transpose_reshape], single_chip))

    # grids = [shard_bs/tb_shape_transpose_reshape[0], shard_seq/tb_shape_transpose_reshape[1], shard_hidden/tb_shape_transpose_reshape[2]]
    grids = [shard_bs/tb_shape_transpose_reshape[0], shard_seq/tb_shape_transpose_reshape[1], (shard_num_head * kv_nope_head_dim)/tb_shape_transpose_reshape[2]]

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    # log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_stage4_2 = smem_fusion_post_data[0]
    # log.info("mla decode coarse stage 4_2 overall time: %s s", time_stage4_2)
    return time_stage4_2

def mla_decode_coarse_stage5(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    #  o_up_b_trans = o_nope@Wo_up_b_trans -> [bs/dp, seq/sp, num_head/tp,(kv_nope_head_dim)] @ [num_head/tp, kv_nope_head_dim, head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,head_dim]
    # num_head/tp is batch dimension.
    # transform to [bs/dp, seq/sp, wq_hidden/tp]

    # x [bs/dp, seq/sp, shard_num_head, kv_nope_head_dim]
    # w_kv_up_b_trans [num_head/tp, kv_nope_head_dim, head_dim]
    # kv_up_b_trans [bs/dp, seq/sp, shard_num_head, head_dim]

    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    # num_kv_head = mla_arch.num_kv_head
    num_kv_head = 1
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    # shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)
    smem_fusion_list=[] 
    
    #  o_up_b_trans = o_nope@Wo_up_b_trans -> [bs/dp, seq/sp, num_head/tp,(kv_nope_head_dim)] @ [num_head/tp, kv_nope_head_dim, head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,head_dim]
    # num_head/tp is batch dimension.
    # transform to [bs/dp, seq/sp, wq_hidden/tp]

    # x [bs/dp, seq/sp, shard_num_head, kv_nope_head_dim]
    # w_kv_up_b_trans [num_head/tp, kv_nope_head_dim, head_dim]
    # kv_up_b_trans [bs/dp, seq/sp, shard_num_head, head_dim]
    # 5 first we do o_up_b_trans
    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_q_up_b_trans, ext_max_q_up_b_trans, noc_overall_time_q_up_b_trans= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = shard_num_head * kv_nope_head_dim * head_dim * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_q_up_b_trans, ext_max_q_up_b_trans, noc_overall_time_q_up_b_trans, noc_traffic_q_up_b_trans= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_q_up_b_trans, hop_time_s=hop_time_q_up_b_trans, link_time_s=ext_max_q_up_b_trans)
        log.info("mla decode coarse stage 3 q_up_b_trans fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_q_up_b_trans, ext_max_q_up_b_trans, noc_overall_time_q_up_b_trans)
    
    gemm_q_up_a_trans_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    hete_post_data_gemm_q_up_b_trans, gemm_smem_fusion_post_data_q_up_b_trans, gemm_tiling_config_q_up_b_trans=gemm_wrapper(M=shard_bs*shard_seq, N= head_dim, K=kv_nope_head_dim, gemm_bytes=gemm_q_up_a_trans_bytes, granularity=granularity, single_chip=single_chip, batch=shard_num_head)
    log.info("M,N,K,batch,tiling config: %s, %s, %s, %s, %s", shard_bs*shard_seq, head_dim, kv_nope_head_dim, shard_num_head, gemm_tiling_config_q_up_b_trans)
    grids = [shard_bs * shard_seq / gemm_tiling_config_q_up_b_trans[0], head_dim / gemm_tiling_config_q_up_b_trans[1], shard_num_head]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm_q_up_b_trans], single_chip))
    

    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time_q_up_b_trans=max(gemm_smem_fusion_post_data_q_up_b_trans[0],noc_overall_time_q_up_b_trans)-gemm_smem_fusion_post_data_q_up_b_trans[0]
        log.info("mla decode coarse stage 3 fsdp additional time: %s s", additional_time_q_up_b_trans)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time_q_up_b_trans = noc_overall_time_q_up_b_trans

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    single_chip_time = smem_fusion_post_data[0] + additional_time_q_up_b_trans

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_stage5 = single_chip_time

    return time_stage5

def mla_decode_coarse_stage6(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # x [bs/dp, seq/sp, wq_hidden/tp]
    # W_o [wq_hidden/tp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # all-reduce level tp

    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    # num_kv_head = mla_arch.num_kv_head
    num_kv_head = 1
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden/ (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    # shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)
    smem_fusion_list=[] 

    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # x [bs/dp, seq/sp, wq_hidden/tp]
    # W_o [wq_hidden/tp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # all-reduce level tp

    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_w_o, ext_max_w_o, noc_overall_time_w_o= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = shard_hidden * hidden * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_w_o, ext_max_w_o, noc_overall_time_w_o, noc_traffic_w_o= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_w_o, hop_time_s=hop_time_w_o, link_time_s=ext_max_w_o)
        log.info("mla decode coarse stage 3 w_o fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_w_o, ext_max_w_o, noc_overall_time_w_o)
    
    gemm_w_o_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )    
    # Thanks Xianqi for spotting the bug here!
    hete_post_data_gemm_w_o, gemm_smem_fusion_post_data_w_o, gemm_tiling_config_w_o=gemm_wrapper(M=shard_bs*shard_seq, N= hidden, K=shard_hidden, gemm_bytes=gemm_w_o_bytes, granularity=granularity, single_chip=single_chip, batch=1)
    log.info("gemm_w_o: M,N,K,batch,tiling config: %s, %s, %s, %s, %s", shard_bs*shard_seq, hidden, shard_hidden, 1, gemm_tiling_config_w_o)
    grids = [shard_bs * shard_seq / gemm_tiling_config_w_o[0], hidden / gemm_tiling_config_w_o[1], 1]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm_w_o], single_chip))

    waves=np.prod(grids)/single_chip.sm_count
    

    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time_w_o=max(gemm_smem_fusion_post_data_w_o[0],noc_overall_time_w_o)-gemm_smem_fusion_post_data_w_o[0]
        log.info("mla decode coarse stage 3 fsdp additional time: %s s", additional_time_w_o)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time_w_o = noc_overall_time_w_o

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    single_chip_time = smem_fusion_post_data[0] + additional_time_w_o

    if (parallel.tp > 1 and next_parallel.tp == parallel.tp):
        # all_reduce_time=all_reduce_wrapper(parallel=parallel, noc_hierarchy=noc_hierarchy, bytes=shard_bs*shard_seq*shard_hidden*out_bytes, dim_to_process="tp")
        # each device holding [shard_bs, shard_seq, hidden], tp-level all-reduce
        all_reduce_latency, all_reduce_ext_max, all_reduce_traffic = all_reduce_wrapper(all_reduce_op_bytes=atten_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy, granularity=granularity, 
            dim_to_process="tp", bytes=shard_bs*shard_seq*hidden*out_bytes)
    elif (parallel.tp > 1 and next_parallel.ep !=1 and (parallel.tp * parallel.sp *parallel.dp == next_parallel.tp * next_parallel.ep * next_parallel.sp * next_parallel.dp)):
        all_reduce_latency, all_reduce_ext_max, all_reduce_traffic = reduce_partial_brodcast_wrapper(all_reduce_op_bytes=atten_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy, granularity=granularity,
            dim_to_process="tp", broadcast_degree=next_parallel.tp, bytes=shard_bs*shard_seq*hidden*out_bytes)

        # tmp fix
        all_reduce_latency_2, all_reduce_ext_max_2, all_reduce_traffic_2 = all_reduce_wrapper(all_reduce_op_bytes=atten_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy, granularity=granularity, 
            dim_to_process="tp", bytes=shard_bs*shard_seq*hidden*out_bytes)

        all_reduce_latency = min(all_reduce_latency, all_reduce_latency_2)
        all_reduce_ext_max = min(all_reduce_ext_max, all_reduce_ext_max_2)

    elif (parallel.tp == 1):
        all_reduce_latency, all_reduce_ext_max, all_reduce_traffic = 0, 0, None
    else:
        all_reduce_latency, all_reduce_ext_max, all_reduce_traffic = 0, 0, None

    if stats is not None and all_reduce_traffic is not None:
        stats.append_traffic(all_reduce_traffic, hop_time_s=all_reduce_latency, link_time_s=all_reduce_ext_max)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)

    # time_stage6 = single_chip_time
    e2e_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=all_reduce_latency, network_link_time=all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())

    time_stage6 = e2e_time
    return time_stage6

def mla_decode_coarse(bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    '''
    Decode: with weight absorption, MQA form. (kv head = 1)
    While for weight, we keep both absorption and no absorption footprint
    (no absorb:w_kv_up, absorb:  q_up_a_trans + o_up_b_trans)
    '''
    assert granularity.get_mode() == "coarse"
    assert parallel.ep==1

    if seq == 1:
        assert parallel.sp == 1

    assert model_arch.mla_arch is not None

    mla_arch = model_arch.mla_arch
    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    # num_kv_head = mla_arch.num_kv_head
    num_kv_head = 1
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    # 其中x [bs/dp, seq/sp, hidden], 
    # Wq_a, [hidden, q_down_hidden], Wq_b [q_down_hidden, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # Wkv_up_a_trans [num_head/tp, head_dim, kv_nope_head_dim]
    # Wkv_down [hidden, (kv_rope_head_dim+kv_nope_head_dim)]
    # Wkv_up_b_trans [num_head/tp, kv_nope_head_dim, head_dim]
    # W_o [num_head*head_dim/tp, hidden]

    # input: [bs/dp, seq/sp, hidden]

    # stage 1:
    # 1.1 q_a = x@Wq_a -> [bs/dp, seq/sp, q_down_hidden]
    # 1.2 kv_down = x@Wkv_down -> [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]
    # 1.3 rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]

    # stage 2:
    # q_b = q_a@Wq_b -> [bs/dp, seq/sp, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # rope(q_rope) -> [bs/dp, seq/sp, num_head/tp*q_rope_head_dim]

    # stage 3:
    # q_up_a_trans = q_nope@Wq_up_a_trans -> [bs/dp, seq/sp, num_head/tp,(q_nope_head_dim)] @ [num_head/tp, head_dim, kv_nope_head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,kv_nope_head_dim]
    # num_head/tp is batch dimension.


    # stage 4:
    # MQA, copy previous code
    # need to manually set num_kv_head = 1
    # flash attention, just copy previous code
    # in mla_decode_coarse, time_stage4 and time_stage5

    # stage 5:
    #  o_up_b_trans = o_nope@Wo_up_b_trans -> [bs/dp, seq/sp, num_head/tp,(kv_nope_head_dim)] @ [num_head/tp, kv_nope_head_dim, head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,head_dim]
    # num_head/tp is batch dimension.
    # transform to [bs/dp, seq/sp, wq_hidden/tp]

    # stage 6:
    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]

    # let's model it one by one 

    # ------------------------------------- stage 1 -------------------------------------
    # 1.1 q_a = x@Wq_a -> [bs/dp, seq/sp, q_down_hidden]
    # 1.2 kv_down = x@Wkv_down -> [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]
    # 1.3 rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]
    time_stage1 = mla_decode_coarse_stage1(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # ------------------------------------- stage 2 -------------------------------------
    # q_b = q_a@Wq_b -> [bs/dp, seq/sp, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # rope(q_rope) -> [bs/dp, seq/sp, num_head/tp*q_rope_head_dim]
    time_stage2 = mla_decode_coarse_stage2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    # ------------------------------------- stage 3 -------------------------------------
    # q_up_a_trans = q_nope@Wq_up_a_trans -> [bs/dp, seq/sp, num_head/tp,(q_nope_head_dim)] @ [num_head/tp, head_dim, kv_nope_head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,kv_nope_head_dim]
    # num_head/tp is batch dimension.
    time_stage3 = mla_decode_coarse_stage3(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    # ------------------------------------- stage 4 -------------------------------------   
    # MQA, copy previous code
    # need to manually set num_kv_head = 1
    # flash attention, just copy previous code
    # in mla_decode_coarse, time_stage4 and time_stage5
    # 4.1 mqa to atten.cp-level all-reduce + scatter p,
    time_stage4_1 = mla_decode_coarse_stage4_1(bs=bs, seq=seq, cached_kv=cached_kv, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    # 4.2 rescale + reshape to [bs/dp, seq/sp, num_head/tp, nope_head_dim]
    time_stage4_2 = mla_decode_coarse_stage4_2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    # ------------------------------------- stage 5 -------------------------------------
    #  o_up_b_trans = o_nope@Wo_up_b_trans -> [bs/dp, seq/sp, num_head/tp,(kv_nope_head_dim)] @ [num_head/tp, kv_nope_head_dim, head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,head_dim]
    # num_head/tp is batch dimension.
    # transform to [bs/dp, seq/sp, wq_hidden/tp]
    time_stage5 = mla_decode_coarse_stage5(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    # stage 6:
    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # x [bs/dp, seq/sp, wq_hidden/tp]
    # W_o [wq_hidden/tp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # all-reduce level tp
    time_stage6 = mla_decode_coarse_stage6(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    log.info("mla decode coarse time: stage 1: %s. Note: no tp for w_q_a and w_kv_down projection which could be potential performance bottleneck", time_stage1)
    log.info("mla decode coarse time: stage 2: %s", time_stage2)
    log.info("mla decode coarse time: stage 3: %s", time_stage3)
    log.info("mla decode coarse time: stage 4_1 (mqa decode): %s", time_stage4_1)
    log.info("mla decode coarse time: stage 4_2 (rescale + reshape): %s", time_stage4_2)
    log.info("mla decode coarse time: stage 5: %s", time_stage5)
    log.info("mla decode coarse time: stage 6: %s", time_stage6)

    return time_stage1 + time_stage2 + time_stage3 + time_stage4_1 + time_stage4_2 + time_stage5 + time_stage6

def mla_decode_kv_list_coarse(bs:int, seq:int, cached_kv_list:list[int], model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    '''
    Decode: with weight absorption, MQA form. (kv head = 1)
    While for weight, we keep both absorption and no absorption footprint
    (no absorb:w_kv_up, absorb:  q_up_a_trans + o_up_b_trans)
    '''
    assert granularity.get_mode() == "coarse"
    assert parallel.ep==1

    if seq == 1:
        assert parallel.sp == 1

    assert model_arch.mla_arch is not None

    mla_arch = model_arch.mla_arch
    assert model_arch.mla_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    # num_kv_head = mla_arch.num_kv_head
    num_kv_head = 1
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    # 其中x [bs/dp, seq/sp, hidden], 
    # Wq_a, [hidden, q_down_hidden], Wq_b [q_down_hidden, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # Wkv_up_a_trans [num_head/tp, head_dim, kv_nope_head_dim]
    # Wkv_down [hidden, (kv_rope_head_dim+kv_nope_head_dim)]
    # Wkv_up_b_trans [num_head/tp, kv_nope_head_dim, head_dim]
    # W_o [num_head*head_dim/tp, hidden]

    # input: [bs/dp, seq/sp, hidden]

    # stage 1:
    # 1.1 q_a = x@Wq_a -> [bs/dp, seq/sp, q_down_hidden]
    # 1.2 kv_down = x@Wkv_down -> [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]
    # 1.3 rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]

    # stage 2:
    # q_b = q_a@Wq_b -> [bs/dp, seq/sp, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # rope(q_rope) -> [bs/dp, seq/sp, num_head/tp*q_rope_head_dim]

    # stage 3:
    # q_up_a_trans = q_nope@Wq_up_a_trans -> [bs/dp, seq/sp, num_head/tp,(q_nope_head_dim)] @ [num_head/tp, head_dim, kv_nope_head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,kv_nope_head_dim]
    # num_head/tp is batch dimension.


    # stage 4:
    # MQA, copy previous code
    # need to manually set num_kv_head = 1
    # flash attention, just copy previous code
    # in mla_decode_coarse, time_stage4 and time_stage5

    # stage 5:
    #  o_up_b_trans = o_nope@Wo_up_b_trans -> [bs/dp, seq/sp, num_head/tp,(kv_nope_head_dim)] @ [num_head/tp, kv_nope_head_dim, head_dim] 
    # -> [bs/dp, seq/sp, num_head/tp,head_dim]
    # num_head/tp is batch dimension.
    # transform to [bs/dp, seq/sp, wq_hidden/tp]

    # stage 6:
    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]

    # let's model it one by one 

    if granularity.dump_perf_log:
        _base_stats = OpPerfStats(op_name="_base")
        # log.info("dump perf log true in mla_decode_kv_list_coarse")
    else:
        _base_stats = None

    # ------------------------------------- stage 1 -------------------------------------
    # 1.1 q_a = x@Wq_a -> [bs/dp, seq/sp, q_down_hidden]
    # 1.2 kv_down = x@Wkv_down -> [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]
    # 1.3 rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]
    time_stage1 = mla_decode_coarse_stage1(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)

    # ------------------------------------- stage 2 -------------------------------------
    # q_b = q_a@Wq_b -> [bs/dp, seq/sp, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # rope(q_rope) -> [bs/dp, seq/sp, num_head/tp*q_rope_head_dim]
    time_stage2 = mla_decode_coarse_stage2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    # ------------------------------------- stage 3 -------------------------------------
    # q_up_a_trans = q_nope@Wq_up_a_trans -> [bs/dp, seq/sp, num_head/tp,(q_nope_head_dim)] @ [num_head/tp, head_dim, kv_nope_head_dim]
    # -> [bs/dp, seq/sp, num_head/tp,kv_nope_head_dim]
    # num_head/tp is batch dimension.
    time_stage3 = mla_decode_coarse_stage3(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    # 4.2 rescale + reshape to [bs/dp, seq/sp, num_head/tp, nope_head_dim]
    time_stage4_2 = mla_decode_coarse_stage4_2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    # ------------------------------------- stage 5 -------------------------------------
    #  o_up_b_trans = o_nope@Wo_up_b_trans -> [bs/dp, seq/sp, num_head/tp,(kv_nope_head_dim)] @ [num_head/tp, kv_nope_head_dim, head_dim]
    # -> [bs/dp, seq/sp, num_head/tp,head_dim]
    # num_head/tp is batch dimension.
    # transform to [bs/dp, seq/sp, wq_hidden/tp]
    time_stage5 = mla_decode_coarse_stage5(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    # stage 6:
    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # x [bs/dp, seq/sp, wq_hidden/tp]
    # W_o [wq_hidden/tp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # all-reduce level tp
    time_stage6 = mla_decode_coarse_stage6(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)

    overall_time_list = []
    stats_list = [] if granularity.dump_perf_log else None
    # ------------------------------------- stage 4 -------------------------------------
    # MQA, copy previous code
    # need to manually set num_kv_head = 1
    # flash attention, just copy previous code
    # in mla_decode_coarse, time_stage4 and time_stage5
    # 4.1 mqa to atten.cp-level all-reduce + scatter p,
    for cached_kv in cached_kv_list:
        stats = OpPerfStats(op_name=f"mla_decode_bs{bs}_seq{seq}_kv{cached_kv}", dump_perf_log=granularity.dump_perf_log) if granularity.dump_perf_log else None
        if stats is not None:
            stats.absorb(_base_stats)
        time_stage4_1 = mla_decode_coarse_stage4_1(bs=bs, seq=seq, cached_kv=cached_kv, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

        overall_time = time_stage1 + time_stage2 + time_stage3 + time_stage4_1 + time_stage4_2 + time_stage5 + time_stage6
        log.info("kv length: %s, over all time: %s", cached_kv, overall_time)
        log.info("mla decode coarse time: stage 1: %s. Note: no tp for w_q_a and w_kv_down projection which could be potential performance bottleneck", time_stage1)
        log.info("mla decode coarse time: stage 2: %s", time_stage2)
        log.info("mla decode coarse time: stage 3: %s", time_stage3)
        log.info("mla decode coarse time: stage 4_1 (mqa decode): %s", time_stage4_1)
        log.info("mla decode coarse time: stage 4_2 (rescale + reshape): %s", time_stage4_2)
        log.info("mla decode coarse time: stage 5: %s", time_stage5)
        log.info("mla decode coarse time: stage 6: %s", time_stage6)
        if granularity.dump_perf_log:
            stats.finalize(total_time_s=overall_time, h=noc_hierarchy, arch=single_chip)
            # log.info("dump perf log finalize true in mla_decode_kv_list_coarse")
            
            stats_list.append(stats)
        overall_time_list.append(overall_time)

    return overall_time_list, stats_list