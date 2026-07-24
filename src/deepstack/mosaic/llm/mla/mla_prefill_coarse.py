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
from mosaic.op_dtype.mla_fa_prefill_wrapper import mla_fa_prefill_wrapper
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

from mosaic.llm.rope.ropeq_coarse import ropeq_coarse
from mosaic.llm.rope.ropek_coarse import ropek_coarse
from mosaic.collectives import all_gather_sp_fission, reduce_scatter_wrapper,reduce_partial_brodcast_wrapper
from mosaic.cost.op_perf_stats import OpPerfStats

def mla_prefill_coarse_stage1(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):

    # stage 1:
    # q_a = x@Wq_a -> [bs/dp, seq/sp, q_down_hidden]
    # x: [bs/dp, seq/sp, hidden]
    # Wq_a: [hidden, q_down_hidden]
    # output: [bs/dp, seq/sp, q_down_hidden]
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)
    

    # 
    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_1, ext_max_1, noc_overall_time_1= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = hidden * q_down_hidden * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_1, ext_max_1, noc_overall_time_1, noc_traffic_1= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=hop_time_1, link_time_s=ext_max_1)
        
    gemm1_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    smem_fusion_list=[]

    # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
    # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
    hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_bs*shard_seq, N=q_down_hidden, K=hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
    grids = [shard_bs * shard_seq / gemm_tiling_config[0], q_down_hidden / gemm_tiling_config[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

    
    
    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
        log.info("mla prefill coarse stage 1 fsdp additional time: %s s", additional_time)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time = noc_overall_time_1
    
    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    
    time_stage1 = smem_fusion_post_data[0]
    return time_stage1

def mla_prefill_coarse_stage2(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    # stage 2:
    # q_b = q_a@Wq_b -> [bs/dp, seq/sp, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # rope(q_rope) -> [bs/dp, seq/sp, num_head/tp*q_rope_head_dim]

    # q_a [bs/dp, seq/sp, q_down_hidden]
    # Wq_b [q_down_hidden, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # output: [bs/dp, seq/sp, num_head/tp, (q_nope_head_dim+q_rope_head_dim)]
    # rope_q [bs/dp, seq/sp, num_head/tp, q_rope_head_dim]
    # nope_q [bs/dp, seq/sp, num_head/tp, q_nope_head_dim]


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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)
    

    # 
    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_1, ext_max_1, noc_overall_time_1= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = q_down_hidden * shard_num_head * (q_nope_head_dim+q_rope_head_dim) * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_1, ext_max_1, noc_overall_time_1, noc_traffic_1= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=hop_time_1, link_time_s=ext_max_1)
        
    gemm1_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    smem_fusion_list=[]

    # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
    # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
    hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_bs*shard_seq, N=shard_num_head * (q_nope_head_dim+q_rope_head_dim), K=q_down_hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
    grids = [shard_bs * shard_seq / gemm_tiling_config[0], shard_num_head * (q_nope_head_dim+q_rope_head_dim) / gemm_tiling_config[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

    
    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
        log.info("mla prefill coarse stage 1 fsdp additional time: %s s", additional_time)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time = noc_overall_time_1
    
    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    
    time_gemm_q_b = smem_fusion_post_data[0]

    rope_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
        input2=Tensor_Loc(torch.float16, 'ddr'),
        output=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
    )

    time_rope_q = ropeq_coarse(bs=bs, head=num_head, seq=seq, head_dim=q_rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # log.info("time rope q:%s, time gemm q_b:%s, additional time:%s", time_rope_q, time_gemm_q_b, additional_time)
    log.info("mla prefill coarse stage 2 gemm q_b:%s, fsdp additional time:%s, time rope q:%s", time_gemm_q_b, additional_time, time_rope_q)

    time_stage2 = time_gemm_q_b + time_rope_q + additional_time
    return time_stage2

def mla_prefill_coarse_stage3(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    pass
    # stage 3:
    # kv_down = x@Wkv_down -> [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]
    # rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]
    # dup (logically) -> [bs/dp, seq/sp, num_kv_head, kv_rope_head_dim]
    # input: [bs/dp, seq/sp, hidden]
    # w_kv_down [hidden, (kv_rope_head_dim+kv_nope_head_dim)]
    # rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]
    # output: [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]

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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_1, ext_max_1, noc_overall_time_1= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = hidden * (kv_rope_head_dim+kv_nope_head_dim) * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_1, ext_max_1, noc_overall_time_1, noc_traffic_1= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=hop_time_1, link_time_s=ext_max_1)
        
    gemm1_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    smem_fusion_list=[]

    # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
    # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
    hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_bs*shard_seq, N=(kv_rope_head_dim+kv_nope_head_dim), K=hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
    grids = [shard_bs * shard_seq / gemm_tiling_config[0], (kv_rope_head_dim+kv_nope_head_dim) / gemm_tiling_config[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

    
    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
        log.info("mla prefill coarse stage 1 fsdp additional time: %s s", additional_time)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time = noc_overall_time_1
    
    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    
    time_gemm_kv_down = smem_fusion_post_data[0]

    rope_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
        input2=Tensor_Loc(torch.float16, 'ddr'),
        output=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
    )

    time_rope_kv = ropeq_coarse(bs=bs, head=1, seq=seq, head_dim=kv_rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # log.info("time rope q:%s, time gemm q_b:%s, additional time:%s", time_rope_q, time_gemm_q_b, additional_time)
    log.info("mla prefill coarse stage 3 gemm kv_down:%s, fsdp additional time:%s, time rope q:%s", time_gemm_kv_down, additional_time, time_rope_kv)

    time_stage3 = time_gemm_kv_down + time_rope_kv + additional_time
    return time_stage3  

def mla_prefill_coarse_stage4(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    pass
    # kv_up = kv_down@Wkv_up -> [bs/dp, seq/sp, kv_nope_up_hidden/tp]
    # writing the split-> part of k and part of v
    # k: [bs/dp, seq/sp, num_kv_head, (kv_rope_head_dim+head_dim)]
    # v: [bs/dp, seq/sp, num_kv_head, head_dim]

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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_1, ext_max_1, noc_overall_time_1= 0, 0, 0
    elif parallel.fsdp==True:
        # reconstruct wq at dp level
        tm = TrafficMatrix(parallel.world_size())
        bytes_each_pair = kv_nope_head_dim * shard_num_kv_head * (head_dim+head_dim) * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_1, ext_max_1, noc_overall_time_1, noc_traffic_1= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=hop_time_1, link_time_s=ext_max_1)
        
    gemm1_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    smem_fusion_list=[]

    # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
    # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
    hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_bs*shard_seq, N=shard_num_kv_head * (head_dim+head_dim), K=kv_nope_head_dim, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
    grids = [shard_bs * shard_seq / gemm_tiling_config[0], shard_num_kv_head * (head_dim+head_dim) / gemm_tiling_config[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

    
    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
        log.info("mla prefill coarse stage 1 fsdp additional time: %s s", additional_time)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time = noc_overall_time_1
    
    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    
    time_gemm_kv_up = smem_fusion_post_data[0]

    
    # log.info("time rope q:%s, time gemm q_b:%s, additional time:%s", time_rope_q, time_gemm_q_b, additional_time)
    log.info("mla prefill coarse stage 4 gemm kv_up:%s, fsdp additional time:%s", time_gemm_kv_up, additional_time)

    time_stage4 = time_gemm_kv_up + additional_time
    return time_stage4

def mla_prefill_coarse_stage5_1(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    pass
    # ------------------------------------- stage 5 -------------------------------------
    # MHA, copy previous code
    # flash attention, just copy previous code
    # in mla_prefill_coarse, time_stage4 and time_stage5

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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_seq_kv = math.ceil(seq / atten_parallel.cp)

    shard_grouped_seq_q = shard_seq_q * math.ceil(shard_num_head / shard_num_kv_head)
    shard_group_size = math.ceil(shard_num_head / shard_num_kv_head)
    
    
    # grids, single_chip_time = mla_fa_prefill_wrapper(bs=bs, seq=seq, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, qk_rope_head_dim=qk_rope_head_dim, parallel=parallel, atten_parallel=atten_parallel, atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    grids, single_chip_time = mla_fa_prefill_wrapper(bs=bs, seq=seq, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, qk_rope_head_dim=q_rope_head_dim, parallel=parallel, atten_parallel=atten_parallel, 
        atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    waves= np.prod(grids)/single_chip.sm_count

    if (atten_parallel.cp == 1):
        log.info("mla prefill coarse stage 5_1 overall time: %s s", single_chip_time)
        return single_chip_time
    elif (atten_parallel.cp > 1):
        # reduce-scatter along atten.cp
        # each device holding: [shard_bs, shard_num_head, shard_seq_q, head_dim]
        cp_reduce_latency, cp_reduce_ext_max, cp_reduce_traffic = reduce_scatter_wrapper(all_reduce_op_bytes=atten_bytes, parallel=atten_parallel, 
        noc_hierarchy=noc_hierarchy, granularity=granularity, dim_to_process="cp", bytes=shard_bs*shard_num_head*shard_seq_q*head_dim*in_bytes)
        if stats is not None:
            stats.append_traffic(cp_reduce_traffic, hop_time_s=cp_reduce_latency, link_time_s=cp_reduce_ext_max)

        log.info("cp_reduce_latency: %s, cp_reduce_ext_max: %s", cp_reduce_latency, cp_reduce_ext_max)

        overall_time =get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=cp_reduce_latency, network_link_time=cp_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
        log.info("mla prefill coarse stage 5_1 overall time: %s s", overall_time)
        return overall_time


def mla_prefill_coarse_stage5_2(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    pass
    # ------------------------------------- stage 5_2 -------------------------------------
    # stage 5_2
    # smem fused
    # then a reconstruct, a element-wise op, new [shard_bs, shard_num_head, shard_seq_q, head_dim] / or [shard_bs, shard_num_head, shard_seq, head_dim] do a [shard_bs, shard_num_head, shard_seq, 1x2] (rescale： sum and max)
    # transpose p + reshape p  : 
    # [bs/dp, num_head/tp, seq/(atten.sp*atten.cp), head_dim] == [bs/dp, num_head/tp, seq/sp, head_dim]
    # transpose to [bs/dp, seq/sp, num_head/tp, head_dim]
    # reshape to [bs/dp, seq/sp, hidden/tp]    

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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)
    shard_hidden = math.ceil(wq_hidden / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_seq_kv = math.ceil(seq / atten_parallel.cp)

    shard_grouped_seq_q = shard_seq_q * math.ceil(shard_num_head / shard_num_kv_head)
    shard_group_size = math.ceil(shard_num_head / shard_num_kv_head)

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

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
    
    time_stage5_2 = smem_fusion_post_data[0] 
    log.info("mla prefill coarse stage 5_2 overall time: %s s", time_stage5_2)

    return time_stage5_2

def mla_prefill_coarse_stage6(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    pass
    # ------------------------------------- stage 6 -------------------------------------
    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # x [bs/dp, seq/sp, wq_hidden]
    # W_o [wq_hidden, hidden]
    # output: [bs/dp, seq/sp, hidden]
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
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)



    group_size = math.ceil (num_head/num_kv_head)

    wq_hidden = num_head * head_dim
    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)
    shard_hidden=math.ceil(wq_hidden / parallel.tp)
    shard_kv_hidden = math.ceil(wq_hidden / (parallel.tp) / group_size)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_num_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_seq_kv = math.ceil(seq / atten_parallel.cp)

    shard_grouped_seq_q = shard_seq_q * math.ceil(shard_num_head / shard_num_kv_head)
    shard_group_size = math.ceil(shard_num_head / shard_num_kv_head)

    # mla:
    # head_dim = hidden/num_head, g = num_head/num_kv_head

    # x[bs/dp,seq/sp,hidden]
    # wq[hidden,hidden/tp]
    # wk, wv, [hidden, hiddden/tp/g]
    # wo [hidden/tp, hidden]
    # parallel.cp == 1, atten_parallel.cp * atten_parallel.sp == parallel.sp 
    # ------------------------------------- stage 6 -------------------------------------
    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # x [bs/dp, seq/sp, wq_hidden]
    # W_o [wq_hidden, hidden]
    # output: [bs/dp, seq/sp, hidden]
    
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
        
        log.info("mla prefill coarse stage 6 fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_1, ext_max_1, noc_overall_time_1)
    
    gemm1_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'smem'),
    )

    smem_fusion_list=[]

    # (M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch)
    # happen to be no fusion, so this would be the final result, we will use the smem_fusion data only
    # log.info("GEMM M: %s, N: %s, K: %s", shard_bs * shard_seq, hidden, shard_hidden)
    hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config=gemm_wrapper(M=shard_bs*shard_seq, N=hidden, K=shard_hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
    grids = [shard_bs * shard_seq / gemm_tiling_config[0], hidden / gemm_tiling_config[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)

    log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
    log.info("hete modeling single chip metrics: %s, gemm_tiling_config: %s", get_hete_metrics(smem_fusion_post_data), gemm_tiling_config)

    # log.error("gemm_smem_fusion_post_data: %s", gemm_smem_fusion_post_data)
    

    if (granularity.get_comp_comm_overlap()==True):
        # time_stage2=max(gemm_smem_fusion_post_data[0],overall_time_1)
        additional_time=max(gemm_smem_fusion_post_data[0],noc_overall_time_1)-gemm_smem_fusion_post_data[0]
        log.info("mla prefill coarse stage 6 fsdp additional time: %s s", additional_time)
    elif (granularity.get_comp_comm_overlap()==False):
        # time_stage2=gemm_smem_fusion_post_data[0] + noc_overall_time_1
        additional_time = noc_overall_time_1

    single_chip_time=smem_fusion_post_data[0] + additional_time

    waves=np.prod(grids)/single_chip.sm_count


    all_reduce_traffic = None
    if (parallel.tp > 1 and next_parallel.tp == parallel.tp):
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
        all_reduce_latency, all_reduce_ext_max = 0, 0
    else:
        all_reduce_latency, all_reduce_ext_max = 0, 0
        # raise NotImplementedError("not desinged for this case", parallel, next_parallel)
    
    if stats is not None and all_reduce_traffic is not None:
        stats.append_traffic(all_reduce_traffic, hop_time_s=all_reduce_latency, link_time_s=all_reduce_ext_max)

    time_stage6 = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=all_reduce_latency, network_link_time=all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
    # next we need to do a rope
    # ropeq_coarse(bs:int, head:int, seq:int, head_dim:int, parallel:ParallelScheme, next_parallel:ParallelScheme, rope_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy)

    # log.info("mla prefill coarse stage 6 single chip time before all reduce: %s s", single_chip_time)
    # log.info("mla prefill coarse stage 6 all reduce time: %s s", all_reduce_latency + all_reduce_ext_max)
    # log.info("mla prefill coarse stage 6 overall time: %s s", e2e_time)
    return time_stage6

def mla_prefill_coarse(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    '''
    Prefill:
    no weight absorption.
    '''
    if granularity.dump_perf_log == True:
        stats = OpPerfStats(op_name="mla_prefill", dump_perf_log=True)
    else:
        stats = None

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

    # 其中x [bs/dp, seq/sp, hidden], 
    # Wq_a, [hidden, q_down_hidden], Wq_b [q_down_hidden, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # Wkv_down [hidden, (kv_rope_head_dim+kv_nope_head_dim)]
    # Wkv_up [kv_nope_head_dim, kv_nope_up_hidden/tp]
    # W_o [num_head*head_dim/tp, hidden]

    # input: [bs/dp, seq/sp, hidden]

    # stage 1:
    # q_a = x@Wq_a -> [bs/dp, seq/sp, q_down_hidden]

    # stage 2:
    # q_b = q_a@Wq_b -> [bs/dp, seq/sp, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # rope(q_rope) -> [bs/dp, seq/sp, num_head/tp*q_rope_head_dim]

    # stage 3:
    # kv_down = x@Wkv_down -> [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]
    # rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]
    # dup (logically) -> [bs/dp, seq/sp, num_kv_head, kv_rope_head_dim]

    # stage 4:
    # kv_up = kv_down@Wkv_up -> [bs/dp, seq/sp, kv_nope_up_hidden/tp]
    # writing the split-> part of k and part of v
    # k: [bs/dp, seq/sp, num_kv_head, (kv_rope_head_dim+head_dim)]
    # v: [bs/dp, seq/sp, num_kv_head, head_dim]

    # stage 5:
    # MHA, copy previous code
    # flash attention, just copy previous code
    # in mla_prefill_coarse, time_stage4 and time_stage5

    # stage 6:
    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]

    # let's model it one by one 

    # ------------------------------------- stage 1 -------------------------------------
    # q_a = x@Wq_a -> [bs/dp, seq/sp, q_down_hidden]
    time_stage1 = mla_prefill_coarse_stage1(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # ------------------------------------- stage 2 -------------------------------------
    # q_b = q_a@Wq_b -> [bs/dp, seq/sp, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # rope(q_rope) -> [bs/dp, seq/sp, num_head/tp*q_rope_head_dim]
    time_stage2 = mla_prefill_coarse_stage2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # ------------------------------------- stage 3 -------------------------------------
    # kv_down = x@Wkv_down -> [bs/dp, seq/sp, (kv_rope_head_dim+kv_nope_head_dim)]
    # rope(kv_rope) -> [bs/dp, seq/sp, kv_rope_head_dim]
    # dup (logically) -> [bs/dp, seq/sp, num_kv_head, kv_rope_head_dim]
    time_stage3 = mla_prefill_coarse_stage3(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # ------------------------------------- stage 4 -------------------------------------
    # kv_up = kv_down@Wkv_up -> [bs/dp, seq/sp, kv_nope_up_hidden/tp]
    # writing the split-> part of k and part of v
    # k: [bs/dp, seq/sp, num_kv_head, (kv_rope_head_dim+head_dim)]
    # v: [bs/dp, seq/sp, num_kv_head, head_dim]
    time_stage4 = mla_prefill_coarse_stage4(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # ------------------------------------- stage 5 -------------------------------------
    # MHA, copy previous code
    # flash attention, just copy previous code
    # in mla_prefill_coarse, time_stage4 and time_stage5

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

    time_stage5_1 = mla_prefill_coarse_stage5_1(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_stage5_2 = mla_prefill_coarse_stage5_2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    # time_stage5 = time_stage5_1 + time_stage5_2 

    # ------------------------------------- stage 6 -------------------------------------
    # o = x@W_o -> [bs/dp, seq/sp, num_head*head_dim/tp] @ [num_head*head_dim/tp, hidden] -> [bs/dp, seq/sp, hidden]
    # tp-level all-reduce o -> [bs/dp, seq/sp, hidden]
    # output: [bs/dp, seq/sp, hidden]
    # x [bs/dp, seq/sp, wq_hidden]
    # W_o [wq_hidden, hidden]
    # output: [bs/dp, seq/sp, hidden]
    time_stage6 = mla_prefill_coarse_stage6(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    
    
    log.info("time stage1:%s. Note: no tp for w_q_a projection which could be potential performance bottleneck", time_stage1)
    log.info("time stage2:%s", time_stage2)
    log.info("time stage3:%s. Note: no tp for w_kv_down projection which could be potential performance bottleneck", time_stage3)
    log.info("time stage4:%s", time_stage4)
    log.info("time stage5_1:%s", time_stage5_1)
    log.info("time stage5_2:%s", time_stage5_2)
    log.info("time stage6:%s", time_stage6)
    # log.info("time stage5:%s", time_stage5)
    time = time_stage1 + time_stage2 + time_stage3 + time_stage4 + time_stage5_1 + time_stage5_2 + time_stage6
    if stats is not None:
        stats.finalize(total_time_s=time, h=noc_hierarchy, arch=single_chip)
        

    return time, stats
