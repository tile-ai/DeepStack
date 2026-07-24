import torch
import math
import numpy as np
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, make_mesh_or_torus, make_switch, TopoKind, PortSpread, get_extend_max_routes, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
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
from mosaic.utils import estimate_moe_routing_max, estimate_moe_routing_imbalance_overhead
from mosaic.utils import estimate_experts_activated
from mosaic.llm.swiglu.swiglu_coarse import swiglu_coarse
from mosaic.llm.swiglu.swiglu_coarse_by_token import swiglu_coarse_by_token
from mosaic.llm.swiglu.swiglu_coarse_group_gemm import swiglu_coarse_group_gemm
from mosaic.op_dtype.reduce_wrapper import reduce_wrapper
from mosaic.collectives import ep_all_to_all_wrapper
import numpy as np
from mosaic.utils import count_expert_frequency_flatten_wrapper
from mosaic.cost.op_perf_stats import OpPerfStats

def moe_coarse_gate_shared_all_to_all(bs:int, seq:int, hidden:int, moe_down_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, expert_bytes:OpBytes, gate_bytes:OpBytes, num_shared_experts:int, num_routed_experts:int, num_activated_experts:int, routing_array:np.ndarray, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    
    # stage 1 gate and normalize, dispatch
    # x [bs/dp/ep1, seq/sp/ep2, hidden]
    # gate [hidden, num_routed_experts]
    # x [bs/dp/ep1, seq/sp/ep2, hidden] @ gate [hidden, num_routed_experts] -> [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    # gated scores [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    # do some normalization by scores for weights
    # normliad_scores = gated_scores / sum(gated_scores) -> [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    
    
    # stage2 shared experts
    # doing this locally on each device
    # call a swiglu

    # all-to-all routing fusing in aforemenioned ops

    

    def gate():
        # stage 1 gate and normalize, dispatch
        # x [bs/dp/ep1, seq/sp/ep2, hidden]
        # gate [hidden, num_routed_experts]
        # x [bs/dp/ep1, seq/sp/ep2, hidden] @ gate [hidden, num_routed_experts] -> [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
        # gated scores [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
        # do some normalization by scores for weights
        # normliad_scores = gated_scores / sum(gated_scores) -> [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
        shard_tokens = math.ceil(bs/parallel.dp/parallel.ep1 * seq/parallel.sp/parallel.ep2 )
        shard_bs = math.ceil(bs/parallel.dp/parallel.ep1)
        shard_seq = math.ceil(seq/parallel.sp/parallel.ep2)
        shard_hidden = math.ceil(hidden/parallel.tp)


        smem_fusion_list=[]

        # 1.gemm gate
        gemm_gate_bytes=OpBytes(
        input1=Tensor_Loc(gate_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(gate_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(gate_bytes.output.dtype, 'smem'),
        )
        hete_post_data_gemm, smem_fusion_post_data_gemm, gemm_tiling_config=gemm_wrapper(M=shard_tokens, N=num_routed_experts, K=hidden, gemm_bytes=gemm_gate_bytes, granularity=granularity, single_chip=single_chip)
        grids = [shard_tokens / gemm_tiling_config[0], num_routed_experts / gemm_tiling_config[1]]
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

        # 2.top-k
        topk_op_bytes=OpBytes(
        input1=Tensor_Loc(gate_bytes.output.dtype, 'smem',[shard_tokens, num_routed_experts]),
        # input2=Tensor_Loc(rms_norm_bytes.input2.dtype, rms_norm_bytes.input2.loc,[shard_bs, shard_seq, shard_hidden]),
        input2=None,
        output=Tensor_Loc(gate_bytes.output.dtype, 'smem',[shard_tokens, num_activated_experts]),
        )

        
        hete_post_data_top_k, out_tb_shape = reduce_wrapper(topk_op_bytes, granularity, single_chip, tb_tiling_config=None)
        # hete_post_data_mean, out_tb_shape = reduce_wrapper(mean_op_bytes, granularity, single_chip, tb_tiling_config=[2,2,1])
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_top_k], single_chip))

        # 3.1norm_weights
        # counting bin scores for each token
        # doing some rescale and normalization by scores for weights
        
        sum_weights_op_bytes=OpBytes(
        input1=Tensor_Loc(gate_bytes.output.dtype, 'smem',[shard_tokens, num_activated_experts]),
        # input2=Tensor_Loc(rms_norm_bytes.input2.dtype, rms_norm_bytes.input2.loc,[shard_bs, shard_seq, shard_hidden]),
        input2=None,
        output=Tensor_Loc(gate_bytes.output.dtype, 'reg',[shard_tokens, 1]),
        )

        hete_post_data_sum_weights, out_tb_shape = reduce_wrapper(sum_weights_op_bytes, granularity, single_chip, tb_tiling_config=None)
        
        # 3.2 reciprocal sum 
        reciprocal_op_bytes=OpBytes(
        input1=Tensor_Loc(gate_bytes.output.dtype, 'reg',[shard_tokens, 1]),
        input2=None,
        output=Tensor_Loc(gate_bytes.output.dtype, 'reg',[shard_tokens, 1]),
        )

        hete_post_data_reciprocal, out_tb_shape = element_wrapper(reciprocal_op_bytes, granularity, single_chip, batch=1, type="sfu_core", tb_tiling_config=None)
        
        # 3.3 rescale and normalization by scores for weights

        rescale_op_bytes=OpBytes(
        input1=Tensor_Loc(gate_bytes.output.dtype, 'smem',[shard_tokens, num_activated_experts]),
        input2=Tensor_Loc(gate_bytes.output.dtype, 'reg',[shard_tokens, 1]),
        output=Tensor_Loc(gate_bytes.output.dtype, 'smem',[shard_tokens, num_activated_experts]),
        )

        hete_post_data_rescale, out_tb_shape = element_wrapper(rescale_op_bytes, granularity, single_chip, batch=1, type="cuda_core", tb_tiling_config=None)
        # smem_fusion_list.append(hete_reg_fusion([hete_post_data_rescale], single_chip))
        
        # print("hete_post_data_sum_weights: ", hete_post_data_sum_weights)
        # print("hete_post_data_reciprocal: ", hete_post_data_reciprocal)
        # print("hete_post_data_rescale: ", hete_post_data_rescale)
        smem_fusion_list.append(hete_reg_fusion([hete_post_data_sum_weights,hete_post_data_reciprocal,hete_post_data_rescale], single_chip))

        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling metrics: %s", get_hete_metrics(smem_fusion_post_data))

        time_gate = smem_fusion_post_data[0]
        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        return time_gate
        # return hete_post_data_gemm, grids
    
    def shared_experts():
        transformed_parallel = ParallelScheme(tp=parallel.tp, ep=1, sp=parallel.sp * parallel.ep2, cp=parallel.cp, dp=parallel.dp * parallel.ep1, fsdp=parallel.fsdp, pp=parallel.pp)
        shard_tokens = math.ceil(bs*seq/parallel.dp/parallel.sp/parallel.ep)
        time_shared_experts, waves_shared_experts = swiglu_coarse_by_token(shard_tokens, hidden, moe_down_hidden, transformed_parallel, transformed_parallel, expert_bytes, granularity, single_chip, noc_hierarchy, stats=stats)
        time_shared_experts *= num_shared_experts
        return time_shared_experts
    
    def all_to_all():

        # all-to-all routing back

        in_bytes, expert_weight_bytes, out_bytes = expert_bytes.get_dtype_bytes()
        _, gate_weight_bytes, routing_output_bytes = gate_bytes.get_dtype_bytes()

        bytes_each_token = in_bytes * hidden 

        tokens_each_ep_group = math.ceil(bs/parallel.dp) * math.ceil(seq/parallel.sp)

        imbalance_overhead_ratio = estimate_moe_routing_imbalance_overhead(num_routed_experts, num_activated_experts, parallel.ep, tokens_each_ep_group, R=parallel.tp*parallel.sp*parallel.dp)
        
        noc_hop_time_1, noc_ext_max_1, noc_traffic_1 = ep_all_to_all_wrapper(parallel, noc_hierarchy, granularity, bytes_each_token, routing_array, bs, seq, num_routed_experts, num_activated_experts, imbalance_overhead_ratio)
        # if ep ==1 then no traffic 
        if stats is not None and noc_traffic_1 is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=noc_hop_time_1, link_time_s=noc_ext_max_1)
        # print("routing_array type: %s", type(routing_array))
        return noc_hop_time_1, noc_ext_max_1

    time_gate = gate()
    if num_shared_experts > 0:
        time_shared_experts = shared_experts()
    else:
        time_shared_experts = 0
    noc_hop_time_1, noc_ext_max_1 = all_to_all()


    log.info("time_gate: %s, time_shared_experts: %s, noc_hop_time_1: %s, noc_ext_max_1: %s", time_gate, time_shared_experts, noc_hop_time_1, noc_ext_max_1)
    if (granularity.get_comp_comm_overlap() == False):
        return time_gate + time_shared_experts + noc_hop_time_1 + noc_ext_max_1
    else:
        return time_gate + max(time_shared_experts, noc_hop_time_1 + noc_ext_max_1)


            # grids = [shard_bs * shard_seq / tiling_config[0], hidden / tiling_config[1]]
    # waves = np.prod(grids) / single_chip.sm_count
    # e2e_time = get_comp_comm_e2e_time(compute_time=time_stage3, network_hop_latency=all_reduce_hop, network_link_time=all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
    # return time_gate + time_shared_experts + 
    
def moe_coarse_routed_experts_all_to_all(bs:int, seq:int, hidden:int, moe_down_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, expert_bytes:OpBytes, gate_bytes:OpBytes, num_shared_experts:int, num_routed_experts:int, num_activated_experts:int, routing_array:np.ndarray, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    # log.info("routing_array type: %s", type(routing_array))
    # log.info("routing_array shape: %s", routing_array.shape)
    def routed_experts():     
        # log.info("routing_array type: %s", type(routing_array))

        if (bs * seq >= 4096):
            """
            tokens总数较多, 使用overhead率进行评估
            """
            # each device is responsible for doing: num_activated_experts * [bs/dp/ep1, seq/sp/ep2, hidden] tokens' swiglu
            # assume math.ceil(bs/parallel.dp/parallel.ep1) = 4, math.ceil(seq/parallel.sp/parallel.ep2) = 2, num_activated_experts = 8
            num_tokens_per_device = math.ceil(bs/parallel.dp * seq / parallel.sp / parallel.ep) * num_activated_experts

            # each device holds num_routed_experts / ep experts, but how many activated?
            num_experts_per_device = math.ceil(num_routed_experts / parallel.ep)

            tokens_each_ep_group = math.ceil(bs/parallel.dp * seq / parallel.sp) 
            # assume imbalance_overhead_ratio = 1.3
            # estimate_moe_routing_imbalance_overhead(M, n, EP, T, R, method="qwen")
            imbalance_overhead_ratio = estimate_moe_routing_imbalance_overhead(num_routed_experts, num_activated_experts, parallel.ep, tokens_each_ep_group, R=parallel.tp*parallel.sp*parallel.dp)
            # imbalance_overhead_ratio = 1
            log.info("imbalance_overhead_ratio: %s, num_routed_experts: %s, num_activated_experts: %s, parallel.ep: %s, tokens_each_ep_group: %s, R: %s", imbalance_overhead_ratio, num_routed_experts, num_activated_experts, parallel.ep, tokens_each_ep_group, parallel.tp*parallel.sp*parallel.dp)

            # then this would be 4 * 2 * 8 * 1.3 = 83.2, rounded to 84
            num_tokens_per_device = round(num_tokens_per_device * imbalance_overhead_ratio)

            # and this would be 42 for 32 experts, 10 *2 times, 22 * 1 times
            num_few_activated_experts, few_activated_times, num_more_activated_experts, more_activated_times = estimate_experts_activated(num_tokens_per_device, num_experts_per_device)

            # swiglu for few activated experts
            # transformed_bs = math.ceil(bs / parallel.ep1)
            # if few_activated_times == 0:
            #     time_few_activated_experts = 0
            #     waves_few_activated_experts = 0
            # else:

            #     shard_tokens = few_activated_times

            #     transformed_parallel = ParallelScheme(tp=parallel.tp, ep=1, sp=parallel.sp * parallel.ep2, cp=parallel.cp, dp=parallel.dp * parallel.ep1, fsdp=parallel.fsdp, pp=parallel.pp)
            #     time_few_activated_experts, waves_few_activated_experts = swiglu_coarse_by_token(shard_tokens, hidden, moe_down_hidden, transformed_parallel, transformed_parallel, expert_bytes, granularity, single_chip, noc_hierarchy)
            #     log.info("moe coarse few activated experts time: %s", time_few_activated_experts)
            #     time_few_activated_experts *= num_few_activated_experts

            #     waves_few_activated_experts = waves_few_activated_experts * num_few_activated_experts
            #     log.info("time_few_activated_experts: %s, waves_few_activated_experts: %s", time_few_activated_experts, waves_few_activated_experts)
                
                

            # # swiglu for more activated experts
            # if more_activated_times == 0:
            #     time_more_activated_experts = 0
            #     waves_more_activated_experts = 0
            # else:


            #     shard_tokens = more_activated_times

            #     transformed_parallel = ParallelScheme(tp=parallel.tp, ep=1, sp=parallel.sp * parallel.ep2, cp=parallel.cp, dp=parallel.dp * parallel.ep1, fsdp=parallel.fsdp, pp=parallel.pp)
            #     time_more_activated_experts, waves_more_activated_experts = swiglu_coarse_by_token(shard_tokens, hidden, moe_down_hidden, transformed_parallel, transformed_parallel, expert_bytes, granularity, single_chip, noc_hierarchy)
            #     log.info("moe coarse more activated experts time: %s", time_more_activated_experts)
            #     time_more_activated_experts *= num_more_activated_experts
            #     waves_more_activated_experts = waves_more_activated_experts * num_more_activated_experts
            #     log.info("time_more_activated_experts: %s, waves_more_activated_experts: %s", time_more_activated_experts, waves_more_activated_experts)
            
            log.info("num_few_activated_experts: %s, few_activated_times: %s, num_more_activated_experts: %s, more_activated_times: %s", num_few_activated_experts, few_activated_times, num_more_activated_experts, more_activated_times)

            expert_row = [few_activated_times] * num_few_activated_experts + [more_activated_times] * num_more_activated_experts
            # to ndarry
            expert_row = np.array(expert_row)
            counts = np.bincount(expert_row)

            transformed_parallel = ParallelScheme(tp=parallel.tp, ep=1, sp=parallel.sp * parallel.ep2, cp=parallel.cp, dp=parallel.dp * parallel.ep1, fsdp=parallel.fsdp, pp=parallel.pp)
            time_computing_routed_experts, waves_computing_routed_experts = swiglu_coarse_group_gemm(expert_row, counts, hidden, moe_down_hidden, transformed_parallel, transformed_parallel, expert_bytes, granularity, single_chip, noc_hierarchy, stats=stats)

            # time_computing_routed_experts = time_few_activated_experts + time_more_activated_experts

            # waves_computing_routed_experts = waves_few_activated_experts + waves_more_activated_experts

            # construct the expert row and counts

            # 

            return time_computing_routed_experts, waves_computing_routed_experts
        
        else:
            """
            tokens总数较少, 使用真实routing array进行评估
            """

            # 策略: 统计这些tokens 累计出来，激活的expert最多的ep组，来评估时间上限。
            expected_num_tokens_per_device = math.ceil(bs/parallel.dp * seq / parallel.sp / parallel.ep) * num_activated_experts
            # well we see from the trace
            repeated_times = parallel.dp * parallel.sp
            # tp are identical, no impact on token routing

            # each group we got: math.ceil(bs/parallel.dp * seq / parallel.sp / parallel.ep) tokens to see the results

            local_routing_array = routing_array.reshape(-1, routing_array.shape[-1])
            local_routing_array = local_routing_array[:bs*seq]
            tokens_per_ep_group = math.ceil(bs/parallel.dp * seq / parallel.sp)

            expert_row = count_expert_frequency_flatten_wrapper(flatten_numpy=local_routing_array, group_tokens=tokens_per_ep_group, total_experts=num_routed_experts, EP=parallel.ep)

            # counts = np.bincount(expert_row, minlength=expert_row.max()+1)
            counts = np.bincount(expert_row)
            # hist_matrix = np.column_stack((np.arange(counts.size), counts))
            # log.info("counts: %s", counts)
            # log.info("hist_matrix: %s", hist_matrix.tolist())
            # expert_row: [4 0 0 0 1 0 2 2 4 2 1 6 0 2 1 2 0 1 1 4 2 0 1 0 2 1 1 0 1 5 4 1]
            # counts: [ 9 10  7  0  4  1  1]
            # 19:22:59 INFO [mosaic.llm.moe.moe_coarse] expert_row: [23 18 16 15 15 13 20 19 16 15 16 30 12 16 21 24 19 16 19 20 17 12 15 16 13 15 16  7 14 18 20 20]
            # 19:22:59 INFO [mosaic.llm.moe.moe_coarse] counts: [0 0 0 0 0 0 0 1 0 0 0 0 2 2 1 5 7 1 2 3 4 1 0 1 1 0 0 0 0 0 1]
            # expert_row.shape = [total_experts/EP]
            log.info("parallel information: %s", parallel)
            log.info("expert_row: %s", expert_row)
            log.info("counts: %s", counts)

            activated_tokens = 0
            time_total_activated_experts = 0
            waves_total_activated_experts = 0
            
            # expert_row 表示, 每个ep组(total_experts/EP个), 每个expert激活次数
            # counts则是bin count,第index个数的值num_expert表示, 有index个token的expert总共有num_expert个
            # for expert_count in counts:
            #     log.info("expert_count: %s", expert_count)
            #     if (activated_tokens == 0):
            #         # skip as no tokens are routed to these experts
            #         activated_tokens += 1
            #     else:
            #         log.info("activated_tokens: %s", activated_tokens)
            #         activated_times = activated_tokens
            #         minibatch = expert_count
                
            #         # log.info("activated_times: %s, minibatch: %s", activated_times, minibatch)
            #         shard_tokens = activated_times
            #         # shard_tokens: 1
            #         log.info("shard_tokens: %s", shard_tokens)

            #         transformed_parallel = ParallelScheme(tp=parallel.tp, ep=1, sp=parallel.sp * parallel.ep2, cp=parallel.cp, dp=parallel.dp * parallel.ep1, fsdp=parallel.fsdp, pp=parallel.pp)
            #         time_activated_experts, waves_activated_experts = swiglu_coarse_by_token(shard_tokens, hidden, moe_down_hidden, transformed_parallel, transformed_parallel, expert_bytes, granularity, single_chip, noc_hierarchy)
                    
            #         # log.info("time_activated_experts: %s, minibatch: %s", time_activated_experts, minibatch)
            #         time_activated_experts *= minibatch
            #         # log.info("time_activated_experts: %s", time_activated_experts)

            #         waves_activated_experts = waves_activated_experts * minibatch

            #         time_total_activated_experts += time_activated_experts
            #         waves_total_activated_experts += waves_activated_experts

            #         activated_tokens += 1

            transformed_parallel = ParallelScheme(tp=parallel.tp, ep=1, sp=parallel.sp * parallel.ep2, cp=parallel.cp, dp=parallel.dp * parallel.ep1, fsdp=parallel.fsdp, pp=parallel.pp)
            
            # print(f"expert_row.type: {type(expert_row)}")
            # print(f"counts.type: {type(counts)}")
            # print(f"expert_row: {expert_row}")
            # print(f"counts: {counts}")
            time_activated_experts, waves_activated_experts = swiglu_coarse_group_gemm(expert_row, counts, hidden, moe_down_hidden, transformed_parallel, transformed_parallel, expert_bytes, granularity, single_chip, noc_hierarchy, stats=stats)
            
            # log.info("time_activated_experts: %s, minibatch: %s", time_activated_experts, minibatch)
            # log.info("time_activated_experts: %s", time_activated_experts)

            time_total_activated_experts = time_activated_experts
            waves_total_activated_experts = waves_activated_experts

                    
            log.info("time_routed_experts: %s, waves_routed_experts: %s", time_total_activated_experts, waves_total_activated_experts)
            return time_total_activated_experts, waves_total_activated_experts

    # all-to-all routing back
    def all_to_all_back():
        # all-to-all routing back

        in_bytes, expert_weight_bytes, out_bytes = expert_bytes.get_dtype_bytes()
        _, gate_weight_bytes, routing_output_bytes = gate_bytes.get_dtype_bytes()

        bytes_each_token = out_bytes * hidden 

        tokens_each_ep_group = math.ceil(bs/parallel.dp) * math.ceil(seq/parallel.sp)

        imbalance_overhead_ratio = estimate_moe_routing_imbalance_overhead(num_routed_experts, num_activated_experts, parallel.ep, tokens_each_ep_group, R=parallel.tp*parallel.sp*parallel.dp)
        
        
        noc_hop_time_1, noc_ext_max_1, noc_traffic_1 = ep_all_to_all_wrapper(parallel, noc_hierarchy, granularity, bytes_each_token, routing_array, bs, seq, num_routed_experts, num_activated_experts, imbalance_overhead_ratio)
        # if ep ==1 then no traffic 
        if stats is not None and noc_traffic_1 is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=noc_hop_time_1, link_time_s=noc_ext_max_1)
        return noc_hop_time_1, noc_ext_max_1

    time_computing_routed_experts, waves_computing_routed_experts = routed_experts()
    
    noc_hop_time_1, noc_ext_max_1 = all_to_all_back()

    e2e_time = get_comp_comm_e2e_time(compute_time=time_computing_routed_experts, network_hop_latency=noc_hop_time_1, network_link_time=noc_ext_max_1, waves=waves_computing_routed_experts, overlap=granularity.get_comp_comm_overlap())

    log.info("time_computing_routed_experts: %s, waves_computing_routed_experts: %s, noc_hop_time_1: %s, noc_ext_max_1: %s, e2e_time: %s", time_computing_routed_experts, waves_computing_routed_experts, noc_hop_time_1, noc_ext_max_1, e2e_time)

    return e2e_time
    
def moe_coarse_token_weighted_summation(bs:int, seq:int, hidden:int, moe_down_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, expert_bytes:OpBytes, gate_bytes:OpBytes, num_shared_experts:int, num_routed_experts:int, num_activated_experts:int, routing_array:np.ndarray, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):

    # weighted summation for shared experts and routed experts

    shard_tokens = math.ceil(bs/parallel.dp * seq/parallel.sp/parallel.ep )

    smem_fusion_list=[]

    # 1. scaled tokens for shared experts and routed experts

    scaled_tokens_op_bytes=OpBytes(
    input1=Tensor_Loc(gate_bytes.output.dtype, 'ddr',[shard_tokens * ((num_shared_experts + num_activated_experts)), hidden]),
    input2=Tensor_Loc(gate_bytes.output.dtype, 'ddr',[1, 1]),
    output=Tensor_Loc(gate_bytes.output.dtype, 'smem',[shard_tokens * ((num_shared_experts + num_activated_experts)), hidden]),
    )
    # log.info("shard_tokens: %s, num_shared_experts: %s, num_activated_experts: %s", shard_tokens, num_shared_experts, num_activated_experts)
    
    hete_post_data_scaled_tokens, out_tb_shape = element_wrapper(scaled_tokens_op_bytes, granularity, single_chip, batch=1, type="cuda_core", tb_tiling_config=None)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_scaled_tokens], single_chip))

    grids = [shard_tokens * ((num_shared_experts + num_activated_experts)) / out_tb_shape[0], hidden / out_tb_shape[1]]

    # 2. reduce sum 

    # reduce_sum_op_bytes=OpBytes(
    # input1=Tensor_Loc(gate_bytes.output.dtype, 'smem',[shard_tokens * ((num_shared_experts + num_activated_experts)), hidden]),
    # input2=None,
    # output=Tensor_Loc(gate_bytes.output.dtype, 'ddr',[shard_tokens , hidden]),
    # )

    reduce_sum_op_bytes=OpBytes(
    input1=Tensor_Loc(gate_bytes.output.dtype, 'smem',[hidden, shard_tokens * ((num_shared_experts + num_activated_experts)) ]),
    input2=None,
    output=Tensor_Loc(gate_bytes.output.dtype, 'ddr',[hidden, shard_tokens]),
    )

    hete_post_data_reduce_sum, out_tb_shape = reduce_wrapper(reduce_sum_op_bytes, granularity, single_chip, tb_tiling_config=None)
    log.info("hete_post_data_reduce_sum: %s", hete_post_data_reduce_sum)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_reduce_sum], single_chip))
    

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
    log.info("hete modeling metrics: %s", get_hete_metrics(smem_fusion_post_data))

    time_token_weighted_summation = smem_fusion_post_data[0]

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    return time_token_weighted_summation

def moe_coarse(bs:int, seq:int, hidden:int, moe_down_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, expert_bytes:OpBytes, gate_bytes:OpBytes, num_shared_experts:int, num_routed_experts:int, num_activated_experts:int, routing_array:np.ndarray, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    assert granularity.get_mode() == "coarse"

    stats = OpPerfStats(op_name="moe") if granularity.dump_perf_log else None

    shard_bs = math.ceil(bs / parallel.dp / parallel.ep1)
    shard_seq = math.ceil(seq / parallel.sp / parallel.ep2)
    # shard_hidden = math.ceil(hidden / parallel.tp)
    shard_moe_down_hidden = math.ceil(moe_down_hidden / parallel.tp)
    num_shard_routed_experts = math.ceil(num_routed_experts / parallel.ep)

    # extracting the trace file
    tokens_total = bs * seq 

    tokens_each_ep_group = math.ceil(bs/parallel.dp) * math.ceil(seq/parallel.sp)

    # estimate the max EP-bin
    # this is the "partial expert", as there is TP inside each expert 
    max_ep_bin = estimate_moe_routing_max(M=num_routed_experts, n=num_activated_experts, EP=parallel.ep, T=tokens_each_ep_group, R=parallel.tp*parallel.sp*parallel.dp, method="qwen")
    # print(f"max_ep_bin: {max_ep_bin}")
    # routing_array

    # stage 1 gate and normalize, dispatch
    # x [bs/dp/ep1, seq/sp/ep2, hidden]
    # gate [hidden, num_routed_experts]
    # x [bs/dp/ep1, seq/sp/ep2, hidden] @ gate [hidden, num_routed_experts] -> [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    # gated scores [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    # do some normalization by scores for weights
    # normliad_scores = gated_scores / sum(gated_scores) -> [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    
    
    # stage2 shared experts
    # doing this locally on each device
    # call a swiglu

    # all-to-all routing fusing in aforemenioned ops

    # stage 3 non-shared experts
    # estimating: max_ep_bin swiglu + all-to-all routing back
    # max_ep_bin times -> [bs/dp/ep1, seq/sp/ep2, hidden]
    # all-to-all routing back -> [bs/dp/ep1, seq/sp/ep2, hidden]

    # stage 4 accumulation
    # after routing back, do a reduce weighted sum for non-shared experts and shared experts
    # reduce-sum
    # -> [bs/dp/ep1, seq/sp/ep2, hidden]

    # ------------------------------------- stage 1 + 2 -------------------------------------
    # gate and normalize, dispatch
    # x [bs/dp/ep1, seq/sp/ep2, hidden]
    # gate [hidden, num_routed_experts]
    # x [bs/dp/ep1, seq/sp/ep2, hidden] @ gate [hidden, num_routed_experts] -> [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    # gated scores [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    # do some normalization by scores for weights
    # normliad_scores = gated_scores / sum(gated_scores) -> [bs/dp/ep1, seq/sp/ep2, num_routed_experts]
    # stage2 shared experts
    # doing this locally on each device
    # call a swiglu
    # all-to-all routing fusing in aforemenioned ops
    
    time_gate_shared_all_to_all = moe_coarse_gate_shared_all_to_all(bs, seq, hidden, moe_down_hidden, parallel, next_parallel, expert_bytes, gate_bytes, num_shared_experts, num_routed_experts, num_activated_experts, routing_array, granularity, single_chip, noc_hierarchy, stats=stats)

    time_routed_experts = moe_coarse_routed_experts_all_to_all(bs, seq, hidden, moe_down_hidden, parallel, next_parallel, expert_bytes, gate_bytes, num_shared_experts, num_routed_experts, num_activated_experts, routing_array, granularity, single_chip, noc_hierarchy, stats=stats)

    time_token_weighted_summation = moe_coarse_token_weighted_summation(bs, seq, hidden, moe_down_hidden, parallel, next_parallel, expert_bytes, gate_bytes, num_shared_experts, num_routed_experts, num_activated_experts, routing_array, granularity, single_chip, noc_hierarchy, stats=stats)
    
    log.info("time_gate_shared_all_to_all: %s", time_gate_shared_all_to_all)
    log.info("time_routed_experts: %s", time_routed_experts)
    log.info("time_token_weighted_summation: %s", time_token_weighted_summation)

    time_moe_coarse = time_gate_shared_all_to_all + time_routed_experts + time_token_weighted_summation
    log.info("time_moe_coarse: %s", time_moe_coarse)

    if stats is not None:
        stats.finalize(time_moe_coarse, h=noc_hierarchy, arch=single_chip)
    return time_moe_coarse, stats
