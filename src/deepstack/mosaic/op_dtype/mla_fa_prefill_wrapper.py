from mosaic.utils import OpBytes, Modeling_Granularity, Tensor_Loc
from tilesight.arch import Arch, uses_dram_wave_quantization
import math
from mosaic.parallelism.parallel import ParallelScheme
# from tilesight.fused_op_dtype.matmul_fused_op_new_api import calculate_matmul_resource_utilization_new
from tilesight.fused_op_dtype_wave.matmul_fused_op_new_api_wave import calculate_matmul_resource_utilization_new
# from tilesight.fusion_support.post_process_single_op import hyper_post_process_tensor_core_op,hyper_post_process_sfu_core_op,hyper_post_process_cuda_core_op
from tilesight.fusion_support.hete_post_process_single_op import hete_post_process_tensor_core_op,hete_post_process_sfu_core_op,hete_post_process_cuda_core_op
from tilesight.fused_op_dtype_wave.N_0_general_reduce_fused_op_wave import calculate_N_0_general_ruduce_resource_utilization
from tilesight.fused_op_dtype_wave.N_0_general_inter_thread_reduce_fused_op_wave import calculate_N_0_general_ruduce_inter_thread_resource_utilization
from tilesight.fused_op_dtype_wave.N_0_element_wise_fused_op_wave import calculate_N_0_elementwise_resource_utilization
from tilesight.fused_op_dtype_wave.N_1_element_wise_fused_op_wave import calculate_N_1_elementwise_resource_utilization
from tilesight.fused_op_dtype_wave.N_N_element_wise_fused_op_wave import calculate_N_N_elementwise_resource_utilization

from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
from mosaic.op_dtype.element_wrapper import element_wrapper
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy
from mosaic.cost.op_perf_stats import OpPerfStats

# tiling_configs
# tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages
from functools import lru_cache
import numpy as np
import logging

from tilesight.util.post_process_simulation import bare_post_process_tensor_core_op
log = logging.getLogger(__name__) 

def get_default_fa_tiling(shard_bs, shard_num_kv_head, shard_grouped_seq_q, shard_seq_kv, head_dim):
    
    tb_m = min(128, shard_grouped_seq_q)
    # block_k = head_dim
    tb_k = min(32, head_dim)
    tb_n = min(128,shard_seq_kv)

    wp_m = math.ceil(tb_m/2)
    wp_n = math.ceil(tb_n/2)
    wp_k = tb_k

    stages=2

    return tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages


def get_autontune_fa_tiling(shard_bs, shard_num_kv_head, shard_grouped_seq_q, shard_seq_kv, head_dim):
    
    full_configs = get_tiling_configs_medium()

    max_block_m = min(128, shard_grouped_seq_q)
    max_block_k = min(32, head_dim)
    max_tb_n = min(128,shard_seq_kv)

    filtered_configs = []

    for config in full_configs:
        tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages = config
        if tb_m <= max_block_m and tb_k <= max_block_k and tb_n <= max_tb_n:
            filtered_configs.append(config)

    return filtered_configs
    
@lru_cache(maxsize=1)
def get_tiling_configs_medium():
    tb_m_values = [1, 16, 32, 64, 128]
    tb_n_values = [1, 16, 32, 64, 128]
    tb_k_values = [1, 32]
    stages_values = [1, 2, 3]

    configs = []
    for tb_m in tb_m_values:
        for tb_n in tb_n_values:
            for tb_k in tb_k_values:
                wp_m = tb_m // 2 if tb_m != 1 else 1
                wp_n = tb_n // 2 if tb_n != 1 else 1
                wp_k = tb_k
                for stages in stages_values:
                    configs.append((tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages))
    return configs


# def mla_fa_prefill_wrapper(shard_bs, shard_num_head, shard_seq_q, shard_seq_kv, head_dim, atten_parallel:ParallelScheme, atten_bytes:OpBytes,granularity:Modeling_Granularity, single_chip:Arch):
def mla_fa_prefill_wrapper(bs:int, seq:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, qk_rope_head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme,
    atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None, seq_kv: "int | None" = None):

    # seq_kv: 每个 query 实际 attend 的 kv 长度。默认 None 表示 dense (kv 长度 = seq)。
    # DSA (sparse attention) 场景下传 min(index_topk, seq), 即每个 query 只 attend 被 indexer 选中的 topk 个 token。
    if seq_kv is None:
        seq_kv = seq

    mode=granularity.get_mode()
    tune_flag=granularity.get_auto_tune()

    # 在 GQA 里, g = K/V head 数，而 group size = Q head 数 ÷ K/V head 数。
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
    shard_seq_kv = math.ceil(seq_kv / atten_parallel.cp)

    shard_grouped_seq_q = shard_seq_q * math.ceil(shard_num_head / shard_num_kv_head)
    shard_group_size = math.ceil(shard_num_head / shard_num_kv_head)

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

    # get tiling
    

    # stage1: 
    # load the first q, k

    # 

    if mode == "coarse":
        if tune_flag==False:
            


            tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages = get_default_fa_tiling(shard_bs, shard_num_kv_head, shard_grouped_seq_q, shard_seq_kv, head_dim)
            tiling_config =(tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages)
            grids, time = model_fa_prefill_coarse_four_stage_pessimistic(bs, seq, hidden, num_head, num_kv_head, head_dim, qk_rope_head_dim, parallel, atten_parallel, atten_bytes, granularity, single_chip, noc_hierarchy, tiling_config, stats=stats, seq_kv=seq_kv)
            # time = model_fa_prefill_coarse_four_stage_pessimistic(shard_bs, shard_num_head, shard_seq_q, shard_seq_kv, head_dim, atten_parallel, atten_bytes,granularity, single_chip, tiling_config)
            return grids, time

        else:
            filtered_configs = get_autontune_fa_tiling(shard_bs, shard_num_head, shard_seq_q, shard_seq_kv, head_dim)
            min_time = float('inf')
            best_grids = None
            for config in filtered_configs:
                tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages = config
                tiling_config =(tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages)
                grids, time = model_fa_prefill_coarse_four_stage_pessimistic(bs, seq, hidden, num_head, num_kv_head, head_dim, qk_rope_head_dim, parallel, atten_parallel, atten_bytes, granularity, single_chip, noc_hierarchy, tiling_config, seq_kv=seq_kv)
                if time < min_time:
                    min_time = time
                    best_grids = grids
                    best_config = tiling_config
            if stats is not None and best_grids is not None:
                model_fa_prefill_coarse_four_stage_pessimistic(bs, seq, hidden, num_head, num_kv_head, head_dim, qk_rope_head_dim, parallel, atten_parallel, atten_bytes, granularity, single_chip, noc_hierarchy, best_config, stats=stats, seq_kv=seq_kv)
            return best_grids, min_time
            # raise NotImplementedError("Auto tune is not implemented for fa prefill")


def model_fa_prefill_coarse_four_stage_pessimistic(bs:int, seq:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, qk_rope_head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme,
    atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, tiling_config:tuple, stats: "OpPerfStats | None" = None, seq_kv: "int | None" = None):
    tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages = tiling_config

    if seq_kv is None:
        seq_kv = seq
    
    assert hasattr(single_chip, 'support_wgmma'), "single_chip.support_wgmma does not exist"
    if single_chip.support_wgmma:
        # wg_1_m=tb_m
        # wg_1_n=tb_n
        wp_m=tb_m
        wp_n=tb_n
    else:
        pass
    accum_bytes=4
    log.info("tb_m: %s, tb_n: %s, tb_k: %s, wp_m: %s, wp_n: %s, wp_k: %s, stages: %s", tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages)

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

    # stage1
    # load the first q

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
    shard_seq_kv = math.ceil(seq_kv / atten_parallel.cp)

    shard_grouped_seq_q = shard_seq_q * math.ceil(shard_num_head / shard_num_kv_head)
    shard_group_size = math.ceil(shard_num_head / shard_num_kv_head)

    # grids=[shard_bs, shard_num_kv_head, shard_grouped_seq_q/tb_m, head_dim/1]
    grids=[shard_bs, shard_num_kv_head, shard_grouped_seq_q/tb_m]
    waves=np.prod(grids)/single_chip.sm_count

    KV_l2_hit_rate=1-(tb_m/shard_grouped_seq_q)

    # smem_footprint=tb_m*head_dim*in_bytes + tb_n*head_dim*in_bytes*stages + tb_m*head_dim*in_bytes*stages + tb_m * tb_n * stages * in_bytes
    smem_footprint=tb_m*head_dim*in_bytes + tb_n*head_dim*in_bytes*stages + tb_m*head_dim*in_bytes*stages 

    log.info("Flash Attention prefill each device processing: shard_bs: %s, shard_num_kv_head: %s, shard_seq_q: %s, shard_grouped_seq_q: %s, shard_seq_kv: %s, head_dim: %s", shard_bs, shard_num_kv_head, shard_seq_q, shard_grouped_seq_q, shard_seq_kv, head_dim)

    def stage1():
        # load q

        load_q_bytes=OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr',[shard_bs, shard_num_kv_head, shard_grouped_seq_q, head_dim+qk_rope_head_dim]),
        input2=None,
        output=Tensor_Loc(atten_bytes.output.dtype, 'smem',[shard_bs, shard_num_kv_head, shard_grouped_seq_q, head_dim+qk_rope_head_dim]),
        )

        smem_fusion_list=[]

        stage1_tb_tile = (tb_m, 1)
    
        hete_post_data_load_q, tb_load_q=element_wrapper(element_op_bytes=load_q_bytes, granularity=granularity, single_chip=single_chip, batch=1, type="cuda_core", tb_tiling_config=stage1_tb_tile)

        smem_fusion_list.append(hete_reg_fusion([hete_post_data_load_q], single_chip))
        

        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling single chip metrics: %s", get_hete_metrics(smem_fusion_post_data))

        time_1 = smem_fusion_post_data[0]
        log.info("mla flash attention v3 coarse pessimistic stage1: %s", time_1)

        return time_1
    
    def stage2():

        # load k, gemm1

        smem_fusion_list=[]
        
        

        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = 0,0,0,0,0,0,0,0,0
            # load k and compute gemm_tile
        l2_read_io+=np.prod(grids)*(head_dim+qk_rope_head_dim)*shard_seq_kv*in_bytes

        assert hasattr(single_chip, 'ddr_wave_bytes'), "arch.ddr_wave_bytes does not exist"
        if l2_read_io > 0 and uses_dram_wave_quantization(single_chip):
            l2_read_waves = l2_read_io / single_chip.ddr_wave_bytes
            log.info("l2 read waves: %s, original l2 read io: %s", l2_read_waves, l2_read_io)
            l2_read_io = l2_read_io * math.ceil(l2_read_waves) / l2_read_waves

        l2_store_io=0
        if l2_store_io > 0 and uses_dram_wave_quantization(single_chip):
            l2_store_waves = l2_store_io / single_chip.ddr_wave_bytes
            log.info("l2 store waves: %s, original l2 store io: %s", l2_store_waves, l2_store_io)
            l2_store_io = l2_store_io * math.ceil(l2_store_waves) / l2_store_waves

        ddr_read_io+=l2_read_io*(1-KV_l2_hit_rate) 

        compute_flops+=np.prod(grids)*(head_dim+qk_rope_head_dim)*shard_seq_kv*2

        reg_footprint+=tb_m*tb_n/128*accum_bytes/4

        # smem_l1_io+=np.prod(grids)*(block_m*(dim)*seq_len/block_n*(block_n/wg_1_n) + (dim)*block_n*seq_len/block_n*(1+block_m/wg_1_m))*dtype
        # smem_l1_io+=np.prod(grids)*(tb_m*(head_dim)*shard_seq_kv/tb_n*(tb_n/wg_1_n) + (head_dim)*tb_n*shard_seq_kv/tb_n*(1+tb_m/wg_1_m))*in_bytes
        if single_chip.support_wgmma:
            smem_l1_io+=np.prod(grids)*(tb_m*(head_dim+qk_rope_head_dim)*shard_seq_kv/tb_n + (head_dim+qk_rope_head_dim)*shard_seq_kv*(1+tb_m/wp_m))*in_bytes
        else:
            # q is in the shared memory, always
            # k is loaded continuouslys
            smem_l1_io+=np.prod(grids)*(tb_m*(head_dim+qk_rope_head_dim)*shard_seq_kv/tb_n + (head_dim+qk_rope_head_dim)*shard_seq_kv*(1+tb_m/wp_m))*in_bytes


        ddr_io=ddr_read_io
        l2_hit_rate=KV_l2_hit_rate
        l2_io=l2_read_io+l2_store_io

        smem_footprint += (head_dim+qk_rope_head_dim)*in_bytes*(tb_m + tb_n*stages)

        ret = ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        # return ddr_io*batch, l2_hit_rate, l2_io*batch, smem_footprint, smem_l1_io*batch, reg_footprint, 
        # compute_flops*batch, ddr_read_io*batch, l2_read_io*batch

        hete_post_data=hete_post_process_tensor_core_op(ret, single_chip, in_bytes)
        smem_fusion_list.append(hete_reg_fusion([hete_post_data], single_chip))
        
        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling single chip metrics: %s", get_hete_metrics(smem_fusion_post_data))

        time_2 = smem_fusion_post_data[0]
        log.info("mla flash attention v3 coarse pessimistic stage2: %s", time_2)

        return time_2
    
    def stage3():
        # softmax, s@v=p, rescale ....
        smem_fusion_list=[]
        reg_array=[]

        # smem fused stage 1
        reg_array=[]
        # max m
        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = 0,0,0,0,0,0,0,0,0

        reg_footprint+=math.ceil(tb_m/128)
        compute_flops+=np.prod(grids)*tb_m*shard_seq_kv*math.ceil(128/tb_m)*2

        # fmnmx m+ following(m-mprev)
        reg_footprint+=math.ceil(tb_m/128)
        compute_flops+=np.prod(grids)*tb_m*(shard_seq_kv/tb_n)*math.ceil(128/tb_m)*2*2

        ret = ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_cuda_core_op(ret, single_chip, accum_bytes)
        reg_array.append(post_data)   
    
        # scale = exp(m,m_prev)
        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = 0,0,0,0,0,0,0,0,0

        reg_footprint+=math.ceil(tb_m/128)
        compute_flops+=np.prod(grids)*tb_m*(shard_seq_kv/tb_n)*2

        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_sfu_core_op(ret, single_chip)
        reg_array.append(post_data)

        # o *scale
        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0
        compute_flops+=np.prod(grids)*tb_m*(shard_seq_kv/tb_n)*head_dim*2

        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_cuda_core_op(ret, single_chip, accum_bytes)
        reg_array.append(post_data)        
        
        # s=s-m
        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0
        compute_flops+=np.prod(grids)*tb_m*(shard_seq_kv)*2

        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_cuda_core_op(ret, single_chip, accum_bytes)
        reg_array.append(post_data)

        # s= exp(s-m)
        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0
        compute_flops+=np.prod(grids)*tb_m*(shard_seq_kv) * 2
    
        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_sfu_core_op(ret, single_chip)
        reg_array.append(post_data)        

        # s->s_cast fp16
        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0
        compute_flops+=np.prod(grids)*tb_m*(shard_seq_kv)*2

        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_cuda_core_op(ret, single_chip, in_bytes)
        reg_array.append(post_data)

        post_data=hete_reg_fusion(reg_array, single_chip)
        smem_fusion_list.append(post_data)   

        # smem fused stage 2, gemm p=s@v
        reg_array=[]

        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0
        # actually v cache is the same as k cache, so we can use the same l2_read_io
        # l2_read_io=np.prod(grids)*head_dim*shard_seq_kv*in_bytes



        assert hasattr(single_chip, 'ddr_wave_bytes'), "arch.ddr_wave_bytes does not exist"
        if l2_read_io > 0 and uses_dram_wave_quantization(single_chip):
            l2_read_waves = l2_read_io / single_chip.ddr_wave_bytes
            log.info("l2 read waves: %s, original l2 read io: %s", l2_read_waves, l2_read_io)
            l2_read_io = l2_read_io * math.ceil(l2_read_waves) / l2_read_waves

        l2_store_io=0
        if l2_store_io > 0 and uses_dram_wave_quantization(single_chip):
            l2_store_waves = l2_store_io / single_chip.ddr_wave_bytes
            log.info("l2 store waves: %s, original l2 store io: %s", l2_store_waves, l2_store_io)
            l2_store_io = l2_store_io * math.ceil(l2_store_waves) / l2_store_waves

        ddr_read_io+=l2_read_io*(1-KV_l2_hit_rate)
        compute_flops+=np.prod(grids)*tb_m*head_dim*shard_seq_kv*2
        # reg_footprint+=block_m*dim/128*accum_dtype/4+(wg_2_m*wg_2_n)*dtype/4/128
        reg_footprint+=(tb_m*tb_n)*accum_bytes/4/128
        ddr_io=ddr_read_io
        l2_io=l2_read_io
        # smem_l1_io+=np.prod(grids)*(block_m*seq_len*(1+block_n/warpg1_n)+seq*block_n*(1+block_m/warpg1_m))*dtype
        # smem_l1_io+=np.prod(grids)*(seq_len*block_m*(dim/wg_2_n)+seq_len*dim*(block_m/wg_2_m+1))*dtype
        smem_l1_io+=np.prod(grids)*(shard_seq_kv*head_dim*(tb_m/wp_m+1))*in_bytes
        # smem_footprint+=sw_pipe*dtype*(block_k*block_m)
        smem_footprint+=tb_m*head_dim*in_bytes*stages
        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_tensor_core_op(ret, single_chip, in_bytes)
        reg_array.append(post_data)
        
        post_data=hete_reg_fusion(reg_array, single_chip)
        smem_fusion_list.append(post_data)

        # smem fused stage 3, rescale, reduce sum
        reg_array=[]

        # l=reduce_sum(s)
        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = 0,0,0,0,0,0,0,0,0
        reg_footprint+=math.ceil(tb_m/128)
        compute_flops+=np.prod(grids)*tb_m*shard_seq_kv*math.ceil(128/tb_m)*2
        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_cuda_core_op(ret, single_chip, accum_bytes)
        reg_array.append(post_data)

        # l_sum+=scale*l_sum+l
        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0
        reg_footprint+=math.ceil(tb_m/128)
        compute_flops+=np.prod(grids)*tb_m*(shard_seq_kv/tb_n)*math.ceil(128/tb_m)*2

        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_cuda_core_op(ret, single_chip, accum_bytes)
        reg_array.append(post_data)

        # l_sum-> 1/l_sum
        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0
        compute_flops+=np.prod(grids)*tb_m*(shard_seq_kv/tb_n)*math.ceil(128/tb_m)*2
        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_sfu_core_op(ret, single_chip)
        reg_array.append(post_data)
        

        post_data=hete_reg_fusion(reg_array, single_chip)
        smem_fusion_list.append(post_data)

        


        # processiong smem fusion list
        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling single chip metrics: %s", get_hete_metrics(smem_fusion_post_data))

        time_3 = smem_fusion_post_data[0]
        log.info("mla flash attention v3 coarse pessimistic stage3: %s", time_3)

        return time_3

    def stage4():
        # store p
        smem_fusion_list=[]

        # smem fused stage 4, rescale, store p
        # o=o*(1/l_sum)
        # store o to global memory

        ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0
        compute_flops+=np.prod(grids)*tb_m*head_dim*2
        smem_l1_io+=np.prod(grids)*tb_m*head_dim*in_bytes*2
        l2_store_io=np.prod(grids)*tb_m*head_dim*in_bytes
        assert hasattr(single_chip, 'ddr_wave_bytes'), "arch.ddr_wave_bytes does not exist"
        if l2_read_io > 0 and uses_dram_wave_quantization(single_chip):
            l2_read_waves = l2_read_io / single_chip.ddr_wave_bytes
            log.info("l2 read waves: %s, original l2 read io: %s", l2_read_waves, l2_read_io)
            l2_read_io = l2_read_io * math.ceil(l2_read_waves) / l2_read_waves

        if l2_store_io > 0 and uses_dram_wave_quantization(single_chip):
            l2_store_waves = l2_store_io / single_chip.ddr_wave_bytes
            log.info("l2 store waves: %s, original l2 store io: %s", l2_store_waves, l2_store_io)
            l2_store_io = l2_store_io * math.ceil(l2_store_waves) / l2_store_waves

        ddr_io+=l2_store_io


        ret=ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        post_data=hete_post_process_cuda_core_op(ret, single_chip, in_bytes)

        smem_fusion_list.append(hete_reg_fusion([post_data], single_chip))
        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        if stats is not None:
            stats.append_hete(smem_fusion_post_data)

        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling single chip metrics: %s", get_hete_metrics(smem_fusion_post_data))

        time_4 = smem_fusion_post_data[0]
        log.info("mla flash attention v3 coarse pessimistic stage4: %s", time_4)

        return time_4


        


    time_1 = stage1()
    time_2 = stage2()
    time_3 = stage3()
    time_4 = stage4()

    return grids, time_1 + time_2 + time_3 + time_4

    # stage2, sliding k, gemm1

    # stage3, local softmax rescale, gemm2, rescale

    # stage4, store scaled p
