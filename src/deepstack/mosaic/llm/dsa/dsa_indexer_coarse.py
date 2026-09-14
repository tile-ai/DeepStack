import torch
import math
import numpy as np
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
from mosaic.op_dtype.element_wrapper import element_wrapper
from mosaic.llm_arch import LLM_Arch
from tilesight.arch import *
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
import logging
log = logging.getLogger(__name__)

from mosaic.llm.rope.ropeq_coarse import ropeq_coarse
from mosaic.cost.op_perf_stats import OpPerfStats

# DeepSeek Sparse Attention (DSA) lightning indexer, see
# mosaic/llm_arch/transformers_modeling/modeling_deepseek_v32.py: DeepseekV32Indexer
#
#   q_i = q_resid @ Wq_b           [bs, seq, q_down_hidden] @ [q_down_hidden, H_i*D_i]
#   k_i = k_norm(x @ Wk)           [bs, seq, hidden] @ [hidden, D_i]
#   rope(q_i[:rope_dim]), rope(k_i[:rope_dim])
#   w   = x @ W_weights            [bs, seq, hidden] @ [hidden, H_i]
#   topk(index_scores) -> indices  [bs, seq, topk]
#
#
INDEXER_OP_LATENCY_S = 20e-6


def dsa_indexer_proj_coarse(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    """Model KV-length-independent indexer work: wq_b, wk with key normalization,
    weights projection, and query/key RoPE. Shared by prefill and decode (seq=1).
    """
    assert model_arch.dsa_arch is not None
    assert model_arch.mla_arch is not None
    dsa_arch = model_arch.dsa_arch
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    q_down_hidden = mla_arch.q_down_hidden
    index_n_heads = dsa_arch.index_n_heads
    index_head_dim = dsa_arch.index_head_dim
    rope_head_dim = mla_arch.kv_rope_head_dim
    atten_bytes = dsa_arch.atten_bytes

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    if (parallel.fsdp == False or parallel.dp == 1):
        hop_time_w, ext_max_w, noc_overall_time_w = 0, 0, 0
    elif parallel.fsdp == True:
        tm = TrafficMatrix(parallel.world_size())
        indexer_weight_numel = q_down_hidden * index_n_heads * index_head_dim + hidden * index_head_dim + hidden * index_n_heads
        bytes_each_pair = indexer_weight_numel * weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1)
        tm.add_intra_group_traffic("dp", bytes_each_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        hop_time_w, ext_max_w, noc_overall_time_w, noc_traffic_w = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_w, hop_time_s=hop_time_w, link_time_s=ext_max_w)
        log.info("dsa indexer proj fsdp, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", hop_time_w, ext_max_w, noc_overall_time_w)

    gemm_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(atten_bytes.output.dtype, 'ddr'),
    )

    smem_fusion_list = []
    hete_post_data_wq_b, smem_post_wq_b, tiling_wq_b = gemm_wrapper(M=shard_bs*shard_seq, N=index_n_heads*index_head_dim, K=q_down_hidden, gemm_bytes=gemm_bytes, granularity=granularity, single_chip=single_chip)
    log.info("dsa indexer wq_b M,N,K,tiling config: %s, %s, %s, %s", shard_bs*shard_seq, index_n_heads*index_head_dim, q_down_hidden, tiling_wq_b)
    grids = [shard_bs * shard_seq / tiling_wq_b[0], index_n_heads*index_head_dim / tiling_wq_b[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_wq_b], single_chip))
    smem_fusion_post_data = hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_wq_b = smem_fusion_post_data[0]

    smem_fusion_list = []
    hete_post_data_wk, smem_post_wk, tiling_wk = gemm_wrapper(M=shard_bs*shard_seq, N=index_head_dim, K=hidden, gemm_bytes=gemm_bytes, granularity=granularity, single_chip=single_chip)
    log.info("dsa indexer wk M,N,K,tiling config: %s, %s, %s, %s", shard_bs*shard_seq, index_head_dim, hidden, tiling_wk)
    grids = [shard_bs * shard_seq / tiling_wk[0], index_head_dim / tiling_wk[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_wk], single_chip))
    smem_fusion_post_data = hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_wk = smem_fusion_post_data[0]

    # 3. w = x @ W_weights -> [bs/dp, seq/sp, H_i]
    smem_fusion_list = []
    hete_post_data_wp, smem_post_wp, tiling_wp = gemm_wrapper(M=shard_bs*shard_seq, N=index_n_heads, K=hidden, gemm_bytes=gemm_bytes, granularity=granularity, single_chip=single_chip)
    log.info("dsa indexer weights_proj M,N,K,tiling config: %s, %s, %s, %s", shard_bs*shard_seq, index_n_heads, hidden, tiling_wp)
    grids = [shard_bs * shard_seq / tiling_wp[0], max(index_n_heads / tiling_wp[1], 1)]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_wp], single_chip))
    smem_fusion_post_data = hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_wp = smem_fusion_post_data[0]

    rope_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
        input2=Tensor_Loc(torch.float16, 'ddr'),
        output=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
    )
    time_rope_q = ropeq_coarse(bs=bs, head=index_n_heads, seq=seq, head_dim=rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_rope_k = ropeq_coarse(bs=bs, head=1, seq=seq, head_dim=rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    L = INDEXER_OP_LATENCY_S
    time_wq_b = max(time_wq_b, L)
    time_wk = max(time_wk, L)
    time_wp = max(time_wp, L)
    time_rope_q = max(time_rope_q, L)
    time_rope_k = max(time_rope_k, L)
    time_comp = time_wq_b + time_wk + time_wp + time_rope_q + time_rope_k

    if (granularity.get_comp_comm_overlap() == True):
        additional_time = max(time_comp, noc_overall_time_w) - time_comp
    else:
        additional_time = noc_overall_time_w

    time_proj = time_comp + additional_time
    log.info("dsa indexer proj time: wq_b %s, wk %s, weights_proj %s, rope_q %s, rope_k %s, fsdp additional %s", time_wq_b, time_wk, time_wp, time_rope_q, time_rope_k, additional_time)
    return time_proj


def dsa_indexer_score_topk_coarse(bs:int, seq:int, kv_len:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    """Model KV-dependent indexer score computation and top-k selection.
    kv_len is cached_kv during decode and seq during prefill.
    The score path corresponds to the fused fp8_index kernel: ReLU of q_i @ k_cache.T,
    followed by a weighted head reduction to fp32 index_scores [bs, seq, T].
    Use a batched GEMM with M=seq*H_i, N=T, K=D_i, batch=bs. ReLU and reduction work
    is about 1/D_i of GEMM work and is ignored. Intermediate scores remain in SMEM;
    only fp32 index_scores reach DDR, accounted for by the top-k pass.
    """
    assert model_arch.dsa_arch is not None
    dsa_arch = model_arch.dsa_arch

    index_n_heads = dsa_arch.index_n_heads
    index_head_dim = dsa_arch.index_head_dim
    atten_bytes = dsa_arch.atten_bytes

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_kv = math.ceil(kv_len / atten_parallel.cp)

    score_gemm_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(torch.float32, 'smem'),
    )

    smem_fusion_list = []
    hete_post_data_score, smem_post_score, tiling_score = gemm_wrapper(M=shard_seq_q*index_n_heads, N=shard_kv, K=index_head_dim, gemm_bytes=score_gemm_bytes, granularity=granularity, single_chip=single_chip, batch=shard_bs)
    log.info("dsa indexer score M,N,K,batch,tiling config: %s, %s, %s, %s, %s", shard_seq_q*index_n_heads, shard_kv, index_head_dim, shard_bs, tiling_score)
    grids = [shard_seq_q*index_n_heads / tiling_score[0], shard_kv / tiling_score[1], shard_bs]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_score], single_chip))
    smem_fusion_post_data = hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_score = smem_fusion_post_data[0]

    TOPK_RW_PASSES = 3
    topk_bytes = OpBytes(
        input1=Tensor_Loc(torch.float32, 'ddr', [TOPK_RW_PASSES * shard_bs, shard_seq_q, shard_kv]),
        input2=None,
        output=Tensor_Loc(torch.float32, 'ddr', [TOPK_RW_PASSES * shard_bs, shard_seq_q, shard_kv]),
    )
    smem_fusion_list = []
    hete_post_data_topk, tb_shape_topk = element_wrapper(element_op_bytes=topk_bytes, granularity=granularity, single_chip=single_chip, batch=1, type="cuda_core", tb_tiling_config=None)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_topk], single_chip))
    grids = [TOPK_RW_PASSES*shard_bs/tb_shape_topk[0], shard_seq_q/tb_shape_topk[1], shard_kv/tb_shape_topk[2]]
    smem_fusion_post_data = hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_topk = smem_fusion_post_data[0]

    time_score = max(time_score, INDEXER_OP_LATENCY_S)
    time_topk = max(time_topk, INDEXER_OP_LATENCY_S)
    log.info("dsa indexer score time: %s, topk time: %s (kv_len: %s)", time_score, time_topk, kv_len)
    return time_score + time_topk
