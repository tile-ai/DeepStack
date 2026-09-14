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
# Computation graph (H_i = index_n_heads, D_i = index_head_dim, T = kv length):
#   q_i = q_resid @ Wq_b           [bs, seq, q_down_hidden] @ [q_down_hidden, H_i*D_i]
#   k_i = k_norm(x @ Wk)           [bs, seq, hidden] @ [hidden, D_i]
#   rope(q_i[:rope_dim]), rope(k_i[:rope_dim])
#   w   = x @ W_weights            [bs, seq, hidden] @ [hidden, H_i]
#   scores = relu(q_i @ k_cache^T) [bs, seq, H_i, T]   <- main cost: reading the entire index-k cache
#   index_scores = w @ scores      [bs, seq, T] (fp32, weighted sum over heads)
#   topk(index_scores) -> indices  [bs, seq, topk]
#
# Parallelism assumptions (at coarse granularity):
# - The indexer is fully replicated across tp (heads are not sharded), matching the reference implementation: topk indices must match
#   across all tp ranks (main attention uses MQA and shares the same gathered kv).
#   * During decode, score computation is memory-bound (reading the index-k cache is the bottleneck, and the k cache has
#     a single "head" shared by all index heads); sharding heads saves no read traffic, so replication is nearly free;
#   * During prefill, score computation is compute-bound, so replication prevents indexer FLOPs from decreasing with tp
#     (sharding heads + a tp all-reduce on [bs, seq, T] fp32 scores is a potential optimization,
#     currently not modeled here) -- another reason DSA deployments favor DP attention.
#   Indexer weights are small (~25M parameters), making the memory cost of replication acceptable.
# - Like the main kv cache, the index-k cache is sharded along atten_parallel.cp; each cp rank performs local topk
#   on its local scores (approximation: select topk/cp per rank to avoid communication for a cross-rank topk merge).
# - dp shards bs and sp shards seq, as in other stages.
#
# Per-op latency floor (INDEXER_OP_LATENCY_S):
#   The indexer in each layer is a serial chain of small kernels (wq_b -> wk -> weights_proj -> rope_q -> rope_k
#   -> score -> topk). During decode with small batches, each kernel's compute/bandwidth time is tiny
#   (~µs); the real cost is each kernel's fixed latency floor: launch dispatch + grid ramp-up +
#   initial memory-access latency + dependency-chain stalls (the next kernel must wait for the previous output). The coarse
#   model sums only compute/bandwidth time and omits this floor, substantially underestimating indexer cost
#   (measured DSA adds ~9ms/step to decode, or ~150µs/layer, almost independent of batch size; the model gives only ~1-6ms).
#   Apply max(compute, INDEXER_OP_LATENCY_S) to each indexer op: small batches are bounded by the floor
#   (approximately 7 ops × floor), while large batches retain their original compute time once it exceeds the floor. This is a physical lower bound,
#   not a parameter fitted to measured curves; its precise value should come from microbenchmarks of these small kernels on B200.
#   Here we use a conservative lower bound on serial latency for small kernels.
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
    # The indexer's rope dimension matches qk_rope_head_dim in main attention (DeepSeek-V3.2: 64)
    rope_head_dim = mla_arch.kv_rope_head_dim
    atten_bytes = dsa_arch.atten_bytes

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    # fsdp: reconstruct indexer weights at dp level (wq_b + wk + weights_proj; full weights because they are replicated across tp)
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

    # 1. q_i = q_resid @ Wq_b -> [bs/dp, seq/sp, H_i*D_i]; q_resid reuses the q_a output from main attention stage1
    smem_fusion_list = []
    hete_post_data_wq_b, smem_post_wq_b, tiling_wq_b = gemm_wrapper(M=shard_bs*shard_seq, N=index_n_heads*index_head_dim, K=q_down_hidden, gemm_bytes=gemm_bytes, granularity=granularity, single_chip=single_chip)
    log.info("dsa indexer wq_b M,N,K,tiling config: %s, %s, %s, %s", shard_bs*shard_seq, index_n_heads*index_head_dim, q_down_hidden, tiling_wq_b)
    grids = [shard_bs * shard_seq / tiling_wq_b[0], index_n_heads*index_head_dim / tiling_wq_b[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_wq_b], single_chip))
    smem_fusion_post_data = hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_wq_b = smem_fusion_post_data[0]

    # 2. k_i = k_norm(x @ Wk) -> [bs/dp, seq/sp, D_i]; k_norm (LayerNorm) is folded into the epilogue with negligible cost
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

    # 4. rope(q_i), rope(k_i): rotate only the first rope_head_dim dimensions of each head
    rope_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
        input2=Tensor_Loc(torch.float16, 'ddr'),
        output=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
    )
    time_rope_q = ropeq_coarse(bs=bs, head=index_n_heads, seq=seq, head_dim=rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_rope_k = ropeq_coarse(bs=bs, head=1, seq=seq, head_dim=rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # Apply a per-kernel latency floor to each op (see INDEXER_OP_LATENCY_S at the top of this file)
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

    # 1. fused score kernel: q [bs, seq*H_i, D_i] @ k_cache^T [D_i, T] -> scores (on-chip)
    score_gemm_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),   # index-k cache, D_i per token
        output=Tensor_Loc(torch.float32, 'smem'),             # scores are not written to DDR
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

    # 2. topk pass: multiple radix-select read/write passes over fp32 index_scores [bs, seq, T].
    #    Calibration (2026-06-11, RTX PRO 6000 Blackwell, torch.topk/CUB radix select, k=2048):
    #    In the bandwidth-bound regime (sufficiently large rows*T), measured time is approximately 2.6~2.9x a single (read+write) pass,
    #    equivalent to ~5.6 one-way passes; use TOPK_RW_PASSES=3 read/write pairs here (6 passes, conservative by ~7%).
    #    Note: (a) optimized fused topk kernels (e.g., sglang) may fall below this value;
    #    (b) small workloads (rows<256 and T<=8k) have a ~40-60us launch/occupancy floor, which is not modeled.
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

    # Per-kernel latency floor (see INDEXER_OP_LATENCY_S at the top of this file)
    time_score = max(time_score, INDEXER_OP_LATENCY_S)
    time_topk = max(time_topk, INDEXER_OP_LATENCY_S)
    log.info("dsa indexer score time: %s, topk time: %s (kv_len: %s)", time_score, time_topk, kv_len)
    return time_score + time_topk
