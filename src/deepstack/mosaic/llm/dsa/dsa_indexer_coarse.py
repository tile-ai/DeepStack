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
# 计算图 (H_i = index_n_heads, D_i = index_head_dim, T = kv 长度):
#   q_i = q_resid @ Wq_b           [bs, seq, q_down_hidden] @ [q_down_hidden, H_i*D_i]
#   k_i = k_norm(x @ Wk)           [bs, seq, hidden] @ [hidden, D_i]
#   rope(q_i[:rope_dim]), rope(k_i[:rope_dim])
#   w   = x @ W_weights            [bs, seq, hidden] @ [hidden, H_i]
#   scores = relu(q_i @ k_cache^T) [bs, seq, H_i, T]   <- 主要开销: 读整个 index-k cache
#   index_scores = w @ scores      [bs, seq, T] (fp32, 对 head 加权求和)
#   topk(index_scores) -> indices  [bs, seq, topk]
#
# 并行假设 (在 coarse 粒度下):
# - indexer 在 tp 维度上完整复制 (不切 head), 与参考实现一致: topk indices 必须在
#   所有 tp rank 上一致 (主注意力是 MQA 形式, 共享同一份被 gather 的 kv)。
#   * decode 下 score 计算是 memory-bound (瓶颈是读 index-k cache, 而 k cache 是
#     单"head"的, 所有 index head 共享), 切 head 不省读流量, 复制基本零代价;
#   * prefill 下 score 计算是 compute-bound, 复制会让 indexer flops 不随 tp 下降
#     (切 head + 对 [bs, seq, T] fp32 score 做 tp all-reduce 是潜在的优化方案,
#     本模型暂不建模) —— 这也是 DSA 模型部署偏向 DP attention 的原因之一。
#   indexer 权重很小 (~25M 参数), 复制的存储代价可以接受。
# - index-k cache 与主 kv cache 一样沿 atten_parallel.cp 切分; 每个 cp rank 在本地
#   score 上做 local topk (近似: 每 rank 选 topk/cp, 免去跨 rank 的 topk merge 通信)。
# - dp 切 bs, sp 切 seq, 与其他 stage 一致。
#
# Per-op 延迟地板 (INDEXER_OP_LATENCY_S):
#   indexer 每层是一条串行小 kernel 链 (wq_b -> wk -> weights_proj -> rope_q -> rope_k
#   -> score -> topk)。在 decode + 小 batch 下这些 kernel 单个的 compute/带宽都极小
#   (~µs),真正的成本是每个 kernel 的固定下限:launch dispatch + grid ramp-up +
#   首次访存延迟 + 依赖链上的 stall (下一个 kernel 必须等上一个的输出就绪)。coarse
#   模型只累加了 compute/带宽,完全没有这个下限,于是把 indexer 算得太便宜
#   (实测 DSA 给 decode 加 ~9ms/步 ≈ ~150µs/层,几乎与 batch 无关;模型只给 ~1-6ms)。
#   这里给每个 indexer op 套 max(compute, INDEXER_OP_LATENCY_S):小 batch 被地板托住
#   (≈ 7 op × floor),大 batch 下 compute 超过地板就回到原值。这是物理下限,不是为
#   了拟合实测曲线而设;精确值应来自 B200 上对这些小 kernel 的 micro-benchmark,
#   这里取一个保守的小 kernel 串行延迟下限。
INDEXER_OP_LATENCY_S = 20e-6


def dsa_indexer_proj_coarse(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    '''
    indexer 中与 kv 长度无关的部分: wq_b / wk(+k_norm) / weights_proj 三个投影 + q/k rope。
    prefill 与 decode 共用 (decode 时 seq=1)。
    '''
    assert model_arch.dsa_arch is not None
    assert model_arch.mla_arch is not None
    dsa_arch = model_arch.dsa_arch
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    q_down_hidden = mla_arch.q_down_hidden
    index_n_heads = dsa_arch.index_n_heads
    index_head_dim = dsa_arch.index_head_dim
    # indexer 的 rope 维度与主注意力的 qk_rope_head_dim 一致 (DeepSeek-V3.2: 64)
    rope_head_dim = mla_arch.kv_rope_head_dim
    atten_bytes = dsa_arch.atten_bytes

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    # fsdp: dp-level 重建 indexer 权重 (wq_b + wk + weights_proj, tp 上复制所以是全量权重)
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

    # 1. q_i = q_resid @ Wq_b -> [bs/dp, seq/sp, H_i*D_i]; q_resid 复用主注意力 stage1 的 q_a 输出
    smem_fusion_list = []
    hete_post_data_wq_b, smem_post_wq_b, tiling_wq_b = gemm_wrapper(M=shard_bs*shard_seq, N=index_n_heads*index_head_dim, K=q_down_hidden, gemm_bytes=gemm_bytes, granularity=granularity, single_chip=single_chip)
    log.info("dsa indexer wq_b M,N,K,tiling config: %s, %s, %s, %s", shard_bs*shard_seq, index_n_heads*index_head_dim, q_down_hidden, tiling_wq_b)
    grids = [shard_bs * shard_seq / tiling_wq_b[0], index_n_heads*index_head_dim / tiling_wq_b[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_wq_b], single_chip))
    smem_fusion_post_data = hete_smem_fusion(smem_fusion_list, grids, single_chip)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    time_wq_b = smem_fusion_post_data[0]

    # 2. k_i = k_norm(x @ Wk) -> [bs/dp, seq/sp, D_i]; k_norm (LayerNorm) 折叠进 epilogue, 开销可忽略
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

    # 4. rope(q_i), rope(k_i): 只旋转每个 head 的前 rope_head_dim 维
    rope_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
        input2=Tensor_Loc(torch.float16, 'ddr'),
        output=Tensor_Loc(atten_bytes.input1.dtype, 'smem'),
    )
    time_rope_q = ropeq_coarse(bs=bs, head=index_n_heads, seq=seq, head_dim=rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_rope_k = ropeq_coarse(bs=bs, head=1, seq=seq, head_dim=rope_head_dim, parallel=parallel, next_parallel=atten_parallel, rope_bytes=rope_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # 每个 op 套 per-kernel 延迟地板 (见文件头 INDEXER_OP_LATENCY_S 说明)
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
    '''
    indexer 中与 kv 长度有关的部分: score 计算 (fused kernel) + topk 选择。
    kv_len: decode 时为 cached_kv, prefill 时为 seq。

    score 部分对应真实实现里的 fp8_index fused kernel:
      scores = relu(q_i @ k_cache^T) -> 加权 head 求和 -> index_scores [bs, seq, T] fp32
    这里用一个 batched GEMM 建模 (M=seq*H_i, N=T, K=D_i, batch=bs), relu / 加权求和的
    flops 是 GEMM 的 1/D_i (~1%), 忽略; 中间 scores 留在片上 (output loc=smem), 只有
    index_scores [bs, seq, T] fp32 落 DDR, 由 topk pass 统一计费。
    '''
    assert model_arch.dsa_arch is not None
    dsa_arch = model_arch.dsa_arch

    index_n_heads = dsa_arch.index_n_heads
    index_head_dim = dsa_arch.index_head_dim
    atten_bytes = dsa_arch.atten_bytes

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_kv = math.ceil(kv_len / atten_parallel.cp)

    # 1. fused score kernel: q [bs, seq*H_i, D_i] @ k_cache^T [D_i, T] -> scores (片上)
    score_gemm_bytes = OpBytes(
        input1=Tensor_Loc(atten_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(atten_bytes.input2.dtype, 'ddr'),   # index-k cache, 每 token D_i
        output=Tensor_Loc(torch.float32, 'smem'),             # scores 不落 DDR
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

    # 2. topk pass: 对 index_scores [bs, seq, T] fp32 的多轮 radix-select 读写。
    #    校准 (2026-06-11, RTX PRO 6000 Blackwell, torch.topk/CUB radix select, k=2048):
    #    带宽受限区 (rows*T 足够大) 实测时间 ≈ 2.6~2.9x 单次(读+写) pass,
    #    即等效 ~5.6 个单向 pass; 这里取 TOPK_RW_PASSES=3 组读写 (6 pass, 偏保守 ~7%)。
    #    注意: (a) 优化过的 fused topk kernel (如 sglang) 可能低于此值;
    #    (b) 小工作量时 (rows<256 且 T<=8k) 有 ~40-60us 的 launch/占用率下限, 未建模。
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

    # per-kernel 延迟地板 (见文件头 INDEXER_OP_LATENCY_S 说明)
    time_score = max(time_score, INDEXER_OP_LATENCY_S)
    time_topk = max(time_topk, INDEXER_OP_LATENCY_S)
    log.info("dsa indexer score time: %s, topk time: %s (kv_len: %s)", time_score, time_topk, kv_len)
    return time_score + time_topk
