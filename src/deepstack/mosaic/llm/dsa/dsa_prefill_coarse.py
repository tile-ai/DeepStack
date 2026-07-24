import math
import numpy as np
import logging
from mosaic.parallelism import ParallelScheme
from mosaic.noc.noc_topo import Hierarchy
from mosaic.utils import Modeling_Granularity
from mosaic.llm_arch import LLM_Arch
from tilesight.arch import *
from mosaic.op_dtype.mla_fa_prefill_wrapper import mla_fa_prefill_wrapper
from mosaic.collectives import reduce_scatter_wrapper
from mosaic.utils import get_comp_comm_e2e_time
from mosaic.cost.op_perf_stats import OpPerfStats

from mosaic.llm.mla.mla_prefill_coarse import (
    mla_prefill_coarse_stage1,
    mla_prefill_coarse_stage2,
    mla_prefill_coarse_stage3,
    mla_prefill_coarse_stage4,
    mla_prefill_coarse_stage5_2,
    mla_prefill_coarse_stage6,
)
from .dsa_indexer_coarse import dsa_indexer_proj_coarse, dsa_indexer_score_topk_coarse

log = logging.getLogger(__name__)

# DeepSeek-V3.2 DSA prefill:
# 与 MLA prefill (no weight absorption, MHA form) 的区别:
# 1. 多一个 lightning indexer: 投影 + 对 seq x seq 的 score 计算 + topk。
#    注意 indexer 的 score GEMM 是 O(seq^2 * H_i * D_i), 没有被稀疏化,
#    但 D_i=128 远小于主注意力的 (nope+rope)*num_head, 且真实 kernel 用 fp8。
# 2. 主注意力每个 query 只 attend 被选中的 min(index_topk, seq) 个 token,
#    通过 mla_fa_prefill_wrapper 的 seq_kv 参数表达。
#    (与现有 FA prefill 建模一致, 不做 causal 折半; sparse 下每 query kv 数
#    上限为 topk, 取 seq_kv = min(index_topk, seq)。)


def dsa_sparse_fa_prefill_coarse(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats: "OpPerfStats | None" = None):
    # 与 mla_prefill_coarse_stage5_1 相同, 但有效 kv 长度截断到 index_topk
    assert model_arch.mla_arch is not None
    assert model_arch.dsa_arch is not None
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    num_head = mla_arch.num_head
    num_kv_head = mla_arch.num_kv_head
    head_dim = mla_arch.head_dim
    q_rope_head_dim = mla_arch.q_rope_head_dim
    atten_bytes = mla_arch.atten_bytes

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_num_head = math.ceil(num_head / parallel.tp)
    shard_seq_q = math.ceil(seq / atten_parallel.sp)

    seq_kv = min(model_arch.dsa_arch.index_topk, seq)

    grids, single_chip_time = mla_fa_prefill_wrapper(bs=bs, seq=seq, hidden=hidden, num_head=num_head, num_kv_head=num_kv_head, head_dim=head_dim, qk_rope_head_dim=q_rope_head_dim, parallel=parallel, atten_parallel=atten_parallel,
        atten_bytes=atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats, seq_kv=seq_kv)
    waves = np.prod(grids)/single_chip.sm_count

    if (atten_parallel.cp == 1):
        log.info("dsa sparse fa prefill overall time: %s s", single_chip_time)
        return single_chip_time
    elif (atten_parallel.cp > 1):
        # reduce-scatter along atten.cp (每个 cp rank 在本地 topk/cp 的 kv 上算 partial attention)
        cp_reduce_latency, cp_reduce_ext_max, cp_reduce_traffic = reduce_scatter_wrapper(all_reduce_op_bytes=atten_bytes, parallel=atten_parallel,
        noc_hierarchy=noc_hierarchy, granularity=granularity, dim_to_process="cp", bytes=shard_bs*shard_num_head*shard_seq_q*head_dim*in_bytes)
        if stats is not None:
            stats.append_traffic(cp_reduce_traffic, hop_time_s=cp_reduce_latency, link_time_s=cp_reduce_ext_max)

        log.info("cp_reduce_latency: %s, cp_reduce_ext_max: %s", cp_reduce_latency, cp_reduce_ext_max)

        overall_time = get_comp_comm_e2e_time(compute_time=single_chip_time, network_hop_latency=cp_reduce_latency, network_link_time=cp_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
        log.info("dsa sparse fa prefill overall time: %s s", overall_time)
        return overall_time


def dsa_mla_prefill_coarse(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    '''
    Prefill: MLA (no weight absorption, MHA form) + DSA lightning indexer + sparse FA。
    返回与 mla_prefill_coarse 相同: (time, stats)
    '''
    if granularity.dump_perf_log == True:
        stats = OpPerfStats(op_name="dsa_mla_prefill", dump_perf_log=True)
    else:
        stats = None

    assert model_arch.mla_arch is not None
    assert model_arch.dsa_arch is not None

    # ------------------------------------- MLA 投影 stage -------------------------------------
    time_stage1 = mla_prefill_coarse_stage1(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_stage2 = mla_prefill_coarse_stage2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_stage3 = mla_prefill_coarse_stage3(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_stage4 = mla_prefill_coarse_stage4(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # ------------------------------------- DSA indexer -------------------------------------
    time_indexer_proj = dsa_indexer_proj_coarse(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_indexer_score = dsa_indexer_score_topk_coarse(bs=bs, seq=seq, kv_len=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    # ------------------------------------- sparse FA + 输出投影 -------------------------------------
    time_stage5_1 = dsa_sparse_fa_prefill_coarse(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_stage5_2 = mla_prefill_coarse_stage5_2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_stage6 = mla_prefill_coarse_stage6(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    log.info("dsa mla prefill time stage1:%s", time_stage1)
    log.info("dsa mla prefill time stage2:%s", time_stage2)
    log.info("dsa mla prefill time stage3:%s", time_stage3)
    log.info("dsa mla prefill time stage4:%s", time_stage4)
    log.info("dsa mla prefill time indexer proj:%s", time_indexer_proj)
    log.info("dsa mla prefill time indexer score+topk:%s", time_indexer_score)
    log.info("dsa mla prefill time stage5_1 (sparse fa, seq_kv=%s):%s", min(model_arch.dsa_arch.index_topk, seq), time_stage5_1)
    log.info("dsa mla prefill time stage5_2:%s", time_stage5_2)
    log.info("dsa mla prefill time stage6:%s", time_stage6)

    time = time_stage1 + time_stage2 + time_stage3 + time_stage4 + time_indexer_proj + time_indexer_score + time_stage5_1 + time_stage5_2 + time_stage6
    if stats is not None:
        stats.finalize(total_time_s=time, h=noc_hierarchy, arch=single_chip)

    return time, stats
