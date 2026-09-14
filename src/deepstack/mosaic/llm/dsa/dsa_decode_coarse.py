import math
import logging
from mosaic.parallelism import ParallelScheme
from mosaic.noc.noc_topo import Hierarchy
from mosaic.utils import Modeling_Granularity
from mosaic.llm_arch import LLM_Arch
from tilesight.arch import *
from mosaic.cost.op_perf_stats import OpPerfStats

from mosaic.llm.mla.mla_decode_coarse import (
    mla_decode_coarse_stage1,
    mla_decode_coarse_stage2,
    mla_decode_coarse_stage3,
    mla_decode_coarse_stage4_1,
    mla_decode_coarse_stage4_2,
    mla_decode_coarse_stage5,
    mla_decode_coarse_stage6,
)
from .dsa_indexer_coarse import dsa_indexer_proj_coarse, dsa_indexer_score_topk_coarse

log = logging.getLogger(__name__)

# DeepSeek-V3.2 DSA decode:
# The only differences from MLA decode:
# 1. An additional lightning indexer (projections + score/topk over all cached_kv)
# 2. Main attention (MQA, weight absorption) attends only to the selected topk tokens,
#    so the effective kv length in stage4_1 = min(index_topk, cached_kv).
#    The gathered kv involves random access, approximated as contiguous access at coarse granularity.
# Indexer score computation still reads the entire index-k cache (cached_kv * D_i),
# which is how DSA reduces O(T) read traffic from 576B/token (c_kv) to ~128B/token (index-k).


def dsa_mla_decode_kv_list_coarse(bs:int, seq:int, cached_kv_list:list[int], model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    """Model absorbed-weight MLA decode in MQA form plus the DSA lightning indexer.
    Return (overall_time_list, stats_list), as in mla_decode_kv_list_coarse.
    """
    assert granularity.get_mode() == "coarse"
    assert parallel.ep == 1

    if seq == 1:
        assert parallel.sp == 1

    assert model_arch.mla_arch is not None
    assert model_arch.dsa_arch is not None
    index_topk = model_arch.dsa_arch.index_topk

    if granularity.dump_perf_log:
        _base_stats = OpPerfStats(op_name="_base")
    else:
        _base_stats = None

    # ------------------------------------- MLA: kv-independent stages -------------------------------------
    time_stage1 = mla_decode_coarse_stage1(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    time_stage2 = mla_decode_coarse_stage2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    time_stage3 = mla_decode_coarse_stage3(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    time_stage4_2 = mla_decode_coarse_stage4_2(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    time_stage5 = mla_decode_coarse_stage5(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)
    time_stage6 = mla_decode_coarse_stage6(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)

    # ------------------------------------- DSA indexer: kv-independent projections -------------------------------------
    time_indexer_proj = dsa_indexer_proj_coarse(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=_base_stats)

    overall_time_list = []
    stats_list = [] if granularity.dump_perf_log else None

    for cached_kv in cached_kv_list:
        stats = OpPerfStats(op_name=f"dsa_mla_decode_bs{bs}_seq{seq}_kv{cached_kv}", dump_perf_log=granularity.dump_perf_log) if granularity.dump_perf_log else None
        if stats is not None:
            stats.absorb(_base_stats)

        # indexer score + topk: scan the entire index-k cache
        time_indexer_score = dsa_indexer_score_topk_coarse(bs=bs, seq=seq, kv_len=cached_kv, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

        # Main attention: attend only to the selected topk tokens
        effective_kv = min(index_topk, cached_kv)
        time_stage4_1 = mla_decode_coarse_stage4_1(bs=bs, seq=seq, cached_kv=effective_kv, model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

        overall_time = time_stage1 + time_stage2 + time_stage3 + time_indexer_proj + time_indexer_score + time_stage4_1 + time_stage4_2 + time_stage5 + time_stage6
        log.info("kv length: %s, effective kv (topk): %s, over all time: %s", cached_kv, effective_kv, overall_time)
        log.info("dsa mla decode coarse time: stage 1: %s", time_stage1)
        log.info("dsa mla decode coarse time: stage 2: %s", time_stage2)
        log.info("dsa mla decode coarse time: stage 3: %s", time_stage3)
        log.info("dsa mla decode coarse time: indexer proj: %s", time_indexer_proj)
        log.info("dsa mla decode coarse time: indexer score+topk: %s", time_indexer_score)
        log.info("dsa mla decode coarse time: stage 4_1 (sparse mqa decode, kv=%s): %s", effective_kv, time_stage4_1)
        log.info("dsa mla decode coarse time: stage 4_2 (rescale + reshape): %s", time_stage4_2)
        log.info("dsa mla decode coarse time: stage 5: %s", time_stage5)
        log.info("dsa mla decode coarse time: stage 6: %s", time_stage6)

        if granularity.dump_perf_log:
            stats.finalize(total_time_s=overall_time, h=noc_hierarchy, arch=single_chip)
            stats_list.append(stats)
        overall_time_list.append(overall_time)

    return overall_time_list, stats_list


def dsa_mla_decode_coarse(bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    """Convenience entry point for one cached_kv value; return a scalar time."""
    overall_time_list, _ = dsa_mla_decode_kv_list_coarse(bs=bs, seq=seq, cached_kv_list=[cached_kv], model_arch=model_arch, parallel=parallel, atten_parallel=atten_parallel, next_parallel=next_parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    return overall_time_list[0]
