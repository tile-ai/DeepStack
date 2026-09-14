import math
import numpy as np
import logging
from mosaic.parallelism import ParallelScheme
from mosaic.llm_arch import LLM_Arch
from mosaic.llm.mla.mla_footprint import (
    get_mla_absorb_and_no_absorb_footprint,
    get_mla_no_absorb_footprint,
)

log = logging.getLogger(__name__)


def get_dsa_indexer_footprint(bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme):
    """Estimate the DSA indexer's additional footprint relative to MLA.
    Return (max_activation, mem_weight, mem_index_cache).
    Indexer weights are replicated across TP. The index-key cache holds one D_i
    vector per token, sharded across CP and replicated across TP.
    """
    assert model_arch.dsa_arch is not None
    assert model_arch.mla_arch is not None
    dsa_arch = model_arch.dsa_arch
    mla_arch = model_arch.mla_arch

    hidden = model_arch.hidden_size
    q_down_hidden = mla_arch.q_down_hidden
    index_n_heads = dsa_arch.index_n_heads
    index_head_dim = dsa_arch.index_head_dim
    index_topk = dsa_arch.index_topk
    atten_bytes = dsa_arch.atten_bytes

    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)

    # Weights (replicated across tp): wq_b + wk + weights_proj (+ k_norm, negligible)
    wq_b_weight = q_down_hidden * index_n_heads * index_head_dim * weight_bytes
    wk_weight = hidden * index_head_dim * weight_bytes
    weights_proj_weight = hidden * index_n_heads * weight_bytes
    mem_weight = wq_b_weight + wk_weight + weights_proj_weight

    dp_divisor = np.uint64(parallel.dp) if parallel.fsdp else np.uint64(1)
    mem_weight = mem_weight // dp_divisor

    # index-k cache (sharded along cp)
    mem_index_cache = in_bytes * shard_bs * shard_cached_kv * index_head_dim

    # activation: q_i + index_scores (fp32) + topk indices (int32)
    q_act = in_bytes * shard_bs * shard_seq_q * index_n_heads * index_head_dim
    index_scores_act = 4 * shard_bs * shard_seq_q * shard_cached_kv
    topk_indices_act = 4 * shard_bs * shard_seq_q * min(index_topk, shard_cached_kv)
    max_activation = max(q_act, index_scores_act + topk_indices_act)

    return max_activation, mem_weight, mem_index_cache


def get_dsa_mla_absorb_and_no_absorb_footprint(bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme):
    """Estimate decode memory for MLA with both absorbed and unabsorbed weights plus
    the DSA indexer. Match get_mla_absorb_and_no_absorb_footprint; the KV-cache
    component includes the index-key cache.
    """
    mla_max_activation, mla_mem_weight, mla_mem_kv_cache = get_mla_absorb_and_no_absorb_footprint(bs, seq, cached_kv, model_arch, parallel, atten_parallel)
    idx_max_activation, idx_mem_weight, idx_mem_index_cache = get_dsa_indexer_footprint(bs, seq, cached_kv, model_arch, parallel, atten_parallel)

    max_activation = max(mla_max_activation, idx_max_activation)
    mem_weight = mla_mem_weight + idx_mem_weight
    mem_kv_cache = mla_mem_kv_cache + idx_mem_index_cache

    return max_activation, mem_weight, mem_kv_cache


def get_dsa_mla_no_absorb_footprint(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme):
    """Estimate prefill memory for unabsorbed MLA plus the DSA indexer."""
    mla_max_activation, mla_mem_weight, mla_mem_kv_cache = get_mla_no_absorb_footprint(bs, seq, model_arch, parallel, atten_parallel)
    idx_max_activation, idx_mem_weight, idx_mem_index_cache = get_dsa_indexer_footprint(bs, seq, seq, model_arch, parallel, atten_parallel)

    max_activation = max(mla_max_activation, idx_max_activation)
    mem_weight = mla_mem_weight + idx_mem_weight
    mem_kv_cache = mla_mem_kv_cache + idx_mem_index_cache

    return max_activation, mem_weight, mem_kv_cache
