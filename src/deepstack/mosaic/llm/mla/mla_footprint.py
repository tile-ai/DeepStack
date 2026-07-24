import torch
import math
import numpy as np
from mosaic.parallelism import ParallelScheme
import cProfile
import logging
from mosaic.llm_arch import MLA_Arch, GQA_Arch, MoE_Arch, Dense_FFN_Arch, LLM_Arch

log = logging.getLogger(__name__) 

def get_mla_no_absorb_footprint(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme):
    '''
    Prefill:
    no weight absorption footprint
    '''
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


    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    shard_head = math.ceil(num_head / parallel.tp)
    shard_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    shard_seq_kv = math.ceil(seq / atten_parallel.cp)

    # prefill footprint: no absorption
    # weight bytes:
    q_a_weight = hidden * q_down_hidden * weight_bytes
    q_b_weight = q_down_hidden * shard_head*(q_nope_head_dim+q_rope_head_dim) * weight_bytes
    kv_down_weight = hidden * (kv_rope_head_dim+kv_nope_head_dim) * weight_bytes
    kv_up_weight = kv_nope_head_dim * kv_nope_up_hidden/parallel.tp * weight_bytes
    o_weight = num_head*head_dim/parallel.tp * hidden * weight_bytes
    
    
    input_act = in_bytes * shard_bs *shard_seq * hidden
    q_a_act = in_bytes * shard_bs * shard_seq_q * q_down_hidden
    q_b_act = in_bytes * shard_bs * shard_seq_q * shard_head*(q_nope_head_dim+q_rope_head_dim)
    q_rope_nope_act = q_b_act
    q_act = q_b_act
    kv_down_act = in_bytes * shard_bs *shard_seq_kv * (kv_rope_head_dim+kv_nope_head_dim)
    kv_up_act = in_bytes * shard_bs *shard_seq_kv * kv_nope_up_hidden/parallel.tp
    k_act = in_bytes * shard_bs *shard_seq_kv *shard_kv_head*  (head_dim+kv_rope_head_dim)
    v_act = in_bytes * shard_bs *shard_seq_kv *shard_kv_head*  (head_dim)
    o_act = in_bytes * shard_bs * shard_seq * num_head*head_dim/parallel.tp
    
    mem_weight = q_a_weight + q_b_weight + kv_down_weight + kv_up_weight + o_weight
    mem_kv_cache = k_act + v_act
    max_activation = max((input_act+q_a_act+kv_down_act),(q_a_act+q_b_act ), (kv_down_act+kv_up_act), (q_act+k_act+v_act),2 * o_act)
    # log.info("max_activation candidates: %s, %s, %s, %s",( q_act + k_act + v_act + max(p_act,input_act)), (p_act + p_reduce_act) , (p_reduce_act + o_act), (o_act + o_reduce_act) )
    dp_divisor = np.uint64(parallel.dp) if parallel.fsdp else np.uint64(1)

    mem_weight = mem_weight // dp_divisor

    return max_activation, mem_weight, mem_kv_cache

def get_mla_absorb_footprint(bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme):
    '''
    Decode:
    weight absorption footprint,
    '''
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
    # Wkv_up_a_trans [num_head/tp, head_dim, kv_nope_head_dim]
    # Wkv_down [hidden, (kv_rope_head_dim+kv_nope_head_dim)]
    # Wkv_up_b_trans [num_head/tp, kv_nope_head_dim, head_dim] 
    # W_o [num_head*head_dim/tp, hidden]


    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    shard_head = math.ceil(num_head / parallel.tp)
    shard_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    # shard_seq_kv = math.ceil(seq / atten_parallel.cp)
    shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)

    # decode footprint: weight absorption
    # weight bytes:
    q_a_weight = hidden * q_down_hidden * weight_bytes
    q_b_weight = q_down_hidden * shard_head*(q_nope_head_dim+q_rope_head_dim) * weight_bytes
    q_up_a_trans_weight = shard_head * head_dim * kv_nope_head_dim * weight_bytes
    kv_down_weight = hidden * (kv_rope_head_dim+kv_nope_head_dim) * weight_bytes
    o_up_b_trans_weight = shard_head * kv_nope_head_dim * head_dim * weight_bytes
    o_weight = num_head*head_dim/parallel.tp * hidden * weight_bytes
    
    
    input_act = in_bytes * shard_bs *shard_seq * hidden
    q_a_act = in_bytes * shard_bs * shard_seq_q * q_down_hidden
    q_b_act = in_bytes * shard_bs * shard_seq_q * shard_head*(q_nope_head_dim+q_rope_head_dim)
    q_up_a_trans_act = in_bytes * shard_bs * shard_seq_q * shard_head * kv_nope_head_dim
    q_act = in_bytes * shard_bs * shard_seq_q * shard_head * (kv_nope_head_dim+ kv_rope_head_dim)

    kv_down_act = in_bytes * shard_bs *shard_seq_q * (kv_rope_head_dim+kv_nope_head_dim)

    k_act = in_bytes * shard_bs *shard_cached_kv * (kv_nope_head_dim+kv_rope_head_dim)
    # v is included in the compressed latent c_kv
    v_act = 0
    o_up_b_trans_act = in_bytes * shard_bs * shard_seq_q * shard_head * kv_nope_head_dim
    o_act = in_bytes * shard_bs * shard_seq * num_head*head_dim/parallel.tp
    
    mem_weight = q_a_weight + q_b_weight + q_up_a_trans_weight + kv_down_weight + o_up_b_trans_weight + o_weight
    # debug
    # log.info("q_a_weight: %s GiB, q_b_weight: %s GiB, q_up_a_trans_weight: %s GiB, kv_down_weight: %s GiB, o_up_b_trans_weight: %s GiB, o_weight: %s GiB", q_a_weight/(1024**3), q_b_weight/(1024**3), q_up_a_trans_weight/(1024**3), kv_down_weight/(1024**3), o_up_b_trans_weight/(1024**3), o_weight/(1024**3))
    mem_kv_cache = k_act + v_act
    max_activation = max((input_act+q_a_act+kv_down_act),(q_a_act+q_b_act),(q_b_act+q_up_a_trans_act), (kv_down_act+k_act+v_act), (q_act+k_act+v_act),2 * o_act)
    # log.info("max_activation candidates: %s, %s, %s, %s",( q_act + k_act + v_act + max(p_act,input_act)), (p_act + p_reduce_act) , (p_reduce_act + o_act), (o_act + o_reduce_act) )
    dp_divisor = np.uint64(parallel.dp) if parallel.fsdp else np.uint64(1)

    mem_weight = mem_weight // dp_divisor

    return max_activation, mem_weight, mem_kv_cache


def get_mla_absorb_and_no_absorb_footprint(bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme):
    '''
    Decode:
    While for weight, we keep both absorption and no absorption footprint
    (no absorb:w_kv_up, absorb:  q_up_a_trans + o_up_b_trans)
    '''
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

    # 其中x [bs/dp, seq/sp, hidden], 
    # Wq_a, [hidden, q_down_hidden], Wq_b [q_down_hidden, num_head/tp*(q_nope_head_dim+q_rope_head_dim)]
    # Wkv_up_a_trans [num_head/tp, head_dim, kv_nope_head_dim]
    # Wkv_down [hidden, (kv_rope_head_dim+kv_nope_head_dim)]
    # Wkv_up_b_trans [num_head/tp, kv_nope_head_dim, head_dim] 
    # W_o [num_head*head_dim/tp, hidden]


    in_bytes, weight_bytes, out_bytes = atten_bytes.get_dtype_bytes()

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    shard_head = math.ceil(num_head / parallel.tp)
    shard_kv_head = math.ceil(num_kv_head / parallel.tp)

    shard_seq_q = math.ceil(seq / atten_parallel.sp)
    # shard_seq_kv = math.ceil(seq / atten_parallel.cp)
    shard_cached_kv = math.ceil(cached_kv / atten_parallel.cp)

    # decode footprint: weight absorption
    # weight bytes:
    q_a_weight = hidden * q_down_hidden * weight_bytes
    q_b_weight = q_down_hidden * shard_head*(q_nope_head_dim+q_rope_head_dim) * weight_bytes
    q_up_a_trans_weight = shard_head * head_dim * kv_nope_head_dim * weight_bytes
    kv_down_weight = hidden * (kv_rope_head_dim+kv_nope_head_dim) * weight_bytes
    o_up_b_trans_weight = shard_head * kv_nope_head_dim * head_dim * weight_bytes
    o_weight = num_head*head_dim/parallel.tp * hidden * weight_bytes
    # no absorb: w_kv_up
    kv_up_weight = kv_nope_head_dim * kv_nope_up_hidden/parallel.tp * weight_bytes
    
    
    input_act = in_bytes * shard_bs *shard_seq * hidden
    q_a_act = in_bytes * shard_bs * shard_seq_q * q_down_hidden
    q_b_act = in_bytes * shard_bs * shard_seq_q * shard_head*(q_nope_head_dim+q_rope_head_dim)
    q_up_a_trans_act = in_bytes * shard_bs * shard_seq_q * shard_head * kv_nope_head_dim
    q_act = in_bytes * shard_bs * shard_seq_q * shard_head * (kv_nope_head_dim+ kv_rope_head_dim)

    kv_down_act = in_bytes * shard_bs *shard_seq_q * (kv_rope_head_dim+kv_nope_head_dim)

    k_act = in_bytes * shard_bs *shard_cached_kv * (kv_nope_head_dim+kv_rope_head_dim)
    # v is included in the compressed latent c_kv
    v_act = 0
    o_up_b_trans_act = in_bytes * shard_bs * shard_seq_q * shard_head * kv_nope_head_dim
    o_act = in_bytes * shard_bs * shard_seq * num_head*head_dim/parallel.tp
    
    mem_weight = q_a_weight + q_b_weight + q_up_a_trans_weight + kv_down_weight + o_up_b_trans_weight + o_weight
    mem_kv_cache = k_act + v_act
    max_activation = max((input_act+q_a_act+kv_down_act),(q_a_act+q_b_act),(q_b_act+q_up_a_trans_act), (kv_down_act+k_act+v_act), (q_act+k_act+v_act),2 * o_act)
    # log.info("max_activation candidates: %s, %s, %s, %s",( q_act + k_act + v_act + max(p_act,input_act)), (p_act + p_reduce_act) , (p_reduce_act + o_act), (o_act + o_reduce_act) )
    dp_divisor = np.uint64(parallel.dp) if parallel.fsdp else np.uint64(1)

    mem_weight = mem_weight // dp_divisor
    kv_up_weight = kv_up_weight // dp_divisor

    return max_activation, (mem_weight+kv_up_weight), mem_kv_cache


if __name__ == "__main__":
    from mosaic.llm_arch import DeepSeekV3
    logging.basicConfig(
    level=logging.INFO,                              # 全局日志级别
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    )

    model_arch = DeepSeekV3()

    parallel =          ParallelScheme(tp=8, ep=1, sp=8, cp=1, dp=2, pp=2,fsdp=True)
    # parallel = ParallelScheme(tp=1, ep=1, sp=64, cp=1, dp=2, pp=2,fsdp=True)
    # parallel = ParallelScheme(tp=1, ep=1, sp=8, cp=1, dp=2, pp=16,fsdp=True)
    # parallel = ParallelScheme(tp=8, ep=1, sp=4, cp=1, dp=8, pp=1,fsdp=True)
    # atten_parallel = ParallelScheme(tp=8, ep=1, sp=2, cp=4, dp=2, pp=2,fsdp=True)
    atten_parallel =    ParallelScheme(tp=8, ep=1, sp=4, cp=2, dp=2, pp=2,fsdp=True)
    next_parallel =     ParallelScheme(tp=4, ep=4, sp=1, cp=4, dp=2, pp=2,fsdp=False)

    bs=16
    seq=1
    cached_kv=128*1024
    # (bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme)
    
    log.info("bs: %s, seq: %s, cached_kv: %s", bs, seq, cached_kv)
    max_activation, mem_weight, mem_kv_cache = get_mla_absorb_and_no_absorb_footprint(bs, seq, cached_kv, model_arch, parallel, atten_parallel)
    log.info("mla absorb and no absorb footprint, MQA:")
    log.info("max_activation: %s GiB, mem_weight: %s GiB, mem_kv_cache: %s GiB", max_activation/(1024**3), mem_weight/(1024**3), mem_kv_cache/(1024**3))

    max_activation, mem_weight, mem_kv_cache = get_mla_absorb_footprint(bs, seq, cached_kv, model_arch, parallel, atten_parallel)
    log.info("mla absorb footprint, MQA:")
    log.info("max_activation: %s GiB, mem_weight: %s GiB, mem_kv_cache: %s GiB", max_activation/(1024**3), mem_weight/(1024**3), mem_kv_cache/(1024**3))

    bs = 16
    seq = 4096
    log.info("bs: %s, seq: %s", bs, seq)
    max_activation, mem_weight, mem_kv_cache = get_mla_no_absorb_footprint(bs, seq, model_arch, parallel, atten_parallel)
    log.info("mla no absorb footprint, MHA:")
    log.info("max_activation: %s GiB, mem_weight: %s GiB, mem_kv_cache: %s GiB", max_activation/(1024**3), mem_weight/(1024**3), mem_kv_cache/(1024**3))