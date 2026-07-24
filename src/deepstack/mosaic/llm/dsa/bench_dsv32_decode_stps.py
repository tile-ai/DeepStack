"""
DeepSeek-V3.2 (DSA) decode STPS sweep on 8x B200.

Full-model per-token decode time = attention(DSA) + dense FFN + MoE + rms_norm + add_residual,
assembled exactly like dse_framework_multi_process_v4.modeling_decode but with the MLA
attention swapped for the DSA path (dsa_mla_decode_kv_list_top).

Parallel scheme: single-node DP-attention + EP (DeepSeek's recommended decode deployment,
also the fastest DSA decode config):
  - moe_parallel    = tp1 ep8 dp1  (experts sharded EP=8 across the node)
  - non_moe_parallel= tp1 ep1 dp8  (attention / dense / norm replicated as DP over the 8 GPUs)

STPS = batch / per_token_time. Reported at the representative KV = prompt + max_new/2
(midpoint of the decode window), since per-token time grows with KV length.
"""
import math
import logging
import dataclasses
from pathlib import Path
import numpy as np

logging.disable(logging.INFO)

from mosaic.parallelism import ParallelScheme
from mosaic.utils import Modeling_Granularity
from mosaic.utils.allocate_ep import allocate_ep
from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
from mosaic.llm_arch.deepseek_v3_2 import DeepSeekV3_2
from mosaic.dse_space.arch_noc_combinations import b200_8x1_8_arch_noc_combinations

from mosaic.llm.dsa.dsa_top import dsa_mla_decode_kv_list_top
from mosaic.llm.dsa.dsa_footprint import get_dsa_mla_absorb_and_no_absorb_footprint
from mosaic.llm.swiglu.swiglu_top import swiglu_top, get_swiglu_footprint
from mosaic.llm.moe.moe_top import moe_top, get_moe_footprint
from mosaic.llm.rms_norm.rms_norm_top import rms_norm_top, get_rms_norm_footprint
from mosaic.llm.add_residual.add_residual_top import add_residual_top, get_add_residual_footprint
from mosaic.llm.rope.get_rope_global_footprint import get_rope_global_footprint


SCENARIOS = {
    "io1024_1024":  {"prompt_tokens": 1024,  "max_new_tokens": 1024},
    "io3072_1024":  {"prompt_tokens": 3072,  "max_new_tokens": 1024},
    "io4096_1024":  {"prompt_tokens": 4096,  "max_new_tokens": 1024},
    "io7168_1024":  {"prompt_tokens": 7168,  "max_new_tokens": 1024},
    "io15360_1024": {"prompt_tokens": 15360, "max_new_tokens": 1024},
    "io31744_1024": {"prompt_tokens": 31744, "max_new_tokens": 1024},
    "io4096_4096":  {"prompt_tokens": 4096,  "max_new_tokens": 4096},
}

BS_SWEEP = [512, 256, 128, 64, 32]


def build():
    combos = b200_8x1_8_arch_noc_combinations()
    arch, noc = combos[0][0], combos[0][1]
    model = DeepSeekV3_2()
    g = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False, dump_perf_log=False)
    routing = load_npz_routing_keep_shape(
        str(Path(__file__).resolve().parents[3] / "mosaic" / "data" / "aime_ds_r1" / "moe_activations_batch0.npz"),
        as_list=False)[1]
    return arch, noc, model, g, routing


def full_model_decode_time(model, bs, kv_len, moe_parallel, non_moe_parallel, arch, noc, g, routing):
    """整模单 token decode 时间 (秒)，组装方式对齐 modeling_decode。"""
    hidden = model.hidden_size
    num_layer = model.num_layer
    seq = 1

    # MoE 用纯 EP: tp 折进 ep (与 modeling_decode 的 debug_tp_ep_parallel_scheme 一致)
    moe_compute = dataclasses.replace(
        moe_parallel, tp=1,
        ep=moe_parallel.ep * moe_parallel.tp,
        ep1=moe_parallel.ep * moe_parallel.tp, ep2=1)

    # 1. attention (DSA), 全部 61 层
    t_attn_list, _ = dsa_mla_decode_kv_list_top(
        bs=bs, seq=seq, cached_kv_list=[kv_len], model_arch=model,
        parallel=non_moe_parallel, atten_parallel=non_moe_parallel, next_parallel=non_moe_parallel,
        granularity=g, single_chip=arch, noc_hierarchy=noc)
    t_attn = t_attn_list[0] * num_layer

    # 2. dense FFN, 3 层
    t_dense, _ = swiglu_top(bs=bs, seq=seq, hidden=hidden, up_hidden=model.dense_ffn_arch.up_hidden,
        parallel=non_moe_parallel, next_parallel=non_moe_parallel, swiglu_bytes=model.dense_ffn_arch.swiglu_bytes,
        granularity=g, single_chip=arch, noc_hierarchy=noc)
    t_dense = t_dense * model.num_dense_layer

    # 3. MoE, 58 层
    t_moe, _ = moe_top(bs=bs, seq=seq, hidden=hidden, moe_down_hidden=model.moe_arch.moe_down_hidden,
        parallel=moe_compute, next_parallel=moe_compute,
        expert_bytes=model.moe_arch.expert_bytes, gate_bytes=model.moe_arch.gate_bytes,
        num_shared_experts=model.moe_arch.num_shared_experts, num_routed_experts=model.moe_arch.num_routed_experts,
        num_activated_experts=model.moe_arch.num_activated_experts, routing_array=routing,
        granularity=g, single_chip=arch, noc_hierarchy=noc)
    t_moe = t_moe * model.num_moe_layer

    # 4. rms_norm + add_residual, 每层 2 个
    t_rms, _ = rms_norm_top(bs=bs, seq=seq, hidden=hidden, parallel=non_moe_parallel, next_parallel=non_moe_parallel,
        rms_norm_bytes=model.rms_norm_bytes, granularity=g, single_chip=arch, noc_hierarchy=noc)
    t_rms = t_rms * num_layer * 2
    t_res, _ = add_residual_top(bs=bs, seq=seq, hidden=hidden, parallel=non_moe_parallel, next_parallel=non_moe_parallel,
        add_residual_bytes=model.add_residual_bytes, granularity=g, single_chip=arch, noc_hierarchy=noc)
    t_res = t_res * num_layer * 2

    total = t_attn + t_dense + t_moe + t_rms + t_res
    return total, dict(attn=t_attn, dense=t_dense, moe=t_moe, rms=t_rms, res=t_res)


def footprint_per_gpu(model, minibatch, max_kv, moe_parallel, non_moe_parallel):
    """每 GPU 总占用 (bytes): weights + kv cache + max activation. 对齐 get_max_footprint_decode。
    传入完整 minibatch (= bs, pp=1); footprint 函数内部按 parallel.dp / ep 自行分片。"""
    hidden = model.hidden_size
    pp = non_moe_parallel.pp
    shard_layer = math.ceil(model.num_layer / pp)

    # attention (DSA) footprint, per layer; 内部按 non_moe_parallel.dp 切 bs
    a_act, a_w, a_kv = get_dsa_mla_absorb_and_no_absorb_footprint(
        minibatch, 1, max_kv, model, non_moe_parallel, non_moe_parallel)
    a_w *= shard_layer
    a_kv *= model.num_layer

    # dense
    d_act, d_w = get_swiglu_footprint(minibatch, 1, hidden, model.dense_ffn_arch.up_hidden, non_moe_parallel, model.dense_ffn_arch.swiglu_bytes)
    d_w *= math.ceil(model.num_dense_layer / pp)

    # moe (EP): get_moe_footprint 内部按 moe_parallel 分 expert
    m_act, m_w = get_moe_footprint(minibatch, 1, hidden, model.moe_arch.moe_down_hidden, moe_parallel,
        model.moe_arch.expert_bytes, model.moe_arch.gate_bytes, model.moe_arch.num_shared_experts, model.moe_arch.num_routed_experts)
    m_w *= math.ceil(model.num_moe_layer / pp)

    r_act, r_w = get_rms_norm_footprint(bs=minibatch, seq=1, hidden=hidden, parallel=non_moe_parallel, rms_norm_bytes=model.rms_norm_bytes)
    r_w *= shard_layer * 2
    rope_w = get_rope_global_footprint(bs=minibatch, head_dim=model.mla_arch.head_dim, parallel=non_moe_parallel, rope_bytes=model.rope_bytes, MAX_ROPE_SEQ=model.max_seq_len)
    res_act, res_w = get_add_residual_footprint(bs=minibatch, seq=1, hidden=hidden, parallel=non_moe_parallel, add_residual_bytes=model.add_residual_bytes)
    res_w *= shard_layer * 2

    max_act = max(a_act, d_act, m_act, r_act, res_act)
    total_w = a_w + d_w + m_w + r_w + rope_w + res_w
    total_kv = a_kv
    return max_act, total_w, total_kv


def main():
    arch, noc, model, g, routing = build()
    cap = arch.ddr_capacity  # per-GPU bytes
    GiB = 1024**3

    # 单节点 TP-attention + EP (paper 同款: attention tp=8, MoE 用 ep=8)
    moe_parallel = ParallelScheme(tp=8, ep=1, sp=1, cp=1, dp=1, pp=1, fsdp=False)
    non_moe_parallel = dataclasses.replace(moe_parallel, ep=1, ep1=1, ep2=1, dp=moe_parallel.dp * moe_parallel.ep)
    DP = non_moe_parallel.dp

    print(f"Model: DeepSeek-V3.2 (DSA)  |  Arch: 8x B200 ({cap/GiB:.0f} GiB/GPU)  |  scheme: TP-attention tp={non_moe_parallel.tp} + EP ep={moe_parallel.ep*moe_parallel.tp}")
    print(f"index_topk={model.dsa_arch.index_topk}, attn dtype={model.dsa_arch.atten_bytes.input1.dtype}")
    print(f"STPS reported at representative KV = prompt + max_new/2 (decode-window midpoint). 'x' = OOM (>{cap/GiB:.0f} GiB/GPU).\n")

    header = "scenario       prompt  new   KVmid  | " + "  ".join(f"bs={b}" for b in BS_SWEEP)
    print(header)
    print("-" * len(header))

    detail = {}
    for sc, cfg in SCENARIOS.items():
        P, N = cfg["prompt_tokens"], cfg["max_new_tokens"]
        kv_mid = P + N // 2
        max_kv = P + N
        row = []
        for bs in BS_SWEEP:
            minibatch = bs  # pp=1
            allocate_ep(parallel=moe_parallel, bs=minibatch, seq=1)
            max_act, total_w, total_kv = footprint_per_gpu(model, minibatch, max_kv, moe_parallel, non_moe_parallel)
            fits = (max_act + total_w + total_kv) <= cap

            t, parts = full_model_decode_time(model, bs, kv_mid, moe_parallel, non_moe_parallel, arch, noc, g, routing)
            stps = bs / t
            detail[(sc, bs)] = (stps, t, parts, fits, (max_act + total_w + total_kv) / GiB)
            row.append(f"{stps:7.0f}" + ("" if fits else "x"))
        print(f"{sc:14s} {P:6d} {N:5d} {kv_mid:6d} | " + "  ".join(row))

    # breakdown for one representative point
    print("\n--- per-block breakdown @ io4096_1024, bs=64, KVmid ---")
    _, t, parts, fits, memgib = detail[("io4096_1024", 64)]
    for k, v in parts.items():
        print(f"  {k:6s}: {v*1e3:8.3f} ms  ({100*v/t:4.1f}%)")
    print(f"  total : {t*1e3:8.3f} ms  -> STPS {64/t:.0f}, mem/GPU {memgib:.1f} GiB ({'fits' if fits else 'OOM'})")


if __name__ == "__main__":
    main()
