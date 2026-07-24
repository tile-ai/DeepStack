# Qwen3-Omni 类多组件模型的组件级建模与端到端编排 (Phase 1: 同集群串行)。
#
# 依赖链: vision/audio encoder -> thinker (prefill + AR decode)
#          -> resize MLP -> talker (prefill + 每 codec 帧 1 步 decode)
#          -> code predictor (每帧 num_code_groups-1 步微型 decode)
#          -> code2wav (按 chunk 流式 vocoder)
#
# Phase 1 假设所有组件共享同一组设备, 按依赖链串行占用; 输出指标:
#   TTFT (首 text token), text TPOT, first-audio-packet latency, 音频 RTF。
# Phase 2 (分离式部署 + 组件间 NoC 传输 + 流水稳态) 见 modeling_omni_disagg (待做)。
import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from mosaic.llm_arch import LLM_Arch
from mosaic.llm_arch.omni_base import Omni_Arch, ViT_Encoder_Arch, Audio_Encoder_Arch, Code2Wav_Arch
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, get_extend_max_routes_with_traffic
from mosaic.utils import Modeling_Granularity
from tilesight.arch import Arch

from mosaic.llm.gqa.gqa_prefill_top import gqa_prefill_top
from mosaic.llm.gqa.gqa_decode_top import gqa_decode_kv_list_top
from mosaic.llm.swiglu.swiglu_top import swiglu_top
from mosaic.llm.moe.moe_top import moe_top
from mosaic.llm.rms_norm.rms_norm_top import rms_norm_top
from mosaic.llm.add_residual.add_residual_top import add_residual_top
from mosaic.llm.mlp_gelu import mlp_gelu_top
from mosaic.llm.conv import (
    activation_1d_top,
    conv1d_top,
    conv2d_top,
    conv3d_patch_embed_top,
    conv_transpose1d_top,
    depthwise_conv1d_top,
)

import logging
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# workload / 并行方案定义
# ---------------------------------------------------------------------------

@dataclass
class OmniWorkload:
    bs: int = 1
    text_in_tokens: int = 128
    num_images: int = 0
    image_height: int = 1080
    image_width: int = 1920
    video_seconds: float = 0.0
    video_fps: float = 2.0                # 抽帧率 (Qwen3-Omni 默认 ~2fps)
    video_height: int = 720
    video_width: int = 1280
    audio_in_seconds: float = 0.0
    text_out_tokens: int = 256
    audio_out_seconds: float = 0.0
    # talker 启动前等待 thinker 先出多少个 text token (流式启动阈值)
    talker_wait_text_tokens: int = 4
    # code2wav 首个可播 chunk 的帧数 (12.5Hz; 25 帧 = 2s, 对应 config seconds_per_chunk)
    first_chunk_frames: int = 25


@dataclass
class OmniParallelPlan:
    """每个组件一组并行方案 (Phase 1: 同一组设备上串行, world_size 应一致)。
    decode 类方案 seq==1, 要求 sp=1。"""
    vision: ParallelScheme
    audio: ParallelScheme
    thinker_prefill: ParallelScheme
    thinker_prefill_atten: ParallelScheme
    thinker_moe: ParallelScheme
    thinker_decode: ParallelScheme
    talker_prefill: ParallelScheme
    talker_decode: ParallelScheme
    talker_moe: ParallelScheme
    code_predictor: ParallelScheme
    code2wav: ParallelScheme


def make_uniform_routing(num_tokens: int, num_experts: int, topk: int, seed: int = 0) -> np.ndarray:
    """合成均匀路由 (fallback): 每 token 无重复地选 topk 个专家。"""
    rng = np.random.default_rng(seed)
    scores = rng.random((max(1, num_tokens), num_experts))
    return np.argpartition(scores, -topk, axis=-1)[:, -topk:].astype(np.int64)


# 模拟路由生成一次后按 (n_experts, n_group, topk_group, topk) 缓存到文件, 进程内再缓存 ndarray
_SIM_ROUTING_CACHE: dict = {}
_SIM_ROUTING_BASE_TOKENS = 2048  # 生成的基础 token 数; 更长的请求平铺复用
                                 # (ep_all_to_all 对 >=1024 token 走平均+不均衡因子, 平铺不损失信息)


def get_simulated_routing(num_tokens: int, num_experts: int, topk: int,
                          n_group: int = 1, topk_group: int = 1,
                          cache_dir: "str | None" = None) -> np.ndarray:
    """无真实 trace 时, 用 mosaic.utils.moe_router_sim 的路由模拟器合成路由
    (随机 linear router 打分 + top-k, 比均匀采样更接近真实的专家负载不均衡)。
    Qwen 系列无分组受限路由, 用 n_group=1, topk_group=1 即普通 top-k。"""
    from mosaic.utils.moe_router_sim import (
        extract_selected_experts_lists,
        generate_filename,
        import_results_as_expert_id_lists,
        run_simulation,
        export_results,
    )
    import os
    import mosaic

    key = (num_experts, n_group, topk_group, topk)
    if key not in _SIM_ROUTING_CACHE:
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(mosaic.__file__), "..", "data", "routing_outputs")
        config = {
            "iter": _SIM_ROUTING_BASE_TOKENS,
            "n_routed_experts": num_experts,
            "n_group": n_group,
            "topk_group": topk_group,
            "num_experts_per_tok": topk,
            "output_dir": cache_dir,
        }
        filepath = os.path.join(cache_dir, generate_filename(config))
        if os.path.exists(filepath):
            expert_id_lists = import_results_as_expert_id_lists(filepath)
        else:
            routing_data = run_simulation(config)
            export_results(filepath, config, routing_data)
            expert_id_lists = extract_selected_experts_lists(routing_data)
        _SIM_ROUTING_CACHE[key] = np.asarray(expert_id_lists, dtype=np.int64)

    base = _SIM_ROUTING_CACHE[key]
    num_tokens = max(1, num_tokens)
    if num_tokens <= base.shape[0]:
        return base[:num_tokens]
    reps = math.ceil(num_tokens / base.shape[0])
    return np.tile(base, (reps, 1))[:num_tokens]


def get_routing_from_npz(npz_path: str, phase: str = "prefill", topk: "int | None" = None) -> np.ndarray:
    """读取 routing trace：prefill [layer, iter, topk] 或
    decode [iter, layer, batch, topk]，并展平成 [tokens, topk]
    直接可作 modeling_llm_prefill/decode_step 的 routing_array。
    topk 缺省时自动探测末维 (thinker=8, talker=6)。"""
    from mosaic.utils.moe_router_sim import load_npz_routing_flatten_last_dim
    if topk is None:
        with np.load(npz_path, allow_pickle=True) as zf:
            arr = zf.get("prefill", zf.get("decode"))
            topk = int(np.asarray(arr).shape[-1])
    prefill_rows, decode_rows = load_npz_routing_flatten_last_dim(npz_path, as_list=False, expected_last_dim=topk)
    rows = prefill_rows if phase == "prefill" else decode_rows
    if rows is None or len(rows) == 0:
        raise ValueError(f"npz {npz_path} 中没有 {phase} 路由数据")
    return np.asarray(rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# encoder 组件
# ---------------------------------------------------------------------------

def modeling_vision_encoder(arch: ViT_Encoder_Arch, bs: int, num_patches: int, parallel: ParallelScheme,
                            granularity: Modeling_Granularity, single_chip: Arch, noc_hierarchy: Hierarchy):
    """双向 ViT encoder 一次 forward (prefill-only)。返回 (total_time, breakdown)。"""
    if num_patches == 0:
        return 0.0, {}

    merged_tokens = math.ceil(num_patches / (arch.spatial_merge_size ** 2))

    t_patch_embed, _ = conv3d_patch_embed_top(
        num_patches=bs * num_patches, cin=arch.in_channels, embed_dim=arch.hidden_size,
        kernel_elems=arch.temporal_patch_size * arch.patch_size * arch.patch_size,
        parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
        single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # 双向 MHA: num_kv_head == num_head, fa_prefill 本身按全量 S_q x S_kv 建模 (无 causal 折扣)
    t_atten, _ = gqa_prefill_top(
        bs=bs, seq=num_patches, hidden=arch.hidden_size, num_head=arch.num_head,
        num_kv_head=arch.num_head, head_dim=arch.head_dim,
        parallel=parallel, atten_parallel=parallel, next_parallel=parallel,
        atten_bytes=arch.atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    t_mlp, _ = mlp_gelu_top(
        bs=bs, seq=num_patches, hidden=arch.hidden_size, up_hidden=arch.ffn_hidden,
        parallel=parallel, next_parallel=parallel, mlp_bytes=arch.mlp_bytes,
        granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # LayerNorm 以 rms_norm 建模 (同为访存受限归一化), residual 同现有框架
    t_norm, _ = rms_norm_top(bs=bs, seq=num_patches, hidden=arch.hidden_size, parallel=parallel, next_parallel=parallel,
                             rms_norm_bytes=arch.mlp_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    t_res, _ = add_residual_top(bs=bs, seq=num_patches, hidden=arch.hidden_size, parallel=parallel, next_parallel=parallel,
                                add_residual_bytes=arch.mlp_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # patch merger (主 merger + deepstack mergers): 非门控 MLP [merge_hidden -> merge_hidden -> out_hidden]
    t_merger, _ = mlp_gelu_top(
        bs=bs, seq=merged_tokens, hidden=arch.merger_hidden, up_hidden=arch.merger_hidden,
        parallel=parallel, next_parallel=parallel, mlp_bytes=arch.mlp_bytes,
        granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy,
        out_hidden=arch.out_hidden_size)
    num_mergers = 1 + arch.num_deepstack_mergers

    breakdown = {
        "patch_embed": t_patch_embed,
        "atten": t_atten * arch.depth,
        "mlp": t_mlp * arch.depth,
        "norm": t_norm * 2 * arch.depth,
        "residual": t_res * 2 * arch.depth,
        "merger": t_merger * num_mergers,
    }
    total = sum(breakdown.values())
    log.info("vision encoder (%s patches -> %s tokens): %s s, %s", num_patches, merged_tokens, total, breakdown)
    return total, breakdown


def modeling_audio_encoder(arch: Audio_Encoder_Arch, bs: int, audio_seconds: float, parallel: ParallelScheme,
                           granularity: Modeling_Granularity, single_chip: Arch, noc_hierarchy: Hierarchy):
    """AuT audio encoder 一次 forward (prefill-only)。返回 (total_time, breakdown)。"""
    if audio_seconds <= 0:
        return 0.0, {}

    mel_frames = arch.num_mel_frames(audio_seconds)
    tokens = arch.num_tokens(audio_seconds)

    # conv 前端: 3x Conv2d(k=3, s=2), (H=mel_bins, W=mel_frames), cin 1 -> ds -> ds -> ds
    h, w = arch.num_mel_bins, mel_frames
    conv_specs = [(1, arch.downsample_hidden), (arch.downsample_hidden, arch.downsample_hidden), (arch.downsample_hidden, arch.downsample_hidden)]
    t_conv_front = 0.0
    for cin, cout in conv_specs:
        t_c, _ = conv2d_top(bs=bs, height=h, width=w, cin=cin, cout=cout, kernel=3, stride=2,
                            parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
                            single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_conv_front += t_c
        h, w = math.ceil(h / 2), math.ceil(w / 2)

    # conv_out: Linear(ds * mel/8, d_model), 以 k=1 conv1d (纯 GEMM) 建模
    t_conv_out, _ = conv1d_top(bs=bs, length=tokens, cin=arch.downsample_hidden * arch.downsampled_mel_bins,
                               cout=arch.d_model, kernel=1, stride=1,
                               parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
                               single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # 窗口注意力: 块大小 = n_window_infer(mel 帧) 对应的 token 数; 块间不注意
    chunk_mel = 2 * arch.n_window
    chunk_tokens = arch.num_tokens(chunk_mel / arch.mel_frame_rate)
    window_tokens = chunk_tokens * max(1, arch.n_window_infer // chunk_mel)
    num_windows = max(1, math.ceil(tokens / window_tokens))
    eff_window = min(window_tokens, tokens)

    t_atten, _ = gqa_prefill_top(
        bs=bs * num_windows, seq=eff_window, hidden=arch.d_model, num_head=arch.num_head,
        num_kv_head=arch.num_head, head_dim=arch.head_dim,
        parallel=parallel, atten_parallel=parallel, next_parallel=parallel,
        atten_bytes=arch.atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    t_mlp, _ = mlp_gelu_top(
        bs=bs, seq=tokens, hidden=arch.d_model, up_hidden=arch.ffn_hidden,
        parallel=parallel, next_parallel=parallel, mlp_bytes=arch.mlp_bytes,
        granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    t_norm, _ = rms_norm_top(bs=bs, seq=tokens, hidden=arch.d_model, parallel=parallel, next_parallel=parallel,
                             rms_norm_bytes=arch.mlp_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    t_res, _ = add_residual_top(bs=bs, seq=tokens, hidden=arch.d_model, parallel=parallel, next_parallel=parallel,
                                add_residual_bytes=arch.mlp_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    # 出口: proj1(d_model->d_model) + gelu + proj2(d_model->output_dim)
    t_out_proj, _ = mlp_gelu_top(
        bs=bs, seq=tokens, hidden=arch.d_model, up_hidden=arch.d_model,
        parallel=parallel, next_parallel=parallel, mlp_bytes=arch.mlp_bytes,
        granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy,
        out_hidden=arch.output_dim)

    breakdown = {
        "conv_front": t_conv_front,
        "conv_out": t_conv_out,
        "atten": t_atten * arch.num_layers,
        "mlp": t_mlp * arch.num_layers,
        "norm": t_norm * 2 * arch.num_layers,
        "residual": t_res * 2 * arch.num_layers,
        "out_proj": t_out_proj,
    }
    total = sum(breakdown.values())
    log.info("audio encoder (%.1fs -> %s tokens, window %s): %s s, %s", audio_seconds, tokens, eff_window, total, breakdown)
    return total, breakdown


# ---------------------------------------------------------------------------
# causal LM 组件 (thinker / talker / code predictor 通用)
# ---------------------------------------------------------------------------

def _fill_ep_split(p: ParallelScheme) -> ParallelScheme:
    # moe_coarse 需要 ep1/ep2 (分层 EP); ep==1 时 allocate_ep 不会被触发, 这里兜底
    if p.ep == 1 and (getattr(p, "ep1", None) is None or getattr(p, "ep2", None) is None):
        return dataclasses.replace(p, ep1=1, ep2=1)
    return p


def modeling_llm_prefill(model_arch: LLM_Arch, bs: int, seq: int,
                         parallel: ParallelScheme, atten_parallel: ParallelScheme, moe_parallel: ParallelScheme,
                         granularity: Modeling_Granularity, single_chip: Arch, noc_hierarchy: Hierarchy,
                         routing_array: "np.ndarray | None" = None):
    """通用 causal LM prefill (与 dse v3/v4 的 modeling_prefill 同构)。返回 (total_time, breakdown)。"""
    hidden = model_arch.hidden_size
    shard_layer = math.ceil(model_arch.num_layer / parallel.pp)

    def get_pipeline_time():
        if parallel.pp == 1:
            return 0
        shard_bs = math.ceil(bs / parallel.dp)
        shard_seq = math.ceil(seq / parallel.sp)
        activation_bytes = model_arch.add_residual_bytes.input1.num_bytes
        pipeline_activation_bytes = shard_bs * shard_seq * hidden * activation_bytes
        pp_time_list = []
        for i in range(parallel.pp - 1):
            traffic_pair = [[pipeline_activation_bytes, i, (i + 1) % parallel.pp]]
            tm = TrafficMatrix(parallel.world_size())
            tm.add_intra_group_traffic_pair_bulk("pp", traffic_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
            _, _, pp_overall_time, _ = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
            pp_time_list.append(pp_overall_time)
        return max(pp_time_list) if pp_time_list else 0.0

    pp_p2p_time = get_pipeline_time()

    if model_arch.gqa_arch is not None:
        time_gqa, _ = gqa_prefill_top(bs=bs, seq=seq, hidden=hidden, num_head=model_arch.gqa_arch.num_head,
                                      num_kv_head=model_arch.gqa_arch.num_kv_head, head_dim=model_arch.gqa_arch.head_dim,
                                      parallel=parallel, atten_parallel=atten_parallel, next_parallel=parallel,
                                      atten_bytes=model_arch.gqa_arch.atten_bytes, granularity=granularity,
                                      single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        time_gqa = time_gqa * shard_layer
    else:
        time_gqa = 0

    if model_arch.dense_ffn_arch is not None:
        time_dense_ffn, _ = swiglu_top(bs=bs, seq=seq, hidden=hidden, up_hidden=model_arch.dense_ffn_arch.up_hidden,
                                       parallel=parallel, next_parallel=parallel, swiglu_bytes=model_arch.dense_ffn_arch.swiglu_bytes,
                                       granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        time_dense_ffn = time_dense_ffn * math.ceil(model_arch.num_dense_layer / parallel.pp)
    else:
        time_dense_ffn = 0

    if model_arch.moe_arch is not None:
        moe_parallel = _fill_ep_split(moe_parallel)
        if routing_array is None:
            routing_array = get_simulated_routing(bs * seq, model_arch.moe_arch.num_routed_experts, model_arch.moe_arch.num_activated_experts)
        time_moe, _ = moe_top(bs=bs, seq=seq, hidden=hidden, moe_down_hidden=model_arch.moe_arch.moe_down_hidden,
                              parallel=moe_parallel, next_parallel=parallel,
                              expert_bytes=model_arch.moe_arch.expert_bytes, gate_bytes=model_arch.moe_arch.gate_bytes,
                              num_shared_experts=model_arch.moe_arch.num_shared_experts,
                              num_routed_experts=model_arch.moe_arch.num_routed_experts,
                              num_activated_experts=model_arch.moe_arch.num_activated_experts,
                              routing_array=routing_array, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        time_moe = time_moe * math.ceil(model_arch.num_moe_layer / parallel.pp)
    else:
        time_moe = 0

    time_rms_norm, _ = rms_norm_top(bs=bs, seq=seq, hidden=hidden, parallel=parallel, next_parallel=parallel,
                                    rms_norm_bytes=model_arch.rms_norm_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    time_rms_norm = time_rms_norm * shard_layer * 2

    time_add_residual, _ = add_residual_top(bs=bs, seq=seq, hidden=hidden, parallel=parallel, next_parallel=parallel,
                                            add_residual_bytes=model_arch.add_residual_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    time_add_residual = time_add_residual * shard_layer * 2

    breakdown = {"pp_p2p": pp_p2p_time, "gqa": time_gqa, "dense_ffn": time_dense_ffn, "moe": time_moe,
                 "rms_norm": time_rms_norm, "add_residual": time_add_residual}
    total = sum(breakdown.values())
    log.info("%s prefill bs=%s seq=%s: %s s, %s", model_arch.name, bs, seq, total, breakdown)
    return total, breakdown


def modeling_llm_decode_step(model_arch: LLM_Arch, bs: int, cached_kv: int,
                             parallel: ParallelScheme, moe_parallel: ParallelScheme,
                             granularity: Modeling_Granularity, single_chip: Arch, noc_hierarchy: Hierarchy,
                             routing_array: "np.ndarray | None" = None):
    """通用 causal LM 单步 decode (seq=1), 与 dse v4 的 modeling_decode 同构。返回 (step_time, breakdown)。"""
    assert parallel.sp == 1, "decode 单 token, sp 必须为 1"
    hidden = model_arch.hidden_size
    shard_layer = math.ceil(model_arch.num_layer / parallel.pp)
    seq = 1

    def get_pipeline_time():
        if parallel.pp == 1:
            return 0
        shard_bs = math.ceil(bs / parallel.dp)
        activation_bytes = model_arch.add_residual_bytes.input1.num_bytes
        pipeline_activation_bytes = shard_bs * 1 * hidden * activation_bytes
        pp_time_list = []
        for i in range(parallel.pp - 1):
            traffic_pair = [[pipeline_activation_bytes, i, (i + 1) % parallel.pp]]
            tm = TrafficMatrix(parallel.world_size())
            tm.add_intra_group_traffic_pair_bulk("pp", traffic_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
            _, _, pp_overall_time, _ = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
            pp_time_list.append(pp_overall_time)
        return max(pp_time_list) if pp_time_list else 0.0

    pp_p2p_time = get_pipeline_time()

    if model_arch.gqa_arch is not None:
        time_gqa_list, _ = gqa_decode_kv_list_top(bs=bs, seq=seq, cached_kv_list=[cached_kv], hidden=hidden,
                                                  num_head=model_arch.gqa_arch.num_head, num_kv_head=model_arch.gqa_arch.num_kv_head,
                                                  head_dim=model_arch.gqa_arch.head_dim, parallel=parallel, atten_parallel=parallel,
                                                  next_parallel=parallel, atten_bytes=model_arch.gqa_arch.atten_bytes,
                                                  granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        time_gqa = time_gqa_list[0] * shard_layer
    else:
        time_gqa = 0

    if model_arch.dense_ffn_arch is not None:
        time_dense_ffn, _ = swiglu_top(bs=bs, seq=seq, hidden=hidden, up_hidden=model_arch.dense_ffn_arch.up_hidden,
                                       parallel=parallel, next_parallel=parallel, swiglu_bytes=model_arch.dense_ffn_arch.swiglu_bytes,
                                       granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        time_dense_ffn = time_dense_ffn * math.ceil(model_arch.num_dense_layer / parallel.pp)
    else:
        time_dense_ffn = 0

    if model_arch.moe_arch is not None:
        moe_parallel = _fill_ep_split(moe_parallel)
        if routing_array is None:
            routing_array = get_simulated_routing(bs, model_arch.moe_arch.num_routed_experts, model_arch.moe_arch.num_activated_experts)
        time_moe, _ = moe_top(bs=bs, seq=seq, hidden=hidden, moe_down_hidden=model_arch.moe_arch.moe_down_hidden,
                              parallel=moe_parallel, next_parallel=moe_parallel,
                              expert_bytes=model_arch.moe_arch.expert_bytes, gate_bytes=model_arch.moe_arch.gate_bytes,
                              num_shared_experts=model_arch.moe_arch.num_shared_experts,
                              num_routed_experts=model_arch.moe_arch.num_routed_experts,
                              num_activated_experts=model_arch.moe_arch.num_activated_experts,
                              routing_array=routing_array, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        time_moe = time_moe * math.ceil(model_arch.num_moe_layer / parallel.pp)
    else:
        time_moe = 0

    time_rms_norm, _ = rms_norm_top(bs=bs, seq=seq, hidden=hidden, parallel=parallel, next_parallel=parallel,
                                    rms_norm_bytes=model_arch.rms_norm_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    time_rms_norm = time_rms_norm * shard_layer * 2

    time_add_residual, _ = add_residual_top(bs=bs, seq=seq, hidden=hidden, parallel=parallel, next_parallel=parallel,
                                            add_residual_bytes=model_arch.add_residual_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    time_add_residual = time_add_residual * shard_layer * 2

    breakdown = {"pp_p2p": pp_p2p_time, "gqa": time_gqa, "dense_ffn": time_dense_ffn, "moe": time_moe,
                 "rms_norm": time_rms_norm, "add_residual": time_add_residual}
    total = sum(breakdown.values())
    log.info("%s decode step bs=%s kv=%s: %s s", model_arch.name, bs, cached_kv, total)
    return total, breakdown


# ---------------------------------------------------------------------------
# code2wav 组件
# ---------------------------------------------------------------------------

def modeling_code2wav_chunk(arch: Code2Wav_Arch, bs: int, frames: int, parallel: ParallelScheme,
                            granularity: Modeling_Granularity, single_chip: Arch, noc_hierarchy: Hierarchy,
                            with_left_context: bool = True):
    """code2wav 处理一个 chunk (frames 个 codec 帧, 含左上下文) 的时间。返回 (total_time, breakdown)。"""
    L = frames + (arch.left_context if with_left_context else 0)

    # pre_transformer: sliding window 注意力, 每 query 最多看 window 个 kv;
    # 以块近似: bs_eff = ceil(L/w) 块, 每块 seq=w (总 attention 面积 ~ L*w)
    w = min(arch.sliding_window, L)
    num_blocks = max(1, math.ceil(L / arch.sliding_window))
    t_atten, _ = gqa_prefill_top(
        bs=bs * num_blocks, seq=w, hidden=arch.hidden_size, num_head=arch.num_head,
        num_kv_head=arch.num_kv_head, head_dim=arch.head_dim,
        parallel=parallel, atten_parallel=parallel, next_parallel=parallel,
        atten_bytes=arch.atten_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    t_mlp, _ = swiglu_top(bs=bs, seq=L, hidden=arch.hidden_size, up_hidden=arch.ffn_hidden,
                          parallel=parallel, next_parallel=parallel, swiglu_bytes=arch.mlp_bytes,
                          granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    t_norm, _ = rms_norm_top(bs=bs, seq=L, hidden=arch.hidden_size, parallel=parallel, next_parallel=parallel,
                             rms_norm_bytes=arch.mlp_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    t_res, _ = add_residual_top(bs=bs, seq=L, hidden=arch.hidden_size, parallel=parallel, next_parallel=parallel,
                                add_residual_bytes=arch.mlp_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    t_transformer = (t_atten + t_mlp + 2 * t_norm + 2 * t_res) * arch.num_layers

    # upsample 段: 对每个 ratio r: TransConv(h, h, r, r) + ConvNeXt(dwconv k7 + pw h->4h->h)
    t_upsample = 0.0
    cur_len = L
    h = arch.hidden_size
    for r in arch.upsampling_ratios:
        t_tc, _ = conv_transpose1d_top(bs=bs, length=cur_len, cin=h, cout=h, kernel=r, stride=r,
                                       parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
                                       single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        cur_len *= r
        t_dw, _ = depthwise_conv1d_top(bs=bs, length=cur_len, channels=h, kernel=7,
                                       parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
                                       single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_pw, _ = mlp_gelu_top(bs=bs, seq=cur_len, hidden=h, up_hidden=4 * h,
                               parallel=parallel, next_parallel=parallel, mlp_bytes=arch.mlp_bytes,
                               granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_ln, _ = rms_norm_top(bs=bs, seq=cur_len, hidden=h, parallel=parallel, next_parallel=parallel,
                               rms_norm_bytes=arch.mlp_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_upsample += t_tc + t_dw + t_pw + t_ln

    # decoder 段: Conv1d(h, decoder_dim, 7) + 逐级 DecoderBlock + 尾部 Conv1d(out, 1, 7)
    t_decoder = 0.0
    t_c, _ = conv1d_top(bs=bs, length=cur_len, cin=h, cout=arch.decoder_dim, kernel=7, stride=1,
                        parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
                        single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    t_decoder += t_c

    in_dim = arch.decoder_dim
    for i, r in enumerate(arch.upsample_rates):
        out_dim = arch.decoder_dim // (2 ** (i + 1))
        # SnakeBeta + TransConv(in, out, 2r, r)
        t_act, _ = activation_1d_top(bs=bs, length=cur_len, channels=in_dim, parallel=parallel, conv_bytes=arch.conv_bytes,
                                     granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_tc, _ = conv_transpose1d_top(bs=bs, length=cur_len, cin=in_dim, cout=out_dim, kernel=2 * r, stride=r,
                                       parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
                                       single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        cur_len *= r
        t_decoder += t_act + t_tc
        # 3x ResidualUnit: SnakeBeta + Conv1d(out,out,7,dilated) + SnakeBeta + Conv1d(out,out,1)
        t_act2, _ = activation_1d_top(bs=bs, length=cur_len, channels=out_dim, parallel=parallel, conv_bytes=arch.conv_bytes,
                                      granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_c7, _ = conv1d_top(bs=bs, length=cur_len, cin=out_dim, cout=out_dim, kernel=7, stride=1,
                             parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
                             single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_c1, _ = conv1d_top(bs=bs, length=cur_len, cin=out_dim, cout=out_dim, kernel=1, stride=1,
                             parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
                             single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_res_unit, _ = add_residual_top(bs=bs, seq=cur_len, hidden=out_dim, parallel=parallel, next_parallel=parallel,
                                         add_residual_bytes=arch.conv_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_decoder += 3 * (2 * t_act2 + t_c7 + t_c1 + t_res_unit)
        in_dim = out_dim

    t_act_final, _ = activation_1d_top(bs=bs, length=cur_len, channels=in_dim, parallel=parallel, conv_bytes=arch.conv_bytes,
                                       granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    t_c_final, _ = conv1d_top(bs=bs, length=cur_len, cin=in_dim, cout=1, kernel=7, stride=1,
                              parallel=parallel, conv_bytes=arch.conv_bytes, granularity=granularity,
                              single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    t_decoder += t_act_final + t_c_final

    breakdown = {"transformer": t_transformer, "upsample": t_upsample, "decoder": t_decoder}
    total = sum(breakdown.values())
    log.info("code2wav chunk %s frames (+ctx %s) -> %s samples: %s s, %s", frames, arch.left_context if with_left_context else 0, cur_len, total, breakdown)
    return total, breakdown


# ---------------------------------------------------------------------------
# 端到端编排 (Phase 1: 同集群串行)
# ---------------------------------------------------------------------------

def modeling_omni_e2e(omni: Omni_Arch, workload: OmniWorkload, plan: OmniParallelPlan,
                      granularity: Modeling_Granularity, single_chip: Arch, noc_hierarchy: Hierarchy,
                      thinker_routing: "np.ndarray | None" = None, talker_routing: "np.ndarray | None" = None):
    """端到端串行编排。返回 metrics dict (含各组件 breakdown)。"""
    bs = workload.bs
    wl = workload

    # ---- 输入 token 换算 ----
    vision_patches = 0
    vision_tokens = 0
    if omni.vision_encoder is not None:
        ve = omni.vision_encoder
        if wl.num_images > 0:
            vision_patches += wl.num_images * ve.num_patches(wl.image_height, wl.image_width)
            vision_tokens += wl.num_images * ve.num_merged_tokens(wl.image_height, wl.image_width)
        if wl.video_seconds > 0:
            n_frames = math.ceil(wl.video_seconds * wl.video_fps)
            vision_patches += ve.num_patches(wl.video_height, wl.video_width, num_frames=n_frames)
            vision_tokens += ve.num_merged_tokens(wl.video_height, wl.video_width, num_frames=n_frames)

    audio_tokens = omni.audio_encoder.num_tokens(wl.audio_in_seconds) if (omni.audio_encoder is not None and wl.audio_in_seconds > 0) else 0
    seq_in = wl.text_in_tokens + vision_tokens + audio_tokens

    # ---- encoders ----
    t_vision, vision_bd = modeling_vision_encoder(omni.vision_encoder, bs, vision_patches, plan.vision, granularity, single_chip, noc_hierarchy) if vision_patches > 0 else (0.0, {})
    t_audio, audio_bd = modeling_audio_encoder(omni.audio_encoder, bs, wl.audio_in_seconds, plan.audio, granularity, single_chip, noc_hierarchy) if wl.audio_in_seconds > 0 else (0.0, {})

    # ---- thinker ----
    t_thinker_prefill, thinker_prefill_bd = modeling_llm_prefill(
        omni.thinker, bs, seq_in, plan.thinker_prefill, plan.thinker_prefill_atten, plan.thinker_moe,
        granularity, single_chip, noc_hierarchy, routing_array=thinker_routing)

    # text TPOT: 取 decode 中点 kv 为代表
    kv_mid = seq_in + wl.text_out_tokens // 2
    t_tpot, thinker_decode_bd = modeling_llm_decode_step(
        omni.thinker, bs, kv_mid, plan.thinker_decode, plan.thinker_moe,
        granularity, single_chip, noc_hierarchy, routing_array=thinker_routing)

    ttft = t_vision + t_audio + t_thinker_prefill

    metrics = {
        "seq_in": seq_in, "vision_tokens": vision_tokens, "audio_tokens": audio_tokens,
        "t_vision_encoder": t_vision, "t_audio_encoder": t_audio,
        "t_thinker_prefill": t_thinker_prefill,
        "TTFT": ttft, "text_TPOT": t_tpot,
        "t_text_generation": ttft + wl.text_out_tokens * t_tpot,
        "breakdown": {"vision": vision_bd, "audio": audio_bd, "thinker_prefill": thinker_prefill_bd, "thinker_decode_step": thinker_decode_bd},
    }

    # ---- 音频输出路径 (talker + code predictor + code2wav) ----
    if omni.talker is not None and wl.audio_out_seconds > 0:
        num_frames = omni.num_codec_frames(wl.audio_out_seconds)

        # resize MLP: thinker hidden -> talker hidden, 对 talker 消费的 prefix tokens 施加一次
        talker_prefix = seq_in + wl.talker_wait_text_tokens
        t_resize, _ = mlp_gelu_top(bs=bs, seq=talker_prefix, hidden=omni.thinker.hidden_size,
                                   up_hidden=omni.resize_mlp_hidden, parallel=plan.talker_prefill,
                                   next_parallel=plan.talker_prefill, mlp_bytes=omni.resize_mlp_bytes,
                                   granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy,
                                   out_hidden=omni.talker.hidden_size)

        t_talker_prefill, talker_prefill_bd = modeling_llm_prefill(
            omni.talker, bs, talker_prefix, plan.talker_prefill, plan.talker_prefill, plan.talker_moe,
            granularity, single_chip, noc_hierarchy, routing_array=talker_routing)

        # talker 每帧 1 步 decode (kv 取中点)
        talker_kv_mid = talker_prefix + num_frames // 2
        t_talker_step, talker_decode_bd = modeling_llm_decode_step(
            omni.talker, bs, talker_kv_mid, plan.talker_decode, plan.talker_moe,
            granularity, single_chip, noc_hierarchy, routing_array=talker_routing)

        # code predictor: 每帧 num_code_groups-1 步微型 decode (kv 很小)
        cp_steps = omni.code_predictor_steps_per_frame()
        t_cp_step, cp_bd = modeling_llm_decode_step(
            omni.code_predictor, bs, cached_kv=max(2, cp_steps), parallel=plan.code_predictor, moe_parallel=plan.code_predictor,
            granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
        t_frame = t_talker_step + cp_steps * t_cp_step

        # code2wav: 首 chunk + 稳态 chunk
        t_c2w_first, c2w_bd = modeling_code2wav_chunk(omni.code2wav, bs, wl.first_chunk_frames, plan.code2wav,
                                                      granularity, single_chip, noc_hierarchy, with_left_context=False)
        t_c2w_steady, _ = modeling_code2wav_chunk(omni.code2wav, bs, wl.first_chunk_frames, plan.code2wav,
                                                  granularity, single_chip, noc_hierarchy, with_left_context=True)
        t_c2w_per_frame = t_c2w_steady / wl.first_chunk_frames

        # 首音频包: TTFT + 等 thinker 出前几个 text token + talker prefill + 首 chunk 帧生成 + 首 chunk vocoder
        first_audio_latency = (ttft + wl.talker_wait_text_tokens * t_tpot + t_resize + t_talker_prefill
                               + wl.first_chunk_frames * t_frame + t_c2w_first)

        # 音频 RTF: 生成 1s 音频所需时间 / 1s (纯音频路径; 同集群串行下 thinker decode 也占用硬件, 单独给出)
        audio_rtf = omni.codec_frame_rate * (t_frame + t_c2w_per_frame)
        # 串行合成 RTF: 假设 text 与音频同速流式 (每秒音频伴随 tokens_per_sec_text 个 text token 串行执行)
        text_tokens_per_audio_sec = wl.text_out_tokens / max(wl.audio_out_seconds, 1e-9)
        audio_rtf_colocated = audio_rtf + text_tokens_per_audio_sec * t_tpot

        metrics.update({
            "num_codec_frames": num_frames,
            "t_resize_mlp": t_resize,
            "t_talker_prefill": t_talker_prefill,
            "t_talker_step": t_talker_step,
            "t_code_predictor_step": t_cp_step,
            "t_frame": t_frame,
            "t_code2wav_first_chunk": t_c2w_first,
            "t_code2wav_per_frame": t_c2w_per_frame,
            "first_audio_latency": first_audio_latency,
            "audio_RTF": audio_rtf,
            "audio_RTF_colocated": audio_rtf_colocated,
            "t_audio_generation_total": num_frames * (t_frame + t_c2w_per_frame),
        })
        metrics["breakdown"].update({"talker_prefill": talker_prefill_bd, "talker_decode_step": talker_decode_bd,
                                     "code_predictor_step": cp_bd, "code2wav_chunk": c2w_bd})

    return metrics


if __name__ == "__main__":
    import torch
    from mosaic.llm_arch import Qwen3_Omni_30b_a3b
    from mosaic.noc.noc_config_set import torus_mesh_switch_1
    from mosaic.arch import stacked_gpu_base

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    omni = Qwen3_Omni_30b_a3b()

    noc_hierarchy = torus_mesh_switch_1()
    single_chip = stacked_gpu_base()
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False, dump_perf_log=False)

    # 8 卡: encoder/talker/code2wav tp=8; thinker tp=8 / moe ep=8
    p_dense = ParallelScheme(tp=8, ep=1, sp=1, cp=1, dp=1, pp=1, fsdp=False)
    p_moe = ParallelScheme(tp=1, ep=8, sp=1, cp=1, dp=1, pp=1, fsdp=False)
    plan = OmniParallelPlan(
        vision=p_dense, audio=p_dense,
        thinker_prefill=p_dense, thinker_prefill_atten=p_dense, thinker_moe=p_moe,
        thinker_decode=p_dense,
        talker_prefill=p_dense, talker_decode=p_dense, talker_moe=p_moe,
        code_predictor=p_dense, code2wav=p_dense,
    )

    wl = OmniWorkload(
        bs=1,
        text_in_tokens=128,
        num_images=1, image_height=1080, image_width=1920,
        audio_in_seconds=30.0,
        text_out_tokens=256,
        audio_out_seconds=20.0,
    )

    metrics = modeling_omni_e2e(omni, wl, plan, granularity, single_chip, noc_hierarchy)

    print("=" * 60)
    for k, v in metrics.items():
        if k == "breakdown":
            continue
        print(f"{k}: {v}")
