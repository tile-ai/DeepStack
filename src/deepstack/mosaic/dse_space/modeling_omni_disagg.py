# Qwen3-Omni 多组件模型的 Phase 2 编排: 分离式部署 (disaggregated)。
#
# 每个组件占独立设备组 (全局设备空间上连续的一段 rank), 组件间通过 NoC/网络传
# hidden state / codec code。与 Phase 1 (同集群串行) 的区别:
#   - 各组件计算互不抢占 -> thinker decode 与 talker/code2wav 跨请求/跨帧流水重叠
#   - 组件间传输显式建模 (TrafficMatrix 点对点, 走全局 noc_hierarchy)
#   - 输出稳态吞吐 (受最慢 stage 限制) 与流水化的延迟链
import math
from dataclasses import dataclass

import numpy as np

from mosaic.llm_arch.omni_base import Omni_Arch
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, get_extend_max_routes_with_traffic
from mosaic.utils import Modeling_Granularity
from tilesight.arch import Arch

from .modeling_omni import (
    OmniWorkload,
    make_uniform_routing,
    modeling_audio_encoder,
    modeling_code2wav_chunk,
    modeling_llm_decode_step,
    modeling_llm_prefill,
    modeling_vision_encoder,
)
from mosaic.llm.mlp_gelu import mlp_gelu_top

import logging
log = logging.getLogger(__name__)


@dataclass
class OmniStagePlacement:
    """一个组件 stage 的部署: 全局 rank 区间 [offset, offset+size) + 组内并行方案。
    parallel.world_size() 必须等于 size。moe_parallel 为 None 时用 parallel。"""
    offset: int
    size: int
    parallel: ParallelScheme
    moe_parallel: "ParallelScheme | None" = None
    atten_parallel: "ParallelScheme | None" = None

    def __post_init__(self):
        assert self.parallel.world_size() == self.size, \
            f"parallel world_size {self.parallel.world_size()} != stage size {self.size}"

    @property
    def moe(self) -> ParallelScheme:
        return self.moe_parallel if self.moe_parallel is not None else self.parallel

    @property
    def atten(self) -> ParallelScheme:
        return self.atten_parallel if self.atten_parallel is not None else self.parallel


@dataclass
class OmniDisaggPlacement:
    """全部组件的放置。各 stage 的 rank 区间不应重叠 (由调用者保证)。"""
    vision: "OmniStagePlacement | None"
    audio: "OmniStagePlacement | None"
    thinker: OmniStagePlacement
    talker: "OmniStagePlacement | None"
    code_predictor: "OmniStagePlacement | None"
    code2wav: "OmniStagePlacement | None"

    def total_devices(self) -> int:
        stages = [self.vision, self.audio, self.thinker, self.talker, self.code_predictor, self.code2wav]
        return sum(s.size for s in stages if s is not None)


def _inter_stage_transfer_time(bytes_total: float, src: "OmniStagePlacement | None", dst: "OmniStagePlacement | None",
                               total_devices: int, noc_hierarchy: Hierarchy):
    """组件间激活传输: 源组每设备持有 1/n_src 分片, 均匀发往目的组; 总流量 = bytes_total。"""
    if src is None or dst is None or bytes_total <= 0:
        return 0.0
    tm = TrafficMatrix(total_devices)
    per_pair = bytes_total / (src.size * dst.size)
    for i in range(src.offset, src.offset + src.size):
        for j in range(dst.offset, dst.offset + dst.size):
            tm.add(i, j, per_pair)
    _, _, overall_time, _ = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
    return overall_time


def modeling_omni_disagg(omni: Omni_Arch, workload: OmniWorkload, placement: OmniDisaggPlacement,
                         granularity: Modeling_Granularity, single_chip: Arch, noc_hierarchy: Hierarchy,
                         thinker_routing: "np.ndarray | None" = None, talker_routing: "np.ndarray | None" = None):
    """分离式部署的端到端建模。返回 metrics dict。
    noc_hierarchy 是覆盖全部 total_devices 的全局层级拓扑。"""
    bs = workload.bs
    wl = workload
    total_devices = placement.total_devices()
    act_bytes = 2  # bf16 激活

    # ---- 输入 token 换算 (与 Phase 1 一致) ----
    vision_patches, vision_tokens = 0, 0
    if omni.vision_encoder is not None and placement.vision is not None:
        ve = omni.vision_encoder
        if wl.num_images > 0:
            vision_patches += wl.num_images * ve.num_patches(wl.image_height, wl.image_width)
            vision_tokens += wl.num_images * ve.num_merged_tokens(wl.image_height, wl.image_width)
        if wl.video_seconds > 0:
            n_frames = math.ceil(wl.video_seconds * wl.video_fps)
            vision_patches += ve.num_patches(wl.video_height, wl.video_width, num_frames=n_frames)
            vision_tokens += ve.num_merged_tokens(wl.video_height, wl.video_width, num_frames=n_frames)
    audio_tokens = omni.audio_encoder.num_tokens(wl.audio_in_seconds) if (omni.audio_encoder is not None and placement.audio is not None and wl.audio_in_seconds > 0) else 0
    seq_in = wl.text_in_tokens + vision_tokens + audio_tokens

    # ---- 各 stage 计算时间 (复用 Phase 1 组件函数, 各自的并行方案) ----
    t_vision, _ = modeling_vision_encoder(omni.vision_encoder, bs, vision_patches, placement.vision.parallel,
                                          granularity, single_chip, noc_hierarchy) if vision_patches > 0 else (0.0, {})
    t_audio, _ = modeling_audio_encoder(omni.audio_encoder, bs, wl.audio_in_seconds, placement.audio.parallel,
                                        granularity, single_chip, noc_hierarchy) if audio_tokens > 0 else (0.0, {})

    th = placement.thinker
    t_thinker_prefill, _ = modeling_llm_prefill(omni.thinker, bs, seq_in, th.parallel, th.atten, th.moe,
                                                granularity, single_chip, noc_hierarchy, routing_array=thinker_routing)
    kv_mid = seq_in + wl.text_out_tokens // 2
    t_tpot, _ = modeling_llm_decode_step(omni.thinker, bs, kv_mid, th.parallel, th.moe,
                                         granularity, single_chip, noc_hierarchy, routing_array=thinker_routing)

    # ---- 组件间传输 ----
    t_xfer_vis = _inter_stage_transfer_time(bs * vision_tokens * omni.thinker.hidden_size * act_bytes,
                                            placement.vision, placement.thinker, total_devices, noc_hierarchy)
    t_xfer_aud = _inter_stage_transfer_time(bs * audio_tokens * omni.thinker.hidden_size * act_bytes,
                                            placement.audio, placement.thinker, total_devices, noc_hierarchy)

    # encoder 两路可并行 (不同设备组)
    ttft = max(t_vision + t_xfer_vis, t_audio + t_xfer_aud) + t_thinker_prefill

    metrics = {
        "seq_in": seq_in, "vision_tokens": vision_tokens, "audio_tokens": audio_tokens,
        "t_vision_encoder": t_vision, "t_audio_encoder": t_audio,
        "t_xfer_vision_to_thinker": t_xfer_vis, "t_xfer_audio_to_thinker": t_xfer_aud,
        "t_thinker_prefill": t_thinker_prefill,
        "TTFT": ttft, "text_TPOT": t_tpot,
    }

    # ---- 音频输出路径 ----
    if omni.talker is not None and placement.talker is not None and wl.audio_out_seconds > 0:
        num_frames = omni.num_codec_frames(wl.audio_out_seconds)
        tk = placement.talker
        cp = placement.code_predictor
        cw = placement.code2wav

        talker_prefix = seq_in + wl.talker_wait_text_tokens
        t_resize, _ = mlp_gelu_top(bs=bs, seq=talker_prefix, hidden=omni.thinker.hidden_size,
                                   up_hidden=omni.resize_mlp_hidden, parallel=tk.parallel, next_parallel=tk.parallel,
                                   mlp_bytes=omni.resize_mlp_bytes, granularity=granularity,
                                   single_chip=single_chip, noc_hierarchy=noc_hierarchy,
                                   out_hidden=omni.talker.hidden_size)
        t_talker_prefill, _ = modeling_llm_prefill(omni.talker, bs, talker_prefix, tk.parallel, tk.atten, tk.moe,
                                                   granularity, single_chip, noc_hierarchy, routing_array=talker_routing)
        talker_kv_mid = talker_prefix + num_frames // 2
        t_talker_step, _ = modeling_llm_decode_step(omni.talker, bs, talker_kv_mid, tk.parallel, tk.moe,
                                                    granularity, single_chip, noc_hierarchy, routing_array=talker_routing)

        cp_steps = omni.code_predictor_steps_per_frame()
        t_cp_step, _ = modeling_llm_decode_step(omni.code_predictor, bs, cached_kv=max(2, cp_steps),
                                                parallel=cp.parallel, moe_parallel=cp.moe,
                                                granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

        t_c2w_first, _ = modeling_code2wav_chunk(omni.code2wav, bs, wl.first_chunk_frames, cw.parallel,
                                                 granularity, single_chip, noc_hierarchy, with_left_context=False)
        t_c2w_steady, _ = modeling_code2wav_chunk(omni.code2wav, bs, wl.first_chunk_frames, cw.parallel,
                                                  granularity, single_chip, noc_hierarchy, with_left_context=True)
        t_c2w_per_frame = t_c2w_steady / wl.first_chunk_frames

        # 传输: thinker -> talker (prefix hidden, 一次性), talker -> cp (每帧), cp -> c2w (codes, 每帧)
        t_xfer_th_tk = _inter_stage_transfer_time(bs * talker_prefix * omni.thinker.hidden_size * act_bytes,
                                                  placement.thinker, placement.talker, total_devices, noc_hierarchy)
        t_xfer_tk_cp = _inter_stage_transfer_time(bs * omni.talker.hidden_size * act_bytes,
                                                  placement.talker, placement.code_predictor, total_devices, noc_hierarchy)
        t_xfer_cp_cw = _inter_stage_transfer_time(bs * omni.num_code_groups * 4.0,
                                                  placement.code_predictor, placement.code2wav, total_devices, noc_hierarchy)

        # 每帧: talker / code predictor / code2wav 在不同设备组上, 跨帧流水;
        # 帧节拍受最慢者限制, 单帧延迟为三段之和 (+ 帧级传输)
        frame_stage_times = [t_talker_step, cp_steps * t_cp_step + t_xfer_tk_cp, t_c2w_per_frame + t_xfer_cp_cw]
        t_frame_pipelined = max(frame_stage_times)
        t_frame_latency = sum(frame_stage_times)

        # 首音频包: encoder -> thinker prefill -> 等 wait 个 text token -> 传 hidden -> talker prefill
        #           -> 首 chunk 逐帧生成 (帧级流水: 首帧延迟 + (n-1) 个节拍) -> 首 chunk vocoder
        first_audio_latency = (ttft + wl.talker_wait_text_tokens * t_tpot + t_xfer_th_tk + t_resize + t_talker_prefill
                               + t_frame_latency + (wl.first_chunk_frames - 1) * max(t_talker_step, cp_steps * t_cp_step + t_xfer_tk_cp)
                               + t_c2w_first)

        # 稳态音频 RTF: 帧节拍 x 帧率 (thinker decode 在独立设备组, 不占音频路径)
        audio_rtf = omni.codec_frame_rate * t_frame_pipelined

        # 稳态吞吐 (连续请求流): 每请求各 stage 忙时, 受最慢 stage 限制
        stage_busy = {
            "vision": t_vision,
            "audio": t_audio,
            "thinker": t_thinker_prefill + wl.text_out_tokens * t_tpot,
            "talker": t_talker_prefill + num_frames * t_talker_step,
            "code_predictor": num_frames * cp_steps * t_cp_step,
            "code2wav": num_frames * t_c2w_per_frame,
        }
        bottleneck_stage = max(stage_busy, key=stage_busy.get)
        throughput_rps = 1.0 / stage_busy[bottleneck_stage]

        metrics.update({
            "num_codec_frames": num_frames,
            "t_xfer_thinker_to_talker": t_xfer_th_tk,
            "t_xfer_talker_to_cp": t_xfer_tk_cp,
            "t_xfer_cp_to_code2wav": t_xfer_cp_cw,
            "t_talker_prefill": t_talker_prefill,
            "t_talker_step": t_talker_step,
            "t_code_predictor_step": t_cp_step,
            "t_frame_pipelined": t_frame_pipelined,
            "t_frame_latency": t_frame_latency,
            "t_code2wav_per_frame": t_c2w_per_frame,
            "first_audio_latency": first_audio_latency,
            "audio_RTF": audio_rtf,
            "stage_busy_per_request": stage_busy,
            "bottleneck_stage": bottleneck_stage,
            "steady_state_throughput_rps": throughput_rps,
        })

    return metrics


if __name__ == "__main__":
    from mosaic.llm_arch import Qwen3_Omni_30b_a3b
    from mosaic.noc.noc_config_set import torus_mesh_switch_1
    from mosaic.arch import stacked_gpu_base

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    omni = Qwen3_Omni_30b_a3b()

    # 16 设备全局拓扑: thinker 8, talker 4, vision 1, audio 1, cp 1, c2w 1
    noc_hierarchy = torus_mesh_switch_1()
    single_chip = stacked_gpu_base()
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False, dump_perf_log=False)

    def p(tp=1, ep=1, dp=1):
        return ParallelScheme(tp=tp, ep=ep, sp=1, cp=1, dp=dp, pp=1, fsdp=False)

    placement = OmniDisaggPlacement(
        vision=OmniStagePlacement(offset=0, size=1, parallel=p()),
        audio=OmniStagePlacement(offset=1, size=1, parallel=p()),
        thinker=OmniStagePlacement(offset=2, size=8, parallel=p(tp=8), moe_parallel=p(ep=8)),
        talker=OmniStagePlacement(offset=10, size=4, parallel=p(tp=4), moe_parallel=p(ep=4)),
        code_predictor=OmniStagePlacement(offset=14, size=1, parallel=p()),
        code2wav=OmniStagePlacement(offset=15, size=1, parallel=p()),
    )

    wl = OmniWorkload(
        bs=1,
        text_in_tokens=128,
        num_images=1, image_height=1080, image_width=1920,
        audio_in_seconds=30.0,
        text_out_tokens=256,
        audio_out_seconds=20.0,
    )

    metrics = modeling_omni_disagg(omni, wl, placement, granularity, single_chip, noc_hierarchy)
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v}")
