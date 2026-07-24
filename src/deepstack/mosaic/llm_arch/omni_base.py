# Omni (multi-component multimodal) 模型的架构基类。
#
# Qwen3-Omni 类模型由多个异构组件组成:
#   vision encoder (双向 ViT) / audio encoder (窗口注意力 AuT)
#   -> thinker (MoE causal LM, 输出 text)
#   -> talker (MoE causal LM, 消费 thinker 第 accept_hidden_layer 层 hidden state, 每音频帧输出 1 个 codec token)
#   -> code predictor (小型 dense causal LM, 每帧 AR 生成剩余 num_code_groups-1 个 codebook)
#   -> code2wav (sliding-window transformer + 转置卷积上采样, codec 帧 -> 波形)
#
# thinker / talker / code_predictor 直接复用 llm_base.LLM_Arch,
# 使 mosaic/llm 下现有的 gqa/moe/swiglu/rms_norm 等 kernel 模型与并行方案原样可用。
# 与现有框架保持一致: embedding / lm_head 不计入建模。

import math
from dataclasses import dataclass, field
from typing import Optional

from mosaic.utils import OpBytes
from .llm_base import LLM_Arch, default_op_bytes


@dataclass
class ViT_Encoder_Arch:
    """双向注意力 ViT encoder (Qwen3OmniMoeVisionEncoder)。

    - patch embed: Conv3d(in_channels, hidden, kernel=stride=[temporal_patch, patch, patch])
    - depth x [MHA (双向, 无 KV cache) + 非门控 gelu MLP (hidden -> ffn_hidden -> hidden)]
    - patch merger: LN + Linear(hidden*merge^2, hidden*merge^2) + gelu + Linear(hidden*merge^2, out_hidden)
      deepstack 在 num_deepstack_mergers 个中间层各挂一个同结构 merger
    """
    depth: int
    hidden_size: int
    num_head: int
    ffn_hidden: int                 # 非门控 MLP intermediate (gelu, 2 GEMM)
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int            # 输出到 thinker 的 hidden
    in_channels: int = 3
    num_deepstack_mergers: int = 0  # len(deepstack_visual_indexes)
    atten_bytes: OpBytes = field(default_factory=default_op_bytes)
    mlp_bytes: OpBytes = field(default_factory=default_op_bytes)
    conv_bytes: OpBytes = field(default_factory=default_op_bytes)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_head

    @property
    def merger_hidden(self) -> int:
        return self.hidden_size * (self.spatial_merge_size ** 2)

    def num_patches(self, height: int, width: int, num_frames: int = 1) -> int:
        """transformer 看到的 patch token 数 (merge 之前)。图片按 num_frames=1 (temporal 维 pad 到 temporal_patch_size)。"""
        grid_h = math.ceil(height / self.patch_size)
        grid_w = math.ceil(width / self.patch_size)
        grid_t = max(1, math.ceil(num_frames / self.temporal_patch_size))
        return grid_t * grid_h * grid_w

    def num_merged_tokens(self, height: int, width: int, num_frames: int = 1) -> int:
        """进入 thinker 的 token 数 (spatial merge 之后)。"""
        return math.ceil(self.num_patches(height, width, num_frames) / (self.spatial_merge_size ** 2))


@dataclass
class Audio_Encoder_Arch:
    """窗口注意力 audio encoder (Qwen3OmniMoeAudioEncoder / AuT)。

    - 前端: 3x Conv2d(k=3, s=2) 在 (mel_bins, time) 上, 1->downsample_hidden->downsample_hidden->downsample_hidden,
      时间 8x 下采样 (100Hz mel -> 12.5Hz token); 之后 Linear(downsample_hidden * mel_bins/8, d_model)
    - num_layers x [双向 MHA (块内注意力, 块大小 n_window_infer) + 非门控 gelu MLP (d_model -> ffn_hidden -> d_model)]
    - 出口: proj1 Linear(d_model, d_model) + gelu + proj2 Linear(d_model, output_dim)
    """
    num_layers: int
    d_model: int
    num_head: int
    ffn_hidden: int                 # 非门控 MLP intermediate (gelu)
    num_mel_bins: int
    downsample_hidden: int
    n_window: int                   # 训练窗口 (mel 帧), chunk 长度 = 2*n_window
    n_window_infer: int             # 推理时注意力块大小 (after-cnn token 数)
    output_dim: int                 # 输出到 thinker 的 hidden
    conv_chunksize: int = 500
    mel_frame_rate: float = 100.0   # mel 特征帧率 (Hz)
    atten_bytes: OpBytes = field(default_factory=default_op_bytes)
    mlp_bytes: OpBytes = field(default_factory=default_op_bytes)
    conv_bytes: OpBytes = field(default_factory=default_op_bytes)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_head

    @property
    def downsampled_mel_bins(self) -> int:
        f = self.num_mel_bins
        for _ in range(3):
            f = (f + 1) // 2
        return f

    def num_mel_frames(self, seconds: float) -> int:
        return math.ceil(seconds * self.mel_frame_rate)

    def num_tokens(self, seconds: float) -> int:
        """encoder 输出 token 数 (~12.5 token/s), 与 conv 前端 3x stride-2 一致。"""
        t = self.num_mel_frames(seconds)
        for _ in range(3):
            t = (t - 1) // 2 + 1
        return t


@dataclass
class Code2Wav_Arch:
    """codec token -> 波形 的 vocoder (Qwen3OmniMoeCode2Wav)。

    - code embedding: num_quantizers 组 codebook 查表取 mean (访存, 不计 GEMM)
    - pre_transformer: num_layers x [causal MHA (sliding_window) + 门控 SwiGLU MLP (hidden -> ffn_hidden)]
    - upsample: 对每个 upsampling_ratio r: ConvTranspose1d(hidden, hidden, k=r, s=r) + ConvNeXt block
      (depthwise Conv1d k=7 + LN + pwconv hidden->4*hidden->hidden, gelu)
    - decoder: Conv1d(hidden, decoder_dim, 7); 对每个 upsample_rate r_i:
      DecoderBlock(in=decoder_dim/2^i, out=decoder_dim/2^(i+1)):
        ConvTranspose1d(in, out, 2*r_i, r_i) + 3x ResidualUnit(Conv1d k=7 dilated {1,3,9} + Conv1d k=1, SnakeBeta)
      最后 Conv1d(decoder_dim/2^len, 1, 7)
    - chunked streaming decode: chunk_size 帧 / left_context 帧
    """
    num_layers: int
    hidden_size: int
    ffn_hidden: int                 # 门控 SwiGLU intermediate
    num_head: int
    num_kv_head: int
    sliding_window: int
    codebook_size: int
    num_quantizers: int
    decoder_dim: int
    upsample_rates: tuple[int, ...]
    upsampling_ratios: tuple[int, ...]
    chunk_size: int = 300           # chunked_decode 默认 chunk 帧数
    left_context: int = 25          # chunked_decode 默认左上下文帧数
    atten_bytes: OpBytes = field(default_factory=default_op_bytes)
    mlp_bytes: OpBytes = field(default_factory=default_op_bytes)
    conv_bytes: OpBytes = field(default_factory=default_op_bytes)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_head

    @property
    def total_upsample(self) -> int:
        r = 1
        for x in self.upsample_rates:
            r *= x
        for x in self.upsampling_ratios:
            r *= x
        return r

    @property
    def output_sample_rate(self) -> float:
        # codec 帧率 x 总上采样倍数, e.g. 12.5Hz * 1920 = 24kHz
        return None  # 由 Omni_Arch.codec_frame_rate 推得, 此处不重复存


@dataclass
class Omni_Arch:
    """多组件 omni 模型容器。组件间依赖:
    vision/audio encoder -> thinker -> (resize MLP) -> talker -> code predictor -> code2wav
    """
    name: str
    thinker: LLM_Arch
    talker: Optional[LLM_Arch] = None
    code_predictor: Optional[LLM_Arch] = None
    vision_encoder: Optional[ViT_Encoder_Arch] = None
    audio_encoder: Optional[Audio_Encoder_Arch] = None
    code2wav: Optional[Code2Wav_Arch] = None

    # 组件间接口参数
    talker_accept_hidden_layer: int = 0     # talker 消费 thinker 的第几层 hidden state
    num_code_groups: int = 0                # 每帧 codec codebook 组数 (talker 出第 1 组, code predictor 出其余)
    codec_frame_rate: float = 12.5          # codec 帧率 (Hz), 音频每秒帧数
    position_id_per_seconds: int = 13
    resize_mlp_hidden: int = 0              # thinker hidden -> talker hidden 的非门控 resize MLP intermediate
    resize_mlp_bytes: OpBytes = field(default_factory=default_op_bytes)

    def __post_init__(self):
        if self.talker is not None and self.code_predictor is None:
            raise ValueError("talker 存在时必须提供 code_predictor (残差 codebook 生成)。")
        if self.talker is not None and self.code2wav is None:
            raise ValueError("talker 存在时必须提供 code2wav (codec -> 波形)。")
        if self.talker is not None and self.num_code_groups <= 0:
            raise ValueError("talker 存在时 num_code_groups 必须 > 0。")
        if self.code2wav is not None and self.code2wav.num_quantizers != self.num_code_groups:
            raise ValueError("code2wav.num_quantizers 必须等于 num_code_groups。")

    # ---- workload 换算辅助 ----

    def num_codec_frames(self, audio_out_seconds: float) -> int:
        return math.ceil(audio_out_seconds * self.codec_frame_rate)

    def code_predictor_steps_per_frame(self) -> int:
        return self.num_code_groups - 1

    @property
    def output_sample_rate(self) -> float:
        assert self.code2wav is not None
        return self.codec_frame_rate * self.code2wav.total_upsample
