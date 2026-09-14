#
#

import math
from dataclasses import dataclass, field
from typing import Optional

from mosaic.utils import OpBytes
from .llm_base import LLM_Arch, default_op_bytes


@dataclass
class ViT_Encoder_Arch:
    """Bidirectional ViT encoder (Qwen3OmniMoeVisionEncoder).
    Patch embedding uses Conv3d with kernel=stride=[temporal_patch, patch, patch].
    Each of depth blocks contains bidirectional MHA without a KV cache and a
    non-gated GELU MLP, hidden -> ffn_hidden -> hidden. The patch merger is LN,
    Linear(hidden*merge^2, hidden*merge^2), GELU, and Linear to out_hidden.
    Attach equivalent mergers to num_deepstack_mergers intermediate layers.
    """
    depth: int
    hidden_size: int
    num_head: int
    ffn_hidden: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
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
        """Return the number of pre-merge patch tokens seen by the transformer.
        Images use num_frames=1, padded temporally to temporal_patch_size.
        """
        grid_h = math.ceil(height / self.patch_size)
        grid_w = math.ceil(width / self.patch_size)
        grid_t = max(1, math.ceil(num_frames / self.temporal_patch_size))
        return grid_t * grid_h * grid_w

    def num_merged_tokens(self, height: int, width: int, num_frames: int = 1) -> int:
        """Return the number of tokens entering the thinker after spatial merging."""
        return math.ceil(self.num_patches(height, width, num_frames) / (self.spatial_merge_size ** 2))


@dataclass
class Audio_Encoder_Arch:
    """Window-attention audio encoder (Qwen3OmniMoeAudioEncoder / AuT).
    The frontend applies three Conv2d layers with kernel=3 and stride=2 over
    (mel_bins,time), using 1 -> downsample_hidden -> downsample_hidden -> downsample_hidden
    channels. This downsamples time by 8 (100 Hz mel to 12.5 Hz tokens), followed by
    Linear(downsample_hidden * mel_bins/8, d_model). Each encoder layer combines
    bidirectional MHA within n_window_infer-sized blocks and a non-gated GELU MLP.
    The output is Linear(d_model,d_model), GELU, then Linear(d_model,output_dim).
    """
    num_layers: int
    d_model: int
    num_head: int
    ffn_hidden: int
    num_mel_bins: int
    downsample_hidden: int
    n_window: int
    n_window_infer: int
    output_dim: int
    conv_chunksize: int = 500
    mel_frame_rate: float = 100.0
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
        """Return the encoder output token count, approximately 12.5 tokens/s,
        consistent with the three stride-2 frontend convolutions.
        """
        t = self.num_mel_frames(seconds)
        for _ in range(3):
            t = (t - 1) // 2 + 1
        return t


@dataclass
class Code2Wav_Arch:
    """Codec-token-to-waveform vocoder (Qwen3OmniMoeCode2Wav).
    Average lookups from num_quantizers codebooks; these are memory operations,
    not GEMMs. The pre-transformer uses causal sliding-window MHA and gated SWiGLU.
    Each upsampling ratio r adds ConvTranspose1d(hidden,hidden,kernel=r,stride=r)
    and a ConvNeXt block: depthwise Conv1d(kernel=7), LN, pointwise
    hidden -> 4*hidden -> hidden convolutions, and GELU.
    The decoder starts with Conv1d(hidden,decoder_dim,7). For each upsample rate r_i,
    a DecoderBlock halves channels using ConvTranspose1d(kernel=2*r_i,stride=r_i)
    and three residual units with kernel-7 dilations {1,3,9}, kernel-1 convolutions,
    and SnakeBeta. A final kernel-7 convolution produces one channel.
    Streaming processes chunk_size frames with left_context frames.
    """
    num_layers: int
    hidden_size: int
    ffn_hidden: int
    num_head: int
    num_kv_head: int
    sliding_window: int
    codebook_size: int
    num_quantizers: int
    decoder_dim: int
    upsample_rates: tuple[int, ...]
    upsampling_ratios: tuple[int, ...]
    chunk_size: int = 300
    left_context: int = 25
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
        return None


@dataclass
class Omni_Arch:
    """Container for an omni model's components and dependencies:
    vision/audio encoder -> thinker -> resize MLP -> talker -> code predictor -> code2wav.
    """
    name: str
    thinker: LLM_Arch
    talker: Optional[LLM_Arch] = None
    code_predictor: Optional[LLM_Arch] = None
    vision_encoder: Optional[ViT_Encoder_Arch] = None
    audio_encoder: Optional[Audio_Encoder_Arch] = None
    code2wav: Optional[Code2Wav_Arch] = None

    talker_accept_hidden_layer: int = 0
    num_code_groups: int = 0
    codec_frame_rate: float = 12.5
    position_id_per_seconds: int = 13
    resize_mlp_hidden: int = 0
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


    def num_codec_frames(self, audio_out_seconds: float) -> int:
        return math.ceil(audio_out_seconds * self.codec_frame_rate)

    def code_predictor_steps_per_frame(self) -> int:
        return self.num_code_groups - 1

    @property
    def output_sample_rate(self) -> float:
        assert self.code2wav is not None
        return self.codec_frame_rate * self.code2wav.total_upsample
