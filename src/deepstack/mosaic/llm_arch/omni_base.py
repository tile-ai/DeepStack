# Base architecture class for Omni (multi-component multimodal) models.
#
# Qwen3-Omni-style models consist of several heterogeneous components:
#   vision encoder (bidirectional ViT) / audio encoder (windowed-attention AuT)
#   -> thinker (MoE causal LM, outputs text)
#   -> talker (MoE causal LM, consumes thinker hidden states at layer accept_hidden_layer, outputs 1 codec token per audio frame)
#   -> code predictor (small dense causal LM, autoregressively generates the remaining num_code_groups-1 codebooks per frame)
#   -> code2wav (sliding-window transformer + transposed-convolution upsampling, codec frames -> waveform)
#
# thinker / talker / code_predictor directly reuse llm_base.LLM_Arch,
# allowing existing gqa/moe/swiglu/rms_norm kernel models and parallelism schemes under mosaic/llm to be used unchanged.
# Consistent with the existing framework: embedding / lm_head are excluded from modeling.

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
    ffn_hidden: int                 # Non-gated MLP intermediate dimension (gelu, 2 GEMMs)
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int            # Hidden dimension output to thinker
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
    ffn_hidden: int                 # Non-gated MLP intermediate dimension (gelu)
    num_mel_bins: int
    downsample_hidden: int
    n_window: int                   # Training window (mel frames), chunk length = 2*n_window
    n_window_infer: int             # Attention block size during inference (number of after-cnn tokens)
    output_dim: int                 # Hidden dimension output to thinker
    conv_chunksize: int = 500
    mel_frame_rate: float = 100.0   # mel feature frame rate (Hz)
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
    ffn_hidden: int                 # Gated SwiGLU intermediate dimension
    num_head: int
    num_kv_head: int
    sliding_window: int
    codebook_size: int
    num_quantizers: int
    decoder_dim: int
    upsample_rates: tuple[int, ...]
    upsampling_ratios: tuple[int, ...]
    chunk_size: int = 300           # Default number of chunk frames for chunked_decode
    left_context: int = 25          # Default number of left-context frames for chunked_decode
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
        # codec frame rate x total upsampling factor, e.g., 12.5Hz * 1920 = 24kHz
        return None  # Derived from Omni_Arch.codec_frame_rate; not stored again here


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

    # Interface parameters between components
    talker_accept_hidden_layer: int = 0     # Index of the thinker layer whose hidden states are consumed by talker
    num_code_groups: int = 0                # Number of codec codebook groups per frame (talker produces the first group, code predictor produces the rest)
    codec_frame_rate: float = 12.5          # codec frame rate (Hz), audio frames per second
    position_id_per_seconds: int = 13
    resize_mlp_hidden: int = 0              # Non-gated resize MLP intermediate dimension for thinker hidden -> talker hidden
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

    # ---- Workload conversion helpers ----

    def num_codec_frames(self, audio_out_seconds: float) -> int:
        return math.ceil(audio_out_seconds * self.codec_frame_rate)

    def code_predictor_steps_per_frame(self) -> int:
        return self.num_code_groups - 1

    @property
    def output_sample_rate(self) -> float:
        assert self.code2wav is not None
        return self.codec_frame_rate * self.code2wav.total_upsample
