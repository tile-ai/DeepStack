# Qwen3-Omni-30B-A3B-Instruct
# 参数来源: ref/Qwen3-Omni-30B-A3B-Instruct.json
# 结构参考: transformers_modeling/modeling_qwen3_omni_moe.py (transformers v4.57.0)

import torch

from mosaic.utils import OpBytes, Tensor_Loc
from .llm_base import LLM_Arch, MoE_Arch, GQA_Arch, Dense_FFN_Arch
from .omni_base import Omni_Arch, ViT_Encoder_Arch, Audio_Encoder_Arch, Code2Wav_Arch


def _bf16(loc1="ddr", loc2="ddr", loc3="ddr"):
    return OpBytes(
        input1=Tensor_Loc(dtype=torch.bfloat16, loc=loc1),
        input2=Tensor_Loc(dtype=torch.bfloat16, loc=loc2),
        output=Tensor_Loc(dtype=torch.bfloat16, loc=loc3),
    )


def _make_thinker() -> LLM_Arch:
    # thinker_config.text_config: 与独立版 Qwen3-30B-A3B 同构
    gqa_arch = GQA_Arch(
        num_head=32,
        num_kv_head=4,
        head_dim=128,
        atten_bytes=_bf16(),
    )
    moe_arch = MoE_Arch(
        num_shared_experts=0,
        num_routed_experts=128,
        num_activated_experts=8,
        moe_down_hidden=768,
        expert_bytes=_bf16(),
        gate_bytes=_bf16(loc1="smem"),
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
    )
    rms_norm_bytes = OpBytes(
        input1=Tensor_Loc(dtype=torch.float32, loc="smem"),
        input2=Tensor_Loc(dtype=torch.float32, loc="ddr"),
        output=Tensor_Loc(dtype=torch.float16, loc="ddr"),
    )
    return LLM_Arch(
        hidden_size=2048,
        num_layer=48,
        num_dense_layer=0,
        num_moe_layer=48,
        max_seq_len=65536,
        name="Qwen3-Omni-30B-A3B-Thinker",
        rms_norm_bytes=rms_norm_bytes,
        rope_bytes=_bf16(),
        add_residual_bytes=_bf16(loc2="smem"),
        gqa_arch=gqa_arch,
        moe_arch=moe_arch,
    )


def _make_talker() -> LLM_Arch:
    # talker_config.text_config
    gqa_arch = GQA_Arch(
        num_head=16,
        num_kv_head=2,
        head_dim=128,
        atten_bytes=_bf16(),
    )
    # shared_expert_intermediate_size=768 = 2 x moe_intermediate_size(384);
    # moe_top 假设 shared 与 routed expert 同尺寸, 故按 2 个 shared expert 等效建模 (FLOPs/权重一致)
    moe_arch = MoE_Arch(
        num_shared_experts=2,
        num_routed_experts=128,
        num_activated_experts=6,
        moe_down_hidden=384,
        expert_bytes=_bf16(),
        gate_bytes=_bf16(loc1="smem"),
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
    )
    rms_norm_bytes = OpBytes(
        input1=Tensor_Loc(dtype=torch.float32, loc="smem"),
        input2=Tensor_Loc(dtype=torch.float32, loc="ddr"),
        output=Tensor_Loc(dtype=torch.float16, loc="ddr"),
    )
    return LLM_Arch(
        hidden_size=1024,
        num_layer=20,
        num_dense_layer=0,
        num_moe_layer=20,
        max_seq_len=65536,
        name="Qwen3-Omni-30B-A3B-Talker",
        rms_norm_bytes=rms_norm_bytes,
        rope_bytes=_bf16(),
        add_residual_bytes=_bf16(loc2="smem"),
        gqa_arch=gqa_arch,
        moe_arch=moe_arch,
    )


def _make_code_predictor() -> LLM_Arch:
    # talker_config.code_predictor_config: 5 层 dense causal LM,
    # 每帧 AR 生成剩余 15 个 codebook (MTP 式微型 decode)
    gqa_arch = GQA_Arch(
        num_head=16,
        num_kv_head=8,
        head_dim=128,
        atten_bytes=_bf16(),
    )
    dense_ffn_arch = Dense_FFN_Arch(
        up_hidden=3072,          # 门控 SwiGLU (Qwen3OmniMoeMLP)
        swiglu_bytes=_bf16(loc3="smem"),
    )
    rms_norm_bytes = OpBytes(
        input1=Tensor_Loc(dtype=torch.float32, loc="smem"),
        input2=Tensor_Loc(dtype=torch.float32, loc="ddr"),
        output=Tensor_Loc(dtype=torch.float16, loc="ddr"),
    )
    return LLM_Arch(
        hidden_size=1024,
        num_layer=5,
        num_dense_layer=5,
        num_moe_layer=0,
        max_seq_len=32768,
        name="Qwen3-Omni-30B-A3B-CodePredictor",
        rms_norm_bytes=rms_norm_bytes,
        rope_bytes=_bf16(),
        add_residual_bytes=_bf16(loc2="smem"),
        gqa_arch=gqa_arch,
        dense_ffn_arch=dense_ffn_arch,
    )


class Qwen3_Omni_30b_a3b(Omni_Arch):
    def __init__(self):
        vision_encoder = ViT_Encoder_Arch(
            depth=27,
            hidden_size=1152,
            num_head=16,
            ffn_hidden=4304,          # 非门控 gelu MLP
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            in_channels=3,
            num_deepstack_mergers=3,  # deepstack_visual_indexes [8, 16, 24]
            atten_bytes=_bf16(),
            mlp_bytes=_bf16(),
            conv_bytes=_bf16(),
        )
        audio_encoder = Audio_Encoder_Arch(
            num_layers=32,
            d_model=1280,
            num_head=20,
            ffn_hidden=5120,          # 非门控 gelu MLP
            num_mel_bins=128,
            downsample_hidden=480,
            n_window=50,
            n_window_infer=800,
            output_dim=2048,
            conv_chunksize=500,
            atten_bytes=_bf16(),
            mlp_bytes=_bf16(),
            conv_bytes=_bf16(),
        )
        code2wav = Code2Wav_Arch(
            num_layers=8,
            hidden_size=1024,
            ffn_hidden=3072,          # 门控 SwiGLU
            num_head=16,
            num_kv_head=16,
            sliding_window=72,
            codebook_size=2048,
            num_quantizers=16,
            decoder_dim=1536,
            upsample_rates=(8, 5, 4, 3),
            upsampling_ratios=(2, 2),
            atten_bytes=_bf16(),
            mlp_bytes=_bf16(),
            conv_bytes=_bf16(),
        )
        super().__init__(
            name="Qwen3-Omni-30B-A3B",
            thinker=_make_thinker(),
            talker=_make_talker(),
            code_predictor=_make_code_predictor(),
            vision_encoder=vision_encoder,
            audio_encoder=audio_encoder,
            code2wav=code2wav,
            talker_accept_hidden_layer=24,
            num_code_groups=16,
            codec_frame_rate=12.5,      # 25 token / (seconds_per_chunk=2s); 24kHz / total_upsample(1920)
            position_id_per_seconds=13,
            resize_mlp_hidden=2048,     # TalkerResizeMLP: thinker 2048 -> intermediate 2048 -> talker 1024
            resize_mlp_bytes=_bf16(),
        )


if __name__ == "__main__":
    m = Qwen3_Omni_30b_a3b()
    print(m.name)
    print("thinker:", m.thinker.name, m.thinker.hidden_size, m.thinker.num_layer)
    print("talker:", m.talker.name, m.talker.hidden_size, m.talker.num_layer)
    print("code_predictor:", m.code_predictor.name, m.code_predictor.num_layer)
    print("vision tokens (1080x1920):", m.vision_encoder.num_patches(1080, 1920), "->", m.vision_encoder.num_merged_tokens(1080, 1920))
    print("audio tokens (30s):", m.audio_encoder.num_tokens(30))
    print("codec frames (10s out):", m.num_codec_frames(10))
    print("output sample rate:", m.output_sample_rate)
