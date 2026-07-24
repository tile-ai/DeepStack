import json
import os
from typing import Any, Dict

from .llm_config import (
    LLMModelSpec, RopeConfig, AttentionConfig, DenseFFNConfig, MoEConfig,
    LayerSpec, make_gqa_attention, make_mla_attention, make_dense_ffn, make_moe_ffn
)


def build_llama_3_3_70b_spec() -> LLMModelSpec:
    """直接构建 Llama 3.3-70B 规范对象（不依赖 JSON 文件）。"""
    # 来自 Llama-3.3-70B-Instruct.config.json 的关键参数
    hidden_size = 8192
    num_layers = 80
    num_heads = 64
    num_kv_heads = 8
    head_dim = 128
    inter_size = 28672
    rms_eps = 1e-5
    max_pos = 131072
    vocab_size = 128256
    weight_dtype = "bf16"  # torch_dtype: bfloat16
    attention_dropout = 0.0
    activation = "silu"

    rope = RopeConfig(
        rope_type="llama3",
        rope_theta=500000.0,
        rope_scaling_factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    )

    # Llama 使用 GQA 注意力 + Dense FFN
    attn = make_gqa_attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dropout=attention_dropout,
        qk_norm=None,
        qkv_bias=None,
    )
    dense = make_dense_ffn(intermediate_size=inter_size, activation=activation)
    moe = MoEConfig()  # 不使用 MoE

    spec = LLMModelSpec(
        arch_name="llama-3.3-70b",
        model_family="llama",
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        rms_norm_eps=rms_eps,
        max_seq_len=max_pos,
        vocab_size=vocab_size,
        weight_dtype=weight_dtype,
        rope=rope,
        attention=attn,
        dense_ffn=dense,
        moe_ffn=moe,
        extra={
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
            "initializer_range": 0.02,
            "mlp_bias": False,
            "pretraining_tp": 1,
            "tie_word_embeddings": False,
            "use_cache": True,
            "transformers_version": "4.47.0.dev0",
        },
    )

    # 不逐层展开，仅统计 dense/moe 层数
    spec.num_dense_layers = num_layers
    spec.num_moe_layers = 0

    return spec



if __name__ == "__main__":
#     spec = build_spec_from_json("llm_configs/Llama-3.3-70B-Instruct.config.json")
    spec = build_llama_3_3_70b_spec()
    print(spec)
    print(spec.hidden_size)
    print(spec.num_hidden_layers)
    print(spec.num_dense_layers)
    print(spec.num_moe_layers)
    print(spec.rms_norm_eps)
    print(spec.max_seq_len)
    print(spec.vocab_size)
    print(spec.weight_dtype)
    print(spec.rope)
    print(spec.attention)
    print(spec.hidden_size)
    print(spec.dense_ffn.intermediate_size)