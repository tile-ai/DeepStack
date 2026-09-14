from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# ----- Basic enums/constants (use str for JSON/external integration) -----

WeightDType = str         # e.g. "fp32", "fp16", "bf16", "int8", "fp8-e4m3"
ActivationType = str      # e.g. "silu", "gelu", "relu", "swiglu"
RopeType = str            # e.g. "llama", "llama3", "yarn"
AttentionType = str       # "gqa" | "mla"
FFNType = str             # "dense" | "moe"


# ----- Detailed sub-configurations -----

@dataclass
class RopeConfig:
    rope_type: Optional[RopeType] = None
    rope_theta: Optional[float] = None
    rope_scaling_factor: Optional[float] = None
    low_freq_factor: Optional[float] = None
    high_freq_factor: Optional[float] = None
    original_max_position_embeddings: Optional[int] = None


@dataclass
class AttentionConfig:
    attention_type: AttentionType = "gqa"
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None  # Used only by gqa
    head_dim: Optional[int] = None
    dropout: Optional[float] = None
    qk_norm: Optional[bool] = None
    qkv_bias: Optional[bool] = None


@dataclass
class DenseFFNConfig:
    ffn_type: FFNType = "dense"
    intermediate_size: Optional[int] = None
    activation: Optional[ActivationType] = None


@dataclass
class MoEConfig:
    ffn_type: FFNType = "moe"
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    expert_intermediate_size: Optional[int] = None
    router_aux_loss_coef: Optional[float] = None
    topk_group: Optional[int] = None
    topk_method: Optional[str] = None
    norm_topk_prob: Optional[bool] = None


@dataclass
class LayerSpec:
    index: int
    has_attention: bool = True
    attention: Optional[AttentionConfig] = None
    ffn: Optional[DenseFFNConfig] = None
    moe: Optional[MoEConfig] = None


@dataclass
class LLMModelSpec:
    # Core metadata
    arch_name: str                   # For example, "llama", "qwen3_moe", "deepseek_v3"
    model_family: str                # For example, "llama", "qwen", "deepseek"

    # Global structure parameters
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    # Layer counts aggregated by type (used when layers are not enumerated individually)
    num_dense_layers: Optional[int] = None
    num_moe_layers: Optional[int] = None
    rms_norm_eps: Optional[float] = None
    max_seq_len: Optional[int] = None
    vocab_size: Optional[int] = None

    # Data types/weights
    weight_dtype: Optional[WeightDType] = None

    # Positional encoding
    rope: RopeConfig = field(default_factory=RopeConfig)

    # Attention (a global template by default, overridable by Layer)
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    # Feed-forward (global Dense or MoE templates, either overridable by Layer)
    dense_ffn: DenseFFNConfig = field(default_factory=DenseFFNConfig)
    moe_ffn: MoEConfig = field(default_factory=MoEConfig)

    # Per-layer specifications (if empty, infer from the global templates)
    layers: List[LayerSpec] = field(default_factory=list)

    # Other configuration
    extra: Dict[str, Any] = field(default_factory=dict)


def make_gqa_attention(num_heads: int, num_kv_heads: Optional[int], head_dim: int, dropout: float = 0.0, qk_norm: Optional[bool] = None, qkv_bias: Optional[bool] = None) -> AttentionConfig:
    return AttentionConfig(
        attention_type="gqa",
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        dropout=dropout,
        qk_norm=qk_norm,
        qkv_bias=qkv_bias,
    )


def make_mla_attention(num_heads: int, head_dim: int, dropout: float = 0.0, qk_norm: Optional[bool] = None, qkv_bias: Optional[bool] = None) -> AttentionConfig:
    return AttentionConfig(
        attention_type="mla",
        num_attention_heads=num_heads,
        num_key_value_heads=None,
        head_dim=head_dim,
        dropout=dropout,
        qk_norm=qk_norm,
        qkv_bias=qkv_bias,
    )


def make_dense_ffn(intermediate_size: int, activation: ActivationType = "silu") -> DenseFFNConfig:
    return DenseFFNConfig(ffn_type="dense", intermediate_size=intermediate_size, activation=activation)


def make_moe_ffn(num_experts: int, num_experts_per_tok: int, expert_intermediate_size: int, router_aux_loss_coef: float = 0.0, topk_group: Optional[int] = None, topk_method: Optional[str] = None, norm_topk_prob: Optional[bool] = None) -> MoEConfig:
    return MoEConfig(
        ffn_type="moe",
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        expert_intermediate_size=expert_intermediate_size,
        router_aux_loss_coef=router_aux_loss_coef,
        topk_group=topk_group,
        topk_method=topk_method,
        norm_topk_prob=norm_topk_prob,
    )


