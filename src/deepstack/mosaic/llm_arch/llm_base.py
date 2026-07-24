# from mosaic.utils import OpBytes

# class MoE_Arch:
#     def __init__(self):
#         # (bs: int, seq: int, hidden: int, moe_down_hidden: int, parallel: ParallelScheme, expert_bytes: OpBytes, gate_bytes: OpBytes, num_shared_experts: int, num_routed_experts: int)
#         self.num_moe_layer = None

#         self.num_shared_experts = None
#         self.num_routed_experts = None
#         self.num_activated_experts = None
#         self.moe_down_hidden = None

#         self.expert_bytes = None
#         self.gate_bytes = None

# class Dense_FFN_Arch:
#     def __init__(self):
#         # (bs:int, seq:int, hidden:int, up_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, swiglu_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy)
#         self.num_dense_layer = None
        
#         self.up_hidden = None

#         self.swiglu_bytes = None

# class GQA_Arch:
#     def __init__(self):
#         # (bs:int, seq:int, hidden:int, num_head:int, num_kv_head:int, head_dim:int, parallel:ParallelScheme, atten_parallel:ParallelScheme, next_parallel:ParallelScheme, atten_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
#         self.num_head = None
#         self.num_kv_head = None
#         self.head_dim = None

#         self.atten_bytes = None

# class MLA_Arch:
#     def __init__(self):
#         self.num_head = None
#         self.num_kv_head = None
#         self.head_dim = None

#         self.q_down_hidden = None
#         self.q_rope_head_dim = None
#         self.q_nope_head_dim = None


#         self.kv_rope_head_dim = None
#         self.kv_nope_head_dim = None 
#         # for decode only, with absorb   
#         # self.kv_nope_up_head_dim = self.num_head * (128+128)
#         self.kv_nope_up_hidden = None

#         # for prefill only, no absorb
#         self.kv_up_head_dim = None


# class LLM_Arch:
#     def __init__(self):
        
#         self.num_layer = None
#         self.num_dense_layer = None
#         self.num_moe_layer = None

#         if self.num_layer is not None:
#             assert self.num_layer == self.num_dense_layer + self.num_moe_layer

#         self.hidden_size = None
        
#         # 实例化各个架构组件
#         self.moe_arch = MoE_Arch()
#         self.dense_ffn_arch = Dense_FFN_Arch()
#         self.gqa_arch = GQA_Arch()
#         self.mla_arch = MLA_Arch()

from dataclasses import dataclass, field
from typing import Optional
import torch
from mosaic.utils import OpBytes, Tensor_Loc

# 定义一个默认的OpBytes工厂函数，以避免可变默认参数问题
def default_op_bytes():
    return OpBytes(
        input1=Tensor_Loc(dtype=torch.float16, loc="ddr"),
        input2=Tensor_Loc(dtype=torch.float16, loc="ddr"),
        output=Tensor_Loc(dtype=torch.float16, loc="ddr"),
    )

@dataclass
class MoE_Arch:
    num_shared_experts: int
    num_routed_experts: int
    num_activated_experts: int
    moe_down_hidden: int
    expert_bytes: OpBytes = field(default_factory=default_op_bytes)
    gate_bytes: OpBytes = field(default_factory=default_op_bytes)
    scoring_func: Optional[str] = None
    topk_method: Optional[str] = None
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    norm_topk_prob: Optional[bool] = None
    routed_scaling_factor: Optional[float] = None
    router_aux_loss_coef: Optional[float] = None

@dataclass
class Dense_FFN_Arch:
    up_hidden: int
    swiglu_bytes: OpBytes = field(default_factory=default_op_bytes)

@dataclass
class GQA_Arch:
    num_head: int
    num_kv_head: int
    head_dim: int
    atten_bytes: OpBytes = field(default_factory=default_op_bytes)

@dataclass
class MLA_Arch:
    num_head: int
    num_kv_head: int
    head_dim: int
    q_down_hidden: int
    q_rope_head_dim: int
    q_nope_head_dim: int
    kv_rope_head_dim: int
    kv_nope_head_dim: int
    kv_nope_up_hidden: int
    atten_bytes: OpBytes = field(default_factory=default_op_bytes)

@dataclass
class DSA_Arch:
    num_head: int
    head_dim: int
    topk: int
    rope_interleave: Optional[bool] = None
    atten_bytes: OpBytes = field(default_factory=default_op_bytes)

    @property
    def index_n_heads(self) -> int:
        return self.num_head

    @property
    def index_head_dim(self) -> int:
        return self.head_dim

    @property
    def index_topk(self) -> int:
        return self.topk

@dataclass
class Attention_Pattern_Arch:
    layer_types: tuple[str, ...] = ()
    sliding_window: Optional[int] = None

    @property
    def num_sliding_layer(self) -> int:
        return self.layer_types.count("sliding_attention")

    @property
    def num_full_attention_layer(self) -> int:
        return self.layer_types.count("full_attention")

@dataclass
class LLM_Arch:
    """
    一个通用的LLM架构基类,负责持有和校验架构组件。
    """
    # 核心参数
    hidden_size: int
    num_layer: int
    num_dense_layer: int
    num_moe_layer: int
    max_seq_len: int
    name: str = ""
    rms_norm_bytes: OpBytes  = field(default_factory=default_op_bytes)
    rope_bytes: OpBytes = field(default_factory=default_op_bytes)
    add_residual_bytes: OpBytes = field(default_factory=default_op_bytes)


    # Attention组件保持不变
    # attention_arch: Union[GQA_Arch, MLA_Arch]

    gqa_arch: Optional[GQA_Arch] = None
    mla_arch: Optional[MLA_Arch] = None
    dense_ffn_arch: Optional[Dense_FFN_Arch] = None
    moe_arch: Optional[MoE_Arch] = None
    dsa_arch: Optional[DSA_Arch] = None
    attention_pattern_arch: Optional[Attention_Pattern_Arch] = None

    def __post_init__(self):
        # 初始化后的校验逻辑保持不变
        # 校验1: 总层数必须匹配
        if self.num_layer != self.num_dense_layer + self.num_moe_layer:
            raise ValueError("总层数必须是 dense 层和 MoE 层的总和。")

        # 校验 2: 如果声明了有dense层，那么dense_ffn_arch必须存在
        if self.num_dense_layer > 0 and self.dense_ffn_arch is None:
            raise ValueError("模型声明了 dense 层, 但 dense_ffn_arch 未被提供。")
        
        # 校验 3: 如果声明了有moe层，那么moe_arch必须存在
        if self.num_moe_layer > 0 and self.moe_arch is None:
            raise ValueError("模型声明了 moe 层, 但 moe_arch 未被提供。")

        if (
            self.attention_pattern_arch is not None
            and self.attention_pattern_arch.layer_types
            and len(self.attention_pattern_arch.layer_types) != self.num_layer
        ):
            raise ValueError("attention layer_types 的长度必须等于 num_layer。")
