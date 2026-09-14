from .llm_base import DSA_Arch, Dense_FFN_Arch, LLM_Arch, MLA_Arch, MoE_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch


class Glm5_1(LLM_Arch):
    def __init__(self):
        bf16_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
        )

        moe_arch = MoE_Arch(
            num_shared_experts=1,
            num_routed_experts=256,
            num_activated_experts=8,
            moe_down_hidden=2048,
            expert_bytes=bf16_bytes,
            gate_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="smem"),
                input2=Tensor_Loc(dtype=torch.float32, loc="ddr"),
                output=Tensor_Loc(dtype=torch.float32, loc="ddr"),
            ),
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
        )

        dense_ffn_arch = Dense_FFN_Arch(
            up_hidden=12288,
            swiglu_bytes=bf16_bytes,
        )

        mla_arch = MLA_Arch(
            num_head=64,
            num_kv_head=64,
            head_dim=256,
            q_down_hidden=2048,
            q_rope_head_dim=64,
            q_nope_head_dim=192,
            kv_rope_head_dim=64,
            kv_nope_head_dim=512,
            kv_nope_up_hidden=64 * (192 + 256),
            atten_bytes=bf16_bytes,
        )

        dsa_arch = DSA_Arch(
            num_head=32,
            head_dim=128,
            topk=2048,
            rope_interleave=True,
            atten_bytes=bf16_bytes,
        )

        add_residual_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            input2=Tensor_Loc(dtype=torch.bfloat16, loc="smem"),
            output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
        )
        rms_norm_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.float32, loc="smem"),
            input2=Tensor_Loc(dtype=torch.float32, loc="ddr"),
            output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
        )

        super().__init__(
            hidden_size=6144,
            num_layer=78,
            num_dense_layer=3,
            num_moe_layer=75,
            max_seq_len=202752,
            name="GLM-5.1",
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=bf16_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=None,
            mla_arch=mla_arch,
            dsa_arch=dsa_arch,
            attention_pattern_arch=None,
            dense_ffn_arch=dense_ffn_arch,
            moe_arch=moe_arch,
        )


if __name__ == "__main__":
    glm5_1 = Glm5_1()
    print(glm5_1)
