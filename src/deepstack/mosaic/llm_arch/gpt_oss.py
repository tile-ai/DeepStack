from .llm_base import Attention_Pattern_Arch, GQA_Arch, LLM_Arch, MoE_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch


class GptOss(LLM_Arch):
    def __init__(self):
        bf16_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
        )

        gqa_arch = GQA_Arch(
            num_head=64,
            num_kv_head=8,
            head_dim=64,
            atten_bytes=bf16_bytes,
        )

        moe_arch = MoE_Arch(
            num_shared_experts=0,
            num_routed_experts=128,
            num_activated_experts=4,
            moe_down_hidden=2880,
            expert_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.float4_e2m1fn_x2, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
            gate_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="smem"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.float32, loc="ddr"),
            ),
            router_aux_loss_coef=0.9,
        )

        attention_pattern_arch = Attention_Pattern_Arch(
            layer_types=("sliding_attention", "full_attention") * 18,
            sliding_window=128,
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
            hidden_size=2880,
            num_layer=36,
            num_dense_layer=0,
            num_moe_layer=36,
            max_seq_len=131072,
            name="GPT-OSS",
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=bf16_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dsa_arch=None,
            attention_pattern_arch=attention_pattern_arch,
            dense_ffn_arch=None,
            moe_arch=moe_arch,
        )


if __name__ == "__main__":
    gpt_oss = GptOss()
    print(gpt_oss)
