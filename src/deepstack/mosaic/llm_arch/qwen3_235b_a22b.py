from .llm_base import LLM_Arch, MoE_Arch, Dense_FFN_Arch, MLA_Arch, GQA_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch


class Qwen3_235b_a22b(LLM_Arch):
    def __init__(self):


        gqa_arch = GQA_Arch(
            num_head=64,
            num_kv_head=4,
            head_dim=128,
            atten_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
        )

        moe_arch = MoE_Arch(
            num_shared_experts=0,
            num_routed_experts=128,
            num_activated_experts=8,
            moe_down_hidden=1536,
            expert_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
            gate_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="smem"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
        )

        rope_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
        )
        add_residual_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            input2=Tensor_Loc(dtype=torch.bfloat16, loc="smem"),
            output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
        )
        rms_norm_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.float32, loc="smem"),
            input2=Tensor_Loc(dtype=torch.float32, loc="ddr"),
            output=Tensor_Loc(dtype=torch.float16, loc="ddr"),
        )
        
        super().__init__(
            hidden_size=4096,
            num_layer=94,
            num_dense_layer=0,
            num_moe_layer=94,
            max_seq_len=262144,
            name="Qwen3-235B-A22B",
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=None,
            moe_arch=moe_arch,
        )
if __name__ == "__main__":
    llama3_70b = Qwen3_235b_a22b()
    print(llama3_70b)