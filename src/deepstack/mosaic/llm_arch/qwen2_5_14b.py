from .llm_base import LLM_Arch, MoE_Arch, Dense_FFN_Arch, MLA_Arch, GQA_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch


class Qwen2_5_14b(LLM_Arch):
    def __init__(self):


        dense_ffn_arch = Dense_FFN_Arch(
            up_hidden=13824,
            swiglu_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
        )

        gqa_arch = GQA_Arch(
            num_head=40,
            num_kv_head=8,
            head_dim=128,
            atten_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
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
            hidden_size=5120,
            num_layer=48,
            num_dense_layer=48,
            num_moe_layer=0,
            max_seq_len=32768,
            name="Qwen2.5-14B",
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=dense_ffn_arch,
            moe_arch=None,
        )
if __name__ == "__main__":
    qwen2_5_14b = Qwen2_5_14b()
    print(qwen2_5_14b)
