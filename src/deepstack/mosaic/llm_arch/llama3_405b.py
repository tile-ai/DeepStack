from .llm_base import LLM_Arch, MoE_Arch, Dense_FFN_Arch, MLA_Arch, GQA_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch


class Llama3_405b(LLM_Arch):
    def __init__(self):


        dense_ffn_arch = Dense_FFN_Arch(
            up_hidden=53248,
            swiglu_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
        )

        gqa_arch = GQA_Arch(
            num_head=128,
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
            hidden_size=16384,
            num_layer=126,
            num_dense_layer=126,
            num_moe_layer=0,
            max_seq_len=131072,
            name="Llama3-405B",
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=dense_ffn_arch,
            moe_arch=None,
        )
if __name__ == "__main__":
    llama3_405b = Llama3_405b()
    print(llama3_405b)