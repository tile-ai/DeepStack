from .llm_base import LLM_Arch, MoE_Arch, Dense_FFN_Arch, MLA_Arch, GQA_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch


class DeepSeekV3_A8W8(LLM_Arch):
    def __init__(self):
        moe_arch = MoE_Arch(
            num_shared_experts=1,
            num_routed_experts=256,
            num_activated_experts=8,
            moe_down_hidden=2048,
            expert_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
                output=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
            ),
            # router_logits = F.linear(hidden_states.type(torch.float8_e4m3fn), self.weight.type(torch.float8_e4m3fn))
            gate_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="smem"),
                input2=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
                output=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
            ),
            
        )

        dense_ffn_arch = Dense_FFN_Arch(
            up_hidden=18432,
            swiglu_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
                output=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
            ),
            
        )

        rope_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
            input2=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
            output=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
        )
        add_residual_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
            input2=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="smem"),
            output=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
        )
        rms_norm_bytes = OpBytes(
            input1=Tensor_Loc(dtype=torch.float32, loc="smem"),
            input2=Tensor_Loc(dtype=torch.float32, loc="ddr"),
            output=Tensor_Loc(dtype=torch.float16, loc="ddr"),
        )

        mla_arch = MLA_Arch(
            num_head=128,
            num_kv_head=128,
            head_dim=128,
            q_down_hidden=1536,
            q_rope_head_dim=64,
            q_nope_head_dim=128,
            kv_rope_head_dim=64,
            kv_nope_head_dim=512,
            kv_nope_up_hidden=128*(128+128),
            atten_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
                output=Tensor_Loc(dtype=torch.float8_e4m3fn, loc="ddr"),
            ),
        )

        super().__init__(
            hidden_size=7168,
            num_layer=61,
            num_dense_layer=3,
            num_moe_layer=58,
            max_seq_len=163840,
            name="DeepSeek-V3",
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=None,
            mla_arch=mla_arch,
            dense_ffn_arch=dense_ffn_arch,
            moe_arch=moe_arch,
        )
if __name__ == "__main__":
    deepseek_v3_a8w8 = DeepSeekV3_A8W8()
    print(deepseek_v3_a8w8)