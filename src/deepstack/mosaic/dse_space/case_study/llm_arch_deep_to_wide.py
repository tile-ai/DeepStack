from mosaic.llm_arch import DeepSeekV3, LLM_Arch, Qwen3_235b_a22b, Qwen3_480b_a35b, Llama3_70b, Llama3_405b
from mosaic.arch import stacked_gpu_base, stacked_gpu_large_matrix, stacked_gpu_large_vector, stacked_gpu_high_l1, stacked_gpu_high_l2, stacked_gpu_high_noc, stacked_gpu_low_noc, stacked_gpu_reduced_sm, stacked_gpu_wgmma
from mosaic.noc.noc_config_set import torus_mesh_switch_1, torus_mesh_switch_2, torus_mesh_mesh_3, strong_torus_mesh_switch_4, weak_torus_mesh_switch_5, torus_mesh_switch_7, torus_mesh_switch_8, torus_mesh_switch_9

from mosaic.llm_arch.llm_base import LLM_Arch, MoE_Arch, Dense_FFN_Arch, MLA_Arch, GQA_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch


class LLM_DEEP_TO_WIDE_1(LLM_Arch):
    def __init__(self):

        gqa_arch = GQA_Arch(
            num_head=64//2,
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
            hidden_size=4096//2,
            num_layer=94*2,
            num_dense_layer=0,
            num_moe_layer=94*2,
            max_seq_len=262144,
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=None,
            moe_arch=moe_arch,
        )

class LLM_DEEP_TO_WIDE_2(LLM_Arch):
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
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=None,
            moe_arch=moe_arch,
        )

class LLM_DEEP_TO_WIDE_3(LLM_Arch):
    def __init__(self):

        gqa_arch = GQA_Arch(
            num_head=64*2,
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
            hidden_size=4096*2,
            num_layer=94//2,
            num_dense_layer=0,
            num_moe_layer=94//2,
            max_seq_len=262144,
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=None,
            moe_arch=moe_arch,
        )
    
class LLM_DEEP_TO_WIDE_4(LLM_Arch):
    def __init__(self):

        gqa_arch = GQA_Arch(
            num_head=64*4,
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
            hidden_size=4096*4,
            num_layer=94//4,
            num_dense_layer=0,
            num_moe_layer=94//4,
            max_seq_len=262144,
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=None,
            moe_arch=moe_arch,
        )

if __name__ == "__main__":
    # llama3_70b = Llama3_70b()
    # print(llama3_70b)
    model_arch = LLM_DEEP_TO_WIDE_1()
    print(model_arch)
    print(model_arch.max_seq_len)