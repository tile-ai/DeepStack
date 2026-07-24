from mosaic.llm_arch import DeepSeekV3, LLM_Arch, Qwen3_235b_a22b, Qwen3_480b_a35b, Llama3_70b, Llama3_405b
from mosaic.arch import stacked_gpu_base, stacked_gpu_large_matrix, stacked_gpu_large_vector, stacked_gpu_high_l1, stacked_gpu_high_l2, stacked_gpu_high_noc, stacked_gpu_low_noc, stacked_gpu_reduced_sm, stacked_gpu_wgmma
from mosaic.noc.noc_config_set import torus_mesh_switch_1, torus_mesh_switch_2, torus_mesh_mesh_3, strong_torus_mesh_switch_4, weak_torus_mesh_switch_5, torus_mesh_switch_7, torus_mesh_switch_8, torus_mesh_switch_9

from mosaic.llm_arch.llm_base import LLM_Arch, MoE_Arch, Dense_FFN_Arch, MLA_Arch, GQA_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch

class LLM_MOE_SPARSE_TO_DENSE_1(LLM_Arch):
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
            num_activated_experts=1,
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


class LLM_MOE_SPARSE_TO_DENSE_8(LLM_Arch):
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

class LLM_MOE_SPARSE_TO_DENSE_32(LLM_Arch):
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
            num_activated_experts=32,
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


class LLM_MOE_SPARSE_TO_DENSE_128(LLM_Arch):
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
            num_activated_experts=128,
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

def get_model_weight_size(model_arch: LLM_Arch):
    
    weight_bytes = 2
    
    # wq
    wq_weight = model_arch.gqa_arch.num_head * model_arch.gqa_arch.head_dim * model_arch.hidden_size * weight_bytes
    # wk 
    wk_weight = model_arch.gqa_arch.num_kv_head * model_arch.gqa_arch.head_dim * model_arch.hidden_size * weight_bytes
    # wv
    wv_weight = model_arch.gqa_arch.num_kv_head * model_arch.gqa_arch.head_dim * model_arch.hidden_size * weight_bytes
    # wo
    wo_weight = model_arch.gqa_arch.num_head * model_arch.gqa_arch.head_dim * model_arch.hidden_size * weight_bytes
    # intermediate weight
    # 3 * hidden * hidden / group_size * weight_bytes
    # intermediate_weight = 3 * model_arch.hidden_size * model_arch.dense_ffn_arch.up_hidden * weight_bytes
    moe_weight = (model_arch.moe_arch.num_shared_experts+model_arch.moe_arch.num_routed_experts) * model_arch.hidden_size * model_arch.moe_arch.moe_down_hidden * weight_bytes * 3

    return wq_weight*model_arch.num_layer + wk_weight*model_arch.num_layer + wv_weight*model_arch.num_layer + wo_weight*model_arch.num_layer + moe_weight*model_arch.num_layer

if __name__ == "__main__":
    model_arch = LLM_MOE_SPARSE_RATE_1()
    print(get_model_weight_size(model_arch))