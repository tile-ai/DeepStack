from mosaic.llm_arch import DeepSeekV3, LLM_Arch, Qwen3_235b_a22b, Qwen3_480b_a35b, Llama3_70b, Llama3_405b
from mosaic.arch import stacked_gpu_base, stacked_gpu_large_matrix, stacked_gpu_large_vector, stacked_gpu_high_l1, stacked_gpu_high_l2, stacked_gpu_high_noc, stacked_gpu_low_noc, stacked_gpu_reduced_sm, stacked_gpu_wgmma
from mosaic.noc.noc_config_set import torus_mesh_switch_1, torus_mesh_switch_2, torus_mesh_mesh_3, strong_torus_mesh_switch_4, weak_torus_mesh_switch_5, torus_mesh_switch_7, torus_mesh_switch_8, torus_mesh_switch_9

from mosaic.llm_arch.llm_base import LLM_Arch, MoE_Arch, Dense_FFN_Arch, MLA_Arch, GQA_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch

class LLM_GQA_RATE_1(LLM_Arch):
    def __init__(self):


        dense_ffn_arch = Dense_FFN_Arch(
            up_hidden=28672,
            swiglu_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
        )

        gqa_arch = GQA_Arch(
            num_head=64,
            num_kv_head=1,
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
            hidden_size=8192,
            num_layer=80,
            num_dense_layer=80,
            num_moe_layer=0,
            max_seq_len=131072,
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=dense_ffn_arch,
            moe_arch=None,
        )


class LLM_GQA_RATE_2(LLM_Arch):
    def __init__(self):


        dense_ffn_arch = Dense_FFN_Arch(
            up_hidden=28672,
            swiglu_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
        )

        gqa_arch = GQA_Arch(
            num_head=64,
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
            hidden_size=8192,
            num_layer=80,
            num_dense_layer=80,
            num_moe_layer=0,
            max_seq_len=131072,
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=dense_ffn_arch,
            moe_arch=None,
        )

class LLM_GQA_RATE_3(LLM_Arch):
    def __init__(self):


        dense_ffn_arch = Dense_FFN_Arch(
            up_hidden=28672,
            swiglu_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
        )

        gqa_arch = GQA_Arch(
            num_head=64,
            num_kv_head=16,
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
            hidden_size=8192,
            num_layer=80,
            num_dense_layer=80,
            num_moe_layer=0,
            max_seq_len=131072,
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=dense_ffn_arch,
            moe_arch=None,
        )

class LLM_GQA_RATE_4(LLM_Arch):
    def __init__(self):


        dense_ffn_arch = Dense_FFN_Arch(
            up_hidden=28672,
            swiglu_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
        )

        gqa_arch = GQA_Arch(
            num_head=64,
            num_kv_head=64,
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
            hidden_size=8192,
            num_layer=80,
            num_dense_layer=80,
            num_moe_layer=0,
            max_seq_len=131072,
            rms_norm_bytes=rms_norm_bytes,
            rope_bytes=rope_bytes,
            add_residual_bytes=add_residual_bytes,
            gqa_arch=gqa_arch,
            mla_arch=None,
            dense_ffn_arch=dense_ffn_arch,
            moe_arch=None,
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
    intermediate_weight = 3 * model_arch.hidden_size * model_arch.dense_ffn_arch.up_hidden * weight_bytes

    return wq_weight*model_arch.num_layer + wk_weight*model_arch.num_layer + wv_weight*model_arch.num_layer + wo_weight*model_arch.num_layer + intermediate_weight*model_arch.num_layer

if __name__ == "__main__":
    model_arch = LLM_GQA_RATE_4()
    print(get_model_weight_size(model_arch))


    # 134553272320 kv_head=1
    # 136902082560 kv_head=8
    # 139586437120 kv_head=16
    # 155692564480 kv_head=64