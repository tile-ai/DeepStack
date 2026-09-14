from .deepseek_v3 import DeepSeekV3
from .llm_base import DSA_Arch
from mosaic.utils import OpBytes, Tensor_Loc
import torch


class DeepSeekV3_2(DeepSeekV3):
    def __init__(self):
        super().__init__()
        self.name = "DeepSeek-V3.2"

        if self.moe_arch is not None:
            self.moe_arch.scoring_func = "sigmoid"
            self.moe_arch.topk_method = "noaux_tc"
            self.moe_arch.n_group = 8
            self.moe_arch.topk_group = 4
            self.moe_arch.norm_topk_prob = True
            self.moe_arch.routed_scaling_factor = 2.5

        self.dsa_arch = DSA_Arch(
            num_head=64,
            head_dim=128,
            topk=2048,
            atten_bytes=OpBytes(
                input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
                output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
            ),
        )


if __name__ == "__main__":
    deepseek_v3_2 = DeepSeekV3_2()
    print(deepseek_v3_2)
