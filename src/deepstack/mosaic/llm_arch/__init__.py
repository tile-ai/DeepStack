from .llm_base import (
    Attention_Pattern_Arch,
    DSA_Arch,
    Dense_FFN_Arch,
    GQA_Arch,
    LLM_Arch,
    MLA_Arch,
    MoE_Arch,
)
from .omni_base import (
    Audio_Encoder_Arch,
    Code2Wav_Arch,
    Omni_Arch,
    ViT_Encoder_Arch,
)
import importlib

__all__ = [
    "LLM_Arch",
    "MoE_Arch",
    "Dense_FFN_Arch",
    "GQA_Arch",
    "MLA_Arch",
    "DSA_Arch",
    "Attention_Pattern_Arch",
    "Omni_Arch",
    "ViT_Encoder_Arch",
    "Audio_Encoder_Arch",
    "Code2Wav_Arch",
    "Llama3_8b",
    "Llama3_70b",
    "Llama3_405b",
    "DeepSeekV3",
    "DeepSeekV3_A8W8",
    "DeepSeekV3_2",
    "DeepSeekR1",
    "GptOss",
    "Glm5_1",
    "Qwen2_5_7b",
    "Qwen2_5_14b",
    "Qwen2_5_72b",
    "Qwen3_32b",
    "Qwen3_30b_a3b",
    "Qwen3_480b_a35b",
    "Qwen3_235b_a22b",
    "Qwen3_Omni_30b_a3b",
]


def __getattr__(name):
    module_map = {
        "Llama3_8b": ".llama3_8b",
        "Llama3_70b": ".llama3_70b",
        "Llama3_405b": ".llama3_405b",
        "DeepSeekV3": ".deepseek_v3",
        "DeepSeekV3_A8W8": ".deepseek_v3_a8w8",
        "DeepSeekV3_2": ".deepseek_v3_2",
        "DeepSeekR1": ".deepseek_r1",
        "GptOss": ".gpt_oss",
        "Glm5_1": ".glm5_1",
        "Qwen2_5_7b": ".qwen2_5_7b",
        "Qwen2_5_14b": ".qwen2_5_14b",
        "Qwen2_5_72b": ".qwen2_5_72b",
        "Qwen3_32b": ".qwen3_32b",
        "Qwen3_30b_a3b": ".qwen3_30b_a3b",
        "Qwen3_480b_a35b": ".qwen3_480b_a35b",
        "Qwen3_235b_a22b": ".qwen3_235b_a22b",
        "Qwen3_Omni_30b_a3b": ".qwen3_omni_30b_a3b",
    }
    if name in module_map:
        module = importlib.import_module(module_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
