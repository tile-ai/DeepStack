"""Public DSE helpers with lazy compatibility exports.

Historically this package eagerly imported the architecture/NoC combination,
parallel-scheme, and task-combination modules.  Besides making every DSE
import expensive, that pulled the binary-backed reference model into
source-only custom-policy workflows.  The public names remain available, but
their defining module is now imported only when a caller requests that name.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any


_ARCH_NOC_EXPORTS = (
    "B200",
    "H100",
    "H100_SCALED",
    "H200",
    "H200_SCALED",
    "arch_noc_combinations1",
    "arch_noc_combinations1110",
    "arch_noc_combinations_scale_64_512_nodes",
    "b200_8x1_8_arch_noc_combinations",
    "b200_8x1_8_single_arch_noc_combinations",
    "b200_e2e_compare_arch_noc_combinations",
    "compare_stacked_gpu_vs_h200",
    "h100_8_to_64_arch_noc_combinations",
    "h100x32_arch_noc_combinations",
    "h100x32_medium",
    "h100x32_medium_arch_noc_combinations",
    "h100x32_strong",
    "h100x32_strong_arch_noc_combinations",
    "h100x8_arch_noc_combinations",
    "h200_no_nvswitch_arch_noc_combinations",
    "h200x16_arch_noc_combinations",
    "h200x32_arch_noc_combinations",
    "h200x8_arch_noc_combinations",
    "stacked_gpu_base",
    "stacked_gpu_dse_vs_h200_scaled_0309",
    "stacked_gpu_high_l1",
    "stacked_gpu_high_l2",
    "stacked_gpu_high_noc",
    "stacked_gpu_large_matrix",
    "stacked_gpu_large_vector",
    "stacked_gpu_low_noc",
    "stacked_gpu_reduced_sm",
    "stacked_gpu_wgmma",
    "strong_torus_mesh_switch_4",
    "torus_mesh_mesh_3",
    "torus_mesh_switch_1",
    "torus_mesh_switch_2",
    "torus_mesh_switch_7",
    "torus_mesh_switch_8",
    "torus_mesh_switch_9",
    "multi_gpu_validation_arch_noc_combinations",
    "weak_torus_mesh_switch_5",
)

_TASK_EXPORTS = (
    "DeepSeekV3",
    "LLM_Arch",
    "Llama3_405b",
    "Llama3_70b",
    "Qwen3_235b_a22b",
    "Qwen3_480b_a35b",
    "customized_decoding_task_combination",
    "customized_prefill_task_combination",
    "dpsk_decode_max_stps_task_combination",
    "dpsk_decode_task_combination",
    "hybrid_decode_task_combination_0301",
    "hybrid_prefill_task_combination_0301",
    "llama3_405b_decode_task_combination",
    "llama3_70b_decode_task_combination",
    "qwen3_235b_a22b_decode_task_combination",
    "scale_64_512_nodes_task_combination_decode",
    "scale_64_512_nodes_task_combination_prefill",
    "task_combination_2",
    "task_combination_3",
    "task_combination_4",
)

_SUBMODULE_EXPORTS = {
    "arch_noc_combinations": ".arch_noc_combinations",
    "parallel_schemes": ".parallel_schemes",
    "task_combination": ".task_combination",
}

_EXPORT_MODULES = {
    **{name: ".arch_noc_combinations" for name in _ARCH_NOC_EXPORTS},
    "all_parallel_schemes": ".parallel_schemes",
    **{name: ".task_combination" for name in _TASK_EXPORTS},
    **_SUBMODULE_EXPORTS,
}

# Match the names exposed by the former eager imports, including the three
# submodule attributes that import machinery installed on the package.
__all__ = tuple(_EXPORT_MODULES)


def __getattr__(name: str) -> _Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = _import_module(module_name, __name__)
    value = module if name in _SUBMODULE_EXPORTS else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
