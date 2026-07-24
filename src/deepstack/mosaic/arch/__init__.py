from importlib import import_module

__all__ = [
    "stacked_gpu_base",
    "stacked_gpu_large_matrix",
    "stacked_gpu_large_vector",
    "stacked_gpu_high_l1",
    "stacked_gpu_high_l2",
    "stacked_gpu_high_noc",
    "stacked_gpu_low_noc",
    "stacked_gpu_reduced_sm",
    "stacked_gpu_wgmma",
    "stacked_gpu_cluster_8",
    "stacked_gpu_cluster_16",
    "stacked_gpu_cluster_32",
    "ConfigurableStackedGpu",
    "DramInterfaceConfig",
    "DramTimingConfig",
    "StackedGpuConfig",
    "dram_connectivity_efficiency",
    "make_custom_stacked_gpu",
    "H100_SCALED",
    "H200_SCALED",
    "H100",
    "H200",
    "B200",
    "MI325X",
    "MI325XSpec",
    "B6000",
    "A100",
]

_LAZY_IMPORTS = {
    "stacked_gpu_base": ".stacked_gpu_base",
    "stacked_gpu_large_matrix": ".stacked_gpu_large_matrix",
    "stacked_gpu_large_vector": ".stacked_gpu_large_vector",
    "stacked_gpu_high_l1": ".stacked_gpu_high_l1",
    "stacked_gpu_high_l2": ".stacked_gpu_high_l2",
    "stacked_gpu_high_noc": ".stacked_gpu_high_noc",
    "stacked_gpu_low_noc": ".stacked_gpu_low_noc",
    "stacked_gpu_reduced_sm": ".stacked_gpu_reduced_sm",
    "stacked_gpu_wgmma": ".stacked_gpu_wgmma",
    "stacked_gpu_cluster_8": ".stacked_gpu_cluster",
    "stacked_gpu_cluster_16": ".stacked_gpu_cluster",
    "stacked_gpu_cluster_32": ".stacked_gpu_cluster",
    "ConfigurableStackedGpu": ".custom_profile",
    "DramInterfaceConfig": ".custom_profile",
    "DramTimingConfig": ".custom_profile",
    "StackedGpuConfig": ".custom_profile",
    "dram_connectivity_efficiency": ".custom_profile",
    "make_custom_stacked_gpu": ".custom_profile",
    "H100_SCALED": ".h100_scaled",
    "H200_SCALED": ".h200_scaled",
    "H100": ".h100",
    "H200": ".h200",
    "B200": ".b200",
    "MI325X": ".mi325x",
    "MI325XSpec": ".mi325x",
    "B6000": ".b6000",
    "A100": ".a100",
}

def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_IMPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
