# from .rms_norm_top import rms_norm_top

import importlib

__all__ = [
    "rms_norm_top",
    "get_rms_norm_footprint",
    "rms_norm_coarse",
]

_LAZY_MODULES = {
    "rms_norm_top": "mosaic.llm.rms_norm.rms_norm_top",
    "get_rms_norm_footprint": "mosaic.llm.rms_norm.rms_norm_top",
    "rms_norm_coarse": "mosaic.llm.rms_norm.rms_norm_coarse",
}

def __getattr__(name):
    module_path = _LAZY_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(module_path)
    return getattr(module, name)