import importlib

__all__ = [
    "swiglu_top",
    "get_swiglu_footprint",
    "swiglu_coarse",
    "swiglu_coarse_by_token",
]

_LAZY_MODULES = {
    "swiglu_top": "mosaic.llm.swiglu.swiglu_top",
    "get_swiglu_footprint": "mosaic.llm.swiglu.swiglu_top",
    "swiglu_coarse": "mosaic.llm.swiglu.swiglu_coarse",
    "swiglu_coarse_by_token": "mosaic.llm.swiglu.swiglu_coarse_by_token",
}

def __getattr__(name):
    module_path = _LAZY_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(module_path)
    return getattr(module, name)