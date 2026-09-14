# from .add_residual_top import add_residual_top

import importlib

__all__ = [
    "add_residual_top",
    "get_add_residual_footprint",
    "add_residual_coarse",
]

_LAZY_MODULES = {
    "add_residual_top": "mosaic.llm.add_residual.add_residual_top",
    "get_add_residual_footprint": "mosaic.llm.add_residual.add_residual_top",
    "add_residual_coarse": "mosaic.llm.add_residual.add_residual_coarse",
}

def __getattr__(name):
    module_path = _LAZY_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(module_path)
    return getattr(module, name)