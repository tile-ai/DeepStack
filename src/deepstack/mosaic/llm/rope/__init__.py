import importlib

__all__ = [
    "ropeq_top",
    "ropeq_coarse",
    "ropek_top",
    "ropek_coarse",
    "get_rope_global_footprint",
]

_LAZY_MODULES = {
    "ropeq_top": "mosaic.llm.rope.ropeq_top",
    "ropeq_coarse": "mosaic.llm.rope.ropeq_coarse",
    "ropek_top": "mosaic.llm.rope.ropek_top",
    "ropek_coarse": "mosaic.llm.rope.ropek_coarse",
    "get_rope_global_footprint": "mosaic.llm.rope.get_rope_global_footprint",
}

def __getattr__(name):
    module_path = _LAZY_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(module_path)
    return getattr(module, name)