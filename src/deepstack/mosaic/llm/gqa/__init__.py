import importlib

__all__ = [
    "gqa_prefill_top",
    "get_gqa_prefill_footprint",
    "gqa_decode_top",
    "get_gqa_decode_footprint",
    "gqa_decode_kv_list_top",
]

_LAZY_MODULES = {
    "gqa_prefill_top": "mosaic.llm.gqa.gqa_prefill_top",
    "get_gqa_prefill_footprint": "mosaic.llm.gqa.gqa_prefill_top",
    "gqa_decode_top": "mosaic.llm.gqa.gqa_decode_top",
    "get_gqa_decode_footprint": "mosaic.llm.gqa.gqa_decode_top",
    "gqa_decode_kv_list_top": "mosaic.llm.gqa.gqa_decode_top",
}

def __getattr__(name):
    module_path = _LAZY_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(module_path)
    return getattr(module, name)