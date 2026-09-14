import importlib

# __all__ = [
#     "moe_top",
#     "get_moe_footprint",
#     "moe_coarse",
#     "moe_coarse_by_token",
# ]

# _LAZY_MODULES = {
#     "moe_top": "mosaic.llm.moe.moe_top",
#     "get_moe_footprint": "mosaic.llm.moe.moe_top",
#     "moe_coarse": "mosaic.llm.moe.moe_coarse",
# }

# def __getattr__(name):
#     module_path = _LAZY_MODULES.get(name)
#     if module_path is None:
#         raise AttributeError(f"module {__name__} has no attribute {name}")
#     module = importlib.import_module(module_path)
#     return getattr(module, name)