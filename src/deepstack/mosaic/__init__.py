"""
惰性重导出子模块符号，避免在包初始化阶段导入大量子包，
从而避免运行 `python -m mosaic.noc.noc_topo` 时提前间接导入 `noc_topo` 触发警告。
"""

from importlib import import_module as _import_module  # noqa: F401

_SUBMODULES = (
    "data",
    "llm",
    "noc",
    "parallelism",
    "dse_space",
    "cost",
    # "perf",
    "utils",
    "arch",
    "op_dtype",
    "collectives",
)


def __getattr__(name):  # PEP 562
    for modname in _SUBMODULES:
        try:
            mod = _import_module(f"{__name__}.{modname}")
        except Exception:
            continue
        if hasattr(mod, name):
            attr = getattr(mod, name)
            globals()[name] = attr  # 缓存
            return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = set(globals().keys())
    for modname in _SUBMODULES:
        try:
            mod = _import_module(f"{__name__}.{modname}")
        except Exception:
            continue
        for n in dir(mod):
            if not n.startswith("_"):
                names.add(n)
    return sorted(names)