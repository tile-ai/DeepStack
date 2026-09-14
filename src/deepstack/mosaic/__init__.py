"""Lazily re-export submodule symbols without importing every subpackage at initialization.
This avoids importing noc_topo too early when running python -m mosaic.noc.noc_topo.
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
            globals()[name] = attr  # Cache.
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