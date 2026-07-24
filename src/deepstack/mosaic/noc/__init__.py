
from importlib import import_module as _import_module  # noqa: F401

__all__ = []  # 通过 __dir__ 动态汇总
_SUBMODULES = ("traffic_matrix", "noc_topo", "energy_config", "noc_energy", "route_stats")


def __getattr__(name):  # PEP 562
    for modname in _SUBMODULES:
        mod = _import_module(f"{__name__}.{modname}")
        if hasattr(mod, name):
            attr = getattr(mod, name)
            globals()[name] = attr  # 缓存查找到的符号
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
