"""Supplementary reference cluster-width sweep architectures."""

from __future__ import annotations

from . import _reference_model as _provider
from .stacked_gpu_wgmma import stacked_gpu_wgmma


class _StackedGpuClusterBase(stacked_gpu_wgmma):
    """Reference architecture widened to ``CLUSTER_SM`` processing elements."""

    __slots__ = ()

    CLUSTER_SM: int

    def __init__(self):
        super().__init__()
        _provider.q06(self, int(self.CLUSTER_SM))


def make_stacked_gpu_cluster(sm_count: int) -> _StackedGpuClusterBase:
    """Construct a reference cluster-width point for a positive PE count."""

    if isinstance(sm_count, bool) or not isinstance(sm_count, int):
        raise TypeError("sm_count must be an integer")
    if sm_count <= 0:
        raise ValueError("sm_count must be greater than zero")
    cls = type(
        f"stacked_gpu_cluster_{sm_count}",
        (_StackedGpuClusterBase,),
        {"CLUSTER_SM": sm_count, "__slots__": ()},
    )
    return cls()


class stacked_gpu_cluster_8(_StackedGpuClusterBase):
    __slots__ = ()

    CLUSTER_SM = 8


class stacked_gpu_cluster_16(_StackedGpuClusterBase):
    __slots__ = ()

    CLUSTER_SM = 16


class stacked_gpu_cluster_32(_StackedGpuClusterBase):
    __slots__ = ()

    CLUSTER_SM = 32
