"""Paper-facing stacked-GPU profile backed by the bundled reference provider."""

from __future__ import annotations

from . import _reference_model as _provider


class stacked_gpu_base(_provider.ReferenceArch):
    """Compatibility architecture for the paper's standard design point.

    The no-argument constructor is retained for all AE workflows.  Calibrated
    reference values are loaded from the bundled provider; callers who want to
    supply their own clocks, memory interface, or compute resources should use
    :mod:`mosaic.arch.custom_profile`.
    """

    __slots__ = ()

    def __init__(self, *, _profile_name: str = "standard"):
        super().__init__(_profile_name, public_name=type(self).__name__)

    def set_to_spec(self) -> "stacked_gpu_base":
        return self

    def set_to_microbench(self) -> "stacked_gpu_base":
        return self

    def set_to_ncu(self) -> "stacked_gpu_base":
        return self
