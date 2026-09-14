"""Binary-backed scaled H100 baseline used by the archived AE search."""

from . import _reference_model as _provider


class H100_SCALED(_provider.ReferenceArch):
    __slots__ = ()

    def __init__(self):
        super().__init__(
            "h100_scaled",
            public_name=type(self).__name__,
            scaled=True,
        )

    def set_to_spec(self) -> "H100_SCALED":
        return self

    def set_to_microbench(self) -> "H100_SCALED":
        return self

    def set_to_ncu(self) -> "H100_SCALED":
        return self
