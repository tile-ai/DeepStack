"""Caller-owned NoC energy coefficients and binary-backed release defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


__all__ = ["NocEnergyConfig"]


def _nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class NocEnergyConfig:
    """Energy per transferred bit for the L1, L2, and L3 NoC levels.

    All three values are explicit caller inputs in pJ/bit.  Attach an instance
    to a :class:`~mosaic.noc.noc_topo.Hierarchy` or pass it to the route-stat
    helpers to evaluate a caller-owned energy model.
    """

    l1_pj_per_bit: float
    l2_pj_per_bit: float
    l3_pj_per_bit: float

    def __post_init__(self) -> None:
        for field in (
            "l1_pj_per_bit",
            "l2_pj_per_bit",
            "l3_pj_per_bit",
        ):
            object.__setattr__(
                self,
                field,
                _nonnegative(getattr(self, field), field),
            )

    def for_layer(self, li: int, layer_count: int) -> float:
        """Return the pJ/bit value for outer-to-inner layer index ``li``."""

        if (
            isinstance(layer_count, bool)
            or not isinstance(layer_count, int)
            or layer_count <= 0
        ):
            raise ValueError("layer_count must be a positive integer")
        if (
            isinstance(li, bool)
            or not isinstance(li, int)
            or not 0 <= li < layer_count
        ):
            raise ValueError("li must index a layer in [0, layer_count)")
        constants = (
            self.l1_pj_per_bit,
            self.l2_pj_per_bit,
            self.l3_pj_per_bit,
        )
        inner_index = layer_count - 1 - li
        return (
            constants[inner_index]
            if inner_index < len(constants)
            else constants[-1]
        )
