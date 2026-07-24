"""Reference die-capacity facade.

The bundled native provider exposes one decision only: the maximum SM count
that fits the sealed reference die budget for an already configured reference
architecture and NoC.  It intentionally does not expose total area,
component-area estimates, calibration constants, or a caller-selectable area
budget.

Callers that define their own architecture should supply their own capacity
provider (or an explicit source-visible area estimator) through the DSE policy
interfaces instead of querying the protected reference provider.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any


MIN_SM_COUNT = 1
MAX_SM_COUNT = 128


def max_sm_count(reference_arch: Any, noc: Any) -> int:
    """Return the feasible SM capacity for one sealed reference design point."""

    from mosaic.arch import _reference_model

    if not _reference_model.is_reference_arch(reference_arch):
        raise TypeError(
            "the bundled capacity provider accepts only a reference "
            "architecture; custom architectures must supply a capacity_provider"
        )

    from . import _capacity

    try:
        result = _capacity.p00(reference_arch, noc)
    except ValueError:
        raise ValueError(
            "configuration is outside the bundled AE capacity LUT; "
            "contact the artifact authors for additional area-model coverage"
        ) from None
    if isinstance(result, bool) or not isinstance(result, Integral):
        raise RuntimeError("reference capacity provider returned a non-integer")
    value = int(result)
    if not MIN_SM_COUNT <= value <= MAX_SM_COUNT:
        raise RuntimeError(
            "reference capacity provider returned an out-of-range SM count"
        )
    return value


__all__ = ["MAX_SM_COUNT", "MIN_SM_COUNT", "max_sm_count"]
