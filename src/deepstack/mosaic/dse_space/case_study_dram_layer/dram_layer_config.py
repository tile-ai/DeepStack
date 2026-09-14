"""DRAM-layer DSE helpers with explicit custom-policy support.

The default path reproduces the paper using the bundled reference provider.
Every calibration that a caller may want to replace is represented by
``DramDsePolicy`` and handled in source when that policy is supplied.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any


SMEM_CAPACITIES = [128 * 1024, 256 * 1024, 512 * 1024]
L1_THROUGHPUTS = [128, 256, 512, 1024]
REFERENCE_THERMAL_R_BASE_C_PER_W = 0.56
REFERENCE_THERMAL_R_PER_LAYER_C_PER_W = 0.01


def _positive(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and greater than zero")
    return result


def _nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class DramDsePolicy:
    """Caller-defined policy for a reusable DRAM-layer design-space sweep.

    All values use SI units except ``round_trip_latency_cycles`` and the
    dimensionless exponents/factors. ``round_trip_latency_clock_hz`` may
    explicitly select the clock domain used to convert latency cycles. If it
    is omitted, an architecture-provided ``dram_latency_clock_hz`` takes
    precedence over the legacy ``core_freq`` fallback.

    Set ``round_trip_latency_cycles`` to ``None`` to consume
    ``arch.dram_round_trip_latency_cycles`` together with that same clock
    resolution. No paper calibration is used when a policy is passed to the
    helpers below.
    """

    capacity_per_layer_bytes: int
    round_trip_latency_cycles: float | None
    thermal_resistance_base_c_per_w: float
    thermal_resistance_per_layer_c_per_w: float
    thermal_baseline_layers: int
    thermal_design_power_w: float
    static_power_w: float
    area_budget_um2: float
    dynamic_power_exponent: float
    buffering_factor: float
    round_trip_latency_clock_hz: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.capacity_per_layer_bytes, bool)
            or not isinstance(self.capacity_per_layer_bytes, int)
            or self.capacity_per_layer_bytes <= 0
        ):
            raise ValueError("capacity_per_layer_bytes must be a positive integer")
        if (
            isinstance(self.thermal_baseline_layers, bool)
            or not isinstance(self.thermal_baseline_layers, int)
            or self.thermal_baseline_layers <= 0
        ):
            raise ValueError("thermal_baseline_layers must be a positive integer")
        for field in (
            "thermal_resistance_base_c_per_w",
            "thermal_design_power_w",
            "area_budget_um2",
            "dynamic_power_exponent",
            "buffering_factor",
        ):
            _positive(getattr(self, field), field)
        for field in (
            "thermal_resistance_per_layer_c_per_w",
        ):
            _nonnegative(getattr(self, field), field)
        if self.round_trip_latency_cycles is not None:
            _nonnegative(
                self.round_trip_latency_cycles,
                "round_trip_latency_cycles",
            )
        if self.round_trip_latency_clock_hz is not None:
            _positive(
                self.round_trip_latency_clock_hz,
                "round_trip_latency_clock_hz",
            )
        static = _nonnegative(self.static_power_w, "static_power_w")
        if static >= self.thermal_design_power_w:
            raise ValueError("static_power_w must be below thermal_design_power_w")

    def resolve_round_trip_latency_seconds(self, arch: Any) -> float:
        """Resolve caller-owned latency without consulting a reference model.

        Resolution is explicit and deterministic:

        1. latency cycles come from this policy unless they are ``None``, in
           which case ``arch.dram_round_trip_latency_cycles`` is required;
        2. an explicit policy clock wins, followed by
           ``arch.dram_latency_clock_hz``;
        3. ``arch.core_freq`` is retained only as the compatibility fallback
           for callers that do not expose an independent DRAM latency clock.
        """

        cycles = self.round_trip_latency_cycles
        if cycles is None:
            cycles = getattr(arch, "dram_round_trip_latency_cycles", None)
            if cycles is None:
                raise ValueError(
                    "round_trip_latency_cycles=None requires "
                    "arch.dram_round_trip_latency_cycles"
                )
            cycles = _nonnegative(
                cycles,
                "arch.dram_round_trip_latency_cycles",
            )

        clock = self.round_trip_latency_clock_hz
        if clock is None:
            clock = getattr(arch, "dram_latency_clock_hz", None)
        if clock is None:
            clock = getattr(arch, "core_freq", None)
        if clock is None:
            raise ValueError(
                "custom latency requires round_trip_latency_clock_hz, "
                "arch.dram_latency_clock_hz, or arch.core_freq"
            )
        clock_hz = _positive(clock, "resolved round-trip latency clock")
        return float(cycles) / clock_hz


ArchFactory = Callable[[], Any]
AreaEstimator = Callable[[Any, Any], float]
CapacityProvider = Callable[[Any, Any], int]


def _reference_provider() -> Any:
    """Load the binary reference provider only on a reference-path call."""

    from mosaic.arch import _reference_model

    return _reference_model


def _make_reference_arch() -> Any:
    """Construct the historical default architecture through a lazy import."""

    from mosaic.arch.stacked_gpu_wgmma import stacked_gpu_wgmma

    return stacked_gpu_wgmma()


def _make_reference_noc() -> Any:
    """Construct the historical default NoC through a lazy import."""

    from mosaic.noc.noc_config_set import torus_mesh_switch_1

    return torus_mesh_switch_1()


def _reference_capacity(arch: Any, noc: Any) -> int:
    """Resolve the sealed reference capacity through the one-decision facade."""

    from mosaic.cost.capacity import max_sm_count

    return int(max_sm_count(arch, noc))


def generate_dram_layer_configs() -> list[tuple[int, int, int, int]]:
    """Generate the paper search-space coordinates."""

    raw: list[tuple[int, int, int, int]] = []
    for total in range(1, 13):
        for smem_capacity in SMEM_CAPACITIES:
            for l1_throughput in L1_THROUGHPUTS:
                raw.append(
                    (total, total, smem_capacity, l1_throughput)
                )
    for total in (2, 4, 6, 8, 10, 12, 14, 16):
        for connected in range(2, total, 2):
            for smem_capacity in SMEM_CAPACITIES:
                for l1_throughput in L1_THROUGHPUTS:
                    raw.append(
                        (total, connected, smem_capacity, l1_throughput)
                    )

    configs: list[tuple[int, int, int, int]] = []
    for total, connected, smem_capacity, l1_throughput in raw:
        if connected >= 4 and l1_throughput <= 128:
            continue
        if connected >= 6 and l1_throughput <= 256:
            continue
        if connected >= 8 and l1_throughput <= 512:
            continue
        if total >= 10 and smem_capacity <= 128 * 1024:
            continue
        if total >= 12 and smem_capacity <= 256 * 1024:
            continue
        configs.append(
            (total, connected, smem_capacity, l1_throughput)
        )
    return configs


def compute_thermal_freq_scale(
    total_layers: int,
    *,
    policy: DramDsePolicy | None = None,
) -> float:
    """Compute a thermal frequency multiplier for reference or custom inputs."""

    if policy is None:
        return float(
            _reference_provider().q09(total_layers)
        )
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    resistance = (
        policy.thermal_resistance_base_c_per_w
        + policy.thermal_resistance_per_layer_c_per_w * total_layers
    )
    baseline_resistance = (
        policy.thermal_resistance_base_c_per_w
        + policy.thermal_resistance_per_layer_c_per_w
        * policy.thermal_baseline_layers
    )
    max_power = (
        policy.thermal_design_power_w * baseline_resistance / resistance
    )
    dynamic_available = max_power - policy.static_power_w
    dynamic_budget = policy.thermal_design_power_w - policy.static_power_w
    if dynamic_available <= 0.0:
        raise ValueError("custom thermal policy leaves no dynamic-power budget")
    return (dynamic_available / dynamic_budget) ** (
        1.0 / policy.dynamic_power_exponent
    )


def reference_thermal_resistance(total_layers: int) -> float:
    """Return the source-visible aggregate resistance used by the AE plots."""

    if isinstance(total_layers, bool) or not isinstance(total_layers, int):
        raise TypeError("total_layers must be an integer")
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    return (
        REFERENCE_THERMAL_R_BASE_C_PER_W
        + REFERENCE_THERMAL_R_PER_LAYER_C_PER_W * total_layers
    )


def apply_littles_law(
    arch: Any,
    *,
    policy: DramDsePolicy | None = None,
) -> tuple[bool, float]:
    """Cap ``arch.ddr_bandwidth`` using reference or caller-supplied latency."""

    if policy is None:
        limited, required = (
            _reference_provider().q08(arch)
        )
        return bool(limited), float(required)
    latency_s = policy.resolve_round_trip_latency_seconds(arch)
    bandwidth_per_sm = arch.ddr_bandwidth / arch.sm_count
    required_buffer = (
        bandwidth_per_sm * latency_s * policy.buffering_factor
    )
    if arch.configurable_smem_capacity >= required_buffer:
        return False, required_buffer
    arch.ddr_bandwidth *= (
        arch.configurable_smem_capacity / required_buffer
    )
    return True, required_buffer


def is_l1_bound(arch: Any) -> bool:
    """Return whether shared-memory throughput caps usable DRAM bandwidth."""

    # ``smem_bandwidth`` is the already-derived whole-chip aggregate.  Using
    # it here keeps the reference path on the public capability interface;
    # source-defined architectures still produce the same value through
    # ``update_derived``.
    smem_bandwidth_per_sm = arch.smem_bandwidth / arch.sm_count
    dram_bandwidth_per_sm = arch.ddr_bandwidth / arch.sm_count
    return smem_bandwidth_per_sm < 2 * dram_bandwidth_per_sm


def get_actual_bw(arch: Any) -> float:
    """Return usable bandwidth after Little's-law and shared-memory caps."""

    if is_l1_bound(arch):
        return float(arch.smem_bandwidth) / 2
    return float(arch.ddr_bandwidth)


def _configure_arch_dram(
    arch: Any,
    total_layers: int,
    active_layers: int,
    *,
    policy: DramDsePolicy | None,
) -> Any:
    if policy is None:
        return arch.with_dram(total_layers, active_layers)

    arch.update_ddr(
        total_layers=total_layers,
        active_layers=active_layers,
    )
    arch.ddr_capacity = (
        total_layers * policy.capacity_per_layer_bytes
    )
    return arch


def _make_raw_arch(
    total_layers: int,
    active_layers: int,
    smem_capacity: int,
    l1_throughput: int,
    sm_count: int | None = None,
    *,
    policy: DramDsePolicy | None = None,
    arch_factory: ArchFactory = _make_reference_arch,
) -> Any:
    """Build an unscaled architecture for reference or custom DSE inputs."""

    arch = arch_factory()
    if policy is None:
        return arch.with_design_point(
            sm_count=arch.sm_count if sm_count is None else sm_count,
            total_layers=total_layers,
            active_layers=active_layers,
            smem_capacity=smem_capacity,
            l1_throughput_bpc=l1_throughput,
            thermal_scale=1.0,
        )

    if sm_count is not None:
        arch.sm_count = sm_count
    arch = _configure_arch_dram(
        arch, total_layers, active_layers, policy=policy
    )
    arch.configurable_smem_capacity = smem_capacity
    arch.l1_smem_throughput_per_cycle = l1_throughput
    arch.update_derived()
    return arch


def find_max_sm_count(
    total_layers: int,
    active_layers: int,
    smem_capacity: int,
    l1_throughput: int,
    area_budget_um2: float | None = None,
    *,
    policy: DramDsePolicy | None = None,
    arch_factory: ArchFactory = _make_reference_arch,
    noc_factory: Callable[[], Any] = _make_reference_noc,
    capacity_provider: CapacityProvider | None = None,
    area_estimator: AreaEstimator | None = None,
) -> int:
    """Find the largest feasible PE count without exposing reference area.

    The default reference path delegates the complete capacity decision to the
    bundled binary and therefore accepts neither a caller-selected area budget
    nor a caller-selected area estimator.  Reusable custom paths remain fully
    source-visible: callers may provide a direct ``capacity_provider`` or an
    explicit ``policy`` plus ``area_estimator``.
    """

    noc = noc_factory()

    if capacity_provider is not None:
        if area_estimator is not None:
            raise ValueError(
                "capacity_provider and area_estimator are mutually exclusive"
            )
        template = _make_raw_arch(
            total_layers,
            active_layers,
            smem_capacity,
            l1_throughput,
            sm_count=1,
            policy=policy,
            arch_factory=arch_factory,
        )
        result = capacity_provider(template, noc)
        if isinstance(result, bool) or not isinstance(result, int):
            raise TypeError("capacity_provider must return an integer SM count")
        if not 1 <= result <= 128:
            raise ValueError("capacity_provider result must be in [1, 128]")
        return result

    if policy is None and area_estimator is None:
        if area_budget_um2 is not None:
            raise ValueError(
                "the sealed reference capacity uses its fixed die budget; "
                "custom budgets require a policy and area_estimator"
            )
        if arch_factory is not _make_reference_arch:
            raise ValueError(
                "a custom arch_factory requires capacity_provider or "
                "policy plus area_estimator"
            )
        template = _make_raw_arch(
            total_layers,
            active_layers,
            smem_capacity,
            l1_throughput,
            sm_count=1,
            arch_factory=arch_factory,
        )
        return _reference_capacity(template, noc)

    if area_estimator is None:
        raise ValueError(
            "custom area policies require an explicit area_estimator or "
            "capacity_provider"
        )
    explicit_budget = (
        _positive(area_budget_um2, "area_budget_um2")
        if area_budget_um2 is not None
        else policy.area_budget_um2
        if policy is not None
        else None
    )
    if explicit_budget is None:
        raise ValueError(
            "area_estimator requires area_budget_um2 or a DramDsePolicy"
        )

    low, high, best = 1, 128, 1
    while low <= high:
        midpoint = (low + high) // 2
        arch = _make_raw_arch(
            total_layers,
            active_layers,
            smem_capacity,
            l1_throughput,
            sm_count=midpoint,
            policy=policy,
            arch_factory=arch_factory,
        )
        area = _nonnegative(
            area_estimator(arch, noc),
            "area_estimator result",
        )
        fits = area <= explicit_budget
        if fits:
            best = midpoint
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def make_arch_for_config(
    total_layers: int,
    active_layers: int,
    smem_capacity: int,
    l1_throughput: int,
    apply_thermal: bool = True,
    area_budget_um2: float | None = None,
    *,
    policy: DramDsePolicy | None = None,
    arch_factory: ArchFactory = _make_reference_arch,
    noc_factory: Callable[[], Any] = _make_reference_noc,
    capacity_provider: CapacityProvider | None = None,
    area_estimator: AreaEstimator | None = None,
) -> tuple[Any, float, bool, float, int]:
    """Construct a complete architecture for one DRAM-layer DSE coordinate."""

    sm_count = find_max_sm_count(
        total_layers,
        active_layers,
        smem_capacity,
        l1_throughput,
        area_budget_um2,
        policy=policy,
        arch_factory=arch_factory,
        noc_factory=noc_factory,
        capacity_provider=capacity_provider,
        area_estimator=area_estimator,
    )
    frequency_scale = (
        compute_thermal_freq_scale(total_layers, policy=policy)
        if apply_thermal
        else 1.0
    )
    arch = arch_factory()
    if policy is None:
        # Reference primitives stay inside the provider.  The returned
        # architecture exposes only the aggregate capability snapshot used
        # by the source-visible models.
        arch = arch.with_design_point(
            sm_count=sm_count,
            total_layers=total_layers,
            active_layers=active_layers,
            smem_capacity=smem_capacity,
            l1_throughput_bpc=l1_throughput,
            thermal_scale=frequency_scale,
        )
        arch, limited, required_buffer = arch.with_littles_law()
    else:
        arch.sm_count = sm_count
        arch = _configure_arch_dram(
            arch, total_layers, active_layers, policy=policy
        )
        arch.configurable_smem_capacity = smem_capacity
        arch.l1_smem_throughput_per_cycle = l1_throughput

        # Custom-policy architectures intentionally retain their explicit
        # clock-domain interface.
        arch.core_freq *= frequency_scale
        arch.memory_freq *= frequency_scale
        arch.noc_freq *= frequency_scale
        arch.update_derived()
        limited, required_buffer = apply_littles_law(
            arch,
            policy=policy,
        )
    return (
        arch,
        frequency_scale,
        limited,
        required_buffer,
        sm_count,
    )


def compute_power_wall(
    chip_power_w: float,
    noc_power_per_device_w: float,
    tdp: float | None = None,
    *,
    policy: DramDsePolicy | None = None,
) -> tuple[bool, float]:
    """Return power-wall status and an optional cubic frequency scale."""

    chip = _nonnegative(chip_power_w, "chip_power_w")
    noc = _nonnegative(
        noc_power_per_device_w,
        "noc_power_per_device_w",
    )
    if policy is None and tdp is None:
        hit, scale = _reference_provider().q11(chip, noc)
        return bool(hit), float(scale)
    limit = (
        policy.thermal_design_power_w
        if policy is not None
        else _positive(tdp, "tdp")
    )
    exponent = (
        policy.dynamic_power_exponent if policy is not None else 3.0
    )
    total = chip + noc
    if total > limit:
        return True, 1.0 / ((total / limit) ** (1.0 / exponent))
    return False, 1.0
