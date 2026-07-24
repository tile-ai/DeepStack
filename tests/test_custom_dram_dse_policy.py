from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace

import pytest

from ae.paths import DEEPSTACK_SRC, TILESIGHT_SRC, activate_vendored_sources


activate_vendored_sources()

from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (  # noqa: E402
    DramDsePolicy,
    apply_littles_law,
    compute_power_wall,
    compute_thermal_freq_scale,
    find_max_sm_count,
    make_arch_for_config,
)


def _synthetic_policy(**overrides: object) -> DramDsePolicy:
    values: dict[str, object] = {
        "capacity_per_layer_bytes": 111_111_111,
        "round_trip_latency_cycles": 17.5,
        "thermal_resistance_base_c_per_w": 0.42,
        "thermal_resistance_per_layer_c_per_w": 0.007,
        "thermal_baseline_layers": 3,
        "thermal_design_power_w": 73.0,
        "static_power_w": 8.0,
        "area_budget_um2": 55.0,
        "dynamic_power_exponent": 2.5,
        "buffering_factor": 1.75,
    }
    values.update(overrides)
    return DramDsePolicy(**values)


class _SyntheticArch:
    def __init__(self) -> None:
        self.sm_count = 1
        self.core_freq = 1.25e9
        self.memory_freq = 0.31e9
        self.noc_freq = 0.87e9
        self.max_freq = self.core_freq
        self.configurable_smem_capacity = 256
        self.l1_smem_throughput_per_cycle = 32
        self.ddr_peak_bandwidth = 0.0
        self.ddr_bandwidth = 0.0
        self.ddr_capacity = 0

    def update_ddr(self, total_layers: int, active_layers: int) -> None:
        self.dram_layers_per_cluster = total_layers
        self.dram_active_layers = active_layers
        self.ddr_peak_bandwidth = active_layers * 1_000.0
        self.ddr_bandwidth = self.ddr_peak_bandwidth

    def update_derived(self) -> None:
        self.synthetic_compute_rate = self.sm_count * self.max_freq


def test_custom_dse_import_and_execution_do_not_load_binary_providers() -> None:
    script = f"""
import importlib.abc
import sys

sys.path[:0] = [{str(DEEPSTACK_SRC)!r}, {str(TILESIGHT_SRC)!r}]
blocked = {{
    "mosaic.arch._reference_model",
    "mosaic.cost.area",
    "mosaic.cost._capacity",
    "mosaic.noc._model_support",
    "tilesight.distributed.noc._model_support",
}}

class BlockBinaryProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise ImportError("blocked proprietary provider: " + fullname)
        return None

sys.meta_path.insert(0, BlockBinaryProviders())

import mosaic.dse_space as dse_space

assert {{
    "all_parallel_schemes",
    "arch_noc_combinations1",
    "task_combination_2",
}} <= set(dse_space.__all__)
from mosaic.dse_space import all_parallel_schemes, task_combination_2

assert callable(all_parallel_schemes)
assert callable(task_combination_2)
assert blocked.isdisjoint(sys.modules)

from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
    DramDsePolicy,
    make_arch_for_config,
)

assert blocked.isdisjoint(sys.modules)

class SourceOnlyArch:
    def __init__(self):
        self.sm_count = 1
        self.core_freq = 1.1e9
        self.memory_freq = 0.3e9
        self.noc_freq = 0.7e9
        self.max_freq = self.core_freq
        self.configurable_smem_capacity = 512
        self.l1_smem_throughput_per_cycle = 48
        self.ddr_bandwidth = 0.0
        self.ddr_capacity = 0

    def update_ddr(self, total_layers, active_layers):
        self.dram_layers_per_cluster = total_layers
        self.dram_active_layers = active_layers
        self.ddr_bandwidth = active_layers * 1000.0

    def update_derived(self):
        self.synthetic_compute_rate = self.sm_count * self.max_freq

policy = DramDsePolicy(
    capacity_per_layer_bytes=12345,
    round_trip_latency_cycles=11.0,
    round_trip_latency_clock_hz=0.25e9,
    thermal_resistance_base_c_per_w=0.4,
    thermal_resistance_per_layer_c_per_w=0.002,
    thermal_baseline_layers=2,
    thermal_design_power_w=80.0,
    static_power_w=10.0,
    area_budget_um2=4.5,
    dynamic_power_exponent=3.0,
    buffering_factor=1.25,
)
arch, _, _, _, sm_count = make_arch_for_config(
    6,
    3,
    768,
    64,
    policy=policy,
    arch_factory=SourceOnlyArch,
    noc_factory=object,
    area_estimator=lambda candidate, _noc: float(candidate.sm_count),
)
assert sm_count == 4
assert arch.ddr_capacity == 6 * policy.capacity_per_layer_bytes
assert blocked.isdisjoint(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_custom_policy_drives_thermal_equation() -> None:
    policy = _synthetic_policy()
    total_layers = 9
    resistance = (
        policy.thermal_resistance_base_c_per_w
        + policy.thermal_resistance_per_layer_c_per_w * total_layers
    )
    baseline = (
        policy.thermal_resistance_base_c_per_w
        + policy.thermal_resistance_per_layer_c_per_w
        * policy.thermal_baseline_layers
    )
    available = policy.thermal_design_power_w * baseline / resistance
    expected = (
        (available - policy.static_power_w)
        / (policy.thermal_design_power_w - policy.static_power_w)
    ) ** (1.0 / policy.dynamic_power_exponent)

    assert compute_thermal_freq_scale(
        total_layers,
        policy=policy,
    ) == pytest.approx(expected)


def test_zero_latency_and_layer_thermal_increment_are_supported() -> None:
    policy = _synthetic_policy(
        round_trip_latency_cycles=0.0,
        thermal_resistance_per_layer_c_per_w=0.0,
        static_power_w=0.0,
    )
    arch = _SyntheticArch()
    arch.update_ddr(5, 3)

    limited, required = apply_littles_law(arch, policy=policy)

    assert limited is False
    assert required == 0.0
    assert compute_thermal_freq_scale(11, policy=policy) == pytest.approx(1.0)
    assert compute_power_wall(0.0, 0.0, policy=policy) == (False, 1.0)


def test_littles_law_consumes_caller_owned_arch_dram_timing() -> None:
    arch = _SyntheticArch()
    arch.sm_count = 4
    arch.ddr_bandwidth = 8.0e9
    arch.configurable_smem_capacity = 10_000
    arch.dram_round_trip_latency_cycles = 29.0
    arch.dram_latency_clock_hz = 0.2e9
    policy = _synthetic_policy(
        round_trip_latency_cycles=None,
        buffering_factor=1.5,
    )

    limited, required = apply_littles_law(arch, policy=policy)
    expected = (
        arch.ddr_bandwidth
        / arch.sm_count
        * (arch.dram_round_trip_latency_cycles / arch.dram_latency_clock_hz)
        * policy.buffering_factor
    )
    legacy_core_clock_result = (
        arch.ddr_bandwidth
        / arch.sm_count
        * (arch.dram_round_trip_latency_cycles / arch.core_freq)
        * policy.buffering_factor
    )

    assert limited is False
    assert required == pytest.approx(expected)
    assert required != pytest.approx(legacy_core_clock_result)


def test_explicit_policy_latency_clock_overrides_arch_clock() -> None:
    arch = _SyntheticArch()
    arch.sm_count = 5
    arch.ddr_bandwidth = 9.0e9
    arch.configurable_smem_capacity = 10_000
    arch.dram_latency_clock_hz = 0.19e9
    policy = _synthetic_policy(
        round_trip_latency_cycles=23.0,
        round_trip_latency_clock_hz=0.47e9,
        buffering_factor=1.25,
    )

    limited, required = apply_littles_law(arch, policy=policy)
    expected = (
        arch.ddr_bandwidth
        / arch.sm_count
        * (
            policy.round_trip_latency_cycles
            / policy.round_trip_latency_clock_hz
        )
        * policy.buffering_factor
    )

    assert limited is False
    assert required == pytest.approx(expected)


def test_none_latency_cycles_require_arch_dram_timing() -> None:
    arch = _SyntheticArch()
    arch.update_ddr(5, 3)

    with pytest.raises(
        ValueError,
        match="arch.dram_round_trip_latency_cycles",
    ):
        apply_littles_law(
            arch,
            policy=_synthetic_policy(round_trip_latency_cycles=None),
        )


def test_custom_area_estimator_replaces_binary_area_policy() -> None:
    policy = _synthetic_policy(area_budget_um2=55.0)
    visited: list[int] = []

    def area_estimator(arch: _SyntheticArch, noc: object) -> float:
        del noc
        visited.append(arch.sm_count)
        return 10.0 + arch.sm_count * 10.0

    best = find_max_sm_count(
        6,
        2,
        512,
        48,
        policy=policy,
        arch_factory=_SyntheticArch,
        noc_factory=object,
        area_estimator=area_estimator,
    )

    assert best == 4
    assert visited


def test_make_arch_uses_all_caller_owned_dse_hooks() -> None:
    policy = _synthetic_policy(
        round_trip_latency_cycles=0.0,
        area_budget_um2=45.0,
    )
    arch, scale, limited, required, sm_count = make_arch_for_config(
        7,
        3,
        640,
        52,
        apply_thermal=False,
        policy=policy,
        arch_factory=_SyntheticArch,
        noc_factory=object,
        area_estimator=lambda candidate, _noc: 5.0 + 10.0 * candidate.sm_count,
    )

    assert sm_count == 4
    assert arch.sm_count == 4
    assert arch.dram_layers_per_cluster == 7
    assert arch.dram_active_layers == 3
    assert arch.ddr_capacity == 7 * policy.capacity_per_layer_bytes
    assert arch.configurable_smem_capacity == 640
    assert arch.l1_smem_throughput_per_cycle == 52
    assert scale == 1.0
    assert limited is False
    assert required == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("round_trip_latency_cycles", -1.0),
        ("round_trip_latency_clock_hz", 0.0),
        ("round_trip_latency_clock_hz", -1.0),
        ("round_trip_latency_clock_hz", math.inf),
        ("thermal_resistance_per_layer_c_per_w", -0.1),
        ("static_power_w", -0.1),
        ("dynamic_power_exponent", 0.0),
        ("area_budget_um2", math.inf),
    ],
)
def test_custom_policy_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_synthetic_policy(), **{field: value})


def test_custom_area_estimator_must_return_nonnegative_finite_value() -> None:
    with pytest.raises(ValueError, match="area_estimator result"):
        find_max_sm_count(
            3,
            2,
            256,
            32,
            policy=_synthetic_policy(),
            arch_factory=_SyntheticArch,
            noc_factory=object,
            area_estimator=lambda _arch, _noc: -1.0,
        )
