"""Source-visible builders for caller-defined stacked-GPU architectures.

The paper-facing reference profiles are packaged separately from this module.
This file contains only generic equations and validation so that artifact users
can substitute their own clocks, compute resources, cache sizes, and DRAM
interface assumptions without depending on any built-in calibration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

from tilesight.arch.arch_base import Arch


if TYPE_CHECKING:
    from mosaic.cost.energy import ChipEnergyConfig


__all__ = [
    "ConfigurableStackedGpu",
    "DramInterfaceConfig",
    "DramTimingConfig",
    "StackedGpuConfig",
    "dram_connectivity_efficiency",
    "make_custom_stacked_gpu",
]


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return result


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _finite_number(
    value: object,
    field: str,
    *,
    allow_zero: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if result < 0.0 or (result == 0.0 and not allow_zero):
        relation = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{field} must be {relation}")
    return result


def _fraction(value: object, field: str) -> float:
    result = _finite_number(value, field, allow_zero=True)
    if result > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def dram_connectivity_efficiency(
    total_layers: int,
    connected_layers: int,
    *,
    fully_connected_efficiency: float,
) -> float:
    """Interpolate a caller-supplied fully-connected DRAM efficiency.

    The generic model assumes that recharge can be fully hidden when no more
    than half of the physical layers are connected.  Between that point and a
    fully connected stack it linearly interpolates to
    ``fully_connected_efficiency``.  Supplying the endpoint explicitly keeps
    the function reusable without embedding a paper-specific timing ratio.
    """

    total = _positive_int(total_layers, "total_layers")
    connected = _positive_int(connected_layers, "connected_layers")
    if connected > total:
        raise ValueError("connected_layers must not exceed total_layers")
    endpoint = _fraction(
        fully_connected_efficiency,
        "fully_connected_efficiency",
    )
    half = total / 2.0
    if connected <= half:
        return 1.0
    if connected >= total:
        return endpoint
    position = (connected - half) / (total - half)
    return 1.0 + position * (endpoint - 1.0)


@dataclass(frozen=True)
class DramTimingConfig:
    """Caller-owned DRAM row service and latency timing.

    This interface deliberately exposes physical construction inputs for
    custom studies without supplying a reference default. ``row_read_cycles``
    and the fully connected efficiency are derived from the caller's row,
    sector, and recharge values. Bank service cycles and round-trip latency
    cycles use independent caller-supplied clock domains.
    """

    row_bytes: int
    sector_bytes: int
    sector_cycles: int
    recharge_cycles: int
    round_trip_latency_cycles: float
    data_rate_hz: float
    latency_clock_hz: float

    def __post_init__(self) -> None:
        row = _positive_int(self.row_bytes, "row_bytes")
        sector = _positive_int(self.sector_bytes, "sector_bytes")
        if row % sector:
            raise ValueError("row_bytes must be an integer multiple of sector_bytes")
        _positive_int(self.sector_cycles, "sector_cycles")
        _nonnegative_int(self.recharge_cycles, "recharge_cycles")
        _finite_number(
            self.round_trip_latency_cycles,
            "round_trip_latency_cycles",
            allow_zero=True,
        )
        _finite_number(self.data_rate_hz, "data_rate_hz")
        _finite_number(self.latency_clock_hz, "latency_clock_hz")

    @property
    def sectors_per_row(self) -> int:
        return self.row_bytes // self.sector_bytes

    @property
    def row_read_cycles(self) -> int:
        return self.sectors_per_row * self.sector_cycles

    @property
    def full_row_cycles(self) -> int:
        return self.row_read_cycles + self.recharge_cycles

    @property
    def fully_connected_efficiency(self) -> float:
        return self.row_read_cycles / self.full_row_cycles

    @property
    def round_trip_latency_seconds(self) -> float:
        return self.round_trip_latency_cycles / self.latency_clock_hz


@dataclass(frozen=True)
class DramInterfaceConfig:
    """Explicit parameters for one configurable stacked-DRAM interface."""

    total_layers: int
    connected_layers: int
    channels_per_connected_layer: int
    bytes_per_channel_transfer: float | None
    transfers_per_memory_clock: float | None
    capacity_per_layer_bytes: int
    transaction_bytes: int
    fully_connected_efficiency: float | None
    uncached_max_utilization: float
    l2_bandwidth_multiplier: float = 1.0
    direct_peak_bandwidth_per_connected_layer_bytes_per_s: float | None = None
    bank_timing: DramTimingConfig | None = None

    def __post_init__(self) -> None:
        total = _positive_int(self.total_layers, "total_layers")
        connected = _positive_int(self.connected_layers, "connected_layers")
        if connected > total:
            raise ValueError("connected_layers must not exceed total_layers")
        _positive_int(
            self.channels_per_connected_layer,
            "channels_per_connected_layer",
        )
        direct_peak = self.direct_peak_bandwidth_per_connected_layer_bytes_per_s
        if direct_peak is None:
            if (
                self.bytes_per_channel_transfer is None
                or self.transfers_per_memory_clock is None
            ):
                raise ValueError(
                    "transfer composition requires bytes_per_channel_transfer "
                    "and transfers_per_memory_clock"
                )
            _finite_number(
                self.bytes_per_channel_transfer,
                "bytes_per_channel_transfer",
            )
            _finite_number(
                self.transfers_per_memory_clock,
                "transfers_per_memory_clock",
            )
        else:
            _finite_number(
                direct_peak,
                "direct_peak_bandwidth_per_connected_layer_bytes_per_s",
            )
            if (
                self.bytes_per_channel_transfer is not None
                or self.transfers_per_memory_clock is not None
            ):
                raise ValueError(
                    "direct peak bandwidth and transfer composition are "
                    "mutually exclusive"
                )
        _positive_int(self.capacity_per_layer_bytes, "capacity_per_layer_bytes")
        _positive_int(self.transaction_bytes, "transaction_bytes")
        if self.bank_timing is None:
            if self.fully_connected_efficiency is None:
                raise ValueError(
                    "supply fully_connected_efficiency or bank_timing"
                )
            _fraction(
                self.fully_connected_efficiency,
                "fully_connected_efficiency",
            )
        else:
            if not isinstance(self.bank_timing, DramTimingConfig):
                raise TypeError("bank_timing must be a DramTimingConfig")
            if self.fully_connected_efficiency is not None:
                raise ValueError(
                    "fully_connected_efficiency and bank_timing are "
                    "mutually exclusive"
                )
        _fraction(
            self.uncached_max_utilization,
            "uncached_max_utilization",
        )
        _finite_number(
            self.l2_bandwidth_multiplier,
            "l2_bandwidth_multiplier",
        )

    @property
    def resolved_fully_connected_efficiency(self) -> float:
        if self.bank_timing is not None:
            return self.bank_timing.fully_connected_efficiency
        assert self.fully_connected_efficiency is not None
        return float(self.fully_connected_efficiency)

    def peak_bandwidth_bytes_per_s(
        self,
        memory_frequency_hz: float,
        *,
        connected_layers: int | None = None,
    ) -> float:
        connected = (
            self.connected_layers
            if connected_layers is None
            else _positive_int(connected_layers, "connected_layers")
        )
        memory_frequency = _finite_number(
            memory_frequency_hz,
            "memory_frequency_hz",
        )
        direct_peak = self.direct_peak_bandwidth_per_connected_layer_bytes_per_s
        if direct_peak is not None:
            return connected * direct_peak
        assert self.bytes_per_channel_transfer is not None
        assert self.transfers_per_memory_clock is not None
        return (
            connected
            * self.channels_per_connected_layer
            * self.bytes_per_channel_transfer
            * self.transfers_per_memory_clock
            * memory_frequency
        )


@dataclass(frozen=True)
class StackedGpuConfig:
    """Complete caller-owned input for :class:`ConfigurableStackedGpu`."""

    name: str
    sm_count: int
    core_frequency_hz: float
    memory_frequency_hz: float
    noc_frequency_hz: float
    tensor_cores_per_sm: int
    tensor_core_shape: tuple[int, int, int]
    fp32_cores_per_sm: int
    int32_cores_per_sm: int
    sfu_cores_per_sm: int
    sm_sub_partitions: int
    shared_memory_throughput_bytes_per_cycle: float
    shared_memory_capacity_bytes: int
    register_capacity_per_sm_bytes: int
    warp_schedulers_per_sm: int
    l2_capacity_bytes: int
    l1_noc_bytes_per_cycle: float
    dram: DramInterfaceConfig
    tensor_instruction_shapes: Mapping[float, tuple[int, int, int]]
    chip_energy_config: "ChipEnergyConfig | None" = None
    ddr_max_utilization: float = 0.9
    l2_max_utilization: float = 0.9
    l1_max_utilization: float = 0.9
    compute_max_utilization: float = 0.9
    support_wgmma: bool = False
    support_utcmma: bool = False
    use_tensor_core_resource_model: bool = False
    apply_dram_wave_quantization: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        for field in (
            "sm_count",
            "tensor_cores_per_sm",
            "fp32_cores_per_sm",
            "int32_cores_per_sm",
            "sfu_cores_per_sm",
            "sm_sub_partitions",
            "shared_memory_capacity_bytes",
            "register_capacity_per_sm_bytes",
            "warp_schedulers_per_sm",
            "l2_capacity_bytes",
        ):
            _positive_int(getattr(self, field), field)
        for field in (
            "core_frequency_hz",
            "memory_frequency_hz",
            "noc_frequency_hz",
            "shared_memory_throughput_bytes_per_cycle",
            "l1_noc_bytes_per_cycle",
        ):
            _finite_number(getattr(self, field), field)
        if (
            not isinstance(self.tensor_core_shape, tuple)
            or len(self.tensor_core_shape) != 3
        ):
            raise TypeError("tensor_core_shape must be a three-integer tuple")
        for dimension in self.tensor_core_shape:
            _positive_int(dimension, "tensor_core_shape dimension")
        if not isinstance(self.tensor_instruction_shapes, Mapping):
            raise TypeError("tensor_instruction_shapes must be a mapping")
        if not self.tensor_instruction_shapes:
            raise ValueError("tensor_instruction_shapes must not be empty")
        for bytes_per_element, shape in self.tensor_instruction_shapes.items():
            _finite_number(bytes_per_element, "tensor instruction byte width")
            if not isinstance(shape, tuple) or len(shape) != 3:
                raise TypeError(
                    "every tensor instruction shape must be a three-integer tuple"
                )
            for dimension in shape:
                _positive_int(
                    dimension,
                    "tensor instruction shape dimension",
                )
        if self.chip_energy_config is not None:
            from mosaic.cost.energy import ChipEnergyConfig

            if not isinstance(self.chip_energy_config, ChipEnergyConfig):
                raise TypeError(
                    "chip_energy_config must be a ChipEnergyConfig or None"
                )
        for field in (
            "ddr_max_utilization",
            "l2_max_utilization",
            "l1_max_utilization",
            "compute_max_utilization",
        ):
            _fraction(getattr(self, field), field)
        if not isinstance(self.support_wgmma, bool):
            raise TypeError("support_wgmma must be a bool")
        if not isinstance(self.support_utcmma, bool):
            raise TypeError("support_utcmma must be a bool")
        if not isinstance(self.use_tensor_core_resource_model, bool):
            raise TypeError("use_tensor_core_resource_model must be a bool")
        if not isinstance(self.apply_dram_wave_quantization, bool):
            raise TypeError("apply_dram_wave_quantization must be a bool")


class ConfigurableStackedGpu(Arch):
    """Architecture object derived only from an explicit user configuration."""

    def __init__(self, config: StackedGpuConfig):
        if not isinstance(config, StackedGpuConfig):
            raise TypeError("config must be a StackedGpuConfig")
        super().__init__()
        self.config = config
        self.core = config.name
        self.sm_count = config.sm_count
        self.core_freq = config.core_frequency_hz
        self.memory_freq = config.memory_frequency_hz
        self.noc_freq = config.noc_frequency_hz
        self.base_freq = self.core_freq
        self.max_freq = self.core_freq
        self.tensor_cores_per_sm = config.tensor_cores_per_sm
        self.tensor_core_shape = config.tensor_core_shape
        self.fp32_cores_per_sm = config.fp32_cores_per_sm
        self.int32_cores_per_sm = config.int32_cores_per_sm
        self.sfu_cores_per_sm = config.sfu_cores_per_sm
        self.sm_sub_partitions = config.sm_sub_partitions
        self.l1_smem_throughput_per_cycle = (
            config.shared_memory_throughput_bytes_per_cycle
        )
        self.configurable_smem_capacity = config.shared_memory_capacity_bytes
        self.register_capacity_per_sm = config.register_capacity_per_sm_bytes
        self.warp_schedulers_per_sm = config.warp_schedulers_per_sm
        self.l2_capacity = config.l2_capacity_bytes
        self._l1_noc_bytes_per_cycle = config.l1_noc_bytes_per_cycle
        self._tensor_instruction_shapes = dict(config.tensor_instruction_shapes)
        self.ddr_max_util = config.ddr_max_utilization
        self.l2_max_util = config.l2_max_utilization
        self.l1_max_util = config.l1_max_utilization
        self.compute_max_util = config.compute_max_utilization
        self.support_wgmma = config.support_wgmma
        self.support_utcmma = config.support_utcmma
        self.use_tensor_core_resource_model = (
            config.use_tensor_core_resource_model
        )
        self.apply_dram_wave_quantization = (
            config.apply_dram_wave_quantization
        )
        self.chip_energy_config = config.chip_energy_config
        self.ddr_transaction_size = config.dram.transaction_bytes
        self.ddr_stack = config.dram.channels_per_connected_layer
        self.dram_3d_uncached_max_util = (
            config.dram.uncached_max_utilization
        )
        timing = config.dram.bank_timing
        if timing is not None:
            self.dram_row_bytes = timing.row_bytes
            self.dram_sector_bytes = timing.sector_bytes
            self.dram_sector_cycles = timing.sector_cycles
            self.dram_recharge_cycles = timing.recharge_cycles
            self.dram_round_trip_latency_cycles = (
                timing.round_trip_latency_cycles
            )
            self.dram_data_rate_hz = timing.data_rate_hz
            self.dram_latency_clock_hz = timing.latency_clock_hz
        self.update_ddr(
            config.dram.total_layers,
            config.dram.connected_layers,
        )
        self.update_derived()

    def update_ddr(
        self,
        total_layers: int,
        active_layers: int | None = None,
    ) -> None:
        total = _positive_int(total_layers, "total_layers")
        connected = (
            total
            if active_layers is None
            else _positive_int(active_layers, "active_layers")
        )
        if connected > total:
            raise ValueError("active_layers must not exceed total_layers")
        dram = self.config.dram
        self.dram_layers_per_cluster = total
        self.dram_active_layers = connected
        self.ddr_peak_bandwidth = dram.peak_bandwidth_bytes_per_s(
            self.memory_freq,
            connected_layers=connected,
        )
        efficiency = dram_connectivity_efficiency(
            total,
            connected,
            fully_connected_efficiency=(
                dram.resolved_fully_connected_efficiency
            ),
        )
        self.ddr_bandwidth = self.ddr_peak_bandwidth * efficiency
        self.ddr_capacity = total * dram.capacity_per_layer_bytes
        self.l2_bandwidth = (
            self.ddr_peak_bandwidth * dram.l2_bandwidth_multiplier
        )
        self.ddr_wave_bytes = self.ddr_transaction_size * self.ddr_stack

    def update_derived(self) -> None:
        self.layer1_noc_size = self.sm_count
        self.layer1_noc = "xbar"
        self.layer1_noc_single_direction_bw = (
            self._l1_noc_bytes_per_cycle * self.noc_freq
        )
        self.tensor_core_flops = math.prod(self.tensor_core_shape) * 2
        self.fp16_tensor_flops = (
            self.sm_count
            * self.max_freq
            * self.tensor_cores_per_sm
            * self.tensor_core_flops
        )
        self.fp32_tensor_flops = (
            self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        )
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2
        self.int8_tensor_flops = self.fp16_tensor_flops * 2
        self.fp32_cuda_core_flops = self.fp32_tensor_flops
        self.fp16_cuda_core_flops = self.fp32_tensor_flops
        self.fp8_cuda_core_flops = self.fp32_tensor_flops
        self.fp64_cuda_core_flops = self.fp32_tensor_flops * 0.5
        self.int32_cuda_core_flops = (
            self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        )
        self.sfu_flops = (
            self.sm_count * self.max_freq * self.sfu_cores_per_sm
        )
        self.smem_bandwidth = (
            self.sm_count
            * self.max_freq
            * self.l1_smem_throughput_per_cycle
        )
        self.smem_l2_bandwidth = self.smem_bandwidth
        self.l2_to_smem_bandwidth = self.smem_bandwidth * 0.5
        self.smem_to_l2_bandwidth = self.smem_bandwidth * 0.5
        self.smem_register_bandwidth = self.smem_bandwidth
        self.register_bandwidth = (
            self.sm_count
            * self.max_freq
            * self.sm_sub_partitions
            * 32
            * 4
        )

    def get_tensor_core_minimum_ptx(
        self,
        bytes: float = 2,
    ) -> tuple[int, int, int]:
        try:
            return self._tensor_instruction_shapes[bytes]
        except KeyError as exc:
            choices = ", ".join(
                str(value) for value in sorted(self._tensor_instruction_shapes)
            )
            raise ValueError(
                f"unsupported tensor element width {bytes}; choose one of: {choices}"
            ) from exc

    def set_to_spec(self) -> "ConfigurableStackedGpu":
        return self

    def set_to_microbench(self) -> "ConfigurableStackedGpu":
        return self

    def set_to_ncu(self) -> "ConfigurableStackedGpu":
        return self


def make_custom_stacked_gpu(config: StackedGpuConfig) -> ConfigurableStackedGpu:
    """Build a configurable architecture without loading reference defaults."""

    return ConfigurableStackedGpu(config)
