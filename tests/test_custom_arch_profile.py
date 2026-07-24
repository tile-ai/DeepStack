from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace

import pytest

from ae.paths import DEEPSTACK_SRC, TILESIGHT_SRC, activate_vendored_sources


activate_vendored_sources()

from bank_oracle import DramBankSpec  # noqa: E402
from mosaic.arch.custom_profile import (  # noqa: E402
    ConfigurableStackedGpu,
    DramInterfaceConfig,
    DramTimingConfig,
    StackedGpuConfig,
    dram_connectivity_efficiency,
    make_custom_stacked_gpu,
)
from mosaic.cost.energy import ChipEnergyConfig  # noqa: E402
from tilesight.arch import (  # noqa: E402
    uses_dram_wave_quantization,
    uses_gpu_resource_model,
)


def _synthetic_dram() -> DramInterfaceConfig:
    """Return conspicuously fictional values for public-interface testing."""

    return DramInterfaceConfig(
        total_layers=10,
        connected_layers=3,
        channels_per_connected_layer=7,
        bytes_per_channel_transfer=5.5,
        transfers_per_memory_clock=1.25,
        capacity_per_layer_bytes=123_456_789,
        transaction_bytes=96,
        fully_connected_efficiency=0.37,
        uncached_max_utilization=0.63,
        l2_bandwidth_multiplier=1.75,
    )


def _synthetic_chip_energy() -> ChipEnergyConfig:
    return ChipEnergyConfig(
        register_read_pj_per_bit=0.17,
        register_write_pj_per_bit=0.31,
        shared_memory_read_pj_per_bit=0.43,
        shared_memory_write_pj_per_bit=0.67,
        l2_read_pj_per_bit=0.89,
        l2_write_pj_per_bit=1.23,
        dram_read_pj_per_bit=2.47,
        dram_write_pj_per_bit=2.71,
        sfu_pj_per_op=3.19,
        cuda_core_pj_per_op=4.37,
        tensor_core_pj_per_op=5.41,
        static_power_w=6.73,
    )


def _synthetic_config() -> StackedGpuConfig:
    return StackedGpuConfig(
        name="synthetic-reviewer-profile",
        sm_count=13,
        core_frequency_hz=987_654_321.0,
        memory_frequency_hz=345_678_901.0,
        noc_frequency_hz=234_567_891.0,
        tensor_cores_per_sm=3,
        tensor_core_shape=(5, 7, 11),
        fp32_cores_per_sm=17,
        int32_cores_per_sm=19,
        sfu_cores_per_sm=23,
        sm_sub_partitions=5,
        shared_memory_throughput_bytes_per_cycle=37.5,
        shared_memory_capacity_bytes=654_321,
        register_capacity_per_sm_bytes=765_432,
        warp_schedulers_per_sm=7,
        l2_capacity_bytes=98_765_432,
        l1_noc_bytes_per_cycle=43.25,
        dram=_synthetic_dram(),
        tensor_instruction_shapes={
            0.5: (13, 17, 19),
            1.0: (11, 13, 17),
            2.0: (7, 11, 13),
        },
        chip_energy_config=_synthetic_chip_energy(),
        ddr_max_utilization=0.71,
        l2_max_utilization=0.72,
        l1_max_utilization=0.73,
        compute_max_utilization=0.74,
        support_wgmma=True,
        support_utcmma=False,
        use_tensor_core_resource_model=True,
        apply_dram_wave_quantization=True,
    )


def test_custom_arch_module_does_not_load_reference_provider() -> None:
    script = (
        "import sys; "
        f"sys.path[:0] = [{str(DEEPSTACK_SRC)!r}, {str(TILESIGHT_SRC)!r}]; "
        "import mosaic.arch.custom_profile; "
        "assert 'mosaic.arch._reference_model' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True)


def test_custom_arch_reflects_every_explicit_resource_group() -> None:
    config = _synthetic_config()
    arch = make_custom_stacked_gpu(config)
    dram = config.dram

    assert isinstance(arch, ConfigurableStackedGpu)
    assert arch.config is config
    assert arch.core == config.name

    # The local SM topology is derived only from the caller's SM count.
    assert arch.layer1_noc == "xbar"
    assert arch.layer1_noc_size == config.sm_count
    assert arch.sm_count == config.sm_count

    assert arch.core_freq == config.core_frequency_hz
    assert arch.memory_freq == config.memory_frequency_hz
    assert arch.noc_freq == config.noc_frequency_hz
    assert arch.base_freq == config.core_frequency_hz
    assert arch.max_freq == config.core_frequency_hz
    assert arch.chip_energy_config is config.chip_energy_config

    expected_peak = (
        dram.connected_layers
        * dram.channels_per_connected_layer
        * dram.bytes_per_channel_transfer
        * dram.transfers_per_memory_clock
        * config.memory_frequency_hz
    )
    assert arch.dram_layers_per_cluster == dram.total_layers
    assert arch.dram_active_layers == dram.connected_layers
    assert arch.ddr_peak_bandwidth == pytest.approx(expected_peak)
    assert arch.ddr_bandwidth == pytest.approx(expected_peak)
    assert arch.ddr_capacity == (
        dram.total_layers * dram.capacity_per_layer_bytes
    )
    assert arch.ddr_transaction_size == dram.transaction_bytes
    assert arch.ddr_stack == dram.channels_per_connected_layer
    assert arch.ddr_wave_bytes == (
        dram.transaction_bytes * dram.channels_per_connected_layer
    )
    assert arch.l2_bandwidth == pytest.approx(
        expected_peak * dram.l2_bandwidth_multiplier
    )

    assert arch.l2_capacity == config.l2_capacity_bytes
    assert (
        arch.configurable_smem_capacity
        == config.shared_memory_capacity_bytes
    )
    assert (
        arch.register_capacity_per_sm
        == config.register_capacity_per_sm_bytes
    )
    assert arch.l1_smem_throughput_per_cycle == (
        config.shared_memory_throughput_bytes_per_cycle
    )
    assert arch.layer1_noc_single_direction_bw == pytest.approx(
        config.l1_noc_bytes_per_cycle * config.noc_frequency_hz
    )

    tensor_flops_per_instruction = math.prod(config.tensor_core_shape) * 2
    expected_tensor_flops = (
        config.sm_count
        * config.core_frequency_hz
        * config.tensor_cores_per_sm
        * tensor_flops_per_instruction
    )
    assert arch.tensor_core_flops == tensor_flops_per_instruction
    assert arch.fp16_tensor_flops == pytest.approx(expected_tensor_flops)
    assert arch.fp32_tensor_flops == pytest.approx(
        config.sm_count
        * config.core_frequency_hz
        * config.fp32_cores_per_sm
        * 2
    )
    assert arch.int32_cuda_core_flops == pytest.approx(
        config.sm_count
        * config.core_frequency_hz
        * config.int32_cores_per_sm
        * 2
    )
    assert arch.sfu_flops == pytest.approx(
        config.sm_count
        * config.core_frequency_hz
        * config.sfu_cores_per_sm
    )
    assert arch.smem_bandwidth == pytest.approx(
        config.sm_count
        * config.core_frequency_hz
        * config.shared_memory_throughput_bytes_per_cycle
    )
    assert arch.register_bandwidth == pytest.approx(
        config.sm_count
        * config.core_frequency_hz
        * config.sm_sub_partitions
        * 32
        * 4
    )

    assert arch.ddr_max_util == config.ddr_max_utilization
    assert arch.l2_max_util == config.l2_max_utilization
    assert arch.l1_max_util == config.l1_max_utilization
    assert arch.compute_max_util == config.compute_max_utilization
    assert arch.support_wgmma is True
    assert arch.support_utcmma is False
    assert uses_gpu_resource_model(arch)
    assert uses_dram_wave_quantization(arch)
    for byte_width, shape in config.tensor_instruction_shapes.items():
        assert arch.get_tensor_core_minimum_ptx(byte_width) == shape


def test_dram_layer_update_recomputes_capacity_bandwidth_and_efficiency() -> None:
    config = _synthetic_config()
    arch = make_custom_stacked_gpu(config)
    dram = config.dram

    total_layers = 12
    connected_layers = 11
    arch.update_ddr(total_layers, connected_layers)

    expected_peak = (
        connected_layers
        * dram.channels_per_connected_layer
        * dram.bytes_per_channel_transfer
        * dram.transfers_per_memory_clock
        * config.memory_frequency_hz
    )
    interpolation = (connected_layers - total_layers / 2) / (
        total_layers / 2
    )
    expected_efficiency = 1.0 + interpolation * (
        dram.fully_connected_efficiency - 1.0
    )
    assert arch.dram_layers_per_cluster == total_layers
    assert arch.dram_active_layers == connected_layers
    assert arch.ddr_peak_bandwidth == pytest.approx(expected_peak)
    assert arch.ddr_bandwidth == pytest.approx(
        expected_peak * expected_efficiency
    )
    assert arch.ddr_capacity == (
        total_layers * dram.capacity_per_layer_bytes
    )
    assert arch.l2_bandwidth == pytest.approx(
        expected_peak * dram.l2_bandwidth_multiplier
    )


@pytest.mark.parametrize(
    ("total_layers", "connected_layers", "expected"),
    [
        (10, 1, 1.0),
        (10, 5, 1.0),
        (10, 7, 0.748),
        (10, 10, 0.37),
    ],
)
def test_connectivity_efficiency_uses_caller_supplied_endpoint(
    total_layers: int,
    connected_layers: int,
    expected: float,
) -> None:
    assert dram_connectivity_efficiency(
        total_layers,
        connected_layers,
        fully_connected_efficiency=0.37,
    ) == pytest.approx(expected)


def test_peak_bandwidth_supports_a_caller_selected_connected_layer_count() -> None:
    dram = _synthetic_dram()
    memory_frequency_hz = 456_789_123.0
    connected_layers = 6

    expected = (
        connected_layers
        * dram.channels_per_connected_layer
        * dram.bytes_per_channel_transfer
        * dram.transfers_per_memory_clock
        * memory_frequency_hz
    )
    assert dram.peak_bandwidth_bytes_per_s(
        memory_frequency_hz,
        connected_layers=connected_layers,
    ) == pytest.approx(expected)


def test_direct_peak_bandwidth_is_an_explicit_alternative() -> None:
    per_connected_layer_peak = 246_813_579.0
    dram = replace(
        _synthetic_dram(),
        bytes_per_channel_transfer=None,
        transfers_per_memory_clock=None,
        direct_peak_bandwidth_per_connected_layer_bytes_per_s=(
            per_connected_layer_peak
        ),
    )

    assert dram.peak_bandwidth_bytes_per_s(
        135_792_468.0,
        connected_layers=6,
    ) == pytest.approx(6 * per_connected_layer_peak)

    arch = make_custom_stacked_gpu(
        replace(_synthetic_config(), dram=dram)
    )
    arch.update_ddr(12, 11)
    expected_peak = 11 * per_connected_layer_peak
    expected_efficiency = dram_connectivity_efficiency(
        12,
        11,
        fully_connected_efficiency=dram.resolved_fully_connected_efficiency,
    )
    assert arch.ddr_peak_bandwidth == pytest.approx(expected_peak)
    assert arch.ddr_bandwidth == pytest.approx(
        expected_peak * expected_efficiency
    )


def test_bank_timing_derives_efficiency_and_latency_from_caller_inputs() -> None:
    timing = DramTimingConfig(
        row_bytes=4_070,
        sector_bytes=74,
        sector_cycles=3,
        recharge_cycles=41,
        round_trip_latency_cycles=137.5,
        data_rate_hz=321_987_654.0,
        latency_clock_hz=654_321_987.0,
    )
    dram = replace(
        _synthetic_dram(),
        fully_connected_efficiency=None,
        bank_timing=timing,
    )
    arch = make_custom_stacked_gpu(
        replace(_synthetic_config(), dram=dram)
    )

    assert timing.sectors_per_row == 55
    assert timing.row_read_cycles == 165
    assert timing.full_row_cycles == 206
    assert timing.fully_connected_efficiency == pytest.approx(165 / 206)
    assert timing.round_trip_latency_seconds == pytest.approx(
        137.5 / 654_321_987.0
    )
    assert dram.resolved_fully_connected_efficiency == pytest.approx(165 / 206)
    assert arch.dram_row_bytes == timing.row_bytes
    assert arch.dram_sector_bytes == timing.sector_bytes
    assert arch.dram_sector_cycles == timing.sector_cycles
    assert arch.dram_recharge_cycles == timing.recharge_cycles
    assert (
        arch.dram_round_trip_latency_cycles
        == timing.round_trip_latency_cycles
    )
    assert arch.dram_data_rate_hz == timing.data_rate_hz
    assert arch.dram_latency_clock_hz == timing.latency_clock_hz
    assert arch.dram_data_rate_hz != arch.dram_latency_clock_hz
    bank_spec = DramBankSpec.from_arch(arch)
    assert bank_spec.row_bytes == timing.row_bytes
    assert bank_spec.sector_bytes == timing.sector_bytes
    assert bank_spec.sectors_per_row == timing.sectors_per_row
    assert bank_spec.round_trip_latency_cycles == (
        timing.round_trip_latency_cycles
    )

    arch.update_ddr(10, 10)
    assert arch.ddr_bandwidth == pytest.approx(
        arch.ddr_peak_bandwidth * timing.fully_connected_efficiency
    )


@pytest.mark.parametrize(
    "changes",
    [
        {
            "direct_peak_bandwidth_per_connected_layer_bytes_per_s": (
                246_813_579.0
            ),
        },
        {
            "bytes_per_channel_transfer": None,
            "transfers_per_memory_clock": None,
        },
        {
            "bank_timing": DramTimingConfig(
                row_bytes=4_070,
                sector_bytes=74,
                sector_cycles=3,
                recharge_cycles=41,
                round_trip_latency_cycles=137.5,
                data_rate_hz=321_987_654.0,
                latency_clock_hz=654_321_987.0,
            ),
        },
        {"fully_connected_efficiency": None},
    ],
)
def test_dram_config_requires_one_bandwidth_and_timing_form(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        replace(_synthetic_dram(), **changes)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_bytes", 0),
        ("sector_bytes", 64),
        ("sector_cycles", 0),
        ("recharge_cycles", -1),
        ("round_trip_latency_cycles", float("nan")),
        ("data_rate_hz", 0.0),
        ("latency_clock_hz", float("inf")),
    ],
)
def test_bank_timing_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    timing = DramTimingConfig(
        row_bytes=4_070,
        sector_bytes=74,
        sector_cycles=3,
        recharge_cycles=41,
        round_trip_latency_cycles=137.5,
        data_rate_hz=321_987_654.0,
        latency_clock_hz=654_321_987.0,
    )
    with pytest.raises((TypeError, ValueError)):
        replace(timing, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_layers", 0),
        ("total_layers", True),
        ("connected_layers", 11),
        ("channels_per_connected_layer", 2.5),
        ("bytes_per_channel_transfer", 0.0),
        ("transfers_per_memory_clock", float("nan")),
        ("capacity_per_layer_bytes", -1),
        ("transaction_bytes", 0),
        ("fully_connected_efficiency", 1.01),
        ("uncached_max_utilization", -0.01),
        ("l2_bandwidth_multiplier", float("inf")),
    ],
)
def test_dram_config_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_synthetic_dram(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "   "),
        ("sm_count", 0),
        ("sm_count", 13.5),
        ("core_frequency_hz", float("nan")),
        ("memory_frequency_hz", 0.0),
        ("noc_frequency_hz", float("inf")),
        ("tensor_core_shape", [5, 7, 11]),
        ("tensor_core_shape", (5, 0, 11)),
        ("tensor_instruction_shapes", {}),
        ("tensor_instruction_shapes", {1.0: (5, 7)}),
        ("ddr_max_utilization", -0.01),
        ("l2_max_utilization", 1.01),
        ("support_wgmma", 1),
        ("support_utcmma", "false"),
    ],
)
def test_stacked_gpu_config_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_synthetic_config(), **{field: value})


@pytest.mark.parametrize(
    ("total_layers", "active_layers"),
    [(0, 1), (10, 0), (10, 11), (10, True)],
)
def test_arch_rejects_invalid_layer_updates(
    total_layers: object,
    active_layers: object,
) -> None:
    arch = make_custom_stacked_gpu(_synthetic_config())
    with pytest.raises((TypeError, ValueError)):
        arch.update_ddr(total_layers, active_layers)


def test_arch_rejects_unknown_tensor_element_width() -> None:
    arch = make_custom_stacked_gpu(_synthetic_config())
    with pytest.raises(ValueError, match="unsupported tensor element width"):
        arch.get_tensor_core_minimum_ptx(3.0)
