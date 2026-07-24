from __future__ import annotations

import importlib
import importlib.machinery
import math
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor

import pytest

from ae.cli import doctor
from ae.doctor import (
    ARCH_PROFILE_API_VERSION,
    ARCH_PROFILE_MODULE,
    ARCH_PROFILE_REQUIRED_CALLABLES,
    check_arch_reference_provider,
)
from ae.paths import activate_vendored_sources


activate_vendored_sources()

from mosaic.arch.h100_scaled import H100_SCALED  # noqa: E402
from mosaic.arch.h200_scaled import H200_SCALED  # noqa: E402
from mosaic.arch.stacked_gpu_base import stacked_gpu_base  # noqa: E402
from mosaic.arch.stacked_gpu_cluster import (  # noqa: E402
    stacked_gpu_cluster_8,
    stacked_gpu_cluster_16,
    stacked_gpu_cluster_32,
)
from mosaic.arch.stacked_gpu_high_l2 import stacked_gpu_high_l2  # noqa: E402
from mosaic.arch.stacked_gpu_high_l1 import stacked_gpu_high_l1  # noqa: E402
from mosaic.arch.stacked_gpu_high_noc import stacked_gpu_high_noc  # noqa: E402
from mosaic.arch.stacked_gpu_large_matrix import (  # noqa: E402
    stacked_gpu_large_matrix,
)
from mosaic.arch.stacked_gpu_large_vector import (  # noqa: E402
    stacked_gpu_large_vector,
)
from mosaic.arch.stacked_gpu_low_noc import stacked_gpu_low_noc  # noqa: E402
from mosaic.arch.stacked_gpu_reduced_sm import stacked_gpu_reduced_sm  # noqa: E402
from mosaic.arch.stacked_gpu_wgmma import stacked_gpu_wgmma  # noqa: E402
from tilesight.arch import (  # noqa: E402
    uses_dram_wave_quantization,
    uses_gpu_resource_model,
)


REFERENCE_ARCH_CASES = (
    (stacked_gpu_base, "stacked_gpu_base", True),
    (stacked_gpu_high_l1, "stacked_gpu_high_l1", True),
    (stacked_gpu_high_l2, "stacked_gpu_high_l2", True),
    (stacked_gpu_high_noc, "stacked_gpu_high_noc", True),
    (stacked_gpu_low_noc, "stacked_gpu_low_noc", True),
    (stacked_gpu_large_vector, "stacked_gpu_large_vector", True),
    (stacked_gpu_large_matrix, "stacked_gpu_large_matrix", True),
    (stacked_gpu_reduced_sm, "stacked_gpu_reduced_sm", True),
    (stacked_gpu_wgmma, "stacked_gpu_wgmma", True),
    (stacked_gpu_cluster_8, "stacked_gpu_cluster_8", True),
    (stacked_gpu_cluster_16, "stacked_gpu_cluster_16", True),
    (stacked_gpu_cluster_32, "stacked_gpu_cluster_32", True),
    (H100_SCALED, "H100", False),
    (H200_SCALED, "H100", False),
)
REFERENCE_ARCH_FACTORIES = tuple(case[0] for case in REFERENCE_ARCH_CASES)

# These primitive calibration fields must remain inside the proprietary
# provider.  Keep aliases here as a regression guard: reintroducing a legacy
# spelling is just as observable as reintroducing the current spelling.
PROTECTED_ARCH_FIELDS = frozenset(
    {
        "_reference_profile_name",
        "base_freq",
        "clock_frequency",
        "core_freq",
        "dram_latency_clock_hz",
        "dram_round_trip_latency_cycles",
        "fp16_cores_per_sm",
        "fp32_cores_per_sm",
        "fp64_cores_per_sm",
        "freq",
        "int32_cores_per_sm",
        "l1_smem_throughput_per_cycle",
        "max_freq",
        "memory_clock_hz",
        "memory_freq",
        "noc_clock_hz",
        "noc_freq",
        "sfu_cores_per_sm",
        "sfu_units_per_sm",
        "sm_sub_partitions",
        "tensor_core_flops",
        "tensor_core_flops_per_cycle",
        "tensor_core_shape",
        "tensor_cores_per_sm",
        "warp_schedulers_per_sm",
    }
)

POSITIVE_AGGREGATE_FIELDS = (
    "sm_count",
    "fp16_tensor_flops",
    "fp32_tensor_flops",
    "fp8_tensor_flops",
    "fp16_cuda_core_flops",
    "fp32_cuda_core_flops",
    "sfu_flops",
    "ddr_bandwidth",
    "ddr_capacity",
    "ddr_wave_bytes",
    "ddr_transaction_size",
    "l2_bandwidth",
    "l2_capacity",
    "smem_bandwidth",
    "register_bandwidth",
    "layer1_noc_single_direction_bw",
    "configurable_smem_capacity",
    "register_capacity_per_sm",
    "max_blocks_per_sm",
)

PUBLIC_CAPABILITY_FLAGS = (
    "apply_dram_wave_quantization",
    "support_utcmma",
    "support_wgmma",
    "use_tensor_core_resource_model",
)


def _exercise_arch_provider_in_worker() -> tuple[int, bool, bool, bool]:
    module = importlib.import_module("mosaic.arch._reference_model")
    module_path = str(module.__file__)
    is_extension = any(
        module_path.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    has_required_api = all(
        callable(getattr(module, name, None))
        for name in (
            "q01",
            "q03",
            "q05",
            "q07",
            "q12",
            "q13",
        )
    )
    arch = stacked_gpu_base().with_design_point(sm_count=8)
    from mosaic.cost.capacity import max_sm_count
    from mosaic.noc.noc_config_set import torus_mesh_switch_1

    capacity = max_sm_count(arch, torus_mesh_switch_1())
    area_pipeline_works = (
        isinstance(arch, module.ReferenceArch)
        and isinstance(capacity, int)
        and capacity > 0
    )
    return (
        module.model_support_api_version(),
        is_extension,
        has_required_api,
        area_pipeline_works,
    )


def test_arch_reference_provider_is_bundled_extension_with_expected_api() -> None:
    failures: list[str] = []
    path = check_arch_reference_provider(failures)

    assert failures == []
    assert path is not None
    assert path.parent.name == "arch"
    assert any(
        path.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

    module = importlib.import_module(ARCH_PROFILE_MODULE)
    assert module.model_support_api_version() == ARCH_PROFILE_API_VERSION
    assert isinstance(module.ReferenceArch, type)
    assert all(
        callable(getattr(module, name, None))
        for name in ARCH_PROFILE_REQUIRED_CALLABLES
    )


def test_arch_reference_provider_process_pool_import() -> None:
    parent = _exercise_arch_provider_in_worker()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        child = executor.submit(_exercise_arch_provider_in_worker).result(
            timeout=30
        )

    assert child == parent == (ARCH_PROFILE_API_VERSION, True, True, True)


@pytest.mark.parametrize(
    ("factory", "public_name", "wave_quantization"),
    REFERENCE_ARCH_CASES,
)
def test_reference_arch_exposes_public_capabilities(
    factory,
    public_name: str,
    wave_quantization: bool,
) -> None:
    arch = factory()
    provider = importlib.import_module(ARCH_PROFILE_MODULE)

    assert isinstance(arch, provider.ReferenceArch)
    assert arch.core == public_name
    assert uses_gpu_resource_model(arch)
    assert uses_dram_wave_quantization(arch) is wave_quantization
    if wave_quantization:
        assert float(arch.ddr_peak_bandwidth) > 0.0
    for field in POSITIVE_AGGREGATE_FIELDS:
        value = getattr(arch, field)
        assert not isinstance(value, bool), field
        assert math.isfinite(float(value)), field
        assert float(value) > 0.0, field
    for field in PUBLIC_CAPABILITY_FLAGS:
        assert isinstance(getattr(arch, field), bool), field


@pytest.mark.parametrize("factory", REFERENCE_ARCH_FACTORIES)
def test_reference_arch_hides_primitive_calibration_state(factory) -> None:
    arch = factory()

    assert not hasattr(arch, "__dict__")
    with pytest.raises(TypeError):
        vars(arch)

    visible_names = set(dir(arch))
    assert visible_names.isdisjoint(PROTECTED_ARCH_FIELDS)
    for field in PROTECTED_ARCH_FIELDS:
        with pytest.raises(AttributeError):
            getattr(arch, field)

    rendered = repr(arch)
    assert all(field not in rendered for field in PROTECTED_ARCH_FIELDS)

    with pytest.raises(TypeError) as exc_info:
        pickle.dumps(arch)
    assert all(
        field not in str(exc_info.value)
        for field in PROTECTED_ARCH_FIELDS
    )


@pytest.mark.parametrize(
    ("function_name", "args"),
    [
        ("q01", ("standard",)),
        ("q02", ("h100_scaled",)),
        ("q03", ()),
        ("q05", (4, 2)),
        ("q06", (8,)),
        ("q07", (4, 2)),
        ("q08", ()),
    ],
)
def test_native_provider_mutators_reject_python_attribute_sinks(
    function_name: str,
    args: tuple[object, ...],
) -> None:
    class AttributeSink:
        pass

    provider = importlib.import_module(ARCH_PROFILE_MODULE)
    with pytest.raises(TypeError, match="ReferenceArch"):
        getattr(provider, function_name)(AttributeSink(), *args)


def test_descriptive_native_entry_points_are_not_exported() -> None:
    provider = importlib.import_module(ARCH_PROFILE_MODULE)
    legacy_names = {
        "apply_arch_profile",
        "apply_cluster_width",
        "apply_reference_littles_law",
        "apply_scaled_baseline_profile",
        "configure_dram_dse_arch",
        "configure_fig13_switch",
        "configure_fig13_torus",
        "reference_area_fits",
        "reference_power_wall",
        "reference_thermal_frequency_scale",
        "refresh_arch_derived",
        "tensor_instruction_shape",
        "update_arch_dram",
    }
    assert set(dir(provider)).isdisjoint(legacy_names)


def test_reference_thermal_resistance_is_source_visible_and_stable() -> None:
    from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
        reference_thermal_resistance,
    )

    assert [
        reference_thermal_resistance(layers)
        for layers in (1, 4, 16)
    ] == pytest.approx([0.57, 0.60, 0.72], rel=0.0, abs=1.0e-15)
    with pytest.raises(TypeError, match="must be an integer"):
        reference_thermal_resistance(True)
    with pytest.raises(ValueError, match="must be positive"):
        reference_thermal_resistance(0)


def test_doctor_reports_provider_status_without_calibration_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert doctor() == 0
    output = capsys.readouterr()

    assert "arch_reference_provider=binary" in output.out
    assert f"api={ARCH_PROFILE_API_VERSION}" in output.out
    for sensitive_detail in (
        "clock_frequency",
        "dram_latency",
        "hop_latency",
        "link_bandwidth",
        "thermal_resistance",
    ):
        assert sensitive_detail not in output.out
    assert output.err == ""
