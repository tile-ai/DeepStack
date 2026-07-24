from __future__ import annotations

import csv
import importlib
import struct
import sys
from pathlib import Path

import pytest

from ae.paths import ROOT, activate_vendored_sources


activate_vendored_sources()

sys.path.insert(0, str(ROOT / "tools"))
import audit_binary_hardening as binary_audit  # noqa: E402


BINARY_PATTERNS = (
    "src/deepstack/mosaic/arch/_reference_model*.so",
    "src/deepstack/mosaic/cost/_capacity*.so",
    "src/deepstack/mosaic/noc/_model_support*.so",
    "src/tilesight/tilesight/distributed/noc/_model_support*.so",
)

FORBIDDEN_BINARY_MARKERS = binary_audit.COMMON_FORBIDDEN_MARKERS

LEGACY_AREA_API = (
    "AreaModel",
    "get_register_area",
    "get_tensor_core_area",
    "get_smem_area",
    "get_cuda_core_area",
    "get_sfu_core_area",
    "get_sm_area",
    "get_cluster_area",
    "get_cluster_noc_area",
    "get_mem_controller_area",
    "get_dram_peripheral_area",
    "get_l1_noc_area",
    "get_thermal_resistance",
    "get_die_area",
)

REFERENCE_ARCH_METHODS = (
    "with_design_point",
    "with_sm_count",
    "with_dram",
    "with_cluster_width",
    "with_thermal_scale",
    "with_littles_law",
    "row_reduce_time",
)

PROVIDER_CALLABLES = {
    "mosaic.arch._reference_model": (
        "is_reference_arch",
        "model_support_api_version",
        *(f"q{index:02d}" for index in (*range(1, 10), *range(11, 14))),
    ),
    "mosaic.cost._capacity": ("p00",),
    "mosaic.noc._model_support": (
        "p00",
        "p01",
        "p93",
        "p94",
        "p95",
        "p96",
        "p97",
        "p98",
    ),
    "tilesight.distributed.noc._model_support": (
        "p00",
        "p01",
        "p98",
    ),
}


def _model_support_binaries() -> list[Path]:
    binaries: list[Path] = []
    for pattern in BINARY_PATTERNS:
        matches = sorted(ROOT.glob(pattern))
        assert len(matches) == 1, f"expected one binary matching {pattern!r}"
        binaries.extend(matches)
    return binaries


def test_model_support_binaries_omit_build_and_calibration_markers() -> None:
    binaries = _model_support_binaries()
    assert len(binaries) == 4
    for binary in binaries:
        payload = binary.read_bytes()
        for marker in FORBIDDEN_BINARY_MARKERS:
            assert marker not in payload, (
                f"{binary.relative_to(ROOT)} contains forbidden marker "
                f"{marker!r}"
            )


def test_legacy_area_extension_is_absent() -> None:
    legacy = sorted((ROOT / "src/deepstack/mosaic/cost").glob("area*.so"))
    assert legacy == []


def test_capacity_native_surface_is_minimal_and_non_introspectable() -> None:
    provider = importlib.import_module("mosaic.cost._capacity")
    public = {name for name in dir(provider) if not name.startswith("_")}

    assert public == {"p00"}
    assert all(not hasattr(provider, name) for name in LEGACY_AREA_API)

    function = provider.p00
    assert function.__doc__ is None
    assert not hasattr(function, "__code__")
    assert not hasattr(function, "__annotations__")
    assert not hasattr(function, "__dict__")


def test_reference_provider_omits_legacy_area_feasibility_oracle() -> None:
    provider = importlib.import_module("mosaic.arch._reference_model")
    assert not hasattr(provider, "q10")
    assert not hasattr(provider, "q14")


def test_capacity_elf_hardening_and_adjacent_license() -> None:
    matches = sorted(
        (ROOT / "src/deepstack/mosaic/cost").glob("_capacity*.so")
    )
    assert len(matches) == 1
    markers = (
        binary_audit.COMMON_FORBIDDEN_MARKERS
        + binary_audit.CAPACITY_FORBIDDEN_MARKERS
    )
    audit = binary_audit.audit_binary(
        matches[0],
        forbidden_markers=markers,
        require_cet=True,
        require_full_relro=True,
        max_size_bytes=binary_audit.MAX_CAPACITY_BINARY_BYTES,
    )

    assert audit.failures == ()
    assert matches[0].stat().st_size <= binary_audit.MAX_CAPACITY_BINARY_BYTES
    assert audit.dynamic_symbols == (
        binary_audit.expected_init_symbol(matches[0]),
    )


def test_capacity_binary_license_mapping_is_explicit() -> None:
    license_text = (
        ROOT
        / "LICENSES"
        / "LicenseRef-DeepStack-AE-Binary-1.0.txt"
    ).read_text(encoding="utf-8")

    assert "src/deepstack/mosaic/cost/_capacity*.so" in license_text
    assert "src/deepstack/mosaic/cost/area*.so" not in license_text


def test_capacity_release_tree_excludes_private_build_material() -> None:
    forbidden_names = {
        ".capacity_mask_seed",
        "_capacity.c",
        "_capacity.pxd",
        "_capacity.pyx",
        "build_area_capacity.py",
        "build_capacity_lut.py",
        "capacity_build_report.json",
        "capacity_collision_report.json",
        "capacity_coverage_report.json",
        "capacity_final.c",
        "capacity_lut.csv",
        "capacity_lut.json",
    }
    leaked = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.name in forbidden_names:
            leaked.append(relative)
    assert leaked == []


def test_capacity_lut_miss_has_reviewer_friendly_facade_error() -> None:
    from mosaic.arch.stacked_gpu_wgmma import stacked_gpu_wgmma
    from mosaic.cost.capacity import max_sm_count
    from mosaic.noc.noc_config_set import torus_mesh_switch_1

    unsupported = stacked_gpu_wgmma().with_design_point(
        sm_count=1,
        total_layers=17,
        active_layers=17,
        smem_capacity=256 * 1024,
        l1_throughput_bpc=256,
    )
    with pytest.raises(
        ValueError,
        match=(
            "outside the bundled AE capacity LUT; "
            "contact the artifact authors"
        ),
    ) as exc_info:
        max_sm_count(unsupported, torus_mesh_switch_1())

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
    assert "unsupported configuration" not in str(exc_info.value)


def test_capacity_lut_covers_public_ae_manifests_and_design_spaces() -> None:
    from ae import fig18_20, table4
    from mosaic.arch.h100_scaled import H100_SCALED
    from mosaic.arch.h200_scaled import H200_SCALED
    from mosaic.arch.stacked_gpu_base import stacked_gpu_base
    from mosaic.arch.stacked_gpu_cluster import (
        stacked_gpu_cluster_8,
        stacked_gpu_cluster_16,
        stacked_gpu_cluster_32,
    )
    from mosaic.arch.stacked_gpu_high_l1 import stacked_gpu_high_l1
    from mosaic.arch.stacked_gpu_high_l2 import stacked_gpu_high_l2
    from mosaic.arch.stacked_gpu_high_noc import stacked_gpu_high_noc
    from mosaic.arch.stacked_gpu_large_matrix import stacked_gpu_large_matrix
    from mosaic.arch.stacked_gpu_large_vector import stacked_gpu_large_vector
    from mosaic.arch.stacked_gpu_low_noc import stacked_gpu_low_noc
    from mosaic.arch.stacked_gpu_reduced_sm import stacked_gpu_reduced_sm
    from mosaic.arch.stacked_gpu_wgmma import stacked_gpu_wgmma
    from mosaic.cost.capacity import max_sm_count
    from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
        find_max_sm_count as find_dram_capacity,
        generate_dram_layer_configs,
    )
    from mosaic.dse_space.case_study_latency_vs_bandwidth.lat_bw_config import (
        find_max_sm_count as find_latency_bw_capacity,
        generate_latency_bw_configs,
        make_noc_for_config as make_latency_bw_noc,
    )
    from mosaic.dse_space.case_study_which_layer_noc_matter.noc_bw_config import (
        find_max_sm_count as find_layer_bw_capacity,
        generate_noc_bw_configs,
        make_noc_for_config as make_layer_bw_noc,
    )
    from mosaic.noc.noc_config_set import torus_mesh_switch_1

    dram_configs = generate_dram_layer_configs()
    latency_bw_configs = generate_latency_bw_configs()
    layer_bw_configs = generate_noc_bw_configs()
    assert len(dram_configs) == 193
    assert len(latency_bw_configs) == 50
    assert len(layer_bw_configs) == 42

    capacities = [
        find_dram_capacity(*configuration)
        for configuration in dram_configs
    ]
    capacities.extend(
        find_latency_bw_capacity(make_latency_bw_noc(*configuration))
        for configuration in latency_bw_configs
    )
    capacities.extend(
        find_layer_bw_capacity(make_layer_bw_noc(*configuration))
        for configuration in layer_bw_configs
    )

    with fig18_20.DEFAULT_MANIFEST.open(
        newline="", encoding="utf-8"
    ) as handle:
        fig18_rows = list(csv.DictReader(handle))
    assert len(fig18_rows) == 114
    for row in fig18_rows:
        expected = int(row["sm_count"])
        actual = find_dram_capacity(
            int(row["dram_total_layers"]),
            int(row["dram_active_layers"]),
            int(round(float(row["smem_capacity_KiB"]))) * 1024,
            int(row["l1_throughput_Bpc"]),
        )
        assert actual == expected, row["candidate_id"]
        capacities.append(actual)

    with (ROOT / "data/fig21/candidates_march2026.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        fig21_rows = list(csv.DictReader(handle))
    fig21_noc_configs = {
        (
            row["baseline_noc"],
            float(row["latency_multiplier"]),
            float(row["bw_multiplier"]),
        )
        for row in fig21_rows
    }
    assert len(fig21_rows) == 180
    assert len(fig21_noc_configs) == 30
    capacities.extend(
        find_latency_bw_capacity(make_latency_bw_noc(*configuration))
        for configuration in sorted(fig21_noc_configs)
    )

    with table4.DEFAULT_MANIFEST.open(
        newline="", encoding="utf-8"
    ) as handle:
        table4_rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["step"]) >= 6
        ]
    assert len(table4_rows) == 4
    for row in table4_rows:
        arch, _ = table4._build_hardware(row)
        capacities.append(int(arch.sm_count))

    reference_factories = (
        stacked_gpu_base,
        stacked_gpu_wgmma,
        stacked_gpu_large_matrix,
        stacked_gpu_large_vector,
        stacked_gpu_high_l1,
        stacked_gpu_high_l2,
        stacked_gpu_high_noc,
        stacked_gpu_low_noc,
        stacked_gpu_reduced_sm,
        stacked_gpu_cluster_8,
        stacked_gpu_cluster_16,
        stacked_gpu_cluster_32,
        H100_SCALED,
        H200_SCALED,
    )
    baseline_noc = torus_mesh_switch_1()
    for factory in reference_factories:
        default_arch = factory()
        first_dram_variant = default_arch.with_design_point(
            sm_count=1,
            total_layers=4,
            active_layers=4,
            smem_capacity=256 * 1024,
            l1_throughput_bpc=256,
        )
        second_dram_variant = default_arch.with_design_point(
            sm_count=1,
            total_layers=8,
            active_layers=4,
            smem_capacity=512 * 1024,
            l1_throughput_bpc=1024,
        )
        capacities.extend(
            (
                max_sm_count(first_dram_variant, baseline_noc),
                max_sm_count(second_dram_variant, baseline_noc),
            )
        )

    assert capacities
    assert all(
        isinstance(capacity, int) and 1 <= capacity <= 128
        for capacity in capacities
    )


def test_private_number_scanner_detects_ieee_fingerprints() -> None:
    first = 123.456789
    second = 9876.125
    payload = (
        b"prefix"
        + struct.pack("<d", first)
        + b"middle"
        + struct.pack(">f", second)
        + b"suffix"
    )

    matches = binary_audit.find_plain_number_matches(
        payload,
        (first, second),
    )
    found = {
        (match.value, match.representation)
        for match in matches
    }
    assert (first, "float64-le") in found
    assert (second, "float32-be") in found


def test_binary_provider_callables_omit_python_introspection_metadata() -> None:
    reference_module = importlib.import_module("mosaic.arch._reference_model")
    for name in REFERENCE_ARCH_METHODS:
        method = getattr(reference_module.ReferenceArch, name)
        assert method.__doc__ is None
        assert not hasattr(method, "__code__")
        assert not hasattr(method, "__annotations__")
        assert not hasattr(method, "__dict__")

    for module_name, names in PROVIDER_CALLABLES.items():
        module = importlib.import_module(module_name)
        for name in names:
            function = getattr(module, name)
            assert function.__doc__ is None
            assert not hasattr(function, "__code__")
            assert not hasattr(function, "__annotations__")
            assert not hasattr(function, "__dict__")
