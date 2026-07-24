import copy

import pytest

from mosaic.arch.stacked_gpu_wgmma import stacked_gpu_wgmma

from bank_oracle.gemm import (
    GemmBankOptions,
    GemmProblem,
    GemmTiling,
    make_layout_set,
    model_gemm_bank,
)
from bank_oracle.layout import BankSwizzle
from bank_oracle.pipeline import (
    model_gemm_pipeline,
    pipeline_cycles,
    search_layouts_and_stages,
)
from bank_oracle.spec import DramBankSpec


@pytest.mark.parametrize(
    "stage, expected",
    [(1, 1384), (2, 873), (3, 946), (4, 1019)],
)
def test_pipeline_overlap_equation(stage, expected):
    result = pipeline_cycles(8, stage, 73, 100)
    assert result.total_cycles == expected
    if stage >= 2:
        assert result.steady_memory_cycles < result.steady_compute_cycles


def test_pipeline_depth_is_clamped_for_gridk_smaller_than_stage():
    stage1 = pipeline_cycles(1, 1, 73, 100, 17)
    stage4 = pipeline_cycles(1, 4, 73, 100, 17)
    assert stage1.total_cycles == stage4.total_cycles == 190


def test_littles_law_buffer_has_no_phantom_stage_when_gridk_is_one():
    problem = GemmProblem(32, 32, 32)
    spec = DramBankSpec()
    options = GemmBankOptions(
        max_spatial_samples=None,
        max_k_samples=None,
        apply_littles_law=True,
    )
    observed = []
    for stage in (1, 2, 3, 4):
        tiling = GemmTiling(32, 32, 32, stage=stage, ctas_per_wave=8)
        layouts = make_layout_set(problem, tiling, spec)
        observed.append(
            model_gemm_bank(problem, tiling, layouts, spec, options).total_load_cycles
        )
    assert observed == pytest.approx([observed[0]] * 4)


def test_stage_one_does_not_report_serial_read_as_hidden():
    problem = GemmProblem(32, 32, 64)
    tiling = GemmTiling(32, 32, 32, stage=1, ctas_per_wave=8)
    spec = DramBankSpec()
    bank = model_gemm_bank(
        problem,
        tiling,
        make_layout_set(problem, tiling, spec),
        spec,
        GemmBankOptions(
            max_spatial_samples=None,
            max_k_samples=None,
            apply_littles_law=False,
        ),
    )
    result = model_gemm_pipeline(
        bank,
        compute_cycles_per_k=10_000,
        other_memory_cycles_per_k=0,
        store_other_memory_cycles=0,
    )
    assert result.read_hidden_fraction == 0.0


def test_short_k_has_no_steady_state_read_hiding():
    problem = GemmProblem(32, 32, 32)
    tiling = GemmTiling(32, 32, 32, stage=4, ctas_per_wave=8)
    spec = DramBankSpec()
    bank = model_gemm_bank(
        problem,
        tiling,
        make_layout_set(problem, tiling, spec),
        spec,
        GemmBankOptions(
            max_spatial_samples=None,
            max_k_samples=None,
            apply_littles_law=False,
        ),
    )
    result = model_gemm_pipeline(
        bank,
        compute_cycles_per_k=10_000,
        other_memory_cycles_per_k=0,
        store_other_memory_cycles=0,
    )
    assert result.read_hidden_fraction == 0.0


def test_bank_ddr_time_is_independent_of_coarse_architecture_penalties():
    problem = GemmProblem(64, 64, 256)
    tiling = GemmTiling(32, 32, 32, stage=2, ctas_per_wave=8)
    spec = DramBankSpec()
    options = GemmBankOptions(
        max_spatial_samples=None,
        max_k_samples=None,
        apply_littles_law=False,
    )
    layouts = make_layout_set(problem, tiling, spec)
    bank = model_gemm_bank(problem, tiling, layouts, spec, options)

    arch_a = stacked_gpu_wgmma()
    arch_b = copy.copy(arch_a)
    arch_b.ddr_wave_bytes = 1024 * 1024 * 1024
    arch_b.ddr_bandwidth = 1.0
    arch_b.ddr_max_util = 0.01
    a = model_gemm_pipeline(bank, arch_a)
    b = model_gemm_pipeline(bank, arch_b)
    assert a.total_cycles == b.total_cycles
    assert a.total_seconds == b.total_seconds


def test_representative_overlap_case_hides_bank_read():
    problem = GemmProblem(256, 256, 8192)
    tiling = GemmTiling(128, 128, 32, stage=2, ctas_per_wave=8)
    spec = DramBankSpec()
    bank_bits = spec.bank_count.bit_length() - 1
    sector_bits = min(
        4,
        bank_bits,
        spec.sectors_per_row.bit_length() - 1,
    )
    options = GemmBankOptions(max_spatial_samples=4, max_k_samples=4)
    layouts = make_layout_set(
        problem,
        tiling,
        spec,
        preset="tile_major",
        swizzle=BankSwizzle(
            kind="xor",
            sector_bits=sector_bits,
            sector_bank_shift=bank_bits - sector_bits,
            row_bits=bank_bits,
        ),
        b_phase=spec.bank_count // 2,
        c_phase=spec.bank_count // 4,
        name="tile_xor",
    )
    bank = model_gemm_bank(problem, tiling, layouts, spec, options)
    result = model_gemm_pipeline(bank, stacked_gpu_wgmma())
    # One of 256 K iterations fills the two-stage pipeline; the rest overlap.
    assert result.read_hidden_fraction == pytest.approx(255 / 256)
    # Keep the bank-only diagnostic visible even when compute hides it.
    assert result.pipeline_attainment >= result.bank_only_attainment
    assert result.total_seconds > 0


def test_stage_layout_search_can_phase_validate_shortlist():
    problem = GemmProblem(64, 64, 256)
    tiling = GemmTiling(32, 32, 32, stage=2, ctas_per_wave=8)
    result = search_layouts_and_stages(
        problem,
        tiling,
        options=GemmBankOptions(
            max_spatial_samples=4,
            max_k_samples=4,
            apply_littles_law=False,
        ),
        stages=(1, 2, 3, 4),
        compute_cycles_per_k=100,
        workers=4,
        backend="thread",
        validation_top_k_per_stage=1,
        validation_phase_samples=64,
    )
    assert result.validation_performed
    assert {row.bank_result.tiling.stage for row in result.validated_results} == {
        1, 2, 3, 4
    }


def test_stage_search_rejects_smem_infeasible_depths():
    arch = copy.copy(stacked_gpu_wgmma())
    arch.configurable_smem_capacity = 5 * 1024
    result = search_layouts_and_stages(
        GemmProblem(64, 64, 256),
        GemmTiling(32, 32, 32, stage=1, ctas_per_wave=8),
        options=GemmBankOptions(
            max_spatial_samples=2,
            max_k_samples=2,
            apply_littles_law=False,
        ),
        arch=arch,
        stages=(1, 2, 3, 4),
        workers=4,
    )
    assert {row.bank_result.tiling.stage for row in result.results} == {1}


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_stage_layout_search_parallel_backends_match_serial(backend):
    problem = GemmProblem(64, 96, 128)
    tiling = GemmTiling(32, 32, 32, stage=2, ctas_per_wave=8)
    options = GemmBankOptions(
        max_spatial_samples=2,
        max_k_samples=2,
        apply_littles_law=False,
        deterministic_seed=17,
    )
    common = dict(
        options=options,
        stages=(1, 2),
        compute_cycles_per_k=100,
        validation_top_k_per_stage=1,
        validation_phase_samples=4,
    )
    serial = search_layouts_and_stages(
        problem,
        tiling,
        workers=2,
        backend="serial",
        **common,
    )
    parallel = search_layouts_and_stages(
        problem,
        tiling,
        workers=2,
        backend=backend,
        **common,
    )

    def fingerprint(result):
        return (
            (
                result.best.bank_result.tiling.stage,
                result.best.bank_result.layouts.name,
            ),
            tuple(
                (
                    row.bank_result.tiling.stage,
                    row.bank_result.layouts.name,
                    row.total_cycles,
                    row.ideal_total_cycles,
                    row.bank_result.actual_read_bytes,
                )
                for row in result.results
            ),
            tuple(
                (
                    row.bank_result.tiling.stage,
                    row.bank_result.layouts.name,
                    row.total_cycles,
                )
                for row in result.validated_results
            ),
        )

    assert fingerprint(parallel) == fingerprint(serial)


def test_stage_search_enumerates_row_panel_and_raster_axes():
    problem = GemmProblem(128, 192, 128)
    tiling = GemmTiling(
        32,
        32,
        32,
        stage=2,
        ctas_per_wave=8,
        row_panel=1,
        raster_axis="legacy",
    )
    result = search_layouts_and_stages(
        problem,
        tiling,
        options=GemmBankOptions(
            max_spatial_samples=2,
            max_k_samples=2,
            apply_littles_law=False,
        ),
        stages=(2,),
        row_panels=(1, 2),
        raster_axes=("along_m", "along_n"),
        compute_cycles_per_k=100,
        workers=2,
        backend="serial",
        validation_top_k_per_stage=1,
        validation_phase_samples=2,
    )
    expected = {
        (2, row_panel, raster_axis)
        for row_panel in (1, 2)
        for raster_axis in ("along_m", "along_n")
    }
    enumerated = {
        (
            row.bank_result.tiling.stage,
            row.bank_result.tiling.row_panel,
            row.bank_result.tiling.raster_axis,
        )
        for row in result.results
    }
    validated = {
        (
            row.bank_result.tiling.stage,
            row.bank_result.tiling.row_panel,
            row.bank_result.tiling.raster_axis,
        )
        for row in result.validated_results
    }
    best_configuration = (
        result.best.bank_result.tiling.stage,
        result.best.bank_result.tiling.row_panel,
        result.best.bank_result.tiling.raster_axis,
    )
    assert enumerated == expected
    assert validated == expected
    assert best_configuration in expected
