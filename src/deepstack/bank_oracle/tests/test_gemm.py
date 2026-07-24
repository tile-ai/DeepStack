from dataclasses import replace

import numpy as np
import pytest

from bank_oracle.gemm import (
    GemmBankOptions,
    GemmProblem,
    GemmTiling,
    _gemm_model_work_units,
    make_layout_set,
    model_gemm_bank,
    search_gemm_layouts,
    _stable_thin,
    default_layout_candidates,
)
from bank_oracle.layout import BankSwizzle
from bank_oracle.parallel import resolve_backend
from bank_oracle.spec import DramBankSpec


def _exact_options(**kwargs):
    defaults = dict(
        max_spatial_samples=None,
        max_k_samples=None,
        apply_littles_law=False,
        coalesce_scope="wave",
    )
    defaults.update(kwargs)
    return GemmBankOptions(**defaults)


def _xor_swizzle(spec, *, sector_bits=4, include_rows=False):
    bank_bits = spec.bank_count.bit_length() - 1
    width = min(
        sector_bits,
        bank_bits,
        spec.sectors_per_row.bit_length() - 1,
    )
    return BankSwizzle(
        kind="xor",
        sector_bits=width,
        sector_bank_shift=bank_bits - width,
        row_bits=bank_bits if include_rows else 0,
    )


def test_single_tile_a_b_conflict_and_static_phase():
    problem = GemmProblem(32, 32, 32)
    tiling = GemmTiling(32, 32, 32, stage=2, ctas_per_wave=8)
    spec = DramBankSpec()
    same_bank = make_layout_set(problem, tiling, spec, name="same")
    split_bank = make_layout_set(problem, tiling, spec, b_phase=1, name="split")
    conflict = model_gemm_bank(problem, tiling, same_bank, spec, _exact_options())
    balanced = model_gemm_bank(problem, tiling, split_bank, spec, _exact_options())
    operand_sectors = (
        problem.m * problem.k * problem.a_dtype_bytes
        // spec.sector_bytes
    )
    one_operand_cycles = (
        spec.recharge_cycles + operand_sectors * spec.sector_cycles
    )
    assert conflict.total_bank_load_cycles == 2 * one_operand_cycles
    assert balanced.total_bank_load_cycles == one_operand_cycles
    assert conflict.actual_read_bytes == balanced.actual_read_bytes


def test_layout_search_contains_identity_and_can_reach_bound():
    problem = GemmProblem(32, 32, 32)
    tiling = GemmTiling(32, 32, 32, stage=2, ctas_per_wave=8)
    result = search_gemm_layouts(
        problem, tiling, options=_exact_options(), workers=2
    )
    by_name = {row.layouts.name: row for row in result.results}
    assert "linear" in by_name
    assert result.best.total_memory_cycles <= by_name["linear"].total_memory_cycles
    assert result.best.total_memory_cycles > 0
    assert result.best.bank_only_attainment >= (
        by_name["linear"].bank_only_attainment
    )


def test_partial_stack_gemm_keeps_physical_bank_swizzle_candidates():
    problem = GemmProblem(64, 64, 128)
    tiling = GemmTiling(32, 32, 32, stage=2, ctas_per_wave=8)
    spec = DramBankSpec(total_layers=8, connected_layers=5)
    options = GemmBankOptions(
        connectivity="interleaved",
        max_spatial_samples=None,
        max_k_samples=None,
        apply_littles_law=False,
    )
    result = search_gemm_layouts(
        problem, tiling, spec, options, workers=0
    )
    names = {row.layouts.name for row in result.results}
    # A partial connection count must not discard XOR candidates that are
    # legal for the independently configured physical bank field.
    assert spec.physical_bank_count == (
        spec.total_layers * spec.banks_per_layer
    )
    assert spec.port_count == (
        spec.connected_layers * spec.banks_per_layer
    )
    assert any(name.startswith("tile_sector_xor") for name in names)
    assert result.best.fixed_job_load_lower_bound_cycles <= (
        result.best.total_bank_load_cycles
    )


def test_search_exposes_conflict_objective_separately():
    problem = GemmProblem(256, 256, 8192)
    tiling = GemmTiling(128, 128, 32, stage=2, ctas_per_wave=8)
    result = search_gemm_layouts(
        problem,
        tiling,
        options=GemmBankOptions(
            max_spatial_samples=4,
            max_k_samples=4,
            apply_littles_law=False,
        ),
        workers=4,
        validation_top_k=3,
        validation_phase_samples=64,
    )
    names = {row.layouts.name for row in result.results}
    assert "linear" in names
    assert "tile_row_xor" in names
    assert any(name.startswith("tile_sector_xor") for name in names)
    assert result.validation_performed
    assert result.best_conflict.conflict_attainment == pytest.approx(1.0)
    assert result.best_conflict.worst_sample_conflict_attainment == pytest.approx(1.0)


def test_phase_residue_sampling_catches_common_modular_alias():
    problem = GemmProblem(256, 256, 8192)
    tiling = GemmTiling(128, 128, 32, stage=2, ctas_per_wave=8)
    spec = DramBankSpec()
    layouts = make_layout_set(
        problem,
        tiling,
        spec,
        preset="tile_major",
        swizzle=_xor_swizzle(spec, sector_bits=3),
        b_phase=spec.bank_count // 2,
        c_phase=spec.bank_count // 4,
        name="aliased_sector_3",
    )
    folded = model_gemm_bank(
        problem,
        tiling,
        layouts,
        spec,
        GemmBankOptions(
            max_spatial_samples=4,
            max_k_samples=4,
            apply_littles_law=False,
        ),
    )
    phase_residue = model_gemm_bank(
        problem,
        tiling,
        layouts,
        spec,
        GemmBankOptions(
            max_spatial_samples=64,
            max_k_samples=64,
            sampling_mode="phase_residue",
            apply_littles_law=False,
        ),
    )
    assert phase_residue.worst_sample_conflict_attainment <= (
        folded.worst_sample_conflict_attainment
    )
    assert len(phase_residue.sampled_profiles) >= len(
        folded.sampled_profiles
    )


def test_phase_residue_screen_is_not_treated_as_exact_proof():
    problem = GemmProblem(1, 16, 2560)
    tiling = GemmTiling(1, 16, 16, stage=2, ctas_per_wave=8)
    spec = DramBankSpec()
    layouts = make_layout_set(
        problem,
        tiling,
        spec,
        swizzle=BankSwizzle(
            kind="xor",
            row_bits=spec.bank_count.bit_length() - 1,
        ),
        b_phase=spec.bank_count // 2,
        c_phase=spec.bank_count // 4,
        name="late_conflict",
    )
    residue = model_gemm_bank(
        problem,
        tiling,
        layouts,
        spec,
        GemmBankOptions(
            max_spatial_samples=64,
            max_k_samples=64,
            sampling_mode="phase_residue",
            apply_littles_law=False,
        ),
    )
    exact = model_gemm_bank(
        problem,
        tiling,
        layouts,
        spec,
        GemmBankOptions(
            max_spatial_samples=None,
            max_k_samples=None,
            apply_littles_law=False,
        ),
    )
    assert residue.worst_sample_conflict_attainment >= (
        exact.worst_sample_conflict_attainment
    )
    assert len(residue.sampled_profiles) <= len(exact.sampled_profiles)


def test_gemv_tile_xor_reaches_absolute_bank_bound():
    problem = GemmProblem(1, 8192, 8192)
    tiling = GemmTiling(1, 256, 32, stage=2, ctas_per_wave=8)
    spec = DramBankSpec()
    layouts = make_layout_set(
        problem,
        tiling,
        spec,
        preset="tile_major",
        swizzle=_xor_swizzle(spec, include_rows=True),
        b_phase=spec.bank_count // 2,
        c_phase=spec.bank_count // 4,
        name="tile_xor",
    )
    tile_xor = model_gemm_bank(
        problem,
        tiling,
        layouts,
        spec,
        GemmBankOptions(
            max_spatial_samples=64,
            max_k_samples=64,
            sampling_mode="phase_residue",
            apply_littles_law=False,
        ),
    )
    assert tile_xor.conflict_attainment == pytest.approx(1.0)
    assert tile_xor.worst_sample_conflict_attainment == pytest.approx(1.0)
    assert tile_xor.bank_only_attainment == pytest.approx(1.0)
    assert tile_xor.mean_active_bank_fraction == pytest.approx(1.0)


@pytest.mark.parametrize(
    "shape,tile",
    [
        ((33, 35, 37), (32, 32, 16)),
        ((64, 96, 128), (32, 32, 32)),
        ((1, 257, 65), (1, 32, 16)),
    ],
)
def test_ragged_exact_model_is_deterministic(shape, tile):
    problem = GemmProblem(*shape)
    tiling = GemmTiling(*tile, stage=3, ctas_per_wave=8)
    spec = DramBankSpec()
    layouts = make_layout_set(
        problem, tiling, spec, preset="tile_major", name="ragged_tile"
    )
    first = model_gemm_bank(problem, tiling, layouts, spec, _exact_options())
    second = model_gemm_bank(problem, tiling, layouts, spec, _exact_options())
    assert first.total_load_cycles == second.total_load_cycles
    assert first.total_store_cycles == second.total_store_cycles
    assert first.actual_read_bytes == second.actual_read_bytes
    assert first.actual_write_bytes == second.actual_write_bytes


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_layout_search_parallel_backends_match_serial(backend):
    problem = GemmProblem(64, 96, 128)
    tiling = GemmTiling(32, 32, 32, stage=3, ctas_per_wave=8)
    options = GemmBankOptions(
        max_spatial_samples=4,
        max_k_samples=4,
        apply_littles_law=False,
        deterministic_seed=123,
    )
    common = dict(
        options=options,
        validation_top_k=1,
        validation_phase_samples=4,
    )
    serial = search_gemm_layouts(
        problem,
        tiling,
        workers=2,
        backend="serial",
        **common,
    )
    parallel = search_gemm_layouts(
        problem,
        tiling,
        workers=2,
        backend=backend,
        **common,
    )

    def fingerprint(result):
        return (
            result.best.layouts.name,
            tuple(
                (
                    row.layouts.name,
                    row.total_memory_cycles,
                    row.actual_read_bytes,
                    row.actual_write_bytes,
                    row.conflict_attainment,
                )
                for row in result.results
            ),
            tuple(
                (
                    row.layouts.name,
                    row.total_memory_cycles,
                    row.conflict_attainment,
                )
                for row in result.validated_results
            ),
        )

    assert fingerprint(parallel) == fingerprint(serial)


def test_auto_backend_keeps_small_search_serial_and_uses_process_for_large():
    spec = DramBankSpec()
    options = GemmBankOptions()
    small_problem = GemmProblem(256, 256, 1024)
    small_tiling = GemmTiling(128, 128, 32, ctas_per_wave=8)
    large_problem = GemmProblem(8192, 8192, 8192)
    large_tiling = GemmTiling(128, 256, 32, ctas_per_wave=8)
    small_tasks = len(default_layout_candidates(small_problem, small_tiling, spec))
    large_tasks = len(default_layout_candidates(large_problem, large_tiling, spec))
    small_work = (
        _gemm_model_work_units(small_problem, small_tiling, options)
        * small_tasks
    )
    large_work = (
        _gemm_model_work_units(large_problem, large_tiling, options)
        * large_tasks
    )
    assert resolve_backend("auto", 4, small_tasks, small_work) == "serial"
    assert resolve_backend("auto", 4, large_tasks, large_work) == "process"
    assert resolve_backend("thread", 4, small_tasks, small_work) == "thread"


def test_l2_statistical_thinning_is_candidate_order_independent():
    problem = GemmProblem(64, 96, 128)
    tiling = GemmTiling(32, 32, 32, stage=2, ctas_per_wave=8)
    spec = DramBankSpec()
    options = GemmBankOptions(
        a_miss_rate=0.37,
        b_miss_rate=0.61,
        max_spatial_samples=None,
        max_k_samples=None,
        apply_littles_law=False,
        deterministic_seed=99,
    )
    layouts = make_layout_set(problem, tiling, spec)
    first = model_gemm_bank(problem, tiling, layouts, spec, options)
    second = model_gemm_bank(problem, tiling, layouts, spec, options)
    assert first.total_memory_cycles == second.total_memory_cycles
    assert first.actual_read_bytes == second.actual_read_bytes


def test_fractional_traffic_knob_uses_hash_threshold_not_minimum_one():
    kept = _stable_thin(np.arange(10_000, dtype=np.int64), 0.01, 123)
    assert 50 <= kept.size <= 150


def test_mixed_input_precision_requires_explicit_compute_precision():
    with pytest.raises(ValueError, match="compute_dtype_bytes"):
        GemmProblem(32, 32, 32, a_dtype_bytes=2, b_dtype_bytes=1)
    problem = GemmProblem(
        32,
        32,
        32,
        a_dtype_bytes=2,
        b_dtype_bytes=1,
        compute_dtype_bytes=1,
    )
    assert problem.compute_dtype_bytes == 1


def test_output_kept_on_chip_generates_no_ddr_store():
    problem = GemmProblem(64, 64, 128)
    tiling = GemmTiling(32, 32, 32, stage=2, ctas_per_wave=8)
    spec = DramBankSpec()
    result = model_gemm_bank(
        problem,
        tiling,
        make_layout_set(problem, tiling, spec),
        spec,
        GemmBankOptions(
            c_write_rate=0.0,
            max_spatial_samples=None,
            max_k_samples=None,
            apply_littles_law=False,
        ),
    )
    assert result.actual_write_bytes == 0
    assert result.total_store_cycles == 0
    assert result.total_bank_store_cycles == 0


def test_all_cached_mixed_swizzles_have_zero_load_without_division_by_zero():
    problem = GemmProblem(32, 32, 32)
    tiling = GemmTiling(32, 32, 32, stage=2, ctas_per_wave=8)
    spec = DramBankSpec()
    layouts = make_layout_set(problem, tiling, spec, b_phase=32)
    result = model_gemm_bank(
        problem,
        tiling,
        layouts,
        spec,
        GemmBankOptions(
            a_miss_rate=0.0,
            b_miss_rate=0.0,
            c_write_rate=0.0,
            max_spatial_samples=None,
            max_k_samples=None,
            apply_littles_law=False,
        ),
    )
    assert result.actual_read_bytes == 0
    assert result.total_load_cycles == 0
    assert result.total_bank_load_cycles == 0


def test_batched_b_storage_is_explicitly_broadcast_or_private():
    spec = DramBankSpec()
    tiling = GemmTiling(32, 32, 32)
    private_problem = GemmProblem(32, 32, 32, batch=2)
    broadcast_problem = replace(
        private_problem, b_broadcast_across_batch=True
    )
    private = make_layout_set(private_problem, tiling, spec)
    broadcast = make_layout_set(broadcast_problem, tiling, spec)
    assert private.b_batch_stride_bytes > 0
    assert broadcast.b_batch_stride_bytes == 0


def test_four_sample_periodic_fold_matches_exact_enumeration():
    problem = GemmProblem(256, 256, 256)
    tiling = GemmTiling(
        32, 32, 32, stage=3, ctas_per_wave=8, row_panel=2
    )
    spec = DramBankSpec()
    layouts = make_layout_set(
        problem,
        tiling,
        spec,
        preset="tile_major",
        swizzle=_xor_swizzle(spec, include_rows=True),
        b_phase=spec.bank_count // 2,
        c_phase=spec.bank_count // 4,
    )
    exact = model_gemm_bank(problem, tiling, layouts, spec, _exact_options())
    folded = model_gemm_bank(
        problem,
        tiling,
        layouts,
        spec,
        GemmBankOptions(
            max_spatial_samples=4,
            max_k_samples=4,
            apply_littles_law=False,
        ),
    )
    assert folded.total_memory_cycles == exact.total_memory_cycles
    assert folded.actual_read_bytes == exact.actual_read_bytes
    assert folded.actual_write_bytes == exact.actual_write_bytes
