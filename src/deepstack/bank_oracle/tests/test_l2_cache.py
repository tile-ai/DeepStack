from collections import deque
import random

import pytest

from bank_oracle.gemm import (
    GemmBankOptions,
    GemmProblem,
    GemmTiling,
    make_layout_set,
    model_gemm_bank,
    search_gemm_layouts,
)
from bank_oracle.l2_cache import (
    DeterministicFifoCache,
    L2CacheSpec,
)
from bank_oracle.layout import BankSwizzle
from bank_oracle.spec import DramBankSpec


def test_fifo_hit_at_capacity_boundary_does_not_refresh_order():
    cache = DeterministicFifoCache(
        L2CacheSpec(capacity_bytes=4 * 128)
    )
    assert not cache.access("a", 2).hit
    assert not cache.access("b", 2).hit

    boundary = cache.access("a", 2)
    assert boundary.hit
    assert boundary.reuse_distance_lines == 4

    # The hit did not promote A.  One new line evicts the oldest whole block.
    assert not cache.access("c", 1).hit
    again = cache.access("a", 2)
    assert not again.hit
    assert not again.cold
    assert again.reuse_distance_lines == 5


@pytest.mark.parametrize("capacity_lines", [1, 2, 7, 19])
def test_line_fifo_matches_independent_reference(capacity_lines):
    """Exercise the memory-bounded hot path against a simple FIFO oracle."""

    cache = DeterministicFifoCache(
        L2CacheSpec(capacity_bytes=capacity_lines * 128)
    )
    queue = deque()
    resident = set()
    rng = random.Random(0xCACE + capacity_lines)

    for _ in range(5_000):
        key = rng.randrange(64)
        expected_miss = key not in resident
        if expected_miss:
            if len(queue) == capacity_lines:
                resident.remove(queue.popleft())
            queue.append(key)
            resident.add(key)
        assert cache.access_line(key) == int(expected_miss)
        assert cache.resident_lines == len(queue)


def test_oversized_fifo_block_streams_and_clears_prior_residency():
    cache = DeterministicFifoCache(
        L2CacheSpec(capacity_bytes=4 * 128)
    )
    cache.access("resident", 1)
    oversized = cache.access("stream", 5)
    assert not oversized.hit
    assert cache.resident_lines == 0
    assert not cache.access("resident", 1).hit


@pytest.mark.parametrize("capacity", [0, 129])
def test_invalid_l2_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        L2CacheSpec(capacity_bytes=capacity)


def test_l2_spec_can_snapshot_arch_capacity():
    class Arch:
        l2_capacity = 4096

    assert L2CacheSpec.from_arch(Arch()).capacity_lines == 32


def _small_reuse_case(capacity_bytes=None, *, sampled=False):
    problem = GemmProblem(1, 512, 64)
    tiling = GemmTiling(1, 256, 32, stage=2, ctas_per_wave=1)
    spec = DramBankSpec()
    layouts = make_layout_set(problem, tiling, spec, preset="tile_major")
    options = GemmBankOptions(
        max_spatial_samples=1 if sampled else None,
        max_k_samples=None,
        apply_littles_law=False,
        coalesce_scope="request",
        l2_cache=(
            None
            if capacity_bytes is None
            else L2CacheSpec(capacity_bytes=capacity_bytes)
        ),
    )
    return model_gemm_bank(problem, tiling, layouts, spec, options)


def test_gemm_fifo_filters_whole_hit_tiles_before_bank_service():
    uncached = _small_reuse_case()
    cached = _small_reuse_case(64 * 1024)
    tiny = _small_reuse_case(128)

    assert cached.l2_result is not None
    assert cached.l2_result.a.accesses == 4
    # The two 64-B K tiles share one physical 128-B cache line.
    assert cached.l2_result.a.misses == 1
    assert cached.l2_result.a.hits == 3
    assert cached.l2_result.b.hits == 0
    assert uncached.actual_read_bytes - cached.actual_read_bytes == 3 * 128

    # A 16-KiB B tile is larger than this cache and flushes the one-line A.
    assert tiny.l2_result.a.hits == 0
    assert tiny.actual_read_bytes == uncached.actual_read_bytes


def test_output_pressure_can_evict_an_otherwise_reused_a_tile():
    problem = GemmProblem(1, 512, 32)
    tiling = GemmTiling(1, 256, 32, ctas_per_wave=1)
    spec = DramBankSpec()
    layouts = make_layout_set(problem, tiling, spec, preset="tile_major")
    capacity = 129 * 128  # exactly A(1 line) + one B tile(128 lines)

    def run(output_pressure):
        return model_gemm_bank(
            problem,
            tiling,
            layouts,
            spec,
            GemmBankOptions(
                max_spatial_samples=None,
                max_k_samples=None,
                apply_littles_law=False,
                l2_cache=L2CacheSpec(
                    capacity, output_pressure=output_pressure
                ),
            ),
        )

    with_pressure = run(True)
    without_pressure = run(False)
    assert with_pressure.l2_result.a.hits == 0
    assert without_pressure.l2_result.a.hits == 1
    assert without_pressure.actual_read_bytes < with_pressure.actual_read_bytes


def test_fast_bank_sampling_still_walks_the_complete_fifo_trace():
    exact = _small_reuse_case(64 * 1024)
    sampled = _small_reuse_case(64 * 1024, sampled=True)
    assert sampled.l2_result == exact.l2_result
    # This case has one cold wave followed by one hit wave, so the weighted
    # representative is also exact for DDR traffic.
    assert sampled.actual_read_bytes == exact.actual_read_bytes


def test_broadcast_b_reuses_one_fifo_identity_across_batches():
    spec = DramBankSpec()
    tiling = GemmTiling(1, 256, 32, stage=2, ctas_per_wave=1)
    options = GemmBankOptions(
        max_spatial_samples=None,
        max_k_samples=None,
        apply_littles_law=False,
        l2_cache=L2CacheSpec(64 * 1024),
    )

    private = GemmProblem(1, 256, 32, batch=2)
    private_result = model_gemm_bank(
        private,
        tiling,
        make_layout_set(private, tiling, spec, preset="tile_major"),
        spec,
        options,
    )
    broadcast = GemmProblem(
        1, 256, 32, batch=2, b_broadcast_across_batch=True
    )
    broadcast_result = model_gemm_bank(
        broadcast,
        tiling,
        make_layout_set(broadcast, tiling, spec, preset="tile_major"),
        spec,
        options,
    )
    assert private_result.l2_result.b.hits == 0
    assert broadcast_result.l2_result.b.hits == 1
    assert broadcast_result.actual_read_bytes < private_result.actual_read_bytes


def test_l2_and_fractional_hash_thinning_cannot_double_charge_hits():
    with pytest.raises(ValueError, match="cannot be combined"):
        GemmBankOptions(
            a_miss_rate=0.5,
            l2_cache=L2CacheSpec(1024),
        )


def test_l2_line_and_dram_sector_sizes_must_match():
    problem = GemmProblem(1, 256, 32)
    tiling = GemmTiling(1, 256, 32, ctas_per_wave=1)
    spec = DramBankSpec()
    layouts = make_layout_set(problem, tiling, spec)
    with pytest.raises(ValueError, match="line_bytes"):
        model_gemm_bank(
            problem,
            tiling,
            layouts,
            spec,
            GemmBankOptions(l2_cache=L2CacheSpec(1024, line_bytes=64)),
        )


def test_fifo_gemm_result_is_bitwise_deterministic():
    first = _small_reuse_case(64 * 1024)
    second = _small_reuse_case(64 * 1024)
    assert first.l2_result == second.l2_result
    assert first.actual_read_bytes == second.actual_read_bytes
    assert first.total_memory_cycles == second.total_memory_cycles


def test_l2_layout_search_matches_serial_and_process_backends():
    problem = GemmProblem(1, 512, 64)
    tiling = GemmTiling(1, 256, 32, ctas_per_wave=1)
    spec = DramBankSpec()
    options = GemmBankOptions(
        max_spatial_samples=None,
        max_k_samples=None,
        apply_littles_law=False,
        l2_cache=L2CacheSpec(64 * 1024),
    )
    candidates = (
        make_layout_set(problem, tiling, spec, preset="linear", name="linear"),
        make_layout_set(
            problem, tiling, spec, preset="tile_major", name="tile"
        ),
    )
    serial = search_gemm_layouts(
        problem,
        tiling,
        spec=spec,
        options=options,
        candidates=candidates,
        backend="serial",
        workers=2,
    )
    process = search_gemm_layouts(
        problem,
        tiling,
        spec=spec,
        options=options,
        candidates=candidates,
        backend="process",
        workers=2,
    )

    def fingerprint(search):
        return (
            search.best.layouts.name,
            tuple(
                (
                    result.layouts.name,
                    result.actual_read_bytes,
                    result.total_memory_cycles,
                    result.l2_result,
                )
                for result in search.results
            ),
        )

    assert fingerprint(process) == fingerprint(serial)


def test_physical_bank_swizzle_does_not_change_logical_l2_trace():
    problem = GemmProblem(1, 512, 64)
    tiling = GemmTiling(1, 256, 32, ctas_per_wave=1)
    spec = DramBankSpec()
    options = GemmBankOptions(
        max_spatial_samples=None,
        max_k_samples=None,
        apply_littles_law=False,
        l2_cache=L2CacheSpec(64 * 1024),
    )
    plain = make_layout_set(problem, tiling, spec, preset="tile_major")
    swizzled = make_layout_set(
        problem,
        tiling,
        spec,
        preset="tile_major",
        swizzle=BankSwizzle(kind="xor", sector_bits=4),
    )
    plain_result = model_gemm_bank(
        problem, tiling, plain, spec, options
    )
    swizzled_result = model_gemm_bank(
        problem, tiling, swizzled, spec, options
    )
    assert swizzled_result.l2_result == plain_result.l2_result
    assert swizzled_result.actual_read_bytes == plain_result.actual_read_bytes


def test_distinct_subline_tiles_with_same_sector_share_fifo_identity():
    problem = GemmProblem(1, 2, 1)
    tiling = GemmTiling(1, 1, 1, ctas_per_wave=1)
    spec = DramBankSpec()
    result = model_gemm_bank(
        problem,
        tiling,
        make_layout_set(problem, tiling, spec, preset="linear"),
        spec,
        GemmBankOptions(
            max_spatial_samples=None,
            max_k_samples=None,
            apply_littles_law=False,
            coalesce_scope="request",
            c_write_rate=0.0,
            l2_cache=L2CacheSpec(8 * 128, output_pressure=False),
        ),
    )
    assert result.l2_result.a.hits == 1
    assert result.l2_result.b.hits == 1
    assert result.actual_read_bytes == 2 * 128


def test_partially_overlapping_tiles_filter_only_missing_sectors():
    problem = GemmProblem(1, 192, 1)
    tiling = GemmTiling(1, 96, 1, ctas_per_wave=1)
    spec = DramBankSpec()
    result = model_gemm_bank(
        problem,
        tiling,
        make_layout_set(problem, tiling, spec, preset="linear"),
        spec,
        GemmBankOptions(
            max_spatial_samples=None,
            max_k_samples=None,
            apply_littles_law=False,
            coalesce_scope="request",
            c_write_rate=0.0,
            l2_cache=L2CacheSpec(3 * 128, output_pressure=False),
        ),
    )
    # B tiles touch [line0,line1] and [line1,line2].  The second request is
    # partial: line1 hits while only line2 reaches DDR.
    assert result.l2_result.b.partial_hits == 1
    assert result.l2_result.b.hit_lines == 1
    assert result.l2_result.b.miss_lines == 3
    # One A line plus three unique B lines.
    assert result.actual_read_bytes == 4 * 128


def test_small_cache_aware_problem_forces_exact_bank_profiles():
    problem = GemmProblem(1, 256, 32)
    tiling = GemmTiling(1, 1, 16, ctas_per_wave=1)
    spec = DramBankSpec()
    layouts = make_layout_set(problem, tiling, spec, preset="tile_major")
    common = dict(
        apply_littles_law=False,
        coalesce_scope="request",
        c_write_rate=0.0,
        l2_cache=L2CacheSpec(192 * 128, output_pressure=True),
    )
    automatic = model_gemm_bank(
        problem, tiling, layouts, spec, GemmBankOptions(**common)
    )
    exact = model_gemm_bank(
        problem,
        tiling,
        layouts,
        spec,
        GemmBankOptions(
            max_spatial_samples=None,
            max_k_samples=None,
            **common,
        ),
    )
    assert automatic.actual_read_bytes == exact.actual_read_bytes
    assert automatic.total_memory_cycles == exact.total_memory_cycles
    assert automatic.sampled_profiles == exact.sampled_profiles
