import math
from types import SimpleNamespace

import numpy as np
import pytest

from bank_oracle.engine import (
    apply_littles_law_cycles,
    decode_sector_ids,
    fixed_job_parallel_lower_bound_cycles,
    service_request_groups,
    service_requests,
    service_sectors,
)
from bank_oracle.layout import BankSwizzle
from bank_oracle.spec import DramBankSpec


@pytest.fixture
def spec():
    """Synthetic, non-calibrated geometry used by unit tests."""

    return DramBankSpec()


def _scalar_direct_decode(q, bank_count, sectors_per_row):
    block, sector = divmod(q, sectors_per_row)
    bank, row = block % bank_count, block // bank_count
    return bank, row, sector


def _scalar_interleaved_decode(q, bank_count, sectors_per_row):
    bank = q % bank_count
    stream = q // bank_count
    row, sector = divmod(stream, sectors_per_row)
    return bank, row, sector


def _full_sector_xor(spec):
    bits = min(
        spec.bank_count.bit_length() - 1,
        spec.sectors_per_row.bit_length() - 1,
    )
    return BankSwizzle(kind="xor", sector_bits=bits)


def _full_row_xor(spec):
    return BankSwizzle(kind="xor", row_bits=spec.bank_count.bit_length() - 1)


def _scalar_service_requests(requests, spec, swizzle, coalesce_scope="request"):
    jobs = {}
    for request_id, request in enumerate(requests):
        rid = 0 if coalesce_scope == "wave" else request_id
        for q in set(map(int, request)):
            bank, row, sector = _scalar_direct_decode(
                q, spec.bank_count, spec.sectors_per_row
            )
            if swizzle.kind == "xor":
                sector_mask = (1 << swizzle.sector_bits) - 1
                row_mask = (1 << swizzle.row_bits) - 1
                bank ^= (
                    ((sector >> swizzle.sector_source_shift) & sector_mask)
                    << swizzle.sector_bank_shift
                )
                bank ^= (
                    ((row >> swizzle.row_source_shift) & row_mask)
                    << swizzle.row_bank_shift
                )
                bank ^= swizzle.phase
                bank &= spec.bank_count - 1
            key = (rid, bank, row)
            jobs[key] = jobs.get(key, 0) | (1 << sector)
    bank_cycles = [0] * spec.bank_count
    for (_, bank, _), mask in jobs.items():
        bank_cycles[bank] += spec.recharge_cycles + spec.sector_cycles * mask.bit_count()
    return max(bank_cycles, default=0), bank_cycles


def test_default_synthetic_topology_is_internally_derived(spec):
    assert spec.physical_bank_count == (
        spec.total_layers * spec.banks_per_layer
    )
    assert spec.port_count == (
        spec.connected_layers * spec.banks_per_layer
    )
    assert spec.bank_count == spec.physical_bank_count
    assert spec.sectors_per_row == spec.row_bytes // spec.sector_bytes
    assert spec.row_read_cycles == (
        spec.sectors_per_row * spec.sector_cycles
    )
    assert spec.full_row_cycles == (
        spec.row_read_cycles + spec.recharge_cycles
    )
    assert spec.full_bank_wave_bytes == spec.bank_count * spec.row_bytes
    assert spec.raw_peak_bandwidth_bytes_s == pytest.approx(
        spec.port_count
        * spec.sector_bytes
        / spec.sector_cycles
        * spec.data_rate_hz
    )
    assert spec.connectivity_efficiency == pytest.approx(
        spec.row_read_cycles / spec.full_row_cycles
    )
    assert spec.cycles_to_seconds(spec.full_row_cycles) == pytest.approx(
        spec.full_row_cycles / spec.data_rate_hz
    )


def test_partial_stack_separates_physical_banks_from_ports():
    spec = DramBankSpec(total_layers=10, connected_layers=3)
    assert (
        spec.physical_bank_count
        == spec.bank_count
        == spec.total_layers * spec.banks_per_layer
    )
    assert spec.port_count == spec.connected_layers * spec.banks_per_layer
    assert spec.full_bank_wave_bytes == spec.bank_count * spec.row_bytes
    assert spec.raw_peak_bandwidth_bytes_s == pytest.approx(
        spec.port_count
        * spec.sector_bytes
        / spec.sector_cycles
        * spec.data_rate_hz
    )
    assert spec.connectivity_efficiency == 1.0


@pytest.mark.parametrize("layers", [1, 2, 4, 8])
def test_fully_connected_direct_full_rows_follow_supplied_spec(layers):
    spec = DramBankSpec(total_layers=layers, connected_layers=layers)
    result = service_sectors(
        np.arange(spec.physical_bank_count * spec.sectors_per_row), spec
    )
    assert result.cycles == spec.full_row_cycles
    data_cycles = result.sector_count * spec.sector_cycles
    assert data_cycles / (spec.port_count * result.cycles) == pytest.approx(
        spec.row_read_cycles / spec.full_row_cycles
    )


def test_two_of_four_connected_layers_approach_raw_port_bandwidth():
    spec = DramBankSpec(total_layers=4, connected_layers=2)
    rows_per_bank = 64
    result = service_sectors(np.arange(
        spec.physical_bank_count * spec.sectors_per_row * rows_per_bank
    ), spec)
    data_cycles = result.sector_count * spec.sector_cycles
    port_floor = data_cycles / spec.port_count
    # Two backing banks per port cover every interior recharge; only the final
    # caller-configured cooldown remains on a finite trace.
    assert result.cycles == port_floor + spec.recharge_cycles
    assert data_cycles / (spec.port_count * result.cycles) > 0.998
    assert result.sector_lower_bound_cycles == port_floor


def test_three_of_four_connected_layers_respect_analytic_steady_state_cap():
    spec = DramBankSpec(total_layers=4, connected_layers=3)
    rows_per_bank = 30
    result = service_sectors(np.arange(
        spec.physical_bank_count * spec.sectors_per_row * rows_per_bank
    ), spec)
    data_cycles = result.sector_count * spec.sector_cycles
    cap_cycles = data_cycles / (
        spec.port_count * spec.connectivity_efficiency
    )
    assert result.cycles == pytest.approx(cap_cycles)
    assert data_cycles / (spec.port_count * result.cycles) == pytest.approx(
        spec.connectivity_efficiency
    )
    assert result.fixed_job_lower_bound_cycles <= result.cycles


def test_total_layers_changes_partial_service_at_fixed_connected_layers():
    requests = np.arange(4 * 16 * 16 * 30)
    four_layers = service_sectors(
        requests, DramBankSpec(total_layers=4, connected_layers=3)
    )
    eight_layers = service_sectors(
        requests, DramBankSpec(total_layers=8, connected_layers=3)
    )
    assert four_layers.cycles > eight_layers.cycles


@pytest.mark.parametrize(
    "total_layers,connected_layers",
    [(4, 2), (4, 3), (4, 4), (8, 3)],
)
@pytest.mark.parametrize("connectivity", ["direct", "interleaved"])
def test_topology_lower_bounds_are_safe_on_random_waves(
    total_layers, connected_layers, connectivity
):
    spec = DramBankSpec(
        total_layers=total_layers,
        connected_layers=connected_layers,
    )
    rng = np.random.default_rng(
        1000 * total_layers + 10 * connected_layers
        + (connectivity == "interleaved")
    )
    requests = [rng.integers(0, 50_000, size=97) for _ in range(7)]
    result = service_requests(
        requests,
        spec,
        connectivity=connectivity,
        coalesce_scope="request",
    )
    assert result.sector_lower_bound_cycles <= result.cycles
    assert result.sector_lower_bound_cycles <= (
        result.fixed_job_lower_bound_cycles
    )
    assert result.fixed_job_lower_bound_cycles <= result.cycles
    assert 0.0 < result.transaction_efficiency <= 1.0
    assert 0.0 < result.bank_balance_efficiency <= 1.0
    assert 0.0 < result.overall_efficiency <= (
        spec.connectivity_efficiency + 1e-12
    )


def test_spec_from_arch_uses_only_matching_physical_fields():
    arch = SimpleNamespace(
        dram_layers_per_cluster=6,
        dram_active_layers=3,
        dram_banks_per_layer=5,
        dram_row_bytes=1536,
        dram_sector_bytes=96,
        dram_sector_cycles=5,
        dram_recharge_cycles=9,
        dram_data_rate_hz=1.25e9,
        dram_round_trip_latency_cycles=23,
        dram_latency_clock_hz=1.75e9,
        ddr_bandwidth=1.0,
    )
    derived = DramBankSpec.from_arch(arch)
    assert derived.bank_count == (
        arch.dram_layers_per_cluster * arch.dram_banks_per_layer
    )
    assert derived.port_count == (
        arch.dram_active_layers * arch.dram_banks_per_layer
    )
    assert derived.sectors_per_row == (
        arch.dram_row_bytes // arch.dram_sector_bytes
    )
    assert derived.data_rate_hz == arch.dram_data_rate_hz
    assert derived.round_trip_latency_cycles == (
        arch.dram_round_trip_latency_cycles
    )


def test_sector_mask_geometry_accepts_one_through_sixty_four_sectors():
    for count in (1, 7, 16, 31, 64):
        candidate = DramBankSpec(row_bytes=count * 96, sector_bytes=96)
        assert candidate.sectors_per_row == count
    with pytest.raises(ValueError, match=r"\[1, 64\]"):
        DramBankSpec(row_bytes=65 * 96, sector_bytes=96)


@pytest.mark.parametrize("sectors_per_row", [1, 7, 31, 64])
def test_uint64_masks_cover_configurable_row_geometries(sectors_per_row):
    spec = DramBankSpec(
        total_layers=2,
        connected_layers=2,
        banks_per_layer=4,
        row_bytes=sectors_per_row * 96,
        sector_bytes=96,
        sector_cycles=5,
        recharge_cycles=9,
    )
    sectors = np.arange(sectors_per_row, dtype=np.int64)
    result = service_requests(
        [sectors, sectors],
        spec,
        coalesce_scope="wave",
    )
    assert result.sector_count == sectors_per_row
    assert result.transferred_bytes == sectors_per_row * spec.sector_bytes
    assert result.cycles == spec.full_row_cycles


def test_xor_keeps_power_of_two_sector_field_constraint():
    spec = DramBankSpec(row_bytes=7 * 96, sector_bytes=96)
    with pytest.raises(ValueError, match="power-of-two sectors_per_row"):
        service_sectors(
            np.arange(spec.sectors_per_row),
            spec,
            swizzle=BankSwizzle(kind="xor", sector_bits=1),
        )


def test_direct_and_interleaved_decode_are_locked(spec):
    row_span = spec.bank_count * spec.sectors_per_row
    q = np.asarray(
        [
            0,
            spec.sectors_per_row - 1,
            spec.sectors_per_row,
            row_span - 1,
            row_span,
            row_span + spec.sectors_per_row - 1,
        ],
        dtype=np.int64,
    )
    direct = tuple(map(tuple, np.stack(decode_sector_ids(q, spec), axis=1).tolist()))
    interleaved = tuple(map(
        tuple,
        np.stack(
            decode_sector_ids(q, spec, connectivity="interleaved"), axis=1
        ).tolist(),
    ))
    assert direct == tuple(
        _scalar_direct_decode(
            int(x), spec.bank_count, spec.sectors_per_row
        )
        for x in q
    )
    assert interleaved == tuple(
        _scalar_interleaved_decode(
            int(x), spec.bank_count, spec.sectors_per_row
        )
        for x in q
    )


@pytest.mark.parametrize("fraction", [1, 2, 4, 8, 16])
def test_partial_row_transaction_curve(spec, fraction):
    sectors = max(1, spec.sectors_per_row * fraction // 16)
    result = service_sectors(np.arange(sectors), spec)
    assert result.cycles == (
        spec.recharge_cycles + sectors * spec.sector_cycles
    )
    assert result.transferred_bytes == sectors * spec.sector_bytes


@pytest.mark.parametrize("row_count", [1, 3, 11, 32, 64])
def test_direct_contiguous_small_op_parallelism(spec, row_count):
    sector_count = row_count * spec.sectors_per_row
    result = service_sectors(np.arange(sector_count), spec)
    assert result.active_banks == min(row_count, spec.bank_count)
    assert result.cycles == (
        math.ceil(row_count / spec.bank_count) * spec.full_row_cycles
    )


@pytest.mark.parametrize("sectors_per_bank", [1, 2, 7, 32])
def test_sector_interleaved_contiguous(spec, sectors_per_bank):
    sector_count = spec.bank_count * sectors_per_bank
    result = service_sectors(
        np.arange(sector_count), spec, connectivity="interleaved"
    )
    assert result.active_banks == spec.bank_count
    assert result.cycles == (
        spec.recharge_cycles
        + sectors_per_bank * spec.sector_cycles
    )


def test_efficiency_factorization(spec):
    # Deliberately combine a full row, a partial row, and an idle-bank tail.
    row_span = spec.bank_count * spec.sectors_per_row
    sectors = np.concatenate([
        np.arange(spec.sectors_per_row),
        np.arange(row_span, row_span + 5),
    ])
    result = service_sectors(sectors, spec)
    assert result.overall_efficiency == pytest.approx(
        result.transaction_efficiency * result.bank_balance_efficiency
    )


def test_fixed_job_bound_accounts_for_indivisible_row_jobs(spec):
    job_cycles = spec.recharge_cycles + spec.sector_cycles
    job_count = spec.bank_count * 4 + 1
    jobs = np.full(job_count, job_cycles, dtype=np.int64)
    assert fixed_job_parallel_lower_bound_cycles(
        jobs, spec.bank_count
    ) == 5 * job_cycles


def test_fixed_job_bound_is_safe_for_mixed_jobs(spec):
    # Any concrete assignment is an upper bound on the free-assignment
    # optimum, so the strict lower bound must never exceed it.
    jobs = np.asarray(
        [spec.full_row_cycles] * 17
        + [spec.recharge_cycles + 8 * spec.sector_cycles] * 71
        + [spec.recharge_cycles + spec.sector_cycles] * 143,
        dtype=np.int64,
    )
    assigned = np.zeros(spec.bank_count, dtype=np.int64)
    for job in sorted(jobs, reverse=True):
        assigned[np.argmin(assigned)] += job
    lower_bound = fixed_job_parallel_lower_bound_cycles(jobs, spec.bank_count)
    assert lower_bound <= int(assigned.max())
    assert lower_bound >= math.ceil(jobs.sum() / spec.bank_count)


def test_combined_conflict_bound_captures_full_wave_plus_tail(spec):
    result = service_sectors(
        np.arange(spec.bank_count * spec.sectors_per_row + 1),
        spec,
        connectivity="interleaved",
    )
    expected = (
        spec.full_row_cycles
        + spec.recharge_cycles
        + spec.sector_cycles
    )
    assert result.cycles == expected
    assert result.sector_lower_bound_cycles == expected
    assert result.fixed_job_lower_bound_cycles == expected


def test_request_scope_does_not_silently_remove_cross_cta_traffic(spec):
    request = np.arange(spec.sectors_per_row)
    request_scope = service_requests(
        [request, request], spec, coalesce_scope="request"
    )
    wave_scope = service_requests([request, request], spec, coalesce_scope="wave")
    assert request_scope.cycles == 2 * spec.full_row_cycles
    assert request_scope.transferred_bytes == 2 * spec.row_bytes
    assert wave_scope.cycles == spec.full_row_cycles
    assert wave_scope.transferred_bytes == spec.row_bytes


def test_request_groups_combine_distinct_swizzles_before_service(spec):
    request = np.arange(spec.sectors_per_row)
    sector_xor = _full_sector_xor(spec)
    groups = [([request], BankSwizzle()), ([request], sector_xor)]
    request_scope = service_request_groups(groups, spec, coalesce_scope="request")
    wave_scope = service_request_groups(groups, spec, coalesce_scope="wave")
    # Cross-CTA requests may coalesce within an operand, but two distinct
    # tensors must remain distinct even if their swizzles land on the same
    # physical coordinate.
    assert request_scope.cycles >= spec.full_row_cycles
    assert request_scope.transferred_bytes == 2 * spec.row_bytes
    assert wave_scope.cycles == request_scope.cycles
    assert wave_scope.transferred_bytes == request_scope.transferred_bytes


def test_sector_xor_reaches_small_transaction_upper_bound(spec):
    sectors = np.arange(spec.bank_count)
    linear = service_sectors(sectors, spec)
    swizzle = _full_sector_xor(spec)
    xor = service_sectors(sectors, spec, swizzle=swizzle)
    assert linear.cycles == spec.full_row_cycles
    assert linear.active_banks == 1
    expected = spec.recharge_cycles + spec.sector_cycles
    assert xor.cycles == expected
    assert xor.active_banks == spec.bank_count
    assert xor.sector_lower_bound_cycles == expected
    assert xor.bank_attainment == 1.0


def test_row_xor_resolves_adversarial_full_row_conflict(spec):
    sectors = np.concatenate(
        [
            np.arange(row * spec.bank_count * spec.sectors_per_row,
                      row * spec.bank_count * spec.sectors_per_row
                      + spec.sectors_per_row)
            for row in range(spec.bank_count)
        ]
    )
    linear = service_sectors(sectors, spec)
    swizzle = _full_row_xor(spec)
    xor = service_sectors(sectors, spec, swizzle=swizzle)
    assert linear.cycles == spec.bank_count * spec.full_row_cycles
    assert linear.active_banks == 1
    assert xor.cycles == spec.full_row_cycles
    assert xor.active_banks == spec.bank_count
    assert xor.sector_lower_bound_cycles == spec.full_row_cycles
    assert xor.bank_attainment == 1.0
    assert xor.overall_efficiency == pytest.approx(
        spec.connectivity_efficiency
    )


def test_xor_and_cyclic_bank_permutations_are_reversible(spec):
    raw_bank = np.arange(spec.bank_count, dtype=np.int64)
    row = (np.arange(spec.bank_count, dtype=np.int64) * 7) % 113
    sector = (
        np.arange(spec.bank_count, dtype=np.int64)
        % spec.sectors_per_row
    )
    bank_bits = spec.bank_count.bit_length() - 1
    candidates = [
        BankSwizzle(
            kind="xor",
            phase=spec.bank_count // 3,
            sector_source_shift=1,
            sector_bits=3,
            sector_bank_shift=max(bank_bits - 3, 0),
            row_source_shift=2,
            row_bits=2,
            row_bank_shift=0,
        ),
        BankSwizzle(
            kind="cyclic",
            phase=spec.bank_count // 4,
            cyclic_sector_alpha=5,
            cyclic_row_beta=7,
        ),
    ]
    for swizzle in candidates:
        mapped = swizzle.map_bank(raw_bank, row, sector, spec)
        restored = swizzle.unmap_bank(mapped, row, sector, spec)
        np.testing.assert_array_equal(restored, raw_bank)


def test_vectorized_histogram_matches_independent_scalar_oracle(spec):
    rng = np.random.default_rng(7)
    requests = [rng.integers(0, 5000, size=73) for _ in range(11)]
    swizzle = BankSwizzle(
        kind="xor",
        phase=spec.bank_count // 4,
        sector_bits=3,
        sector_bank_shift=spec.bank_count.bit_length() - 1 - 3,
        row_bits=min(4, spec.bank_count.bit_length() - 1),
        row_bank_shift=0,
    )
    for scope in ("request", "wave"):
        expected_cycles, expected_banks = _scalar_service_requests(
            requests, spec, swizzle, scope
        )
        actual = service_requests(
            requests,
            spec,
            swizzle=swizzle,
            coalesce_scope=scope,
        )
        assert actual.cycles == expected_cycles
        np.testing.assert_array_equal(actual.bank_cycles, expected_banks)


def test_littles_law_stage_buffer_is_applied_once():
    spec = DramBankSpec(
        data_rate_hz=1.0,
        latency_clock_hz=1.0,
        round_trip_latency_cycles=20,
    )
    observed = []
    for stage in (1, 2, 3, 4):
        final, limited, ll = apply_littles_law_cycles(
            bank_cycles=8,
            transferred_bytes=1024,
            outstanding_bytes=640 * stage,
            spec=spec,
        )
        observed.append(final)
        assert limited == (ll > 8)
    assert observed == pytest.approx([32, 20, 20, 20])


def test_littles_law_finite_window_cannot_finish_before_round_trip():
    spec = DramBankSpec(
        data_rate_hz=1.0,
        latency_clock_hz=1.0,
        round_trip_latency_cycles=20,
    )
    final, limited, ll = apply_littles_law_cycles(
        bank_cycles=8,
        transferred_bytes=1024,
        outstanding_bytes=10**9,
        spec=spec,
    )
    assert limited
    assert ll == 20
    assert final == 20
