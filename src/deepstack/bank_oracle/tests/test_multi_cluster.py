from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from bank_oracle.gemm import (
    GemmBankOptions,
    GemmProblem,
    GemmTiling,
)
from bank_oracle.l2_cache import L2CacheSpec
from bank_oracle.layout import BankSwizzle
from bank_oracle.multi_cluster import (
    CannonPlan,
    ClusterGrid,
    ClusterUnitSpec,
    DistributedGemmPlan,
    DistributedL2Session,
    LineAccessSource,
    LineRequestEvent,
    LocalModelConfig,
    NoCSpec,
    PhysicalLineId,
    TensorPlacement,
    TensorReadRequest,
    TensorWriteRequest,
    TrafficFlow,
    model_cannon_gemm,
    model_distributed_gemm,
    model_read_stage,
    model_write_stage,
    split_extent,
)


def _unit(count, *, fill_policy="home_only", l2_bytes=64 * 1024):
    return ClusterUnitSpec(
        clusters_per_unit=count,
        l2_per_cluster=L2CacheSpec(l2_bytes),
        noc=NoCSpec(
            link_bandwidth_bytes_s=16e9,
            hop_latency_seconds=2e-9,
            topology="ideal_switch",
        ),
        l2_bandwidth_bytes_s=1e12,
        remote_fill_policy=fill_policy,
    )


def _local(tile=2, compute_cycles=20):
    return LocalModelConfig(
        GemmTiling(tile, tile, tile, stage=2, ctas_per_wave=1),
        bank_options=GemmBankOptions(
            max_spatial_samples=None,
            max_k_samples=None,
            apply_littles_law=False,
        ),
        compute_cycles_per_k=compute_cycles,
        store_other_memory_cycles=0.0,
    )


def _event(requester, owner, allocation="x", line=0):
    return LineRequestEvent(
        requester=requester,
        owner=owner,
        physical_line=PhysicalLineId(allocation, owner, line),
        owner_sector=line,
        tensor="A",
    )


def test_epoch_replay_distinguishes_owner_hit_miss_and_mshr_merge():
    session = DistributedL2Session.homogeneous(3, L2CacheSpec(4 * 128))
    result = session.replay_epoch((_event(2, 0), _event(1, 0)))

    assert result.ddr_miss_lines == 1
    assert result.mshr_merged_lines == 1
    assert result.owner_l2_hit_lines == 0
    assert result.remote_response_bytes == 2 * 128

    warm = session.replay_epoch((_event(1, 0),))
    assert warm.ddr_miss_lines == 0
    assert warm.owner_l2_hit_lines == 1
    assert warm.remote_owner_l2_hit_bytes == 128


def test_epoch_replay_is_input_order_independent_and_coalesces_duplicates():
    events = (_event(2, 0), _event(1, 0), _event(1, 0))
    forward = DistributedL2Session.homogeneous(
        3, L2CacheSpec(4 * 128)
    ).replay_epoch(events)
    backward = DistributedL2Session.homogeneous(
        3, L2CacheSpec(4 * 128)
    ).replay_epoch(reversed(events))
    assert forward == backward
    assert forward.accessed_lines == 2
    assert forward.ddr_miss_lines == 1


def test_detailed_replay_reports_an_unhashable_mapping_before_deduplication():
    session = DistributedL2Session.homogeneous(1, L2CacheSpec(4 * 128))
    invalid = replace(
        _event(0, 0), mapping_signature=["not", "hashable"]
    )
    with pytest.raises(ValueError, match="hashable"):
        session.replay_epoch((invalid,))
    assert session.resident_lines == (0,)


def test_requester_fill_can_remove_second_remote_response():
    unit = _unit(2, fill_policy="requester_and_home")
    session = DistributedL2Session.homogeneous(2, unit.l2_per_cluster)
    request = TensorReadRequest(
        requester=1,
        shard_owner=0,
        tensor="A",
        allocation_id="weights:A",
        sectors=(0,),
        placement=TensorPlacement.sharded(),
    )
    cold = model_read_stage((request,), unit, session)
    warm = model_read_stage((request,), unit, session)

    assert cold.ddr_read_bytes == 128
    assert cold.noc.total_bytes == 128
    assert warm.ddr_read_bytes == 0
    assert warm.l2.requester_l2_hit_lines == 1
    assert warm.noc.total_bytes == 0


def test_same_logical_line_in_disjoint_allocations_does_not_cache_alias():
    session = DistributedL2Session.homogeneous(1, L2CacheSpec(4 * 128))
    first = session.replay_epoch((_event(0, 0, "A", 7),))
    second = session.replay_epoch(
        (
            LineRequestEvent(
                0,
                0,
                PhysicalLineId("B", 0, 7),
                8,
                "A",
            ),
        )
    )
    assert first.ddr_miss_lines == second.ddr_miss_lines == 1


def test_cross_epoch_owner_address_alias_is_rejected_atomically():
    session = DistributedL2Session.homogeneous(1, L2CacheSpec(4 * 128))
    session.replay_epoch((_event(0, 0, "A", 7),))
    before = session.resident_lines
    with pytest.raises(ValueError, match="overlap"):
        session.replay_epoch((_event(0, 0, "B", 7),))
    assert session.resident_lines == before


def test_physical_line_home_is_stable_across_epochs_and_session_clones():
    session = DistributedL2Session.homogeneous(2, L2CacheSpec(4 * 128))
    line = PhysicalLineId("stable", 0, 0)
    first = LineRequestEvent(1, 0, line, 0, "A")
    moved = LineRequestEvent(0, 1, line, 0, "A")
    session.replay_epoch((first,))
    cloned = session.clone()
    assert cloned.resident_lines == session.resident_lines == (1, 0)
    with pytest.raises(ValueError, match="multiple homes"):
        cloned.replay_epoch((moved,))
    session.reset()
    assert session.resident_lines == (0, 0)
    # Reset intentionally starts a new physical-allocation session.
    assert session.replay_epoch((moved,)).ddr_miss_lines == 1


def test_persistent_allocation_cannot_change_its_bank_mapping():
    unit = _unit(1)
    session = DistributedL2Session.homogeneous(1, unit.l2_per_cluster)
    base = TensorReadRequest(
        0,
        0,
        "A",
        "stable_mapping",
        (0,),
        TensorPlacement.sharded(),
        BankSwizzle(),
    )
    model_read_stage((base,), unit, session)
    with pytest.raises(ValueError, match="multiple bank mappings"):
        model_read_stage(
            (replace(base, swizzle=BankSwizzle(phase=1)),),
            unit,
            session,
        )

    session.reset()
    model_read_stage(
        (replace(base, swizzle=BankSwizzle(phase=1)),),
        unit,
        session,
    )


def test_persistent_mapping_signature_includes_the_dram_decoder():
    unit = _unit(1)
    session = DistributedL2Session.homogeneous(1, unit.l2_per_cluster)
    request = TensorReadRequest(
        0, 0, "A", "decoder", (0,), TensorPlacement.sharded()
    )
    model_read_stage((request,), unit, session, connectivity="direct")
    with pytest.raises(ValueError, match="multiple bank mappings"):
        model_read_stage(
            (request,), unit, session, connectivity="interleaved"
        )


def test_persistent_cache_can_be_reused_for_a_pure_dram_timing_sweep():
    unit = _unit(1)
    session = DistributedL2Session.homogeneous(1, unit.l2_per_cluster)
    request = TensorReadRequest(
        0, 0, "A", "timing_sweep", (0,), TensorPlacement.sharded()
    )
    assert model_read_stage((request,), unit, session).ddr_read_bytes == 128
    faster = replace(
        unit,
        dram=replace(unit.dram, data_rate_hz=unit.dram.data_rate_hz * 2),
    )
    assert model_read_stage((request,), faster, session).ddr_read_bytes == 0


def test_high_level_mapping_signature_includes_dtype_and_storage_extent():
    unit = _unit(1)
    session = DistributedL2Session.homogeneous(1, unit.l2_per_cluster)
    base = DistributedGemmPlan(
        GemmProblem(
            4,
            8,
            4,
            a_dtype_bytes=1,
            b_dtype_bytes=1,
            c_dtype_bytes=1,
        ),
        ClusterGrid(),
        unit,
        _local(),
        allocation_id="dtype_stable",
    )
    model_distributed_gemm(base, session=session)
    with pytest.raises(ValueError, match="multiple bank mappings"):
        model_distributed_gemm(
            replace(
                base,
                problem=GemmProblem(
                    4,
                    8,
                    4,
                    a_dtype_bytes=2,
                    b_dtype_bytes=2,
                    c_dtype_bytes=2,
                ),
            ),
            session=session,
        )


def test_detailed_and_bulk_paths_cannot_mix_unsigned_allocation_layouts():
    unit = _unit(1)
    session = DistributedL2Session.homogeneous(1, unit.l2_per_cluster)
    session.replay_epoch((_event(0, 0, "mixed", 0),))
    with pytest.raises(ValueError, match="multiple bank mappings"):
        model_read_stage(
            (
                TensorReadRequest(
                    0,
                    0,
                    "A",
                    "mixed",
                    (0,),
                    TensorPlacement.sharded(),
                ),
            ),
            unit,
            session,
        )


def test_invalid_swizzle_fails_before_mutating_persistent_l2():
    unit = _unit(1)
    session = DistributedL2Session.homogeneous(1, unit.l2_per_cluster)
    invalid = TensorReadRequest(
        0,
        0,
        "A",
        "invalid_swizzle",
        (0,),
        TensorPlacement.sharded(),
        BankSwizzle(phase=unit.dram.bank_count),
    )
    with pytest.raises(ValueError, match="phase"):
        model_read_stage((invalid,), unit, session)
    assert session.resident_lines == (0,)


def test_requester_fill_policy_rejects_a_later_write_to_stale_copies():
    unit = _unit(2, fill_policy="requester_and_home")
    session = DistributedL2Session.homogeneous(2, unit.l2_per_cluster)
    model_read_stage(
        (
            TensorReadRequest(
                1,
                0,
                "A",
                "immutable",
                (0,),
                TensorPlacement.sharded(),
            ),
        ),
        unit,
        session,
    )
    before = session.resident_lines
    write = TensorWriteRequest(
        0,
        0,
        "A",
        "immutable",
        (0,),
        TensorPlacement.sharded(),
    )
    with pytest.raises(ValueError, match="read-only"):
        model_write_stage(
            (write,),
            unit,
            session=session,
        )
    assert session.resident_lines == before

    cloned = session.clone()
    with pytest.raises(ValueError, match="read-only"):
        model_write_stage((write,), unit, session=cloned)
    cloned.reset()
    model_write_stage((write,), unit, session=cloned)

    # The sticky read-only marker is allocation-specific.
    model_write_stage(
        (replace(write, allocation_id="other", sectors=(1,)),),
        unit,
        session=session,
    )


def test_remote_cold_read_charges_only_the_owner_dram_banks():
    unit = _unit(2)
    session = DistributedL2Session.homogeneous(2, unit.l2_per_cluster)
    result = model_read_stage(
        (
            TensorReadRequest(
                requester=1,
                shard_owner=0,
                tensor="A",
                allocation_id="owner_only",
                sectors=(0, 1, 2),
                placement=TensorPlacement.sharded(),
            ),
        ),
        unit,
        session,
    )
    assert [row.owner for row in result.owner_bank_results] == [0]
    assert result.owner_bank_results[0].service.transferred_bytes == 3 * 128
    assert result.noc.total_bytes == 3 * 128


def test_each_owner_has_an_independent_configured_bank_stack():
    unit = _unit(2)
    session = DistributedL2Session.homogeneous(2, unit.l2_per_cluster)
    result = model_read_stage(
        (
            TensorReadRequest(
                0,
                0,
                "A",
                "owner0",
                tuple(range(unit.dram.sectors_per_row)),
                TensorPlacement.sharded(),
            ),
            TensorReadRequest(
                1,
                1,
                "A",
                "owner1",
                tuple(range(unit.dram.sectors_per_row)),
                TensorPlacement.sharded(),
            ),
        ),
        unit,
        session,
        apply_littles_law=False,
    )
    assert len(result.owner_bank_results) == 2
    assert [row.service.cycles for row in result.owner_bank_results] == [
        unit.dram.full_row_cycles,
        unit.dram.full_row_cycles,
    ]
    assert result.bank_seconds == pytest.approx(
        unit.dram.cycles_to_seconds(unit.dram.full_row_cycles)
    )
    assert result.ddr_read_bytes == 2 * unit.dram.row_bytes


def test_output_pressure_evicts_private_l2_input_state():
    unit = ClusterUnitSpec(
        1,
        l2_per_cluster=L2CacheSpec(128, output_pressure=True),
    )
    session = DistributedL2Session.homogeneous(1, unit.l2_per_cluster)
    read = TensorReadRequest(
        0, 0, "A", "input", (0,), TensorPlacement.sharded()
    )
    assert model_read_stage((read,), unit, session).ddr_read_bytes == 128
    model_write_stage(
        (
            TensorWriteRequest(
                0, 0, "C", "output", (1,), TensorPlacement.sharded()
            ),
        ),
        unit,
        session=session,
    )
    assert model_read_stage((read,), unit, session).ddr_read_bytes == 128


def test_l2_session_capacity_must_match_cluster_unit():
    unit = _unit(2, l2_bytes=1024)
    wrong = DistributedL2Session.homogeneous(2, L2CacheSpec(2048))
    with pytest.raises(ValueError, match="capacity/policy"):
        model_read_stage((), unit, wrong)


def test_overlapping_physical_allocations_are_rejected_before_bank_timing():
    unit = _unit(1)
    session = DistributedL2Session.homogeneous(1, unit.l2_per_cluster)
    requests = (
        TensorReadRequest(
            0, 0, "A", "alloc0", (0,), TensorPlacement.sharded()
        ),
        TensorReadRequest(
            0, 0, "A", "alloc1", (0,), TensorPlacement.sharded()
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        model_read_stage(requests, unit, session)
    assert session.resident_lines == (0,)


@pytest.mark.parametrize(
    "grid, cluster_count",
    [(ClusterGrid(1, 4, 1), 4), (ClusterGrid(1, 8, 1), 8)],
)
def test_generic_unit_cluster_count_is_configurable(grid, cluster_count):
    plan = DistributedGemmPlan(
        GemmProblem(2, cluster_count * 2, 4),
        grid,
        _unit(cluster_count),
        _local(),
        a_placement=TensorPlacement.replicated(),
    )
    result = model_distributed_gemm(plan)
    assert len(result.local_results) == cluster_count
    assert result.useful_flops == 2 * 2 * (cluster_count * 2) * 4
    assert not result.requires_reduction
    assert result.compute_seconds == max(
        local.seconds for local in result.local_results
    )


@pytest.mark.parametrize("cluster_count", [4, 8])
def test_generic_canonical_inputs_hit_remote_owner_l2_on_second_run(
    cluster_count,
):
    unit = _unit(cluster_count)
    session = DistributedL2Session.homogeneous(
        cluster_count, unit.l2_per_cluster
    )
    plan = DistributedGemmPlan(
        GemmProblem(4, cluster_count * 2, 4),
        ClusterGrid(1, cluster_count, 1),
        unit,
        _local(),
        a_placement=TensorPlacement.canonical(0),
        b_placement=TensorPlacement.canonical(0),
        allocation_id="warm_weights",
    )
    cold = model_distributed_gemm(plan, session=session)
    warm = model_distributed_gemm(plan, session=session)

    assert cold.total_ddr_read_bytes > 0
    assert warm.total_ddr_read_bytes == 0
    assert warm.remote_l2_hit_bytes > 0
    assert warm.input_stage.owner_bank_results == ()
    assert warm.input_stage.bank_seconds == 0
    assert warm.input_stage.l2_seconds > 0
    # A remote owner-L2 hit still returns the line through the NoC.
    assert warm.input_stage.noc.total_bytes > 0


def test_sharded_storage_is_locally_packed_but_global_mode_is_available():
    base = DistributedGemmPlan(
        GemmProblem(5, 11, 9, batch=2),
        ClusterGrid(2, 2, 1),
        _unit(4, l2_bytes=1 << 20),
        _local(),
    )
    packed = model_distributed_gemm(base)
    global_view = model_distributed_gemm(
        replace(base, sharded_storage="global", allocation_id="global")
    )

    # Ragged local rows use their local pitch/tiles instead of retaining the
    # full tensor's gaps.  A/B/C still keep disjoint aligned region anchors.
    assert packed.total_ddr_read_bytes < global_view.total_ddr_read_bytes
    assert packed.total_ddr_write_bytes < global_view.total_ddr_write_bytes
    assert packed.input_stage.bank_seconds != global_view.input_stage.bank_seconds

    with pytest.raises(ValueError, match="sharded_storage"):
        replace(base, sharded_storage="unknown")


def test_packed_shard_signature_fixes_global_extent_grid_and_logical_slice():
    unit = _unit(4, l2_bytes=1 << 20)
    session = DistributedL2Session.homogeneous(4, unit.l2_per_cluster)
    first = DistributedGemmPlan(
        GemmProblem(1, 2, 1),
        ClusterGrid(1, 4, 1),
        unit,
        _local(tile=1),
        allocation_id="same_packed_storage",
    )
    model_distributed_gemm(first, session=session)
    remapped = replace(
        first,
        problem=GemmProblem(2, 1, 1),
        grid=ClusterGrid(4, 1, 1),
    )
    with pytest.raises(ValueError, match="multiple bank mappings"):
        model_distributed_gemm(remapped, session=session)


def test_distinct_persistent_allocations_require_disjoint_physical_bases():
    unit = _unit(1, l2_bytes=1 << 20)
    session = DistributedL2Session.homogeneous(1, unit.l2_per_cluster)
    first = DistributedGemmPlan(
        GemmProblem(4, 4, 4),
        ClusterGrid(),
        unit,
        _local(),
        allocation_id="allocation0",
    )
    model_distributed_gemm(first, session=session)
    with pytest.raises(ValueError, match="overlap"):
        model_distributed_gemm(
            replace(first, allocation_id="allocation1"),
            session=session,
        )

    model_distributed_gemm(
        replace(
            first,
            allocation_id="allocation1",
            allocation_base_sector=1 << 20,
        ),
        session=session,
    )
    with pytest.raises(ValueError, match="allocation_base_sector"):
        replace(first, allocation_base_sector=-1)


def test_canonical_and_replicated_storage_keep_global_address_views():
    placements = dict(
        a_placement=TensorPlacement.canonical(0),
        b_placement=TensorPlacement.replicated(),
        c_placement=TensorPlacement.canonical(0),
    )
    base = DistributedGemmPlan(
        GemmProblem(5, 11, 9),
        ClusterGrid(2, 2, 1),
        _unit(4, l2_bytes=1 << 20),
        _local(),
        **placements,
    )
    packed_setting = model_distributed_gemm(base)
    global_setting = model_distributed_gemm(
        replace(base, sharded_storage="global", allocation_id="full_global")
    )
    assert packed_setting.total_ddr_read_bytes == global_setting.total_ddr_read_bytes
    assert packed_setting.total_ddr_write_bytes == global_setting.total_ddr_write_bytes
    assert (
        packed_setting.input_stage.bank_seconds
        == global_setting.input_stage.bank_seconds
    )


def test_generic_canonical_output_supports_disjoint_subline_shards():
    result = model_distributed_gemm(
        DistributedGemmPlan(
            GemmProblem(2, 8, 4),
            ClusterGrid(1, 4, 1),
            _unit(4),
            _local(),
            c_placement=TensorPlacement.canonical(0),
        )
    )
    assert result.total_ddr_write_bytes > 0
    assert result.output_stage.noc.total_bytes > 0


def test_generic_propagates_connectivity_and_littles_law_options():
    unit = _unit(1, l2_bytes=1 << 20)
    base_options = GemmBankOptions(
        max_spatial_samples=None,
        max_k_samples=None,
        apply_littles_law=False,
        connectivity="direct",
    )
    base_local = LocalModelConfig(
        GemmTiling(1, 256, 32, stage=3, ctas_per_wave=1),
        bank_options=base_options,
        compute_cycles_per_k=10,
    )
    base_plan = DistributedGemmPlan(
        GemmProblem(1, 1024, 256),
        ClusterGrid(),
        unit,
        base_local,
    )
    direct = model_distributed_gemm(base_plan)
    interleaved = model_distributed_gemm(
        replace(
            base_plan,
            local=replace(
                base_local,
                bank_options=replace(
                    base_options, connectivity="interleaved"
                ),
            ),
            allocation_id="interleaved",
        )
    )
    assert direct.input_stage.bank_seconds != interleaved.input_stage.bank_seconds

    tiny_local = LocalModelConfig(
        GemmTiling(1, 1, 1, ctas_per_wave=1),
        bank_options=replace(base_options, apply_littles_law=True),
        compute_cycles_per_k=1,
    )
    limited = model_distributed_gemm(
        DistributedGemmPlan(
            GemmProblem(1, 1, 1),
            ClusterGrid(),
            _unit(1),
            tiny_local,
            allocation_id="ll",
        )
    )
    assert limited.input_stage.owner_bank_results[0].little_law_limited


def test_generic_can_overlap_streamed_input_with_compute():
    base = DistributedGemmPlan(
        GemmProblem(4, 8, 8),
        ClusterGrid(1, 4, 1),
        _unit(4),
        _local(compute_cycles=100),
    )
    overlap = model_distributed_gemm(base)
    serial = model_distributed_gemm(
        replace(base, overlap_input_compute=False, allocation_id="serial")
    )
    assert overlap.input_compute_seconds == max(
        overlap.input_stage.total_seconds, overlap.compute_seconds
    )
    assert overlap.total_seconds <= serial.total_seconds


def test_generic_aggregate_overlap_covers_input_bound_and_launch_semantics():
    launch_seconds = 11e-9
    local = replace(
        _local(compute_cycles=0.01), kernel_launch_seconds=launch_seconds
    )
    base = DistributedGemmPlan(
        GemmProblem(4, 64, 64),
        ClusterGrid(1, 4, 1),
        _unit(4),
        local,
    )
    overlap = model_distributed_gemm(base)
    serial = model_distributed_gemm(
        replace(base, overlap_input_compute=False, allocation_id="serial_input")
    )
    assert overlap.input_stage.total_seconds > overlap.compute_seconds
    assert overlap.kernel_launch_seconds == launch_seconds
    assert overlap.input_compute_seconds == pytest.approx(
        launch_seconds + overlap.input_stage.total_seconds
    )
    assert serial.input_compute_seconds == pytest.approx(
        launch_seconds
        + serial.input_stage.total_seconds
        + serial.compute_seconds
    )
    # Launch is a prerequisite and cannot disappear behind either stage.
    no_launch = model_distributed_gemm(
        replace(
            base,
            local=replace(local, kernel_launch_seconds=0.0),
            allocation_id="no_launch",
        )
    )
    assert overlap.total_seconds - no_launch.total_seconds == pytest.approx(
        launch_seconds
    )


def test_default_arch_path_includes_local_c_epilogue():
    arch = SimpleNamespace(
        sm_count=1,
        fp16_tensor_flops=1e12,
        fp32_tensor_flops=1e12,
        fp32_cuda_core_flops=1e12,
        compute_max_util=1.0,
        l2_bandwidth=1e9,
        smem_bandwidth=1e9,
        l2_max_util=1.0,
        l1_max_util=1.0,
    )
    inferred = LocalModelConfig(
        GemmTiling(2, 2, 2, ctas_per_wave=1),
        bank_options=GemmBankOptions(apply_littles_law=False),
        arch=arch,
    )
    plan = DistributedGemmPlan(
        GemmProblem(2, 2, 2), ClusterGrid(), _unit(1), inferred
    )
    with_epilogue = model_distributed_gemm(plan)
    without_epilogue = model_distributed_gemm(
        replace(
            plan,
            local=replace(inferred, store_other_memory_cycles=0.0),
            allocation_id="no_epilogue",
        )
    )
    assert (
        with_epilogue.local_results[0].seconds
        > without_epilogue.local_results[0].seconds
    )


def test_k_split_requires_explicit_partial_output_contract():
    with pytest.raises(ValueError, match="partial C"):
        DistributedGemmPlan(
            GemmProblem(4, 4, 8),
            ClusterGrid(1, 1, 4),
            _unit(4),
            _local(),
        )
    result = model_distributed_gemm(
        DistributedGemmPlan(
            GemmProblem(4, 4, 8),
            ClusterGrid(1, 1, 4),
            _unit(4),
            _local(),
            output_mode="partial",
        )
    )
    assert result.requires_reduction
    assert not result.end_to_end_complete


def test_batched_partial_c_has_disjoint_k_storage_and_accumulator_dtype():
    base = DistributedGemmPlan(
        GemmProblem(8, 8, 8, batch=2, c_dtype_bytes=2),
        ClusterGrid(1, 1, 2),
        _unit(2),
        _local(),
        output_mode="partial",
        partial_c_dtype_bytes=2,
    )
    fp16_partial = model_distributed_gemm(base)
    fp32_partial = model_distributed_gemm(
        replace(
            base,
            partial_c_dtype_bytes=4,
            allocation_id="partial_fp32",
        )
    )
    assert fp16_partial.total_ddr_write_bytes == 2 * 2 * 8 * 8 * 2
    assert fp32_partial.total_ddr_write_bytes == 2 * 2 * 8 * 8 * 4


def test_classic_cannon_rejects_eight_cluster_rectangular_grid():
    with pytest.raises(ValueError, match="perfect square"):
        CannonPlan(GemmProblem(8, 8, 8), _unit(8), _local())


def test_cannon_q1_is_a_zero_communication_single_cluster_degeneracy():
    result = model_cannon_gemm(
        CannonPlan(
            GemmProblem(7, 8, 9, batch=2),
            _unit(1),
            _local(),
            restore_initial_skew=True,
        )
    )
    assert len(result.rounds) == 1
    assert result.active_clusters == 1
    assert result.initial_skew.total_bytes == 0
    assert result.rounds[0].shift.total_bytes == 0
    assert result.total_noc_bytes == 0
    assert result.useful_flops == result.modeled_flops == 2016


@pytest.mark.parametrize(
    "broadcast_b, expected_skew, expected_shift",
    [(False, 648, 972), (True, 432, 648)],
)
def test_cannon_q3_batch_schedule_and_restore_traffic(
    broadcast_b, expected_skew, expected_shift
):
    result = model_cannon_gemm(
        CannonPlan(
            GemmProblem(
                9,
                9,
                9,
                batch=3,
                b_broadcast_across_batch=broadcast_b,
            ),
            _unit(9, l2_bytes=1 << 20),
            _local(tile=3),
            restore_initial_skew=True,
        )
    )
    assert result.initial_skew.total_bytes == expected_skew
    assert [round_.shift.total_bytes for round_ in result.rounds] == [
        expected_shift
    ] * 3
    assert result.useful_flops == result.modeled_flops == 4374
    for cluster in range(9):
        assert {
            round_.clusters[cluster].k_block for round_ in result.rounds
        } == {0, 1, 2}
    for round_ in result.rounds:
        for flow in round_.shift.flows:
            row, column = divmod(flow.source, 3)
            if flow.kind.endswith(":A"):
                assert flow.destination == row * 3 + (column - 1) % 3
            else:
                assert flow.kind.endswith(":B")
                assert flow.destination == ((row - 1) % 3) * 3 + column


def test_cannon_charges_the_local_c_epilogue_only_in_the_final_round():
    local = replace(_local(), store_other_memory_cycles=17.0)
    result = model_cannon_gemm(
        CannonPlan(GemmProblem(8, 8, 8), _unit(4), local)
    )
    first = result.rounds[0].clusters[0].local.pipeline.breakdowns
    final = result.rounds[-1].clusters[0].local.pipeline.breakdowns
    assert all(
        final_breakdown.epilogue_cycles
        - first_breakdown.epilogue_cycles
        == 17.0
        for first_breakdown, final_breakdown in zip(first, final)
    )


def test_generic_torus_routes_remote_traffic_on_its_logical_grid():
    unit = ClusterUnitSpec(
        4,
        l2_per_cluster=L2CacheSpec(64 * 1024),
        noc=NoCSpec(16e9, 2e-9, topology="torus_2d"),
    )
    result = model_distributed_gemm(
        DistributedGemmPlan(
            GemmProblem(2, 8, 4),
            ClusterGrid(1, 4, 1),
            unit,
            _local(),
            a_placement=TensorPlacement.canonical(0),
            b_placement=TensorPlacement.canonical(0),
        )
    )
    assert result.input_stage.noc.total_bytes > 0
    assert result.input_stage.noc.maximum_path_hops <= 2


def test_torus_does_not_invent_an_ideal_switch_fabric_bottleneck():
    noc = NoCSpec(
        1e9,
        topology="torus_2d",
        switch_fabric_bandwidth_bytes_s=1.0,
    )
    result = noc.estimate(
        (
            # One adjacent 100-B transfer should take 100 ns, not 100 s.
            TrafficFlow(0, 1, 100),
        ),
        4,
        grid_2d=(2, 2),
    )
    assert result.serialization_seconds == pytest.approx(100 / 1e9)


def test_cannon_schedule_ownership_flops_and_traffic_are_exact():
    problem = GemmProblem(8, 8, 8)
    result = model_cannon_gemm(CannonPlan(problem, _unit(4), _local()))

    assert len(result.rounds) == 2
    for cluster in range(4):
        seen = {
            round_.clusters[cluster].k_block for round_ in result.rounds
        }
        assert seen == {0, 1}
    assert result.useful_flops == result.modeled_flops == 2 * 8 * 8 * 8

    input_payload = (
        problem.m * problem.k * problem.a_dtype_bytes
        + problem.k * problem.n * problem.b_dtype_bytes
    )
    assert result.initial_skew.total_bytes == input_payload // 2
    assert result.rounds[0].shift.total_bytes == input_payload
    assert result.rounds[1].shift.total_bytes == 0


def test_cannon_schedule_reconstructs_a_ragged_matrix_product():
    problem = GemmProblem(5, 7, 9)
    result = model_cannon_gemm(CannonPlan(problem, _unit(4), _local()))
    rng = np.random.default_rng(7)
    a = rng.normal(size=(problem.m, problem.k))
    b = rng.normal(size=(problem.k, problem.n))
    reconstructed = np.zeros((problem.m, problem.n))

    for round_ in result.rounds:
        for cluster in round_.clusters:
            m_slice = split_extent(problem.m, 2, cluster.row)
            n_slice = split_extent(problem.n, 2, cluster.column)
            k_slice = split_extent(problem.k, 2, cluster.k_block)
            reconstructed[
                m_slice.start:m_slice.stop,
                n_slice.start:n_slice.stop,
            ] += a[
                m_slice.start:m_slice.stop,
                k_slice.start:k_slice.stop,
            ] @ b[
                k_slice.start:k_slice.stop,
                n_slice.start:n_slice.stop,
            ]

    np.testing.assert_allclose(reconstructed, a @ b)
    assert result.useful_flops == 2 * problem.m * problem.n * problem.k


def test_cannon_pre_skew_and_restore_are_explicit():
    base = CannonPlan(
        GemmProblem(8, 8, 8),
        _unit(4),
        _local(),
        initial_skew="pre_skewed",
        restore_initial_skew=True,
    )
    result = model_cannon_gemm(base)
    assert result.initial_skew.total_bytes == 0
    assert all(round_.shift.total_bytes > 0 for round_ in result.rounds)


def test_cannon_packed_signature_distinguishes_runtime_and_preskewed_homes():
    unit = _unit(4, l2_bytes=1 << 20)
    session = DistributedL2Session.homogeneous(4, unit.l2_per_cluster)
    runtime = CannonPlan(
        GemmProblem(8, 8, 8),
        unit,
        _local(),
        allocation_id="cannon_arrangement",
    )
    model_cannon_gemm(runtime, session=session)
    with pytest.raises(ValueError, match="multiple bank mappings"):
        model_cannon_gemm(
            replace(runtime, initial_skew="pre_skewed"),
            session=session,
        )


def test_cannon_shift_overlap_never_exceeds_serial_schedule():
    base = CannonPlan(
        GemmProblem(8, 8, 8),
        _unit(4),
        _local(compute_cycles=100),
    )
    overlap = model_cannon_gemm(base)
    serial = model_cannon_gemm(replace(base, overlap_shifts=False))
    assert overlap.total_seconds <= serial.total_seconds
    assert overlap.rounds[0].total_seconds == max(
        overlap.rounds[0].compute_seconds,
        overlap.rounds[0].shift.total_seconds,
    )


def test_cannon_kernel_launch_is_outside_shift_overlap():
    base_local = _local(compute_cycles=100)
    base = CannonPlan(GemmProblem(8, 8, 8), _unit(4), base_local)
    no_launch = model_cannon_gemm(base)
    launch_seconds = 11e-9
    launched = model_cannon_gemm(
        replace(
            base,
            local=replace(base_local, kernel_launch_seconds=launch_seconds),
            allocation_id="launched",
        )
    )
    assert launched.total_seconds - no_launch.total_seconds == pytest.approx(
        launch_seconds
    )


def test_cannon_remote_canonical_input_can_hit_owner_private_l2():
    unit = _unit(4, l2_bytes=256 * 1024)
    session = DistributedL2Session.homogeneous(4, unit.l2_per_cluster)
    plan = CannonPlan(
        GemmProblem(8, 8, 8),
        unit,
        _local(),
        a_placement=TensorPlacement.canonical(0),
        b_placement=TensorPlacement.canonical(0),
        allocation_id="canonical_cannon",
    )
    cold = model_cannon_gemm(plan, session=session)
    warm = model_cannon_gemm(plan, session=session)
    assert cold.total_ddr_read_bytes > 0
    assert warm.total_ddr_read_bytes == 0
    assert warm.remote_l2_hit_bytes > 0
    assert warm.initial_load.noc.total_bytes > 0


def test_cannon_gemv_uses_only_half_the_clusters_unless_m_is_padded():
    exact_plan = CannonPlan(GemmProblem(1, 8, 8), _unit(4), _local())
    exact = model_cannon_gemm(exact_plan)
    padded = model_cannon_gemm(replace(exact_plan, pad=True))

    assert exact.active_clusters == 2
    assert exact.padding_efficiency == 1.0
    assert padded.active_clusters == 4
    assert padded.padding_efficiency == pytest.approx(0.5)
