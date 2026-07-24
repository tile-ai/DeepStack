"""Ownership-, traffic-, cache-, and overlap-aware Cannon GEMM model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

from ..gemm import GemmLayoutSet, GemmProblem
from ..layout import MatrixAccess
from .cache import DistributedL2Session
from .common import (
    ClusterUnitSpec,
    LocalModelConfig,
    NoCResult,
    PlacementKind,
    TensorPlacement,
    TrafficFlow,
    ceil_div,
    split_extent,
)
from .local import (
    distributed_memory_outstanding_bytes,
    LocalPhaseResult,
    make_configured_layouts,
    make_packed_shard_layouts,
    model_onchip_phase,
    tensor_allocation_sector_interval,
    tensor_block_sectors,
    tensor_sectors,
)
from .memory import (
    MemoryReadStageResult,
    MemoryWriteStageResult,
    TensorReadRequest,
    TensorWriteRequest,
    model_read_stage,
    model_write_stage,
)


@dataclass(frozen=True)
class CannonPlan:
    problem: GemmProblem
    unit: ClusterUnitSpec
    local: LocalModelConfig
    a_placement: TensorPlacement = field(default_factory=TensorPlacement.sharded)
    b_placement: TensorPlacement = field(default_factory=TensorPlacement.sharded)
    c_placement: TensorPlacement = field(default_factory=TensorPlacement.sharded)
    initial_skew: str = "runtime"
    overlap_shifts: bool = True
    restore_initial_skew: bool = False
    pad: bool = False
    sharded_storage: str = "packed"
    allocation_base_sector: int = 0
    allocation_id: str = "cannon"

    def __post_init__(self) -> None:
        q = math.isqrt(self.unit.clusters_per_unit)
        if q * q != self.unit.clusters_per_unit:
            raise ValueError(
                "classic Cannon requires clusters_per_unit to be a perfect square"
            )
        if self.initial_skew not in {"runtime", "pre_skewed"}:
            raise ValueError("initial_skew must be runtime or pre_skewed")
        if self.sharded_storage not in {"packed", "global"}:
            raise ValueError("sharded_storage must be packed or global")
        if self.allocation_base_sector < 0:
            raise ValueError("allocation_base_sector must be non-negative")
        if not self.allocation_id:
            raise ValueError("allocation_id must be stable and non-empty")

    @property
    def q(self) -> int:
        return math.isqrt(self.unit.clusters_per_unit)


@dataclass(frozen=True)
class CannonClusterRound:
    cluster: int
    row: int
    column: int
    k_block: int
    local: LocalPhaseResult


@dataclass(frozen=True)
class CannonRoundResult:
    index: int
    clusters: Tuple[CannonClusterRound, ...]
    shift: NoCResult
    compute_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class CannonResult:
    plan: CannonPlan
    kernel_launch_seconds: float
    initial_load: MemoryReadStageResult
    initial_skew: NoCResult
    rounds: Tuple[CannonRoundResult, ...]
    output_store: MemoryWriteStageResult
    total_seconds: float
    useful_flops: int
    modeled_flops: int
    l2_resident_lines: Tuple[int, ...]

    @property
    def total_ddr_read_bytes(self) -> int:
        return self.initial_load.ddr_read_bytes

    @property
    def total_ddr_write_bytes(self) -> int:
        return self.output_store.ddr_write_bytes

    @property
    def remote_l2_hit_bytes(self) -> int:
        return self.initial_load.remote_l2_hit_bytes

    @property
    def total_noc_bytes(self) -> int:
        return (
            self.initial_load.noc.total_bytes
            + self.initial_skew.total_bytes
            + sum(round_.shift.total_bytes for round_ in self.rounds)
            + self.output_store.noc.total_bytes
        )

    @property
    def padding_efficiency(self) -> float:
        if self.modeled_flops == 0:
            return 1.0
        return self.useful_flops / self.modeled_flops

    @property
    def active_clusters(self) -> int:
        return len(
            {
                cluster.cluster
                for round_ in self.rounds
                for cluster in round_.clusters
                if cluster.local.problem is not None
            }
        )


def _rank(q: int, row: int, column: int) -> int:
    return row * q + column


def _physical_problem(plan: CannonPlan) -> GemmProblem:
    if not plan.pad:
        return plan.problem
    q = plan.q
    return GemmProblem(
        ceil_div(plan.problem.m, q) * q,
        ceil_div(plan.problem.n, q) * q,
        ceil_div(plan.problem.k, q) * q,
        batch=plan.problem.batch,
        a_dtype_bytes=plan.problem.a_dtype_bytes,
        b_dtype_bytes=plan.problem.b_dtype_bytes,
        c_dtype_bytes=plan.problem.c_dtype_bytes,
        b_broadcast_across_batch=plan.problem.b_broadcast_across_batch,
        compute_dtype_bytes=plan.problem.compute_dtype_bytes,
    )


def _input_requests(
    plan: CannonPlan,
    physical: GemmProblem,
) -> Tuple[List[TensorReadRequest], GemmLayoutSet, GemmLayoutSet]:
    q = plan.q
    layouts = make_configured_layouts(physical, plan.local, plan.unit.dram)
    maximum_shard = GemmProblem(
        ceil_div(physical.m, q),
        ceil_div(physical.n, q),
        ceil_div(physical.k, q),
        batch=physical.batch,
        a_dtype_bytes=physical.a_dtype_bytes,
        b_dtype_bytes=physical.b_dtype_bytes,
        c_dtype_bytes=physical.c_dtype_bytes,
        b_broadcast_across_batch=physical.b_broadcast_across_batch,
        compute_dtype_bytes=physical.compute_dtype_bytes,
    )
    packed_layouts = make_packed_shard_layouts(
        layouts, maximum_shard, plan.local, plan.unit.dram
    )
    a_interval = tensor_allocation_sector_interval(
        layouts,
        physical,
        "A",
        plan.unit.dram,
        sector_offset=plan.allocation_base_sector,
    )
    b_interval = tensor_allocation_sector_interval(
        layouts,
        physical,
        "B",
        plan.unit.dram,
        sector_offset=plan.allocation_base_sector,
    )
    distribution_signature = (plan.problem, q, plan.pad)
    requests: List[TensorReadRequest] = []
    for i in range(q):
        m_slice = split_extent(plan.problem.m, q, i, pad=plan.pad)
        for k_index in range(q):
            k_slice = split_extent(
                plan.problem.k, q, k_index, pad=plan.pad
            )
            if not m_slice.size or not k_slice.size:
                continue
            source = _rank(q, i, k_index)
            destination = _rank(q, i, (k_index - i) % q)
            requester = source if plan.initial_skew == "runtime" else destination
            shard_owner = source if plan.initial_skew == "runtime" else destination
            for batch in range(physical.batch):
                packed = (
                    plan.sharded_storage == "packed"
                    and plan.a_placement.kind == PlacementKind.SHARDED
                )
                if packed:
                    shape = (m_slice.size, k_slice.size)
                    sectors = tensor_block_sectors(
                        packed_layouts,
                        "A",
                        shape,
                        physical.a_dtype_bytes,
                        batch,
                        physical.batch,
                        plan.unit.dram,
                    )
                    mapping = (
                        "packed",
                        packed_layouts.a,
                        packed_layouts.a_batch_stride_bytes,
                        shape,
                        physical.a_dtype_bytes,
                        physical.batch,
                        distribution_signature,
                        physical.a_shape,
                        (
                            m_slice.start,
                            m_slice.stop,
                            k_slice.start,
                            k_slice.stop,
                        ),
                    )
                else:
                    sectors = tensor_sectors(
                        layouts,
                        physical,
                        "A",
                        MatrixAccess(
                            m_slice.start,
                            m_slice.stop,
                            k_slice.start,
                            k_slice.stop,
                        ),
                        batch,
                        plan.unit.dram,
                    )
                    mapping = (
                        "global",
                        layouts.a,
                        layouts.a_batch_stride_bytes,
                        physical.a_shape,
                        physical.a_dtype_bytes,
                        physical.batch,
                    )
                if plan.allocation_base_sector:
                    sectors = sectors + plan.allocation_base_sector
                mapping = mapping + (
                    "allocation_base_sector",
                    plan.allocation_base_sector,
                )
                requests.append(
                    TensorReadRequest(
                        requester,
                        shard_owner,
                        "A",
                        "%s:A" % plan.allocation_id,
                        sectors,
                        plan.a_placement,
                        layouts.a.swizzle,
                        mapping_signature=mapping,
                        address_interval=a_interval,
                    )
                )

    b_batches = 1 if physical.b_broadcast_across_batch else physical.batch
    for k_index in range(q):
        k_slice = split_extent(plan.problem.k, q, k_index, pad=plan.pad)
        for j in range(q):
            n_slice = split_extent(plan.problem.n, q, j, pad=plan.pad)
            if not k_slice.size or not n_slice.size:
                continue
            source = _rank(q, k_index, j)
            destination = _rank(q, (k_index - j) % q, j)
            requester = source if plan.initial_skew == "runtime" else destination
            shard_owner = source if plan.initial_skew == "runtime" else destination
            for batch in range(b_batches):
                packed = (
                    plan.sharded_storage == "packed"
                    and plan.b_placement.kind == PlacementKind.SHARDED
                )
                if packed:
                    shape = (k_slice.size, n_slice.size)
                    sectors = tensor_block_sectors(
                        packed_layouts,
                        "B",
                        shape,
                        physical.b_dtype_bytes,
                        batch,
                        physical.batch,
                        plan.unit.dram,
                        b_broadcast_across_batch=(
                            physical.b_broadcast_across_batch
                        ),
                    )
                    mapping = (
                        "packed",
                        packed_layouts.b,
                        packed_layouts.b_batch_stride_bytes,
                        shape,
                        physical.b_dtype_bytes,
                        physical.b_broadcast_across_batch,
                        b_batches,
                        distribution_signature,
                        physical.b_shape,
                        (
                            k_slice.start,
                            k_slice.stop,
                            n_slice.start,
                            n_slice.stop,
                        ),
                    )
                else:
                    sectors = tensor_sectors(
                        layouts,
                        physical,
                        "B",
                        MatrixAccess(
                            k_slice.start,
                            k_slice.stop,
                            n_slice.start,
                            n_slice.stop,
                        ),
                        batch,
                        plan.unit.dram,
                    )
                    mapping = (
                        "global",
                        layouts.b,
                        layouts.b_batch_stride_bytes,
                        physical.b_shape,
                        physical.b_dtype_bytes,
                        physical.b_broadcast_across_batch,
                        b_batches,
                    )
                if plan.allocation_base_sector:
                    sectors = sectors + plan.allocation_base_sector
                mapping = mapping + (
                    "allocation_base_sector",
                    plan.allocation_base_sector,
                )
                requests.append(
                    TensorReadRequest(
                        requester,
                        shard_owner,
                        "B",
                        "%s:B" % plan.allocation_id,
                        sectors,
                        plan.b_placement,
                        layouts.b.swizzle,
                        mapping_signature=mapping,
                        address_interval=b_interval,
                    )
                )
    return requests, layouts, packed_layouts


def _initial_skew_flows(plan: CannonPlan) -> Tuple[TrafficFlow, ...]:
    if plan.initial_skew == "pre_skewed":
        return ()
    q = plan.q
    flows = []
    for i in range(q):
        m_slice = split_extent(plan.problem.m, q, i, pad=plan.pad)
        for k_index in range(q):
            k_slice = split_extent(plan.problem.k, q, k_index, pad=plan.pad)
            nbytes = (
                m_slice.size
                * k_slice.size
                * plan.problem.a_dtype_bytes
                * plan.problem.batch
            )
            flows.append(
                TrafficFlow(
                    _rank(q, i, k_index),
                    _rank(q, i, (k_index - i) % q),
                    nbytes,
                    "cannon:initial_skew:A",
                )
            )
    b_batches = 1 if plan.problem.b_broadcast_across_batch else plan.problem.batch
    for k_index in range(q):
        k_slice = split_extent(plan.problem.k, q, k_index, pad=plan.pad)
        for j in range(q):
            n_slice = split_extent(plan.problem.n, q, j, pad=plan.pad)
            nbytes = (
                k_slice.size
                * n_slice.size
                * plan.problem.b_dtype_bytes
                * b_batches
            )
            flows.append(
                TrafficFlow(
                    _rank(q, k_index, j),
                    _rank(q, (k_index - j) % q, j),
                    nbytes,
                    "cannon:initial_skew:B",
                )
            )
    return tuple(flows)


def _round_shift_flows(plan: CannonPlan, round_index: int) -> Tuple[TrafficFlow, ...]:
    q = plan.q
    flows = []
    b_batches = 1 if plan.problem.b_broadcast_across_batch else plan.problem.batch
    for i in range(q):
        m_slice = split_extent(plan.problem.m, q, i, pad=plan.pad)
        for j in range(q):
            k_index = (i + j + round_index) % q
            k_slice = split_extent(plan.problem.k, q, k_index, pad=plan.pad)
            n_slice = split_extent(plan.problem.n, q, j, pad=plan.pad)
            rank = _rank(q, i, j)
            flows.extend(
                (
                    TrafficFlow(
                        rank,
                        _rank(q, i, (j - 1) % q),
                        m_slice.size
                        * k_slice.size
                        * plan.problem.a_dtype_bytes
                        * plan.problem.batch,
                        "cannon:shift:A",
                        round_index,
                    ),
                    TrafficFlow(
                        rank,
                        _rank(q, (i - 1) % q, j),
                        k_slice.size
                        * n_slice.size
                        * plan.problem.b_dtype_bytes
                        * b_batches,
                        "cannon:shift:B",
                        round_index,
                    ),
                )
            )
    return tuple(flows)


def _output_requests(
    plan: CannonPlan,
    physical: GemmProblem,
    layouts: GemmLayoutSet,
    packed_layouts: GemmLayoutSet,
) -> List[TensorWriteRequest]:
    q = plan.q
    requests = []
    c_interval = tensor_allocation_sector_interval(
        layouts,
        physical,
        "C",
        plan.unit.dram,
        sector_offset=plan.allocation_base_sector,
    )
    for i in range(q):
        m_slice = split_extent(plan.problem.m, q, i, pad=plan.pad)
        for j in range(q):
            n_slice = split_extent(plan.problem.n, q, j, pad=plan.pad)
            if not m_slice.size or not n_slice.size:
                continue
            rank = _rank(q, i, j)
            for batch in range(physical.batch):
                packed = (
                    plan.sharded_storage == "packed"
                    and plan.c_placement.kind == PlacementKind.SHARDED
                )
                if packed:
                    shape = (m_slice.size, n_slice.size)
                    sectors = tensor_block_sectors(
                        packed_layouts,
                        "C",
                        shape,
                        physical.c_dtype_bytes,
                        batch,
                        physical.batch,
                        plan.unit.dram,
                    )
                    mapping = (
                        "packed",
                        packed_layouts.c,
                        packed_layouts.c_batch_stride_bytes,
                        shape,
                        physical.c_dtype_bytes,
                        physical.batch,
                        (plan.problem, q, plan.pad),
                        physical.c_shape,
                        (
                            m_slice.start,
                            m_slice.stop,
                            n_slice.start,
                            n_slice.stop,
                        ),
                    )
                else:
                    sectors = tensor_sectors(
                        layouts,
                        physical,
                        "C",
                        MatrixAccess(
                            m_slice.start,
                            m_slice.stop,
                            n_slice.start,
                            n_slice.stop,
                        ),
                        batch,
                        plan.unit.dram,
                    )
                    mapping = (
                        "global",
                        layouts.c,
                        layouts.c_batch_stride_bytes,
                        physical.c_shape,
                        physical.c_dtype_bytes,
                        physical.batch,
                    )
                if plan.allocation_base_sector:
                    sectors = sectors + plan.allocation_base_sector
                mapping = mapping + (
                    "allocation_base_sector",
                    plan.allocation_base_sector,
                )
                requests.append(
                    TensorWriteRequest(
                        rank,
                        rank,
                        "C",
                        "%s:C" % plan.allocation_id,
                        sectors,
                        plan.c_placement,
                        layouts.c.swizzle,
                        mapping_signature=mapping,
                        address_interval=c_interval,
                    )
                )
    return requests


def model_cannon_gemm(
    plan: CannonPlan,
    *,
    session: Optional[DistributedL2Session] = None,
) -> CannonResult:
    """Model classic square-grid Cannon with optional shift/compute overlap."""

    if session is None:
        session = DistributedL2Session.homogeneous(
            plan.unit.clusters_per_unit,
            plan.unit.l2_per_cluster,
        )
    q = plan.q
    grid_2d = (q, q)
    physical = _physical_problem(plan)
    input_requests, layouts, packed_layouts = _input_requests(plan, physical)
    bank_options = plan.local.bank_options
    outstanding = distributed_memory_outstanding_bytes(physical, plan.local)
    initial_load = model_read_stage(
        input_requests,
        plan.unit,
        session,
        grid_2d=grid_2d,
        connectivity=bank_options.connectivity,
        coalesce_scope=bank_options.coalesce_scope,
        apply_littles_law=bank_options.apply_littles_law,
        outstanding_bytes=outstanding,
    )
    initial_skew = plan.unit.noc.estimate(
        _initial_skew_flows(plan),
        plan.unit.clusters_per_unit,
        grid_2d=grid_2d,
    )

    rounds = []
    useful_flops = 0
    modeled_flops = 0
    for round_index in range(q):
        clusters = []
        for i in range(q):
            m_slice = split_extent(plan.problem.m, q, i, pad=plan.pad)
            for j in range(q):
                n_slice = split_extent(plan.problem.n, q, j, pad=plan.pad)
                k_index = (i + j + round_index) % q
                k_slice = split_extent(
                    plan.problem.k, q, k_index, pad=plan.pad
                )
                rank = _rank(q, i, j)
                local_problem = None
                if min(m_slice.size, n_slice.size, k_slice.size) > 0:
                    local_problem = GemmProblem(
                        m_slice.size,
                        n_slice.size,
                        k_slice.size,
                        batch=plan.problem.batch,
                        a_dtype_bytes=plan.problem.a_dtype_bytes,
                        b_dtype_bytes=plan.problem.b_dtype_bytes,
                        c_dtype_bytes=plan.problem.c_dtype_bytes,
                        b_broadcast_across_batch=plan.problem.b_broadcast_across_batch,
                        compute_dtype_bytes=plan.problem.compute_dtype_bytes,
                    )
                local_useful_flops = (
                    2
                    * m_slice.logical_size
                    * n_slice.logical_size
                    * k_slice.logical_size
                    * plan.problem.batch
                )
                local = model_onchip_phase(
                    rank,
                    local_problem,
                    plan.local,
                    plan.unit.dram,
                    useful_flops=local_useful_flops,
                    # One fused Cannon kernel launch is charged outside the
                    # compute/shift max.  A launch is a prerequisite, not DMA
                    # work that can disappear behind the first shift.
                    charge_launch=False,
                    include_store_epilogue=round_index == q - 1,
                )
                useful_flops += local.useful_flops
                modeled_flops += local.modeled_flops
                clusters.append(
                    CannonClusterRound(rank, i, j, k_index, local)
                )

        has_shift = round_index < q - 1 or plan.restore_initial_skew
        shift = plan.unit.noc.estimate(
            _round_shift_flows(plan, round_index) if has_shift else (),
            plan.unit.clusters_per_unit,
            grid_2d=grid_2d,
        )
        compute_seconds = max(
            (cluster.local.seconds for cluster in clusters), default=0.0
        )
        round_seconds = (
            max(compute_seconds, shift.total_seconds)
            if plan.overlap_shifts
            else compute_seconds + shift.total_seconds
        )
        rounds.append(
            CannonRoundResult(
                round_index,
                tuple(clusters),
                shift,
                compute_seconds,
                round_seconds,
            )
        )

    output_store = model_write_stage(
        _output_requests(plan, physical, layouts, packed_layouts),
        plan.unit,
        session=session,
        grid_2d=grid_2d,
        connectivity=bank_options.connectivity,
        coalesce_scope=bank_options.coalesce_scope,
        apply_littles_law=bank_options.apply_littles_law,
        outstanding_bytes=outstanding,
    )
    total_seconds = (
        plan.local.kernel_launch_seconds
        + initial_load.total_seconds
        + initial_skew.total_seconds
        + sum(round_.total_seconds for round_ in rounds)
        + output_store.total_seconds
    )
    return CannonResult(
        plan=plan,
        kernel_launch_seconds=plan.local.kernel_launch_seconds,
        initial_load=initial_load,
        initial_skew=initial_skew,
        rounds=tuple(rounds),
        output_store=output_store,
        total_seconds=total_seconds,
        useful_flops=useful_flops,
        modeled_flops=modeled_flops,
        l2_resident_lines=session.resident_lines,
    )
