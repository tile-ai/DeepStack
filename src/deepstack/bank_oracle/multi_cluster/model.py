"""Aggregate distributed GEMM model over a configurable cluster unit."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

from ..gemm import GemmProblem
from ..layout import MatrixAccess
from .cache import DistributedL2Session
from .common import (
    ClusterGrid,
    ClusterUnitSpec,
    LocalModelConfig,
    PlacementKind,
    TensorPlacement,
    ceil_div,
    effective_problem,
    split_extent,
)
from .local import (
    LocalPhaseResult,
    distributed_memory_outstanding_bytes,
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
class DistributedGemmPlan:
    """One global GEMM mapped onto an ``M x N x K`` cluster grid.

    The outer memory phase and local pipeline are aggregate statistical
    stages.  They can form a perfect-overlap bound or a serial prefetch bound;
    this is not a per-transaction execution simulator.
    """

    problem: GemmProblem
    grid: ClusterGrid
    unit: ClusterUnitSpec
    local: LocalModelConfig
    a_placement: TensorPlacement = field(default_factory=TensorPlacement.sharded)
    b_placement: TensorPlacement = field(default_factory=TensorPlacement.sharded)
    c_placement: TensorPlacement = field(default_factory=TensorPlacement.sharded)
    pad: bool = False
    output_mode: str = "final"
    partial_c_dtype_bytes: Optional[int] = None
    overlap_input_compute: bool = True
    sharded_storage: str = "packed"
    allocation_base_sector: int = 0
    allocation_id: str = "gemm"

    def __post_init__(self) -> None:
        if self.grid.size != self.unit.clusters_per_unit:
            raise ValueError("cluster grid does not match clusters_per_unit")
        if self.output_mode not in {"final", "partial"}:
            raise ValueError("output_mode must be final or partial")
        if self.grid.k > 1 and self.output_mode != "partial":
            raise ValueError(
                "K-sharded GEMM produces partial C; request partial output "
                "or use an algorithm with an explicit reduction"
            )
        if self.grid.k == 1 and self.output_mode == "partial":
            raise ValueError("partial output is meaningful only for a K split")
        if (
            self.partial_c_dtype_bytes is not None
            and self.partial_c_dtype_bytes <= 0
        ):
            raise ValueError("partial C dtype width must be positive")
        if self.grid.k == 1 and self.partial_c_dtype_bytes is not None:
            raise ValueError("partial C dtype applies only to a K split")
        if self.sharded_storage not in {"packed", "global"}:
            raise ValueError("sharded_storage must be packed or global")
        if self.allocation_base_sector < 0:
            raise ValueError("allocation_base_sector must be non-negative")
        if not self.allocation_id:
            raise ValueError("allocation_id must be stable and non-empty")


@dataclass(frozen=True)
class DistributedGemmResult:
    plan: DistributedGemmPlan
    local_results: Tuple[LocalPhaseResult, ...]
    input_stage: MemoryReadStageResult
    output_stage: MemoryWriteStageResult
    kernel_launch_seconds: float
    compute_seconds: float
    input_compute_seconds: float
    total_seconds: float
    useful_flops: int
    modeled_flops: int
    requires_reduction: bool
    l2_resident_lines: Tuple[int, ...]

    @property
    def total_ddr_read_bytes(self) -> int:
        return self.input_stage.ddr_read_bytes

    @property
    def total_ddr_write_bytes(self) -> int:
        return self.output_stage.ddr_write_bytes

    @property
    def total_noc_bytes(self) -> int:
        return self.input_stage.noc.total_bytes + self.output_stage.noc.total_bytes

    @property
    def remote_l2_hit_bytes(self) -> int:
        return self.input_stage.remote_l2_hit_bytes

    @property
    def padding_efficiency(self) -> float:
        if self.modeled_flops == 0:
            return 1.0
        return self.useful_flops / self.modeled_flops

    @property
    def end_to_end_complete(self) -> bool:
        """False when ``total_seconds`` ends at unreduced partial C."""

        return not self.requires_reduction


def model_distributed_gemm(
    plan: DistributedGemmPlan,
    *,
    session: Optional[DistributedL2Session] = None,
) -> DistributedGemmResult:
    """Model one distributed GEMM bound and update private L2 state."""

    if session is None:
        session = DistributedL2Session.homogeneous(
            plan.unit.clusters_per_unit,
            plan.unit.l2_per_cluster,
        )
    physical = effective_problem(plan.problem, plan.grid, plan.pad)
    layouts = make_configured_layouts(physical, plan.local, plan.unit.dram)
    partial_dtype = (
        plan.partial_c_dtype_bytes
        or plan.problem.compute_dtype_bytes
        or plan.problem.c_dtype_bytes
    )
    partial_physical = (
        replace(physical, c_dtype_bytes=partial_dtype)
        if plan.grid.k > 1
        else physical
    )
    partial_layouts = (
        make_configured_layouts(
            partial_physical, plan.local, plan.unit.dram
        )
        if plan.grid.k > 1
        else layouts
    )
    maximum_shard = GemmProblem(
        ceil_div(physical.m, plan.grid.m),
        ceil_div(physical.n, plan.grid.n),
        ceil_div(physical.k, plan.grid.k),
        batch=physical.batch,
        a_dtype_bytes=physical.a_dtype_bytes,
        b_dtype_bytes=physical.b_dtype_bytes,
        c_dtype_bytes=physical.c_dtype_bytes,
        b_broadcast_across_batch=physical.b_broadcast_across_batch,
        compute_dtype_bytes=physical.compute_dtype_bytes,
    )
    maximum_partial_shard = (
        replace(maximum_shard, c_dtype_bytes=partial_dtype)
        if plan.grid.k > 1
        else maximum_shard
    )
    packed_layouts = make_packed_shard_layouts(
        layouts, maximum_shard, plan.local, plan.unit.dram
    )
    packed_partial_layouts = (
        make_packed_shard_layouts(
            partial_layouts,
            maximum_partial_shard,
            plan.local,
            plan.unit.dram,
        )
        if plan.grid.k > 1
        else packed_layouts
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
    input_requests: List[TensorReadRequest] = []
    output_requests: List[TensorWriteRequest] = []
    local_results: List[LocalPhaseResult] = []
    distribution_signature = (plan.problem, plan.grid, plan.pad)

    # Partial C copies need disjoint owner-local addresses if a canonical
    # output placement sends several K ranks to one physical cluster.
    c_all_batches_bytes = (
        partial_layouts.c_batch_stride_bytes * partial_physical.batch
    )
    c_storage_lines = (
        c_all_batches_bytes
        + plan.unit.dram.sector_bytes
        - 1
    ) // plan.unit.dram.sector_bytes
    alignment_lines = plan.unit.dram.full_bank_wave_bytes // plan.unit.dram.sector_bytes
    partial_stride = (
        (c_storage_lines + alignment_lines - 1) // alignment_lines
    ) * alignment_lines

    for rank in range(plan.grid.size):
        m_index, n_index, k_index = plan.grid.coordinate(rank)
        m_slice = split_extent(plan.problem.m, plan.grid.m, m_index, pad=plan.pad)
        n_slice = split_extent(plan.problem.n, plan.grid.n, n_index, pad=plan.pad)
        k_slice = split_extent(plan.problem.k, plan.grid.k, k_index, pad=plan.pad)

        local_problem = None
        if min(m_slice.size, n_slice.size, k_slice.size) > 0:
            local_problem = GemmProblem(
                m_slice.size,
                n_slice.size,
                k_slice.size,
                batch=plan.problem.batch,
                a_dtype_bytes=plan.problem.a_dtype_bytes,
                b_dtype_bytes=plan.problem.b_dtype_bytes,
                c_dtype_bytes=(
                    partial_dtype
                    if plan.grid.k > 1
                    else plan.problem.c_dtype_bytes
                ),
                b_broadcast_across_batch=plan.problem.b_broadcast_across_batch,
                compute_dtype_bytes=plan.problem.compute_dtype_bytes,
            )
        useful_flops = (
            2
            * m_slice.logical_size
            * n_slice.logical_size
            * k_slice.logical_size
            * plan.problem.batch
        )
        local_results.append(
            model_onchip_phase(
                rank,
                local_problem,
                plan.local,
                plan.unit.dram,
                useful_flops=useful_flops,
                charge_launch=False,
            )
        )
        if local_problem is None:
            continue

        a_owner = plan.grid.rank(m_index, 0, k_index)
        b_owner = plan.grid.rank(0, n_index, k_index)
        for batch in range(physical.batch):
            packed_a = (
                plan.sharded_storage == "packed"
                and plan.a_placement.kind == PlacementKind.SHARDED
            )
            if packed_a:
                a_sectors = tensor_block_sectors(
                    packed_layouts,
                    "A",
                    local_problem.a_shape,
                    local_problem.a_dtype_bytes,
                    batch,
                    local_problem.batch,
                    plan.unit.dram,
                )
                a_mapping = (
                    "packed",
                    packed_layouts.a,
                    packed_layouts.a_batch_stride_bytes,
                    local_problem.a_shape,
                    local_problem.a_dtype_bytes,
                    local_problem.batch,
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
                a_sectors = tensor_sectors(
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
                a_mapping = (
                    "global",
                    layouts.a,
                    layouts.a_batch_stride_bytes,
                    physical.a_shape,
                    physical.a_dtype_bytes,
                    physical.batch,
                )
            if plan.allocation_base_sector:
                a_sectors = a_sectors + plan.allocation_base_sector
            a_mapping = a_mapping + (
                "allocation_base_sector",
                plan.allocation_base_sector,
            )

            packed_b = (
                plan.sharded_storage == "packed"
                and plan.b_placement.kind == PlacementKind.SHARDED
            )
            if packed_b:
                b_sectors = tensor_block_sectors(
                    packed_layouts,
                    "B",
                    local_problem.b_shape,
                    local_problem.b_dtype_bytes,
                    batch,
                    local_problem.batch,
                    plan.unit.dram,
                    b_broadcast_across_batch=(
                        local_problem.b_broadcast_across_batch
                    ),
                )
                b_mapping = (
                    "packed",
                    packed_layouts.b,
                    packed_layouts.b_batch_stride_bytes,
                    local_problem.b_shape,
                    local_problem.b_dtype_bytes,
                    local_problem.b_broadcast_across_batch,
                    (
                        1
                        if local_problem.b_broadcast_across_batch
                        else local_problem.batch
                    ),
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
                b_sectors = tensor_sectors(
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
                b_mapping = (
                    "global",
                    layouts.b,
                    layouts.b_batch_stride_bytes,
                    physical.b_shape,
                    physical.b_dtype_bytes,
                    physical.b_broadcast_across_batch,
                    1 if physical.b_broadcast_across_batch else physical.batch,
                )
            if plan.allocation_base_sector:
                b_sectors = b_sectors + plan.allocation_base_sector
            b_mapping = b_mapping + (
                "allocation_base_sector",
                plan.allocation_base_sector,
            )
            input_requests.extend(
                (
                    TensorReadRequest(
                        rank,
                        a_owner,
                        "A",
                        "%s:A" % plan.allocation_id,
                        a_sectors,
                        plan.a_placement,
                        layouts.a.swizzle,
                        mapping_signature=a_mapping,
                        address_interval=a_interval,
                    ),
                    TensorReadRequest(
                        rank,
                        b_owner,
                        "B",
                        "%s:B" % plan.allocation_id,
                        b_sectors,
                        plan.b_placement,
                        layouts.b.swizzle,
                        mapping_signature=b_mapping,
                        address_interval=b_interval,
                    ),
                )
            )

            packed_c = (
                plan.sharded_storage == "packed"
                and plan.c_placement.kind == PlacementKind.SHARDED
            )
            if packed_c:
                c_sectors = tensor_block_sectors(
                    packed_partial_layouts,
                    "C",
                    local_problem.c_shape,
                    local_problem.c_dtype_bytes,
                    batch,
                    local_problem.batch,
                    plan.unit.dram,
                )
                c_mapping = (
                    "packed",
                    packed_partial_layouts.c,
                    packed_partial_layouts.c_batch_stride_bytes,
                    local_problem.c_shape,
                    local_problem.c_dtype_bytes,
                    local_problem.batch,
                    distribution_signature,
                    partial_physical.c_shape,
                    (
                        m_slice.start,
                        m_slice.stop,
                        n_slice.start,
                        n_slice.stop,
                    ),
                )
            else:
                c_sectors = tensor_sectors(
                    partial_layouts,
                    partial_physical,
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
                c_mapping = (
                    "global",
                    partial_layouts.c,
                    partial_layouts.c_batch_stride_bytes,
                    partial_physical.c_shape,
                    partial_physical.c_dtype_bytes,
                    partial_physical.batch,
                )
            c_sector_offset = plan.allocation_base_sector
            if plan.grid.k > 1:
                c_sector_offset += k_index * partial_stride
                c_mapping = c_mapping + (
                    "partial_slot",
                    k_index * partial_stride,
                )
                c_tensor = "C_partial_%d" % k_index
                c_allocation = "%s:C_partial_%d" % (
                    plan.allocation_id,
                    k_index,
                )
                c_owner = rank
            else:
                c_tensor = "C"
                c_allocation = "%s:C" % plan.allocation_id
                c_owner = plan.grid.rank(m_index, n_index, 0)
            if c_sector_offset:
                c_sectors = c_sectors + c_sector_offset
            c_mapping = c_mapping + (
                "allocation_base_sector",
                plan.allocation_base_sector,
            )
            c_interval = tensor_allocation_sector_interval(
                partial_layouts,
                partial_physical,
                "C",
                plan.unit.dram,
                sector_offset=(
                    plan.allocation_base_sector
                    + (k_index * partial_stride if plan.grid.k > 1 else 0)
                ),
            )
            output_requests.append(
                TensorWriteRequest(
                    rank,
                    c_owner,
                    c_tensor,
                    c_allocation,
                    c_sectors,
                    plan.c_placement,
                    partial_layouts.c.swizzle,
                    mapping_signature=c_mapping,
                    address_interval=c_interval,
                )
            )

    bank_options = plan.local.bank_options
    outstanding = distributed_memory_outstanding_bytes(physical, plan.local)
    grid_2d = (plan.grid.m, plan.grid.n * plan.grid.k)
    input_stage = model_read_stage(
        input_requests,
        plan.unit,
        session,
        grid_2d=grid_2d,
        connectivity=bank_options.connectivity,
        coalesce_scope=bank_options.coalesce_scope,
        apply_littles_law=bank_options.apply_littles_law,
        outstanding_bytes=outstanding,
    )
    output_stage = model_write_stage(
        output_requests,
        plan.unit,
        session=session,
        grid_2d=grid_2d,
        connectivity=bank_options.connectivity,
        coalesce_scope=bank_options.coalesce_scope,
        apply_littles_law=bank_options.apply_littles_law,
        outstanding_bytes=outstanding,
    )
    compute_seconds = max(
        (result.seconds for result in local_results), default=0.0
    )
    useful_flops = sum(result.useful_flops for result in local_results)
    modeled_flops = sum(result.modeled_flops for result in local_results)
    input_compute_seconds = plan.local.kernel_launch_seconds + (
        max(input_stage.total_seconds, compute_seconds)
        if plan.overlap_input_compute
        else input_stage.total_seconds + compute_seconds
    )
    total_seconds = input_compute_seconds + output_stage.total_seconds
    return DistributedGemmResult(
        plan=plan,
        local_results=tuple(local_results),
        input_stage=input_stage,
        output_stage=output_stage,
        kernel_launch_seconds=plan.local.kernel_launch_seconds,
        compute_seconds=compute_seconds,
        input_compute_seconds=input_compute_seconds,
        total_seconds=total_seconds,
        useful_flops=useful_flops,
        modeled_flops=modeled_flops,
        requires_reduction=plan.grid.k > 1,
        l2_resident_lines=session.resident_lines,
    )
