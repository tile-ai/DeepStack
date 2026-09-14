"""Home-L2/owner-DRAM stages shared by generic and Cannon models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..engine import (
    BankServiceResult,
    apply_littles_law_cycles,
    service_request_groups,
)
from ..layout import BankSwizzle
from .cache import (
    BulkLineRequest,
    DistributedL2Stats,
    DistributedL2Session,
)
from .common import (
    ClusterUnitSpec,
    NoCResult,
    TensorPlacement,
    TrafficFlow,
)


@dataclass(frozen=True)
class TensorReadRequest:
    requester: int
    shard_owner: int
    tensor: str
    allocation_id: str
    sectors: Sequence[int]
    placement: TensorPlacement
    swizzle: BankSwizzle = BankSwizzle()
    epoch: int = 0
    mapping_signature: object = None
    address_interval: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class TensorWriteRequest:
    requester: int
    shard_owner: int
    tensor: str
    allocation_id: str
    sectors: Sequence[int]
    placement: TensorPlacement
    swizzle: BankSwizzle = BankSwizzle()
    phase: int = 0
    mapping_signature: object = None
    address_interval: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class OwnerBankAccessResult:
    owner: int
    service: BankServiceResult
    final_cycles: float
    final_seconds: float
    little_law_cycles: float
    little_law_limited: bool


@dataclass(frozen=True)
class MemoryReadStageResult:
    l2: DistributedL2Stats
    owner_bank_results: Tuple[OwnerBankAccessResult, ...]
    noc: NoCResult
    bank_seconds: float
    l2_seconds: float
    source_seconds: float
    total_seconds: float

    @property
    def ddr_read_bytes(self) -> int:
        return self.l2.ddr_bytes

    @property
    def remote_l2_hit_bytes(self) -> int:
        return self.l2.remote_owner_l2_hit_bytes


@dataclass(frozen=True)
class MemoryWriteStageResult:
    owner_bank_results: Tuple[OwnerBankAccessResult, ...]
    noc: NoCResult
    written_lines: int
    line_bytes: int
    bank_seconds: float
    total_seconds: float

    @property
    def ddr_write_bytes(self) -> int:
        return self.written_lines * self.line_bytes


def _validate_stage_contract(
    requests: Sequence[object],
    unit: ClusterUnitSpec,
    *,
    grid_2d: Optional[Tuple[int, int]],
    connectivity: str,
    coalesce_scope: str,
) -> None:
    """Reject decoder/layout errors before a persistent cache can mutate."""

    if connectivity not in {"direct", "interleaved"}:
        raise ValueError("connectivity must be direct or interleaved")
    if coalesce_scope not in {"request", "wave"}:
        raise ValueError("coalesce_scope must be request or wave")
    if unit.noc.topology == "torus_2d" and (
        grid_2d is None
        or grid_2d[0] <= 0
        or grid_2d[1] <= 0
        or grid_2d[0] * grid_2d[1] != unit.clusters_per_unit
    ):
        raise ValueError("torus_2d requires a matching logical grid")
    for request in requests:
        request.swizzle.validate(unit.dram)


def _dram_mapping_signature(
    unit: ClusterUnitSpec,
    connectivity: str,
) -> Tuple[object, ...]:
    """Fields that change sector-to-(bank,row,sector) decoding.

    Transfer/recharge clocks and connected port count affect timing but not
    cache identity, so warm sessions remain usable for pure timing sweeps.
    """

    return (
        connectivity,
        unit.dram.total_layers,
        unit.dram.banks_per_layer,
        unit.dram.row_bytes,
        unit.dram.sector_bytes,
    )


def _finalize_bank_service(
    owner: int,
    service: BankServiceResult,
    unit: ClusterUnitSpec,
    *,
    apply_littles_law: bool,
    outstanding_bytes: Optional[int],
) -> OwnerBankAccessResult:
    final_cycles = float(service.cycles)
    limited = False
    ll_cycles = 0.0
    outstanding = (
        unit.outstanding_bytes_per_cluster
        if unit.outstanding_bytes_per_cluster is not None
        else outstanding_bytes
    )
    if apply_littles_law and outstanding is not None:
        final_cycles, limited, ll_cycles = apply_littles_law_cycles(
            service.cycles,
            service.transferred_bytes,
            outstanding,
            unit.dram,
        )
    return OwnerBankAccessResult(
        owner=owner,
        service=service,
        final_cycles=final_cycles,
        final_seconds=unit.dram.cycles_to_seconds(final_cycles),
        little_law_cycles=ll_cycles,
        little_law_limited=limited,
    )


def _service_grouped_sectors(
    grouped: Dict[Tuple[object, ...], Sequence[int]],
    swizzles: Dict[str, BankSwizzle],
    unit: ClusterUnitSpec,
    *,
    connectivity: str,
    coalesce_scope: str,
    apply_littles_law: bool,
    outstanding_bytes: Optional[int],
) -> Tuple[OwnerBankAccessResult, ...]:
    by_owner: Dict[int, Dict[str, List[np.ndarray]]] = {}
    for key, sectors in grouped.items():
        owner = int(key[0])
        tensor = str(key[1])
        values = np.asarray(sectors, dtype=np.int64).reshape(-1)
        if values.size:
            unique = np.unique(values)
            by_owner.setdefault(owner, {}).setdefault(tensor, []).append(unique)

    results = []
    for owner in sorted(by_owner):
        request_groups = []
        for tensor, sector_requests in sorted(by_owner[owner].items()):
            request_groups.append(
                (tuple(sector_requests), swizzles[tensor])
            )
        service = service_request_groups(
            tuple(request_groups),
            unit.dram,
            connectivity=connectivity,
            coalesce_scope=coalesce_scope,
        )
        results.append(
            _finalize_bank_service(
                owner,
                service,
                unit,
                apply_littles_law=apply_littles_law,
                outstanding_bytes=outstanding_bytes,
            )
        )
    return tuple(results)


def model_read_stage(
    requests: Sequence[TensorReadRequest],
    unit: ClusterUnitSpec,
    session: DistributedL2Session,
    *,
    grid_2d: Optional[Tuple[int, int]] = None,
    connectivity: str = "direct",
    coalesce_scope: str = "wave",
    apply_littles_law: bool = True,
    outstanding_bytes: Optional[int] = None,
) -> MemoryReadStageResult:
    """Read immutable lines through their physical owner's private L2."""

    _validate_stage_contract(
        requests,
        unit,
        grid_2d=grid_2d,
        connectivity=connectivity,
        coalesce_scope=coalesce_scope,
    )

    if session.cluster_count != unit.clusters_per_unit:
        raise ValueError("L2 session and cluster unit have different sizes")
    if session.cache_specs != (
        (unit.l2_per_cluster,) * unit.clusters_per_unit
    ):
        raise ValueError("L2 session capacity/policy does not match the cluster unit")
    if session.line_bytes != unit.dram.sector_bytes:
        raise ValueError("private L2 line size must equal the DRAM sector size")
    epochs = {request.epoch for request in requests}
    if len(epochs) > 1:
        raise ValueError("one read stage must contain exactly one epoch")

    swizzles: Dict[str, BankSwizzle] = {}
    bulk_requests: List[BulkLineRequest] = []
    for request in requests:
        previous = swizzles.setdefault(request.tensor, request.swizzle)
        if previous != request.swizzle:
            raise ValueError("one tensor cannot use multiple bank swizzles in a stage")
        owner = request.placement.owner(
            request.requester,
            request.shard_owner,
            unit.clusters_per_unit,
        )
        replica = request.placement.replica_id(
            request.requester,
            request.shard_owner,
            unit.clusters_per_unit,
        )
        bulk_requests.append(
            BulkLineRequest(
                requester=request.requester,
                owner=owner,
                allocation_id=request.allocation_id,
                replica_id=replica,
                sectors=np.asarray(request.sectors, dtype=np.int64),
                tensor=request.tensor,
                epoch=request.epoch,
                mapping_signature=(
                    _dram_mapping_signature(unit, connectivity),
                    request.swizzle,
                    request.mapping_signature,
                ),
                address_interval=request.address_interval,
            )
        )

    replay = session.replay_bulk_epoch(
        bulk_requests,
        remote_fill_policy=unit.remote_fill_policy,
    )
    l2_result = replay.stats
    bank_results = _service_grouped_sectors(
        dict(replay.ddr_misses),
        swizzles,
        unit,
        connectivity=connectivity,
        coalesce_scope=coalesce_scope,
        apply_littles_law=apply_littles_law,
        outstanding_bytes=outstanding_bytes,
    )
    bank_read_bytes = sum(
        result.service.transferred_bytes for result in bank_results
    )
    if bank_read_bytes != l2_result.ddr_bytes:
        raise ValueError(
            "distinct physical allocations overlap one owner-local DRAM "
            "sector; provide non-overlapping layout/base offsets"
        )
    bank_seconds = max(
        (result.final_seconds for result in bank_results), default=0.0
    )

    l2_seconds = 0.0
    l2_lines = l2_result.l2_served_lines_by_owner()
    for cluster, lines in l2_result.l2_fill_lines_by_cluster().items():
        l2_lines[cluster] = l2_lines.get(cluster, 0) + lines
    for lines in l2_lines.values():
        seconds = lines * session.line_bytes / unit.l2_bandwidth_bytes_s
        if lines:
            seconds += unit.l2_hit_latency_seconds
        l2_seconds = max(l2_seconds, seconds)

    flows = tuple(
        TrafficFlow(
            source,
            destination,
            nbytes,
            "%s:read_response" % tensor,
            epoch,
        )
        for (source, destination, tensor, epoch), nbytes in sorted(
            replay.remote_flow_bytes.items()
        )
    )
    noc = unit.noc.estimate(
        flows,
        unit.clusters_per_unit,
        grid_2d=grid_2d,
    )
    source_seconds = max(bank_seconds, l2_seconds)
    total_seconds = (
        max(source_seconds, noc.total_seconds)
        if unit.memory_noc_overlap
        else source_seconds + noc.total_seconds
    )
    return MemoryReadStageResult(
        l2=l2_result,
        owner_bank_results=bank_results,
        noc=noc,
        bank_seconds=bank_seconds,
        l2_seconds=l2_seconds,
        source_seconds=source_seconds,
        total_seconds=total_seconds,
    )


def model_write_stage(
    requests: Sequence[TensorWriteRequest],
    unit: ClusterUnitSpec,
    *,
    session: Optional[DistributedL2Session] = None,
    grid_2d: Optional[Tuple[int, int]] = None,
    connectivity: str = "direct",
    coalesce_scope: str = "wave",
    apply_littles_law: bool = True,
    outstanding_bytes: Optional[int] = None,
) -> MemoryWriteStageResult:
    """Store output lines at their physical owner without cache coherence.

    Remote writes are routed to the owner and then enter only the owner's DDR
    bank scheduler.  The prototype intentionally does not claim write-back or
    invalidation support for immutable input replicas.
    """

    _validate_stage_contract(
        requests,
        unit,
        grid_2d=grid_2d,
        connectivity=connectivity,
        coalesce_scope=coalesce_scope,
    )
    if session is not None:
        if session.cluster_count != unit.clusters_per_unit or (
            session.cache_specs
            != (unit.l2_per_cluster,) * unit.clusters_per_unit
        ):
            raise ValueError("L2 session does not match the cluster unit")

    physical_groups: Dict[
        Tuple[int, str, str, int], List[np.ndarray]
    ] = {}
    mapping_signatures: Dict[Tuple[int, str, str, int], object] = {}
    address_intervals: Dict[
        Tuple[int, str, str, int], Optional[Tuple[int, int]]
    ] = {}
    swizzles: Dict[str, BankSwizzle] = {}
    flow_bytes: Dict[Tuple[int, int, str, int], int] = {}
    for request in requests:
        previous = swizzles.setdefault(request.tensor, request.swizzle)
        if previous != request.swizzle:
            raise ValueError("one tensor cannot use multiple bank swizzles in a stage")
        owner = request.placement.owner(
            request.requester,
            request.shard_owner,
            unit.clusters_per_unit,
        )
        replica = request.placement.replica_id(
            request.requester,
            request.shard_owner,
            unit.clusters_per_unit,
        )
        sectors = np.unique(
            np.asarray(request.sectors, dtype=np.int64).reshape(-1)
        )
        physical_key = (
            owner,
            request.tensor,
            request.allocation_id,
            replica,
        )
        physical_groups.setdefault(physical_key, []).append(sectors)
        previous_mapping = mapping_signatures.setdefault(
            physical_key, request.mapping_signature
        )
        if previous_mapping != request.mapping_signature:
            raise ValueError(
                "one physical allocation cannot use multiple layout signatures"
            )
        if (
            physical_key in address_intervals
            and address_intervals[physical_key] != request.address_interval
        ):
            raise ValueError(
                "one physical allocation cannot use multiple address intervals"
            )
        address_intervals[physical_key] = request.address_interval
        if request.requester != owner:
            flow_key = (
                request.requester,
                owner,
                request.tensor,
                request.phase,
            )
            flow_bytes[flow_key] = (
                flow_bytes.get(flow_key, 0)
                + int(sectors.size) * unit.dram.sector_bytes
            )

    # Disjoint M/N shards can share a configured boundary line.  Coalesce within
    # one physical allocation while keeping true partial-C copies distinct by
    # their allocation IDs.
    grouped_chunks: Dict[Tuple[int, str], List[np.ndarray]] = {}
    pressure_requests = []
    written_lines = 0
    for physical_key, chunks in sorted(
        physical_groups.items()
    ):
        owner, tensor, _allocation, _replica = physical_key
        sectors = np.unique(
            chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        )
        written_lines += int(sectors.size)
        grouped_chunks.setdefault((owner, tensor), []).append(sectors)
        pressure_requests.append(
            BulkLineRequest(
                requester=owner,
                owner=owner,
                allocation_id=_allocation,
                replica_id=_replica,
                sectors=sectors,
                tensor=tensor,
                mapping_signature=(
                    _dram_mapping_signature(unit, connectivity),
                    swizzles[tensor],
                    mapping_signatures[physical_key],
                ),
                address_interval=address_intervals[physical_key],
            )
        )
    grouped = {
        key: chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        for key, chunks in grouped_chunks.items()
    }

    flows = tuple(
        TrafficFlow(
            source,
            destination,
            nbytes,
            "%s:write" % tensor,
            phase,
        )
        for (source, destination, tensor, phase), nbytes in sorted(
            flow_bytes.items()
        )
    )

    bank_results = _service_grouped_sectors(
        grouped,
        swizzles,
        unit,
        connectivity=connectivity,
        coalesce_scope=coalesce_scope,
        apply_littles_law=apply_littles_law,
        outstanding_bytes=outstanding_bytes,
    )
    bank_write_bytes = sum(
        result.service.transferred_bytes for result in bank_results
    )
    if bank_write_bytes != written_lines * unit.dram.sector_bytes:
        raise ValueError(
            "distinct output allocations overlap one owner-local DRAM "
            "sector; provide non-overlapping layout/base offsets"
        )
    bank_seconds = max(
        (result.final_seconds for result in bank_results), default=0.0
    )
    noc = unit.noc.estimate(
        flows,
        unit.clusters_per_unit,
        grid_2d=grid_2d,
    )
    total_seconds = (
        max(bank_seconds, noc.total_seconds)
        if unit.memory_noc_overlap
        else bank_seconds + noc.total_seconds
    )
    if session is not None:
        session.apply_output_pressure(pressure_requests)
    return MemoryWriteStageResult(
        owner_bank_results=bank_results,
        noc=noc,
        written_lines=written_lines,
        line_bytes=unit.dram.sector_bytes,
        bank_seconds=bank_seconds,
        total_seconds=total_seconds,
    )
