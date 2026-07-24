"""Epoch-based private-L2 replay for remote cluster memory accesses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..l2_cache import DeterministicFifoCache, L2CacheSpec


@dataclass(frozen=True, order=True)
class PhysicalLineId:
    """Stable identity of one line in one physical tensor copy."""

    allocation_id: str
    replica_id: int
    line_index: int

    def __post_init__(self) -> None:
        if self.replica_id < 0 or self.line_index < 0:
            raise ValueError("invalid physical line identity")


@dataclass(frozen=True)
class LineRequestEvent:
    """One requester reading one immutable line from its canonical home."""

    requester: int
    owner: int
    physical_line: PhysicalLineId
    owner_sector: int
    tensor: str
    epoch: int = 0
    mapping_signature: object = None
    address_interval: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if min(self.requester, self.owner, self.owner_sector, self.epoch) < 0:
            raise ValueError("invalid distributed line request")
        if not self.tensor:
            raise ValueError("distributed line request requires a tensor name")


class LineAccessSource(str, Enum):
    REQUESTER_L2_HIT = "requester_l2_hit"
    OWNER_L2_HIT = "owner_l2_hit"
    OWNER_DDR_MISS = "owner_ddr_miss"
    OWNER_MSHR_MERGE = "owner_mshr_merge"


@dataclass(frozen=True)
class LineAccessOutcome:
    event: LineRequestEvent
    source: LineAccessSource

    @property
    def remote(self) -> bool:
        return self.event.requester != self.event.owner


@dataclass(frozen=True)
class DistributedL2EpochResult:
    """Deterministic outcomes for one concurrent request epoch."""

    outcomes: Tuple[LineAccessOutcome, ...]
    line_bytes: int

    @property
    def accessed_lines(self) -> int:
        return len(self.outcomes)

    def count(self, source: LineAccessSource) -> int:
        return sum(outcome.source == source for outcome in self.outcomes)

    @property
    def requester_l2_hit_lines(self) -> int:
        return self.count(LineAccessSource.REQUESTER_L2_HIT)

    @property
    def owner_l2_hit_lines(self) -> int:
        return self.count(LineAccessSource.OWNER_L2_HIT)

    @property
    def ddr_miss_lines(self) -> int:
        return self.count(LineAccessSource.OWNER_DDR_MISS)

    @property
    def mshr_merged_lines(self) -> int:
        return self.count(LineAccessSource.OWNER_MSHR_MERGE)

    @property
    def ddr_bytes(self) -> int:
        return self.ddr_miss_lines * self.line_bytes

    @property
    def remote_response_lines(self) -> int:
        return sum(
            outcome.remote
            and outcome.source != LineAccessSource.REQUESTER_L2_HIT
            for outcome in self.outcomes
        )

    @property
    def remote_response_bytes(self) -> int:
        return self.remote_response_lines * self.line_bytes

    @property
    def remote_owner_l2_hit_lines(self) -> int:
        return sum(
            outcome.remote
            and outcome.source == LineAccessSource.OWNER_L2_HIT
            for outcome in self.outcomes
        )

    @property
    def remote_owner_l2_hit_bytes(self) -> int:
        return self.remote_owner_l2_hit_lines * self.line_bytes

    def ddr_misses_by_owner_tensor(self) -> Dict[Tuple[int, str], Tuple[int, ...]]:
        grouped: Dict[Tuple[int, str], List[int]] = {}
        for outcome in self.outcomes:
            if outcome.source != LineAccessSource.OWNER_DDR_MISS:
                continue
            event = outcome.event
            grouped.setdefault((event.owner, event.tensor), []).append(
                event.owner_sector
            )
        return {
            key: tuple(sorted(set(sectors)))
            for key, sectors in grouped.items()
        }

    def l2_served_lines_by_owner(self) -> Dict[int, int]:
        """Lines whose data came from a resident owner/requester L2."""

        grouped: Dict[int, int] = {}
        for outcome in self.outcomes:
            if outcome.source not in {
                LineAccessSource.REQUESTER_L2_HIT,
                LineAccessSource.OWNER_L2_HIT,
            }:
                continue
            cluster = (
                outcome.event.requester
                if outcome.source == LineAccessSource.REQUESTER_L2_HIT
                else outcome.event.owner
            )
            grouped[cluster] = grouped.get(cluster, 0) + 1
        return grouped


@dataclass(frozen=True)
class DistributedL2Stats:
    """Memory-bounded aggregate with the same public rate counters."""

    line_bytes: int
    accessed_lines: int
    requester_l2_hit_lines: int
    owner_l2_hit_lines: int
    ddr_miss_lines: int
    mshr_merged_lines: int
    remote_response_lines: int
    remote_owner_l2_hit_lines: int
    l2_served_by_cluster: Tuple[Tuple[int, int], ...]
    owner_fill_by_cluster: Tuple[Tuple[int, int], ...]
    requester_fill_by_cluster: Tuple[Tuple[int, int], ...]

    @property
    def ddr_bytes(self) -> int:
        return self.ddr_miss_lines * self.line_bytes

    @property
    def remote_response_bytes(self) -> int:
        return self.remote_response_lines * self.line_bytes

    @property
    def remote_owner_l2_hit_bytes(self) -> int:
        return self.remote_owner_l2_hit_lines * self.line_bytes

    def l2_served_lines_by_owner(self) -> Dict[int, int]:
        return dict(self.l2_served_by_cluster)

    def l2_fill_lines_by_cluster(self) -> Dict[int, int]:
        combined = dict(self.owner_fill_by_cluster)
        for cluster, lines in self.requester_fill_by_cluster:
            combined[cluster] = combined.get(cluster, 0) + lines
        return combined


@dataclass(frozen=True)
class BulkLineRequest:
    """Vectorized line request used by the scalable operator path."""

    requester: int
    owner: int
    allocation_id: str
    replica_id: int
    sectors: np.ndarray
    tensor: str
    epoch: int = 0
    # Physical bank-address transform for this allocation.  The cache itself
    # does not interpret the value; it only requires a persistent allocation
    # to keep the same hashable signature across operator calls.
    mapping_signature: object = None
    # Half-open owner-local sector range reserved by this physical copy.
    # When omitted, the replay conservatively infers/extends it from sectors.
    address_interval: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class BulkL2ReplayResult:
    stats: DistributedL2Stats
    ddr_misses: Mapping[Tuple[int, str, str, int], np.ndarray]
    remote_flow_bytes: Mapping[Tuple[int, int, str, int], int]


class DistributedL2Session:
    """Mutable private cache state explicitly owned by the unit caller.

    All events in one replay epoch probe the state that existed at the start
    of the epoch.  A cold physical line generates exactly one owner-DDR miss;
    concurrent requests for that line are MSHR merges rather than false hits.
    Fills are committed only after every probe has been classified.
    """

    def __init__(
        self,
        cluster_count: int,
        cache_specs: Sequence[L2CacheSpec],
    ) -> None:
        if cluster_count <= 0:
            raise ValueError("cluster_count must be positive")
        specs = tuple(cache_specs)
        if len(specs) != cluster_count:
            raise ValueError("one private L2 specification is required per cluster")
        line_sizes = {spec.line_bytes for spec in specs}
        if len(line_sizes) != 1:
            raise ValueError("all private L2 caches must use one line size")
        self._specs = specs
        self._caches = [DeterministicFifoCache(spec) for spec in specs]
        # Allocation-level home mapping keeps state proportional to physical
        # copies rather than every line ever touched.  The final value is the
        # constant owner-sector offset relative to PhysicalLineId.line_index.
        self._homes: Dict[Tuple[str, int], Tuple[int, int]] = {}
        self._bank_mappings: Dict[Tuple[str, int], object] = {}
        self._address_intervals: Dict[
            Tuple[str, int], Tuple[int, int, int]
        ] = {}
        # requester_and_home is deliberately a read-only sensitivity.  Keep
        # enough allocation-level state to reject a later write that would
        # otherwise leave an unmodelled stale requester copy.
        self._requester_copy_allocations = set()

    @classmethod
    def homogeneous(
        cls,
        cluster_count: int,
        cache_spec: L2CacheSpec,
    ) -> "DistributedL2Session":
        return cls(cluster_count, (cache_spec,) * cluster_count)

    @property
    def cluster_count(self) -> int:
        return len(self._caches)

    @property
    def line_bytes(self) -> int:
        return self._specs[0].line_bytes

    @property
    def cache_specs(self) -> Tuple[L2CacheSpec, ...]:
        return self._specs

    @property
    def resident_lines(self) -> Tuple[int, ...]:
        return tuple(cache.resident_lines for cache in self._caches)

    @property
    def peak_resident_lines(self) -> Tuple[int, ...]:
        return tuple(cache.peak_resident_lines for cache in self._caches)

    def reset(self) -> None:
        self._caches = [DeterministicFifoCache(spec) for spec in self._specs]
        self._homes.clear()
        self._bank_mappings.clear()
        self._address_intervals.clear()
        self._requester_copy_allocations.clear()

    def clone(self) -> "DistributedL2Session":
        cloned = DistributedL2Session(self.cluster_count, self._specs)
        cloned._caches = [cache.clone() for cache in self._caches]
        cloned._homes = dict(self._homes)
        cloned._bank_mappings = dict(self._bank_mappings)
        cloned._address_intervals = dict(self._address_intervals)
        cloned._requester_copy_allocations = set(
            self._requester_copy_allocations
        )
        return cloned

    @staticmethod
    def _event_key(event: LineRequestEvent) -> Tuple[object, ...]:
        return (
            event.epoch,
            event.owner,
            event.physical_line,
            event.requester,
            event.tensor,
            event.owner_sector,
        )

    def _commit_address_intervals(
        self,
        epoch_intervals: Mapping[
            Tuple[str, int], Tuple[int, int, int]
        ],
    ) -> None:
        """Atomically extend allocation ranges and reject owner-local aliasing."""

        candidate = dict(self._address_intervals)
        for key, (owner, start, stop) in epoch_intervals.items():
            if start < 0 or stop <= start:
                raise ValueError("invalid physical allocation address interval")
            previous = candidate.get(key)
            if previous is not None:
                previous_owner, previous_start, previous_stop = previous
                if previous_owner != owner:
                    raise ValueError(
                        "one physical allocation resolved to multiple homes"
                    )
                start = min(start, previous_start)
                stop = max(stop, previous_stop)
            candidate[key] = (owner, start, stop)

        by_owner: Dict[int, List[Tuple[int, int, Tuple[str, int]]]] = {}
        for key, (owner, start, stop) in candidate.items():
            by_owner.setdefault(owner, []).append((start, stop, key))
        for intervals in by_owner.values():
            intervals.sort()
            previous_start, previous_stop, previous_key = intervals[0]
            for start, stop, key in intervals[1:]:
                if start < previous_stop and key != previous_key:
                    raise ValueError(
                        "distinct physical allocations overlap one owner-local "
                        "DRAM sector; provide non-overlapping layout/base offsets"
                    )
                if stop > previous_stop:
                    previous_start, previous_stop, previous_key = (
                        start,
                        stop,
                        key,
                    )
        self._address_intervals = candidate

    def replay_epoch(
        self,
        events: Iterable[LineRequestEvent],
        *,
        remote_fill_policy: str = "home_only",
    ) -> DistributedL2EpochResult:
        if remote_fill_policy not in {"home_only", "requester_and_home"}:
            raise ValueError("invalid remote L2 fill policy")
        # Identical requester/line events in one epoch are one coalesced L2
        # lookup.  This also handles broadcast-B batch aliases without making
        # accounting depend on how many logical consumers emitted the same
        # physical request.
        materialized = tuple(events)
        for event in materialized:
            try:
                hash(event.mapping_signature)
            except TypeError as error:
                raise ValueError(
                    "bank mapping signature must be hashable"
                ) from error
        ordered = tuple(sorted(set(materialized), key=self._event_key))
        if not ordered:
            return DistributedL2EpochResult((), self.line_bytes)

        epochs = {event.epoch for event in ordered}
        if len(epochs) != 1:
            raise ValueError("replay_epoch accepts exactly one logical epoch")
        for event in ordered:
            if max(event.requester, event.owner) >= self.cluster_count:
                raise ValueError("line request endpoint is outside the cluster unit")

        # One physical identity must always resolve to one immutable home and
        # owner-local sector, independent of the current requester.
        epoch_homes: Dict[Tuple[str, int], Tuple[int, int]] = {}
        epoch_mappings: Dict[Tuple[str, int], object] = {}
        epoch_intervals: Dict[
            Tuple[str, int], Tuple[int, int, int]
        ] = {}
        for event in ordered:
            home_key = (
                event.physical_line.allocation_id,
                event.physical_line.replica_id,
            )
            home = (
                event.owner,
                event.owner_sector - event.physical_line.line_index,
            )
            epoch_previous = epoch_homes.setdefault(home_key, home)
            session_previous = self._homes.get(home_key)
            if epoch_previous != home or (
                session_previous is not None and session_previous != home
            ):
                raise ValueError("one physical line resolved to multiple homes")
            epoch_mapping = epoch_mappings.setdefault(
                home_key, event.mapping_signature
            )
            if epoch_mapping != event.mapping_signature or (
                home_key in self._bank_mappings
                and self._bank_mappings[home_key]
                != event.mapping_signature
            ):
                raise ValueError(
                    "one physical allocation used multiple bank mappings"
                )
            interval = event.address_interval or (
                event.owner_sector,
                event.owner_sector + 1,
            )
            if not interval[0] <= event.owner_sector < interval[1]:
                raise ValueError("requested sector is outside its allocation")
            previous_interval = epoch_intervals.get(home_key)
            if previous_interval is None:
                epoch_intervals[home_key] = (
                    event.owner,
                    interval[0],
                    interval[1],
                )
            else:
                interval_owner, start, stop = previous_interval
                if interval_owner != event.owner:
                    raise ValueError(
                        "one physical allocation resolved to multiple homes"
                    )
                epoch_intervals[home_key] = (
                    interval_owner,
                    min(start, interval[0]),
                    max(stop, interval[1]),
                )
        self._commit_address_intervals(epoch_intervals)
        self._homes.update(epoch_homes)
        self._bank_mappings.update(epoch_mappings)

        resolved: Dict[LineRequestEvent, LineAccessSource] = {}
        cold_groups: Dict[
            Tuple[int, PhysicalLineId], List[LineRequestEvent]
        ] = {}
        for event in ordered:
            requester_cache = self._caches[event.requester]
            owner_cache = self._caches[event.owner]
            if (
                remote_fill_policy == "requester_and_home"
                and event.requester != event.owner
                and requester_cache.probe_line(event.physical_line)
            ):
                resolved[event] = LineAccessSource.REQUESTER_L2_HIT
            elif owner_cache.probe_line(event.physical_line):
                resolved[event] = (
                    LineAccessSource.REQUESTER_L2_HIT
                    if event.requester == event.owner
                    else LineAccessSource.OWNER_L2_HIT
                )
            else:
                cold_groups.setdefault(
                    (event.owner, event.physical_line), []
                ).append(event)

        owner_fills: List[Tuple[int, PhysicalLineId]] = []
        for (owner, physical_line), group in sorted(cold_groups.items()):
            first = min(group, key=self._event_key)
            resolved[first] = LineAccessSource.OWNER_DDR_MISS
            for event in group:
                if event != first:
                    resolved[event] = LineAccessSource.OWNER_MSHR_MERGE
            owner_fills.append((owner, physical_line))

        requester_fills = set()
        if remote_fill_policy == "requester_and_home":
            for event, source in resolved.items():
                if (
                    event.requester != event.owner
                    and source != LineAccessSource.REQUESTER_L2_HIT
                ):
                    requester_fills.add((event.requester, event.physical_line))
                    self._requester_copy_allocations.add(
                        (
                            event.physical_line.allocation_id,
                            event.physical_line.replica_id,
                        )
                    )

        # Commit only after classification, preserving epoch-snapshot probes.
        for cluster, physical_line in sorted(owner_fills):
            self._caches[cluster].fill_line(physical_line)
        for cluster, physical_line in sorted(requester_fills):
            self._caches[cluster].fill_line(physical_line)

        outcomes = tuple(
            LineAccessOutcome(event, resolved[event]) for event in ordered
        )
        return DistributedL2EpochResult(outcomes, self.line_bytes)

    def replay_bulk_epoch(
        self,
        requests: Sequence[BulkLineRequest],
        *,
        remote_fill_policy: str = "home_only",
    ) -> BulkL2ReplayResult:
        """Replay vector requests with memory bounded by cache and one block.

        The detailed API above is convenient for line-level diagnostics.  It
        intentionally retains one Python outcome object per request, which is
        inappropriate for a 100-MiB matrix.  This path preserves the same
        epoch-snapshot hit/miss/MSHR semantics while retaining only aggregate
        counters, NumPy miss-sector chunks, and the final private-cache state.
        """

        if remote_fill_policy not in {"home_only", "requester_and_home"}:
            raise ValueError("invalid remote L2 fill policy")
        if not requests:
            return BulkL2ReplayResult(
                DistributedL2Stats(
                    line_bytes=self.line_bytes,
                    accessed_lines=0,
                    requester_l2_hit_lines=0,
                    owner_l2_hit_lines=0,
                    ddr_miss_lines=0,
                    mshr_merged_lines=0,
                    remote_response_lines=0,
                    remote_owner_l2_hit_lines=0,
                    l2_served_by_cluster=(),
                    owner_fill_by_cluster=(),
                    requester_fill_by_cluster=(),
                ),
                {},
                {},
            )
        epochs = {request.epoch for request in requests}
        if len(epochs) != 1:
            raise ValueError("replay_bulk_epoch accepts one logical epoch")

        grouped: Dict[
            Tuple[int, str, int, str],
            Dict[int, List[np.ndarray]],
        ] = {}
        epoch_homes: Dict[Tuple[str, int], Tuple[int, int]] = {}
        epoch_mappings: Dict[Tuple[str, int], object] = {}
        epoch_intervals: Dict[
            Tuple[str, int], Tuple[int, int, int]
        ] = {}
        requester_copy_allocations = set()
        for request in requests:
            if max(request.requester, request.owner) >= self.cluster_count:
                raise ValueError("bulk request endpoint is outside the cluster unit")
            if min(request.requester, request.owner, request.replica_id) < 0:
                raise ValueError("invalid bulk line request")
            sectors = np.asarray(request.sectors, dtype=np.int64).reshape(-1)
            if sectors.size and int(sectors.min()) < 0:
                raise ValueError("owner sectors must be non-negative")
            home_key = (request.allocation_id, request.replica_id)
            home = (request.owner, 0)
            epoch_previous = epoch_homes.setdefault(home_key, home)
            session_previous = self._homes.get(home_key)
            if epoch_previous != home or (
                session_previous is not None and session_previous != home
            ):
                raise ValueError("one physical allocation resolved to multiple homes")
            try:
                hash(request.mapping_signature)
            except TypeError as error:
                raise ValueError(
                    "bank mapping signature must be hashable"
                ) from error
            epoch_mapping = epoch_mappings.setdefault(
                home_key, request.mapping_signature
            )
            if epoch_mapping != request.mapping_signature or (
                home_key in self._bank_mappings
                and self._bank_mappings[home_key]
                != request.mapping_signature
            ):
                raise ValueError(
                    "one physical allocation used multiple bank mappings"
                )
            interval = request.address_interval
            if interval is None and sectors.size:
                interval = (int(sectors.min()), int(sectors.max()) + 1)
            if interval is not None:
                if (
                    interval[0] < 0
                    or interval[1] <= interval[0]
                    or (
                        sectors.size
                        and (
                            int(sectors.min()) < interval[0]
                            or int(sectors.max()) >= interval[1]
                        )
                    )
                ):
                    raise ValueError(
                        "requested sector is outside its allocation"
                    )
                previous_interval = epoch_intervals.get(home_key)
                if previous_interval is None:
                    epoch_intervals[home_key] = (
                        request.owner,
                        interval[0],
                        interval[1],
                    )
                else:
                    interval_owner, start, stop = previous_interval
                    if interval_owner != request.owner:
                        raise ValueError(
                            "one physical allocation resolved to multiple homes"
                        )
                    epoch_intervals[home_key] = (
                        interval_owner,
                        min(start, interval[0]),
                        max(stop, interval[1]),
                    )
            if (
                remote_fill_policy == "requester_and_home"
                and request.requester != request.owner
            ):
                requester_copy_allocations.add(home_key)
            key = (
                request.owner,
                request.allocation_id,
                request.replica_id,
                request.tensor,
            )
            grouped.setdefault(key, {}).setdefault(
                request.requester, []
            ).append(sectors)
        self._commit_address_intervals(epoch_intervals)
        self._homes.update(epoch_homes)
        self._bank_mappings.update(epoch_mappings)
        self._requester_copy_allocations.update(
            requester_copy_allocations
        )

        snapshots = tuple(cache.resident_keys for cache in self._caches)
        requester_fill_queues = [
            deque(maxlen=spec.capacity_lines) for spec in self._specs
        ]
        miss_chunks: Dict[
            Tuple[int, str, str, int], List[np.ndarray]
        ] = {}
        remote_flow_bytes: Dict[Tuple[int, int, str, int], int] = {}
        l2_served: Dict[int, int] = {}
        owner_fill_counts: Dict[int, int] = {}
        requester_fill_counts: Dict[int, int] = {}
        accessed_lines = 0
        requester_hits = 0
        owner_hits = 0
        ddr_misses = 0
        merged_misses = 0
        remote_responses = 0
        remote_owner_hits = 0
        epoch = next(iter(epochs))

        for (
            owner,
            allocation_id,
            replica_id,
            tensor,
        ), requester_chunks in sorted(grouped.items()):
            sector_arrays = []
            requester_arrays = []
            for requester, chunks in sorted(requester_chunks.items()):
                sectors = (
                    chunks[0]
                    if len(chunks) == 1
                    else np.concatenate(chunks)
                )
                sectors = np.unique(sectors)
                if not sectors.size:
                    continue
                sector_arrays.append(sectors)
                requester_arrays.append(
                    np.full(sectors.size, requester, dtype=np.int32)
                )
            if not sector_arrays:
                continue
            all_sectors = (
                sector_arrays[0]
                if len(sector_arrays) == 1
                else np.concatenate(sector_arrays)
            )
            all_requesters = (
                requester_arrays[0]
                if len(requester_arrays) == 1
                else np.concatenate(requester_arrays)
            )
            order = np.lexsort((all_requesters, all_sectors))
            sorted_sectors = all_sectors[order]
            sorted_requesters = all_requesters[order]
            starts = np.flatnonzero(
                np.r_[True, sorted_sectors[1:] != sorted_sectors[:-1]]
            )
            stops = np.r_[starts[1:], sorted_sectors.size]
            unique_sectors = sorted_sectors[starts]
            miss_mask = np.zeros(unique_sectors.size, dtype=bool)

            for line_position, (start, stop) in enumerate(zip(starts, stops)):
                sector = int(sorted_sectors[start])
                requesters = sorted_requesters[start:stop]
                physical_line = PhysicalLineId(
                    allocation_id, replica_id, sector
                )
                owner_resident = physical_line in snapshots[owner]
                cold_requesters = []
                for requester_value in requesters:
                    requester = int(requester_value)
                    accessed_lines += 1
                    requester_resident = (
                        remote_fill_policy == "requester_and_home"
                        and requester != owner
                        and physical_line in snapshots[requester]
                    )
                    if requester_resident:
                        requester_hits += 1
                        l2_served[requester] = l2_served.get(requester, 0) + 1
                        continue
                    if owner_resident:
                        if requester == owner:
                            requester_hits += 1
                        else:
                            owner_hits += 1
                            remote_owner_hits += 1
                        l2_served[owner] = l2_served.get(owner, 0) + 1
                    else:
                        cold_requesters.append(requester)

                    if requester != owner:
                        remote_responses += 1
                        flow_key = (owner, requester, tensor, epoch)
                        remote_flow_bytes[flow_key] = (
                            remote_flow_bytes.get(flow_key, 0)
                            + self.line_bytes
                        )
                        if remote_fill_policy == "requester_and_home":
                            requester_fill_queues[requester].append(
                                physical_line
                            )
                            requester_fill_counts[requester] = (
                                requester_fill_counts.get(requester, 0) + 1
                            )

                if cold_requesters:
                    ddr_misses += 1
                    merged_misses += len(cold_requesters) - 1
                    miss_mask[line_position] = True
                    # The owner-fill stream is ordered by
                    # (owner, allocation, replica, sector).
                    self._caches[owner].fill_line(physical_line)
                    owner_fill_counts[owner] = (
                        owner_fill_counts.get(owner, 0) + 1
                    )

            if miss_mask.any():
                miss_chunks.setdefault(
                    (owner, tensor, allocation_id, replica_id), []
                ).append(
                    unique_sectors[miss_mask].copy()
                )

        # Requester copies complete after all owner probes/fills in the epoch.
        # Each bounded queue retains exactly the suffix that can survive in
        # that requester's FIFO, avoiding memory growth with tensor size.
        for requester, fills in enumerate(requester_fill_queues):
            for physical_line in fills:
                self._caches[requester].fill_line(physical_line)

        ddr_by_owner_tensor = {
            key: chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            for key, chunks in miss_chunks.items()
        }
        stats = DistributedL2Stats(
            line_bytes=self.line_bytes,
            accessed_lines=accessed_lines,
            requester_l2_hit_lines=requester_hits,
            owner_l2_hit_lines=owner_hits,
            ddr_miss_lines=ddr_misses,
            mshr_merged_lines=merged_misses,
            remote_response_lines=remote_responses,
            remote_owner_l2_hit_lines=remote_owner_hits,
            l2_served_by_cluster=tuple(sorted(l2_served.items())),
            owner_fill_by_cluster=tuple(sorted(owner_fill_counts.items())),
            requester_fill_by_cluster=tuple(
                sorted(requester_fill_counts.items())
            ),
        )
        return BulkL2ReplayResult(
            stats,
            ddr_by_owner_tensor,
            remote_flow_bytes,
        )

    def apply_output_pressure(
        self,
        requests: Sequence[BulkLineRequest],
    ) -> None:
        """Insert written lines into each owner's FIFO for capacity pressure.

        This is not a writable-coherence protocol: it models only the same
        output-footprint aging used by the single-cluster cache model.  It
        does not create requester copies or invalidate read-only sensitivity
        copies in other clusters.
        """

        grouped: Dict[
            Tuple[int, str, int], List[np.ndarray]
        ] = {}
        epoch_homes: Dict[Tuple[str, int], Tuple[int, int]] = {}
        epoch_mappings: Dict[Tuple[str, int], object] = {}
        epoch_intervals: Dict[
            Tuple[str, int], Tuple[int, int, int]
        ] = {}
        for request in requests:
            if max(request.requester, request.owner) >= self.cluster_count:
                raise ValueError("output endpoint is outside the cluster unit")
            home_key = (request.allocation_id, request.replica_id)
            home = (request.owner, 0)
            epoch_previous = epoch_homes.setdefault(home_key, home)
            session_previous = self._homes.get(home_key)
            if epoch_previous != home or (
                session_previous is not None and session_previous != home
            ):
                raise ValueError("one physical allocation resolved to multiple homes")
            if home_key in self._requester_copy_allocations:
                raise ValueError(
                    "requester_and_home copies are read-only; reset the "
                    "session before writing this allocation"
                )
            try:
                hash(request.mapping_signature)
            except TypeError as error:
                raise ValueError(
                    "bank mapping signature must be hashable"
                ) from error
            epoch_mapping = epoch_mappings.setdefault(
                home_key, request.mapping_signature
            )
            if epoch_mapping != request.mapping_signature or (
                home_key in self._bank_mappings
                and self._bank_mappings[home_key]
                != request.mapping_signature
            ):
                raise ValueError(
                    "one physical allocation used multiple bank mappings"
                )
            sectors = np.asarray(request.sectors, dtype=np.int64).reshape(-1)
            interval = request.address_interval
            if interval is None and sectors.size:
                interval = (int(sectors.min()), int(sectors.max()) + 1)
            if interval is not None:
                if (
                    interval[0] < 0
                    or interval[1] <= interval[0]
                    or (
                        sectors.size
                        and (
                            int(sectors.min()) < interval[0]
                            or int(sectors.max()) >= interval[1]
                        )
                    )
                ):
                    raise ValueError(
                        "requested sector is outside its allocation"
                    )
                previous_interval = epoch_intervals.get(home_key)
                if previous_interval is None:
                    epoch_intervals[home_key] = (
                        request.owner,
                        interval[0],
                        interval[1],
                    )
                else:
                    interval_owner, start, stop = previous_interval
                    if interval_owner != request.owner:
                        raise ValueError(
                            "one physical allocation resolved to multiple homes"
                        )
                    epoch_intervals[home_key] = (
                        interval_owner,
                        min(start, interval[0]),
                        max(stop, interval[1]),
                    )
            grouped.setdefault(
                (request.owner, request.allocation_id, request.replica_id), []
            ).append(sectors)
        self._commit_address_intervals(epoch_intervals)
        self._homes.update(epoch_homes)
        self._bank_mappings.update(epoch_mappings)

        for (owner, allocation_id, replica_id), chunks in sorted(
            grouped.items()
        ):
            if not self._specs[owner].output_pressure:
                continue
            sectors = np.unique(
                chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            )
            for sector_value in sectors:
                self._caches[owner].fill_line(
                    PhysicalLineId(
                        allocation_id, replica_id, int(sector_value)
                    )
                )
