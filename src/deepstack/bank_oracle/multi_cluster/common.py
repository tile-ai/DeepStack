"""Shared contracts for modeling several physical clusters as one unit.

The single-cluster bank oracle deliberately remains unaware of ranks.  This
module describes the outer distribution boundary: how many independent
clusters participate, where physical tensor copies live, and how point-to-
point traffic is charged between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..gemm import GemmBankOptions, GemmProblem, GemmTiling
from ..l2_cache import L2CacheSpec
from ..layout import BankSwizzle
from ..spec import DramBankSpec


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class ExtentSlice:
    """One exact or padded contiguous shard of a one-dimensional extent."""

    start: int
    stop: int
    logical_stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def logical_size(self) -> int:
        return max(self.logical_stop - self.start, 0)


def split_extent(
    extent: int,
    parts: int,
    index: int,
    *,
    pad: bool = False,
) -> ExtentSlice:
    """Return a deterministic, non-overlapping shard.

    Exact mode distributes the remainder to the first ranks.  Padded mode
    gives every rank ``ceil(extent / parts)`` elements and records how much of
    the final block is logically useful.
    """

    if extent <= 0 or parts <= 0 or not 0 <= index < parts:
        raise ValueError("invalid extent split")
    if pad:
        width = ceil_div(extent, parts)
        start = index * width
        stop = start + width
        return ExtentSlice(start, stop, min(stop, extent))
    base, remainder = divmod(extent, parts)
    width = base + int(index < remainder)
    start = index * base + min(index, remainder)
    return ExtentSlice(start, start + width, start + width)


class PlacementKind(str, Enum):
    """Physical-copy policy for one tensor allocation."""

    SHARDED = "sharded"
    REPLICATED = "replicated"
    CANONICAL = "canonical"


@dataclass(frozen=True)
class TensorPlacement:
    """Resolve one logical line to an immutable physical home.

    ``SHARDED`` keeps one copy at the algorithm-provided shard owner.
    ``REPLICATED`` gives every requester a distinct local physical copy.
    ``CANONICAL`` places the entire tensor at ``canonical_cluster``.
    """

    kind: PlacementKind = PlacementKind.SHARDED
    canonical_cluster: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == PlacementKind.CANONICAL:
            if self.canonical_cluster is None or self.canonical_cluster < 0:
                raise ValueError("canonical placement requires a non-negative owner")
        elif self.canonical_cluster is not None:
            raise ValueError("canonical_cluster is valid only for canonical placement")

    @classmethod
    def sharded(cls) -> "TensorPlacement":
        return cls(PlacementKind.SHARDED)

    @classmethod
    def replicated(cls) -> "TensorPlacement":
        return cls(PlacementKind.REPLICATED)

    @classmethod
    def canonical(cls, cluster: int = 0) -> "TensorPlacement":
        return cls(PlacementKind.CANONICAL, cluster)

    def owner(self, requester: int, shard_owner: int, cluster_count: int) -> int:
        if not 0 <= requester < cluster_count:
            raise ValueError("requester is outside the cluster unit")
        if not 0 <= shard_owner < cluster_count:
            raise ValueError("shard owner is outside the cluster unit")
        if self.kind == PlacementKind.REPLICATED:
            return requester
        if self.kind == PlacementKind.CANONICAL:
            assert self.canonical_cluster is not None
            if self.canonical_cluster >= cluster_count:
                raise ValueError("canonical owner is outside the cluster unit")
            return self.canonical_cluster
        return shard_owner

    def replica_id(self, requester: int, shard_owner: int, cluster_count: int) -> int:
        """Return a stable physical-copy identifier for cache tags."""

        return self.owner(requester, shard_owner, cluster_count)


@dataclass(frozen=True)
class ClusterGrid:
    """Logical ``M x N x K`` process grid with row-major rank numbering."""

    m: int = 1
    n: int = 1
    k: int = 1

    def __post_init__(self) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError("cluster-grid dimensions must be positive")

    @property
    def size(self) -> int:
        return self.m * self.n * self.k

    def rank(self, m_index: int, n_index: int, k_index: int = 0) -> int:
        if not (
            0 <= m_index < self.m
            and 0 <= n_index < self.n
            and 0 <= k_index < self.k
        ):
            raise ValueError("cluster-grid coordinate is out of range")
        return (m_index * self.n + n_index) * self.k + k_index

    def coordinate(self, rank: int) -> Tuple[int, int, int]:
        if not 0 <= rank < self.size:
            raise ValueError("cluster rank is out of range")
        mn, k_index = divmod(rank, self.k)
        m_index, n_index = divmod(mn, self.n)
        return m_index, n_index, k_index


@dataclass(frozen=True)
class TrafficFlow:
    source: int
    destination: int
    nbytes: int
    kind: str = ""
    phase: int = 0

    def __post_init__(self) -> None:
        if self.source < 0 or self.destination < 0 or self.nbytes < 0:
            raise ValueError("invalid point-to-point traffic flow")


@dataclass(frozen=True)
class NoCResult:
    flows: Tuple[TrafficFlow, ...]
    total_bytes: int
    maximum_link_bytes: int
    maximum_path_hops: int
    serialization_seconds: float
    latency_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class NoCSpec:
    """Small deterministic point-to-point network model.

    ``ideal_switch`` charges the busiest injection/ejection link and an
    optional central-fabric bandwidth.  ``torus_2d`` routes each flow with a
    deterministic shortest XY path over the supplied logical grid.  The
    latter is useful for Cannon sensitivity; the actual four-cluster design
    may instead select the switch topology.
    """

    link_bandwidth_bytes_s: float = float("inf")
    hop_latency_seconds: float = 0.0
    topology: str = "ideal_switch"
    switch_fabric_bandwidth_bytes_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.link_bandwidth_bytes_s <= 0:
            raise ValueError("NoC link bandwidth must be positive")
        if self.hop_latency_seconds < 0:
            raise ValueError("NoC hop latency must be non-negative")
        if self.topology not in {"ideal_switch", "torus_2d"}:
            raise ValueError("NoC topology must be ideal_switch or torus_2d")
        if (
            self.switch_fabric_bandwidth_bytes_s is not None
            and self.switch_fabric_bandwidth_bytes_s <= 0
        ):
            raise ValueError("switch fabric bandwidth must be positive")

    @staticmethod
    def _axis_steps(source: int, destination: int, extent: int) -> Tuple[int, ...]:
        if source == destination:
            return ()
        positive = (destination - source) % extent
        negative = (source - destination) % extent
        direction = 1 if positive <= negative else -1
        count = positive if direction == 1 else negative
        values = []
        current = source
        for _ in range(count):
            current = (current + direction) % extent
            values.append(current)
        return tuple(values)

    def estimate(
        self,
        flows: Iterable[TrafficFlow],
        cluster_count: int,
        *,
        grid_2d: Optional[Tuple[int, int]] = None,
    ) -> NoCResult:
        retained = tuple(
            flow
            for flow in flows
            if flow.nbytes and flow.source != flow.destination
        )
        for flow in retained:
            if max(flow.source, flow.destination) >= cluster_count:
                raise ValueError("NoC flow endpoint is outside the cluster unit")
        if not retained:
            return NoCResult((), 0, 0, 0, 0.0, 0.0, 0.0)

        link_loads: Dict[Tuple[int, int], int] = {}
        maximum_hops = 1
        if self.topology == "ideal_switch":
            outgoing = [0] * cluster_count
            incoming = [0] * cluster_count
            for flow in retained:
                outgoing[flow.source] += flow.nbytes
                incoming[flow.destination] += flow.nbytes
            maximum_link = max(max(outgoing), max(incoming))
        else:
            if grid_2d is None or grid_2d[0] * grid_2d[1] != cluster_count:
                raise ValueError("torus_2d requires a matching logical grid")
            rows, columns = grid_2d
            for flow in retained:
                source_row, source_column = divmod(flow.source, columns)
                target_row, target_column = divmod(flow.destination, columns)
                current_row = source_row
                current_column = source_column
                hops = 0
                for next_column in self._axis_steps(
                    current_column, target_column, columns
                ):
                    source_rank = current_row * columns + current_column
                    target_rank = current_row * columns + next_column
                    edge = (source_rank, target_rank)
                    link_loads[edge] = link_loads.get(edge, 0) + flow.nbytes
                    current_column = next_column
                    hops += 1
                for next_row in self._axis_steps(current_row, target_row, rows):
                    source_rank = current_row * columns + current_column
                    target_rank = next_row * columns + current_column
                    edge = (source_rank, target_rank)
                    link_loads[edge] = link_loads.get(edge, 0) + flow.nbytes
                    current_row = next_row
                    hops += 1
                maximum_hops = max(maximum_hops, hops)
            maximum_link = max(link_loads.values(), default=0)

        total_bytes = sum(flow.nbytes for flow in retained)
        serialization = maximum_link / self.link_bandwidth_bytes_s
        if (
            self.topology == "ideal_switch"
            and self.switch_fabric_bandwidth_bytes_s is not None
        ):
            serialization = max(
                serialization,
                total_bytes / self.switch_fabric_bandwidth_bytes_s,
            )
        latency = maximum_hops * self.hop_latency_seconds
        return NoCResult(
            retained,
            total_bytes,
            maximum_link,
            maximum_hops,
            serialization,
            latency,
            serialization + latency,
        )


@dataclass(frozen=True)
class ClusterUnitSpec:
    """Physical resources owned by one configurable multi-cluster unit.

    Bundled defaults are synthetic conveniences; callers can replace the DRAM,
    L2, NoC, and outstanding-window parameters independently.
    """

    clusters_per_unit: int
    dram: DramBankSpec = field(default_factory=DramBankSpec)
    l2_per_cluster: L2CacheSpec = field(
        default_factory=lambda: L2CacheSpec(8 * 1024 * 1024)
    )
    noc: NoCSpec = field(default_factory=NoCSpec)
    l2_bandwidth_bytes_s: float = float("inf")
    l2_hit_latency_seconds: float = 0.0
    remote_fill_policy: str = "home_only"
    memory_noc_overlap: bool = True
    outstanding_bytes_per_cluster: Optional[int] = None

    def __post_init__(self) -> None:
        if self.clusters_per_unit <= 0:
            raise ValueError("clusters_per_unit must be positive")
        if self.l2_bandwidth_bytes_s <= 0:
            raise ValueError("L2 bandwidth must be positive")
        if self.l2_hit_latency_seconds < 0:
            raise ValueError("L2 hit latency must be non-negative")
        if self.remote_fill_policy not in {"home_only", "requester_and_home"}:
            raise ValueError("invalid remote L2 fill policy")
        if (
            self.outstanding_bytes_per_cluster is not None
            and self.outstanding_bytes_per_cluster <= 0
        ):
            raise ValueError("outstanding bytes must be positive")


@dataclass(frozen=True)
class LocalModelConfig:
    """How each cluster invokes the existing bank and pipeline oracles."""

    tiling: GemmTiling
    bank_options: GemmBankOptions = field(default_factory=GemmBankOptions)
    arch: Optional[Any] = None
    compute_cycles_per_k: Optional[float] = None
    other_memory_cycles_per_k: Optional[float] = None
    store_other_memory_cycles: Optional[float] = None
    kernel_launch_seconds: float = 0.0
    layout_preset: str = "tile_major"
    swizzle: BankSwizzle = field(default_factory=BankSwizzle)
    a_phase: int = 0
    b_phase: int = 0
    c_phase: int = 0
    pitch_pad_sectors: int = 0

    def __post_init__(self) -> None:
        if self.arch is None and self.compute_cycles_per_k is None:
            raise ValueError("local arch or compute_cycles_per_k is required")
        if self.kernel_launch_seconds < 0:
            raise ValueError("kernel launch time must be non-negative")
        if self.layout_preset not in {"linear", "pitch_pad", "tile_major"}:
            raise ValueError("invalid local layout preset")


def effective_problem(
    problem: GemmProblem,
    grid: ClusterGrid,
    pad: bool,
) -> GemmProblem:
    """Return the physical padded problem used for address allocation."""

    if not pad:
        return problem
    return GemmProblem(
        m=ceil_div(problem.m, grid.m) * grid.m,
        n=ceil_div(problem.n, grid.n) * grid.n,
        k=ceil_div(problem.k, grid.k) * grid.k,
        batch=problem.batch,
        a_dtype_bytes=problem.a_dtype_bytes,
        b_dtype_bytes=problem.b_dtype_bytes,
        c_dtype_bytes=problem.c_dtype_bytes,
        b_broadcast_across_batch=problem.b_broadcast_across_batch,
        compute_dtype_bytes=problem.compute_dtype_bytes,
    )
