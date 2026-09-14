"""General overlap analysis framework implementing the paper's recursive
loop traversal algorithm.

For kernels with multiple operation groups, such as flash attention, model:
1. Overlap between different hardware units within a group (roofline maximum).
2. Pipeline overlap between groups across adjacent iterations (software pipelining).
3. Wave head/tail effects.

Core data structures:
- OpGroup: One operation or a group of overlapping operations using specific hardware units.
- LoopNode: One loop level containing multiple subgroups.

Dependency-aware scheduling:
- OpGroup.depends_on declares data dependencies as a DAG.
- model_overlap enumerates all valid topological orders and selects the optimal schedule.
"""
import math
import itertools
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

log = logging.getLogger(__name__)


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class HardwareUsage:
    """Time spent by an operation group on each hardware unit, in seconds.

    Resource quantities have already been converted to time using bandwidth,
    so the values can be compared directly. Different units can overlap
    (take the maximum); work on the same unit is serial (sum the times).
    """
    ddr_time: float = 0.0
    l2_time: float = 0.0
    l1_5_time: float = 0.0        # L1.5 cache (per-group passive cache)
    smem_time: float = 0.0
    tmem_time: float = 0.0        # Tensor Memory (Blackwell tcgen05 ld/st datapath)
    tensor_time: float = 0.0      # tensor core
    cuda_time: float = 0.0        # cuda core (FMA)
    sfu_time: float = 0.0         # special function unit
    network_time: float = 0.0     # Network communication time (for distributed compute-comm overlap)

    @property
    def total_no_overlap(self):
        """Fully serial execution time, summed across all units."""
        return (self.ddr_time + self.l2_time + self.l1_5_time + self.smem_time
                + self.tmem_time + self.tensor_time + self.cuda_time + self.sfu_time
                + self.network_time)

    @property
    def total_full_overlap(self):
        """Fully overlapped execution time, taking the maximum across units."""
        return max(self.ddr_time, self.l2_time, self.l1_5_time, self.smem_time,
                   self.tmem_time, self.tensor_time, self.cuda_time, self.sfu_time,
                   self.network_time)

    def __add__(self, other):
        return HardwareUsage(
            ddr_time=self.ddr_time + other.ddr_time,
            l2_time=self.l2_time + other.l2_time,
            l1_5_time=self.l1_5_time + other.l1_5_time,
            smem_time=self.smem_time + other.smem_time,
            tmem_time=self.tmem_time + other.tmem_time,
            tensor_time=self.tensor_time + other.tensor_time,
            cuda_time=self.cuda_time + other.cuda_time,
            sfu_time=self.sfu_time + other.sfu_time,
            network_time=self.network_time + other.network_time,
        )


@dataclass
class OpGroup:
    """An operation group containing one or more operations that can overlap.

    Operations within a group use different hardware units and can overlap.
    Groups are separated by synchronization and execute serially by default.

    depends_on: List of predecessor groups on which this group depends for data.
        This group must be scheduled after every predecessor.
        An empty list indicates no dependencies, allowing unrestricted ordering.
    """
    name: str
    usage: HardwareUsage
    depends_on: List['OpGroup'] = field(default_factory=list)
    is_loop: bool = False
    loop_node: Optional['LoopNode'] = None

    @property
    def latency(self):
        """Latency of this group: the maximum across hardware units (roofline)."""
        return self.usage.total_full_overlap


@dataclass
class LoopNode:
    """A single loop level.

    sub_groups: Operation groups in the loop body, separated by synchronization.
    num_iters: Number of loop iterations.
    sw_pipeline_stage: Software pipeline depth (1 means no pipelining).
    """
    name: str
    sub_groups: List[OpGroup]
    num_iters: int
    sw_pipeline_stage: int = 1
    is_inner_loop: bool = False


# =====================================================================
# Enumerate DAG topological sorts (implemented locally, no external dependencies)
# =====================================================================

def _build_dag(groups: List[OpGroup]) -> Tuple[List[List[int]], List[int], bool]:
    """Build a DAG (adjacency list and in-degrees) from the groups' depends_on lists.

    Returns:
        (adj, in_degree, has_deps)
        adj[i] = [j, ...] represents i -> j (i is a predecessor of j).
        in_degree[j] is the in-degree of j.
        has_deps indicates whether any dependencies exist.
    """
    n = len(groups)
    group_to_idx = {id(g): i for i, g in enumerate(groups)}
    adj = [[] for _ in range(n)]
    in_degree = [0] * n
    has_deps = False

    for j, g in enumerate(groups):
        for dep in g.depends_on:
            dep_id = id(dep)
            if dep_id in group_to_idx:
                i = group_to_idx[dep_id]
                adj[i].append(j)
                in_degree[j] += 1
                has_deps = True

    return adj, in_degree, has_deps


def all_topological_sorts_builtin(groups: List[OpGroup]) -> List[List[int]]:
    """Enumerate all valid topological orders of a DAG using a custom recursive DFS.

    Algorithm: a variant of Kahn's algorithm. Choose one node with in_degree=0,
    recurse on the remaining graph, and restore the state when backtracking.

    Complexity: O(n! / dependency constraints), suitable for small DAGs with n <= 8.
    With no dependencies, this enumerates all n! permutations.
    """
    adj, in_degree, has_deps = _build_dag(groups)
    n = len(groups)

    if not has_deps:
        # No dependencies -> all permutations
        return [list(p) for p in itertools.permutations(range(n))]

    results = []
    current = []

    def dfs():
        if len(current) == n:
            results.append(list(current))
            return
        for i in range(n):
            if in_degree[i] == 0 and i not in current:
                # Add i to the permutation
                current.append(i)
                # Update successor in-degrees
                for j in adj[i]:
                    in_degree[j] -= 1
                dfs()
                # Backtrack
                current.pop()
                for j in adj[i]:
                    in_degree[j] += 1

    dfs()
    return results


def all_topological_sorts_networkx(groups: List[OpGroup]) -> List[List[int]]:
    """Enumerate all topological orders with NetworkX for cross-validation."""
    import networkx as nx

    adj, in_degree, has_deps = _build_dag(groups)
    n = len(groups)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            G.add_edge(i, j)

    return [list(order) for order in nx.all_topological_sorts(G)]


def verify_topological_sorts(groups: List[OpGroup]) -> bool:
    """Verify that the custom implementation matches NetworkX."""
    builtin = all_topological_sorts_builtin(groups)
    nx_result = all_topological_sorts_networkx(groups)

    builtin_set = set(tuple(s) for s in builtin)
    nx_set = set(tuple(s) for s in nx_result)

    if builtin_set != nx_set:
        log.error("Topological sort mismatch! builtin=%d, networkx=%d",
                  len(builtin_set), len(nx_set))
        log.error("Only in builtin: %s", builtin_set - nx_set)
        log.error("Only in networkx: %s", nx_set - builtin_set)
        return False

    log.info("Topological sort verified: %d legal orderings", len(builtin_set))
    return True


# =====================================================================
# Helper functions
# =====================================================================

def hw_usage_from_resources(ddr_io, l2_io, smem_io, arch,
                            tensor_flops=0, cuda_flops=0, sfu_flops=0,
                            max_util=0.9, l1_5_io=0, tmem_io=0):
    """Build HardwareUsage (per-SM times) from resource quantities and architecture bandwidth."""
    sm = arch.sm_count
    ddr_t = ddr_io / arch.ddr_bandwidth * sm / max_util if arch.ddr_bandwidth > 0 else 0
    l2_t = l2_io / arch.l2_bandwidth * sm / max_util if arch.l2_bandwidth > 0 else 0
    l1_5_bw = getattr(arch, 'l1_5_bandwidth', 0)
    l1_5_t = l1_5_io / l1_5_bw * sm / max_util if l1_5_bw > 0 and l1_5_io > 0 else 0
    smem_t = smem_io / arch.smem_bandwidth * sm / max_util if arch.smem_bandwidth > 0 else 0
    tmem_bw = getattr(arch, 'tmem_bandwidth', 0)
    tmem_t = tmem_io / tmem_bw * sm / max_util if tmem_bw > 0 and tmem_io > 0 else 0

    tensor_t = tensor_flops / arch.fp16_tensor_flops * sm / max_util if arch.fp16_tensor_flops > 0 and tensor_flops > 0 else 0
    cuda_t = cuda_flops / arch.fp32_cuda_core_flops * sm / max_util if arch.fp32_cuda_core_flops > 0 and cuda_flops > 0 else 0
    sfu_t = sfu_flops / arch.sfu_flops * sm / max_util if hasattr(arch, 'sfu_flops') and arch.sfu_flops > 0 and sfu_flops > 0 else 0

    return HardwareUsage(ddr_time=ddr_t, l2_time=l2_t, l1_5_time=l1_5_t,
                         smem_time=smem_t, tmem_time=tmem_t, tensor_time=tensor_t,
                         cuda_time=cuda_t, sfu_time=sfu_t)


# =====================================================================
# ModelOverlap: simulate overlap between op groups at a given stage
# =====================================================================

def simulate_schedule(groups: List[OpGroup], stage: int,
                      order: List[int]) -> Tuple[float, Dict]:
    """Simulate the latency of an operation group ordering at a given pipeline stage count.

    stage=1 (no pipeline): groups execute serially; units within a group overlap.
    stage>=2 (pipeline): hardware units pipeline independently;
        latency = max(total time for each unit).

    Returns:
        (latency_per_iter, util_dict)
    """
    total = HardwareUsage()
    for idx in order:
        total = total + groups[idx].usage

    if stage <= 1:
        lat = sum(groups[idx].latency for idx in order)
    else:
        lat = total.total_full_overlap

    return lat, {
        'ddr_time': total.ddr_time,
        'l2_time': total.l2_time,
        'l1_5_time': total.l1_5_time,
        'smem_time': total.smem_time,
        'tmem_time': total.tmem_time,
        'tensor_time': total.tensor_time,
        'cuda_time': total.cuda_time,
        'sfu_time': total.sfu_time,
    }


def model_overlap(groups: List[OpGroup], stage: int,
                  try_all_orders: bool = False) -> Tuple[float, Dict]:
    """ModelOverlap: enumerate all valid dependency-respecting schedules and select the best.

    Args:
        groups: Operation groups, optionally with depends_on dependencies.
        stage: Effective pipeline depth.
        try_all_orders: If True, enumerate all valid topological orders.
            If False, use only the original order.

    Returns:
        (best_latency_per_iter, best_util)
    """
    n = len(groups)
    if n == 0:
        return 0.0, {}

    if not try_all_orders:
        return simulate_schedule(groups, stage, list(range(n)))

    # Enumerate all valid topological sorts (respecting depends_on dependencies)
    legal_orders = all_topological_sorts_builtin(groups)

    if not legal_orders:
        # Cycle or other issue detected; fall back to the original order
        log.warning("No legal topological sort found, using original order")
        return simulate_schedule(groups, stage, list(range(n)))

    best_lat = float('inf')
    best_util = {}
    best_order = None
    for order in legal_orders:
        lat, util = simulate_schedule(groups, stage, order)
        if lat < best_lat:
            best_lat = lat
            best_util = util
            best_order = order

    if best_order is not None:
        order_names = [groups[i].name for i in best_order]
        log.info("Best schedule (%d candidates): %s, lat=%.3e",
                 len(legal_orders), order_names, best_lat)

    return best_lat, best_util


# =====================================================================
# AnalyzeLoop: recursively analyze the loop hierarchy
# =====================================================================

def analyze_loop(node: LoopNode, stage: int) -> Tuple[float, Dict]:
    """Recursively analyze a loop node.

    Corresponds to AnalyzeLoop in the paper's algorithm.

    Args:
        node: LoopNode.
        stage: Effective pipeline stage count inherited from the parent
            (occupancy x outer pipeline depth).

    Returns:
        (total_latency, util_dict)
    """
    new_stage = node.sw_pipeline_stage * stage

    if node.is_inner_loop:
        # Inner loop: all sub_groups form the body of one iteration
        lat_per_iter, util_per_iter = model_overlap(
            node.sub_groups, new_stage, try_all_orders=True)

        if new_stage <= 1:
            total = node.num_iters * lat_per_iter
        else:
            depth = new_stage - 1
            serial_per_iter = sum(g.latency for g in node.sub_groups)
            prologue = min(depth, node.num_iters) * serial_per_iter if depth > 0 else 0
            steady_iters = max(node.num_iters - depth, 0)
            steady = steady_iters * lat_per_iter
            epilogue_per_iter = sum(
                max(g.usage.tensor_time, g.usage.cuda_time, g.usage.sfu_time,
                    g.usage.tmem_time)
                for g in node.sub_groups)
            epilogue = min(depth, node.num_iters) * epilogue_per_iter if depth > 0 else 0
            total = prologue + steady + epilogue

        # Scale util to the total level (per_iter -> total)
        total_util = {k: v * node.num_iters for k, v in util_per_iter.items()}

        return total, total_util

    else:
        # Outer loop: process sub_groups one by one
        metrics_list = []
        for sg in node.sub_groups:
            if sg.is_loop and sg.loop_node is not None:
                lat, util = analyze_loop(sg.loop_node, new_stage)
            else:
                lat, util = model_overlap([sg], stage)
            metrics_list.append((lat, util))

        total_lat = sum(m[0] for m in metrics_list)

        merged_util = {}
        if metrics_list:
            for key in metrics_list[0][1]:
                merged_util[key] = sum(m[1].get(key, 0) for m in metrics_list)

        return total_lat * node.num_iters, merged_util


# =====================================================================
# OverlapAnalysis: top-level entry point
# =====================================================================

def overlap_analysis(root_node: LoopNode, tiles_per_sm: int = 1,
                     num_iters_override: int = None) -> Tuple[float, Dict]:
    """Perform top-level overlap analysis."""
    if num_iters_override is not None:
        root_node.num_iters = num_iters_override

    per_tile_lat, util = analyze_loop(root_node, tiles_per_sm)
    return per_tile_lat, util


# =====================================================================
# Convenience constructors
# =====================================================================

def make_op_group(name: str, arch, ddr_io=0, l2_io=0, smem_io=0,
                  tensor_flops=0, cuda_flops=0, sfu_flops=0,
                  max_util=0.9, depends_on=None, l1_5_io=0, tmem_io=0) -> OpGroup:
    """Convenience constructor for OpGroup.

    Args:
        depends_on: List of predecessor OpGroup objects; this group must be
            scheduled after all of them.
        l1_5_io: L1.5 cache I/O in bytes; 0 if there is no L1.5 cache.
        tmem_io: Tensor Memory I/O in bytes through the tcgen05 ld/st datapath;
            0 if there is no TMEM.
    """
    usage = hw_usage_from_resources(
        ddr_io, l2_io, smem_io, arch,
        tensor_flops=tensor_flops, cuda_flops=cuda_flops,
        sfu_flops=sfu_flops, max_util=max_util, l1_5_io=l1_5_io, tmem_io=tmem_io)
    return OpGroup(name=name, usage=usage,
                   depends_on=depends_on if depends_on else [])


def make_loop(name: str, sub_groups: List[OpGroup], num_iters: int,
              sw_pipeline_stage: int = 1, is_inner: bool = True) -> LoopNode:
    """Convenience constructor for LoopNode."""
    return LoopNode(
        name=name, sub_groups=sub_groups,
        num_iters=num_iters, sw_pipeline_stage=sw_pipeline_stage,
        is_inner_loop=is_inner)


def make_loop_group(name: str, loop_node: LoopNode) -> OpGroup:
    """Wrap a LoopNode as an OpGroup for nesting."""
    return OpGroup(name=name, usage=HardwareUsage(),
                   is_loop=True, loop_node=loop_node)


# =====================================================================
# Interface compatible with hete_reg_fusion / hete_smem_fusion
# =====================================================================

def overlap_to_hete_post(per_tile_lat: float, util: Dict,
                         smem_footprint: float = 0,
                         reg_footprint: float = 0,
                         ddr_read_io: float = 0,
                         l2_read_io: float = 0,
                         l2_hit_rate: float = 0) -> tuple:
    """Convert overlap_analysis output to the 12-tuple format used by hete_post_process."""
    if per_tile_lat <= 0:
        return (0,) * 12

    ddr_util = util.get('ddr_time', 0) / per_tile_lat
    l2_util = util.get('l2_time', 0) / per_tile_lat
    smem_l1_util = util.get('smem_time', 0) / per_tile_lat
    tensor_util = util.get('tensor_time', 0) / per_tile_lat
    cuda_util = util.get('cuda_time', 0) / per_tile_lat
    sfu_util = util.get('sfu_time', 0) / per_tile_lat

    return (per_tile_lat, ddr_util, l2_hit_rate, l2_util,
            smem_footprint, smem_l1_util, reg_footprint,
            ddr_read_io, l2_read_io,
            tensor_util, cuda_util, sfu_util)


def overlap_analysis_full(root_node: LoopNode, grids, arch,
                          tiles_per_sm: int = 1,
                          smem_footprint: float = 0,
                          reg_footprint: float = 0,
                          ddr_read_io: float = 0,
                          l2_read_io: float = 0,
                          l2_hit_rate: float = 0) -> tuple:
    """Run full overlap analysis with wave adjustment and return a 12-tuple."""
    import numpy as np

    per_tile_lat, util = overlap_analysis(root_node, tiles_per_sm=tiles_per_sm)

    waves = np.prod(grids) / arch.sm_count
    total_lat = per_tile_lat * math.ceil(waves) / max(waves, 1e-30)

    return overlap_to_hete_post(
        total_lat, util,
        smem_footprint=smem_footprint,
        reg_footprint=reg_footprint,
        ddr_read_io=ddr_read_io,
        l2_read_io=l2_read_io,
        l2_hit_rate=l2_hit_rate,
    )
