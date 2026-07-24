"""通用 Overlap Analysis 框架 — 实现论文中的递归循环遍历算法。

用于多 op group 的 kernel（如 flash attention），建模:
1. 同一 group 内不同硬件单元的 overlap (roofline max)
2. 相邻迭代之间 group 的 pipeline overlap (软件流水线)
3. Wave head/tail 效应

核心数据结构:
- OpGroup: 一个 op 或一组可 overlap 的 op，占用特定硬件单元
- LoopNode: 一个循环层次，包含多个 subgroup

依赖感知调度:
- OpGroup.depends_on 声明数据依赖 (DAG)
- model_overlap 枚举所有合法拓扑排序，选最优调度
"""
import math
import itertools
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

log = logging.getLogger(__name__)


# =====================================================================
# 数据结构
# =====================================================================

@dataclass
class HardwareUsage:
    """一个 op group 在各硬件单元上的时间 (秒)。

    时间已经过 bandwidth 转换，直接可比较。
    不同单元之间可 overlap (取 max)，同单元串行 (求和)。
    """
    ddr_time: float = 0.0
    l2_time: float = 0.0
    l1_5_time: float = 0.0        # L1.5 cache (per-group passive cache)
    smem_time: float = 0.0
    tmem_time: float = 0.0        # Tensor Memory (Blackwell tcgen05 ld/st datapath)
    tensor_time: float = 0.0      # tensor core
    cuda_time: float = 0.0        # cuda core (FMA)
    sfu_time: float = 0.0         # special function unit
    network_time: float = 0.0     # 网络通信时间 (用于分布式 compute-comm overlap)

    @property
    def total_no_overlap(self):
        """完全串行时间（所有单元求和）。"""
        return (self.ddr_time + self.l2_time + self.l1_5_time + self.smem_time
                + self.tmem_time + self.tensor_time + self.cuda_time + self.sfu_time
                + self.network_time)

    @property
    def total_full_overlap(self):
        """完全 overlap 时间（各单元取 max）。"""
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
    """一个操作组：一个或多个可互相 overlap 的操作。

    同一 group 内的操作使用不同硬件单元，可以 overlap。
    group 与 group 之间由 sync 分隔，默认串行。

    depends_on: 此 group 依赖的前驱 group 列表 (数据依赖)。
        调度时此 group 必须排在所有前驱之后。
        空列表 = 无依赖，可自由排序。
    """
    name: str
    usage: HardwareUsage
    depends_on: List['OpGroup'] = field(default_factory=list)
    is_loop: bool = False
    loop_node: Optional['LoopNode'] = None

    @property
    def latency(self):
        """此 group 的延迟 = 各单元取 max (roofline)。"""
        return self.usage.total_full_overlap


@dataclass
class LoopNode:
    """一个循环层次。

    sub_groups: 循环体内的 op groups（按 sync 分隔）
    num_iters: 循环迭代次数
    sw_pipeline_stage: 软件流水线深度 (1=无)
    """
    name: str
    sub_groups: List[OpGroup]
    num_iters: int
    sw_pipeline_stage: int = 1
    is_inner_loop: bool = False


# =====================================================================
# DAG 拓扑排序枚举 (自己实现, 无外部依赖)
# =====================================================================

def _build_dag(groups: List[OpGroup]) -> Tuple[List[List[int]], List[int], bool]:
    """从 groups 的 depends_on 构建 DAG (邻接表 + 入度)。

    Returns:
        (adj, in_degree, has_deps)
        adj[i] = [j, ...] 表示 i → j (i 是 j 的前驱)
        in_degree[j] = 入度
        has_deps = 是否存在任何依赖
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
    """枚举 DAG 的所有合法拓扑排序 (自己实现, 递归 DFS)。

    算法: Kahn 变体 — 每次从所有 in_degree=0 的节点中选一个,
    递归处理剩余图, 回溯时恢复状态。

    复杂度: O(n! / 依赖约束), 适用于 n <= 8 的小 DAG。
    无依赖时退化为 n! 全排列。
    """
    adj, in_degree, has_deps = _build_dag(groups)
    n = len(groups)

    if not has_deps:
        # 无任何依赖 → 全排列
        return [list(p) for p in itertools.permutations(range(n))]

    results = []
    current = []

    def dfs():
        if len(current) == n:
            results.append(list(current))
            return
        for i in range(n):
            if in_degree[i] == 0 and i not in current:
                # 选 i 加入排列
                current.append(i)
                # 更新后继入度
                for j in adj[i]:
                    in_degree[j] -= 1
                dfs()
                # 回溯
                current.pop()
                for j in adj[i]:
                    in_degree[j] += 1

    dfs()
    return results


def all_topological_sorts_networkx(groups: List[OpGroup]) -> List[List[int]]:
    """用 NetworkX 枚举所有拓扑排序 (用于交叉验证)。"""
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
    """验证自己实现与 NetworkX 结果一致。"""
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
# 辅助函数
# =====================================================================

def hw_usage_from_resources(ddr_io, l2_io, smem_io, arch,
                            tensor_flops=0, cuda_flops=0, sfu_flops=0,
                            max_util=0.9, l1_5_io=0, tmem_io=0):
    """从资源量 + arch bandwidth 构造 HardwareUsage (per-SM 时间)。"""
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
# ModelOverlap: 在给定 stage 下模拟 op groups 的 overlap
# =====================================================================

def simulate_schedule(groups: List[OpGroup], stage: int,
                      order: List[int]) -> Tuple[float, Dict]:
    """模拟一种 op group 排列顺序在给定 pipeline stage 下的延迟。

    stage=1 (无 pipeline): groups 间串行，同 group 内各单元 overlap
    stage>=2 (有 pipeline): 各硬件单元独立流水，延迟 = max(各单元总时间)

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
    """ModelOverlap: 枚举所有合法调度顺序 (尊重依赖), 选最优。

    Args:
        groups: op groups (可能带 depends_on 依赖)
        stage: 等效 pipeline depth
        try_all_orders: True = 枚举所有合法拓扑排序
                        False = 只用原始顺序

    Returns:
        (best_latency_per_iter, best_util)
    """
    n = len(groups)
    if n == 0:
        return 0.0, {}

    if not try_all_orders:
        return simulate_schedule(groups, stage, list(range(n)))

    # 枚举所有合法拓扑排序 (尊重 depends_on 依赖)
    legal_orders = all_topological_sorts_builtin(groups)

    if not legal_orders:
        # 有环或其他问题, 回退到原始顺序
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
# AnalyzeLoop: 递归分析循环层次
# =====================================================================

def analyze_loop(node: LoopNode, stage: int) -> Tuple[float, Dict]:
    """递归分析一个循环节点。

    对应论文算法的 AnalyzeLoop。

    Args:
        node: LoopNode
        stage: 从上层传入的等效 pipeline stage (occupancy × 外层 pipeline)

    Returns:
        (total_latency, util_dict)
    """
    new_stage = node.sw_pipeline_stage * stage

    if node.is_inner_loop:
        # 内层循环: 所有 sub_groups 作为一次迭代的 body
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

        # util 缩放为 total 级别 (per_iter → total)
        total_util = {k: v * node.num_iters for k, v in util_per_iter.items()}

        return total, total_util

    else:
        # 外层循环: 逐个处理 sub_groups
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
# OverlapAnalysis: 顶层入口
# =====================================================================

def overlap_analysis(root_node: LoopNode, tiles_per_sm: int = 1,
                     num_iters_override: int = None) -> Tuple[float, Dict]:
    """顶层 overlap 分析。"""
    if num_iters_override is not None:
        root_node.num_iters = num_iters_override

    per_tile_lat, util = analyze_loop(root_node, tiles_per_sm)
    return per_tile_lat, util


# =====================================================================
# 便捷构造函数
# =====================================================================

def make_op_group(name: str, arch, ddr_io=0, l2_io=0, smem_io=0,
                  tensor_flops=0, cuda_flops=0, sfu_flops=0,
                  max_util=0.9, depends_on=None, l1_5_io=0, tmem_io=0) -> OpGroup:
    """快速创建 OpGroup。

    Args:
        depends_on: 前驱 OpGroup 列表, 调度时此 group 必须排在它们之后。
        l1_5_io: L1.5 cache IO (bytes), 0 if no L1.5.
        tmem_io: Tensor Memory IO (bytes) via tcgen05 ld/st datapath, 0 if no TMEM.
    """
    usage = hw_usage_from_resources(
        ddr_io, l2_io, smem_io, arch,
        tensor_flops=tensor_flops, cuda_flops=cuda_flops,
        sfu_flops=sfu_flops, max_util=max_util, l1_5_io=l1_5_io, tmem_io=tmem_io)
    return OpGroup(name=name, usage=usage,
                   depends_on=depends_on if depends_on else [])


def make_loop(name: str, sub_groups: List[OpGroup], num_iters: int,
              sw_pipeline_stage: int = 1, is_inner: bool = True) -> LoopNode:
    """快速创建 LoopNode。"""
    return LoopNode(
        name=name, sub_groups=sub_groups,
        num_iters=num_iters, sw_pipeline_stage=sw_pipeline_stage,
        is_inner_loop=is_inner)


def make_loop_group(name: str, loop_node: LoopNode) -> OpGroup:
    """将 LoopNode 包装为 OpGroup (用于嵌套)。"""
    return OpGroup(name=name, usage=HardwareUsage(),
                   is_loop=True, loop_node=loop_node)


# =====================================================================
# 与 hete_reg_fusion / hete_smem_fusion 兼容的接口
# =====================================================================

def overlap_to_hete_post(per_tile_lat: float, util: Dict,
                         smem_footprint: float = 0,
                         reg_footprint: float = 0,
                         ddr_read_io: float = 0,
                         l2_read_io: float = 0,
                         l2_hit_rate: float = 0) -> tuple:
    """将 overlap_analysis 的输出转换为 hete_post_process 的 12-tuple 格式。"""
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
    """完整的 overlap 分析 + wave 调整, 输出 12-tuple。"""
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
