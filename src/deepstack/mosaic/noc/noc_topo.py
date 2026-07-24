from __future__ import annotations
from .traffic_matrix import TrafficMatrix

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Iterable, Optional, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from .energy_config import NocEnergyConfig
import math
import numpy as np
import logging
log = logging.getLogger(__name__)
# ---------------- 基本类型 ----------------

# 一个"物理链路"用 (layer_id, u, v) 唯一标识（有向）
# u, v 是该层的节点 ID（0..M*N-1 或 switch 的 -1 特殊点）
Link = Tuple[int, int, int]  # (layer, u, v)

# 一条"链条"是一串物理链路（hop 列表），并带一个权重（该条链子承载的 bytes）
@dataclass
class Chain:
    hops: List[Link]
    bytes: int
    # 层级状态轨迹：每一步后的全层坐标快照（外->内），用于人类可读
    states_coords: List[List[Tuple[int, int]]] = field(default_factory=list)


class PortSpread(Enum):
    EVEN = "even"        # 均分：把 (src,dst) 流量在多个出入口均摊为 k 份
    NEAREST = "nearest"  # 最近：选最近的一个出/入口（要求拓扑支持 bypass）


class TopoKind(Enum):
    SWITCH = "switch"        # [1 x N]，所有端口连到中心（记作 -1 节点）
    ALL2ALL = "all_to_all"   # [M x N]，跨组一跳抽象（常数 hop）
    RING = "ring"            # [1 x N], N 偶数
    CHAIN = "chain"          # [1 x N]，N 偶数
    MESH2D = "mesh2d"        # [M x N], M,N 偶数
    TORUS2D = "torus2d"      # [M x N], M,N 偶数
    MCHAIN_NRING = "mchain_nring"   # 行方向为 chain（不环绕），列方向为 ring（环绕）
    MRING_NCHAIN = "mring_nchain"   # 行方向为 ring（环绕），列方向为 chain（不环绕）


# ---------------- Topology 抽象 ----------------

@dataclass
class Topology:
    kind: TopoKind
    shape: Tuple[int, int]           # (M, N)，对 ring/chain/switch 用 (1, size)
    # 每层的单 hop 固有时延与链路带宽
    hop_latency: float = 1.0         # 单位自洽（例如秒/跳 或 cycle/跳）
    link_bandwidth: float = 1.0      # 单位与 bytes 匹配（例如 bytes/秒 或 bytes/cycle）
    # Switch 专用：中心聚合带宽（分别约束进入中心与离开中心的总带宽）
    switch_center_in_bw: Optional[float] = None
    switch_center_out_bw: Optional[float] = None
    # 端口分区规则（按你给的描述定义 in/out 的坐标集合）
    # 这些字段只对 mesh/torus/ring/chain 有意义，用于"上/下/左/右"的 in/out 划分
    # 字段是函数，返回一组"端口单元"的列表，每个端口单元对应一个物理节点 ID。
    left_in: Optional[callable] = None
    left_out: Optional[callable] = None
    right_in: Optional[callable] = None
    right_out: Optional[callable] = None
    up_in: Optional[callable] = None
    up_out: Optional[callable] = None
    down_in: Optional[callable] = None
    down_out: Optional[callable] = None
    _show_rates_in_repr: bool = field(
        default=False,
        repr=False,
        compare=False,
    )

    def __repr__(self) -> str:
        """Return rates only for explicitly caller-owned topology objects."""

        summary = (
            f"Topology(kind={self.kind.value!r}, shape={self.shape!r}"
        )
        if self._show_rates_in_repr:
            summary += (
                f", hop_latency={self.hop_latency!r}, "
                f"link_bandwidth={self.link_bandwidth!r}"
            )
            if self.switch_center_in_bw is not None:
                summary += (
                    f", switch_center_in_bw={self.switch_center_in_bw!r}, "
                    f"switch_center_out_bw={self.switch_center_out_bw!r}"
                )
        return summary + ")"

    # 把二维坐标映射到线性 ID
    def to_id(self, r: int, c: int) -> int:
        M, N = self.shape
        return r * N + c

    # 把线性 ID 映射回 (r, c)
    def to_rc(self, idx: int) -> Tuple[int, int]:
        M, N = self.shape
        return divmod(idx, N)

    # 某方向上的"端口列表"（每个端口是一个物理节点 ID）
    def ports(self, side: str) -> List[int]:
        # SWITCH：抽象为中心端口 -1（任意方向都走中心）
        if self.kind == TopoKind.SWITCH:
            return [-1]
        # ALL2ALL：若工厂提供了端口函数，优先使用；否则按线性 ID 均匀分配到四个方向
        if self.kind == TopoKind.ALL2ALL:
            fn = {
                "left_in": self.left_in, "left_out": self.left_out,
                "right_in": self.right_in, "right_out": self.right_out,
                "up_in": self.up_in, "up_out": self.up_out,
                "down_in": self.down_in, "down_out": self.down_out,
            }.get(side)
            if fn is not None:
                coords = fn(self.shape)
                return [self.to_id(r, c) for (r, c) in coords]
            M, N = self.shape
            S = M * N
            if S < 4:
                raise ValueError("ALL2ALL 需要至少 4 个节点以均匀分配四个方向的端口")
            idxs = list(range(S))
            group = {0: [], 1: [], 2: [], 3: []}
            for i in idxs:
                group[i % 4].append(i)
            mapping = {
                "left_in": group[0],  "left_out": group[0],
                "right_in": group[1], "right_out": group[1],
                "up_in": group[2],    "up_out": group[2],
                "down_in": group[3],  "down_out": group[3],
            }
            return mapping.get(side, [])
        assert self.kind in {TopoKind.MESH2D, TopoKind.TORUS2D, TopoKind.RING, TopoKind.CHAIN, TopoKind.MCHAIN_NRING, TopoKind.MRING_NCHAIN}
        fn = {
            "left_in": self.left_in, "left_out": self.left_out,
            "right_in": self.right_in, "right_out": self.right_out,
            "up_in": self.up_in, "up_out": self.up_out,
            "down_in": self.down_in, "down_out": self.down_out,
        }.get(side)
        if fn is None:
            return []
        coords = fn(self.shape)
        return [self.to_id(r, c) for (r, c) in coords]

    # 计算层内最短"曼哈顿距离"（对 torus 支持环绕），用于 NEAREST 端口选择
    def distance(self, a_id: int, b_id: int) -> int:
        if a_id == b_id:
            return 0
        if self.kind in {TopoKind.SWITCH, TopoKind.ALL2ALL}:
            return 1
        M, N = self.shape
        (ar, ac) = self.to_rc(a_id)
        (br, bc) = self.to_rc(b_id)
        if self.kind == TopoKind.MESH2D:
            return abs(ar - br) + abs(ac - bc)
        if self.kind == TopoKind.TORUS2D:
            dr = min((br - ar) % M, (ar - br) % M)
            dc = min((bc - ac) % N, (ac - bc) % N)
            return dr + dc
        if self.kind == TopoKind.MCHAIN_NRING:
            dr = abs(ar - br)
            dc = min((bc - ac) % N, (ac - bc) % N)
            return dr + dc
        if self.kind == TopoKind.MRING_NCHAIN:
            dr = min((br - ar) % M, (ar - br) % M)
            dc = abs(ac - bc)
            return dr + dc
        if self.kind == TopoKind.RING:
            # 1xN 环：只考虑列方向
            delta = abs(bc - ac)
            return min(delta, N - delta)
        if self.kind == TopoKind.CHAIN:
            return abs(bc - ac)
        return 0

    # 层内 XY 路由：返回有向边序列 [(u->v), ...]
    def route_intra(self, s_id: int, d_id: int) -> List[Tuple[int, int]]:
        if s_id == d_id:
            return []
        if self.kind == TopoKind.SWITCH:
            # switch 层抽象：节点到中心(-1)，再出去
            # 细化边界：避免 (-1,-1) 自环
            if s_id == -1 and d_id == -1:
                return []
            if s_id == -1:
                return [(-1, d_id)]
            if d_id == -1:
                return [(s_id, -1)]
            return [(s_id, -1), (-1, d_id)]
        if self.kind == TopoKind.ALL2ALL:
            # 两种可能：
            # 1) 同方向端口可一跳直达（抽象化）：s->d
            # 2) 或者理解为通过"该方向的汇聚节点"（此处仍保持简化为直达以避免虚增 hop）
            if s_id == -1 and d_id == -1:
                return []
            if s_id == -1:
                return [(-1, d_id)]
            if d_id == -1:
                return [(s_id, -1)]
            return [(s_id, d_id)]

        M, N = self.shape
        (sr, sc) = self.to_rc(s_id)
        (dr, dc) = self.to_rc(d_id)

        hops: List[Tuple[int, int]] = []

        def id_of(r, c): return self.to_id(r % M, c % N)

        if self.kind in {TopoKind.MESH2D, TopoKind.TORUS2D, TopoKind.MCHAIN_NRING, TopoKind.MRING_NCHAIN}:
            # 先行后列（X->Y）
            # 行方向：mesh 走直线；torus 走更短的方向（考虑环绕）
            r_path: List[int] = []
            if self.kind in {TopoKind.MESH2D, TopoKind.MCHAIN_NRING}:
                step = 1 if dr >= sr else -1
                for r in range(sr, dr, step):
                    r_path.append((r, sc))
                    hops.append((id_of(r, sc), id_of(r + step, sc)))
            else:  # row is ring (TORUS2D or MRING_NCHAIN)
                # 选择 |Δ| 与 M-|Δ| 的更小者方向
                delta = (dr - sr) % M
                neg = (sr - dr) % M
                if delta <= neg:
                    for k in range(delta):
                        r = (sr + k) % M
                        hops.append((id_of(r, sc), id_of(r + 1, sc)))
                else:
                    for k in range(neg):
                        r = (sr - k) % M
                        hops.append((id_of(r, sc), id_of(r - 1, sc)))

            # 列方向
            if self.kind in {TopoKind.MESH2D, TopoKind.MRING_NCHAIN}:
                step = 1 if dc >= sc else -1
                for c in range(sc, dc, step):
                    hops.append((id_of(dr, c), id_of(dr, c + step)))
            else:  # col is ring (TORUS2D or MCHAIN_NRING)
                delta = (dc - sc) % N
                neg = (sc - dc) % N
                if delta <= neg:
                    for k in range(delta):
                        c = (sc + k) % N
                        hops.append((id_of(dr, c), id_of(dr, c + 1)))
                else:
                    for k in range(neg):
                        c = (sc - k) % N
                        hops.append((id_of(dr, c), id_of(dr, c - 1)))

            return hops

        if self.kind == TopoKind.RING:
            # 1xN，按短向走（左右之一）
            N = self.shape[1]
            delta = (dc - sc) % N
            neg = (sc - dc) % N
            if delta <= neg:
                for k in range(delta):
                    c = (sc + k) % N
                    hops.append((self.to_id(0, c), self.to_id(0, (c + 1) % N)))
            else:
                for k in range(neg):
                    c = (sc - k) % N
                    hops.append((self.to_id(0, c), self.to_id(0, (c - 1) % N)))
            return hops

        if self.kind == TopoKind.CHAIN:
            step = 1 if dc >= sc else -1
            for c in range(sc, dc, step):
                hops.append((self.to_id(0, c), self.to_id(0, c + step)))
            return hops

        raise NotImplementedError(f"route_intra not implemented for {self.kind}")


# ---------------- 端口规则（按你给出的划分） ----------------

def _range2(a, b):
    # [a:b] 的闭开区间转列表
    return list(range(a, b)) if a < b else []


def make_mesh_or_torus(M: int, N: int, kind: TopoKind, *, hop_latency: float = 1.0, link_bandwidth: float = 1.0) -> Topology:
    assert kind in (TopoKind.MESH2D, TopoKind.TORUS2D, TopoKind.MCHAIN_NRING, TopoKind.MRING_NCHAIN)
    assert M >= 1 and N >= 1, "M,N must be >= 1 for mesh/torus"
    # 兼容 M==1 或 N==1：维度为 1 时，in/out 端口映射到同一组坐标（且不能为空）
    assert (M == 1 or M % 2 == 0) and (N == 1 or N % 2 == 0), "M,N must be even unless it's 1 for mesh/torus"

    def _split_in_out_indices(dim: int):
        """
        把 [0..dim-1] 沿中点拆成两组索引：
        - dim>1：in 用后半段 [dim/2 .. dim-1]，out 用前半段 [0 .. dim/2-1]
        - dim==1：in/out 都是 [0]（同一个点，且不为空）
        """
        if dim == 1:
            return (0,), (0,)
        return tuple(range(dim // 2, dim)), tuple(range(0, dim // 2))

    rows_in, rows_out = _split_in_out_indices(M)
    cols_in, cols_out = _split_in_out_indices(N)

    # 预先把 8 个端口坐标算好，便于阅读/审阅；Topology 回调里只返回常量列表的拷贝
    left_in_ports = tuple((r, 0) for r in rows_in)
    left_out_ports = tuple((r, 0) for r in rows_out)
    right_in_ports = tuple((r, N - 1) for r in rows_out)
    right_out_ports = tuple((r, N - 1) for r in rows_in)

    up_in_ports = tuple((M - 1, c) for c in cols_in)
    up_out_ports = tuple((M - 1, c) for c in cols_out)
    down_in_ports = tuple((0, c) for c in cols_out)
    down_out_ports = tuple((0, c) for c in cols_in)

    def constant_ports(ports):
        return lambda shape, _ports=ports: list(_ports)

    return Topology(
        kind=kind, shape=(M, N), hop_latency=hop_latency, link_bandwidth=link_bandwidth,
        left_in=constant_ports(left_in_ports),
        left_out=constant_ports(left_out_ports),
        right_in=constant_ports(right_in_ports),
        right_out=constant_ports(right_out_ports),
        up_in=constant_ports(up_in_ports),
        up_out=constant_ports(up_out_ports),
        down_in=constant_ports(down_in_ports),
        down_out=constant_ports(down_out_ports),
    )


def make_ring(N: int, *, hop_latency: float = 1.0, link_bandwidth: float = 1.0) -> Topology:
    assert N % 2 == 0, "N must be even for ring"
    def left_in(shape):
        return [(0, 0)]
    def left_out(shape):
        return [(0, 0)]
    def right_in(shape):
        M, N = shape
        return [(0, N-1)]
    def right_out(shape):
        M, N = shape
        return [(0, N-1)]
    def up_out(shape):
        M, N = shape
        return [(0, c) for c in range(0, N//2)]
    def up_in(shape):
        M, N = shape
        return [(0, c) for c in range(N//2, N)]
    def down_out(shape):
        M, N = shape
        return [(0, c) for c in range(N//2, N)]
    def down_in(shape):
        M, N = shape
        return [(0, c) for c in range(0, N//2)]
    return Topology(
        kind=TopoKind.RING, shape=(1, N), hop_latency=hop_latency, link_bandwidth=link_bandwidth,
        left_in=left_in, left_out=left_out, right_in=right_in, right_out=right_out,
        up_in=up_in, up_out=up_out, down_in=down_in, down_out=down_out
    )


def make_chain(N: int, *, hop_latency: float = 1.0, link_bandwidth: float = 1.0) -> Topology:
    assert N % 2 == 0, "N must be even for chain"
    def left_in(shape): return [(0, 0)]
    def left_out(shape): return [(0, 0)]
    def right_in(shape):
        M, N = shape
        return [(0, N-1)]
    def right_out(shape):
        M, N = shape
        return [(0, N-1)]
    # 上下端口按 ring 规则拆半（用于多层汇聚/下沉时的均分）
    def up_out(shape):
        M, N = shape
        return [(0, c) for c in range(0, N//2)]
    def up_in(shape):
        M, N = shape
        return [(0, c) for c in range(N//2, N)]
    def down_out(shape):
        M, N = shape
        return [(0, c) for c in range(N//2, N)]
    def down_in(shape):
        M, N = shape
        return [(0, c) for c in range(0, N//2)]
    return Topology(
        kind=TopoKind.CHAIN, shape=(1, N), hop_latency=hop_latency, link_bandwidth=link_bandwidth,
        left_in=left_in, left_out=left_out, right_in=right_in, right_out=right_out,
        up_in=up_in, up_out=up_out, down_in=down_in, down_out=down_out
    )


def make_switch(
    N: int,
    *,
    hop_latency: float = 1.0,
    link_bandwidth: float = 1.0,
    switch_center_in_bw: Optional[float] = None,
    switch_center_out_bw: Optional[float] = None,
) -> Topology:
    # [1 x N]，所有端口连中心 -1（中心端口由 Topology.ports 返回 [-1] 表示）
    return Topology(
        kind=TopoKind.SWITCH,
        shape=(1, N),
        hop_latency=hop_latency,
        link_bandwidth=link_bandwidth,
        switch_center_in_bw=switch_center_in_bw,
        switch_center_out_bw=switch_center_out_bw,
    )


def make_all2all(M: int, N: int, *, hop_latency: float = 1.0, link_bandwidth: float = 1.0) -> Topology:
    # [M x N]，为四个方向提供均匀分配的端口函数（每节点至少一进一出）
    assert M * N >= 4, "ALL2ALL 需要至少 4 个节点"

    def _mk_group(shape):
        Mx, Nx = shape
        S = Mx * Nx
        ids = list(range(S))
        g = {0: [], 1: [], 2: [], 3: []}
        for i in ids:
            g[i % 4].append((i // Nx, i % Nx))
        return g

    def _left_in(shape):
        return _mk_group(shape)[0]
    def _left_out(shape):
        return _mk_group(shape)[0]
    def _right_in(shape):
        return _mk_group(shape)[1]
    def _right_out(shape):
        return _mk_group(shape)[1]
    def _up_in(shape):
        return _mk_group(shape)[2]
    def _up_out(shape):
        return _mk_group(shape)[2]
    def _down_in(shape):
        return _mk_group(shape)[3]
    def _down_out(shape):
        return _mk_group(shape)[3]

    return Topology(
        kind=TopoKind.ALL2ALL, shape=(M, N), hop_latency=hop_latency, link_bandwidth=link_bandwidth,
        left_in=_left_in, left_out=_left_out,
        right_in=_right_in, right_out=_right_out,
        up_in=_up_in, up_out=_up_out,
        down_in=_down_in, down_out=_down_out,
    )


# ---------------- 分层拓扑与路由 ----------------


@dataclass
class Hierarchy:
    # 从外到内：L3 (index 2), L2 (1), L1 (0)——为了直观，这里我们按 [L3, L2, L1] 存放
    layers: List[Topology]          # 长度 <= 3
    port_spread: PortSpread = PortSpread.EVEN  # 端口选择策略（EVEN/NEAREST）
    name: str = "default_noc_hierarchy"
    # 对于 EVENS：为每个 (src,dst) 在每次"跨层"动作时，均匀分裂成 K 份（K=端口数）
    # 对于 NEAREST：只选择一个最近端口（需拓扑具备 bypass）

    # 将逻辑"节点 ID (0..N-1)"映射到每层的坐标（r,c）。
    # 默认实现按层形状把线性 ID 展开到内层（L1）的 (r,c)，外层 (L2/L3) 则对 (r,c) 进行整分块上取整。
    # 你也可以传入自定义函数，精确描述你在注释里的对齐（如 [[0,0],[0,0],[0,2]] 等）。
    node_mapper: Optional[callable] = None  # f(node_id: int, layers: List[Topology]) -> List[Tuple[int,int]]
    # Optional caller-owned energy profile; None selects the bundled reference path.
    energy_config: Optional["NocEnergyConfig"] = None

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        """Return topology structure without exposing latency/BW in logs."""

        layers = ", ".join(repr(layer) for layer in self.layers)
        return (
            f"Hierarchy(name={self.name!r}, layers=[{layers}], "
            f"port_spread={self.port_spread.value!r})"
        )

    @property
    def num_devices(self) -> int:
        """返回该 Hierarchy 覆盖的设备总数，等于各层 shape 的连乘积。"""
        n = 1
        for layer in self.layers:
            n *= layer.shape[0] * layer.shape[1]
        return n

    # -------- 双向映射：coords <-> nid --------
    def coords_to_nid(self, coords: List[Tuple[int, int]]) -> int:
        """
        按"最内层连续"的规则，将外->内各层坐标 [(r_L3,c_L3), (r_L2,c_L2), (r_L1,c_L1)]
        映射为单个线性 nid。公式：
          nid = k_L1
                + k_L2 * (M1*N1)
                + k_L3 * (M1*N1*M2*N2)
        其中 k_Lx = r_Lx * N_Lx + c_Lx。
        通用到 <=3 任意层数。
        """
        assert len(coords) == len(self.layers), "coords 层数需与 layers 一致"
        nid = 0
        stride = 1
        # 从内到外叠加（内层 stride=1）
        for (r, c), topo in zip(reversed(coords), reversed(self.layers)):
            M, N = topo.shape
            assert 0 <= r < M and 0 <= c < N
            k = r * N + c
            nid += k * stride
            stride *= (M * N)
        return nid

    def nid_to_coords(self, nid: int) -> List[Tuple[int, int]]:
        """
        线性 nid 还原为外->内各层坐标，遵循与 coords_to_nid 相反的分解：
          依次对 size_L1, size_L2, size_L3 取余/整除，得到 k_L1,k_L2,k_L3，
          再分解 k_Lx -> (r_Lx, c_Lx) 其中 r=k//N, c=k%N。
        """
        coords_rev: List[Tuple[int, int]] = []  # 内->外
        rem = int(nid)
        for topo in reversed(self.layers):  # 先内层
            M, N = topo.shape
            size = M * N
            k = rem % size
            rem //= size
            r = k // N
            c = k % N
            coords_rev.append((r, c))
        # 若 rem>0，说明 nid 超容量，这里保留容错：直接忽略高位（也可改为抛错）
        return list(reversed(coords_rev))

    def map_node(self, nid: int) -> List[Tuple[int, int]]:
        if self.node_mapper:
            return self.node_mapper(nid, self.layers)
        # 默认：使用"最内层连续"的双向映射
        return self.nid_to_coords(nid)

    # 找到从外到里，第一层"二者不相等"的层 index（外层优先）
    def first_diff_layer(self, s_coords: List[Tuple[int,int]], d_coords: List[Tuple[int,int]]) -> int:
        for i, (a, b) in enumerate(zip(s_coords, d_coords)):
            if a != b:
                return i
        return len(s_coords) - 1  # 全相同则返回最内层

    # 计算一次跨层动作时的端口集合（例如 L3 -> L2 的"down_out/down_in"）
    def ports_for_cross(self, topo: Topology, direction: str) -> List[int]:
        # direction in {"up_out","up_in","down_out","down_in","left_out","left_in","right_out","right_in"}
        return topo.ports(direction)

    # 核心路由：把 (src, dst, bytes) 映射为若干条 Chain（考虑 split）
    def route(self, src: int, dst: int, bytes_value: int) -> List[Chain]:
        # 映射到各层坐标（外->内）
        s_coords = self.map_node(src)
        d_coords = self.map_node(dst)
        # print(f"s_coords: {s_coords}, d_coords: {d_coords}")
        log.debug("s_coords: %s, d_coords: %s", s_coords, d_coords)
        L = len(self.layers)
        chains: List[Chain] = [Chain(hops=[], bytes=bytes_value)]
        # 初始快照
        for ch in chains:
            ch.states_coords.append(list(s_coords))
        # 若最内层为 switch，先将 L1 置为中心 (-1,-1)
        if self.layers[-1].kind == TopoKind.SWITCH:
            for ch in chains:
                init = list(ch.states_coords[-1])
                init[-1] = (-1, -1)
                ch.states_coords.append(init)

        # 起始层：从"最外层发生分歧"的层开始（若希望总从最外层，可改为 0）
        start_layer = self.first_diff_layer(s_coords, d_coords)

        # 当前各层"起点"ID（会在推进时按跨层入口更新内层起点）
        cur_start_ids = [self.layers[i].to_id(*s_coords[i]) for i in range(L)]

        # 根据单步 hop 判断方向（up/down/left/right）
        def hop_dir(topo: Topology, u: int, v: int) -> str:
            (ur, uc) = topo.to_rc(u)
            (vr, vc) = topo.to_rc(v)
            M, N = topo.shape
            if ur != vr:
                # 行方向
                if topo.kind in {TopoKind.TORUS2D, TopoKind.MRING_NCHAIN}:
                    # 约定：一步 r := (r-1)%M 视作 "down"（符合 0->3 的期望），否则 "up"
                    return "down" if (vr == (ur - 1) % M) else "up"
                else:
                    return "up" if vr > ur else "down"
            else:
                # 列方向
                if topo.kind in {TopoKind.TORUS2D, TopoKind.RING, TopoKind.MCHAIN_NRING}:
                    # 一步 c := (c-1)%N 视作 "left"，否则 "right"
                    return "left" if (vc == (uc - 1) % N) else "right"
                else:
                    return "right" if vc > uc else "left"

        # 端口集合名映射
        def port_sides(direction: str) -> tuple[str, str]:
            if direction == "down":
                return "down_out", "up_in"
            if direction == "up":
                return "up_out", "down_in"
            if direction == "right":
                return "right_out", "left_in"
            if direction == "left":
                return "left_out", "right_in"
            raise ValueError(f"Invalid direction: {direction}")


        # 均匀选择端口索引（确定性；不分裂链条，避免指数膨胀）
        def select_index(num: int, layer_idx: int, hop_idx: int) -> int:
            if num <= 0:
                return -1
            # 稳定整数混合，避免 Python 进程随机盐影响
            x = (int(src) * 1315423911) ^ (int(dst) * 2654435761) ^ (int(layer_idx) * 97) ^ int(hop_idx)
            if x < 0:
                x = -x
            return x % num

        # 去重追加快照
        def append_state(ch: Chain, snap: List[Tuple[int, int]]):
            if not ch.states_coords or ch.states_coords[-1] != snap:
                ch.states_coords.append(snap)

        # 外层驱动：逐层向内推进
        for layer_idx in range(start_layer, L):
            topo = self.layers[layer_idx]
            s_id = cur_start_ids[layer_idx]
            d_id = self.layers[layer_idx].to_id(*d_coords[layer_idx])

            # 1) 在本层：从当前起点到目标坐标的 XY（或环绕）路径（先只记录 hop，不立即生成快照）
            outer_hops = topo.route_intra(s_id, d_id)
            for ch in chains:
                for (u, v) in outer_hops:
                    ch.hops.append((layer_idx, u, v))

            # 若还有更内层：
            if layer_idx < L - 1:
                inner_topo = self.layers[layer_idx + 1]

                # 2) 对每个外层 hop，依据方向在两层之间放置端口占位，并把"内层起点"推进到对应入口端口
                last_snapshot = None
                for hop_idx, (u, v) in enumerate(outer_hops):
                    direction = hop_dir(topo, u, v)
                    out_side, in_side = port_sides(direction)
                    # 外层用于链路统计的端口（保持不变）
                    out_ports_outer = topo.ports(out_side)
                    in_ports_inner = inner_topo.ports(in_side)
                    # 为了人类可读快照：在内层也给"出方向"分配端口（egress on inner layer）
                    out_ports_inner = inner_topo.ports(out_side)
                    K = min(len(out_ports_outer), len(in_ports_inner), len(out_ports_inner) if out_ports_inner else len(in_ports_inner))
                    if K <= 0:
                        continue
                    sel = select_index(K, layer_idx, hop_idx)
                    out_id = out_ports_outer[sel]
                    in_id = in_ports_inner[sel]
                    inner_out_id = out_ports_inner[sel] if out_ports_inner else in_id

                    # 先在内层从当前坐标路由到"本次外层 hop 所需的内层出方向端口"
                    inner_cur = cur_start_ids[layer_idx + 1]
                    if inner_cur != inner_out_id:
                        inner_steps_to_out = inner_topo.route_intra(inner_cur, inner_out_id)
                        for ch in chains:
                            snap_base = list(ch.states_coords[-1]) if last_snapshot is None else list(last_snapshot)
                            for (uu, vv) in inner_steps_to_out:
                                ch.hops.append((layer_idx + 1, uu, vv))
                                if self.layers[-1].kind == TopoKind.SWITCH:
                                    vr2, vc2 = self.layers[layer_idx + 1].to_rc(vv) if vv != -1 else (-1, -1)
                                    snap_step = list(snap_base)
                                    snap_step[layer_idx + 1] = (vr2, vc2)

                                    append_state(ch, snap_step)
                                    snap_base = snap_step
                                else:
                                    ll_topo = self.layers[-1]
                                    ll_dir = hop_dir(ll_topo, uu, vv)
                                    ll_out_side, ll_in_side = port_sides(ll_dir)
                                    # 外层用于链路统计的端口（保持不变）
                                    ll_out_ports_outer = inner_topo.ports(ll_out_side)
                                    ll_in_ports_inner = ll_topo.ports(ll_in_side)
                                    # 为了人类可读快照：在内层也给"出方向"分配端口（egress on inner layer）
                                    ll_out_ports_inner = ll_topo.ports(ll_out_side)
                                    ll_K = min(len(ll_out_ports_outer), len(ll_in_ports_inner), len(ll_out_ports_inner) if ll_out_ports_inner else len(ll_in_ports_inner))
                                    if ll_K <= 0:
                                        continue
                                    ll_sel = select_index(ll_K, layer_idx+1, hop_idx)
                                    ll_out_id = ll_out_ports_outer[ll_sel]
                                    ll_in_id = ll_in_ports_inner[ll_sel]
                                    ll_inner_out_id = ll_out_ports_inner[ll_sel] if ll_out_ports_inner else ll_in_id
                                    ll_inner_cur = cur_start_ids[-1]
                                    # 在最内层从当前坐标路由到"本次内层 hop 所需的最内层出方向端口"
                                    if ll_inner_cur != ll_inner_out_id:
                                        ll_inner_steps_to_out = ll_topo.route_intra(ll_inner_cur, ll_inner_out_id)
                                        for (uuu, vvv) in ll_inner_steps_to_out:
                                            ch.hops.append((-1, uuu, vvv))
                                            vr3, vc3 = self.layers[-1].to_rc(vvv) if vvv != -1 else (-1, -1)
                                            snap_step2 = list(snap_base)
                                            snap_step2[-1] = (vr3, vc3)
                                            append_state(ch, snap_step2)
                                            snap_base = snap_step2
                                    # 跨层
                                    snap_step3 = list(snap_base)
                                    snap_step3[layer_idx + 1] = self.layers[layer_idx + 1].to_rc(vv) if vv != -1 else (-1, -1)
                                    snap_step3[-1] = self.layers[-1].to_rc(ll_in_id) if ll_in_id != -1 else (-1, -1)
                                    append_state(ch, snap_step3)
                                    # 路由到上一层出口
                                    if ll_in_id != vv:
                                        ll_steps_to_outer = ll_topo.route_intra(ll_in_id, vv)
                                        for (uuu, vvv) in ll_steps_to_outer:
                                            ch.hops.append((-1, uuu, vvv))
                                            vr3, vc3 = self.layers[-1].to_rc(vvv) if vvv != -1 else (-1, -1)
                                            snap_step4 = list(snap_step3)
                                            snap_step4[-1] = (vr3, vc3)
                                            append_state(ch, snap_step4)
                                            snap_base = snap_step4
                                    cur_start_ids[-1] = ll_inner_out_id
                            last_snapshot = snap_base
                        cur_start_ids[layer_idx + 1] = inner_out_id

                    for ch in chains:
                        ch.hops.append((layer_idx, out_id, out_id))
                        ch.hops.append((layer_idx + 1, in_id, in_id))
                        # 状态轨迹：先记录"内层出方向端口占位"，外层仍在 u 上
                        ur, uc = self.layers[layer_idx].to_rc(u) if u != -1 else (-1, -1)
                        ior, ioc = self.layers[layer_idx + 1].to_rc(inner_out_id) if inner_out_id != -1 else (-1, -1)
                        prev = list(ch.states_coords[-1]) if last_snapshot is None else list(last_snapshot)
                        snap_out = list(prev)
                        snap_out[layer_idx] = (ur, uc)
                        snap_out[layer_idx + 1] = (ior, ioc)
                        append_state(ch, snap_out)
                        last_snapshot = snap_out
                        # 状态轨迹：执行外层 hop 后，外层到 v，内层到入口端口
                        vr, vc = self.layers[layer_idx].to_rc(v) if v != -1 else (-1, -1)
                        inr, inc = self.layers[layer_idx + 1].to_rc(in_id) if in_id != -1 else (-1, -1)
                        snap_in = list(last_snapshot)
                        snap_in[layer_idx] = (vr, vc)
                        snap_in[layer_idx + 1] = (inr, inc)
                        if self.layers[-1].kind == TopoKind.SWITCH:
                            append_state(ch, snap_in)
                        else:
                            ll_in_ports_inner = self.layers[-1].ports(in_side)
                            ll_sel = select_index(len(ll_in_ports_inner), -1, hop_idx)
                            ll_in_id = ll_in_ports_inner[ll_sel]
                            snap_in[-1] = self.layers[-1].to_rc(ll_in_id)
                            append_state(ch, snap_in)
                            cur_start_ids[-1] = ll_in_id
                        last_snapshot = snap_in
                    # 推进下一层的"当前起点"
                    cur_start_ids[layer_idx + 1] = in_id

                    # 在本次外层 hop 之后，立刻在内层从入口路由到"下一步需要的出口"或最终目标
                    inner_cur = cur_start_ids[layer_idx + 1]
                    if hop_idx < len(outer_hops) - 1:
                        # 还有下一步外层 hop：计算下一个方向所需的内层出口
                        next_u, next_v = outer_hops[hop_idx + 1]
                        next_dir = hop_dir(topo, next_u, next_v)
                        next_out_side, _ = port_sides(next_dir)
                        next_out_ports_inner = inner_topo.ports(next_out_side)
                        if next_out_ports_inner:
                            sel_next = select_index(len(next_out_ports_inner), layer_idx, hop_idx + 1)
                            inner_out_next = next_out_ports_inner[sel_next]
                        else:
                            inner_out_next = inner_cur
                        inner_steps = inner_topo.route_intra(inner_cur, inner_out_next)
                        for ch in chains:
                            snap_base = list(last_snapshot)
                            for (uu, vv) in inner_steps:
                                ch.hops.append((layer_idx + 1, uu, vv))
                                if self.layers[-1].kind == TopoKind.SWITCH:
                                    vr2, vc2 = self.layers[layer_idx + 1].to_rc(vv) if vv != -1 else (-1, -1)
                                    snap_step = list(snap_base)
                                    snap_step[layer_idx + 1] = (vr2, vc2)
                                    append_state(ch, snap_step)
                                    snap_base = snap_step
                                else:
                                    ll_topo = self.layers[-1]
                                    ll_dir = hop_dir(ll_topo, uu, vv)
                                    ll_out_side, ll_in_side = port_sides(ll_dir)
                                    # 外层用于链路统计的端口（保持不变）
                                    ll_out_ports_outer = inner_topo.ports(ll_out_side)
                                    ll_in_ports_inner = ll_topo.ports(ll_in_side)
                                    # 为了人类可读快照：在内层也给"出方向"分配端口（egress on inner layer）
                                    ll_out_ports_inner = ll_topo.ports(ll_out_side)
                                    ll_K = min(len(ll_out_ports_outer), len(ll_in_ports_inner), len(ll_out_ports_inner) if ll_out_ports_inner else len(ll_in_ports_inner))
                                    if ll_K <= 0:
                                        continue
                                    ll_sel = select_index(ll_K, layer_idx+1, hop_idx)
                                    ll_out_id = ll_out_ports_outer[ll_sel]
                                    ll_in_id = ll_in_ports_inner[ll_sel]
                                    ll_inner_out_id = ll_out_ports_inner[ll_sel] if ll_out_ports_inner else ll_in_id
                                    ll_inner_cur = cur_start_ids[-1]
                                    # 在最内层从当前坐标路由到"本次内层 hop 所需的最内层出方向端口"
                                    if ll_inner_cur != ll_inner_out_id:
                                        ll_inner_steps_to_out = ll_topo.route_intra(ll_inner_cur, ll_inner_out_id)
                                        for (uuu, vvv) in ll_inner_steps_to_out:
                                            ch.hops.append((-1, uuu, vvv))
                                            vr3, vc3 = self.layers[-1].to_rc(vvv) if vvv != -1 else (-1, -1)
                                            snap_step2 = list(snap_base)
                                            snap_step2[-1] = (vr3, vc3)
                                            append_state(ch, snap_step2)
                                            snap_base = snap_step2
                                    # 跨层
                                    snap_step3 = list(snap_base)
                                    snap_step3[layer_idx + 1] = self.layers[layer_idx + 1].to_rc(vv) if vv != -1 else (-1, -1)
                                    snap_step3[-1] = self.layers[-1].to_rc(ll_in_id) if ll_in_id != -1 else (-1, -1)
                                    append_state(ch, snap_step3)
                                    # 路由到上一层出口
                                    if ll_in_id != vv:
                                        ll_steps_to_outer = ll_topo.route_intra(ll_in_id, vv)
                                        for (uuu, vvv) in ll_steps_to_outer:
                                            ch.hops.append((-1, uuu, vvv))
                                            vr3, vc3 = self.layers[-1].to_rc(vvv) if vvv != -1 else (-1, -1)
                                            snap_step4 = list(snap_step3)
                                            snap_step4[-1] = (vr3, vc3)
                                            append_state(ch, snap_step4)
                                            snap_base = snap_step4
                                    cur_start_ids[-1] = ll_inner_out_id
                        cur_start_ids[layer_idx + 1] = inner_out_next
                        last_snapshot = chains[0].states_coords[-1]
                    else:
                        # 最后一个外层 hop：路由到内层最终目标
                        inner_target = inner_topo.to_id(*d_coords[layer_idx + 1])
                        inner_steps = inner_topo.route_intra(inner_cur, inner_target)
                        for ch in chains:
                            snap_base = list(last_snapshot)
                            for (uu, vv) in inner_steps:
                                ch.hops.append((layer_idx + 1, uu, vv))
                                if self.layers[-1].kind == TopoKind.SWITCH:
                                    vr2, vc2 = self.layers[layer_idx + 1].to_rc(vv) if vv != -1 else (-1, -1)
                                    snap_step = list(snap_base)
                                    snap_step[layer_idx + 1] = (vr2, vc2)
                                    append_state(ch, snap_step)
                                    snap_base = snap_step
                                else:
                                    ll_topo = self.layers[-1]
                                    ll_dir = hop_dir(ll_topo, uu, vv)
                                    ll_out_side, ll_in_side = port_sides(ll_dir)
                                    # 外层用于链路统计的端口（保持不变）
                                    ll_out_ports_outer = inner_topo.ports(ll_out_side)
                                    ll_in_ports_inner = ll_topo.ports(ll_in_side)
                                    # 为了人类可读快照：在内层也给"出方向"分配端口（egress on inner layer）
                                    ll_out_ports_inner = ll_topo.ports(ll_out_side)
                                    ll_K = min(len(ll_out_ports_outer), len(ll_in_ports_inner), len(ll_out_ports_inner) if ll_out_ports_inner else len(ll_in_ports_inner))
                                    if ll_K <= 0:
                                        continue
                                    ll_sel = select_index(ll_K, layer_idx+1, hop_idx)
                                    ll_out_id = ll_out_ports_outer[ll_sel]
                                    ll_in_id = ll_in_ports_inner[ll_sel]
                                    ll_inner_out_id = ll_out_ports_inner[ll_sel] if ll_out_ports_inner else ll_in_id
                                    ll_inner_cur = cur_start_ids[-1]
                                    # 在最内层从当前坐标路由到"本次内层 hop 所需的最内层出方向端口"
                                    if ll_inner_cur != ll_inner_out_id:
                                        ll_inner_steps_to_out = ll_topo.route_intra(ll_inner_cur, ll_inner_out_id)
                                        for (uuu, vvv) in ll_inner_steps_to_out:
                                            ch.hops.append((-1, uuu, vvv))
                                            vr3, vc3 = self.layers[-1].to_rc(vvv) if vvv != -1 else (-1, -1)
                                            snap_step2 = list(snap_base)
                                            snap_step2[-1] = (vr3, vc3)
                                            append_state(ch, snap_step2)
                                            snap_base = snap_step2
                                    # 跨层
                                    snap_step3 = list(snap_base)
                                    snap_step3[layer_idx + 1] = self.layers[layer_idx + 1].to_rc(vv) if vv != -1 else (-1, -1)
                                    snap_step3[-1] = self.layers[-1].to_rc(ll_in_id) if ll_in_id != -1 else (-1, -1)
                                    append_state(ch, snap_step3)
                                    # 路由到上一层出口
                                    if ll_in_id != vv:
                                        ll_steps_to_outer = ll_topo.route_intra(ll_in_id, vv)
                                        for (uuu, vvv) in ll_steps_to_outer:
                                            ch.hops.append((-1, uuu, vvv))
                                            vr3, vc3 = self.layers[-1].to_rc(vvv) if vvv != -1 else (-1, -1)
                                            snap_step4 = list(snap_step3)
                                            snap_step4[-1] = (vr3, vc3)
                                            append_state(ch, snap_step4)
                                            snap_base = snap_step4
                                    cur_start_ids[-1] = ll_inner_out_id
                        cur_start_ids[layer_idx + 1] = inner_target
                        last_snapshot = chains[0].states_coords[-1]

                # 3) 外层走完后：内层已在每个外层 hop 之后即时路由，无需再整体路由

        # 尾声：若最内层仍停留在入口（如 switch 的 -1,-1），补记最后从入口到目标端口的一步快照
        for ch in chains:
            if ch.states_coords:
                last = ch.states_coords[-1]
                target = list(d_coords)
                if last[-1] != target[-1]:
                    if self.layers[-1].kind == TopoKind.SWITCH:
                        final_snap = list(last)
                        final_snap[-1] = target[-1]
                        if final_snap != last:
                            ch.states_coords.append(final_snap)
                    else:
                        self_id = self.layers[-1].to_id(*last[-1])
                        target_id = self.layers[-1].to_id(*target[-1])
                        final_steps = self.layers[-1].route_intra(self_id, target_id)
                        for (uu, vv) in final_steps:
                            ch.hops.append((L - 1, uu, vv))
                            vr, vc = self.layers[-1].to_rc(vv) if vv != -1 else (-1, -1)
                            final_snap = list(ch.states_coords[-1])
                            final_snap[-1] = (vr, vc)
                            ch.states_coords.append(final_snap)

        return chains


# ---------------- 统计与延迟计算 ----------------

@dataclass
class RouteStats:
    max_logical_hops: int                 # 所有 (src,dst) 的链条里，最大的 hop 数（单条链计算）
    max_hop_latency: float                # 按层 hop 延迟求和后的最大链路总 hop 时延
    # 注意：此处的 Link 端点 u,v 现表示"扩展全局节点 ID（含外层坐标与内层锚点）"，不是本层局部 ID。
    link_bytes: Dict[Link, int]           # 每条"全局物理边"累计的字节数 (layer_idx, ext_u, ext_v)
    max_link_load: Tuple[Link, int]       # (链路, bytes)
    max_link_time: Tuple[Link, float]     # (链路, bytes/bw)
    # 单 stage latency = max_hop_latency + max_link_time[1]
    stage_latency: float

# ---------------- 便捷工厂与示例 ----------------

# def example_hierarchy_for_your_comment() -> Hierarchy:
#     """
#     示例：
#       L1: switch N=4
#       L2: 2D mesh = (2x2)
#       L3: 2D torus = (4x4)
#     从外向内 layers = [L3, L2, L1]
#     """
#     # 每层的 hop_latency、link_bandwidth 与 switch-center 带宽均由调用者提供。
#     # 使用默认的"最内层连续"映射，不再提供自定义 mapper
#     return Hierarchy(layers=[L3, L2, L1], port_spread=PortSpread.EVEN, node_mapper=None)




# ======== 扩展矩阵与可视化（带宽/通信量/利用率） =============================

def _build_extended_indexer(h: Hierarchy):
    """
    在最内层引入一个额外的"锚点"坐标 (-1,-1)，形成扩展节点空间：
      - 若最内层为 SWITCH，则锚点等价于中心节点；
      - 若最内层不是 SWITCH，则锚点为虚拟节点，仅用于跨层连边的占位，不产生 L1 内联带宽或自环。
    返回：
      - ext_size: 扩展后的总节点数
      - coords_to_ext_id(coords)
      - ext_id_to_coords(eid)
      - anchor_token: (-1,-1)
    """
    layers = h.layers
    L = len(layers)
    assert L >= 1, "hierarchy 至少一层"

    # 每层的基数（flat size）
    base_sizes = []
    for i, topo in enumerate(reversed(layers)):
        M, N = topo.shape
        base_sizes.append(M * N)
    base_sizes = list(reversed(base_sizes))

    # 无论最内层类型，均在最内层追加 1 个锚点
    l1_extra = 1
    # 扩展后的每层 size（仅最后一层 +1）
    ext_layer_sizes = base_sizes[:-1] + [base_sizes[-1] + l1_extra]

    # 计算 strides（内到外）
    strides = [1] * L
    for i in range(L - 2, -1, -1):
        strides[i] = strides[i + 1] * ext_layer_sizes[i + 1]

    ext_size = 1
    for s in ext_layer_sizes:
        ext_size *= s

    def coords_to_ext_id(coords: list[tuple[int, int]]) -> int:
        assert len(coords) == L
        eid = 0
        for li in range(L):
            r, c = coords[li]
            M, N = layers[li].shape
            if li == L - 1 and (r, c) == (-1, -1):
                k = base_sizes[-1]  # 追加的锚点索引
            else:
                k = r * N + c
            eid += k * strides[li]
        return eid

    def ext_id_to_coords(eid: int) -> list[tuple[int, int]]:
        rem = int(eid)
        coords_rev: list[tuple[int, int]] = []
        # 内->外分解
        for li in range(L - 1, -1, -1):
            topo = layers[li]
            M, N = topo.shape
            size_li = base_sizes[li] + (1 if (li == L - 1) else 0)
            k = rem % size_li
            rem //= size_li
            if li == L - 1 and k == base_sizes[li]:
                coords_rev.append((-1, -1))
            else:
                coords_rev.append((k // N, k % N))
        return list(reversed(coords_rev))

    anchor_token = (-1, -1)
    return ext_size, coords_to_ext_id, ext_id_to_coords, anchor_token


def _neighbors_2d(kind: TopoKind, M: int, N: int):
    """生成二维网格的有向相邻边列表 (u_idx, v_idx, direction)。
    支持 MESH2D/TORUS2D/RING/CHAIN/ALL2ALL 的相邻定义。"""
    edges: list[tuple[int, int, str]] = []
    def id_of(r: int, c: int) -> int:
        return r * N + c
    if kind == TopoKind.ALL2ALL:
        for u in range(M * N):
            for v in range(M * N):
                if u != v:
                    edges.append((u, v, "direct"))
        return edges
    if kind == TopoKind.MESH2D:
        for r in range(M):
            for c in range(N):
                if r + 1 < M:
                    edges.append((id_of(r, c), id_of(r + 1, c), "down"))
                    edges.append((id_of(r + 1, c), id_of(r, c), "up"))
                if c + 1 < N:
                    edges.append((id_of(r, c), id_of(r, c + 1), "right"))
                    edges.append((id_of(r, c + 1), id_of(r, c), "left"))
        return edges
    if kind == TopoKind.TORUS2D:
        for r in range(M):
            for c in range(N):
                edges.append((id_of(r, c), id_of((r + 1) % M, c), "down"))
                edges.append((id_of(r, c), id_of((r - 1) % M, c), "up"))
                edges.append((id_of(r, c), id_of(r, (c + 1) % N), "right"))
                edges.append((id_of(r, c), id_of(r, (c - 1) % N), "left"))
        return edges
    if kind == TopoKind.MCHAIN_NRING:
        for r in range(M):
            for c in range(N):
                # 行方向 chain（无环绕）
                if r + 1 < M:
                    edges.append((id_of(r, c), id_of(r + 1, c), "down"))
                if r - 1 >= 0:
                    edges.append((id_of(r, c), id_of(r - 1, c), "up"))
                # 列方向 ring（有环绕）
                edges.append((id_of(r, c), id_of(r, (c + 1) % N), "right"))
                edges.append((id_of(r, c), id_of(r, (c - 1) % N), "left"))
        return edges
    if kind == TopoKind.MRING_NCHAIN:
        for r in range(M):
            for c in range(N):
                # 行方向 ring（有环绕）
                edges.append((id_of(r, c), id_of((r + 1) % M, c), "down"))
                edges.append((id_of(r, c), id_of((r - 1) % M, c), "up"))
                # 列方向 chain（无环绕）
                if c + 1 < N:
                    edges.append((id_of(r, c), id_of(r, c + 1), "right"))
                if c - 1 >= 0:
                    edges.append((id_of(r, c), id_of(r, c - 1), "left"))
        return edges
    if kind == TopoKind.RING:
        assert M == 1
        for c in range(N):
            edges.append((id_of(0, c), id_of(0, (c + 1) % N), "right"))
            edges.append((id_of(0, c), id_of(0, (c - 1) % N), "left"))
        return edges
    if kind == TopoKind.CHAIN:
        assert M == 1
        for c in range(N - 1):
            edges.append((id_of(0, c), id_of(0, c + 1), "right"))
            edges.append((id_of(0, c + 1), id_of(0, c), "left"))
        return edges
    return edges


def _side_map(direction: str) -> tuple[str, str]:
    if direction == "down":
        return "down_out", "up_in"
    if direction == "up":
        return "up_out", "down_in"
    if direction == "right":
        return "right_out", "left_in"
    if direction == "left":
        return "left_out", "right_in"
    if direction == "direct":
        return "right_out", "left_in"
    raise ValueError(f"invalid direction {direction}")


def build_extended_bandwidth_matrix(h: Hierarchy) -> np.ndarray:
    """
    生成扩展后的带宽矩阵（320x320 在示例里）。
    规则：
      - L1（SWITCH）层：center <-> ports 带宽 = L1.link_bandwidth；中心自环 = center_in_bw+center_out_bw。
      - L2 层：在同一 L3 tile 内，按拓扑相邻的 L2 节点间建立带宽 = L2.link_bandwidth，L1 坐标固定为中心 (-1,-1)。
      - L3 层：跨 tile 相邻，按方向枚举 L2 的 out/in 端口对，建立 [u, out_k, center] -> [v, in_k, center]，带宽 = L3.link_bandwidth。
    """
    L = len(h.layers)
    assert L >= 1
    ext_size, coords2eid, _, anchor = _build_extended_indexer(h)
    bw = np.zeros((ext_size, ext_size), dtype=np.float64)

    # 方便访问
    def with_coords(li: int, k: int) -> tuple[int, int]:
        topo = h.layers[li]
        M, N = topo.shape
        return (k // N, k % N)

    # 遍历所有外层坐标笛卡尔积
    def iter_outer_coords():
        if L == 1:
            yield []
            return
        # L>=2
        # 从外到内（不含 L1）
        ranges = []
        for topo in h.layers[:-1]:
            Mx, Nx = topo.shape
            ranges.append([(r, c) for r in range(Mx) for c in range(Nx)])
        # 笛卡尔积
        def rec(idx, acc):
            if idx == len(ranges):
                yield list(acc)
            else:
                for v in ranges[idx]:
                    acc.append(v)
                    yield from rec(idx + 1, acc)
                    acc.pop()
        yield from rec(0, [])
    # L1: switch
    if h.layers[-1].kind == TopoKind.SWITCH:
        L1 = h.layers[-1]
        M1, N1 = L1.shape
        # 对每个 (外层 L3/L2) 组合，建立 center <-> port
        outer_sizes = []
        for topo in h.layers[:-1]:
            Mx, Nx = topo.shape
            outer_sizes.append(Mx * Nx)

        center_bw = float(
            (L1.switch_center_in_bw if L1.switch_center_in_bw is not None else 0.0)
            + (L1.switch_center_out_bw if L1.switch_center_out_bw is not None else 0.0)
        )
        
        for outer in iter_outer_coords():
            # 每个 L1 端口
            for p in range(N1):  # M1==1
                src_coords = list(outer) + [(0, p)]
                cen_coords = list(outer) + [anchor]
                a = coords2eid(src_coords)
                b = coords2eid(cen_coords)
                bw[a, b] = max(bw[a, b], float(L1.link_bandwidth))
                bw[b, a] = max(bw[b, a], float(L1.link_bandwidth))
            # 中心自环
            if center_bw > 0.0:
                c = coords2eid(list(outer) + [anchor])
                bw[c, c] = max(bw[c, c], center_bw)
    else:
        assert h.layers[-1].kind in {TopoKind.MESH2D, TopoKind.TORUS2D, TopoKind.RING, TopoKind.CHAIN, TopoKind.MCHAIN_NRING, TopoKind.MRING_NCHAIN, TopoKind.ALL2ALL}
        L1 = h.layers[-1]
        M1, N1 = L1.shape
        edges1 = _neighbors_2d(L1.kind, M1, N1)
        for outer in iter_outer_coords():
            for (u, v, _dir) in edges1:
                ur, uc = with_coords(-1, u)
                vr, vc = with_coords(-1, v)
                coords_a = list(outer) + [(ur, uc)]
                coords_b = list(outer) + [(vr, vc)]
                a = coords2eid(coords_a)
                b = coords2eid(coords_b)
                bw[a, b] = max(bw[a, b], float(L1.link_bandwidth))
                
    # L2: 层内相邻
    if L >= 2:
        L2 = h.layers[-2]
        if L2.kind in {TopoKind.MESH2D, TopoKind.TORUS2D, TopoKind.RING, TopoKind.CHAIN, TopoKind.MCHAIN_NRING, TopoKind.MRING_NCHAIN}:
            M2, N2 = L2.shape
            edges2 = _neighbors_2d(L2.kind, M2, N2)
            # 遍历每个外层 L3 坐标
            if L == 2:
                outer3 = [()]
            else:
                L3 = h.layers[-3]
                M3, N3 = L3.shape
                outer3 = [(r3, c3) for r3 in range(M3) for c3 in range(N3)]
            for oc in outer3:
                for (u, v, _dir) in edges2:
                    ur, uc = with_coords(-2, u)
                    vr, vc = with_coords(-2, v)
                    if h.layers[-1].kind == TopoKind.SWITCH:
                        if L == 2:
                            coords_a = [(ur, uc), anchor]
                            coords_b = [(vr, vc), anchor]
                        else:  # L == 3
                            coords_a = [oc, (ur, uc), anchor]
                            coords_b = [oc, (vr, vc), anchor]
                        a = coords2eid(coords_a)
                        b = coords2eid(coords_b)
                        bw[a, b] = max(bw[a, b], float(L2.link_bandwidth))
                    else:
                        out_side, in_side = _side_map(_dir)
                        out_ports = L1.ports(out_side)
                        in_ports = L1.ports(in_side)
                        K = min(len(out_ports), len(in_ports))
                        for k in range(K):
                            or1, oc1 = L1.to_rc(out_ports[k]) if out_ports[k] != -1 else (-1, -1)
                            ir1, ic1 = L1.to_rc(in_ports[k]) if in_ports[k] != -1 else (-1, -1)
                            if L == 2:
                                coords_a = [(ur, uc), (or1, oc1)]
                                coords_b = [(vr, vc), (ir1, ic1)]
                            else:  # L == 3
                                coords_a = [oc, (ur, uc), (or1, oc1)]
                                coords_b = [oc, (vr, vc), (ir1, ic1)]
                            a = coords2eid(coords_a)
                            b = coords2eid(coords_b)
                            bw[a, b] = max(bw[a, b], float(L2.link_bandwidth))

    # L3: 层内相邻（通过 L2 端口对接）
    if L >= 3:
        L3 = h.layers[-3]
        L2 = h.layers[-2]
        M3, N3 = L3.shape
        edges3 = _neighbors_2d(L3.kind, M3, N3)
        for (u, v, direction) in edges3:
            ur3, uc3 = with_coords(-3, u)
            vr3, vc3 = with_coords(-3, v)
            out_side, in_side = _side_map(direction)
            out_ports = L2.ports(out_side)
            in_ports = L2.ports(in_side)
            K = min(len(out_ports), len(in_ports))
            for k in range(K):
                or2, oc2 = L2.to_rc(out_ports[k]) if out_ports[k] != -1 else (-1, -1)
                ir2, ic2 = L2.to_rc(in_ports[k]) if in_ports[k] != -1 else (-1, -1)
                if h.layers[-1].kind == TopoKind.SWITCH:
                    a = coords2eid([(ur3, uc3), (or2, oc2), anchor])
                    b = coords2eid([(vr3, vc3), (ir2, ic2), anchor])
                    bw[a, b] = max(bw[a, b], float(L3.link_bandwidth))
                else:
                    L1 = h.layers[-1]
                    out_side1, in_side1 = _side_map(direction)
                    out_ports1 = L1.ports(out_side1)
                    in_ports1 = L1.ports(in_side1)
                    K1 = min(len(out_ports1), len(in_ports1))
                    for k1 in range(K1):
                        or1, oc1 = L1.to_rc(out_ports1[k1]) if out_ports1[k1] != -1 else (-1, -1)
                        ir1, ic1 = L1.to_rc(in_ports1[k1]) if in_ports1[k1] != -1 else (-1, -1)
                        a = coords2eid([(ur3, uc3), (or2, oc2), (or1, oc1)])
                        b = coords2eid([(vr3, vc3), (ir2, ic2), (ir1, ic1)])
                        bw[a, b] = max(bw[a, b], float(L3.link_bandwidth))

    return bw


def build_extended_traffic_matrix(tm: "TrafficMatrix", h: Hierarchy) -> tuple[np.ndarray, float]:
    """
    使用 states_coords 将所有链条的通信量叠加到扩展矩阵上。
    返回 (traffic_matrix, max_hop_latency_states)。
    若最内层为 SWITCH，则对每个出现的"进/出中心" hop，额外在中心自环上加同样的 bytes（统计中心自吞吐）。
    """
    ext_size, coords2eid, _, anchor = _build_extended_indexer(h)
    traffic = np.zeros((ext_size, ext_size), dtype=np.float64)

    # 生成所有链条（与 compute_stage_latency 相同入口）
    flat = tm.counts
    nz = np.nonzero(flat)
    max_hop_latency = 0.0
    for s, d in zip(nz[0], nz[1]):
        val = int(flat[s, d])
        if val <= 0 or s == d:
            continue
        chains = h.route(s, d, val)
        for ch in chains:
            # 遍历相邻快照
            hops_lat = 0.0
            for i in range(1, len(ch.states_coords)):
                prev = ch.states_coords[i - 1]
                cur = ch.states_coords[i]
                # 找到最外层发生变化的层
                layer_idx = None
                for li, (a, b) in enumerate(zip(prev, cur)):
                    if a != b:
                        layer_idx = li
                        break
                if layer_idx is None:
                    continue
                # 端点映射
                a_eid = coords2eid(prev)
                b_eid = coords2eid(cur)
                traffic[a_eid, b_eid] += float(ch.bytes)
                # 统计 hop latency
                hops_lat += float(h.layers[layer_idx].hop_latency)
                # 若最内层为 SWITCH 且涉及中心锚点，增加中心自环流量
                if layer_idx == len(h.layers) - 1 and h.layers[-1].kind == TopoKind.SWITCH:
                    if prev[-1] == anchor or cur[-1] == anchor:
                        c_eid = coords2eid(list(cur[:-1]) + [anchor])
                        traffic[c_eid, c_eid] += float(ch.bytes)
            if hops_lat > max_hop_latency:
                max_hop_latency = hops_lat

    return traffic, max_hop_latency


def compute_utilization(traffic: np.ndarray, bandwidth: np.ndarray, base_time: float) -> tuple[np.ndarray, float]:
    """
    计算：
      - max_link_time = max(traffic_ij / bw_ij)
      - total_time = base_time + max_link_time
      - util_ij = traffic_ij / (total_time * bw_ij)
    对 bw==0 的单元，util 记为 0（且不会参与 max_link_time）。
    """
    eps = 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(bandwidth > 0.0, traffic / bandwidth, 0.0)
    max_link_time = float(np.max(ratio)) if ratio.size > 0 else 0.0
    total_time = float(base_time) + max_link_time
    with np.errstate(divide='ignore', invalid='ignore'):
        util = np.where(bandwidth > 0.0, traffic / (total_time * bandwidth), 0.0)
    util = np.clip(util, 0.0, 1.0)
    return util, total_time


def compute_energy(traffic: np.ndarray, energy_per_bit: np.ndarray) -> tuple[np.ndarray, float]:
    """
    计算链路能耗：
      - noc_energy_pj[i,j] = traffic[i,j] (bytes) * 8 * energy_per_bit[i,j] (pJ/bit)
      - total_noc_energy_pj = sum(noc_energy_pj)
    对 energy_per_bit==0 的单元，energy 记为 0。
    """
    noc_energy_pj = np.where(energy_per_bit > 0.0, traffic * 8.0 * energy_per_bit, 0.0)
    total_noc_energy_pj = float(np.sum(noc_energy_pj))
    return noc_energy_pj, total_noc_energy_pj


def get_total_traffic_bytes(traffic_mats: list) -> float:
    """
    将多个已展开的 extended traffic 矩阵累加，返回所有物理链路的总字节数之和。

    适用于 OpPerfStats._traffic_mats（append_traffic 时收集的各 collective 矩阵）。
    若列表为空，返回 0.0。
    """
    if not traffic_mats:
        return 0.0
    total = traffic_mats[0].copy()
    for mat in traffic_mats[1:]:
        total += mat
    return float(np.sum(total))


def save_matrix_heatmap(matrix: np.ndarray, path: str, title: str, bar_label: str = "Value") -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise ImportError("需要安装 matplotlib：pip install matplotlib") from e
    # 透明-绿-黄-红 实现（保留为注释）：
    # from matplotlib.colors import LinearSegmentedColormap
    # data = matrix.astype(float, copy=True)
    # vmax = float(np.nanmax(data)) if data.size > 0 else 1.0
    # if not np.isfinite(vmax) or vmax <= 0.0:
    #     vmax = 1.0
    # gyr_alpha = LinearSegmentedColormap.from_list(
    #     "gyr_alpha",
    #     [
    #         (0.0, (0.0, 1.0, 0.0, 0.0)),
    #         (1.0/10.0, (0.0, 1.0, 0.0, 1.0)),
    #         (2.0/3.0, (1.0, 1.0, 0.0, 1.0)),
    #         (1.0, (1.0, 0.0, 0.0, 1.0)),
    #     ], N=256,
    # )
    # fig = plt.figure(figsize=(8, 7), dpi=500)
    # ax = fig.add_subplot(111)
    # im = ax.imshow(data, cmap=gyr_alpha, vmin=0.0, vmax=vmax, origin="lower", interpolation="nearest")
    # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # ax.set_title(title)
    # ax.set_xlabel("dst (extended id)")
    # ax.set_ylabel("src (extended id)")
    # fig.tight_layout()
    # fig.savefig(path)
    # plt.close(fig)

    # 经典热力图：与老的 save_heatmap 风格类似（viridis，无透明）
    data = matrix.astype(float, copy=True)
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap="viridis", origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ==== 字体增强 BEGIN ====
    cbar.set_label(bar_label, fontsize=16)           # 色条标题
    cbar.ax.tick_params(labelsize=16)              # 色条刻度
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)  # colorbar 顶部科学计数法字体

    ax.set_title(title, fontsize=16)
    # ax.set_xlabel("dst (extended id)", fontsize=16)
    # ax.set_ylabel("src (extended id)", fontsize=16)
    ax.set_xlabel("Destination", fontsize=16)
    ax.set_ylabel("Source", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    # ==== 字体增强 END ====

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_extended_matrices_and_plots(
    tm: "TrafficMatrix",
    h: Hierarchy,
    out_prefix: str = "extended"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    生成并保存：
      - 带宽矩阵 PDF
      - 通信量矩阵 PDF（基于 states_coords）
      - 利用率矩阵 PDF（总时间 = states hop latency + max(traffic/bw)）
      - 能耗矩阵 PDF（pJ/link = traffic * 8 * energy_per_bit）
    返回 (bw, traffic, util, total_time, noc_energy_pj, total_noc_energy_pj)。
    """
    from .noc_energy import build_extended_energy_matrix  # 延迟导入，避免循环依赖

    bw = build_extended_bandwidth_matrix(h)
    traffic, hop_time = build_extended_traffic_matrix(tm, h)
    util, total_time = compute_utilization(traffic, bw, hop_time)
    energy_per_bit = build_extended_energy_matrix(h)
    noc_energy_pj, total_noc_energy_pj = compute_energy(traffic, energy_per_bit)

    save_matrix_heatmap(bw, f"{out_prefix}_bandwidth.pdf", "Link Bandwidth", "Bandwidth (bytes/s)")
    save_matrix_heatmap(traffic, f"{out_prefix}_traffic.pdf", "Link Traffic (bytes)", "Traffic (bytes)")
    save_matrix_heatmap(util, f"{out_prefix}_utilization.pdf", "Link Utilization", "Utilization")
    save_matrix_heatmap(noc_energy_pj, f"{out_prefix}_energy.pdf", "Link Energy (pJ)", "Energy (pJ)")

    return bw, traffic, util, total_time, noc_energy_pj, total_noc_energy_pj

def build_extended_traffic_matrix_switch_only(tm: "TrafficMatrix", h: Hierarchy) -> tuple[np.ndarray, float]:
    """
    基于 states_coords 叠加流量到"多层均为 SWITCH"的扩展坐标系。
    与原版本不同点：
      - 扩展索引器对每一层都有锚点；当某一步跨过某层的中心（prev[li] 或 cur[li] 为 (-1,-1)）时，
        在该层中心自环上也累加同样的 bytes（统计中心吞吐）。
    """
    for topo in h.layers:
        if topo.kind != TopoKind.SWITCH:
            raise ValueError("build_extended_traffic_matrix_switch_only 仅支持 SWITCH 层")

    ext_size, coords2eid, _, anchors = _build_extended_indexer_switch_only(h)
    traffic = np.zeros((ext_size, ext_size), dtype=np.float64)

    flat = tm.counts
    nz = np.nonzero(flat)
    max_hop_latency = 0.0

    for s, d in zip(nz[0], nz[1]):
        val = int(flat[s, d])
        if val <= 0 or s == d:
            continue
        chains = h.route(int(s), int(d), int(val))
        for ch in chains:
            hops_lat = 0.0
            for i in range(1, len(ch.states_coords)):
                prev = ch.states_coords[i - 1]
                cur  = ch.states_coords[i]
                # 找到最外层发生变化的层
                layer_idx = None
                for li, (a, b) in enumerate(zip(prev, cur)):
                    if a != b:
                        layer_idx = li; break
                if layer_idx is None:
                    continue
                # 规范化坐标：仅保留 layer_idx 层的实际坐标，其余层一律置为锚点。
                # 这与 build_extended_bandwidth_matrix_switch_only 的坐标约定一致，
                # 确保 traffic[a,b] 和 bw[a,b] 对齐同一物理链路条目。
                L = len(h.layers)
                prev_norm = [anchors[li] if li != layer_idx else prev[li] for li in range(L)]
                cur_norm  = [anchors[li] if li != layer_idx else cur[li]  for li in range(L)]
                a_eid = coords2eid(prev_norm)
                b_eid = coords2eid(cur_norm)
                traffic[a_eid, b_eid] += float(ch.bytes)
                # hop latency 叠加
                hops_lat += float(h.layers[layer_idx].hop_latency)

            if hops_lat > max_hop_latency:
                max_hop_latency = hops_lat

    return traffic, max_hop_latency


def _build_extended_indexer_switch_only(h: Hierarchy):
    """
    为每一层（外->内）都是 SWITCH 的层级构建扩展索引器：
      - 每一层都追加一个"中心锚点"坐标 (-1,-1)
      - 返回：
          ext_size
          coords_to_ext_id(coords)
          ext_id_to_coords(eid)
          anchors: List[Tuple[int,int]]  # 每层的锚点，均为 (-1,-1)
    均假定所有层都是 TopoKind.SWITCH。
    """
    assert len(h.layers) >= 1, "hierarchy 至少一层"
    for topo in h.layers:
        if topo.kind != TopoKind.SWITCH:
            raise ValueError("本索引器仅支持所有层均为 SWITCH 的情形")

    L = len(h.layers)
    base_sizes = [(topo.shape[0] * topo.shape[1]) for topo in h.layers]  # 对 switch 即为端口数 N
    # 每层都 +1 作为中心锚点（-1,-1）
    ext_layer_sizes = [s + 1 for s in base_sizes]

    # 计算 stride（内->外）
    strides = [1] * L
    for i in range(L - 2, -1, -1):
        strides[i] = strides[i + 1] * ext_layer_sizes[i + 1]

    ext_size = 1
    for s in ext_layer_sizes:
        ext_size *= s

    def _coords_to_k(li: int, rc: tuple[int, int]) -> int:
        """把 (r,c) 映射为该层的线性索引；(-1,-1) 是追加的中心，索引==base_size。"""
        topo = h.layers[li]
        M, N = topo.shape
        if rc == (-1, -1):
            return base_sizes[li]
        (r, c) = rc
        return r * N + c

    def coords_to_ext_id(coords: list[tuple[int, int]]) -> int:
        assert len(coords) == L
        eid = 0
        for li in range(L):
            k = _coords_to_k(li, coords[li])
            eid += k * strides[li]
        return eid

    def ext_id_to_coords(eid: int) -> list[tuple[int, int]]:
        rem = int(eid)
        coords_rev: list[tuple[int, int]] = []
        for li in range(L - 1, -1, -1):
            size_li = base_sizes[li] + 1  # +1 for anchor
            k = rem % size_li
            rem //= size_li
            if k == base_sizes[li]:
                coords_rev.append((-1, -1))
            else:
                N = h.layers[li].shape[1]
                coords_rev.append((k // N, k % N))
        return list(reversed(coords_rev))

    anchors = [(-1, -1) for _ in range(L)]
    return ext_size, coords_to_ext_id, ext_id_to_coords, anchors


def build_extended_bandwidth_matrix_switch_only(h: Hierarchy) -> np.ndarray:
    """
    构建"只支持三层均为 SWITCH"的扩展带宽矩阵。
    建模方式（与 L1 的旧逻辑一致，推广到任意层）：
      - 对每一层 li：
          * 给该层的"中心锚点 (-1,-1)" 与 "每个物理端口 (0,p)"之间建立双向边，
            边带宽 = layers[li].link_bandwidth（取 max 累积，不求和）
          * 如果设置了 center_in/out_bw，则在该层中心自环上写入
            bw = center_in_bw + center_out_bw（同样取 max）
      - 更内层/更外层坐标用"锚点 (-1,-1)"占位，使得边位于统一的扩展坐标系。
    """
    # 校验
    for topo in h.layers:
        if topo.kind != TopoKind.SWITCH:
            raise ValueError("build_extended_bandwidth_matrix_switch_only 仅支持 SWITCH 层")

    L = len(h.layers)
    ext_size, coords2eid, _, anchors = _build_extended_indexer_switch_only(h)
    bw = np.zeros((ext_size, ext_size), dtype=np.float64)

    # 枚举"外层坐标"的笛卡尔积（到 li-1），li 及其内层均使用锚点占位（或在 li 层枚举端口）
    def iter_prefix_coords(upto_exclusive: int):
        """返回所有 0..upto_exclusive-1 层的坐标组合（每层只会是 (-1,-1) 或端口坐标？对 SWITCH 用端口坐标只在当前层）"""
        if upto_exclusive <= 0:
            yield []
            return
        ranges = []
        for li in range(upto_exclusive):
            topo = h.layers[li]
            # 在"外层前缀"我们只放置锚点，避免指数爆炸；路由时会具体化
            ranges.append([anchors[li]])
        # 笛卡尔积
        def rec(idx, acc):
            if idx == len(ranges):
                yield list(acc); return
            for v in ranges[idx]:
                acc.append(v)
                yield from rec(idx + 1, acc)
                acc.pop()
        yield from rec(0, [])

    for li in range(L):
        topo = h.layers[li]
        M, N = topo.shape  # 对 switch，M=1, N=端口数
        center_loop_bw = float(
            (topo.switch_center_in_bw or 0.0) + (topo.switch_center_out_bw or 0.0)
        )
        # 外层前缀坐标（0..li-1），都固定用锚点；内层后缀（li+1..）也固定锚点。
        for prefix in iter_prefix_coords(li):
            # 组装 "该层中心节点的完整扩展坐标"
            center_coords = list(prefix) + [anchors[li]] + anchors[li+1:]
            center_eid = coords2eid(center_coords)
            # 该层每个端口与中心的双向带宽边
            for p in range(N):  # M==1
                port_coords = list(prefix) + [(0, p)] + anchors[li+1:]
                port_eid = coords2eid(port_coords)
                # 双向
                bw[port_eid, center_eid] = max(bw[port_eid, center_eid], float(topo.link_bandwidth))
                bw[center_eid, port_eid] = max(bw[center_eid, port_eid], float(topo.link_bandwidth))
            # 中心自环
            if center_loop_bw > 0.0:
                bw[center_eid, center_eid] = max(bw[center_eid, center_eid], center_loop_bw)

    return bw

def export_extended_matrices_and_plots_switch_only(
    tm: "TrafficMatrix",
    h: Hierarchy,
    out_prefix: str = "extended_switch_only"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    只支持所有层均为 SWITCH 的导出版本。
    生成并保存：
      - 带宽矩阵 PNG
      - 通信量矩阵 PNG
      - 利用率矩阵 PNG
      - 能耗矩阵 PNG（pJ/link = traffic * 8 * energy_per_bit）
    返回 (bw, traffic, util, total_time, noc_energy_pj, total_noc_energy_pj)。
    """
    from .noc_energy import build_extended_energy_matrix_switch_only  # 延迟导入，避免循环依赖

    bw = build_extended_bandwidth_matrix_switch_only(h)
    traffic, hop_time = build_extended_traffic_matrix_switch_only(tm, h)
    util, total_time = compute_utilization(traffic, bw, hop_time)
    energy_per_bit = build_extended_energy_matrix_switch_only(h)
    noc_energy_pj, total_noc_energy_pj = compute_energy(traffic, energy_per_bit)

    save_matrix_heatmap(bw,           f"{out_prefix}_bandwidth.png",   "Extended Link Bandwidth (Switch-only)")
    save_matrix_heatmap(traffic,      f"{out_prefix}_traffic.png",     "Extended Link Traffic (bytes)")
    save_matrix_heatmap(util,         f"{out_prefix}_utilization.png", "Extended Link Utilization")
    save_matrix_heatmap(noc_energy_pj, f"{out_prefix}_energy.png",    "Extended Link Energy (pJ)")

    return bw, traffic, util, total_time, noc_energy_pj, total_noc_energy_pj

def get_extend_max_routes_switch_only(
    tm: "TrafficMatrix",
    h: Hierarchy,
) -> tuple[float, float, float]:
    '''
    仅支持所有层均为 SWITCH：
      - 使用已构建的扩展 traffic（states_coords 叠加）作为真值
      - 按需在 traffic>0 的位置补带宽，带宽所属层 = "最外层发生变化的层"
        * i==j（自环）：该层 center 自环带宽 = center_in_bw + center_out_bw
        * i!=j：带宽 = h.layers[layer_idx].link_bandwidth
      - 返回 (hop_time, ext_max, overall_time)
    '''
    # 1) 检查 & traffic / hop_time
    if not all(t.kind == TopoKind.SWITCH for t in h.layers):
        raise ValueError("get_extend_max_routes_switch_only 仅支持所有层均为 SWITCH")

    traffic, hop_time = build_extended_traffic_matrix_switch_only(tm, h)

    # 2) 按需补带宽：只给 traffic 非零的位置赋带宽，且带宽归属按"最外层发生变化的层"
    _, _, extid2coords, _anchors = _build_extended_indexer_switch_only(h)
    bw = np.zeros_like(traffic, dtype=np.float64)

    nz_src, nz_dst = np.nonzero(traffic)
    for i, j in zip(nz_src.tolist(), nz_dst.tolist()):
        tij = float(traffic[i, j])
        if tij <= 0.0:
            continue

        ci = extid2coords(int(i))
        cj = extid2coords(int(j))

        # 找"最外层发生变化的层"layer_idx
        layer_idx = None
        for li, (a, b) in enumerate(zip(ci, cj)):
            if a != b:
                layer_idx = li
                break

        if layer_idx is None:
            # i==j: 自环。这个自环必然是某一层的 center 自环，
            # 我们根据坐标里为 (-1,-1) 的层给出该层的 center_in+center_out。
            loop_bw = 0.0
            for li, rc in enumerate(ci):
                if rc == (-1, -1):
                    topo = h.layers[li]
                    loop_bw = max(loop_bw, float((topo.switch_center_in_bw or 0.0) + (topo.switch_center_out_bw or 0.0)))
            # 防守：如果没有任何层是 (-1,-1)，则不写入（或写 0）
            if loop_bw > 0.0:
                bw[i, j] = max(bw[i, j], loop_bw)
            continue

        # 非自环：这一跳就是 layer_idx 层的 hop，带宽取该层的 link_bandwidth
        topo = h.layers[layer_idx]
        hop_bw = float(topo.link_bandwidth)
        if hop_bw > 0.0:
            bw[i, j] = max(bw[i, j], hop_bw)

    # 3) 计算瓶颈时间和总时间
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(bw > 0.0, traffic / bw, -np.inf)

    if ratio.size == 0 or not np.isfinite(np.max(ratio)):
        # 流量矩阵为空（ep=1 等无跨节点通信的合法情况），直接返回 0
        log.debug("[extended/switch-only] traffic matrix is empty (no inter-node comm); returning (0,0,0)")
        return 0.0, 0.0, 0.0

    flat_idx = int(np.nanargmax(ratio))
    i_star, j_star = divmod(flat_idx, ratio.shape[1])
    ext_max = float(ratio[i_star, j_star]) if np.isfinite(ratio[i_star, j_star]) else 0.0

    # 诊断打印
    src_coords = extid2coords(int(i_star))
    dst_coords = extid2coords(int(j_star))
    log.info("[extended/switch-only] max_link_time:")
    log.info("  i->j: %s -> %s time_s: %.6g", i_star, j_star, ext_max)
    log.info("  src_coords: %s", src_coords)
    log.info("  dst_coords: %s", dst_coords)
    log.info("  bytes: %.0f bw: %.0f", float(traffic[i_star, j_star]), float(bw[i_star, j_star]))
    log.info("  hop_time_s: %.6g total_s: %.6g", float(hop_time), float(hop_time) + float(ext_max))

    overall_time = float(hop_time) + float(ext_max)
    return float(hop_time), float(ext_max), float(overall_time)

def get_extend_max_routes(
    tm: "TrafficMatrix",
    h: Hierarchy,
) -> tuple[float, float, float]:
    """
    打印流水线中 max link time 的分解：
      1) 扩展路径（export_extended_matrices_and_plots）：max(traffic/bw)
    """
    log.info("Diagnosing max link time")

    if all(t.kind == TopoKind.SWITCH for t in h.layers):
        hop_time, ext_max, overall_time = get_extend_max_routes_switch_only(tm, h)
        return hop_time, ext_max, overall_time
    else:
        # 扩展矩阵
        bw = build_extended_bandwidth_matrix(h)
        traffic, hop_time = build_extended_traffic_matrix(tm, h)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(bw > 0.0, traffic / bw, -np.inf)
        if ratio.size > 0:
            flat_idx = int(np.nanargmax(ratio))
            i, j = divmod(flat_idx, ratio.shape[1])
            ext_max = float(ratio[i, j]) if np.isfinite(ratio[i, j]) else 0.0
            # 还原扩展坐标
            ext_size, _, extid2coords, anchor = _build_extended_indexer(h)
            src_coords = extid2coords(i)
            dst_coords = extid2coords(j)
            log.info("[extended] max_link_time:")
            log.info("  i->j: %s -> %s time_s: %.6g", i, j, float(ext_max))
            log.info("  src_coords: %s", src_coords)
            log.info("  dst_coords: %s", dst_coords)
            log.info("  bytes: %.0f bw: %.0f", float(traffic[i, j]), float(bw[i, j]))
            log.info("  hop_time_s: %.6g total_s: %.6g", float(hop_time), float(hop_time) + float(ext_max))

        else:
            log.debug("[extended] traffic matrix is empty (no inter-node comm); returning (0,0,0)")

        overall_time=hop_time + ext_max

        return hop_time, ext_max, overall_time

def get_extend_max_routes_with_traffic(
    tm: "TrafficMatrix",
    h: Hierarchy,
    dump_perf_log: bool = True,
) -> tuple[float, float, float, np.ndarray | None]:
    """
    返回 (hop_time, ext_max, overall_time, traffic)，可直接接入 OpPerfStats：
        hop_time, ext_max, overall_time, traffic = get_extend_max_routes_with_traffic(tm, h)
        stats.append_traffic(traffic, link_time_s=ext_max, hop_time_s=hop_time, comm_time_s=overall_time)

    其中：
        hop_time     → hop_time_s    （交换机 hop latency 之和）
        ext_max      → link_time_s   （瓶颈链路传输时间）
        overall_time → comm_time_s   （总通信时间 = hop + link）

    dump_perf_log=False（默认）：简单路径，直接组合 get_extend_max_routes + build traffic，
                                无额外诊断 log。
    dump_perf_log=True         ：完整内联路径，traffic 只构建一次，并打印详细诊断 log。
    """
    if not dump_perf_log:
        # ── 无 dump_perf_log ──────────────────────────────────────────────
        hop_time, ext_max, overall_time = get_extend_max_routes_with_traffic(tm, h)

        return float(hop_time), float(ext_max), float(overall_time), None

    # ── 完整路径：内联计算，traffic 只构建一次，打印详细诊断 log ──────────────
    if all(t.kind == TopoKind.SWITCH for t in h.layers):
        traffic, hop_time = build_extended_traffic_matrix_switch_only(tm, h)

        _, _, extid2coords, _anchors = _build_extended_indexer_switch_only(h)
        bw = np.zeros_like(traffic, dtype=np.float64)
        nz_src, nz_dst = np.nonzero(traffic)
        for i, j in zip(nz_src.tolist(), nz_dst.tolist()):
            if float(traffic[i, j]) <= 0.0:
                continue
            ci = extid2coords(int(i))
            cj = extid2coords(int(j))
            layer_idx = None
            for li, (a, b) in enumerate(zip(ci, cj)):
                if a != b:
                    layer_idx = li
                    break
            if layer_idx is None:
                loop_bw = 0.0
                for li, rc in enumerate(ci):
                    if rc == (-1, -1):
                        topo = h.layers[li]
                        loop_bw = max(loop_bw, float((topo.switch_center_in_bw or 0.0) + (topo.switch_center_out_bw or 0.0)))
                if loop_bw > 0.0:
                    bw[i, j] = max(bw[i, j], loop_bw)
                continue
            hop_bw = float(h.layers[layer_idx].link_bandwidth)
            if hop_bw > 0.0:
                bw[i, j] = max(bw[i, j], hop_bw)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(bw > 0.0, traffic / bw, -np.inf)

        if ratio.size == 0 or not np.isfinite(np.max(ratio)):
            log.debug("[extended/switch-only] traffic matrix is empty (no inter-node comm); returning (0,0,0)")
            return float(hop_time), 0.0, float(hop_time), traffic

        flat_idx = int(np.nanargmax(ratio))
        i_star, j_star = divmod(flat_idx, ratio.shape[1])
        ext_max = float(ratio[i_star, j_star]) if np.isfinite(ratio[i_star, j_star]) else 0.0
        src_coords = extid2coords(int(i_star))
        dst_coords = extid2coords(int(j_star))
        # log.info("[extended/switch-only] max_link_time:")
        # log.info("  i->j: %s -> %s time_s: %.6g", i_star, j_star, ext_max)
        # log.info("  src_coords: %s  dst_coords: %s", src_coords, dst_coords)
        # log.info("  bytes: %.0f bw: %.0f", float(traffic[i_star, j_star]), float(bw[i_star, j_star]))
        # log.info("  hop_time_s: %.6g total_s: %.6g", float(hop_time), float(hop_time) + ext_max)
        overall_time = float(hop_time) + ext_max

    else:
        bw = build_extended_bandwidth_matrix(h)
        traffic, hop_time = build_extended_traffic_matrix(tm, h)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(bw > 0.0, traffic / bw, -np.inf)
        ext_max = 0.0
        if ratio.size > 0:
            flat_idx = int(np.nanargmax(ratio))
            i, j = divmod(flat_idx, ratio.shape[1])
            ext_max = float(ratio[i, j]) if np.isfinite(ratio[i, j]) else 0.0
            ext_size, _, extid2coords, anchor = _build_extended_indexer(h)
            src_coords = extid2coords(i)
            dst_coords = extid2coords(j)
            log.info("[extended] max_link_time:")
            log.info("  i->j: %s -> %s time_s: %.6g", i, j, float(ext_max))
            log.info("  src_coords: %s  dst_coords: %s", src_coords, dst_coords)
            log.info("  bytes: %.0f bw: %.0f", float(traffic[i, j]), float(bw[i, j]))
            log.info("  hop_time_s: %.6g total_s: %.6g", float(hop_time), float(hop_time) + ext_max)
        else:
            log.error("[extended] empty matrices")
        overall_time = hop_time + ext_max

    return float(hop_time), float(ext_max), float(overall_time), traffic

def diagnose_max_link_time(
    tm: "TrafficMatrix",
    h: Hierarchy,
) -> None:
    """
    打印两条流水线中 max link time 的分解：
      1) 扩展路径（export_extended_matrices_and_plots）：max(traffic/bw)
    """
    # 扩展矩阵
    bw = build_extended_bandwidth_matrix(h)
    traffic, hop_time = build_extended_traffic_matrix(tm, h)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(bw > 0.0, traffic / bw, -np.inf)
    if ratio.size > 0:
        flat_idx = int(np.nanargmax(ratio))
        i, j = divmod(flat_idx, ratio.shape[1])
        ext_max = float(ratio[i, j]) if np.isfinite(ratio[i, j]) else 0.0
        # 还原扩展坐标
        ext_size, _, extid2coords, anchor = _build_extended_indexer(h)
        src_coords = extid2coords(i)
        dst_coords = extid2coords(j)
        log.info("[extended] max_link_time:")
        log.info("  i->j: %s -> %s time_s: %s", i, j, ext_max)
        log.info("  src_coords: %s", src_coords)
        log.info("  dst_coords: %s", dst_coords)
        log.info("  bytes: %s bw: %s", float(traffic[i, j]), float(bw[i, j]))
        log.info("  hop_time_s: %s total_s: %s", hop_time, hop_time + ext_max)
    else:
        log.debug("[extended] traffic matrix is empty (no inter-node comm)")


def get_longest_states_path(
    tm: "TrafficMatrix",
    h: Hierarchy,
) -> tuple[list[list[tuple[int, int]]], float, tuple[int, int]]:
    """
    基于 states_coords，找出 hop 延迟之和最大的那条链（最长路径，按层 hop_latency 求和）。
    返回 (best_path_states, best_hop_time, (src_idx, dst_idx))。
    """
    flat = tm.counts
    nz = np.nonzero(flat)
    best_hop_time: float = 0.0
    best_path: list[list[tuple[int, int]]] | None = None
    best_pair: tuple[int, int] | None = None

    for s, d in zip(nz[0], nz[1]):
        val = int(flat[s, d])
        if val <= 0 or s == d:
            continue
        chains = h.route(int(s), int(d), int(val))
        for ch in chains:
            hops_lat = 0.0
            for i in range(1, len(ch.states_coords)):
                prev = ch.states_coords[i - 1]
                cur = ch.states_coords[i]
                layer_idx = None
                for li, (a, b) in enumerate(zip(prev, cur)):
                    if a != b:
                        layer_idx = li
                        break
                if layer_idx is None:
                    continue
                hops_lat += float(h.layers[layer_idx].hop_latency)
            if hops_lat > best_hop_time:
                best_hop_time = hops_lat
                best_path = list(ch.states_coords)
                best_pair = (int(s), int(d))

    if best_path is None:
        return [], 0.0, (-1, -1)
    return best_path, float(best_hop_time), (best_pair[0], best_pair[1])


def print_longest_states_path(
    tm: "TrafficMatrix",
    h: Hierarchy,
) -> None:
    """打印按 hop 延迟之和最长的路径（基于 states_coords 的相邻快照）。"""
    path, hop_time, pair = get_longest_states_path(tm, h)
    if not path:
        log.info("[longest] no valid paths")
        return
    log.info("[longest] src=%s dst=%s hop_time_s=%.6g steps=%d", pair[0], pair[1], hop_time, max(0, len(path) - 1))
    # 整体起点/终点的 coords 视图
    log.info("  start_coords: %s", path[0])
    log.info("  dst_coords:   %s", path[-1])

    cum = 0.0
    for i in range(1, len(path)):
        prev = path[i - 1]
        cur = path[i]
        # 找到最外层发生变化的层
        layer_idx = None
        for li, (a, b) in enumerate(zip(prev, cur)):
            if a != b:
                layer_idx = li
                break
        step_lat = float(h.layers[layer_idx].hop_latency) if layer_idx is not None else 0.0
        cum += step_lat
        log.info("  step %d:", i - 1)
        log.info("    src_coords: %s", prev)
        log.info("    dst_coords: %s", cur)
        if layer_idx is not None:
            log.info("    layer=%d kind=%s hop_latency_s=%.6g cum_s=%.6g", layer_idx, h.layers[layer_idx].kind.value, step_lat, cum)
        else:
            log.info("    layer=NA hop_latency_s=0 cum_s=%.6g", cum)
