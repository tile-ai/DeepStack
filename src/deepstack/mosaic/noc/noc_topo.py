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
# ---------------- Basic types ----------------

# A "physical link" is uniquely identified by (layer_id, u, v) (directed)
# u, v are node IDs within this layer (0..M*N-1, or the special switch node -1).
Link = Tuple[int, int, int]  # (layer, u, v)

# A "chain" is a sequence of physical links (a hop list) with a weight (bytes carried by that chain)
@dataclass
class Chain:
    hops: List[Link]
    bytes: int
    # Hierarchical state trace: human-readable snapshots of all layer coordinates after each step (outer->inner).
    states_coords: List[List[Tuple[int, int]]] = field(default_factory=list)


class PortSpread(Enum):
    EVEN = "even"        # Even: divide (src,dst) traffic evenly into k shares across multiple ingress/egress ports.
    NEAREST = "nearest"  # Nearest: select the nearest ingress/egress port (requires topology support for bypass).


class TopoKind(Enum):
    SWITCH = "switch"        # [1 x N], all ports connect to the center (node -1).
    ALL2ALL = "all_to_all"   # [M x N], abstract cross-group communication as one hop (constant hop count).
    RING = "ring"            # [1 x N], N is even.
    CHAIN = "chain"          # [1 x N], N is even.
    MESH2D = "mesh2d"        # [M x N], M,N are even.
    TORUS2D = "torus2d"      # [M x N], M,N are even.
    MCHAIN_NRING = "mchain_nring"   # Rows form a chain (no wraparound); columns form a ring (wraparound).
    MRING_NCHAIN = "mring_nchain"   # Rows form a ring (wraparound); columns form a chain (no wraparound).


# ---------------- Topology abstraction ----------------

@dataclass
class Topology:
    kind: TopoKind
    shape: Tuple[int, int]           # (M, N); use (1, size) for ring/chain/switch.
    # Intrinsic per-hop latency and link bandwidth for each layer.
    hop_latency: float = 1.0         # Use consistent units (e.g., seconds/hop or cycles/hop).
    link_bandwidth: float = 1.0      # Use units consistent with bytes (e.g., bytes/second or bytes/cycle).
    # Switch only: aggregate center bandwidth (separate limits for total ingress and egress bandwidth).
    switch_center_in_bw: Optional[float] = None
    switch_center_out_bw: Optional[float] = None
    # Port partition rules (define in/out coordinate sets according to the supplied description).
    # These fields apply only to mesh/torus/ring/chain and partition in/out ports by "up/down/left/right"
    # Each field is a function returning a list of "port units", each corresponding to a physical node ID.
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

    # Map 2D coordinates to a linear ID.
    def to_id(self, r: int, c: int) -> int:
        M, N = self.shape
        return r * N + c

    # Map a linear ID back to (r, c).
    def to_rc(self, idx: int) -> Tuple[int, int]:
        M, N = self.shape
        return divmod(idx, N)

    # A "port list" for a direction (each port is a physical node ID)
    def ports(self, side: str) -> List[int]:
        # SWITCH: abstract as center port -1 (all directions go through the center).
        if self.kind == TopoKind.SWITCH:
            return [-1]
        # ALL2ALL: prefer the factory-provided port functions; otherwise distribute linear IDs evenly across four directions.
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

    # Compute the shortest within-layer "Manhattan distance" (with wraparound for torus) for NEAREST port selection
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
            # 1xN ring: consider only the column direction.
            delta = abs(bc - ac)
            return min(delta, N - delta)
        if self.kind == TopoKind.CHAIN:
            return abs(bc - ac)
        return 0

    # Intralayer XY routing: return a sequence of directed edges [(u->v), ...].
    def route_intra(self, s_id: int, d_id: int) -> List[Tuple[int, int]]:
        if s_id == d_id:
            return []
        if self.kind == TopoKind.SWITCH:
            # Switch-layer abstraction: node to center (-1), then out.
            # Handle the boundary case to avoid a (-1,-1) self-loop.
            if s_id == -1 and d_id == -1:
                return []
            if s_id == -1:
                return [(-1, d_id)]
            if d_id == -1:
                return [(s_id, -1)]
            return [(s_id, -1), (-1, d_id)]
        if self.kind == TopoKind.ALL2ALL:
            # Two possibilities:
            # 1) Ports in the same direction connect directly in one abstract hop: s->d.
            # 2) Alternatively, view this as passing through "the aggregation node for that direction" (still simplified to a direct link here to avoid artificially adding hops)
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
            # Rows first, then columns (X->Y).
            # Row direction: take a straight path for mesh, or the shorter direction for torus (including wraparound).
            r_path: List[int] = []
            if self.kind in {TopoKind.MESH2D, TopoKind.MCHAIN_NRING}:
                step = 1 if dr >= sr else -1
                for r in range(sr, dr, step):
                    r_path.append((r, sc))
                    hops.append((id_of(r, sc), id_of(r + step, sc)))
            else:  # row is ring (TORUS2D or MRING_NCHAIN)
                # Choose the direction with the smaller of |Δ| and M-|Δ|.
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

            # Column direction.
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
            # 1xN: take the shorter direction (left or right).
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


# ---------------- Port rules (using the supplied partition) ----------------

def _range2(a, b):
    # Convert the half-open interval [a:b] to a list.
    return list(range(a, b)) if a < b else []


def make_mesh_or_torus(M: int, N: int, kind: TopoKind, *, hop_latency: float = 1.0, link_bandwidth: float = 1.0) -> Topology:
    assert kind in (TopoKind.MESH2D, TopoKind.TORUS2D, TopoKind.MCHAIN_NRING, TopoKind.MRING_NCHAIN)
    assert M >= 1 and N >= 1, "M,N must be >= 1 for mesh/torus"
    # Handle M==1 or N==1: when a dimension is 1, in/out ports map to the same coordinate set (which must be nonempty)
    assert (M == 1 or M % 2 == 0) and (N == 1 or N % 2 == 0), "M,N must be even unless it's 1 for mesh/torus"

    def _split_in_out_indices(dim: int):
        """Split indices [0, dim-1] into input/output halves.
        For dim>1, inputs use the second half and outputs the first. For dim==1,
        both groups contain [0].
        """
        if dim == 1:
            return (0,), (0,)
        return tuple(range(dim // 2, dim)), tuple(range(0, dim // 2))

    rows_in, rows_out = _split_in_out_indices(M)
    cols_in, cols_out = _split_in_out_indices(N)

    # Precompute the 8 port coordinates for readability/review; Topology callbacks only return copies of constant lists
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
    # Split up/down ports in half using ring rules (for even splitting during multilayer aggregation/descent).
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
    # [1 x N], all ports connect to center -1 (Topology.ports represents the center port as [-1]).
    return Topology(
        kind=TopoKind.SWITCH,
        shape=(1, N),
        hop_latency=hop_latency,
        link_bandwidth=link_bandwidth,
        switch_center_in_bw=switch_center_in_bw,
        switch_center_out_bw=switch_center_out_bw,
    )


def make_all2all(M: int, N: int, *, hop_latency: float = 1.0, link_bandwidth: float = 1.0) -> Topology:
    # [M x N], provide port functions that distribute ports evenly across four directions (at least one ingress and one egress per node).
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


# ---------------- Hierarchical topology and routing ----------------


@dataclass
class Hierarchy:
    # Outer to inner: L3 (index 2), L2 (1), L1 (0); store as [L3, L2, L1] here for clarity.
    layers: List[Topology]          # Length <= 3.
    port_spread: PortSpread = PortSpread.EVEN  # Port selection policy (EVEN/NEAREST).
    name: str = "default_noc_hierarchy"
    # For EVENS: split each (src,dst) evenly into K parts at every "cross-layer" step (K = number of ports)
    # For NEAREST: select only the nearest port (requires topology support for bypass).

    # Map logical "node IDs (0..N-1)" to coordinates (r,c) in each layer.
    # The default mapping expands linear IDs into inner-layer (L1) coordinates (r,c) using layer shapes; outer layers (L2/L3) use integer blocking of (r,c), rounding up.
    # A custom function can specify the exact alignment described in the comments (e.g., [[0,0],[0,0],[0,2]]).
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
        """Return the total number of devices covered by this Hierarchy, equal to the product of all layer shapes."""
        n = 1
        for layer in self.layers:
            n *= layer.shape[0] * layer.shape[1]
        return n

    # -------- Bidirectional mapping: coords <-> nid --------
    def coords_to_nid(self, coords: List[Tuple[int, int]]) -> int:
        """Map outer-to-inner layer coordinates [(r_L3,c_L3), (r_L2,c_L2), (r_L1,c_L1)]
        to a single linear nid, with the innermost layer varying fastest:
          nid = k_L1
                + k_L2 * (M1*N1)
                + k_L3 * (M1*N1*M2*N2)
        where k_Lx = r_Lx * N_Lx + c_Lx.
        Supports any number of layers up to 3.
        """
        assert len(coords) == len(self.layers), "coords 层数需与 layers 一致"
        nid = 0
        stride = 1
        # Accumulate from inner to outer layers (innermost stride=1).
        for (r, c), topo in zip(reversed(coords), reversed(self.layers)):
            M, N = topo.shape
            assert 0 <= r < M and 0 <= c < N
            k = r * N + c
            nid += k * stride
            stride *= (M * N)
        return nid

    def nid_to_coords(self, nid: int) -> List[Tuple[int, int]]:
        """Convert a linear nid back to outer-to-inner layer coordinates, reversing coords_to_nid:
          Successively take the remainder and integer quotient by size_L1, size_L2,
          and size_L3 to obtain k_L1, k_L2, and k_L3. Then decompose each k_Lx
          into (r_Lx, c_Lx), where r=k//N and c=k%N.
        """
        coords_rev: List[Tuple[int, int]] = []  # Inner->outer.
        rem = int(nid)
        for topo in reversed(self.layers):  # Innermost layer first.
            M, N = topo.shape
            size = M * N
            k = rem % size
            rem //= size
            r = k // N
            c = k % N
            coords_rev.append((r, c))
        # rem>0 means nid exceeds capacity; tolerate this by ignoring higher-order digits (could instead raise an error).
        return list(reversed(coords_rev))

    def map_node(self, nid: int) -> List[Tuple[int, int]]:
        if self.node_mapper:
            return self.node_mapper(nid, self.layers)
        # Default: use a bidirectional mapping with "contiguous innermost-layer IDs"
        return self.nid_to_coords(nid)

    # Find the first layer index, from outermost to innermost, where "the two differ" (outer layers first)
    def first_diff_layer(self, s_coords: List[Tuple[int,int]], d_coords: List[Tuple[int,int]]) -> int:
        for i, (a, b) in enumerate(zip(s_coords, d_coords)):
            if a != b:
                return i
        return len(s_coords) - 1  # If all coordinates match, return the innermost layer.

    # Determine the ports for a cross-layer step (e.g., "down_out/down_in" for L3 -> L2)
    def ports_for_cross(self, topo: Topology, direction: str) -> List[int]:
        # direction in {"up_out","up_in","down_out","down_in","left_out","left_in","right_out","right_in"}
        return topo.ports(direction)

    # Core routing: map (src, dst, bytes) to one or more Chains, accounting for splits.
    def route(self, src: int, dst: int, bytes_value: int) -> List[Chain]:
        # Map to coordinates at each layer (outer->inner).
        s_coords = self.map_node(src)
        d_coords = self.map_node(dst)
        # print(f"s_coords: {s_coords}, d_coords: {d_coords}")
        log.debug("s_coords: %s, d_coords: %s", s_coords, d_coords)
        L = len(self.layers)
        chains: List[Chain] = [Chain(hops=[], bytes=bytes_value)]
        # Initial snapshot.
        for ch in chains:
            ch.states_coords.append(list(s_coords))
        # If the innermost layer is a switch, first place L1 at the center (-1,-1).
        if self.layers[-1].kind == TopoKind.SWITCH:
            for ch in chains:
                init = list(ch.states_coords[-1])
                init[-1] = (-1, -1)
                ch.states_coords.append(init)

        # Starting layer: begin at the "outermost layer where they diverge" (set to 0 to always start from the outermost layer)
        start_layer = self.first_diff_layer(s_coords, d_coords)

        # Current "source" ID in each layer (inner-layer sources are updated to cross-layer ingress ports as routing proceeds)
        cur_start_ids = [self.layers[i].to_id(*s_coords[i]) for i in range(L)]

        # Determine the direction (up/down/left/right) from a single hop.
        def hop_dir(topo: Topology, u: int, v: int) -> str:
            (ur, uc) = topo.to_rc(u)
            (vr, vc) = topo.to_rc(v)
            M, N = topo.shape
            if ur != vr:
                # Row direction.
                if topo.kind in {TopoKind.TORUS2D, TopoKind.MRING_NCHAIN}:
                    # Convention: a step r := (r-1)%M is "down" (matching the expected 0->3 direction); otherwise "up".
                    return "down" if (vr == (ur - 1) % M) else "up"
                else:
                    return "up" if vr > ur else "down"
            else:
                # Column direction.
                if topo.kind in {TopoKind.TORUS2D, TopoKind.RING, TopoKind.MCHAIN_NRING}:
                    # A step c := (c-1)%N is "left"; otherwise "right".
                    return "left" if (vc == (uc - 1) % N) else "right"
                else:
                    return "right" if vc > uc else "left"

        # Map port-set names.
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


        # Select port indices uniformly and deterministically, without splitting chains to avoid exponential growth.
        def select_index(num: int, layer_idx: int, hop_idx: int) -> int:
            if num <= 0:
                return -1
            # Use stable integer mixing to avoid Python's per-process random hash salt.
            x = (int(src) * 1315423911) ^ (int(dst) * 2654435761) ^ (int(layer_idx) * 97) ^ int(hop_idx)
            if x < 0:
                x = -x
            return x % num

        # Append snapshots with deduplication.
        def append_state(ch: Chain, snap: List[Tuple[int, int]]):
            if not ch.states_coords or ch.states_coords[-1] != snap:
                ch.states_coords.append(snap)

        # Drive routing from the outer layer, proceeding inward layer by layer.
        for layer_idx in range(start_layer, L):
            topo = self.layers[layer_idx]
            s_id = cur_start_ids[layer_idx]
            d_id = self.layers[layer_idx].to_id(*d_coords[layer_idx])

            # 1) Within this layer: find the XY (or wraparound) path from the current source to the target coordinates (record hops first, without generating snapshots yet).
            outer_hops = topo.route_intra(s_id, d_id)
            for ch in chains:
                for (u, v) in outer_hops:
                    ch.hops.append((layer_idx, u, v))

            # If another inner layer exists:
            if layer_idx < L - 1:
                inner_topo = self.layers[layer_idx + 1]

                # 2) For each outer-layer hop, place port placeholders between the two layers according to direction, then advance the "inner-layer source" to the corresponding ingress port
                last_snapshot = None
                for hop_idx, (u, v) in enumerate(outer_hops):
                    direction = hop_dir(topo, u, v)
                    out_side, in_side = port_sides(direction)
                    # Outer-layer ports used for link statistics (unchanged).
                    out_ports_outer = topo.ports(out_side)
                    in_ports_inner = inner_topo.ports(in_side)
                    # For human-readable snapshots: also assign an "outgoing-direction" port on the inner layer (egress on inner layer)
                    out_ports_inner = inner_topo.ports(out_side)
                    K = min(len(out_ports_outer), len(in_ports_inner), len(out_ports_inner) if out_ports_inner else len(in_ports_inner))
                    if K <= 0:
                        continue
                    sel = select_index(K, layer_idx, hop_idx)
                    out_id = out_ports_outer[sel]
                    in_id = in_ports_inner[sel]
                    inner_out_id = out_ports_inner[sel] if out_ports_inner else in_id

                    # First route within the inner layer from the current coordinates to "the inner-layer egress port required for this outer-layer hop"
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
                                    # Outer-layer ports used for link statistics (unchanged).
                                    ll_out_ports_outer = inner_topo.ports(ll_out_side)
                                    ll_in_ports_inner = ll_topo.ports(ll_in_side)
                                    # For human-readable snapshots: also assign an "outgoing-direction" port on the inner layer (egress on inner layer)
                                    ll_out_ports_inner = ll_topo.ports(ll_out_side)
                                    ll_K = min(len(ll_out_ports_outer), len(ll_in_ports_inner), len(ll_out_ports_inner) if ll_out_ports_inner else len(ll_in_ports_inner))
                                    if ll_K <= 0:
                                        continue
                                    ll_sel = select_index(ll_K, layer_idx+1, hop_idx)
                                    ll_out_id = ll_out_ports_outer[ll_sel]
                                    ll_in_id = ll_in_ports_inner[ll_sel]
                                    ll_inner_out_id = ll_out_ports_inner[ll_sel] if ll_out_ports_inner else ll_in_id
                                    ll_inner_cur = cur_start_ids[-1]
                                    # Route within the innermost layer from the current coordinates to "the innermost-layer egress port required for this inner-layer hop"
                                    if ll_inner_cur != ll_inner_out_id:
                                        ll_inner_steps_to_out = ll_topo.route_intra(ll_inner_cur, ll_inner_out_id)
                                        for (uuu, vvv) in ll_inner_steps_to_out:
                                            ch.hops.append((-1, uuu, vvv))
                                            vr3, vc3 = self.layers[-1].to_rc(vvv) if vvv != -1 else (-1, -1)
                                            snap_step2 = list(snap_base)
                                            snap_step2[-1] = (vr3, vc3)
                                            append_state(ch, snap_step2)
                                            snap_base = snap_step2
                                    # Cross-layer transition.
                                    snap_step3 = list(snap_base)
                                    snap_step3[layer_idx + 1] = self.layers[layer_idx + 1].to_rc(vv) if vv != -1 else (-1, -1)
                                    snap_step3[-1] = self.layers[-1].to_rc(ll_in_id) if ll_in_id != -1 else (-1, -1)
                                    append_state(ch, snap_step3)
                                    # Route to the previous layer's egress port.
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
                        # State trace: first record the "inner-layer egress port placeholder", while the outer layer remains at u
                        ur, uc = self.layers[layer_idx].to_rc(u) if u != -1 else (-1, -1)
                        ior, ioc = self.layers[layer_idx + 1].to_rc(inner_out_id) if inner_out_id != -1 else (-1, -1)
                        prev = list(ch.states_coords[-1]) if last_snapshot is None else list(last_snapshot)
                        snap_out = list(prev)
                        snap_out[layer_idx] = (ur, uc)
                        snap_out[layer_idx + 1] = (ior, ioc)
                        append_state(ch, snap_out)
                        last_snapshot = snap_out
                        # State trace: after the outer-layer hop, the outer layer is at v and the inner layer is at the ingress port.
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
                    # Advance the next layer's "current source"
                    cur_start_ids[layer_idx + 1] = in_id

                    # Immediately after this outer-layer hop, route within the inner layer from the ingress to "the egress needed next" or the final destination
                    inner_cur = cur_start_ids[layer_idx + 1]
                    if hop_idx < len(outer_hops) - 1:
                        # If another outer-layer hop follows, compute the inner-layer egress port required for its direction.
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
                                    # Outer-layer ports used for link statistics (unchanged).
                                    ll_out_ports_outer = inner_topo.ports(ll_out_side)
                                    ll_in_ports_inner = ll_topo.ports(ll_in_side)
                                    # For human-readable snapshots: also assign an "outgoing-direction" port on the inner layer (egress on inner layer)
                                    ll_out_ports_inner = ll_topo.ports(ll_out_side)
                                    ll_K = min(len(ll_out_ports_outer), len(ll_in_ports_inner), len(ll_out_ports_inner) if ll_out_ports_inner else len(ll_in_ports_inner))
                                    if ll_K <= 0:
                                        continue
                                    ll_sel = select_index(ll_K, layer_idx+1, hop_idx)
                                    ll_out_id = ll_out_ports_outer[ll_sel]
                                    ll_in_id = ll_in_ports_inner[ll_sel]
                                    ll_inner_out_id = ll_out_ports_inner[ll_sel] if ll_out_ports_inner else ll_in_id
                                    ll_inner_cur = cur_start_ids[-1]
                                    # Route within the innermost layer from the current coordinates to "the innermost-layer egress port required for this inner-layer hop"
                                    if ll_inner_cur != ll_inner_out_id:
                                        ll_inner_steps_to_out = ll_topo.route_intra(ll_inner_cur, ll_inner_out_id)
                                        for (uuu, vvv) in ll_inner_steps_to_out:
                                            ch.hops.append((-1, uuu, vvv))
                                            vr3, vc3 = self.layers[-1].to_rc(vvv) if vvv != -1 else (-1, -1)
                                            snap_step2 = list(snap_base)
                                            snap_step2[-1] = (vr3, vc3)
                                            append_state(ch, snap_step2)
                                            snap_base = snap_step2
                                    # Cross-layer transition.
                                    snap_step3 = list(snap_base)
                                    snap_step3[layer_idx + 1] = self.layers[layer_idx + 1].to_rc(vv) if vv != -1 else (-1, -1)
                                    snap_step3[-1] = self.layers[-1].to_rc(ll_in_id) if ll_in_id != -1 else (-1, -1)
                                    append_state(ch, snap_step3)
                                    # Route to the previous layer's egress port.
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
                        # Last outer-layer hop: route to the inner layer's final target.
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
                                    # Outer-layer ports used for link statistics (unchanged).
                                    ll_out_ports_outer = inner_topo.ports(ll_out_side)
                                    ll_in_ports_inner = ll_topo.ports(ll_in_side)
                                    # For human-readable snapshots: also assign an "outgoing-direction" port on the inner layer (egress on inner layer)
                                    ll_out_ports_inner = ll_topo.ports(ll_out_side)
                                    ll_K = min(len(ll_out_ports_outer), len(ll_in_ports_inner), len(ll_out_ports_inner) if ll_out_ports_inner else len(ll_in_ports_inner))
                                    if ll_K <= 0:
                                        continue
                                    ll_sel = select_index(ll_K, layer_idx+1, hop_idx)
                                    ll_out_id = ll_out_ports_outer[ll_sel]
                                    ll_in_id = ll_in_ports_inner[ll_sel]
                                    ll_inner_out_id = ll_out_ports_inner[ll_sel] if ll_out_ports_inner else ll_in_id
                                    ll_inner_cur = cur_start_ids[-1]
                                    # Route within the innermost layer from the current coordinates to "the innermost-layer egress port required for this inner-layer hop"
                                    if ll_inner_cur != ll_inner_out_id:
                                        ll_inner_steps_to_out = ll_topo.route_intra(ll_inner_cur, ll_inner_out_id)
                                        for (uuu, vvv) in ll_inner_steps_to_out:
                                            ch.hops.append((-1, uuu, vvv))
                                            vr3, vc3 = self.layers[-1].to_rc(vvv) if vvv != -1 else (-1, -1)
                                            snap_step2 = list(snap_base)
                                            snap_step2[-1] = (vr3, vc3)
                                            append_state(ch, snap_step2)
                                            snap_base = snap_step2
                                    # Cross-layer transition.
                                    snap_step3 = list(snap_base)
                                    snap_step3[layer_idx + 1] = self.layers[layer_idx + 1].to_rc(vv) if vv != -1 else (-1, -1)
                                    snap_step3[-1] = self.layers[-1].to_rc(ll_in_id) if ll_in_id != -1 else (-1, -1)
                                    append_state(ch, snap_step3)
                                    # Route to the previous layer's egress port.
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

                # 3) After completing the outer layer, inner-layer routing has already occurred after each outer-layer hop; no additional full routing pass is needed.

        # Finish: if the innermost layer remains at the ingress (e.g., switch center -1,-1), append the final snapshot from ingress to the target port.
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


# ---------------- Statistics and latency calculation ----------------

@dataclass
class RouteStats:
    max_logical_hops: int                 # Maximum hop count among all (src,dst) chains, measured per chain.
    max_hop_latency: float                # Maximum total hop latency across chains, summing each layer's hop latency.
    # Note: Link endpoints u,v now denote "extended global node IDs (including outer-layer coordinates and inner-layer anchors)", not local IDs within this layer.
    link_bytes: Dict[Link, int]           # Accumulated bytes on each "global physical edge" (layer_idx, ext_u, ext_v)
    max_link_load: Tuple[Link, int]       # (link, bytes)
    max_link_time: Tuple[Link, float]     # (link, bytes/bw)
    # Single-stage latency = max_hop_latency + max_link_time[1].
    stage_latency: float

# ---------------- Convenience factories and examples ----------------

# def example_hierarchy_for_your_comment() -> Hierarchy:
#     """
#     Example:
#       L1: switch N=4
#       L2: 2D mesh = (2x2)
#       L3: 2D torus = (4x4)
#     Outer to inner: layers = [L3, L2, L1].
#     """
#     # The caller supplies each layer's hop_latency, link_bandwidth, and switch-center bandwidth.
#     # Use the default mapping with "contiguous innermost-layer IDs"; no custom mapper is supplied
#     return Hierarchy(layers=[L3, L2, L1], port_spread=PortSpread.EVEN, node_mapper=None)




# ======== Extended matrices and visualization (bandwidth/traffic/utilization) =============================

def _build_extended_indexer(h: Hierarchy):
    """Add an innermost anchor coordinate (-1,-1) to the extended node space.
    For SWITCH this is the center; otherwise it is a virtual cross-layer placeholder
    with no L1 internal bandwidth or self-loop. Return ext_size, coords_to_ext_id,
    ext_id_to_coords, and anchor_token=(-1,-1).
    """
    layers = h.layers
    L = len(layers)
    assert L >= 1, "hierarchy 至少一层"

    # Radix of each layer (flat size).
    base_sizes = []
    for i, topo in enumerate(reversed(layers)):
        M, N = topo.shape
        base_sizes.append(M * N)
    base_sizes = list(reversed(base_sizes))

    # Append 1 anchor to the innermost layer, regardless of its type.
    l1_extra = 1
    # Extended size of each layer (+1 only for the final layer).
    ext_layer_sizes = base_sizes[:-1] + [base_sizes[-1] + l1_extra]

    # Compute strides (inner to outer).
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
                k = base_sizes[-1]  # Index of the appended anchor.
            else:
                k = r * N + c
            eid += k * strides[li]
        return eid

    def ext_id_to_coords(eid: int) -> list[tuple[int, int]]:
        rem = int(eid)
        coords_rev: list[tuple[int, int]] = []
        # Decompose inner->outer.
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
    """Generate directed 2D-neighbor edges (u_idx, v_idx, direction) for
    MESH2D, TORUS2D, RING, CHAIN, and ALL2ALL topologies.
    """
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
                # Row-direction chain (no wraparound).
                if r + 1 < M:
                    edges.append((id_of(r, c), id_of(r + 1, c), "down"))
                if r - 1 >= 0:
                    edges.append((id_of(r, c), id_of(r - 1, c), "up"))
                # Column-direction ring (wraparound).
                edges.append((id_of(r, c), id_of(r, (c + 1) % N), "right"))
                edges.append((id_of(r, c), id_of(r, (c - 1) % N), "left"))
        return edges
    if kind == TopoKind.MRING_NCHAIN:
        for r in range(M):
            for c in range(N):
                # Row-direction ring (wraparound).
                edges.append((id_of(r, c), id_of((r + 1) % M, c), "down"))
                edges.append((id_of(r, c), id_of((r - 1) % M, c), "up"))
                # Column-direction chain (no wraparound).
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
    """Build the extended bandwidth matrix (320x320 in the example).
    L1 SWITCH center-port edges use L1.link_bandwidth; the center self-loop uses
    center_in_bw + center_out_bw. Within each L3 tile, connect neighboring L2 nodes
    using L2.link_bandwidth and the L1 center anchor. Across neighboring L3 tiles,
    connect corresponding directional L2 out/in ports through the L1 anchors
    using L3.link_bandwidth.
    """
    L = len(h.layers)
    assert L >= 1
    ext_size, coords2eid, _, anchor = _build_extended_indexer(h)
    bw = np.zeros((ext_size, ext_size), dtype=np.float64)

    # Convenient access.
    def with_coords(li: int, k: int) -> tuple[int, int]:
        topo = h.layers[li]
        M, N = topo.shape
        return (k // N, k % N)

    # Iterate over the Cartesian product of all outer-layer coordinates.
    def iter_outer_coords():
        if L == 1:
            yield []
            return
        # L>=2
        # Outer to inner (excluding L1).
        ranges = []
        for topo in h.layers[:-1]:
            Mx, Nx = topo.shape
            ranges.append([(r, c) for r in range(Mx) for c in range(Nx)])
        # Cartesian product.
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
        # Create center <-> port connections for each outer-layer (L3/L2) coordinate combination.
        outer_sizes = []
        for topo in h.layers[:-1]:
            Mx, Nx = topo.shape
            outer_sizes.append(Mx * Nx)

        center_bw = float(
            (L1.switch_center_in_bw if L1.switch_center_in_bw is not None else 0.0)
            + (L1.switch_center_out_bw if L1.switch_center_out_bw is not None else 0.0)
        )
        
        for outer in iter_outer_coords():
            # Each L1 port.
            for p in range(N1):  # M1==1
                src_coords = list(outer) + [(0, p)]
                cen_coords = list(outer) + [anchor]
                a = coords2eid(src_coords)
                b = coords2eid(cen_coords)
                bw[a, b] = max(bw[a, b], float(L1.link_bandwidth))
                bw[b, a] = max(bw[b, a], float(L1.link_bandwidth))
            # Center self-loop.
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
                
    # L2: intralayer adjacency.
    if L >= 2:
        L2 = h.layers[-2]
        if L2.kind in {TopoKind.MESH2D, TopoKind.TORUS2D, TopoKind.RING, TopoKind.CHAIN, TopoKind.MCHAIN_NRING, TopoKind.MRING_NCHAIN}:
            M2, N2 = L2.shape
            edges2 = _neighbors_2d(L2.kind, M2, N2)
            # Iterate over each outer-layer L3 coordinate.
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

    # L3: intralayer adjacency (connected through L2 ports).
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
    """Accumulate all chain traffic into the extended matrix using states_coords.
    Return (traffic_matrix, max_hop_latency_states). For an innermost SWITCH,
    center-entry and center-exit hops also add their bytes to the center self-loop.
    """
    ext_size, coords2eid, _, anchor = _build_extended_indexer(h)
    traffic = np.zeros((ext_size, ext_size), dtype=np.float64)

    # Generate all chains (using the same entry point as compute_stage_latency).
    flat = tm.counts
    nz = np.nonzero(flat)
    max_hop_latency = 0.0
    for s, d in zip(nz[0], nz[1]):
        val = int(flat[s, d])
        if val <= 0 or s == d:
            continue
        chains = h.route(s, d, val)
        for ch in chains:
            # Iterate over adjacent snapshots.
            hops_lat = 0.0
            for i in range(1, len(ch.states_coords)):
                prev = ch.states_coords[i - 1]
                cur = ch.states_coords[i]
                # Find the outermost layer that changed.
                layer_idx = None
                for li, (a, b) in enumerate(zip(prev, cur)):
                    if a != b:
                        layer_idx = li
                        break
                if layer_idx is None:
                    continue
                # Map endpoints.
                a_eid = coords2eid(prev)
                b_eid = coords2eid(cur)
                traffic[a_eid, b_eid] += float(ch.bytes)
                # Accumulate hop latency.
                hops_lat += float(h.layers[layer_idx].hop_latency)
                # If the innermost layer is SWITCH and the center anchor is involved, add traffic to the center self-loop.
                if layer_idx == len(h.layers) - 1 and h.layers[-1].kind == TopoKind.SWITCH:
                    if prev[-1] == anchor or cur[-1] == anchor:
                        c_eid = coords2eid(list(cur[:-1]) + [anchor])
                        traffic[c_eid, c_eid] += float(ch.bytes)
            if hops_lat > max_hop_latency:
                max_hop_latency = hops_lat

    return traffic, max_hop_latency


def compute_utilization(traffic: np.ndarray, bandwidth: np.ndarray, base_time: float) -> tuple[np.ndarray, float]:
    """Compute max_link_time=max(traffic_ij/bw_ij), total_time=base_time+max_link_time,
    and util_ij=traffic_ij/(total_time*bw_ij). Zero-bandwidth entries have zero
    utilization and are excluded from max_link_time.
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
    """Compute per-link energy as traffic_bytes * 8 * energy_per_bit in pJ.
    Return link energies and their sum. Zero energy-per-bit entries contribute zero.
    """
    noc_energy_pj = np.where(energy_per_bit > 0.0, traffic * 8.0 * energy_per_bit, 0.0)
    total_noc_energy_pj = float(np.sum(noc_energy_pj))
    return noc_energy_pj, total_noc_energy_pj


def get_total_traffic_bytes(traffic_mats: list) -> float:
    """Sum bytes on all physical links across expanded traffic matrices.
    Accept matrices collected in OpPerfStats._traffic_mats; return 0.0 for an empty list.
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
    # Transparent-green-yellow-red implementation (retained as comments):
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

    # Classic heatmap: similar to the old save_heatmap style (viridis, no transparency)
    data = matrix.astype(float, copy=True)
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap="viridis", origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ==== Font enhancements BEGIN ====
    cbar.set_label(bar_label, fontsize=16)           # Colorbar title
    cbar.ax.tick_params(labelsize=16)              # Colorbar ticks
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)  # Font for scientific notation above the colorbar

    ax.set_title(title, fontsize=16)
    # ax.set_xlabel("dst (extended id)", fontsize=16)
    # ax.set_ylabel("src (extended id)", fontsize=16)
    ax.set_xlabel("Destination", fontsize=16)
    ax.set_ylabel("Source", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    # ==== Font enhancements END ====

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_extended_matrices_and_plots(
    tm: "TrafficMatrix",
    h: Hierarchy,
    out_prefix: str = "extended"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Generate and save bandwidth, traffic, utilization, and energy matrix PDFs.
    Traffic uses states_coords; total time is state-hop latency + max(traffic/bw).
    Per-link energy is traffic * 8 * energy_per_bit.
    Return (bw, traffic, util, total_time, noc_energy_pj, total_noc_energy_pj).
    """
    from .noc_energy import build_extended_energy_matrix  # Lazy import to avoid circular dependencies

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
    """Accumulate states_coords traffic in an all-switch extended coordinate system.
    Every level has a center anchor. A hop entering or leaving a level's center
    also adds the same bytes to its center self-loop to account for center throughput.
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
                # Find the outermost layer that changed.
                layer_idx = None
                for li, (a, b) in enumerate(zip(prev, cur)):
                    if a != b:
                        layer_idx = li; break
                if layer_idx is None:
                    continue
                # Normalize coordinates: retain actual coordinates only in layer layer_idx; set all other layers to their anchors.
                # This matches the coordinate convention in build_extended_bandwidth_matrix_switch_only,
                # ensuring traffic[a,b] and bw[a,b] refer to the same physical link entry.
                L = len(h.layers)
                prev_norm = [anchors[li] if li != layer_idx else prev[li] for li in range(L)]
                cur_norm  = [anchors[li] if li != layer_idx else cur[li]  for li in range(L)]
                a_eid = coords2eid(prev_norm)
                b_eid = coords2eid(cur_norm)
                traffic[a_eid, b_eid] += float(ch.bytes)
                # Accumulate hop latency.
                hops_lat += float(h.layers[layer_idx].hop_latency)

            if hops_lat > max_hop_latency:
                max_hop_latency = hops_lat

    return traffic, max_hop_latency


def _build_extended_indexer_switch_only(h: Hierarchy):
    """Build an extended indexer for a hierarchy containing only SWITCH levels.
    Add a center anchor (-1,-1) at every level, ordered outermost to innermost.
    Return ext_size, coords_to_ext_id, ext_id_to_coords, and the per-level anchors.
    """
    assert len(h.layers) >= 1, "hierarchy 至少一层"
    for topo in h.layers:
        if topo.kind != TopoKind.SWITCH:
            raise ValueError("本索引器仅支持所有层均为 SWITCH 的情形")

    L = len(h.layers)
    base_sizes = [(topo.shape[0] * topo.shape[1]) for topo in h.layers]  # For a switch, this is the number of ports N.
    # Add 1 to each layer for the center anchor (-1,-1).
    ext_layer_sizes = [s + 1 for s in base_sizes]

    # Compute stride (inner->outer).
    strides = [1] * L
    for i in range(L - 2, -1, -1):
        strides[i] = strides[i + 1] * ext_layer_sizes[i + 1]

    ext_size = 1
    for s in ext_layer_sizes:
        ext_size *= s

    def _coords_to_k(li: int, rc: tuple[int, int]) -> int:
        """Map (r,c) to a linear index within this layer; (-1,-1) is the appended center, with index == base_size."""
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
    """Build the extended bandwidth matrix for a three-level all-switch hierarchy.
    At each level, connect its center anchor (-1,-1) bidirectionally to physical
    ports (0,p), using link_bandwidth. Combine duplicate edge capacities by maximum,
    not summation. If center bandwidths are set, the center self-loop uses
    center_in_bw + center_out_bw, also combined by maximum. Other levels use anchor
    coordinates so all edges share one extended coordinate system.
    """
    # Validate.
    for topo in h.layers:
        if topo.kind != TopoKind.SWITCH:
            raise ValueError("build_extended_bandwidth_matrix_switch_only 仅支持 SWITCH 层")

    L = len(h.layers)
    ext_size, coords2eid, _, anchors = _build_extended_indexer_switch_only(h)
    bw = np.zeros((ext_size, ext_size), dtype=np.float64)

    # Enumerate the Cartesian product of "outer-layer coordinates" (through li-1); use anchor placeholders for li and inner layers (or enumerate ports at li)
    def iter_prefix_coords(upto_exclusive: int):
        """Return coordinate combinations for layers 0 through upto_exclusive-1.
        (Does each layer contain only (-1,-1) or port coordinates? For SWITCH,
        port coordinates are used only at the current layer.)
        """
        if upto_exclusive <= 0:
            yield []
            return
        ranges = []
        for li in range(upto_exclusive):
            topo = h.layers[li]
            # Place only anchors in the "outer-layer prefix" to avoid exponential growth; resolve them during routing
            ranges.append([anchors[li]])
        # Cartesian product.
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
        M, N = topo.shape  # For a switch, M=1, N=number of ports.
        center_loop_bw = float(
            (topo.switch_center_in_bw or 0.0) + (topo.switch_center_out_bw or 0.0)
        )
        # Fix outer-prefix coordinates (0..li-1) and inner-suffix coordinates (li+1..) at their anchors.
        for prefix in iter_prefix_coords(li):
            # Assemble "the full extended coordinates of this layer's central node"
            center_coords = list(prefix) + [anchors[li]] + anchors[li+1:]
            center_eid = coords2eid(center_coords)
            # Bidirectional bandwidth edges between each port and the center of this layer.
            for p in range(N):  # M==1
                port_coords = list(prefix) + [(0, p)] + anchors[li+1:]
                port_eid = coords2eid(port_coords)
                # Bidirectional.
                bw[port_eid, center_eid] = max(bw[port_eid, center_eid], float(topo.link_bandwidth))
                bw[center_eid, port_eid] = max(bw[center_eid, port_eid], float(topo.link_bandwidth))
            # Center self-loop.
            if center_loop_bw > 0.0:
                bw[center_eid, center_eid] = max(bw[center_eid, center_eid], center_loop_bw)

    return bw

def export_extended_matrices_and_plots_switch_only(
    tm: "TrafficMatrix",
    h: Hierarchy,
    out_prefix: str = "extended_switch_only"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Export all-switch bandwidth, traffic, utilization, and energy matrix PNGs.
    Per-link energy is traffic * 8 * energy_per_bit in pJ.
    Return (bw, traffic, util, total_time, noc_energy_pj, total_noc_energy_pj).
    """
    from .noc_energy import build_extended_energy_matrix_switch_only  # Lazy import to avoid circular dependencies

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
    """Evaluate an all-switch hierarchy using expanded states_coords traffic.
    Fill missing bandwidth where traffic is positive, assigning the outermost
    changed level. Self-loops use center_in_bw + center_out_bw; other edges use
    that level's link_bandwidth. Return (hop_time, ext_max, overall_time).
    """
    # 1) Validate and compute traffic / hop_time.
    if not all(t.kind == TopoKind.SWITCH for t in h.layers):
        raise ValueError("get_extend_max_routes_switch_only 仅支持所有层均为 SWITCH")

    traffic, hop_time = build_extended_traffic_matrix_switch_only(tm, h)

    # 2) Fill in bandwidth as needed: assign it only where traffic is nonzero, with bandwidth ownership determined by "the outermost layer that changes"
    _, _, extid2coords, _anchors = _build_extended_indexer_switch_only(h)
    bw = np.zeros_like(traffic, dtype=np.float64)

    nz_src, nz_dst = np.nonzero(traffic)
    for i, j in zip(nz_src.tolist(), nz_dst.tolist()):
        tij = float(traffic[i, j])
        if tij <= 0.0:
            continue

        ci = extid2coords(int(i))
        cj = extid2coords(int(j))

        # Find "the outermost layer that changes", layer_idx
        layer_idx = None
        for li, (a, b) in enumerate(zip(ci, cj)):
            if a != b:
                layer_idx = li
                break

        if layer_idx is None:
            # i==j: a self-loop, which must be a layer's center self-loop.
            # Use center_in+center_out from the layer whose coordinates are (-1,-1).
            loop_bw = 0.0
            for li, rc in enumerate(ci):
                if rc == (-1, -1):
                    topo = h.layers[li]
                    loop_bw = max(loop_bw, float((topo.switch_center_in_bw or 0.0) + (topo.switch_center_out_bw or 0.0)))
            # Guard: if no layer has coordinates (-1,-1), do not write an entry (or write 0).
            if loop_bw > 0.0:
                bw[i, j] = max(bw[i, j], loop_bw)
            continue

        # Non-self-loop: this hop belongs to layer_idx; use that layer's link_bandwidth.
        topo = h.layers[layer_idx]
        hop_bw = float(topo.link_bandwidth)
        if hop_bw > 0.0:
            bw[i, j] = max(bw[i, j], hop_bw)

    # 3) Compute the bottleneck time and total time.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(bw > 0.0, traffic / bw, -np.inf)

    if ratio.size == 0 or not np.isfinite(np.max(ratio)):
        # Return 0 directly for an empty traffic matrix (valid cases with no inter-node communication, such as ep=1)
        log.debug("[extended/switch-only] traffic matrix is empty (no inter-node comm); returning (0,0,0)")
        return 0.0, 0.0, 0.0

    flat_idx = int(np.nanargmax(ratio))
    i_star, j_star = divmod(flat_idx, ratio.shape[1])
    ext_max = float(ratio[i_star, j_star]) if np.isfinite(ratio[i_star, j_star]) else 0.0

    # Print diagnostics.
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
    """Print the maximum link-time breakdown for the extended path, max(traffic/bw)."""
    log.info("Diagnosing max link time")

    if all(t.kind == TopoKind.SWITCH for t in h.layers):
        hop_time, ext_max, overall_time = get_extend_max_routes_switch_only(tm, h)
        return hop_time, ext_max, overall_time
    else:
        # Extended matrix.
        bw = build_extended_bandwidth_matrix(h)
        traffic, hop_time = build_extended_traffic_matrix(tm, h)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(bw > 0.0, traffic / bw, -np.inf)
        if ratio.size > 0:
            flat_idx = int(np.nanargmax(ratio))
            i, j = divmod(flat_idx, ratio.shape[1])
            ext_max = float(ratio[i, j]) if np.isfinite(ratio[i, j]) else 0.0
            # Restore extended coordinates.
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
    """Return (hop_time, ext_max, overall_time, traffic) for direct OpPerfStats use.

    Example:
        hop_time, ext_max, overall_time, traffic = get_extend_max_routes_with_traffic(tm, h)
        stats.append_traffic(traffic, link_time_s=ext_max, hop_time_s=hop_time, comm_time_s=overall_time)

    hop_time is accumulated switch-hop latency; ext_max is bottleneck transfer time;
    overall_time is their sum. The default dump_perf_log=False combines the ordinary
    route calculation with traffic construction without diagnostics. With True,
    the inlined diagnostic path builds traffic only once and logs detailed results.
    """
    if not dump_perf_log:
        # -- Without dump_perf_log ------------------------------------------------
        hop_time, ext_max, overall_time = get_extend_max_routes_with_traffic(tm, h)

        return float(hop_time), float(ext_max), float(overall_time), None

    # -- Full path: compute inline, build traffic only once, and print detailed diagnostic logs --
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
    """Print the maximum link-time breakdown for the extended path, max(traffic/bw)."""
    # Extended matrix.
    bw = build_extended_bandwidth_matrix(h)
    traffic, hop_time = build_extended_traffic_matrix(tm, h)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(bw > 0.0, traffic / bw, -np.inf)
    if ratio.size > 0:
        flat_idx = int(np.nanargmax(ratio))
        i, j = divmod(flat_idx, ratio.shape[1])
        ext_max = float(ratio[i, j]) if np.isfinite(ratio[i, j]) else 0.0
        # Restore extended coordinates.
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
    """Find the states_coords chain with the largest sum of per-layer hop latencies.
    Return (best_path_states, best_hop_time, (src_idx, dst_idx)).
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
    """Print the path with the largest hop-latency sum using adjacent states_coords snapshots."""
    path, hop_time, pair = get_longest_states_path(tm, h)
    if not path:
        log.info("[longest] no valid paths")
        return
    log.info("[longest] src=%s dst=%s hop_time_s=%.6g steps=%d", pair[0], pair[1], hop_time, max(0, len(path) - 1))
    # coords view of the overall source/destination.
    log.info("  start_coords: %s", path[0])
    log.info("  dst_coords:   %s", path[-1])

    cum = 0.0
    for i in range(1, len(path)):
        prev = path[i - 1]
        cur = path[i]
        # Find the outermost layer that changed.
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
