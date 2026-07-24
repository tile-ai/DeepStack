from __future__ import annotations
import numpy as np

from .noc_topo import (
    Hierarchy,
    TopoKind,
    _build_extended_indexer,
    _build_extended_indexer_switch_only,
    _neighbors_2d,
    _side_map,
)
from .energy_config import NocEnergyConfig


def _reference_energy_for_layer(layer_index: int, layer_count: int) -> float:
    from . import _model_support

    return float(
        _model_support.p96(
            layer_index,
            layer_count,
        )
    )


def build_extended_energy_matrix(
    h: Hierarchy,
    energy_config: NocEnergyConfig | None = None,
) -> np.ndarray:
    L = len(h.layers)
    assert L >= 1
    cfg = energy_config or h.energy_config
    if cfg is None:
        L1_ENERGY_PER_BIT = _reference_energy_for_layer(L - 1, L)
        L2_ENERGY_PER_BIT = (
            _reference_energy_for_layer(L - 2, L) if L >= 2 else 0.0
        )
        L3_ENERGY_PER_BIT = (
            _reference_energy_for_layer(L - 3, L) if L >= 3 else 0.0
        )
    else:
        L1_ENERGY_PER_BIT = cfg.l1_pj_per_bit
        L2_ENERGY_PER_BIT = cfg.l2_pj_per_bit
        L3_ENERGY_PER_BIT = cfg.l3_pj_per_bit

    ext_size, coords2eid, _, anchor = _build_extended_indexer(h)
    bw = np.zeros((ext_size, ext_size), dtype=np.float64)

    def with_coords(li: int, k: int) -> tuple[int, int]:
        topo = h.layers[li]
        M, N = topo.shape
        return (k // N, k % N)

    def iter_outer_coords():
        if L == 1:
            yield []
            return
        ranges = []
        for topo in h.layers[:-1]:
            Mx, Nx = topo.shape
            ranges.append([(r, c) for r in range(Mx) for c in range(Nx)])
        def rec(idx, acc):
            if idx == len(ranges):
                yield list(acc)
            else:
                for v in ranges[idx]:
                    acc.append(v)
                    yield from rec(idx + 1, acc)
                    acc.pop()
        yield from rec(0, [])

    # L1
    if h.layers[-1].kind == TopoKind.SWITCH:
        L1 = h.layers[-1]
        M1, N1 = L1.shape
        center_bw = float(L1_ENERGY_PER_BIT)
        for outer in iter_outer_coords():
            for p in range(N1):
                src_coords = list(outer) + [(0, p)]
                cen_coords = list(outer) + [anchor]
                a = coords2eid(src_coords)
                b = coords2eid(cen_coords)
                bw[a, b] = L1_ENERGY_PER_BIT
                bw[b, a] = L1_ENERGY_PER_BIT
            if center_bw > 0.0:
                c = coords2eid(list(outer) + [anchor])
                bw[c, c] = L1_ENERGY_PER_BIT
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
                bw[a, b] = L1_ENERGY_PER_BIT

    # L2
    if L >= 2:
        L2 = h.layers[-2]
        if L2.kind in {TopoKind.MESH2D, TopoKind.TORUS2D, TopoKind.RING, TopoKind.CHAIN, TopoKind.MCHAIN_NRING, TopoKind.MRING_NCHAIN}:
            M2, N2 = L2.shape
            edges2 = _neighbors_2d(L2.kind, M2, N2)
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
                        else:
                            coords_a = [oc, (ur, uc), anchor]
                            coords_b = [oc, (vr, vc), anchor]
                        a = coords2eid(coords_a)
                        b = coords2eid(coords_b)
                        bw[a, b] = L2_ENERGY_PER_BIT
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
                            else:
                                coords_a = [oc, (ur, uc), (or1, oc1)]
                                coords_b = [oc, (vr, vc), (ir1, ic1)]
                            a = coords2eid(coords_a)
                            b = coords2eid(coords_b)
                            bw[a, b] = L2_ENERGY_PER_BIT

    # L3
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
                    bw[a, b] = L3_ENERGY_PER_BIT
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
                        bw[a, b] = L3_ENERGY_PER_BIT

    return bw


def build_extended_energy_matrix_switch_only(
    h: Hierarchy,
    energy_config: NocEnergyConfig | None = None,
) -> np.ndarray:
    """
    纯 SWITCH 多层拓扑的扩展能耗矩阵（pJ/bit）。

    与 build_extended_bandwidth_matrix_switch_only 结构完全一致，
    仅将 topo.link_bandwidth 替换为对应层级的 pJ/bit 常数
    （通过 caller config 或 bundled release provider 获取）。
    """
    cfg = energy_config or h.energy_config

    for topo in h.layers:
        if topo.kind != TopoKind.SWITCH:
            raise ValueError("build_extended_energy_matrix_switch_only 仅支持 SWITCH 层")

    L = len(h.layers)
    ext_size, coords2eid, _, anchors = _build_extended_indexer_switch_only(h)
    energy = np.zeros((ext_size, ext_size), dtype=np.float64)

    def iter_prefix_coords(upto_exclusive: int):
        if upto_exclusive <= 0:
            yield []
            return
        ranges = [[anchors[li]] for li in range(upto_exclusive)]
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
        _, N = topo.shape
        e = (
            cfg.for_layer(li, L)
            if cfg is not None
            else _reference_energy_for_layer(li, L)
        )
        for prefix in iter_prefix_coords(li):
            center_coords = list(prefix) + [anchors[li]] + anchors[li + 1:]
            center_eid = coords2eid(center_coords)
            for p in range(N):
                port_coords = list(prefix) + [(0, p)] + anchors[li + 1:]
                port_eid = coords2eid(port_coords)
                energy[port_eid, center_eid] = max(energy[port_eid, center_eid], e)
                energy[center_eid, port_eid] = max(energy[center_eid, port_eid], e)
            # center self-loop omitted: all layers collapse to the same all-anchor node
            # so the energy rate cannot be assigned per-layer; switch fabric energy
            # is not modelled in switch_only (unlike non-switch-only where each L1
            # group has its own center node with correct l1_pj_per_bit).

    return energy
