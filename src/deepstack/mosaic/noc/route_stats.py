from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np

from .noc_topo import (
    Hierarchy,
    TopoKind,
    build_extended_bandwidth_matrix,
    build_extended_bandwidth_matrix_switch_only,
    build_extended_traffic_matrix,
    build_extended_traffic_matrix_switch_only,
)
from .traffic_matrix import TrafficMatrix
from .energy_config import NocEnergyConfig
from .noc_energy import (
    build_extended_energy_matrix,
    build_extended_energy_matrix_switch_only,
)

log = logging.getLogger(__name__)


@dataclass
class RouteStats:
    """
    逻辑 TrafficMatrix 经过 Hierarchy 路由展开后的链路资源快照。

    包含三路并行矩阵（shape [ext_size × ext_size]，每个 (i,j) 对应一条物理有向链路）：
      - 带宽路：traffic (bytes) / total_time_s / bw → util (0..1)
      - 能耗路：traffic (bytes) × 8 × energy_per_bit (pJ/bit) → noc_energy_pj (pJ)

    两种拓扑均支持：
      - 纯 SWITCH 多层（build_extended_*_switch_only）
      - 混合 MESH / TORUS / RING / CHAIN（build_extended_*）
    """

    # ── Bottleneck link（扩展节点 ID）────────────────────────────────────────
    max_link_src: int       # 瓶颈链路 src 的 extended node ID
    max_link_dst: int       # 瓶颈链路 dst 的 extended node ID
    max_link_bytes: float   # 该链路承载的字节数
    max_link_bw: float      # 该链路的带宽 (bytes/s)

    # ── Extended matrices [ext_size × ext_size] ──────────────────────────────
    traffic: np.ndarray = field(repr=False)  # 字节数
    bw:      np.ndarray = field(repr=False)  # 带宽 (bytes/s)
    util:    np.ndarray = field(repr=False)  # 利用率 (0..1) = traffic/total_time_s/bw

    # ── Energy ───────────────────────────────────────────────────────────────
    energy_per_bit:      np.ndarray = field(repr=False)  # pJ/bit per link
    noc_energy_pj:       np.ndarray = field(repr=False)  # pJ per link = traffic * 8 * energy_per_bit
    total_noc_energy_pj: float = 0.0                     # sum(noc_energy_pj)，单位 pJ


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def _build_route_stats_core(
    total_traffic: np.ndarray,
    bw: np.ndarray,
    total_time_s: float,
    h: Hierarchy,
    energy_config: NocEnergyConfig | None,
    switch_only: bool,
) -> RouteStats:
    """
    给定已汇总的 traffic 矩阵和 bw 矩阵，计算 util / energy / bottleneck，返回 RouteStats。

    util[i,j] = (traffic[i,j] / total_time_s) / bw[i,j]   （时间平均利用率）
    """
    # ── 利用率：时间平均带宽 / 峰值带宽 ─────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(bw > 0.0, total_traffic / total_time_s / bw, 0.0)

    # ── 瓶颈链路（total_traffic / bw 最大那条）──────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bw > 0.0, total_traffic / bw, 0.0)
    flat_idx = int(np.argmax(ratio))
    max_i, max_j = divmod(flat_idx, ratio.shape[1])

    # ── 能耗路 ────────────────────────────────────────────────────────────────
    effective_energy_cfg = energy_config or h.energy_config
    if switch_only:
        energy_per_bit = build_extended_energy_matrix_switch_only(h, effective_energy_cfg)
    else:
        energy_per_bit = build_extended_energy_matrix(h, effective_energy_cfg)

    noc_energy_pj = np.where(energy_per_bit > 0.0, total_traffic * 8.0 * energy_per_bit, 0.0)
    total_noc_energy_pj = float(np.sum(noc_energy_pj))

    # log.info(
    #     "_build_route_stats_core [%s]: total_time=%.3e s  "
    #     "bottleneck [%d->%d] %.3e B / %.3e B/s  total_noc_energy=%.3e pJ",
    #     "switch_only" if switch_only else "general",
    #     total_time_s,
    #     max_i, max_j,
    #     float(total_traffic[max_i, max_j]),
    #     float(bw[max_i, max_j]),
    #     total_noc_energy_pj,
    # )

    return RouteStats(
        max_link_src        = int(max_i),
        max_link_dst        = int(max_j),
        max_link_bytes      = float(total_traffic[max_i, max_j]),
        max_link_bw         = float(bw[max_i, max_j]),
        traffic             = total_traffic,
        bw                  = bw,
        util                = util,
        energy_per_bit      = energy_per_bit,
        noc_energy_pj       = noc_energy_pj,
        total_noc_energy_pj = total_noc_energy_pj,
    )


def build_route_stats_from_extended(
    traffic_list: list[np.ndarray],
    h: Hierarchy,
    total_time_s: float,
    energy_config: NocEnergyConfig | None = None,
) -> RouteStats:
    """
    从多个已展开的 extended traffic matrices 构建 RouteStats。

    调用方负责预先调用 build_extended_traffic_matrix[_switch_only] 生成各 collective
    的流量矩阵；本函数对其求和，再统一计算带宽利用率和能耗，不再重新展开逻辑 TM。

    traffic_list 为空时，traffic 视为全零（算子无 collective）。

    参数：
        traffic_list : 已展开的 extended traffic 矩阵列表（相同 shape），将被逐元素求和
        h            : Hierarchy，用于构建 bw_matrix 和 energy_matrix
        total_time_s : e2e 总时间窗口，用作利用率分母
        energy_config: 可选能耗配置（优先级高于 h.energy_config）
    """
    switch_only = all(t.kind == TopoKind.SWITCH for t in h.layers)

    if switch_only:
        bw = build_extended_bandwidth_matrix_switch_only(h)
    else:
        bw = build_extended_bandwidth_matrix(h)

    if traffic_list:
        total_traffic: np.ndarray = traffic_list[0].copy()
        for mat in traffic_list[1:]:
            total_traffic += mat
    else:
        total_traffic = np.zeros_like(bw)

    return _build_route_stats_core(total_traffic, bw, total_time_s, h, energy_config, switch_only)


def build_route_stats(
    tm: TrafficMatrix,
    h: Hierarchy,
    energy_config: NocEnergyConfig | None = None,
) -> RouteStats:
    """
    单 TrafficMatrix 场景：内部展开 TM，以 NoC stage latency（hop + 瓶颈链路传输时间）
    为 total_time_s 计算利用率。

    多 collective 场景请改用 build_route_stats_from_extended。
    """
    switch_only = all(t.kind == TopoKind.SWITCH for t in h.layers)

    if switch_only:
        bw                = build_extended_bandwidth_matrix_switch_only(h)
        traffic, hop_time = build_extended_traffic_matrix_switch_only(tm, h)
    else:
        bw                = build_extended_bandwidth_matrix(h)
        traffic, hop_time = build_extended_traffic_matrix(tm, h)

    # stage_latency = hop_time + 瓶颈链路传输时间 = hop_time + max(traffic/bw)
    with np.errstate(divide="ignore", invalid="ignore"):
        link_time = float(np.max(np.where(bw > 0.0, traffic / bw, 0.0)))
    stage_latency = float(hop_time) + link_time

    log.info(
        "build_route_stats [%s]: hop=%.3e s  link=%.3e s  stage=%.3e s",
        "switch_only" if switch_only else "general",
        float(hop_time), link_time, stage_latency,
    )

    return _build_route_stats_core(traffic, bw, stage_latency, h, energy_config, switch_only)


# ─────────────────────────────────────────────────────────────────────────────
