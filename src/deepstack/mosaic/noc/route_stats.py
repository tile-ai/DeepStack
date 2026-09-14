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
    """Snapshot of link resources after routing a logical TrafficMatrix through a Hierarchy.

    Contains three parallel matrices of shape [ext_size x ext_size], with each
    (i,j) corresponding to a physical directed link:
      - Bandwidth path: traffic (bytes) / total_time_s / bw -> util (0..1)
      - Energy path: traffic (bytes) x 8 x energy_per_bit (pJ/bit) -> noc_energy_pj (pJ)

    Supports both topology families:
      - Multilayer SWITCH-only topologies (build_extended_*_switch_only)
      - Mixed MESH / TORUS / RING / CHAIN topologies (build_extended_*)
    """

    max_link_src: int
    max_link_dst: int
    max_link_bytes: float
    max_link_bw: float

    # ── Extended matrices [ext_size × ext_size] ──────────────────────────────
    traffic: np.ndarray = field(repr=False)
    bw:      np.ndarray = field(repr=False)
    util:    np.ndarray = field(repr=False)

    # ── Energy ───────────────────────────────────────────────────────────────
    energy_per_bit:      np.ndarray = field(repr=False)  # pJ/bit per link
    noc_energy_pj:       np.ndarray = field(repr=False)  # pJ per link = traffic * 8 * energy_per_bit
    total_noc_energy_pj: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _build_route_stats_core(
    total_traffic: np.ndarray,
    bw: np.ndarray,
    total_time_s: float,
    h: Hierarchy,
    energy_config: NocEnergyConfig | None,
    switch_only: bool,
) -> RouteStats:
    """Compute utilization, energy, and the bottleneck from aggregated traffic and
    bandwidth matrices, and return RouteStats.

    util[i,j] = (traffic[i,j] / total_time_s) / bw[i,j]  (time-averaged utilization)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(bw > 0.0, total_traffic / total_time_s / bw, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bw > 0.0, total_traffic / bw, 0.0)
    flat_idx = int(np.argmax(ratio))
    max_i, max_j = divmod(flat_idx, ratio.shape[1])

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
    """Build RouteStats from multiple expanded extended traffic matrices.

    The caller must first generate each collective's traffic matrix using
    build_extended_traffic_matrix[_switch_only]. This function sums those matrices
    and computes bandwidth utilization and energy without re-expanding logical TMs.

    An empty traffic_list is treated as zero traffic (an operator with no collectives).

    Args:
        traffic_list: Expanded extended traffic matrices with identical shapes,
            summed elementwise.
        h: Hierarchy used to build bw_matrix and energy_matrix.
        total_time_s: Total end-to-end time window used as the utilization denominator.
        energy_config: Optional energy configuration, taking precedence over h.energy_config.
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
    """Handle a single TrafficMatrix: expand the TM internally and compute utilization
    using the NoC stage latency (hop latency + bottleneck link transfer time) as total_time_s.

    For multiple collectives, use build_route_stats_from_extended.
    """
    switch_only = all(t.kind == TopoKind.SWITCH for t in h.layers)

    if switch_only:
        bw                = build_extended_bandwidth_matrix_switch_only(h)
        traffic, hop_time = build_extended_traffic_matrix_switch_only(tm, h)
    else:
        bw                = build_extended_bandwidth_matrix(h)
        traffic, hop_time = build_extended_traffic_matrix(tm, h)

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
