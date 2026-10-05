"""All-switch hierarchies must keep the links of different switch instances apart.

The topology is ``n_nodes`` switched nodes, each holding eight switched
devices. Bandwidths and latencies are illustrative.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.collectives.all_reduce_wrapper import all_reduce_ring
from mosaic.noc.energy_config import NocEnergyConfig
from mosaic.noc.noc_topo import (
    Hierarchy,
    PortSpread,
    build_extended_bandwidth_matrix_switch_only,
    build_extended_traffic_matrix_switch_only,
    get_extend_max_routes_with_traffic,
    make_switch,
)
from mosaic.noc.route_stats import build_route_stats
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.parallelism import ParallelScheme
from mosaic.utils import Modeling_Granularity


GPUS_PER_NODE = 8
L1_BW, L1_LAT = 100e9, 1e-6
L2_BW, L2_LAT = 25e9, 5e-6
ENERGY = NocEnergyConfig(l1_pj_per_bit=1.0, l2_pj_per_bit=2.0, l3_pj_per_bit=3.0)


def _nodes(n_nodes: int) -> Hierarchy:
    return Hierarchy(
        layers=[
            make_switch(1, hop_latency=0.0, link_bandwidth=L2_BW),
            make_switch(n_nodes, hop_latency=L2_LAT, link_bandwidth=L2_BW),
            make_switch(GPUS_PER_NODE, hop_latency=L1_LAT, link_bandwidth=L1_BW),
        ],
        port_spread=PortSpread.EVEN,
        name=f"switch_{n_nodes}x{GPUS_PER_NODE}",
        energy_config=ENERGY,
    )


def _intra_node_ring_step(n_nodes: int, bytes_per_flow: int) -> TrafficMatrix:
    """Every device sends to its right neighbour inside its own node."""

    tm = TrafficMatrix(n_nodes * GPUS_PER_NODE)
    for node in range(n_nodes):
        for gpu in range(GPUS_PER_NODE):
            src = node * GPUS_PER_NODE + gpu
            dst = node * GPUS_PER_NODE + (gpu + 1) % GPUS_PER_NODE
            tm.add(src, dst, bytes_per_flow)
    return tm


@pytest.mark.parametrize("n_nodes", [1, 2, 4])
def test_disjoint_nodes_do_not_share_switch_links(n_nodes: int) -> None:
    flow = 1 << 20
    hop, link, _, _ = get_extend_max_routes_with_traffic(
        _intra_node_ring_step(n_nodes, flow), _nodes(n_nodes)
    )
    # Each node's ring uses only its own switch: one flow per port link.
    assert link == pytest.approx(flow / L1_BW)
    assert hop == pytest.approx(2 * L1_LAT)


@pytest.mark.parametrize("n_nodes", [1, 2, 4])
def test_tp_ring_all_reduce_does_not_grow_with_parallel_groups(n_nodes: int) -> None:
    message = 64 << 20
    latency, link, _ = all_reduce_ring(
        None,
        ParallelScheme(tp=GPUS_PER_NODE, dp=n_nodes),
        _nodes(n_nodes),
        Modeling_Granularity("coarse", True, False, True),
        "tp",
        message,
    )
    steps = 2 * (GPUS_PER_NODE - 1)
    assert link == pytest.approx(steps * (message / GPUS_PER_NODE) / L1_BW)
    assert latency == pytest.approx(steps * 2 * L1_LAT)


def test_a_node_uplink_still_carries_all_of_its_flows() -> None:
    """Flows that physically share a link must still add up on it."""

    n_nodes, flow = 4, 1 << 20
    tm = TrafficMatrix(n_nodes * GPUS_PER_NODE)
    for node in range(n_nodes):
        for gpu in range(GPUS_PER_NODE):
            src = node * GPUS_PER_NODE + gpu
            dst = ((node + 1) % n_nodes) * GPUS_PER_NODE + gpu
            tm.add(src, dst, flow)
    hop, link, _, _ = get_extend_max_routes_with_traffic(tm, _nodes(n_nodes))
    # All eight devices of a node leave through the node's single L2 port.
    assert link == pytest.approx(GPUS_PER_NODE * flow / L2_BW)
    assert hop == pytest.approx(2 * L1_LAT + 2 * L2_LAT)


@pytest.mark.parametrize("n_nodes", [1, 2, 4])
def test_route_stats_cover_every_link_with_bandwidth_and_energy(n_nodes: int) -> None:
    flow = 1 << 20
    tm = _intra_node_ring_step(n_nodes, flow)
    hierarchy = _nodes(n_nodes)

    traffic, _ = build_extended_traffic_matrix_switch_only(tm, hierarchy)
    bandwidth = build_extended_bandwidth_matrix_switch_only(hierarchy)
    assert np.all(bandwidth[traffic > 0] > 0)

    stats = build_route_stats(tm, hierarchy)
    flows = n_nodes * GPUS_PER_NODE
    # Each flow crosses two L1 links (port to switch, switch to port).
    assert stats.total_noc_energy_pj == pytest.approx(flows * flow * 8 * 2 * ENERGY.l1_pj_per_bit)
    assert stats.max_link_bytes == pytest.approx(flow)
