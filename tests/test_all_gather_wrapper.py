"""all_gather_wrapper and its algorithms on illustrative topologies."""

from __future__ import annotations

import pytest

from mosaic.collectives import all_gather_wrapper
from mosaic.collectives.all_gather_wrapper import (
    all_gather_all_to_all,
    all_gather_recursive_doubling,
    all_gather_ring,
)
from mosaic.collectives.all_reduce_wrapper import all_reduce_ring
from mosaic.noc.noc_topo import Hierarchy, PortSpread, TopoKind, make_mesh_or_torus, make_ring, make_switch
from mosaic.parallelism import ParallelScheme
from mosaic.utils import Modeling_Granularity


BW, LAT = 450e9, 1e-6
MESSAGE = 64 << 20
GRANULARITY = Modeling_Granularity("coarse", True, False)


def _switch(ports: int) -> Hierarchy:
    return Hierarchy(
        layers=[
            make_switch(1, hop_latency=0.0, link_bandwidth=BW),
            make_switch(1, hop_latency=0.0, link_bandwidth=BW),
            make_switch(ports, hop_latency=LAT, link_bandwidth=BW),
        ],
        port_spread=PortSpread.EVEN,
        name=f"switch{ports}",
    )


def _nodes(n_nodes: int) -> Hierarchy:
    return Hierarchy(
        layers=[
            make_switch(1, hop_latency=0.0, link_bandwidth=50e9),
            make_switch(n_nodes, hop_latency=5e-6, link_bandwidth=50e9),
            make_switch(8, hop_latency=LAT, link_bandwidth=BW),
        ],
        port_spread=PortSpread.EVEN,
        name=f"switch{n_nodes}x8",
    )


def _ring_of_nodes() -> Hierarchy:
    return Hierarchy(
        layers=[
            make_switch(1, hop_latency=0.0, link_bandwidth=50e9),
            make_ring(4, hop_latency=5e-6, link_bandwidth=50e9),
            make_switch(8, hop_latency=LAT, link_bandwidth=BW),
        ],
        port_spread=PortSpread.EVEN,
        name="ring4x8",
    )


def _torus() -> Hierarchy:
    return Hierarchy(
        layers=[
            make_switch(1, hop_latency=0.0, link_bandwidth=BW),
            make_switch(1, hop_latency=0.0, link_bandwidth=BW),
            make_mesh_or_torus(4, 4, TopoKind.TORUS2D, hop_latency=LAT, link_bandwidth=200e9),
        ],
        port_spread=PortSpread.EVEN,
        name="torus4x4",
    )


def test_algorithms_on_one_switch_match_closed_forms() -> None:
    parallel, topology = ParallelScheme(tp=8), _switch(8)
    shard = MESSAGE / 8

    ring = all_gather_ring(None, parallel, topology, GRANULARITY, "tp", MESSAGE)
    assert ring[1] == pytest.approx(7 * shard / BW)
    assert ring[0] == pytest.approx(7 * 2 * LAT)

    doubling = all_gather_recursive_doubling(None, parallel, topology, GRANULARITY, "tp", MESSAGE)
    assert doubling[1] == pytest.approx((shard + 2 * shard + 4 * shard) / BW)
    assert doubling[0] == pytest.approx(3 * 2 * LAT)

    direct = all_gather_all_to_all(None, parallel, topology, GRANULARITY, "tp", MESSAGE)
    assert direct[1] == pytest.approx(7 * shard / BW)
    assert direct[0] == pytest.approx(2 * LAT)

    # On a switch the direct exchange has the fewest hops for the same bytes.
    assert all_gather_wrapper(None, parallel, topology, GRANULARITY, "tp", MESSAGE)[:2] == pytest.approx(direct[:2])


@pytest.mark.parametrize(
    ("parallel", "topology"),
    [
        (ParallelScheme(tp=8), _switch(8)),
        (ParallelScheme(tp=8, dp=4), _nodes(4)),
        (ParallelScheme(tp=8, dp=4), _ring_of_nodes()),
        (ParallelScheme(tp=4, dp=4), _torus()),
    ],
)
def test_ring_all_reduce_is_two_ring_all_gathers(parallel: ParallelScheme, topology: Hierarchy) -> None:
    gather = all_gather_ring(None, parallel, topology, GRANULARITY, "tp", MESSAGE)
    reduce = all_reduce_ring(None, parallel, topology, GRANULARITY, "tp", MESSAGE)
    assert reduce[0] == pytest.approx(2 * gather[0])
    assert reduce[1] == pytest.approx(2 * gather[1])


@pytest.mark.parametrize("n_nodes", [1, 2, 4])
def test_intra_node_all_gather_does_not_grow_with_node_count(n_nodes: int) -> None:
    single = all_gather_wrapper(None, ParallelScheme(tp=8), _nodes(1), GRANULARITY, "tp", MESSAGE)
    many = all_gather_wrapper(None, ParallelScheme(tp=8, dp=n_nodes), _nodes(n_nodes), GRANULARITY, "tp", MESSAGE)
    assert many[:2] == pytest.approx(single[:2])


def test_cross_node_gather_is_bound_by_the_node_uplinks() -> None:
    # Every node's eight devices gather across four nodes through one uplink each.
    hop, link, traffic = all_gather_wrapper(None, ParallelScheme(tp=8, dp=4), _nodes(4), GRANULARITY, "dp", MESSAGE)
    assert traffic is not None and link > 0 and hop > 0
    ring = all_gather_ring(None, ParallelScheme(tp=8, dp=4), _nodes(4), GRANULARITY, "dp", MESSAGE)
    assert ring[1] == pytest.approx(3 * 8 * (MESSAGE / 4) / 50e9)
    assert hop + link <= ring[0] + ring[1]


def test_non_power_of_two_groups_skip_recursive_doubling() -> None:
    parallel, topology = ParallelScheme(tp=6), _switch(6)
    best = all_gather_wrapper(None, parallel, topology, GRANULARITY, "tp", MESSAGE)
    candidates = [
        all_gather_ring(None, parallel, topology, GRANULARITY, "tp", MESSAGE),
        all_gather_all_to_all(None, parallel, topology, GRANULARITY, "tp", MESSAGE),
    ]
    assert best[0] + best[1] == pytest.approx(min(c[0] + c[1] for c in candidates))
    with pytest.raises(AssertionError):
        all_gather_recursive_doubling(None, parallel, topology, GRANULARITY, "tp", MESSAGE)


def test_a_group_of_one_costs_nothing() -> None:
    assert all_gather_wrapper(None, ParallelScheme(tp=1, dp=8), _switch(8), GRANULARITY, "tp", MESSAGE) == (0, 0, None)


def test_unknown_dimension_is_rejected() -> None:
    with pytest.raises(ValueError):
        all_gather_wrapper(None, ParallelScheme(tp=8), _switch(8), GRANULARITY, "xp", MESSAGE)
