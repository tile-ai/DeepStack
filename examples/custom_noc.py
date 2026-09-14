"""Build and inspect illustrative switch, ring, mesh and torus hierarchies."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/deepstack"), str(ROOT / "src/tilesight")]

from mosaic.noc.energy_config import NocEnergyConfig
from mosaic.noc.noc_config_set import h200x32
from mosaic.noc.noc_topo import (
    Hierarchy,
    PortSpread,
    TopoKind,
    make_mesh_or_torus,
    make_ring,
    make_switch,
)


def ring_switch_32(local_bw=100e9, ring_bw=50e9):
    """Four groups on a ring, with eight switched devices per group."""
    # Bandwidth is bytes/s; hop latency is seconds. Values are illustrative.
    L1 = make_switch(
        8, hop_latency=200e-9, link_bandwidth=local_bw,
        switch_center_in_bw=local_bw * 8, switch_center_out_bw=local_bw * 8,
    )
    L2 = make_ring(4, hop_latency=1e-6, link_bandwidth=ring_bw)
    # A singleton outer level preserves the three-level convention.
    L3 = make_switch(1, hop_latency=0, link_bandwidth=ring_bw)
    return Hierarchy(
        layers=[L3, L2, L1], port_spread=PortSpread.EVEN,
        node_mapper=None, name="ring_switch_32",
        # Example energy coefficients in pJ/bit; replace for your design.
        energy_config=NocEnergyConfig(1.0, 2.0, 3.0),
    )


def torus_mesh_switch_64(local_bw=100e9, mesh_bw=50e9, torus_bw=25e9):
    """A 2x2 torus of 2x2 meshes, each mesh position holding four devices."""
    L1 = make_switch(
        4, hop_latency=200e-9, link_bandwidth=local_bw,
        switch_center_in_bw=local_bw * 4, switch_center_out_bw=local_bw * 4,
    )
    L2 = make_mesh_or_torus(
        2, 2, TopoKind.MESH2D, hop_latency=1e-6, link_bandwidth=mesh_bw,
    )
    L3 = make_mesh_or_torus(
        2, 2, TopoKind.TORUS2D, hop_latency=2e-6, link_bandwidth=torus_bw,
    )
    return Hierarchy(
        layers=[L3, L2, L1], port_spread=PortSpread.EVEN,
        node_mapper=None, name="torus_mesh_switch_64",
        energy_config=NocEnergyConfig(1.0, 2.0, 3.0),
    )


def main():
    for factory in (h200x32, ring_switch_32, torus_mesh_switch_64):
        topology = factory()
        print(f"{topology.name}: {topology.num_devices} devices")
        for name, layer in zip(("L3", "L2", "L1"), topology.layers):
            print(
                f"  {name}: {layer.kind.value:8s} {layer.shape}, "
                f"{layer.link_bandwidth / 1e9:g} GB/s, "
                f"{layer.hop_latency * 1e9:g} ns/hop"
            )
        print()


if __name__ == "__main__":
    main()
