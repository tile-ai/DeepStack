"""Source-only builders for caller-defined network-on-chip profiles."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from .energy_config import NocEnergyConfig
from .noc_topo import (
    Hierarchy,
    PortSpread,
    TopoKind,
    Topology,
    make_all2all,
    make_chain,
    make_mesh_or_torus,
    make_ring,
    make_switch,
)


__all__ = [
    "Hierarchy",
    "LAYER_NAMES",
    "NocEnergyConfig",
    "PortSpread",
    "TopoKind",
    "Topology",
    "make_all2all",
    "make_chain",
    "make_custom_profile",
    "make_mesh_or_torus",
    "make_ring",
    "make_switch",
    "make_topology",
]


LAYER_NAMES = ("L3", "L2", "L1")
"""Canonical outer-to-inner layer order used by three-level profiles."""

_MESH_KINDS = {
    TopoKind.MESH2D,
    TopoKind.TORUS2D,
    TopoKind.MCHAIN_NRING,
    TopoKind.MRING_NCHAIN,
}
_REQUIRED_TOPOLOGY_FIELDS = {
    "kind",
    "shape",
    "hop_latency_ns",
    "link_bandwidth_gbytes_per_s",
}
_OPTIONAL_TOPOLOGY_FIELDS = {
    "switch_center_in_gbytes_per_s",
    "switch_center_out_gbytes_per_s",
}


def _number(
    value: object,
    *,
    field: str,
    allow_zero: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if result < 0.0 or (result == 0.0 and not allow_zero):
        relation = (
            "greater than or equal to zero" if allow_zero else "greater than zero"
        )
        raise ValueError(f"{field} must be {relation}")
    return result


def _shape(value: object) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise TypeError("shape must be a two-element integer sequence")
    try:
        dimensions = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("shape must be a two-element integer sequence") from exc
    if len(dimensions) != 2:
        raise ValueError("shape must contain exactly two dimensions")
    if any(
        isinstance(item, bool) or not isinstance(item, Integral)
        for item in dimensions
    ):
        raise TypeError("shape dimensions must be integers")
    shape = (int(dimensions[0]), int(dimensions[1]))
    if min(shape) <= 0:
        raise ValueError("shape dimensions must be greater than zero")
    return shape


def _topology_kind(value: object) -> TopoKind:
    if isinstance(value, TopoKind):
        return value
    if not isinstance(value, str):
        raise TypeError("kind must be a TopoKind or its string value")
    try:
        return TopoKind(value)
    except ValueError as exc:
        choices = ", ".join(kind.value for kind in TopoKind)
        raise ValueError(
            f"unknown topology kind {value!r}; choose one of: {choices}"
        ) from exc


def make_topology(
    kind: TopoKind | str,
    shape: Sequence[int],
    *,
    hop_latency_ns: float,
    link_bandwidth_gbytes_per_s: float,
    switch_center_in_gbytes_per_s: float | None = None,
    switch_center_out_gbytes_per_s: float | None = None,
) -> Topology:
    """Construct one NoC level from explicit, human-readable units.

    ``hop_latency_ns`` is nanoseconds per hop.  Bandwidth fields use decimal
    gigabytes per second (1 GB/s = 1e9 bytes/s).  The topology-specific port
    layout and routing semantics come from the existing ``noc_topo`` factories.
    """

    topology_kind = _topology_kind(kind)
    rows, columns = _shape(shape)
    latency_s = _number(
        hop_latency_ns,
        field="hop_latency_ns",
        allow_zero=True,
    ) * 1.0e-9
    bandwidth_bytes_per_s = _number(
        link_bandwidth_gbytes_per_s,
        field="link_bandwidth_gbytes_per_s",
        allow_zero=False,
    ) * 1.0e9

    center_values = (
        switch_center_in_gbytes_per_s,
        switch_center_out_gbytes_per_s,
    )
    if topology_kind != TopoKind.SWITCH and any(
        value is not None for value in center_values
    ):
        raise ValueError("switch-center bandwidth is valid only for kind='switch'")
    if (center_values[0] is None) != (center_values[1] is None):
        raise ValueError(
            "switch-center input and output bandwidth must be supplied together"
        )

    if topology_kind == TopoKind.SWITCH:
        if rows != 1:
            raise ValueError("switch shape must be (1, N)")
        center_in = (
            None
            if center_values[0] is None
            else _number(
                center_values[0],
                field="switch_center_in_gbytes_per_s",
                allow_zero=False,
            )
            * 1.0e9
        )
        center_out = (
            None
            if center_values[1] is None
            else _number(
                center_values[1],
                field="switch_center_out_gbytes_per_s",
                allow_zero=False,
            )
            * 1.0e9
        )
        topology = make_switch(
            columns,
            hop_latency=latency_s,
            link_bandwidth=bandwidth_bytes_per_s,
            switch_center_in_bw=center_in,
            switch_center_out_bw=center_out,
        )
    elif topology_kind in {TopoKind.RING, TopoKind.CHAIN}:
        if rows != 1:
            raise ValueError(f"{topology_kind.value} shape must be (1, N)")
        if columns % 2:
            raise ValueError(f"{topology_kind.value} size must be even")
        factory = make_ring if topology_kind == TopoKind.RING else make_chain
        topology = factory(
            columns,
            hop_latency=latency_s,
            link_bandwidth=bandwidth_bytes_per_s,
        )
    elif topology_kind == TopoKind.ALL2ALL:
        if rows * columns < 4:
            raise ValueError("all_to_all requires at least four nodes")
        topology = make_all2all(
            rows,
            columns,
            hop_latency=latency_s,
            link_bandwidth=bandwidth_bytes_per_s,
        )
    elif topology_kind in _MESH_KINDS:
        if any(dimension != 1 and dimension % 2 for dimension in (rows, columns)):
            raise ValueError(
                f"{topology_kind.value} dimensions must be 1 or a positive even integer"
            )
        topology = make_mesh_or_torus(
            rows,
            columns,
            topology_kind,
            hop_latency=latency_s,
            link_bandwidth=bandwidth_bytes_per_s,
        )
    else:
        raise AssertionError(f"unhandled topology kind: {topology_kind}")

    topology._show_rates_in_repr = True
    return topology


def _topology_from_config(layer_name: str, config: Mapping[str, Any]) -> Topology:
    actual_fields = set(config)
    missing = sorted(_REQUIRED_TOPOLOGY_FIELDS - actual_fields)
    unexpected = sorted(
        actual_fields - _REQUIRED_TOPOLOGY_FIELDS - _OPTIONAL_TOPOLOGY_FIELDS
    )
    if missing or unexpected:
        raise ValueError(
            f"{layer_name} topology fields mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return make_topology(
        config["kind"],
        config["shape"],
        hop_latency_ns=config["hop_latency_ns"],
        link_bandwidth_gbytes_per_s=config["link_bandwidth_gbytes_per_s"],
        switch_center_in_gbytes_per_s=config.get(
            "switch_center_in_gbytes_per_s"
        ),
        switch_center_out_gbytes_per_s=config.get(
            "switch_center_out_gbytes_per_s"
        ),
    )


def _require_three_layers(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping keyed by L3, L2, and L1")
    keys = set(value)
    expected = set(LAYER_NAMES)
    if keys != expected:
        raise ValueError(
            f"{field} must contain exactly L3, L2, and L1; "
            f"missing={sorted(expected - keys)}, unexpected={sorted(keys - expected)}"
        )
    return value


def make_custom_profile(
    *,
    layers: Mapping[str, Topology | Mapping[str, Any]],
    name: str = "custom_noc_hierarchy",
    port_spread: PortSpread | str = PortSpread.EVEN,
    energy_config: NocEnergyConfig | None = None,
) -> Hierarchy:
    """Build a three-level hierarchy without consulting a reference profile.

    ``layers`` is keyed by ``L3``, ``L2``, and ``L1``.  Each value may be a
    :class:`Topology` returned by :func:`make_topology`, or a mapping containing
    that function's arguments.  The hierarchy always stores layers in the
    model's canonical outer-to-inner order.

    ``energy_config`` supplies caller-owned L1/L2/L3 energy-per-bit values.
    Omitting it leaves the hierarchy unconfigured; the paper path then uses
    the bundled release provider when energy accounting is requested.
    """

    layer_mapping = _require_three_layers(layers, field="layers")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if isinstance(port_spread, str):
        try:
            spread = PortSpread(port_spread)
        except ValueError as exc:
            choices = ", ".join(item.value for item in PortSpread)
            raise ValueError(
                f"unknown port_spread {port_spread!r}; choose one of: {choices}"
            ) from exc
    elif isinstance(port_spread, PortSpread):
        spread = port_spread
    else:
        raise TypeError("port_spread must be a PortSpread or its string value")
    if energy_config is not None and not isinstance(
        energy_config,
        NocEnergyConfig,
    ):
        raise TypeError("energy_config must be a NocEnergyConfig or None")

    topologies: list[Topology] = []
    for layer_name in LAYER_NAMES:
        value = layer_mapping[layer_name]
        if isinstance(value, Topology):
            topology = copy.copy(value)
            topology._show_rates_in_repr = True
        elif isinstance(value, Mapping):
            topology = _topology_from_config(layer_name, value)
        else:
            raise TypeError(
                f"layers[{layer_name!r}] must be a Topology or configuration mapping"
            )
        topologies.append(topology)

    return Hierarchy(
        layers=topologies,
        port_spread=spread,
        node_mapper=None,
        name=name,
        energy_config=energy_config,
    )
