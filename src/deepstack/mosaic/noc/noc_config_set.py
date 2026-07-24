"""Reference profiles plus compatibility exports for custom NoC builders.

Named reference profiles are binary-backed.  Fully caller-defined topologies
live in :mod:`mosaic.noc.custom_profile`, which can be imported without loading
the calibrated provider; their public names are re-exported here for backward
compatibility and convenience.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from . import _model_support as _provider
from .custom_profile import (
    Hierarchy,
    LAYER_NAMES,
    PortSpread,
    TopoKind,
    Topology,
    _number,
    _require_three_layers,
    make_all2all,
    make_chain,
    make_custom_profile,
    make_mesh_or_torus,
    make_ring,
    make_switch,
    make_topology,
)

_PROFILE_IDS = {
    "torus_mesh_switch_1": 0,
    "torus_mesh_switch_cluster8": 1,
    "torus_mesh_switch_cluster16": 2,
    "torus_mesh_switch_cluster32": 3,
    "torus_mesh_switch_2": 4,
    "torus_mesh_mesh_3": 5,
    "strong_torus_mesh_switch_4": 6,
    "weak_torus_mesh_switch_5": 7,
    "torus_mesh_switch_7": 8,
    "torus_mesh_switch_8": 9,
    "torus_mesh_switch_9": 10,
    "h200x8": 11,
    "h100x8": 12,
    "h100x32_strong": 13,
    "h100x32_medium": 14,
    "h100_8x1_8": 15,
    "h100_8x2_16": 16,
    "h100_8x4_32": 17,
    "h100_8x8_64": 18,
    "stacked_gpu_4x4": 19,
    "stacked_gpu_4x8": 20,
    "stacked_gpu_4x16": 21,
    "stacked_gpu_8x8": 22,
    "stacked_gpu_8x16": 23,
    "b200_8x1_8": 24,
    "h200x16": 25,
    "h200x32": 26,
    "A100x1": 27,
    "B6000x2": 28,
    "b200x4": 29,
    "b200x16": 30,
    "b200x32": 31,
    "h200x8_direct_no_nvswitch": 32,
    "h200x4_direct_no_nvswitch": 33,
}

_PROFILE_OUTPUT_NAMES = {
    "h100x8": "h200x8",
    "stacked_gpu_4x4": "stacked_gpu_4x4_64",
    "stacked_gpu_4x8": "stacked_gpu_4x8_128",
    "stacked_gpu_4x16": "stacked_gpu_4x16_256",
    "stacked_gpu_8x8": "stacked_gpu_8x8_256",
    "stacked_gpu_8x16": "stacked_gpu_8x16_512",
}

_GENERATED_PROFILE_IDS = {
    (32, False): 34,
    (64, False): 35,
    (128, False): 36,
    (256, False): 37,
    (512, False): 38,
    (1024, False): 39,
    (2048, False): 40,
    (4096, False): 41,
    (64, True): 42,
    (128, True): 43,
    (256, True): 44,
    (512, True): 45,
    (1024, True): 46,
    (2048, True): 47,
    (4096, True): 48,
}


def _fixed_profile(public_name: str) -> Hierarchy:
    hierarchy = _provider.p00(_PROFILE_IDS[public_name])
    hierarchy.name = _PROFILE_OUTPUT_NAMES.get(public_name, public_name)
    return hierarchy


def _install_fixed_profile(public_name: str) -> None:
    def factory() -> Hierarchy:
        return _fixed_profile(public_name)

    factory.__name__ = public_name
    factory.__qualname__ = public_name
    factory.__doc__ = f"Construct the {public_name} interconnect profile."
    globals()[public_name] = factory


for _profile_name in _PROFILE_IDS:
    _install_fixed_profile(_profile_name)


def h100x(num_nodes: int) -> Hierarchy:
    hierarchy = _provider.p01(0, int(num_nodes))
    hierarchy.name = "h200x8"
    return hierarchy


def mi325x(num_nodes: int) -> Hierarchy:
    nodes = int(num_nodes)
    hierarchy = _provider.p01(1, nodes)
    hierarchy.name = f"mi325x{nodes}"
    return hierarchy


def mi325x_all2all(num_nodes: int) -> Hierarchy:
    nodes = int(num_nodes)
    hierarchy = _provider.p01(2, nodes)
    hierarchy.name = f"mi325x{nodes}_all2all"
    return hierarchy


def mi325x4() -> Hierarchy:
    return mi325x(4)


def mi325x8() -> Hierarchy:
    return mi325x(8)


def mi325x4_all2all() -> Hierarchy:
    return mi325x_all2all(4)


def mi325x8_all2all() -> Hierarchy:
    return mi325x_all2all(8)


def make_generated_profile(
    num_nodes: int,
    variant: bool = False,
) -> Hierarchy:
    nodes = int(num_nodes)
    key = (nodes, bool(variant))
    try:
        profile_id = _GENERATED_PROFILE_IDS[key]
    except KeyError:
        if variant and nodes == 32:
            raise ValueError("the variant profile starts at 64 devices") from None
        raise ValueError(
            "num_nodes must be a supported power-of-two profile size"
        ) from None
    hierarchy = _provider.p00(profile_id)
    hierarchy.name = f"torus_mesh_switch_{nodes}"
    return hierarchy


def make_scaled_profile(
    profile_name: str,
    latency_scale: float = 1.0,
    l3_bandwidth_scale: float = 1.0,
    l2_bandwidth_scale: float = 1.0,
    l1_bandwidth_scale: float = 1.0,
    output_name: str | None = None,
) -> Hierarchy:
    """Construct a protected profile and apply dimensionless scale factors."""

    scales = (
        float(latency_scale),
        float(l3_bandwidth_scale),
        float(l2_bandwidth_scale),
        float(l1_bandwidth_scale),
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("profile scale factors must be finite and positive")
    if profile_name not in {
        "torus_mesh_switch_1",
        "torus_mesh_mesh_3",
    }:
        raise ValueError(f"unsupported calibrated profile: {profile_name}")

    hierarchy = _fixed_profile(profile_name)
    latency = scales[0]
    for layer, bandwidth in zip(hierarchy.layers, scales[1:]):
        layer.hop_latency *= latency
        layer.link_bandwidth *= bandwidth
        if layer.switch_center_in_bw is not None:
            layer.switch_center_in_bw *= bandwidth
        if layer.switch_center_out_bw is not None:
            layer.switch_center_out_bw *= bandwidth
    if output_name is not None:
        hierarchy.name = str(output_name)
    return hierarchy


def model_support_api_version() -> int:
    return int(_provider.p98())


def energy_support_api_version() -> int:
    return int(_provider.p97())


def make_profile_with_latencies(
    profile_name: str,
    *,
    layer_latencies_ns: Mapping[str, float],
    output_name: str | None = None,
) -> Hierarchy:
    """Copy a named profile and replace all per-hop latencies in nanoseconds.

    Requiring all three values prevents a missing entry from silently retaining
    a calibrated latency.  The supplied mapping is never modified.
    """

    latency_mapping = _require_three_layers(
        layer_latencies_ns,
        field="layer_latencies_ns",
    )
    latencies_s = [
        _number(
            latency_mapping[layer_name],
            field=f"layer_latencies_ns[{layer_name!r}]",
            allow_zero=True,
        )
        * 1.0e-9
        for layer_name in LAYER_NAMES
    ]
    hierarchy = make_scaled_profile(
        profile_name,
        output_name=output_name,
    )
    if len(hierarchy.layers) != len(LAYER_NAMES):
        raise ValueError(
            f"profile {profile_name!r} does not contain exactly three layers"
        )
    for topology, hop_latency_s in zip(hierarchy.layers, latencies_s):
        topology.hop_latency = hop_latency_s
    return hierarchy


__all__ = [
    *_PROFILE_IDS,
    "Hierarchy",
    "LAYER_NAMES",
    "PortSpread",
    "TopoKind",
    "Topology",
    "energy_support_api_version",
    "h100x",
    "make_all2all",
    "make_chain",
    "make_custom_profile",
    "make_generated_profile",
    "make_mesh_or_torus",
    "make_profile_with_latencies",
    "make_ring",
    "make_scaled_profile",
    "make_switch",
    "make_topology",
    "mi325x",
    "mi325x4",
    "mi325x4_all2all",
    "mi325x8",
    "mi325x8_all2all",
    "mi325x_all2all",
    "model_support_api_version",
]
