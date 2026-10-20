"""Independent numerical reconstruction of DeepStack's model-support provider.

Uses only Python's standard library and the adjacent recovered NoC table.
Topology/Hierarchy are value objects, not drop-in routing-class replacements.
All bandwidths are bytes/s, latencies seconds, energy pJ.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Topology:
    kind: str
    shape: tuple[int, int]
    hop_latency: float
    link_bandwidth: float
    switch_center_in_bw: float | None = None
    switch_center_out_bw: float | None = None


@dataclass
class Hierarchy:
    layers: list[Topology]
    port_spread: str = "even"
    name: str | None = None

    @property
    def num_devices(self):
        result = 1
        for layer in self.layers:
            result *= layer.shape[0] * layer.shape[1]
        return result


def p00(profile_id):
    profile_id = int(profile_id)
    table = json.loads(Path(__file__).with_name("noc_profile_id_table.json").read_text(encoding="utf-8-sig"))
    if not 0 <= profile_id < len(table) or "error" in table[profile_id]:
        raise ValueError("E8")
    row = table[profile_id]
    layers = []
    for tag in ("L3", "L2", "L1"):
        layers.append(Topology(
            row[f"{tag}_kind"], tuple(map(int, row[f"{tag}_shape"].split("x"))),
            row[f"{tag}_hop_latency_s"], row[f"{tag}_link_bandwidth_Bps"],
            row[f"{tag}_switch_center_in_Bps"], row[f"{tag}_switch_center_out_Bps"],
        ))
    return Hierarchy(layers, row["port_spread"])


def p01(variant, num_nodes):
    variant, n = int(variant), int(num_nodes)
    if variant not in (0, 1, 2):
        raise ValueError("E8")
    if variant == 2 and n < 4:
        raise AssertionError("ALL2ALL requires at least 4 nodes")
    outer_bw = 3.70e12 if variant == 0 else 4.90e11
    inner_bw = 3.70e11 if variant == 0 else 4.90e10
    outer_center = 2.96e13 if variant == 0 else (n or 1) * 4.90e11
    outer = lambda: Topology("switch", (1, 1), 0.0, outer_bw,
                             outer_center, outer_center)
    inner = Topology("all_to_all" if variant == 2 else "switch", (1, n),
                     4.85e-6 if variant == 2 else 1.25e-6, inner_bw)
    if variant != 2:
        inner.switch_center_in_bw = (n or 1) * inner_bw
        inner.switch_center_out_bw = (n or 1) * inner_bw
    return Hierarchy([outer(), outer(), inner])


def p93():
    return 10.0


def p94(unit, operations):
    unit = int(unit)
    if unit not in (0, 1, 2):
        raise ValueError("E5")
    return (10.0, 0.7942, 0.722)[unit] * float(operations)


def p95(level, read_bytes, write_bytes):
    level = int(level)
    if level not in (0, 1, 2, 3):
        raise ValueError("E4")
    read, write = ((1.6, 2.0), (3.2, 4.8), (8.0, 12.0), (31.2, 31.2))[level]
    return read * float(read_bytes) + write * float(write_bytes)


def p96(layer_index, layer_count):
    i, n = int(layer_index), int(layer_count)
    if n < 1:
        raise ValueError("E2")
    if not 0 <= i < n:
        raise ValueError("E3")
    return 2.0 if i == n - 1 else 3.0 if i == n - 2 else 8.0


def p97():
    return 1


def p98():
    return 1
