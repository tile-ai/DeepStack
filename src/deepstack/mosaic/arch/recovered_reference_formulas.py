"""Source-level reconstruction of formulas recovered from DeepStack's provider.

This is an audit artifact, not a replacement for the licensed provider.  The
functions below reproduce the scalar transformations observed through the
public facade; the original profile table and NoC case data remain separate.
"""
from __future__ import annotations

import math
import json
from pathlib import Path


DDR_PEAK_PER_ACTIVE_LAYER = 409.6e9
DDR_UNCACHED_UTIL = 0.8205128205128205
LITTLE_LATENCY_S = 159.0 / 700_000_000.0
POWER_LIMIT_W = 100.0


def reference_profile(profile_name: str) -> dict:
    """Return recovered public scalar state, not a Cython ReferenceArch object."""
    table = json.loads(Path(__file__).with_name("recovered_reference_profiles.json").read_text(encoding="utf-8"))
    try:
        return dict(table[profile_name]["fields"])
    except KeyError:
        raise ValueError("unknown reference architecture profile") from None


def q09_thermal_scale(total_layers: int) -> float:
    """Recovered q09: cubic frequency scale for total DRAM layers."""
    total_layers = int(total_layers)
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    # Preserve the original floating-point evaluation order. The simplified
    # rational expression differs near L=544 (the provider returns ~2.7e-6).
    d = 0.56 + 0.01 * total_layers
    x = (100.0 * (0.56 + 0.01 * 4.0) / d - 10.0) / 90.0
    return math.pow(x, 1.0 / 3.0) if x >= 0.0 else math.nan


def q11_power_wall(chip_power_w: float, noc_power_w: float) -> tuple[bool, float]:
    """Recovered q11 power wall and cubic frequency scale."""
    total = chip_power_w + noc_power_w
    if not total > POWER_LIMIT_W:
        return False, 1.0
    return True, (POWER_LIMIT_W / total) ** (1.0 / 3.0)


def row_reduce_time(a: float, b: float, c: float) -> float:
    """Recovered row-reduction time in seconds."""
    return (10.0 * a * b + 10.0 * c + 5.0) / 7.0e9


def little_required_buffer(ddr_bandwidth: float, sm_count: float) -> float:
    return (ddr_bandwidth / sm_count) * LITTLE_LATENCY_S


def little_law(ddr_bandwidth: float, sm_count: float, smem_capacity: float) -> tuple[bool, float, float]:
    """Return (limited, required_buffer, effective_ddr_bandwidth)."""
    required = little_required_buffer(ddr_bandwidth, sm_count)
    if required <= smem_capacity:
        return False, required, ddr_bandwidth
    return True, required, ddr_bandwidth * smem_capacity / required


def with_littles_law_result(ddr_bandwidth: float, sm_count: float, smem_capacity: float) -> tuple[bool, float, float]:
    """Clone helper returns the post-cap check, rather than the original check."""
    _, _, effective_bandwidth = little_law(ddr_bandwidth, sm_count, smem_capacity)
    limited, required, _ = little_law(effective_bandwidth, sm_count, smem_capacity)
    return limited, required, effective_bandwidth


def dram_layer_rates(total_layers: float, active_layers: float | None = None) -> dict[str, float]:
    """Recovered q05/q07 throughput fields (capacity is q07-only)."""
    total = int(total_layers)
    active = total if active_layers is None else int(active_layers)
    if total <= 0:
        raise ValueError("total_layers must be positive")
    if not 1 <= active <= total:
        raise ValueError("active_layers must be in [1, total_layers]")
    peak = DDR_PEAK_PER_ACTIVE_LAYER * active
    return {
        "dram_active_layers": active,
        "dram_layers_per_cluster": total,
        "ddr_peak_bandwidth": peak,
        "l2_bandwidth": peak,
        "ddr_bandwidth": peak * (1.0 - max(0.0, 2.0 * active / total - 1.0) * (1.0 - DDR_UNCACHED_UTIL)),
    }


def cluster_scale(cluster_width: float) -> dict[str, float]:
    """Recovered q06 geometry and scale factors."""
    width = int(cluster_width)
    if width <= 0:
        raise ValueError("sm_count must be positive")
    return {
        "sm_count": width,
        "ddr_stack": 2 * width,
        "ddr_capacity": float(1 << 30) * width,
        "ddr_wave_bytes": 4096 * width,
        "l2_capacity": 786432 * width,
        "aggregate_scale": width / 8.0,
    }


def q04(profile_name: str, width_bytes: float) -> tuple[int, int, int]:
    if not isinstance(profile_name, str):
        raise TypeError("profile_name must be a string")
    try:
        m = 16 if profile_name == "large_matrix" else 8
        return {0.5: (m, 8, 64), 1.0: (m, 8, 32),
                2.0: (m, 8, 16), 4.0: (m, 8, 8)}[float(width_bytes)]
    except KeyError:
        raise ValueError("unsupported tensor input width") from None


_SWITCH_CASES = {
    "s0": (128e9, 0.0, math.inf, 128e-9, 128e9, 1024e9),
    "s1": (256e9, 512e-9, 32e9, 128e-9, 256e9, 2048e9),
    "s2": (512e9, 0.0, math.inf, 128e-9, 512e9, 4096e9),
    "s3": (512e9, 512e-9, 64e9, 128e-9, 512e9, 4096e9),
    "s4": (512e9, 512e-9, 128e9, 256e-9, 512e9, 4096e9),
    "s5": (32e9, 0.0, math.inf, 32e-9, 32e9, 256e9),
}
_TORUS_CASES = {"t0": (1280e9, 128e-9, 128e9),
                "t1": (5120e9, 128e-9, 512e9),
                "t2": (320e9, 32e-9, 32e9)}


def _set_rates(layer, latency, bandwidth, center=None):
    layer.hop_latency = latency
    layer.link_bandwidth = bandwidth
    if center is not None:
        layer.switch_center_in_bw = center
        layer.switch_center_out_bw = center


def q12(case_id, l3, l2, l1):
    """Mutate duck-typed topology fields; no provider import is required."""
    try:
        b3, t2, b2, t1, b1, c1 = _SWITCH_CASES[case_id]
    except KeyError:
        raise ValueError("unknown switch NoC case") from None
    _set_rates(l3, 0.0, b3, b3)
    _set_rates(l2, t2, b2, b2)
    _set_rates(l1, t1, b1, b1 * l1.shape[0] * l1.shape[1])


def q13(case_id, l3, l2, l1):
    try:
        bw, latency, link = _TORUS_CASES[case_id]
    except KeyError:
        raise ValueError("unknown torus NoC case") from None
    _set_rates(l3, 0.0, bw, bw)
    _set_rates(l2, 0.0, bw, bw)
    _set_rates(l1, latency, link)

