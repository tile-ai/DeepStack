"""
noc_bw_config.py — NoC per-layer BW sensitivity DSE configuration & utilities

Provides:
  - generate_noc_bw_configs()     → all (baseline, layer, bw_mult) combos
  - make_noc_for_config()         → Hierarchy with one layer's BW scaled
  - make_arch_for_noc_config()    → stacked_gpu_wgmma + feasible SM capacity
  - find_max_sm_count()           → sealed reference-capacity decision
  - compute_power_wall()          → power wall detection & freq scaling
"""

from mosaic.arch.stacked_gpu_wgmma import stacked_gpu_wgmma
from mosaic.cost.capacity import max_sm_count as reference_max_sm_count
from mosaic.noc.noc_config_set import make_scaled_profile

TDP_W = 100.0

BASELINES = ("torus_mesh_switch_1", "torus_mesh_mesh_3")

BW_MULTIPLIERS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
SCALED_LAYERS = ["L1", "L2", "L3"]


def generate_noc_bw_configs():
    """Generate all (baseline_name, scaled_layer, bw_multiplier) combinations.

    Returns list of tuples: (baseline_name, scaled_layer, bw_mult)
    Total: 2 baselines × 3 layers × 7 multipliers = 42
    """
    configs = []
    for baseline_name in BASELINES:
        for layer in SCALED_LAYERS:
            for mult in BW_MULTIPLIERS:
                configs.append((baseline_name, layer, mult))
    return configs


def make_noc_for_config(baseline_name, scaled_layer, bw_mult):
    """Create a Hierarchy with one layer's BW scaled.

    Args:
        baseline_name: "torus_mesh_switch_1" or "torus_mesh_mesh_3"
        scaled_layer: "L1", "L2", or "L3"
        bw_mult: bandwidth multiplier for the scaled layer

    Returns:
        Hierarchy object
    """
    name = f"{baseline_name}_scale_{scaled_layer}_bw{bw_mult}"
    scales = {"L1": 1.0, "L2": 1.0, "L3": 1.0}
    scales[scaled_layer] = bw_mult
    return make_scaled_profile(
        baseline_name,
        l3_bandwidth_scale=scales["L3"],
        l2_bandwidth_scale=scales["L2"],
        l1_bandwidth_scale=scales["L1"],
        output_name=name,
    )


def find_max_sm_count(noc):
    """Return the sealed reference capacity for this NoC configuration."""

    return reference_max_sm_count(stacked_gpu_wgmma(), noc)


def make_arch_for_noc_config(noc):
    """Create a stacked_gpu_wgmma at the sealed feasible SM capacity.

    Returns:
        (arch, sm_count)
    """
    sm_count = find_max_sm_count(noc)
    arch = stacked_gpu_wgmma().with_sm_count(sm_count)
    return arch, sm_count


def compute_power_wall(chip_power_w, noc_power_per_device_w, tdp=TDP_W):
    """Check if total per-device power exceeds TDP.

    Returns:
        (hit_power_wall: bool, freq_scale_power: float)
    """
    total = chip_power_w + noc_power_per_device_w
    if total > tdp:
        ratio = total / tdp
        freq_scale = 1.0 / (ratio ** (1.0 / 3.0))
        return True, freq_scale
    return False, 1.0
