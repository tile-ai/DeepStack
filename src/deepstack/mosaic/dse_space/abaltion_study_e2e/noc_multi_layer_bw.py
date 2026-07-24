"""
noc_multi_layer_bw.py — Multi-layer NoC BW scaling for ablation Step 7.

Extends the single-layer scaling in case_study_which_layer_noc_matter/noc_bw_config.py
to support setting all 3 layers' BW simultaneously.
"""

from mosaic.cost.capacity import max_sm_count as reference_max_sm_count
from mosaic.noc.noc_config_set import make_scaled_profile


def make_noc_multi(baseline_name, bw_mults):
    """Create a Hierarchy with all 3 layers' BW scaled independently.

    Args:
        baseline_name: "torus_mesh_switch_1" or "torus_mesh_mesh_3"
        bw_mults: dict, e.g. {"L1": 1.0, "L2": 1.5, "L3": 0.8}

    Returns:
        Hierarchy object with 256 devices
    """
    name = f"{baseline_name}_L1x{bw_mults.get('L1',1.0)}_L2x{bw_mults.get('L2',1.0)}_L3x{bw_mults.get('L3',1.0)}"
    return make_scaled_profile(
        baseline_name,
        l3_bandwidth_scale=bw_mults.get("L3", 1.0),
        l2_bandwidth_scale=bw_mults.get("L2", 1.0),
        l1_bandwidth_scale=bw_mults.get("L1", 1.0),
        output_name=name,
    )


def _make_raw_arch_with_dram(m, n, smem_cap, l1_throughput, sm_count):
    """Create a stacked_gpu_wgmma with DRAM config, NO freq scaling or Little's law."""
    from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
        _make_raw_arch,
    )

    return _make_raw_arch(
        m,
        n,
        smem_cap,
        l1_throughput,
        sm_count=sm_count,
    )


def find_max_sm_count_for_noc(
    noc,
    m=4,
    n=4,
    smem_cap=256 * 1024,
    l1_throughput=256,
):
    """Return the sealed capacity for one DRAM and NoC configuration."""

    template = _make_raw_arch_with_dram(
        m,
        n,
        smem_cap,
        l1_throughput,
        sm_count=1,
    )
    return reference_max_sm_count(template, noc)


def make_arch_for_noc(
    noc,
    m=4,
    n=4,
    smem_cap=256 * 1024,
    l1_throughput=256,
):
    """Create a reference architecture at the sealed feasible SM capacity.

    Args:
        noc: Hierarchy object (with BW scaling)
        m, n, smem_cap, l1_throughput: DRAM layer config from Step 6 candidate
    Returns:
        (arch, sm_count)
    """
    from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
        _make_raw_arch,
        compute_thermal_freq_scale,
    )

    sm_count = find_max_sm_count_for_noc(
        noc,
        m,
        n,
        smem_cap,
        l1_throughput,
    )

    arch = _make_raw_arch(
        m,
        n,
        smem_cap,
        l1_throughput,
        sm_count=sm_count,
    )

    freq_scale = compute_thermal_freq_scale(m)
    arch = arch.with_thermal_scale(freq_scale)
    arch, _, _ = arch.with_littles_law()

    return arch, sm_count


if __name__ == "__main__":
    # Quick test
    noc = make_noc_multi("torus_mesh_switch_1", {"L1": 1.0, "L2": 1.5, "L3": 0.8})
    print(f"NoC: {noc.name}, devices: {noc.num_devices}")
    sm = find_max_sm_count_for_noc(noc)
    print(f"Max SM count: {sm}")
    arch, sm_count = make_arch_for_noc(noc)
    print(f"Arch SM={sm_count}")
