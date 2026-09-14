"""Unified multi-level cache hit rate entry point.

Auto-detects L1.5 capability from arch parameters and dispatches
to the appropriate model:
- L1.5 enabled (l1_5_group_size > 0): two-level SDCM cascade
- L1.5 disabled: falls back to existing L2-only model
"""

from tilesight.util.L2_hit_rate_flow_sim_reuse_distance import (
    L2_hit_rate_flow_sim_reuse_distance,
)
from .reuse_distance_multilevel import multilevel_hit_rate_reuse_distance


def multi_level_hit_rate(M, N, K, tb_m, tb_n, tb_k, arch, mem_levels,
                         row_panel, column_panel=None,
                         raster_axis="legacy"):
    """Compute multi-level cache hit rates, auto-detecting L1.5 from arch.

    The optional `column_panel` argument specifies the M-direction
    sub-block height) and `raster_axis` ('along_m' default, or
    'along_n'). Legacy callers passing only `row_panel` are preserved
    byte-for-byte (column_panel=None → Stride_M = SM_Count // row_panel,
    raster_axis='along_m' → original extract_blocks order).

    Args:
        M, N, K: operation dimensions
        tb_m, tb_n, tb_k: tile block dimensions
        arch: architecture object (with optional l1_5_* attributes)
        mem_levels: dict with 'in1', 'in2', 'out1' memory level descriptors
        row_panel: N-direction sub-block width (Stride_N)
        column_panel: M-direction sub-block height (Stride_M); if None,
            derived as SM_Count // row_panel (legacy behaviour).
        raster_axis: 'along_m' or 'along_n'. Selects whether
            sub-blocks advance along M first (cutlass AlongM) or N
            first (cutlass AlongN).

    Returns:
        dict with keys:
            'l1_5_hit_rate': fraction of total requests served by L1.5
                             (0.0 if L1.5 not present)
            'l2_hit_rate': fraction of L1.5 misses served by L2
            'ddr_miss_rate': fraction of total requests going to DDR
    """
    l1_5_group_size = getattr(arch, 'l1_5_group_size', 0)

    if l1_5_group_size > 0:
        # The L1.5 cascade does not yet accept column_panel/raster_axis;
        # for now we only forward row_panel so existing behaviour is
        # preserved. Extending the cascade to honor the new args is a
        # follow-up; the L2-only fallback below is what the runner
        # exercises for the cutlass profiler sweep.
        return multilevel_hit_rate_reuse_distance(
            M, N, K, tb_m, tb_n, tb_k,
            arch.l2_capacity, arch.sm_count, mem_levels, row_panel,
            l1_5_group_size=l1_5_group_size,
            l1_5_capacity_per_group=getattr(arch, 'l1_5_capacity_per_group', 0),
            l1_5_associativity=getattr(arch, 'l1_5_associativity', 8),
            l1_5_cacheline_bytes=getattr(arch, 'l1_5_cacheline_bytes', 128),
        )
    else:
        # Fallback: existing L2-only model, now with column_panel + raster_axis
        l2_hr = L2_hit_rate_flow_sim_reuse_distance(
            M, N, K, tb_m, tb_n, tb_k,
            arch.l2_capacity, arch.sm_count, mem_levels, row_panel,
            column_panel=column_panel,
            raster_axis=raster_axis,
        )
        return {
            'l1_5_hit_rate': 0.0,
            'l2_hit_rate': float(l2_hr),
            'ddr_miss_rate': float(1 - l2_hr),
        }
