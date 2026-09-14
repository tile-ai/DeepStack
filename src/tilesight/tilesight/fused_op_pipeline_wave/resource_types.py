from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


@dataclass
class PerIterationResources:
    """Resource quantities for one inner-loop iteration, such as one GEMM K-tile.

    For an element-wise operation without an inner loop, these quantities describe
    the entire tile. Different memory levels can overlap (take the maximum),
    but whether the overall load overlaps with compute depends on stage_num,
    as determined by pipeline_overlap.
    """
    ddr_io: float = 0.0
    l2_io: float = 0.0
    l1_5_io: float = 0.0
    smem_io: float = 0.0
    compute_flops: float = 0.0


@dataclass
class PrologueEpilogueResources:
    """Resources for the pipeline prologue (fill) and epilogue (drain + store).

    prologue: Pipeline fill phase, with loads only and no compute.
        Lasts (stage_num - 1) iterations.
    epilogue: Pipeline drain phase, with compute only and no loads.
        Lasts (stage_num - 1) iterations plus the final store.
    When stage_num=1, both prologue and epilogue are zero.
    """
    prologue_ddr_io: float = 0.0
    prologue_l2_io: float = 0.0
    prologue_l1_5_io: float = 0.0
    prologue_smem_io: float = 0.0

    epilogue_compute_flops: float = 0.0

    store_ddr_io: float = 0.0
    store_l2_io: float = 0.0
    store_l1_5_io: float = 0.0   # typically 0 (store bypasses L1.5)
    store_smem_io: float = 0.0


@dataclass
class TileResources:
    """Complete resource description of a tile, broken down by phase.

    For GEMM: num_iterations = gridK, stage_num = software pipeline depth.
    For element-wise operations: num_iterations = 1, stage_num = 1.
    For reduce: num_iterations = reduction_iters, stage_num may be >= 1.
    """
    per_iter: PerIterationResources
    prologue_epilogue: PrologueEpilogueResources
    num_iterations: int
    stage_num: int
    smem_footprint: float
    reg_footprint: float
    warps_per_block: int
    grids: Tuple[int, ...]


@dataclass
class PipelineDetail:
    """Time breakdown by pipeline phase for detailed wave modeling."""
    prologue_time: float = 0.0
    steady_time_per_iter: float = 0.0
    epilogue_time: float = 0.0
    mem_time_per_iter: float = 0.0
    compute_time_per_iter: float = 0.0


@dataclass
class PipelineResult:
    """Complete output of pipeline overlap analysis."""
    per_tile_latency: float
    total_latency: float

    ddr_util: float = 0.0
    l2_util: float = 0.0
    l2_hit_rate: float = 0.0
    smem_util: float = 0.0
    compute_util: float = 0.0

    # Footprint
    smem_footprint: float = 0.0
    reg_footprint: float = 0.0

    tiles_per_sm: int = 1
    waves: float = 1.0

    pipeline_detail: Optional[PipelineDetail] = None
