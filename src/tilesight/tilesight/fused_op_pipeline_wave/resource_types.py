from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


@dataclass
class PerIterationResources:
    """一次内层循环迭代的资源量（如 GEMM 的一次 K-tile）。

    对于 element-wise op（无内层循环），表示整个 tile 的资源量。
    各 mem 层之间可 overlap（取 max），但 load 整体与 compute 之间
    是否 overlap 取决于 stage_num（由 pipeline_overlap 决定）。
    """
    ddr_io: float = 0.0         # DDR 流量 (bytes)
    l2_io: float = 0.0          # L2 流量 (bytes)
    l1_5_io: float = 0.0        # L1.5 流量 (bytes), 0 if no L1.5
    smem_io: float = 0.0        # SMEM 读写 (bytes)
    compute_flops: float = 0.0  # 计算量 (FLOPs)


@dataclass
class PrologueEpilogueResources:
    """Pipeline prologue (填充) 和 epilogue (排空+store) 的资源。

    prologue: pipeline 填充阶段，只有 load，无 compute。
        持续 (stage_num - 1) 次迭代。
    epilogue: pipeline 排空阶段，只有 compute，无 load。
        持续 (stage_num - 1) 次迭代 + 最终 store。
    当 stage_num=1 时，prologue 和 epilogue 均为 0。
    """
    # Prologue 资源（pipeline 填充）
    prologue_ddr_io: float = 0.0
    prologue_l2_io: float = 0.0
    prologue_l1_5_io: float = 0.0
    prologue_smem_io: float = 0.0

    # Epilogue 资源（pipeline 排空）
    epilogue_compute_flops: float = 0.0

    # Output store 资源（epilogue 最后的写回）
    store_ddr_io: float = 0.0
    store_l2_io: float = 0.0
    store_l1_5_io: float = 0.0   # typically 0 (store bypasses L1.5)
    store_smem_io: float = 0.0


@dataclass
class TileResources:
    """一个 tile 的完整资源描述，按阶段分解。

    对于 GEMM: num_iterations = gridK, stage_num = software pipeline depth
    对于 element-wise: num_iterations = 1, stage_num = 1
    对于 reduce: num_iterations = reduction_iters, stage_num 可 >= 1
    """
    per_iter: PerIterationResources
    prologue_epilogue: PrologueEpilogueResources
    num_iterations: int          # 内层循环次数 (gridK for matmul)
    stage_num: int               # 软件流水线深度
    smem_footprint: float        # bytes, 用于 occupancy 计算
    reg_footprint: float         # 寄存器数 (per warp, 4-byte units)
    warps_per_block: int         # 每个 thread block 的 warp 数
    grids: Tuple[int, ...]       # 空间维度 (gridM, gridN) 等


@dataclass
class PipelineDetail:
    """Pipeline 各阶段的时间分解，用于 wave model 精细建模。"""
    prologue_time: float = 0.0          # prologue 总时间 (秒)
    steady_time_per_iter: float = 0.0   # steady state 每迭代时间 (秒)
    epilogue_time: float = 0.0          # epilogue 总时间 (秒)
    mem_time_per_iter: float = 0.0      # 每迭代 mem 时间 (秒)
    compute_time_per_iter: float = 0.0  # 每迭代 compute 时间 (秒)


@dataclass
class PipelineResult:
    """Pipeline overlap 分析的完整输出。"""
    per_tile_latency: float      # 单 tile 延迟 (秒)
    total_latency: float         # 全 kernel 延迟 (秒), 包含 wave 效应

    # 利用率
    ddr_util: float = 0.0
    l2_util: float = 0.0
    l2_hit_rate: float = 0.0
    smem_util: float = 0.0
    compute_util: float = 0.0

    # Footprint
    smem_footprint: float = 0.0
    reg_footprint: float = 0.0

    # Occupancy & wave 信息
    tiles_per_sm: int = 1
    waves: float = 1.0

    # Pipeline 各阶段时间（用于调试/可视化）
    pipeline_detail: Optional[PipelineDetail] = None
