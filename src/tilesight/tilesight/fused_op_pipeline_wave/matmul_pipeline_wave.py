"""Pipeline-aware GEMM performance model.

将原有 matmul_fused_op_new_api_wave.py 的资源计算分解为 per-K-iteration 粒度，
结合软件流水线 overlap + wave head/tail 效应建模。
"""
from ..util import *
from ..tile_cache.cache_model import multi_level_hit_rate
import numpy as np
import math
import logging

from tilesight.arch import uses_dram_wave_quantization, uses_gpu_resource_model

from .resource_types import (
    PerIterationResources, PrologueEpilogueResources,
    TileResources, PipelineResult,
)
from .occupancy import compute_occupancy
from .pipeline_overlap import (
    compute_pipeline_tile_latency,
    compute_pipeline_tile_latency_with_occupancy,
    resources_to_times,
)
from .wave_model import compute_wave_adjusted_latency

log = logging.getLogger(__name__)


def calculate_matmul_pipeline_wave(op_shape, tb_shape, wp_shape, stage_num, arch,
                                   mem_levels, row_panel=1, batch=1,
                                   mma_type="wmma",
                                   column_panel=None,
                                   raster_axis="legacy"):
    """Pipeline-aware GEMM 性能模型入口。

    The optional `column_panel` argument specifies the M-direction sub-block
    height) and `raster_axis` ('legacy' / 'along_m' / 'along_n')
    to map cutlass RasterOrder + swizzle_size into the cache model.

    Cutlass mapping the runner uses:
        RasterOrder::AlongM, swizzle=K → row_panel=K,
                                          column_panel=gridM (or SM_Count),
                                          raster_axis='along_m'
        RasterOrder::AlongN, swizzle=K → row_panel=gridN (or SM_Count),
                                          column_panel=K,
                                          raster_axis='along_n'

    Args:
        op_shape: (M, N, K)
        tb_shape: (tb_m, tb_n, tb_k) thread-block tile
        wp_shape: (wp_m, wp_n, wp_k) warp tile
        stage_num: pipeline stages
        arch: TileSight arch (microbench-applied)
        mem_levels: per-operand cache traversal flags
        row_panel: N-direction sub-block width (legacy default 1)
        column_panel: M-direction sub-block height; None → legacy
            derivation `Stride_M = SM_Count // row_panel`.
        raster_axis: sub-block traversal direction.
        mma_type: 'wmma' / 'wgmma' / 'utcmma_cta1' / 'utcmma_cta2'

    Returns:
        PipelineResult
    """
    m, n, k = op_shape
    tb_m, tb_n, tb_k = tb_shape
    wp_m, wp_n, wp_k = wp_shape

    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    # ---- Grid dimensions ----
    gridM = math.ceil(m / tb_m)
    gridN = math.ceil(n / tb_n)
    gridK = math.ceil(k / tb_k)

    # ---- utcmma_cta2 cluster direction selection ----
    # tcgen05.mma cta_pair=2 forms a super-tile from 2 CTAs. Cutlass
    # picks the cluster axis to keep the super-tile reasonably square
    # (better A/B reuse + L2 hit rate). We mirror that heuristic:
    #   tb_m <= tb_n  →  M-cluster (super_tb_m = 2*tb_m, B broadcast)
    #   tb_m  > tb_n  →  N-cluster (super_tb_n = 2*tb_n, A broadcast)
    # If the preferred axis has insufficient grid (<2), try the other.
    # If both are <2 the cluster degenerates → fall back to cta1.
    cluster_axis = None
    if mma_type == "utcmma_cta2":
        pref = "M" if tb_m <= tb_n else "N"
        if pref == "M" and gridM >= 2:
            cluster_axis = "M"
        elif pref == "N" and gridN >= 2:
            cluster_axis = "N"
        elif pref == "M" and gridN >= 2:
            cluster_axis = "N"
            log.info("utcmma_cta2: preferred M-cluster but gridM=%d<2; "
                     "using N-cluster (gridN=%d)", gridM, gridN)
        elif pref == "N" and gridM >= 2:
            cluster_axis = "M"
            log.info("utcmma_cta2: preferred N-cluster but gridN=%d<2; "
                     "using M-cluster (gridM=%d)", gridN, gridM)
        else:
            log.info("utcmma_cta2 → cta1 fallback: gridM=%d, gridN=%d both <2",
                     gridM, gridN)
            mma_type = "utcmma_cta1"

    # Derived cluster shape, super-tile dims, and DSMEM broadcast splits
    if cluster_axis == "M":
        cm_eff, cn_eff = 2, 1
        super_tb_m, super_tb_n = tb_m * 2, tb_n
        smem_in1_div, smem_in2_div = 1, 2  # B broadcast → DSMEM split
    elif cluster_axis == "N":
        cm_eff, cn_eff = 1, 2
        super_tb_m, super_tb_n = tb_m, tb_n * 2
        smem_in1_div, smem_in2_div = 2, 1  # A broadcast → DSMEM split
    else:
        cm_eff, cn_eff = 1, 1
        super_tb_m, super_tb_n = tb_m, tb_n
        smem_in1_div, smem_in2_div = 1, 1
    cluster_size = cm_eff * cn_eff

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    # ---- Multi-level cache hit rate (L1.5 + L2) ----
    if mma_type == "utcmma_cta2":
        # super-tile combines 2 grid cells along cluster axis → halve
        # both row_panel and column_panel for the scheduler view.
        cp_eff = (
            max(column_panel // 2, 1)
            if column_panel is not None else None
        )
        cache_result = multi_level_hit_rate(
            m, n, k, super_tb_m, super_tb_n, tb_k,
            arch, mem_levels, max(row_panel // 2, 1),
            column_panel=cp_eff, raster_axis=raster_axis,
        )
    else:
        cache_result = multi_level_hit_rate(
            m, n, k, tb_m, tb_n, tb_k,
            arch, mem_levels, row_panel,
            column_panel=column_panel, raster_axis=raster_axis,
        )
    l1_5_hit_rate = cache_result['l1_5_hit_rate']
    l2_hit_rate = cache_result['l2_hit_rate']

    # ---- Compute overheads (tensor core shape mismatch) ----
    assert hasattr(arch, 'get_tensor_core_minimum_ptx'), \
        "arch.get_tensor_core_minimum_ptx does not exist"
    minimum_tc_ptx_shape = arch.get_tensor_core_minimum_ptx(bytes=in1_level[-1])
    compute_overheads = min(1,
                           math.ceil(wp_m) / minimum_tc_ptx_shape[0]
                           * math.ceil(wp_n) / minimum_tc_ptx_shape[1]
                           * math.ceil(wp_k) / minimum_tc_ptx_shape[2])
    if compute_overheads > 1:
        log.warning(" warp shape: warp_m %s, warp_n %s, warp_k %s", wp_m, wp_n, wp_k)
        log.warning(" arch name: %s, minimum ptx shape: %s", arch.core, minimum_tc_ptx_shape)

    # ================================================================
    # Per-K-iteration 资源分解 (per tile, per K-step)
    # ================================================================

    # ---- Per-iter total load IO (load A tile + B tile) ----
    # For cta2: super_tb_m / super_tb_n include the cluster doubling in
    # the active cluster axis; the broadcast operand stays single-tile
    # (loaded once per cluster, then DSMEM-shared).
    total_load_per_iter = tb_k * (super_tb_m * in1_level[0] * in1_level[-1]
                                  + super_tb_n * in2_level[0] * in2_level[-1])

    # Three-level IO decomposition: L1.5 → L2 → DDR
    # L1.5 IO: all requests go through L1.5 (read path).
    # Read and write have independent bandwidth, so bound = max(read, write).
    # read = total_load, write = total_load * (1-hit) [fill on miss].
    # Since read >= write always, l1_5_io = total_load (read-bound).
    l1_5_io_per_iter = total_load_per_iter
    l2_load_per_iter = total_load_per_iter * (1 - l1_5_hit_rate)

    # DRAM wave-byte quantization (应用于 per-iter 级别)
    assert hasattr(arch, 'ddr_wave_bytes'), "arch.ddr_wave_bytes does not exist"

    # DDR read per iter = L2 miss portion (of L1.5 misses)
    ddr_load_per_iter = l2_load_per_iter * (1 - l2_hit_rate) * DDR_non_ideal_para

    if uses_dram_wave_quantization(arch) and ddr_load_per_iter > 0:
        # 对 total DDR IO 做 wave bytes 量化, 然后分摊回 per-iter
        total_l2_read = (gridM / cm_eff) * (gridN / cn_eff) * k * (
            super_tb_m * in1_level[0] * in1_level[-1]
            + super_tb_n * in2_level[0] * in2_level[-1])
        l2_read_waves = total_l2_read / arch.ddr_wave_bytes
        wave_correction = math.ceil(l2_read_waves) / l2_read_waves if l2_read_waves > 0 else 1.0
        l2_load_per_iter *= wave_correction
        ddr_load_per_iter = l2_load_per_iter * (1 - l2_hit_rate) * DDR_non_ideal_para

    # L2 IO per iter (包含 l2 read 的 two-part cache 结构, 只对 L1.5 miss 部分)
    if arch.core in ("A100", "H100", "B200", "A100_LUT"):
        l2_io_per_iter = (l2_load_per_iter * l2_hit_rate
                          + l2_load_per_iter * (1 - l2_hit_rate) * 2)
    else:
        l2_io_per_iter = l2_load_per_iter

    # ---- Per-iter compute FLOPs ----
    # cta2 super-tile: super_tb_m * super_tb_n covers the full 2-CTA work.
    compute_per_iter = super_tb_m * super_tb_n * tb_k * 2 * compute_overheads

    # ---- Per-iter SMEM IO ----
    active_warp_per_tb = (tb_m / wp_m) * (tb_n / wp_n)

    if mma_type in ("wgmma", "utcmma_cta1"):
        # l2 -> smem + smem -> reg/tcgen05
        smem_per_iter = 2 * (tb_m * in1_level[0] * in1_level[-1]
                             + tb_n * in2_level[0] * in2_level[-1]) * tb_k
    elif mma_type == "utcmma_cta2":
        smem_per_iter = 2 * (super_tb_m * in1_level[0] * in1_level[-1]
                             + super_tb_n * in2_level[0] * in2_level[-1]) * tb_k
    elif arch.core == "V100":
        # store shared + shared load (no ldgsts)
        store_shared = tb_k * (tb_m * in1_level[0] * in1_level[-1]
                               + tb_n * in2_level[0] * in2_level[-1])
        shared_load = tb_k * (tb_m * in1_level[-1] + tb_n * in2_level[-1]) * \
                      active_warp_per_tb * \
                      (wp_m * in1_level[1] * in1_level[-1] + wp_n * in2_level[1] * in2_level[-1]) / \
                      max(tb_m * in1_level[-1] + tb_n * in2_level[-1], 1)
        smem_per_iter = store_shared + shared_load
    elif (arch.core in ("A100", "H100")) and mma_type == "wmma":
        # 0.5x ldgsts + shared load
        ldgsts = 0.5 * tb_k * (tb_m * in1_level[0] * in1_level[-1]
                                + tb_n * in2_level[0] * in2_level[-1])
        shared_load = tb_k * (tb_m * in1_level[-1] + tb_n * in2_level[-1]) * \
                      active_warp_per_tb * \
                      (wp_m * in1_level[1] * in1_level[-1] + wp_n * in2_level[1] * in2_level[-1]) / \
                      max(tb_m * in1_level[-1] + tb_n * in2_level[-1], 1)
        smem_per_iter = ldgsts + shared_load
    else:
        # 1x ldgsts + shared load (generic wmma)
        ldgsts = tb_k * (tb_m * in1_level[0] * in1_level[-1]
                         + tb_n * in2_level[0] * in2_level[-1])
        shared_load = tb_k * (tb_m * in1_level[-1] + tb_n * in2_level[-1]) * \
                      active_warp_per_tb * \
                      (wp_m * in1_level[1] * in1_level[-1] + wp_n * in2_level[1] * in2_level[-1]) / \
                      max(tb_m * in1_level[-1] + tb_n * in2_level[-1], 1)
        smem_per_iter = ldgsts + shared_load

    # ================================================================
    # Store (epilogue) 资源 — per tile, 只执行一次
    # ================================================================
    # store_l2_io is per (super-)tile; cta2 super-tile = super_tb_m × super_tb_n
    store_l2_io = super_tb_m * super_tb_n * out1_level[-1] * out1_level[0]
    store_ddr_io = store_l2_io  # store always goes to DDR
    if arch.core in ("A100", "H100", "B200", "A100_LUT"):
        store_l2_io_adjusted = store_l2_io * 2  # two-part L2 (store counts double)
    else:
        store_l2_io_adjusted = store_l2_io

    # DRAM store wave-byte quantization
    if uses_dram_wave_quantization(arch) and store_ddr_io > 0:
        if mma_type == "utcmma_cta2":
            total_store = m * n * 4  # from original code
        else:
            total_store = gridM * gridN * tb_m * tb_n * out1_level[-1] * out1_level[0]
        store_waves = total_store / arch.ddr_wave_bytes
        if store_waves > 0:
            sw_correction = math.ceil(store_waves) / store_waves
            store_ddr_io *= sw_correction
            store_l2_io_adjusted *= sw_correction

    # store smem IO (epilogue L1) — per super-tile for cta2
    store_smem_io = super_tb_m * super_tb_n * out1_level[-1] * out1_level[1]

    # ================================================================
    # Footprint 计算 (复用现有逻辑)
    # ================================================================
    if arch.core == "B200" and mma_type == "utcmma_cta2":
        # Per-CTA: broadcast operand is split across the cluster via
        # DSMEM (smem_in*_div=2 for the broadcast side, =1 for the
        # operand the CTA owns).
        smem_footprint = (tb_m / smem_in1_div * tb_k
                            * (1 - (in1_level[1] - in1_level[0])) * in1_level[-1]
                          + tb_n / smem_in2_div * tb_k
                            * (1 - (in2_level[1] - in2_level[0])) * in2_level[-1]
                          + tb_m * tb_n
                            * (out1_level[1] - out1_level[0]) * out1_level[-1]
                          ) * stage_num
    else:
        smem_footprint = (tb_m * tb_k * (1 - (in1_level[1] - in1_level[0])) * in1_level[-1]
                          + tb_n * tb_k * (1 - (in2_level[1] - in2_level[0])) * in2_level[-1]
                          + tb_m * tb_n * (out1_level[1] - out1_level[0]) * out1_level[-1]
                          ) * stage_num

    if arch.core == "B200" and mma_type in ("utcmma_cta1", "utcmma_cta2"):
        reg_footprint = math.ceil(tb_m * tb_n * out1_level[-1] / 4) / 256
    elif uses_gpu_resource_model(arch, include_lut=True):
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / out1_level[-1]))
                         + math.ceil(wp_m * wp_k / 32 / (4 / in1_level[-1])) * in1_level[1]
                         + math.ceil(wp_n * wp_k / 32 / (4 / in2_level[-1]))) * in2_level[1]
        reg_footprint = math.ceil(reg_footprint * REG_spill_para)
    else:
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / out1_level[-1]))
                         + math.ceil(wp_m * wp_k / 32 / (4 / in1_level[-1])) * in1_level[1]
                         + math.ceil(wp_n * wp_k / 32 / (4 / in2_level[-1]))) * in2_level[1]
        reg_footprint = math.ceil(reg_footprint * REG_spill_para)

    # ================================================================
    # 组装 TileResources
    # ================================================================
    warps_per_block = int((tb_m / wp_m) * (tb_n / wp_n))
    spatial_grids = (math.ceil(gridM / cm_eff), math.ceil(gridN / cn_eff))

    per_iter = PerIterationResources(
        ddr_io=ddr_load_per_iter,
        l2_io=l2_io_per_iter,
        l1_5_io=l1_5_io_per_iter,
        smem_io=smem_per_iter,
        compute_flops=compute_per_iter,
    )

    prologue_epilogue = PrologueEpilogueResources(
        prologue_ddr_io=ddr_load_per_iter * max(stage_num - 1, 0),
        prologue_l2_io=l2_io_per_iter * max(stage_num - 1, 0),
        prologue_l1_5_io=l1_5_io_per_iter * max(stage_num - 1, 0),
        prologue_smem_io=smem_per_iter * max(stage_num - 1, 0),
        epilogue_compute_flops=compute_per_iter * max(stage_num - 1, 0),
        store_ddr_io=store_ddr_io,
        store_l2_io=store_l2_io_adjusted,
        store_smem_io=store_smem_io,
    )

    tile_res = TileResources(
        per_iter=per_iter,
        prologue_epilogue=prologue_epilogue,
        num_iterations=gridK,
        stage_num=stage_num,
        smem_footprint=smem_footprint,
        reg_footprint=reg_footprint,
        warps_per_block=warps_per_block,
        grids=spatial_grids,
    )

    # ================================================================
    # Occupancy
    # ================================================================
    tiles_per_sm = compute_occupancy(
        smem_footprint, reg_footprint, warps_per_block, arch, mma_type)

    # ================================================================
    # Pipeline latency (SM-level, 考虑 occupancy 交织)
    # ================================================================
    # cta2: 1 cluster occupies cluster_size SMs → sm_count // cluster_size
    effective_sm_count = arch.sm_count // cluster_size if cluster_size > 1 else arch.sm_count

    sm_latency, pipeline_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_res, tiles_per_sm, arch, data_bytes=in1_level[-1],
        sm_count=effective_sm_count)

    # ================================================================
    # Wave adjustment
    # ================================================================
    total_tiles = int(np.prod(spatial_grids))

    total_latency, wave_info = compute_wave_adjusted_latency(
        sm_latency, tiles_per_sm, total_tiles, arch,
        pipeline_detail=pipeline_detail, mma_type=mma_type,
        tile_res=tile_res, data_bytes=in1_level[-1],
        sm_count_override=effective_sm_count)

    # 用实际执行条件重算 per_tile_latency 和 pipeline_detail (用于报告)
    # tiles < sm_count 时, 每 SM 只有 1 tile, active_sms = total_tiles
    actual_active_sms = min(total_tiles, effective_sm_count)
    actual_tiles_per_sm = (1 if total_tiles <= effective_sm_count
                           else min(math.ceil(total_tiles / effective_sm_count), tiles_per_sm))
    per_tile_latency, pipeline_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_res, actual_tiles_per_sm, arch, data_bytes=in1_level[-1],
        sm_count=effective_sm_count, active_sms=actual_active_sms)

    total_latency *= batch

    # ================================================================
    # Utilization 计算
    # ================================================================
    # utilization = 该资源在 roofline 下的理论时间 / pipeline 模型的实际时间
    # roofline 理论时间 = total_resource / system_bandwidth (使用全系统带宽)
    total_l2_read = (gridM / cm_eff) * (gridN / cn_eff) * k * (
        super_tb_m * in1_level[0] * in1_level[-1]
        + super_tb_n * in2_level[0] * in2_level[-1])
    total_compute = m * n * k * 2 * compute_overheads

    total_ddr = total_l2_read * (1 - l2_hit_rate) * DDR_non_ideal_para
    total_compute *= batch
    total_ddr *= batch

    # roofline 理论最短时间
    ddr_time_roofline = total_ddr / arch.ddr_bandwidth if arch.ddr_bandwidth > 0 else 0
    compute_time_roofline = resources_to_times(0, 0, 0, total_compute, arch, in1_level[-1])[4]

    ddr_util = ddr_time_roofline / total_latency if total_latency > 0 else 0
    compute_util = compute_time_roofline / total_latency if total_latency > 0 else 0

    result = PipelineResult(
        per_tile_latency=per_tile_latency,
        total_latency=total_latency,
        ddr_util=ddr_util,
        l2_util=0.0,  # can be refined
        l2_hit_rate=l2_hit_rate,
        smem_util=0.0,  # can be refined
        compute_util=compute_util,
        smem_footprint=smem_footprint,
        reg_footprint=reg_footprint,
        tiles_per_sm=tiles_per_sm,
        waves=wave_info['waves_float'],
        pipeline_detail=pipeline_detail,
    )

    return result


def calculate_matmul_triton_swizzle_pipeline_wave(op_shape, tb_shape, wp_shape,
                                                  stage_num, arch, mem_levels,
                                                  group_m=8, batch=1):
    """Pipeline-aware GEMM with Triton swizzle scheduling.

    与标准 matmul 的差异:
    - L2 hit rate 使用 triton swizzle 版本 (group_m 参数)
    - wmma only (无 mma_type 分支)
    - DDR_non_ideal_para=1.0

    参数签名与 fused_op_dtype/matmul_fused_op_triton_swizzle.py 一致。
    """
    m, n, k = op_shape
    tb_m, tb_n, tb_k = tb_shape
    wp_m, wp_n, wp_k = wp_shape

    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    gridM = math.ceil(m / tb_m)
    gridN = math.ceil(n / tb_n)
    gridK = math.ceil(k / tb_k)

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    # ---- L2 hit rate (triton swizzle 版本, L2-only — no multi-level for swizzle yet) ----
    l2_hit_rate = L2_hit_rate_flow_sim_reuse_distance_triton_swizzle(
        m, n, k, tb_m, tb_n, tb_k,
        arch.l2_capacity, arch.sm_count, mem_levels, group_m)
    l1_5_hit_rate = 0.0  # TODO: multi-level triton swizzle

    # ---- Per-K-iter 资源分解 ----
    total_load_per_iter = tb_k * (tb_m * in1_level[0] * in1_level[-1]
                                  + tb_n * in2_level[0] * in2_level[-1])
    # L1.5 IO = total_load (all requests read through L1.5; read/write have independent BW)
    l1_5_io_per_iter = total_load_per_iter
    l2_load_per_iter = total_load_per_iter * (1 - l1_5_hit_rate)
    ddr_load_per_iter = l2_load_per_iter * (1 - l2_hit_rate) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200", "A100_LUT"):
        l2_io_per_iter = (l2_load_per_iter * l2_hit_rate
                          + l2_load_per_iter * (1 - l2_hit_rate) * 2)
    else:
        l2_io_per_iter = l2_load_per_iter

    compute_per_iter = tb_m * tb_n * tb_k * 2

    # ---- Per-iter SMEM IO (wmma only) ----
    active_warp_per_tb = (tb_m / wp_m) * (tb_n / wp_n)

    if uses_gpu_resource_model(arch):
        ldgsts = 0.5 * tb_k * (tb_m * in1_level[0] * in1_level[-1]
                                + tb_n * in2_level[0] * in2_level[-1])
        shared_load = tb_k * (tb_m * in1_level[-1] + tb_n * in2_level[-1]) * \
                      active_warp_per_tb * \
                      (wp_m * in1_level[1] * in1_level[-1] + wp_n * in2_level[1] * in2_level[-1]) / \
                      max(tb_m * in1_level[-1] + tb_n * in2_level[-1], 1)
        smem_per_iter = ldgsts + shared_load
    else:
        ldgsts = tb_k * (tb_m * in1_level[0] * in1_level[-1]
                         + tb_n * in2_level[0] * in2_level[-1])
        shared_load = tb_k * (tb_m * in1_level[-1] + tb_n * in2_level[-1]) * \
                      active_warp_per_tb * \
                      (wp_m * in1_level[1] * in1_level[-1] + wp_n * in2_level[1] * in2_level[-1]) / \
                      max(tb_m * in1_level[-1] + tb_n * in2_level[-1], 1)
        smem_per_iter = ldgsts + shared_load

    # ---- Store ----
    store_l2_io = tb_m * tb_n * out1_level[-1] * out1_level[0]
    store_ddr_io = store_l2_io
    if arch.core in ("A100", "H100", "B200", "A100_LUT"):
        store_l2_io_adjusted = store_l2_io * 2
    else:
        store_l2_io_adjusted = store_l2_io
    store_smem_io = tb_m * tb_n * out1_level[-1] * out1_level[1]

    # ---- Footprint ----
    smem_footprint = (tb_m * tb_k * (1 - (in1_level[1] - in1_level[0])) * in1_level[-1]
                      + tb_n * tb_k * (1 - (in2_level[1] - in2_level[0])) * in2_level[-1]
                      + tb_m * tb_n * (out1_level[1] - out1_level[0]) * out1_level[-1]
                      ) * stage_num

    if uses_gpu_resource_model(arch, include_lut=True):
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / out1_level[-1]))
                         + math.ceil(wp_m * wp_k / 32 / (4 / in1_level[-1])) * in1_level[1]
                         + math.ceil(wp_n * wp_k / 32 / (4 / in2_level[-1]))) * in2_level[1]
        reg_footprint = math.ceil(reg_footprint * REG_spill_para)
    else:
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / out1_level[-1]))
                         + math.ceil(wp_m * wp_k / 32 / (4 / in1_level[-1])) * in1_level[1]
                         + math.ceil(wp_n * wp_k / 32 / (4 / in2_level[-1]))) * in2_level[1]
        reg_footprint = math.ceil(reg_footprint * REG_spill_para)

    # ---- TileResources ----
    warps_per_block = int(active_warp_per_tb)
    spatial_grids = (gridM, gridN)

    per_iter = PerIterationResources(
        ddr_io=ddr_load_per_iter, l2_io=l2_io_per_iter,
        l1_5_io=l1_5_io_per_iter,
        smem_io=smem_per_iter, compute_flops=compute_per_iter,
    )
    pe = PrologueEpilogueResources(
        prologue_ddr_io=ddr_load_per_iter * max(stage_num - 1, 0),
        prologue_l2_io=l2_io_per_iter * max(stage_num - 1, 0),
        prologue_l1_5_io=l1_5_io_per_iter * max(stage_num - 1, 0),
        prologue_smem_io=smem_per_iter * max(stage_num - 1, 0),
        epilogue_compute_flops=compute_per_iter * max(stage_num - 1, 0),
        store_ddr_io=store_ddr_io,
        store_l2_io=store_l2_io_adjusted,
        store_smem_io=store_smem_io,
    )
    tile_res = TileResources(
        per_iter=per_iter, prologue_epilogue=pe,
        num_iterations=gridK, stage_num=stage_num,
        smem_footprint=smem_footprint, reg_footprint=reg_footprint,
        warps_per_block=warps_per_block, grids=spatial_grids,
    )

    # ---- Occupancy + Pipeline + Wave ----
    tiles_per_sm = compute_occupancy(smem_footprint, reg_footprint, warps_per_block, arch)

    sm_latency, pipeline_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_res, tiles_per_sm, arch, data_bytes=in1_level[-1])

    total_tiles = int(np.prod(spatial_grids))
    total_latency, wave_info = compute_wave_adjusted_latency(
        sm_latency, tiles_per_sm, total_tiles, arch,
        pipeline_detail=pipeline_detail,
        tile_res=tile_res, data_bytes=in1_level[-1])
    total_latency *= batch

    # per_tile for reporting
    actual_active_sms = min(total_tiles, arch.sm_count)
    actual_tps = 1 if total_tiles <= arch.sm_count else min(math.ceil(total_tiles / arch.sm_count), tiles_per_sm)
    per_tile_latency, pipeline_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_res, actual_tps, arch, data_bytes=in1_level[-1],
        active_sms=actual_active_sms)

    # ---- Utilization ----
    total_l2_read = gridM * gridN * k * (tb_m * in1_level[0] * in1_level[-1]
                                         + tb_n * in2_level[0] * in2_level[-1])
    total_compute = m * n * k * 2 * batch
    total_ddr = total_l2_read * (1 - l1_5_hit_rate) * (1 - l2_hit_rate) * DDR_non_ideal_para * batch

    ddr_time_rf = total_ddr / arch.ddr_bandwidth if arch.ddr_bandwidth > 0 else 0
    compute_time_rf = resources_to_times(0, 0, 0, total_compute, arch, in1_level[-1])[4]
    ddr_util = ddr_time_rf / total_latency if total_latency > 0 else 0
    compute_util = compute_time_rf / total_latency if total_latency > 0 else 0

    return PipelineResult(
        per_tile_latency=per_tile_latency, total_latency=total_latency,
        ddr_util=ddr_util, l2_hit_rate=l2_hit_rate, compute_util=compute_util,
        smem_footprint=smem_footprint, reg_footprint=reg_footprint,
        tiles_per_sm=tiles_per_sm, waves=wave_info['waves_float'],
        pipeline_detail=pipeline_detail,
    )
