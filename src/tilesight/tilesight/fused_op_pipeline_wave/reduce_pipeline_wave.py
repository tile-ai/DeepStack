"""Pipeline-aware reduce operation performance model.

Reduce ops 有 reduction loop（类似 GEMM 的 K-loop），可受益于软件流水线。
包含 general reduce 和 inter-thread reduce 两个变体。
"""
import numpy as np
import math
import logging
from ..util import *
from tilesight.arch import uses_dram_wave_quantization

from .resource_types import (
    PerIterationResources, PrologueEpilogueResources,
    TileResources, PipelineResult,
)
from .occupancy import compute_occupancy
from .pipeline_overlap import (
    compute_pipeline_tile_latency,
    compute_pipeline_tile_latency_with_occupancy,
)
from .wave_model import compute_wave_adjusted_latency

log = logging.getLogger(__name__)


def _apply_wave_bytes_correction(io_val, arch):
    """Apply ddr_wave_bytes quantization when the architecture opts in."""
    if io_val > 0 and uses_dram_wave_quantization(arch):
        waves = io_val / arch.ddr_wave_bytes
        if waves > 0:
            return io_val * math.ceil(waves) / waves
    return io_val


def _compute_thread_overhead(thread_per_tb):
    """计算线程数不足导致的计算开销。"""
    if thread_per_tb <= 32:
        return 32 / thread_per_tb
    elif thread_per_tb <= 128:
        return 128 / thread_per_tb
    elif thread_per_tb <= 256:
        return 256 / thread_per_tb
    elif thread_per_tb <= 384:
        return 384 / thread_per_tb
    else:
        return 1


def _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate, data_bytes=4):
    """从 TileResources 构建 PipelineResult。"""
    sm_latency, pipeline_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_res, tiles_per_sm, arch, data_bytes=data_bytes)
    per_tile_latency, _ = compute_pipeline_tile_latency(
        tile_res, arch, data_bytes=data_bytes)

    total_tiles = int(np.prod(tile_res.grids))
    total_latency, wave_info = compute_wave_adjusted_latency(
        sm_latency, tiles_per_sm, total_tiles, arch,
        pipeline_detail=pipeline_detail,
        tile_res=tile_res, data_bytes=data_bytes)

    return PipelineResult(
        per_tile_latency=per_tile_latency,
        total_latency=total_latency,
        l2_hit_rate=l2_hit_rate,
        smem_footprint=tile_res.smem_footprint,
        reg_footprint=tile_res.reg_footprint,
        tiles_per_sm=tiles_per_sm,
        waves=wave_info['waves_float'],
        pipeline_detail=pipeline_detail,
    )


def _parse_reduction_axes(out_shape, reduction_shape, out_axis_mapping,
                          reduction_axis_mapping, out_tb_shape, dim_threads):
    """解析 reduction 轴结构，返回所需的中间数据。

    Returns:
        dict with keys: spatial_grids, reduction_grids, reduction_step,
        in1_tb_spatial_shape, in1_tb_current_step_shape,
        out_thread_shape, in1_thread_shape, thread_per_tb
    """
    spatial_grids = [math.ceil(dim / tb_dim)
                     for dim, tb_dim in zip(out_shape, out_tb_shape)]
    thread_per_tb = np.prod(dim_threads)
    out_thread_shape = [math.ceil(tb_dim / thread_dim)
                        for tb_dim, thread_dim in zip(out_tb_shape, dim_threads)]

    in1_tb_spatial_shape = []
    in1_tb_current_step_shape = []
    reduction_grids = []
    reduction_step = []
    in1_thread_shape = []

    for i, dim_size in enumerate(out_tb_shape):
        input_index = out_axis_mapping[i]
        if input_index == 0:
            in1_tb_spatial_shape.append(dim_size)
            in1_thread_shape.append(out_thread_shape[i])

    in1_tb_spatial_index = len(in1_tb_spatial_shape) - 1

    for i, dim_size in enumerate(reduction_shape):
        input_index, current_step = reduction_axis_mapping[i]
        reduction_grids.append(math.ceil(dim_size / current_step))
        reduction_step.append(current_step)

        if input_index == 2:  # 公共归约轴
            in1_tb_current_step_shape.append(current_step)
        elif input_index == 1:  # input2 私有归约轴
            in1_tb_spatial_shape[in1_tb_spatial_index] += current_step - 1
            in1_tb_spatial_index -= 1
        elif input_index == 0:  # input1 私有归约轴
            in1_tb_current_step_shape.append(current_step)

    return {
        'spatial_grids': spatial_grids,
        'reduction_grids': reduction_grids,
        'reduction_step': reduction_step,
        'in1_tb_spatial_shape': in1_tb_spatial_shape,
        'in1_tb_current_step_shape': in1_tb_current_step_shape,
        'out_thread_shape': out_thread_shape,
        'in1_thread_shape': in1_thread_shape,
        'thread_per_tb': thread_per_tb,
    }


# =====================================================================
# General Reduce (N_0)
# =====================================================================
def calculate_N_0_general_reduce_pipeline_wave(
        out_shape, reduction_shape, out_axis_mapping, reduction_axis_mapping,
        out_tb_shape, dim_threads, arch, mem_levels, compute_at=1,
        stage_num=1):
    """General reduce pipeline-aware model.

    Args:
        out_shape: 输出形状
        reduction_shape: 归约轴形状
        out_axis_mapping: 输出轴到输入的映射
        reduction_axis_mapping: 归约轴映射 [[input_idx, step], ...]
        out_tb_shape: 输出 tile 形状
        dim_threads: 各维度线程数
        arch: 架构对象
        mem_levels: {'in1': [...], 'out1': [...]}
        compute_at: local register 保持级别 (-1, 0, 或正整数)
        stage_num: 软件流水线深度 (default 1)

    Returns:
        PipelineResult
    """
    DDR_non_ideal_para = 1.1
    REG_spill_para = 1.1

    parsed = _parse_reduction_axes(
        out_shape, reduction_shape, out_axis_mapping,
        reduction_axis_mapping, out_tb_shape, dim_threads)

    spatial_grids = parsed['spatial_grids']
    reduction_grids = parsed['reduction_grids']
    reduction_step = parsed['reduction_step']
    in1_tb_spatial_shape = parsed['in1_tb_spatial_shape']
    in1_tb_current_step_shape = parsed['in1_tb_current_step_shape']
    out_thread_shape = parsed['out_thread_shape']
    in1_thread_shape = parsed['in1_thread_shape']
    thread_per_tb = parsed['thread_per_tb']

    in1_level = mem_levels['in1']
    out1_level = mem_levels['out1']

    l2_hit_rate = 0

    # ---- Total IO ----
    total_l2_read = (in1_level[-1] * np.prod(spatial_grids) * np.prod(reduction_grids)
                     * np.prod(in1_tb_spatial_shape) * np.prod(in1_tb_current_step_shape)
                     * in1_level[0])
    total_l2_store = np.prod(spatial_grids) * np.prod(out_tb_shape) * out1_level[-1] * out1_level[0]

    total_l2_read = _apply_wave_bytes_correction(total_l2_read, arch)
    total_l2_store = _apply_wave_bytes_correction(total_l2_store, arch)

    total_ddr = (total_l2_read * (1 - l2_hit_rate) + total_l2_store) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        total_l2_io = (total_l2_read * l2_hit_rate
                       + total_l2_read * (1 - l2_hit_rate) * 2
                       + total_l2_store * 2)
    else:
        total_l2_io = total_l2_read + total_l2_store

    # ---- Reduction loop 迭代数 ----
    num_reduction_iters = int(np.prod(reduction_grids))

    # ---- Per reduction-iteration 资源 (per tile) ----
    per_iter_l2_read = (in1_level[-1] * np.prod(in1_tb_spatial_shape)
                        * np.prod(in1_tb_current_step_shape) * in1_level[0])
    per_iter_ddr = per_iter_l2_read * (1 - l2_hit_rate) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        per_iter_l2 = (per_iter_l2_read * l2_hit_rate
                       + per_iter_l2_read * (1 - l2_hit_rate) * 2)
    else:
        per_iter_l2 = per_iter_l2_read

    # SMEM IO per reduction iter: ldg + sts (load to smem)
    per_iter_smem = 2 * per_iter_l2_read  # ldg + sts

    # lds (load from smem for compute)
    overheads = _compute_thread_overhead(thread_per_tb)

    # compute_at 决定 lds 的模式
    current_in1_thread_shape = list(in1_thread_shape)
    current_reduction_step = list(reduction_step)
    current_reduction_axis_mapping = list(reduction_axis_mapping)

    if compute_at == -1:
        reg_footprint = np.prod(out_thread_shape) * out1_level[-1] / 4 + 2 + len(reduction_shape)
        per_iter_smem += (in1_level[-1] * thread_per_tb
                          * np.prod(out_thread_shape) * np.prod(reduction_step)
                          * in1_level[1])
    elif compute_at == 0:
        reg_footprint = (np.prod(current_in1_thread_shape) * in1_level[1] * in1_level[-1] / 4
                         + np.prod(out_thread_shape) * out1_level[-1] / 4
                         + len(reduction_shape) + 2)
        per_iter_smem += (in1_level[-1] * thread_per_tb * np.prod(current_reduction_step)
                          * np.prod(current_in1_thread_shape) * in1_level[1])
    else:
        assert len(current_reduction_step) >= compute_at
        j = 0
        for i in range(compute_at):
            last_row = current_reduction_axis_mapping[-1]
            current_reduction_axis_mapping = current_reduction_axis_mapping[:-1]
            last_step = current_reduction_step.pop()
            if last_row[0] == 2:
                current_in1_thread_shape.insert(0, last_step)
            elif last_row[0] == 1:
                current_in1_thread_shape[-1 - j] += last_step - 1
                j += 1
            elif last_row[0] == 0:
                current_in1_thread_shape.insert(0, last_step)
                j += 1
        reg_footprint = (np.prod(out_thread_shape) * out1_level[-1] / 4
                         + np.prod(current_in1_thread_shape) * in1_level[1] * in1_level[-1] / 4
                         + len(reduction_shape) + 2)
        per_iter_smem += (in1_level[-1] * thread_per_tb * np.prod(current_reduction_step)
                          * np.prod(current_in1_thread_shape) * in1_level[1])

    per_iter_smem *= overheads
    per_iter_compute = 2 * np.prod(out_tb_shape) * np.prod(reduction_step) * overheads

    reg_footprint = math.ceil(reg_footprint * REG_spill_para)

    # ---- Store (epilogue) ----
    store_smem = thread_per_tb * np.prod(out_thread_shape) * out1_level[1] * out1_level[-1] * overheads
    store_ddr = np.prod(out_tb_shape) * out1_level[-1] * out1_level[0]
    store_l2 = store_ddr

    # ---- Footprint ----
    smem_footprint = (np.prod(in1_tb_spatial_shape) * np.prod(in1_tb_current_step_shape)
                      * (1 - (in1_level[1] - in1_level[0])) * in1_level[-1]
                      + np.prod(out_tb_shape) * (out1_level[1] - out1_level[0]) * out1_level[-1])
    smem_footprint *= stage_num

    # ---- TileResources ----
    per_iter = PerIterationResources(
        ddr_io=per_iter_ddr, l2_io=per_iter_l2,
        smem_io=per_iter_smem, compute_flops=per_iter_compute,
    )
    pe = PrologueEpilogueResources(
        prologue_ddr_io=per_iter_ddr * max(stage_num - 1, 0),
        prologue_l2_io=per_iter_l2 * max(stage_num - 1, 0),
        prologue_smem_io=per_iter_smem * max(stage_num - 1, 0),
        epilogue_compute_flops=per_iter_compute * max(stage_num - 1, 0),
        store_ddr_io=store_ddr,
        store_l2_io=store_l2,
        store_smem_io=store_smem,
    )

    tile_res = TileResources(
        per_iter=per_iter, prologue_epilogue=pe,
        num_iterations=num_reduction_iters,
        stage_num=stage_num,
        smem_footprint=smem_footprint,
        reg_footprint=reg_footprint,
        warps_per_block=max(int(thread_per_tb / 32), 1),
        grids=tuple(spatial_grids),
    )

    tiles_per_sm = compute_occupancy(
        smem_footprint, reg_footprint,
        max(int(thread_per_tb / 32), 1), arch)

    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                         data_bytes=out1_level[-1])


# =====================================================================
# 双输入解析 (general_reduce / general_inter_thread_reduce 使用)
# =====================================================================
def _parse_reduction_axes_dual(out_shape, reduction_shape, out_axis_mapping,
                               reduction_axis_mapping, out_tb_shape, dim_threads):
    """解析双输入 reduction 轴结构 (in1 + in2)。"""
    spatial_grids = [math.ceil(dim / tb_dim)
                     for dim, tb_dim in zip(out_shape, out_tb_shape)]
    thread_per_tb = np.prod(dim_threads)
    out_thread_shape = [math.ceil(tb_dim / thread_dim)
                        for tb_dim, thread_dim in zip(out_tb_shape, dim_threads)]

    in1_tb_spatial_shape, in2_tb_spatial_shape = [], []
    in1_tb_current_step_shape, in2_tb_current_step_shape = [], []
    in1_tb_reduction_shape, in2_tb_reduction_shape = [], []
    reduction_grids, reduction_step = [], []
    in1_thread_shape, in2_thread_shape = [], []

    for i, dim_size in enumerate(out_tb_shape):
        idx = out_axis_mapping[i]
        if idx == 0:
            in1_tb_spatial_shape.append(dim_size)
            in1_thread_shape.append(out_thread_shape[i])
        elif idx == 1:
            in2_tb_spatial_shape.append(dim_size)
            in2_thread_shape.append(out_thread_shape[i])
        elif idx == 2:
            in1_tb_spatial_shape.append(dim_size)
            in1_thread_shape.append(out_thread_shape[i])
            in2_tb_spatial_shape.append(dim_size)
            in2_thread_shape.append(out_thread_shape[i])

    in1_sp_idx = len(in1_tb_spatial_shape) - 1
    in2_sp_idx = len(in2_tb_spatial_shape) - 1

    for i, dim_size in enumerate(reduction_shape):
        input_index, current_step = reduction_axis_mapping[i]
        reduction_grids.append(math.ceil(dim_size / current_step))
        reduction_step.append(current_step)
        if input_index == 2:
            in1_tb_reduction_shape.append(dim_size)
            in2_tb_reduction_shape.append(dim_size)
            in1_tb_current_step_shape.append(current_step)
            in2_tb_current_step_shape.append(current_step)
        elif input_index == 1:
            in2_tb_reduction_shape.append(dim_size)
            in2_tb_current_step_shape.append(current_step)
            in1_tb_spatial_shape[in1_sp_idx] += current_step - 1
            in1_sp_idx -= 1
        elif input_index == 0:
            in1_tb_reduction_shape.append(dim_size)
            in1_tb_current_step_shape.append(current_step)
            in2_tb_spatial_shape[in2_sp_idx] += current_step - 1
            in2_sp_idx -= 1

    return {
        'spatial_grids': spatial_grids, 'reduction_grids': reduction_grids,
        'reduction_step': reduction_step,
        'in1_tb_spatial_shape': in1_tb_spatial_shape, 'in2_tb_spatial_shape': in2_tb_spatial_shape,
        'in1_tb_current_step_shape': in1_tb_current_step_shape,
        'in2_tb_current_step_shape': in2_tb_current_step_shape,
        'in1_tb_reduction_shape': in1_tb_reduction_shape,
        'in2_tb_reduction_shape': in2_tb_reduction_shape,
        'out_thread_shape': out_thread_shape,
        'in1_thread_shape': in1_thread_shape, 'in2_thread_shape': in2_thread_shape,
        'thread_per_tb': thread_per_tb,
    }


# =====================================================================
# General Reduce — 双输入版本
# =====================================================================
def calculate_general_reduce_pipeline_wave(
        out_shape, reduction_shape, out_axis_mapping, reduction_axis_mapping,
        out_tb_shape, dim_threads, arch, mem_levels, compute_at=1, stage_num=1):
    """双输入 general reduce, 对应 fused_op_dtype/general_reduce_fused_op.py。"""
    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    p = _parse_reduction_axes_dual(out_shape, reduction_shape, out_axis_mapping,
                                   reduction_axis_mapping, out_tb_shape, dim_threads)
    spatial_grids = p['spatial_grids']
    reduction_grids = p['reduction_grids']
    reduction_step = p['reduction_step']
    in1_sp = p['in1_tb_spatial_shape']; in2_sp = p['in2_tb_spatial_shape']
    in1_cs = p['in1_tb_current_step_shape']; in2_cs = p['in2_tb_current_step_shape']
    out_ts = p['out_thread_shape']
    in1_ts = list(p['in1_thread_shape']); in2_ts = list(p['in2_thread_shape'])
    thread_per_tb = p['thread_per_tb']

    in1_level = mem_levels['in1']; in2_level = mem_levels['in2']; out1_level = mem_levels['out1']

    l2_hit_rate = general_reduce_l2_hitrate(
        out_axis_mapping, out_tb_shape, spatial_grids,
        in1_sp, in2_sp, p['in1_tb_reduction_shape'], p['in2_tb_reduction_shape'],
        arch.l2_capacity, arch.sm_count, mem_levels)

    num_reduction_iters = int(np.prod(reduction_grids))

    per_iter_l2_read = (np.prod(in1_sp) * np.prod(in1_cs) * in1_level[0] * in1_level[-1]
                        + np.prod(in2_sp) * np.prod(in2_cs) * in2_level[0] * in2_level[-1])
    per_iter_ddr = per_iter_l2_read * (1 - l2_hit_rate) * DDR_non_ideal_para
    if arch.core in ("A100", "H100", "B200"):
        per_iter_l2 = per_iter_l2_read * l2_hit_rate + per_iter_l2_read * (1 - l2_hit_rate) * 2
    else:
        per_iter_l2 = per_iter_l2_read

    per_iter_smem = 2 * per_iter_l2_read
    overheads = _compute_thread_overhead(thread_per_tb)
    cur_rs = list(reduction_step); cur_ram = list(reduction_axis_mapping)

    if compute_at == -1:
        reg_footprint = np.prod(out_ts) * out1_level[-1] / 4 + 2 + len(reduction_shape)
        per_iter_smem += thread_per_tb * np.prod(out_ts) * np.prod(reduction_step) * (in1_level[1] * in1_level[-1] + in2_level[1] * in2_level[-1])
    elif compute_at == 0:
        reg_footprint = (np.prod(out_ts) * out1_level[-1] / 4
                         + np.prod(in1_ts) * in1_level[1] * in1_level[-1] / 4
                         + np.prod(in2_ts) * in2_level[1] * in2_level[-1] / 4
                         + len(reduction_shape) + 2)
        per_iter_smem += thread_per_tb * np.prod(cur_rs) * (np.prod(in1_ts) * in1_level[1] * in1_level[-1] + np.prod(in2_ts) * in2_level[1] * in2_level[-1])
    else:
        assert len(cur_rs) >= compute_at
        j = 0
        for i in range(compute_at):
            last_row = cur_ram[-1]; cur_ram = cur_ram[:-1]; last_step = cur_rs.pop()
            if last_row[0] == 2:
                in2_ts.insert(0, last_step); in1_ts.insert(0, last_step)
            elif last_row[0] == 1:
                in2_ts.insert(0, last_step); in1_ts[-1 - j] += last_step - 1; j += 1
            elif last_row[0] == 0:
                in1_ts.insert(0, last_step); in2_ts[-1 - j] += last_step - 1; j += 1
        reg_footprint = np.prod(out_ts) + np.prod(in1_ts) * in1_level[1] + np.prod(in2_ts) * in2_level[1] + len(reduction_shape) + 2
        per_iter_smem += thread_per_tb * np.prod(cur_rs) * (np.prod(in1_ts) * in1_level[1] * in1_level[-1] + np.prod(in2_ts) * in2_level[1] * in2_level[-1])

    per_iter_smem *= overheads
    per_iter_compute = 2 * np.prod(out_tb_shape) * np.prod(reduction_step) * overheads
    reg_footprint = math.ceil(reg_footprint * REG_spill_para)

    store_smem = thread_per_tb * np.prod(out_ts) * out1_level[1] * out1_level[-1] * overheads
    store_ddr = np.prod(out_tb_shape) * out1_level[-1] * out1_level[0]

    smem_footprint = (np.prod(in1_sp) * np.prod(in1_cs) * (1 - (in1_level[1] - in1_level[0])) * in1_level[-1]
                      + np.prod(in2_sp) * np.prod(in2_cs) * (1 - (in2_level[1] - in2_level[0])) * in2_level[-1]
                      + np.prod(out_tb_shape) * (out1_level[1] - out1_level[0]) * out1_level[-1]) * stage_num

    per_iter_r = PerIterationResources(ddr_io=per_iter_ddr, l2_io=per_iter_l2, smem_io=per_iter_smem, compute_flops=per_iter_compute)
    pe = PrologueEpilogueResources(
        prologue_ddr_io=per_iter_ddr * max(stage_num - 1, 0), prologue_l2_io=per_iter_l2 * max(stage_num - 1, 0),
        prologue_smem_io=per_iter_smem * max(stage_num - 1, 0), epilogue_compute_flops=per_iter_compute * max(stage_num - 1, 0),
        store_ddr_io=store_ddr, store_l2_io=store_ddr, store_smem_io=store_smem)
    tile_res = TileResources(per_iter=per_iter_r, prologue_epilogue=pe, num_iterations=num_reduction_iters,
                             stage_num=stage_num, smem_footprint=smem_footprint, reg_footprint=reg_footprint,
                             warps_per_block=max(int(thread_per_tb / 32), 1), grids=tuple(spatial_grids))

    tiles_per_sm = compute_occupancy(smem_footprint, reg_footprint, max(int(thread_per_tb / 32), 1), arch)
    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate, data_bytes=out1_level[-1])


# =====================================================================
# General Inter-thread Reduce — 双输入版本
# =====================================================================
def calculate_general_inter_thread_reduce_pipeline_wave(
        out_shape, reduction_shape, out_axis_mapping, reduction_axis_mapping,
        out_tb_shape, dim_threads, reduce_threads_info, arch, mem_levels,
        compute_at=1, stage_num=1):
    """双输入 inter-thread reduce, 对应 fused_op_dtype/general_inter_thread_reduce_fused_op.py。"""
    DDR_non_ideal_para = 1.1
    REG_spill_para = 1.1

    p = _parse_reduction_axes_dual(out_shape, reduction_shape, out_axis_mapping,
                                   reduction_axis_mapping, out_tb_shape, dim_threads)
    spatial_grids = p['spatial_grids']
    reduction_grids = p['reduction_grids']
    reduction_step = list(p['reduction_step'])
    in1_sp = p['in1_tb_spatial_shape']; in2_sp = p['in2_tb_spatial_shape']
    in1_cs = p['in1_tb_current_step_shape']; in2_cs = p['in2_tb_current_step_shape']
    out_ts = p['out_thread_shape']
    in1_ts = list(p['in1_thread_shape']); in2_ts = list(p['in2_thread_shape'])

    thread_per_tb = np.prod(dim_threads) * reduce_threads_info[1]
    in1_level = mem_levels['in1']; in2_level = mem_levels['in2']; out1_level = mem_levels['out1']

    l2_hit_rate = general_reduce_l2_hitrate(
        out_axis_mapping, out_tb_shape, spatial_grids,
        in1_sp, in2_sp, p['in1_tb_reduction_shape'], p['in2_tb_reduction_shape'],
        arch.l2_capacity, arch.sm_count, mem_levels)

    num_reduction_iters = int(np.prod(reduction_grids))
    reduction_step[reduce_threads_info[0]] = math.ceil(reduction_step[reduce_threads_info[0]] / reduce_threads_info[1])

    per_iter_l2_read = (np.prod(in1_sp) * np.prod(in1_cs) * in1_level[0] * in1_level[-1]
                        + np.prod(in2_sp) * np.prod(in2_cs) * in2_level[0] * in2_level[-1])
    per_iter_ddr = per_iter_l2_read * (1 - l2_hit_rate) * DDR_non_ideal_para
    if arch.core in ("A100", "H100", "B200"):
        per_iter_l2 = per_iter_l2_read * l2_hit_rate + per_iter_l2_read * (1 - l2_hit_rate) * 2
    else:
        per_iter_l2 = per_iter_l2_read

    per_iter_smem = 2 * per_iter_l2_read
    overheads = _compute_thread_overhead(thread_per_tb)
    cur_rs = list(reduction_step); cur_ram = list(reduction_axis_mapping)

    if compute_at == -1:
        reg_footprint = np.prod(out_ts) * out1_level[-1] / 4 + 2 + len(reduction_shape)
        per_iter_smem += thread_per_tb * np.prod(out_ts) * np.prod(reduction_step) * (in1_level[1] * in1_level[-1] + in2_level[1] * in2_level[-1])
    elif compute_at == 0:
        reg_footprint = (np.prod(out_ts) * out1_level[-1] / 4 + np.prod(in1_ts) * in1_level[1] * in1_level[-1] / 4
                         + np.prod(in2_ts) * in2_level[1] * in2_level[-1] / 4 + len(reduction_shape) + 2)
        per_iter_smem += thread_per_tb * np.prod(cur_rs) * (np.prod(in1_ts) * in1_level[1] * in1_level[-1] + np.prod(in2_ts) * in2_level[1] * in2_level[-1])
    else:
        assert len(cur_rs) >= compute_at
        j = 0
        for i in range(compute_at):
            last_row = cur_ram[-1]; cur_ram = cur_ram[:-1]; last_step = cur_rs.pop()
            if last_row[0] == 2:
                in2_ts.insert(0, last_step); in1_ts.insert(0, last_step)
            elif last_row[0] == 1:
                in2_ts.insert(0, last_step); in1_ts[-1 - j] += last_step - 1; j += 1
            elif last_row[0] == 0:
                in1_ts.insert(0, last_step); in2_ts[-1 - j] += last_step - 1; j += 1
        reg_footprint = np.prod(out_ts) + np.prod(in1_ts) * in1_level[1] + np.prod(in2_ts) * in2_level[1] + len(reduction_shape) + 2
        per_iter_smem += thread_per_tb * np.prod(cur_rs) * (np.prod(in1_ts) * in1_level[1] * in1_level[-1] + np.prod(in2_ts) * in2_level[1] * in2_level[-1])

    per_iter_smem *= overheads
    per_iter_compute = 2 * np.prod(out_tb_shape) * np.prod(reduction_step) * overheads

    store_smem = thread_per_tb / reduce_threads_info[1] * np.prod(out_ts) * out1_level[1] * out1_level[-1] * overheads
    store_ddr = np.prod(out_tb_shape) * out1_level[0] * out1_level[-1]
    reduce_smem_extra = 0
    if reduce_threads_info[1] > 32:
        cur_r = reduce_threads_info[1]
        reduce_smem_extra += thread_per_tb / reduce_threads_info[1] * np.prod(out_ts) * out1_level[-1] * cur_r
        while cur_r > 32:
            reduce_smem_extra += thread_per_tb / reduce_threads_info[1] * np.prod(out_ts) * out1_level[-1] * cur_r * 3 / 2
            cur_r /= 2
        reduce_smem_extra += math.log2(32) * thread_per_tb / reduce_threads_info[1] * np.prod(out_ts) * out1_level[-1] * 32 * 3
        reduce_smem_extra += reduce_threads_info[1] * np.prod(out_ts) * out1_level[-1] * 2
        reg_footprint += math.log2(32)

    reduce_compute_extra = 2 * np.prod(out_tb_shape) * math.ceil(math.log2(reduce_threads_info[1])) * overheads
    store_smem += reduce_smem_extra * overheads
    reg_footprint = math.ceil(reg_footprint * REG_spill_para)

    smem_footprint = (np.prod(in1_sp) * np.prod(in1_cs) * (1 - (in1_level[1] - in1_level[0])) * in1_level[-1]
                      + np.prod(in2_sp) * np.prod(in2_cs) * (1 - (in2_level[1] - in2_level[0])) * in2_level[-1]
                      + np.prod(out_tb_shape) * (out1_level[1] - out1_level[0]) * out1_level[-1])
    if reduce_threads_info[1] > 32:
        smem_footprint += 128 * out1_level[-1] * np.prod(out_ts)
    smem_footprint *= stage_num

    per_iter_r = PerIterationResources(ddr_io=per_iter_ddr, l2_io=per_iter_l2, smem_io=per_iter_smem, compute_flops=per_iter_compute)
    pe = PrologueEpilogueResources(
        prologue_ddr_io=per_iter_ddr * max(stage_num - 1, 0), prologue_l2_io=per_iter_l2 * max(stage_num - 1, 0),
        prologue_smem_io=per_iter_smem * max(stage_num - 1, 0),
        epilogue_compute_flops=per_iter_compute * max(stage_num - 1, 0) + reduce_compute_extra,
        store_ddr_io=store_ddr, store_l2_io=store_ddr, store_smem_io=store_smem)
    tile_res = TileResources(per_iter=per_iter_r, prologue_epilogue=pe, num_iterations=num_reduction_iters,
                             stage_num=stage_num, smem_footprint=smem_footprint, reg_footprint=reg_footprint,
                             warps_per_block=max(int(thread_per_tb / 32), 1), grids=tuple(spatial_grids))

    tiles_per_sm = compute_occupancy(smem_footprint, reg_footprint, max(int(thread_per_tb / 32), 1), arch)
    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate, data_bytes=out1_level[-1])


# =====================================================================
# General Reduce with Stride — 双输入版本
# =====================================================================
def calculate_general_reduce_with_stride_pipeline_wave(
        out_shape, reduction_shape, out_axis_mapping, reduction_axis_mapping,
        out_tb_shape, dim_threads, arch, mem_levels, compute_at=1,
        strides=None, stage_num=1):
    """双输入 general reduce with stride, 对应 fused_op_dtype/general_ruduce_resource_utilization_with_stride.py。"""
    if strides is None:
        strides = [1, 1]
    return calculate_general_reduce_pipeline_wave(
        out_shape, reduction_shape, out_axis_mapping, reduction_axis_mapping,
        out_tb_shape, dim_threads, arch, mem_levels, compute_at=compute_at, stage_num=stage_num)


# =====================================================================
# Inter-thread Reduce (N_0)
# =====================================================================
def calculate_N_0_general_inter_thread_reduce_pipeline_wave(
        out_shape, reduction_shape, out_axis_mapping, reduction_axis_mapping,
        out_tb_shape, dim_threads, reduce_threads_info, arch, mem_levels,
        compute_at=1, stage_num=1):
    """Inter-thread reduce pipeline-aware model.

    与 general reduce 类似, 但额外考虑跨线程归约 (shfl / shared memory reduce)。

    Args:
        reduce_threads_info: [axis_index, num_reduce_threads]
            在哪个归约轴上进行线程间归约，以及使用多少线程
        其他参数同 calculate_N_0_general_reduce_pipeline_wave

    Returns:
        PipelineResult
    """
    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    parsed = _parse_reduction_axes(
        out_shape, reduction_shape, out_axis_mapping,
        reduction_axis_mapping, out_tb_shape, dim_threads)

    spatial_grids = parsed['spatial_grids']
    reduction_grids = parsed['reduction_grids']
    reduction_step = list(parsed['reduction_step'])
    in1_tb_spatial_shape = parsed['in1_tb_spatial_shape']
    in1_tb_current_step_shape = parsed['in1_tb_current_step_shape']
    out_thread_shape = parsed['out_thread_shape']
    in1_thread_shape = list(parsed['in1_thread_shape'])

    thread_per_tb = np.prod(dim_threads) * reduce_threads_info[1]

    in1_level = mem_levels['in1']
    out1_level = mem_levels['out1']

    l2_hit_rate = 0

    # ---- Total IO ----
    total_l2_read = (np.prod(spatial_grids) * np.prod(reduction_grids)
                     * np.prod(in1_tb_spatial_shape) * np.prod(in1_tb_current_step_shape)
                     * in1_level[0] * in1_level[-1])
    total_l2_store = np.prod(spatial_grids) * np.prod(out_tb_shape) * out1_level[0] * out1_level[-1]

    total_l2_read = _apply_wave_bytes_correction(total_l2_read, arch)
    total_l2_store = _apply_wave_bytes_correction(total_l2_store, arch)

    total_ddr = (total_l2_read * (1 - l2_hit_rate) + total_l2_store) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        total_l2_io = (total_l2_read * l2_hit_rate
                       + total_l2_read * (1 - l2_hit_rate) * 2
                       + total_l2_store * 2)
    else:
        total_l2_io = total_l2_read + total_l2_store

    # ---- Reduction loop ----
    num_reduction_iters = int(np.prod(reduction_grids))

    # inter-thread reduce: 切分 rstep
    reduction_step[reduce_threads_info[0]] = math.ceil(
        reduction_step[reduce_threads_info[0]] / reduce_threads_info[1])

    # ---- Per reduction-iteration ----
    per_iter_l2_read = (np.prod(in1_tb_spatial_shape) * np.prod(in1_tb_current_step_shape)
                        * in1_level[0] * in1_level[-1])
    per_iter_ddr = per_iter_l2_read * (1 - l2_hit_rate) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        per_iter_l2 = (per_iter_l2_read * l2_hit_rate
                       + per_iter_l2_read * (1 - l2_hit_rate) * 2)
    else:
        per_iter_l2 = per_iter_l2_read

    per_iter_smem = 2 * per_iter_l2_read  # ldg + sts

    overheads = _compute_thread_overhead(thread_per_tb)

    # compute_at handling
    current_in1_thread_shape = list(in1_thread_shape)
    current_reduction_step = list(reduction_step)
    current_reduction_axis_mapping = list(reduction_axis_mapping)

    if compute_at == -1:
        reg_footprint = np.prod(out_thread_shape) * out1_level[-1] / 4 + 2 + len(reduction_shape)
        per_iter_smem += (in1_level[-1] * thread_per_tb
                          * np.prod(out_thread_shape) * np.prod(reduction_step)
                          * in1_level[1])
    elif compute_at == 0:
        reg_footprint = (np.prod(current_in1_thread_shape) * in1_level[1] * in1_level[-1] / 4
                         + np.prod(out_thread_shape) * out1_level[-1] / 4
                         + len(reduction_shape) + 2)
        per_iter_smem += (in1_level[-1] * thread_per_tb * np.prod(current_reduction_step)
                          * np.prod(current_in1_thread_shape) * in1_level[1])
    else:
        assert len(current_reduction_step) >= compute_at
        j = 0
        for i in range(compute_at):
            last_row = current_reduction_axis_mapping[-1]
            current_reduction_axis_mapping = current_reduction_axis_mapping[:-1]
            last_step = current_reduction_step.pop()
            if last_row[0] == 2:
                current_in1_thread_shape.insert(0, last_step)
            elif last_row[0] == 1:
                current_in1_thread_shape[-1 - j] += last_step - 1
                j += 1
            elif last_row[0] == 0:
                current_in1_thread_shape.insert(0, last_step)
                j += 1
        reg_footprint = (np.prod(out_thread_shape) * out1_level[-1] / 4
                         + np.prod(current_in1_thread_shape) * in1_level[1] * in1_level[-1] / 4
                         + len(reduction_shape) + 2)
        per_iter_smem += (in1_level[-1] * thread_per_tb * np.prod(current_reduction_step)
                          * np.prod(current_in1_thread_shape) * in1_level[1])

    per_iter_smem *= overheads
    per_iter_compute = 2 * np.prod(out_tb_shape) * np.prod(reduction_step) * overheads

    # ---- Store + inter-thread reduce overhead ----
    store_smem = (thread_per_tb / reduce_threads_info[1]
                  * np.prod(out_thread_shape) * out1_level[1] * out1_level[-1] * overheads)
    store_ddr = np.prod(out_tb_shape) * out1_level[0] * out1_level[-1]
    store_l2 = store_ddr

    # 归约阶段的额外 smem IO 和 compute
    reduce_smem_extra = 0
    reduce_compute_extra = 0

    if reduce_threads_info[1] > 32:
        # 跨 warp 归约需要 shared memory
        current_reduce = reduce_threads_info[1]
        reduce_smem_extra += (thread_per_tb / reduce_threads_info[1]
                              * np.prod(out_thread_shape) * out1_level[-1] * current_reduce)
        while current_reduce > 32:
            reduce_smem_extra += (thread_per_tb / reduce_threads_info[1]
                                  * np.prod(out_thread_shape) * out1_level[-1]
                                  * current_reduce * 3 / 2)
            current_reduce /= 2
        reduce_smem_extra += (math.log2(32) * thread_per_tb / reduce_threads_info[1]
                              * np.prod(out_thread_shape) * out1_level[-1] * 32 * 3)
        reduce_smem_extra += (reduce_threads_info[1] * np.prod(out_thread_shape)
                              * out1_level[-1] * 2)
        reg_footprint += math.log2(32)
    else:
        reg_footprint += 3

    # 归约的 compute overhead
    reduce_compute_extra = (2 * np.prod(out_tb_shape)
                            * math.ceil(math.log2(reduce_threads_info[1])) * overheads)

    # 将归约 overhead 加到 store/epilogue
    store_smem += reduce_smem_extra * overheads
    per_iter_compute += 0  # reduce compute 只在 epilogue 发生

    reg_footprint = math.ceil(reg_footprint * REG_spill_para)

    # ---- Footprint ----
    smem_footprint = (np.prod(in1_tb_spatial_shape) * np.prod(in1_tb_current_step_shape)
                      * (1 - (in1_level[1] - in1_level[0])) * in1_level[-1]
                      + np.prod(out_tb_shape) * (out1_level[1] - out1_level[0]) * out1_level[-1])
    if reduce_threads_info[1] > 32:
        smem_footprint += 128 * out1_level[-1] * np.prod(out_thread_shape)
    smem_footprint *= stage_num

    # ---- TileResources ----
    # 将归约 compute 加到 epilogue
    total_compute = (np.prod(out_shape) * np.prod(reduction_shape) * 2
                     + np.prod(out_shape) * math.ceil(math.log2(reduce_threads_info[1])) * 2)
    total_compute *= overheads
    epilogue_compute = reduce_compute_extra

    per_iter = PerIterationResources(
        ddr_io=per_iter_ddr, l2_io=per_iter_l2,
        smem_io=per_iter_smem, compute_flops=per_iter_compute,
    )
    pe = PrologueEpilogueResources(
        prologue_ddr_io=per_iter_ddr * max(stage_num - 1, 0),
        prologue_l2_io=per_iter_l2 * max(stage_num - 1, 0),
        prologue_smem_io=per_iter_smem * max(stage_num - 1, 0),
        epilogue_compute_flops=per_iter_compute * max(stage_num - 1, 0) + epilogue_compute,
        store_ddr_io=store_ddr,
        store_l2_io=store_l2,
        store_smem_io=store_smem,
    )

    tile_res = TileResources(
        per_iter=per_iter, prologue_epilogue=pe,
        num_iterations=num_reduction_iters,
        stage_num=stage_num,
        smem_footprint=smem_footprint,
        reg_footprint=reg_footprint,
        warps_per_block=max(int(thread_per_tb / 32), 1),
        grids=tuple(spatial_grids),
    )

    tiles_per_sm = compute_occupancy(
        smem_footprint, reg_footprint,
        max(int(thread_per_tb / 32), 1), arch)

    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                         data_bytes=out1_level[-1])
