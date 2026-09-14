"""Pipeline-aware element-wise operation performance model.

Element-wise operations typically use stage_num=1 (no inner loop), reducing
the pipeline model to a roofline model. The main improvements come from
wave head/tail and occupancy analysis.

Includes N_0, N_1, and N_N variants.
"""
import numpy as np
import math
import logging

from tilesight.arch import uses_dram_wave_quantization

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


def _apply_wave_bytes_correction(io_val, arch):
    """Apply ddr_wave_bytes quantization when the architecture opts in."""
    if io_val > 0 and uses_dram_wave_quantization(arch):
        waves = io_val / arch.ddr_wave_bytes
        if waves > 0:
            return io_val * math.ceil(waves) / waves
    return io_val


def _compute_thread_overhead(thread_per_tb):
    """Compute the overhead caused by insufficient threads."""
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
    """Common procedure for building a PipelineResult from TileResources."""
    # Pipeline latency
    sm_latency, pipeline_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_res, tiles_per_sm, arch, data_bytes=data_bytes)
    per_tile_latency, _ = compute_pipeline_tile_latency(
        tile_res, arch, data_bytes=data_bytes)

    # Wave adjustment (pass tile_res to accurately recalculate active_sms bandwidth for the tail wave)
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


# =====================================================================
# N_0 Element-wise: single input, single output, unchanged shape
# =====================================================================
def calculate_N_0_elementwise_pipeline_wave(op_shape, tb_shape, dim_threads,
                                            arch, mem_levels, num_ops=1):
    """N_0 element-wise pipeline-aware model.

    Args:
        op_shape: Multidimensional operation shape.
        tb_shape: Thread-block tile shape.
        dim_threads: Thread count along each dimension.
        arch: Architecture object.
        mem_levels: {'in1': [...], 'out1': [...]}
        num_ops: Number of fused operations.

    Returns:
        PipelineResult
    """
    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    thread_per_tb = np.prod(dim_threads)
    thread_tiles = [math.ceil(tb_dim / thread_dim)
                    for tb_dim, thread_dim in zip(tb_shape, dim_threads)]
    grids = [math.ceil(dim / tb_dim) for dim, tb_dim in zip(op_shape, tb_shape)]

    in1_level = mem_levels['in1']
    out1_level = mem_levels['out1']

    l2_hit_rate = 0

    # ---- IO calculation ----
    total_elements = np.prod(op_shape)
    l2_read_io = total_elements * in1_level[0] * in1_level[-1]
    l2_store_io = total_elements * out1_level[-1] * out1_level[0]

    l2_read_io = _apply_wave_bytes_correction(l2_read_io, arch)
    l2_store_io = _apply_wave_bytes_correction(l2_store_io, arch)

    ddr_io = (l2_read_io * (1 - l2_hit_rate) + l2_store_io) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        l2_io = l2_read_io * l2_hit_rate + l2_read_io * (1 - l2_hit_rate) * 2 + l2_store_io * 2
    else:
        l2_io = l2_read_io + l2_store_io

    # ---- Per tile ----
    total_tiles = int(np.prod(grids))
    per_tile_elements = np.prod(tb_shape)

    per_tile_ddr = ddr_io / max(total_tiles, 1)
    per_tile_l2 = l2_io / max(total_tiles, 1)
    per_tile_smem = (per_tile_elements * in1_level[1] * in1_level[-1]
                     + per_tile_elements * out1_level[1] * out1_level[-1])

    overheads = _compute_thread_overhead(thread_per_tb)
    per_tile_compute = 2 * per_tile_elements * overheads * num_ops
    per_tile_smem *= overheads

    # ---- Footprint ----
    smem_footprint = per_tile_elements * (out1_level[1] - out1_level[0]) * out1_level[-1]
    reg_footprint = math.ceil(np.prod(thread_tiles)
                              * (in1_level[1] * in1_level[-1] / 4 + out1_level[1] / 4)
                              * REG_spill_para)

    # ---- TileResources (element-wise: 1 iteration, stage=1) ----
    per_iter = PerIterationResources(
        ddr_io=per_tile_ddr, l2_io=per_tile_l2,
        smem_io=per_tile_smem, compute_flops=per_tile_compute,
    )
    pe = PrologueEpilogueResources(
        store_ddr_io=0, store_l2_io=0, store_smem_io=0,
    )
    tile_res = TileResources(
        per_iter=per_iter, prologue_epilogue=pe,
        num_iterations=1, stage_num=1,
        smem_footprint=smem_footprint, reg_footprint=reg_footprint,
        warps_per_block=max(int(thread_per_tb / 32), 1),
        grids=tuple(grids),
    )

    tiles_per_sm = compute_occupancy(
        smem_footprint, reg_footprint,
        max(int(thread_per_tb / 32), 1), arch)

    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                         data_bytes=out1_level[-1])


# =====================================================================
# N_1 Element-wise: two inputs (broadcast), single output
# =====================================================================
def calculate_N_1_elementwise_pipeline_wave(in1_shape, in2_shape, in1_tb_shape,
                                            dim_threads, arch, mem_levels,
                                            num_ops=1):
    """N_1 element-wise pipeline-aware model.

    Args:
        in1_shape: Shape of input 1 (large tensor).
        in2_shape: Shape of input 2 (broadcast tensor).
        in1_tb_shape: Tile shape for input 1.
        dim_threads: Thread count along each dimension.
        arch: Architecture object.
        mem_levels: {'in1': [...], 'in2': [...], 'out1': [...]}
        num_ops: Number of fused operations.

    Returns:
        PipelineResult
    """
    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    thread_per_tb = np.prod(dim_threads)
    in1_thread_shape = [math.ceil(tb_dim / thread_dim)
                        for tb_dim, thread_dim in zip(in1_tb_shape, dim_threads)]
    in2_tb_shape = [math.ceil(float(in2_dim / in1_dim * tb_dim))
                    for in1_dim, in2_dim, tb_dim in zip(in1_shape, in2_shape, in1_tb_shape)]
    in2_thread_shape = [math.ceil(float(in2_dim / in1_dim * tb_dim))
                        for in1_dim, in2_dim, tb_dim
                        in zip(in1_tb_shape, in2_tb_shape, in1_thread_shape)]
    grids = [math.ceil(dim / tb_dim) for dim, tb_dim in zip(in1_shape, in1_tb_shape)]

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    # ---- IO calculation ----
    l2_read_io = (np.prod(in1_shape) * in1_level[0] * in1_level[-1]
                  + np.prod(grids) * np.prod(in2_tb_shape) * in2_level[0] * in2_level[-1])
    l2_store_io = np.prod(in1_shape) * out1_level[-1] * out1_level[0]
    ddr_read_io = (np.prod(in1_shape) * in1_level[0] * in1_level[-1]
                   + np.prod(in2_shape) * in2_level[0] * in2_level[-1])
    ddr_store_io = np.prod(in1_shape) * out1_level[-1] * out1_level[0]

    l2_read_io = _apply_wave_bytes_correction(l2_read_io, arch)
    l2_store_io = _apply_wave_bytes_correction(l2_store_io, arch)
    ddr_read_io = _apply_wave_bytes_correction(ddr_read_io, arch)
    ddr_store_io = _apply_wave_bytes_correction(ddr_store_io, arch)

    l2_hit_rate = max((l2_read_io - ddr_read_io) / max(l2_read_io, 1), 0)
    ddr_io = (ddr_read_io + ddr_store_io) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        l2_io = l2_read_io * l2_hit_rate + l2_read_io * (1 - l2_hit_rate) * 2 + l2_store_io * 2
    else:
        l2_io = l2_read_io + l2_store_io

    # ---- Per tile ----
    total_tiles = int(np.prod(grids))
    per_tile_ddr = ddr_io / max(total_tiles, 1)
    per_tile_l2 = l2_io / max(total_tiles, 1)

    per_tile_smem = (np.prod(in1_tb_shape) * in1_level[1] * in1_level[-1]
                     + np.prod(grids) * np.prod(in2_tb_shape) * in2_level[0] * in2_level[-1] * 2 / max(total_tiles, 1)
                     + thread_per_tb * np.prod(in2_thread_shape) * in2_level[1] * in2_level[-1]
                     + np.prod(in1_tb_shape) * out1_level[1] * out1_level[-1])

    overheads = _compute_thread_overhead(thread_per_tb)
    per_tile_compute = 2 * np.prod(in1_tb_shape) * overheads * num_ops
    per_tile_smem *= overheads

    # ---- Footprint ----
    smem_footprint = (np.prod(in2_tb_shape) * (1 - (in2_level[1] - in2_level[0])) * in2_level[-1]
                      + np.prod(in1_tb_shape) * (out1_level[1] - out1_level[0]) * out1_level[-1])
    reg_footprint = math.ceil(
        (np.prod(in1_thread_shape) * (in1_level[1] * in1_level[-1] / 4 + out1_level[-1] / 4)
         + np.prod(in2_thread_shape) * in2_level[1] * in2_level[-1] / 4)
        * REG_spill_para)

    # ---- TileResources ----
    per_iter = PerIterationResources(
        ddr_io=per_tile_ddr, l2_io=per_tile_l2,
        smem_io=per_tile_smem, compute_flops=per_tile_compute,
    )
    pe = PrologueEpilogueResources()
    tile_res = TileResources(
        per_iter=per_iter, prologue_epilogue=pe,
        num_iterations=1, stage_num=1,
        smem_footprint=smem_footprint, reg_footprint=reg_footprint,
        warps_per_block=max(int(thread_per_tb / 32), 1),
        grids=tuple(grids),
    )

    tiles_per_sm = compute_occupancy(
        smem_footprint, reg_footprint,
        max(int(thread_per_tb / 32), 1), arch)

    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                         data_bytes=out1_level[-1])


# =====================================================================
# N_N Element-wise: two inputs of the same shape, single output
# =====================================================================
def calculate_N_N_elementwise_pipeline_wave(op_shape, tb_shape, dim_threads,
                                            arch, mem_levels, num_ops=1):
    """N_N element-wise pipeline-aware model.

    Args:
        op_shape: Operation shape.
        tb_shape: Thread-block tile shape.
        dim_threads: Thread count along each dimension.
        arch: Architecture object.
        mem_levels: {'in1': [...], 'in2': [...], 'out1': [...]}
        num_ops: Number of fused operations.

    Returns:
        PipelineResult
    """
    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    thread_per_tb = np.prod(dim_threads)
    thread_tiles = [math.ceil(tb_dim / thread_dim)
                    for tb_dim, thread_dim in zip(tb_shape, dim_threads)]
    grids = [math.ceil(dim / tb_dim) for dim, tb_dim in zip(op_shape, tb_shape)]

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    l2_hit_rate = 0
    total_elements = np.prod(op_shape)

    # ---- IO ----
    l2_read_io = total_elements * in1_level[0] * in1_level[-1] + total_elements * in2_level[0] * in2_level[-1]
    l2_store_io = total_elements * out1_level[-1] * out1_level[0]

    l2_read_io = _apply_wave_bytes_correction(l2_read_io, arch)
    l2_store_io = _apply_wave_bytes_correction(l2_store_io, arch)

    ddr_io = (l2_read_io * (1 - l2_hit_rate) + l2_store_io) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        l2_io = l2_read_io * l2_hit_rate + l2_read_io * (1 - l2_hit_rate) * 2 + l2_store_io * 2
    else:
        l2_io = l2_read_io + l2_store_io

    # ---- Per tile ----
    total_tiles = int(np.prod(grids))
    per_tile_ddr = ddr_io / max(total_tiles, 1)
    per_tile_l2 = l2_io / max(total_tiles, 1)

    per_tile_elements = np.prod(tb_shape)
    per_tile_smem = (per_tile_elements * in1_level[1] * in1_level[-1]
                     + per_tile_elements * in2_level[1] * in2_level[-1]
                     + per_tile_elements * out1_level[1] * out1_level[-1])

    overheads = _compute_thread_overhead(thread_per_tb)
    per_tile_compute = 2 * per_tile_elements * overheads * num_ops
    per_tile_smem *= overheads

    # ---- Footprint ----
    smem_footprint = per_tile_elements * (out1_level[1] - out1_level[0]) * out1_level[-1]
    reg_footprint = math.ceil(
        np.prod(thread_tiles)
        * (out1_level[-1] / 4 + in1_level[1] * in1_level[-1] / 4
           + in2_level[1] * in2_level[-1] / 4)
        * REG_spill_para)

    # ---- TileResources ----
    per_iter = PerIterationResources(
        ddr_io=per_tile_ddr, l2_io=per_tile_l2,
        smem_io=per_tile_smem, compute_flops=per_tile_compute,
    )
    pe = PrologueEpilogueResources()
    tile_res = TileResources(
        per_iter=per_iter, prologue_epilogue=pe,
        num_iterations=1, stage_num=1,
        smem_footprint=smem_footprint, reg_footprint=reg_footprint,
        warps_per_block=max(int(thread_per_tb / 32), 1),
        grids=tuple(grids),
    )

    tiles_per_sm = compute_occupancy(
        smem_footprint, reg_footprint,
        max(int(thread_per_tb / 32), 1), arch)

    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                         data_bytes=out1_level[-1])
