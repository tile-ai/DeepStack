"""Pipeline-aware convolution performance models.

包含三种 conv:
- conv_implicit_gemm: 将 conv 映射为 GEMM (基础版)
- conv_implicit_gemm_sdp: 同上 + stride/dilation/padding
- conv_nchw: NCHW layout, 4D spatial grid, 非 GEMM 映射
"""
from ..util import *
import numpy as np
import math
import logging

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


def _warp_overhead(active_warp_per_tb):
    """conv implicit gemm 的 warp overhead 计算。"""
    if active_warp_per_tb <= 4:
        return 4 / active_warp_per_tb
    elif active_warp_per_tb <= 8:
        return 8 / active_warp_per_tb
    elif active_warp_per_tb <= 12:
        return 12 / active_warp_per_tb
    elif active_warp_per_tb <= 16:
        return 16 / active_warp_per_tb
    else:
        return 1


def _thread_overhead(thread_per_tb):
    """element-wise / reduce 的 thread overhead 计算。"""
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


def _build_conv_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                       total_compute, total_ddr, batch, data_bytes):
    """Conv 通用结果构建。"""
    sm_latency, pipeline_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_res, tiles_per_sm, arch, data_bytes=data_bytes)

    total_tiles = int(np.prod(tile_res.grids))
    total_latency, wave_info = compute_wave_adjusted_latency(
        sm_latency, tiles_per_sm, total_tiles, arch,
        pipeline_detail=pipeline_detail,
        tile_res=tile_res, data_bytes=data_bytes)
    total_latency *= batch

    actual_active_sms = min(total_tiles, arch.sm_count)
    actual_tps = 1 if total_tiles <= arch.sm_count else min(math.ceil(total_tiles / arch.sm_count), tiles_per_sm)
    per_tile_latency, pipeline_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_res, actual_tps, arch, data_bytes=data_bytes,
        active_sms=actual_active_sms)

    ddr_time_rf = total_ddr * batch / arch.ddr_bandwidth if arch.ddr_bandwidth > 0 else 0
    compute_time_rf = resources_to_times(0, 0, 0, total_compute * batch, arch, data_bytes)[4]
    ddr_util = ddr_time_rf / total_latency if total_latency > 0 else 0
    compute_util = compute_time_rf / total_latency if total_latency > 0 else 0

    return PipelineResult(
        per_tile_latency=per_tile_latency, total_latency=total_latency,
        ddr_util=ddr_util, l2_hit_rate=l2_hit_rate, compute_util=compute_util,
        smem_footprint=tile_res.smem_footprint, reg_footprint=tile_res.reg_footprint,
        tiles_per_sm=tiles_per_sm, waves=wave_info['waves_float'],
        pipeline_detail=pipeline_detail,
    )


# =====================================================================
# Conv Implicit GEMM (基础版)
# =====================================================================
def calculate_conv_implicit_gemm_pipeline_wave(
        op_shape, tb_shape, wp_shape, bytes_per_num, stage_num, arch,
        mem_levels, batch=1):
    """Conv implicit GEMM pipeline-aware model.

    对应 fused_op_dtype/conv_implicit_gemm_fused_op.py。
    将 conv 映射为 m=N*H*W, n=F, k=KH*KW*C 的 GEMM。
    """
    conv_n, conv_f, conv_h, conv_w, conv_c, conv_kh, conv_kw = op_shape
    tb_m, tb_n, tb_k = tb_shape
    wp_m, wp_n, wp_k = wp_shape

    m = conv_n * conv_h * conv_w
    n = conv_f
    k = conv_kh * conv_kw * conv_c

    DDR_non_ideal_para = 1.1
    REG_spill_para = 1.1

    gridM = math.ceil(m / tb_m)
    gridN = math.ceil(n / tb_n)
    gridK = math.ceil(k / tb_k)

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    compute_flops = 2 * m * n * k
    l2_hit_rate = implicit_gemm_l2_hitrate_reuse_distance(
        m, n, k, tb_m, tb_n, tb_k,
        arch.l2_capacity, arch.sm_count, bytes_per_num,
        conv_c, conv_kh, conv_kw)

    active_warp_per_tb = (tb_m / wp_m) * (tb_n / wp_n)
    overheads = _warp_overhead(active_warp_per_tb)
    compute_flops *= overheads

    # ---- Per-K-iter ----
    l2_load_per_iter = tb_k * (tb_m * in1_level[0] + tb_n * in2_level[0]) * bytes_per_num
    ddr_load_per_iter = l2_load_per_iter * (1 - l2_hit_rate) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        l2_io_per_iter = l2_load_per_iter * l2_hit_rate + l2_load_per_iter * (1 - l2_hit_rate) * 2
    else:
        l2_io_per_iter = l2_load_per_iter

    compute_per_iter = tb_m * tb_n * tb_k * 2 * overheads

    if arch.core in ("A100", "H100", "B200"):
        ldgsts = 0.5 * tb_k * (tb_m * in1_level[0] + tb_n * in2_level[0]) * bytes_per_num
        shared_load = (tb_k * (tb_m + tb_n) * bytes_per_num * active_warp_per_tb
                       * (wp_m * in1_level[1] + wp_n * in2_level[1]) / max(tb_m + tb_n, 1))
        smem_per_iter = (ldgsts + shared_load) * overheads
    else:
        store_shared = tb_k * (tb_m * in1_level[0] + tb_n * in2_level[0]) * bytes_per_num
        shared_load = (tb_k * (tb_m + tb_n) * bytes_per_num * active_warp_per_tb
                       * (wp_m * in1_level[1] + wp_n * in2_level[1]) / max(tb_m + tb_n, 1))
        smem_per_iter = (store_shared + shared_load) * overheads

    # ---- Store ----
    store_l2_io = tb_m * tb_n * bytes_per_num * out1_level[0]
    store_ddr_io = store_l2_io
    if arch.core in ("A100", "H100", "B200"):
        store_l2_io_adj = store_l2_io * 2
    else:
        store_l2_io_adj = store_l2_io
    store_smem_io = tb_m * tb_n * bytes_per_num * out1_level[1]

    # ---- Footprint ----
    smem_footprint = (tb_m + tb_n) * tb_k * bytes_per_num * stage_num

    if arch.core in ("A100", "H100", "B200"):
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / bytes_per_num))
                         + math.ceil(wp_m * 32 / 32 / (4 / bytes_per_num))
                         + math.ceil(wp_n * 32 / 32 / (4 / bytes_per_num)))
        reg_footprint = math.ceil(reg_footprint * REG_spill_para)
    else:
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / bytes_per_num))
                         + math.ceil(wp_m * 32 / 32 / (4 / bytes_per_num))
                         + math.ceil(wp_n * 32 / 32 / (4 / bytes_per_num)))
        reg_footprint = math.ceil(reg_footprint * REG_spill_para)

    # ---- TileResources ----
    warps_per_block = int(active_warp_per_tb)
    per_iter = PerIterationResources(ddr_io=ddr_load_per_iter, l2_io=l2_io_per_iter,
                                     smem_io=smem_per_iter, compute_flops=compute_per_iter)
    pe = PrologueEpilogueResources(
        prologue_ddr_io=ddr_load_per_iter * max(stage_num - 1, 0),
        prologue_l2_io=l2_io_per_iter * max(stage_num - 1, 0),
        prologue_smem_io=smem_per_iter * max(stage_num - 1, 0),
        epilogue_compute_flops=compute_per_iter * max(stage_num - 1, 0),
        store_ddr_io=store_ddr_io, store_l2_io=store_l2_io_adj, store_smem_io=store_smem_io)
    tile_res = TileResources(per_iter=per_iter, prologue_epilogue=pe,
                             num_iterations=gridK, stage_num=stage_num,
                             smem_footprint=smem_footprint, reg_footprint=reg_footprint,
                             warps_per_block=warps_per_block, grids=(gridM, gridN))

    tiles_per_sm = compute_occupancy(smem_footprint, reg_footprint, warps_per_block, arch)
    total_ddr = (gridM * gridN * k * (tb_m * in1_level[0] + tb_n * in2_level[0]) * bytes_per_num
                 * (1 - l2_hit_rate) + gridM * gridN * tb_m * tb_n * bytes_per_num * out1_level[0])
    total_ddr *= DDR_non_ideal_para

    return _build_conv_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                              compute_flops, total_ddr, batch, bytes_per_num)


# =====================================================================
# Conv Implicit GEMM SDP (stride/dilation/padding)
# =====================================================================
def calculate_conv_implicit_gemm_sdp_pipeline_wave(
        op_shape, tb_shape, wp_shape, bytes_per_num, stage_num, arch,
        mem_levels, stride=1, dialation=1, padding=0, batch=1):
    """Conv implicit GEMM with stride/dilation/padding.

    对应 fused_op_dtype/conv_implicit_gemm_sdp_fused_op.py。
    结构与基础版 identical, stride/dilation/padding 目前仅影响 op shape 映射。
    """
    # SDP 版本与基础版在资源计算上完全一致
    # (stride/dilation/padding 影响的是 op_shape → GEMM 的映射, 已在 op_shape 中体现)
    return calculate_conv_implicit_gemm_pipeline_wave(
        op_shape, tb_shape, wp_shape, bytes_per_num, stage_num, arch,
        mem_levels, batch=batch)


# =====================================================================
# Conv NCHW
# =====================================================================
def calculate_conv_nchw_pipeline_wave(
        op_shape, tb_shape, dim_threads, r_step, bytes_per_num, arch,
        mem_levels, stage_num=1, batch=1):
    """Conv NCHW layout pipeline-aware model.

    对应 fused_op_dtype/conv_nchw_fused_op.py。
    4D spatial grid, 有 c_iter 内层循环。
    """
    conv_n, conv_f, conv_h, conv_w, conv_c, conv_kh, conv_kw, conv_s, conv_d, conv_p = op_shape
    tb_n, tb_f, tb_h, tb_w = tb_shape
    n_thread, f_thread, h_thread, w_thread = dim_threads
    threads_per_tb = n_thread * f_thread * h_thread * w_thread
    c_inner_step, kh_step, kw_step = r_step

    c_iter = math.ceil(conv_c / c_inner_step)

    thread_n = math.ceil(tb_n / n_thread)
    thread_f = math.ceil(tb_f / f_thread)
    thread_h = math.ceil(tb_h / h_thread)
    thread_w = math.ceil(tb_w / w_thread)

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    DDR_non_ideal_para = 1.1
    REG_spill_para = 1.1

    # Spatial dimensions
    inh = (conv_h - 1) * conv_s + (conv_kh - 1) * conv_d + 1 - 2 * conv_p
    inw = (conv_w - 1) * conv_s + (conv_kw - 1) * conv_d + 1 - 2 * conv_p
    padh = inh + 2 * conv_p
    padw = inw + 2 * conv_p
    tb_inh = (tb_h - 1) * conv_s + (conv_kh - 1) * conv_d + 1 - 2 * conv_p
    tb_inw = (tb_w - 1) * conv_s + (conv_kw - 1) * conv_d + 1 - 2 * conv_p
    tb_padh = tb_inh + 2 * conv_p
    tb_padw = tb_inw + 2 * conv_p

    compute_flops = 2 * conv_n * conv_f * conv_h * conv_w * conv_c * conv_kh * conv_kw
    l2_hit_rate = nchw_conv_l2_hitrate(
        conv_n, conv_f, conv_h, conv_w, conv_c, conv_kh, conv_kw,
        conv_s, conv_d, conv_p, tb_n, tb_f, tb_h, tb_w, c_inner_step,
        arch.l2_capacity, arch.sm_count, bytes_per_num)

    gridN = math.ceil(conv_n / tb_n)
    gridF = math.ceil(conv_f / tb_f)
    gridH = math.ceil(conv_h / tb_h)
    gridW = math.ceil(conv_w / tb_w)
    total_spatial_tiles = gridN * gridF * gridH * gridW

    # Reduction loop: c_iter iterations (over channels)
    # 每次 c_iter 内还有 kh*kw 的内层, 总 reduction iters = c_iter
    # (kh_step, kw_step 在每次 c_iter 内完成)
    num_reduction_iters = c_iter

    # Total IO
    pad_ratio = inh * inw / max(padh * padw, 1)
    total_l2_read = total_spatial_tiles * (
        tb_n * conv_c * tb_padh * tb_padw * pad_ratio * in1_level[0]
        + tb_f * conv_c * conv_kh * conv_kw * in2_level[0]) * bytes_per_num
    total_l2_store = total_spatial_tiles * tb_n * tb_f * tb_h * tb_w * bytes_per_num * out1_level[0]

    total_ddr = (total_l2_read * (1 - l2_hit_rate) + total_l2_store) * DDR_non_ideal_para

    # ---- Per c_iter resources ----
    per_iter_in1_load = tb_n * c_inner_step * tb_padh * tb_padw * pad_ratio * in1_level[0] * bytes_per_num
    per_iter_in2_load = tb_f * c_inner_step * conv_kh * conv_kw * in2_level[0] * bytes_per_num
    per_iter_l2_read = per_iter_in1_load + per_iter_in2_load
    per_iter_ddr = per_iter_l2_read * (1 - l2_hit_rate) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        per_iter_l2 = per_iter_l2_read * l2_hit_rate + per_iter_l2_read * (1 - l2_hit_rate) * 2
    else:
        per_iter_l2 = per_iter_l2_read

    # smem IO per c_iter: ldg + sts (load to smem)
    per_iter_smem = 2 * per_iter_l2_read
    # lds (load from smem for compute) — kh_step * kw_step inner loops
    per_iter_smem += (threads_per_tb * c_inner_step * kh_step
                      * (thread_n * thread_h * (thread_w + kw_step - 1) * in1_level[1]
                         + thread_f * kw_step * in2_level[1]) * bytes_per_num)

    per_iter_compute = 2 * tb_n * tb_f * tb_h * tb_w * c_inner_step * conv_kh * conv_kw

    # ---- Store ----
    store_ddr = tb_n * tb_f * tb_h * tb_w * bytes_per_num * out1_level[0]
    store_l2 = store_ddr
    store_smem = tb_n * tb_f * tb_h * tb_w * bytes_per_num * out1_level[1]

    # ---- Footprint ----
    smem_footprint = bytes_per_num * (tb_n * c_inner_step * tb_padh * tb_padw
                                      + tb_f * c_inner_step * kh_step * kw_step) * stage_num
    reg_footprint = math.ceil(
        (thread_n * thread_f * thread_h * thread_w + 4
         + thread_n * thread_h * (thread_w + conv_kw - 1)
         + thread_f * conv_kw) * REG_spill_para * bytes_per_num / 4)

    if arch.core in ("A100", "H100", "B200"):
        l2_io_total = total_l2_read * l2_hit_rate + total_l2_read * (1 - l2_hit_rate) * 2 + total_l2_store * 2
    else:
        l2_io_total = total_l2_read + total_l2_store

    # ---- TileResources ----
    per_iter = PerIterationResources(ddr_io=per_iter_ddr, l2_io=per_iter_l2,
                                     smem_io=per_iter_smem, compute_flops=per_iter_compute)
    pe = PrologueEpilogueResources(
        prologue_ddr_io=per_iter_ddr * max(stage_num - 1, 0),
        prologue_l2_io=per_iter_l2 * max(stage_num - 1, 0),
        prologue_smem_io=per_iter_smem * max(stage_num - 1, 0),
        epilogue_compute_flops=per_iter_compute * max(stage_num - 1, 0),
        store_ddr_io=store_ddr, store_l2_io=store_l2, store_smem_io=store_smem)
    tile_res = TileResources(per_iter=per_iter, prologue_epilogue=pe,
                             num_iterations=num_reduction_iters, stage_num=stage_num,
                             smem_footprint=smem_footprint, reg_footprint=reg_footprint,
                             warps_per_block=max(int(threads_per_tb / 32), 1),
                             grids=(gridN, gridF, gridH, gridW))

    tiles_per_sm = compute_occupancy(smem_footprint, reg_footprint,
                                     max(int(threads_per_tb / 32), 1), arch)

    return _build_conv_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                              compute_flops, total_ddr, batch, bytes_per_num)
