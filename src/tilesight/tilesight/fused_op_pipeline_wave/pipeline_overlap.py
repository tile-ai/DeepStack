import math
import logging
from .resource_types import TileResources, PipelineDetail, PipelineResult

log = logging.getLogger(__name__)


def resources_to_times(ddr_io, l2_io, smem_io, compute_flops, arch, data_bytes=2,
                       l1_5_io=0):
    """Convert resource quantities to time in seconds using full-system bandwidth.

    Each return value is the time required to process the corresponding quantity
    at full-system bandwidth. For per-SM analysis, shared-resource times (DDR/L2)
    must be multiplied by the number of contending SMs.

    Args:
        ddr_io: DDR traffic in bytes.
        l2_io: L2 traffic in bytes.
        smem_io: SMEM reads and writes in bytes.
        compute_flops: Computation in FLOPs.
        arch: Architecture object.
        data_bytes: Data type size in bytes, used to select tensor FLOPS.
        l1_5_io: L1.5 traffic in bytes; 0 if there is no L1.5 cache.

    Returns:
        (ddr_time, l2_time, l1_5_time, smem_time, compute_time), all in seconds.
    """
    ddr_time = ddr_io / arch.ddr_bandwidth if arch.ddr_bandwidth > 0 else 0.0
    l2_time = l2_io / arch.l2_bandwidth if arch.l2_bandwidth > 0 else 0.0
    smem_time = smem_io / arch.smem_bandwidth if arch.smem_bandwidth > 0 else 0.0

    l1_5_bw = getattr(arch, 'l1_5_bandwidth', 0)
    l1_5_time = l1_5_io / l1_5_bw if l1_5_bw > 0 else 0.0

    if data_bytes == 2:
        flops_capacity = arch.fp16_tensor_flops
    elif data_bytes == 1:
        if hasattr(arch, 'fp8_tensor_flops'):
            flops_capacity = arch.fp8_tensor_flops
        elif hasattr(arch, 'int8_tensor_flops'):
            flops_capacity = arch.int8_tensor_flops
        else:
            flops_capacity = arch.fp16_tensor_flops
    elif data_bytes == 4:
        flops_capacity = getattr(arch, 'fp32_tensor_flops', arch.fp32_cuda_core_flops)
    else:
        flops_capacity = arch.fp16_tensor_flops

    compute_time = compute_flops / flops_capacity if flops_capacity > 0 else 0.0

    return ddr_time, l2_time, l1_5_time, smem_time, compute_time


def _per_sm_iter_times(per_iter, arch, data_bytes, sm_count, active_sms=None):
    """Compute per-iteration times using per-SM bandwidth.

    Resource distinction:
    - DDR/L2: Full-system shared bandwidth, divided equally among active_sms SMs.
    - SMEM/Compute: Independent per-SM resources; always divide bandwidth by
      sm_count because the bandwidth definitions include sm_count.

    Args:
        per_iter: PerIterationResources.
        arch: Architecture object.
        data_bytes: Data type size in bytes.
        sm_count: Total architecture SM count, used for SMEM/Compute because
            architecture bandwidth includes sm_count.
        active_sms: Actual number of active SMs, used to allocate shared DDR/L2
            bandwidth. None defaults to a full wave, with active_sms = sm_count.

    Returns:
        (mem_time_per_iter, comp_time_per_iter), already divided by max_util.
    """
    if active_sms is None:
        active_sms = sm_count

    ddr_t, l2_t, l1_5_t, smem_t, comp_t = resources_to_times(
        per_iter.ddr_io, per_iter.l2_io, per_iter.smem_io, per_iter.compute_flops,
        arch, data_bytes, l1_5_io=per_iter.l1_5_io
    )

    # DDR/L2: shared bandwidth contested by active_sms SMs
    # Per-SM time = full-system time * active_sms
    ddr_t *= active_sms
    l2_t *= active_sms

    # L1.5: shared per group; bandwidth is defined for the whole chip
    # Like SMEM, multiply by sm_count
    l1_5_t *= sm_count

    # SMEM/Compute: per-SM resources, but arch.smem_bandwidth and flops are full-system totals
    # Per-SM time = full-system time * sm_count
    smem_t *= sm_count
    comp_t *= sm_count

    # Memory levels can overlap (take max), then divide by max_util
    l1_5_max_util = getattr(arch, 'l1_5_max_util', arch.l1_max_util)
    mem_time = max(ddr_t / arch.ddr_max_util,
                   l2_t / arch.l2_max_util,
                   l1_5_t / l1_5_max_util,
                   smem_t / arch.l1_max_util)
    comp_time = comp_t / arch.compute_max_util

    return mem_time, comp_time


def _per_sm_store_time(pe, arch, data_bytes, sm_count, active_sms=None):
    """Compute store time (epilogue writeback) using per-SM bandwidth."""
    if active_sms is None:
        active_sms = sm_count

    store_ddr_t, store_l2_t, store_l1_5_t, store_smem_t, _ = resources_to_times(
        pe.store_ddr_io, pe.store_l2_io, pe.store_smem_io, 0,
        arch, data_bytes, l1_5_io=pe.store_l1_5_io
    )
    # DDR/L2 are shared; L1.5/SMEM are per-SM
    store_ddr_t *= active_sms
    store_l2_t *= active_sms
    store_l1_5_t *= sm_count
    store_smem_t *= sm_count

    l1_5_max_util = getattr(arch, 'l1_5_max_util', arch.l1_max_util)
    store_time = max(store_ddr_t / arch.ddr_max_util,
                     store_l2_t / arch.l2_max_util,
                     store_l1_5_t / l1_5_max_util,
                     store_smem_t / arch.l1_max_util)
    return store_time


def compute_pipeline_tile_latency(tile_res, arch, data_bytes=2,
                                  sm_count=None, active_sms=None):
    """Compute pipeline-aware latency for a single tile.

    Use a per-SM bandwidth model:
    - DDR/L2 (shared): Each SM receives system bandwidth / active_sms.
    - SMEM/Compute (per-SM): Each SM independently has system bandwidth / sm_count.

    When total_tiles < sm_count, active_sms < sm_count, giving each active SM
    more DDR/L2 bandwidth.

    Behavior by stage_num:
    - stage_num=1: No pipelining; load and compute execute serially within each iteration.
    - stage_num>=2: Pipelining with prologue + steady state (overlap) + epilogue.

    Args:
        tile_res: TileResources.
        arch: Architecture object.
        data_bytes: Data type size in bytes.
        sm_count: Total architecture SM count; defaults to arch.sm_count.
        active_sms: Actual active SM count; defaults to sm_count (a full wave).

    Returns:
        (per_tile_latency, PipelineDetail)
    """
    if sm_count is None:
        sm_count = arch.sm_count
    if active_sms is None:
        active_sms = sm_count

    per_iter = tile_res.per_iter
    pe = tile_res.prologue_epilogue
    stage_num = tile_res.stage_num
    num_iters = tile_res.num_iterations

    mem_time_per_iter, comp_time_per_iter = _per_sm_iter_times(
        per_iter, arch, data_bytes, sm_count, active_sms)
    store_time = _per_sm_store_time(pe, arch, data_bytes, sm_count, active_sms)

    if stage_num <= 1:
        # ===== Case 1: No software pipeline =====
        # Load and compute execute serially in each iteration
        per_iter_time = mem_time_per_iter + comp_time_per_iter
        per_tile_latency = num_iters * per_iter_time + store_time

        detail = PipelineDetail(
            prologue_time=0.0,
            steady_time_per_iter=per_iter_time,
            epilogue_time=store_time,
            mem_time_per_iter=mem_time_per_iter,
            compute_time_per_iter=comp_time_per_iter,
        )
    else:
        # ===== Case 2: Software pipeline enabled (stage_num >= 2) =====
        pipeline_depth = stage_num - 1

        prologue_time = pipeline_depth * mem_time_per_iter
        steady_iters = max(num_iters - pipeline_depth, 0)
        steady_time_per_iter = max(mem_time_per_iter, comp_time_per_iter)
        steady_time = steady_iters * steady_time_per_iter
        epilogue_time = pipeline_depth * comp_time_per_iter + store_time

        per_tile_latency = prologue_time + steady_time + epilogue_time

        detail = PipelineDetail(
            prologue_time=prologue_time,
            steady_time_per_iter=steady_time_per_iter,
            epilogue_time=epilogue_time,
            mem_time_per_iter=mem_time_per_iter,
            compute_time_per_iter=comp_time_per_iter,
        )

    log.info("tile latency: stage=%d, iters=%d, active_sms=%d, mem_t=%.3e, "
             "comp_t=%.3e, store_t=%.3e => tile_lat=%.3e",
             stage_num, num_iters, active_sms, mem_time_per_iter,
             comp_time_per_iter, store_time, per_tile_latency)

    return per_tile_latency, detail


def compute_pipeline_tile_latency_with_occupancy(tile_res, tiles_per_sm, arch,
                                                 data_bytes=2, sm_count=None,
                                                 active_sms=None):
    """Compute pipeline latency with multiple tiles interleaved on an SM.

    When tiles_per_sm > 1, multiple tiles alternate execution on the same SM.
    Effective pipeline depth = stage_num * tiles_per_sm, corresponding to
    multiplication of stage counts in the algorithm.

    Args:
        tile_res: TileResources.
        tiles_per_sm: Occupancy, in tiles per SM.
        arch: Architecture object.
        data_bytes: Data type size in bytes.
        sm_count: Total architecture SM count; defaults to arch.sm_count.
        active_sms: Actual active SM count; defaults to sm_count.

    Returns:
        (sm_latency, PipelineDetail)
    """
    if sm_count is None:
        sm_count = arch.sm_count
    if active_sms is None:
        active_sms = sm_count

    if tiles_per_sm <= 1:
        per_tile_lat, detail = compute_pipeline_tile_latency(
            tile_res, arch, data_bytes, sm_count=sm_count, active_sms=active_sms)
        return per_tile_lat, detail

    # tiles_per_sm > 1: interleave multiple tiles
    per_iter = tile_res.per_iter
    pe = tile_res.prologue_epilogue
    stage_num = tile_res.stage_num
    num_iters = tile_res.num_iterations

    mem_time_per_iter, comp_time_per_iter = _per_sm_iter_times(
        per_iter, arch, data_bytes, sm_count, active_sms)
    store_time = _per_sm_store_time(pe, arch, data_bytes, sm_count, active_sms)

    # Effective pipeline depth
    effective_stage = stage_num * tiles_per_sm
    effective_pipeline_depth = effective_stage - 1
    total_iters = num_iters * tiles_per_sm

    prologue_time = min(effective_pipeline_depth, total_iters) * mem_time_per_iter
    steady_iters = max(total_iters - effective_pipeline_depth, 0)
    steady_time_per_iter = max(mem_time_per_iter, comp_time_per_iter)
    steady_time = steady_iters * steady_time_per_iter
    drain_iters = min(effective_pipeline_depth, total_iters)
    epilogue_time = drain_iters * comp_time_per_iter + tiles_per_sm * store_time

    sm_latency = prologue_time + steady_time + epilogue_time

    detail = PipelineDetail(
        prologue_time=prologue_time,
        steady_time_per_iter=steady_time_per_iter,
        epilogue_time=epilogue_time,
        mem_time_per_iter=mem_time_per_iter,
        compute_time_per_iter=comp_time_per_iter,
    )

    log.info("SM latency: tiles_per_sm=%d, active_sms=%d, eff_stage=%d, "
             "total_iters=%d => sm_lat=%.3e",
             tiles_per_sm, active_sms, effective_stage, total_iters, sm_latency)

    return sm_latency, detail
