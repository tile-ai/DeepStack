import math
import logging
from .resource_types import TileResources, PipelineDetail, PipelineResult

log = logging.getLogger(__name__)


def resources_to_times(ddr_io, l2_io, smem_io, compute_flops, arch, data_bytes=2,
                       l1_5_io=0):
    """将资源量转换为时间 (秒), 使用全系统带宽。

    返回值是"在全系统带宽下完成该资源量所需时间"。
    对于 per-SM 分析, 共享资源(DDR/L2)需乘以竞争 SM 数。

    Args:
        ddr_io: DDR 流量 (bytes)
        l2_io: L2 流量 (bytes)
        smem_io: SMEM 读写 (bytes)
        compute_flops: 计算量 (FLOPs)
        arch: 架构对象
        data_bytes: 数据类型字节数 (用于选择 tensor flops)
        l1_5_io: L1.5 流量 (bytes), 0 if no L1.5

    Returns:
        (ddr_time, l2_time, l1_5_time, smem_time, compute_time) 各单位为秒
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
    """计算每迭代在 per-SM 带宽下的时间。

    关键区分:
    - DDR/L2: 全系统共享带宽, 由 active_sms 个 SM 平分
    - SMEM/Compute: per-SM 独立资源, 始终除以 sm_count (因为 bandwidth 定义含 sm_count)

    Args:
        per_iter: PerIterationResources
        arch: 架构对象
        data_bytes: 数据类型字节数
        sm_count: 架构总 SM 数 (用于 SMEM/Compute, 因为 arch bandwidth 包含了 sm_count)
        active_sms: 实际活跃的 SM 数 (用于 DDR/L2 共享带宽分配),
                    默认 None 表示 full wave = sm_count

    Returns:
        (mem_time_per_iter, comp_time_per_iter) — 已除以 max_util
    """
    if active_sms is None:
        active_sms = sm_count

    ddr_t, l2_t, l1_5_t, smem_t, comp_t = resources_to_times(
        per_iter.ddr_io, per_iter.l2_io, per_iter.smem_io, per_iter.compute_flops,
        arch, data_bytes, l1_5_io=per_iter.l1_5_io
    )

    # DDR/L2: 共享带宽, 由 active_sms 个 SM 竞争
    # per-SM 时间 = 全系统时间 * active_sms
    ddr_t *= active_sms
    l2_t *= active_sms

    # L1.5: per-group 共享, bandwidth 定义是 whole chip
    # 与 SMEM 类似, 需要 × sm_count
    l1_5_t *= sm_count

    # SMEM/Compute: per-SM 资源, 但 arch.smem_bandwidth 和 flops 是全系统总和
    # per-SM 时间 = 全系统时间 * sm_count
    smem_t *= sm_count
    comp_t *= sm_count

    # 各 mem 层之间可 overlap (取 max), 再除以 max_util
    l1_5_max_util = getattr(arch, 'l1_5_max_util', arch.l1_max_util)
    mem_time = max(ddr_t / arch.ddr_max_util,
                   l2_t / arch.l2_max_util,
                   l1_5_t / l1_5_max_util,
                   smem_t / arch.l1_max_util)
    comp_time = comp_t / arch.compute_max_util

    return mem_time, comp_time


def _per_sm_store_time(pe, arch, data_bytes, sm_count, active_sms=None):
    """计算 store (epilogue 写回) 在 per-SM 带宽下的时间。"""
    if active_sms is None:
        active_sms = sm_count

    store_ddr_t, store_l2_t, store_l1_5_t, store_smem_t, _ = resources_to_times(
        pe.store_ddr_io, pe.store_l2_io, pe.store_smem_io, 0,
        arch, data_bytes, l1_5_io=pe.store_l1_5_io
    )
    # DDR/L2 共享, L1.5/SMEM per-SM
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
    """计算单个 tile 的 pipeline-aware latency。

    使用 per-SM 带宽模型:
    - DDR/L2 (共享): 每个 SM 分到 系统带宽 / active_sms
    - SMEM/Compute (per-SM): 每个 SM 独立拥有 系统带宽 / sm_count

    当 total_tiles < sm_count 时, active_sms < sm_count,
    每个活跃 SM 分到更多的 DDR/L2 带宽。

    根据 stage_num 分情况：
    - stage_num=1: 无 pipeline，每迭代内 load 与 compute 串行
    - stage_num>=2: 有 pipeline，prologue + steady(overlap) + epilogue

    Args:
        tile_res: TileResources
        arch: 架构对象
        data_bytes: 数据类型字节数
        sm_count: 架构 SM 总数 (默认 arch.sm_count)
        active_sms: 实际活跃 SM 数 (默认 = sm_count, 即 full wave)

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
        # ===== Case 1: 无软件流水线 =====
        # 每迭代 load 与 compute 串行
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
        # ===== Case 2: 有软件流水线 (stage_num >= 2) =====
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
    """考虑 SM 上多 tile 交织的 pipeline latency。

    当 tiles_per_sm > 1 时，多个 tile 在同一 SM 上交替执行。
    等效 pipeline depth = stage_num * tiles_per_sm (算法中的 stage 累乘)。

    Args:
        tile_res: TileResources
        tiles_per_sm: occupancy (tiles per SM)
        arch: 架构对象
        data_bytes: 数据类型字节数
        sm_count: 架构 SM 总数 (默认 arch.sm_count)
        active_sms: 实际活跃 SM 数 (默认 = sm_count)

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

    # tiles_per_sm > 1: 多 tile 交织
    per_iter = tile_res.per_iter
    pe = tile_res.prologue_epilogue
    stage_num = tile_res.stage_num
    num_iters = tile_res.num_iterations

    mem_time_per_iter, comp_time_per_iter = _per_sm_iter_times(
        per_iter, arch, data_bytes, sm_count, active_sms)
    store_time = _per_sm_store_time(pe, arch, data_bytes, sm_count, active_sms)

    # 等效 pipeline depth
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
