import math
import logging
from .resource_types import PipelineDetail

log = logging.getLogger(__name__)


def compute_wave_adjusted_latency(sm_latency, tiles_per_sm, total_tiles, arch,
                                  pipeline_detail=None, mma_type="wmma",
                                  tile_res=None, data_bytes=2, sm_count_override=None):
    """计算考虑 wave head/tail 效应的 kernel 总延迟。

    关键改进: 当 total_tiles < sm_count 时, 活跃 SM 数 < sm_count,
    每个活跃 SM 分到更多的 DDR/L2 共享带宽, per-tile 延迟更短。

    如果提供了 tile_res, 会对 tail wave (以及 total_tiles < sm_count 的情况)
    用正确的 active_sms 重新计算 per-tile latency。

    Args:
        sm_latency: full wave 下一个 SM 的延迟 (秒), 假设所有 SM 活跃
        tiles_per_sm: occupancy
        total_tiles: 空间 tile 总数 (gridM * gridN)
        arch: 架构对象
        pipeline_detail: PipelineDetail from full-wave computation
        mma_type: MMA 类型
        tile_res: TileResources, 用于精确重算 tail wave latency (可选)
        data_bytes: 数据类型字节数
        sm_count_override: 覆盖 sm_count (如 utcmma_cta2 用 sm_count//2)

    Returns:
        (total_latency, wave_info)
    """
    sm_count = sm_count_override if sm_count_override else arch.sm_count
    if mma_type == "utcmma_cta2" and sm_count_override is None:
        sm_count = sm_count // 2

    tiles_per_wave = sm_count * tiles_per_sm
    if tiles_per_wave <= 0:
        tiles_per_wave = 1

    num_full_waves = total_tiles // tiles_per_wave
    tail_tiles = total_tiles % tiles_per_wave
    total_waves = num_full_waves + (1 if tail_tiles > 0 else 0)

    # === Head wave penalty ===
    head_penalty = 1.0
    if pipeline_detail is not None and pipeline_detail.prologue_time > 0:
        prologue_ratio = pipeline_detail.prologue_time / max(sm_latency, 1e-30)
        head_penalty = 1.0 + 0.1 * prologue_ratio

    # === Full waves ===
    if num_full_waves >= 1:
        # Full wave: 所有 sm_count 个 SM 活跃, sm_latency 已正确
        head_wave_time = sm_latency * head_penalty
        middle_waves = max(num_full_waves - 1, 0)
        middle_wave_time = sm_latency
    else:
        head_wave_time = 0.0
        middle_waves = 0
        middle_wave_time = 0.0

    # === Tail wave (包括 total_tiles < sm_count 的情况) ===
    #
    # GPU 调度是 round-robin: 先每 SM 分 1 个 tile, 填满后再分第 2 个。
    # 所以 tail_tiles 个 tile 的分配:
    #   - tail_tiles <= sm_count: 每个活跃 SM 恰好 1 个 tile
    #   - tail_tiles > sm_count: 所有 SM 活跃, busiest SM 有 ceil(tail_tiles/sm_count) 个
    #
    if tail_tiles > 0:
        if tail_tiles <= sm_count:
            # 不够填满所有 SM, 每个活跃 SM 只分到 1 个 tile
            active_sms = tail_tiles
            tail_tiles_per_sm = 1
        else:
            # 多于 SM 数, 所有 SM 活跃
            active_sms = sm_count
            tail_tiles_per_sm = math.ceil(tail_tiles / sm_count)
            # 不能超过 occupancy 容量
            tail_tiles_per_sm = min(tail_tiles_per_sm, tiles_per_sm)

        if tile_res is not None:
            # 精确重算: 用正确的 active_sms 和 tail_tiles_per_sm
            from .pipeline_overlap import compute_pipeline_tile_latency_with_occupancy
            tail_sm_latency, _ = compute_pipeline_tile_latency_with_occupancy(
                tile_res, tail_tiles_per_sm, arch, data_bytes=data_bytes,
                sm_count=sm_count, active_sms=active_sms)
        else:
            tail_sm_latency = sm_latency * tail_tiles_per_sm / max(tiles_per_sm, 1)

        if num_full_waves == 0:
            tail_wave_time = tail_sm_latency * head_penalty
        else:
            tail_wave_time = tail_sm_latency
    else:
        tail_wave_time = 0.0
        active_sms = sm_count

    # === 总延迟 ===
    total_latency = head_wave_time + middle_waves * middle_wave_time + tail_wave_time

    waves = total_tiles / max(tiles_per_wave, 1)

    wave_info = {
        'total_waves': total_waves,
        'num_full_waves': num_full_waves,
        'tail_tiles': tail_tiles,
        'tiles_per_wave': tiles_per_wave,
        'head_penalty': head_penalty,
        'waves_float': waves,
    }

    log.info("wave model: total_tiles=%d, sm_count=%d, tiles_per_wave=%d, "
             "full_waves=%d, tail=%d, active_sms=%s => total_lat=%.3e",
             total_tiles, sm_count, tiles_per_wave, num_full_waves,
             tail_tiles, active_sms, total_latency)

    return total_latency, wave_info
