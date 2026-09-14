import math
import logging
from .resource_types import PipelineDetail

log = logging.getLogger(__name__)


def compute_wave_adjusted_latency(sm_latency, tiles_per_sm, total_tiles, arch,
                                  pipeline_detail=None, mma_type="wmma",
                                  tile_res=None, data_bytes=2, sm_count_override=None):
    """Compute total kernel latency, accounting for wave head/tail effects.

    When total_tiles < sm_count, fewer than sm_count SMs are active, so each active
    SM receives more shared DDR/L2 bandwidth and has a shorter per-tile latency.

    If tile_res is provided, recompute per-tile latency with the correct active_sms
    for the tail wave and for cases where total_tiles < sm_count.

    Args:
        sm_latency: Latency of one SM in a full wave, in seconds, assuming all SMs are active.
        tiles_per_sm: Occupancy.
        total_tiles: Total spatial tile count (gridM * gridN).
        arch: Architecture object.
        pipeline_detail: PipelineDetail from the full-wave computation.
        mma_type: MMA type.
        tile_res: Optional TileResources for precise recomputation of tail-wave latency.
        data_bytes: Data type size in bytes.
        sm_count_override: Override sm_count, such as sm_count//2 for utcmma_cta2.

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
        # Full wave: all sm_count SMs are active; sm_latency is already correct
        head_wave_time = sm_latency * head_penalty
        middle_waves = max(num_full_waves - 1, 0)
        middle_wave_time = sm_latency
    else:
        head_wave_time = 0.0
        middle_waves = 0
        middle_wave_time = 0.0

    # === Tail wave (including total_tiles < sm_count) ===
    #
    # GPU scheduling is round-robin: assign 1 tile per SM first, then assign the second tile after all SMs are filled.
    # Therefore, tail_tiles tiles are assigned as follows:
    #   - tail_tiles <= sm_count: exactly 1 tile per active SM
    #   - tail_tiles > sm_count: all SMs are active, and the busiest SM has ceil(tail_tiles/sm_count) tiles
    #
    if tail_tiles > 0:
        if tail_tiles <= sm_count:
            # Not enough tiles to fill all SMs; each active SM receives only 1 tile
            active_sms = tail_tiles
            tail_tiles_per_sm = 1
        else:
            # More tiles than SMs; all SMs are active
            active_sms = sm_count
            tail_tiles_per_sm = math.ceil(tail_tiles / sm_count)
            # Must not exceed occupancy capacity
            tail_tiles_per_sm = min(tail_tiles_per_sm, tiles_per_sm)

        if tile_res is not None:
            # Recalculate accurately using the correct active_sms and tail_tiles_per_sm
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

    # === Total latency ===
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
