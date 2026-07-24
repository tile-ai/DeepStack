import math
import logging

log = logging.getLogger(__name__)


def compute_occupancy(smem_footprint, reg_footprint, warps_per_block, arch, mma_type="wmma"):
    """计算每 SM 可共驻留的 thread block 数 (tiles_per_sm)。

    三项约束取 min:
    1. configurable_smem_capacity / smem_footprint  (shared memory 限制)
    2. register_capacity_per_sm / (reg_per_block)    (寄存器限制)
    3. max_blocks_per_sm                              (硬件上限)

    Args:
        smem_footprint: 每个 thread block 的 shared memory 用量 (bytes)
        reg_footprint:  每个 warp 的寄存器用量 (4-byte register 数)
        warps_per_block: 每个 thread block 的 warp 数
        arch: 架构对象
        mma_type: MMA 类型，utcmma_cta2 需要特殊处理

    Returns:
        tiles_per_sm: int, 至少为 1
    """
    max_blocks = getattr(arch, 'max_blocks_per_sm', 24)

    # Constraint 1: shared memory
    if smem_footprint > 0:
        smem_limit = int(arch.configurable_smem_capacity / smem_footprint)
    else:
        smem_limit = max_blocks

    # Constraint 2: registers
    # reg_footprint 是每个 warp 使用的 4-byte 寄存器数
    # register_capacity_per_sm 是 bytes
    # 每 block 寄存器用量 (bytes) = reg_footprint * 4 * 32 * warps_per_block
    #   其中 32 = threads_per_warp, 4 = bytes_per_register
    # 但在现有代码中 reg_footprint 的单位因 op 类型而异：
    #   matmul: 已除以 32 和 4（是 register count per thread 的近似）
    #   elementwise: 类似
    # 使用保守估计: reg_footprint 作为 per-warp 的 4B register 数
    if reg_footprint > 0 and warps_per_block > 0:
        reg_per_block_bytes = reg_footprint * 4 * 32 * warps_per_block
        reg_limit = int(arch.register_capacity_per_sm / reg_per_block_bytes)
    else:
        reg_limit = max_blocks

    tiles_per_sm = min(smem_limit, reg_limit, max_blocks)

    # 至少为 1（即使 footprint 超出容量，仍能 launch 一个 block）
    tiles_per_sm = max(tiles_per_sm, 1)

    # utcmma_cta2: 2-CTA cluster，每个 cluster 占 2 个 SM slot
    if mma_type == "utcmma_cta2":
        tiles_per_sm = max(tiles_per_sm // 2, 1)

    log.info("occupancy: smem_limit=%d, reg_limit=%d, max_blocks=%d => tiles_per_sm=%d",
             smem_limit, reg_limit, max_blocks, tiles_per_sm)

    return tiles_per_sm
