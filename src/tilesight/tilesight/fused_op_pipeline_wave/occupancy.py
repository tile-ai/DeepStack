import math
import logging

log = logging.getLogger(__name__)


def compute_occupancy(smem_footprint, reg_footprint, warps_per_block, arch, mma_type="wmma"):
    """Compute the number of thread blocks that can reside concurrently on each SM
    (tiles_per_sm).

    Take the minimum of three constraints:
    1. configurable_smem_capacity / smem_footprint (shared memory limit)
    2. register_capacity_per_sm / reg_per_block (register limit)
    3. max_blocks_per_sm (hardware limit)

    Args:
        smem_footprint: Shared memory per thread block, in bytes.
        reg_footprint: Registers per warp, counted in 4-byte registers.
        warps_per_block: Number of warps per thread block.
        arch: Architecture object.
        mma_type: MMA type; utcmma_cta2 requires special handling.

    Returns:
        tiles_per_sm: Integer, at least 1.
    """
    max_blocks = getattr(arch, 'max_blocks_per_sm', 24)

    # Constraint 1: shared memory
    if smem_footprint > 0:
        smem_limit = int(arch.configurable_smem_capacity / smem_footprint)
    else:
        smem_limit = max_blocks

    # Constraint 2: registers
    # reg_footprint is the number of 4-byte registers used per warp
    # register_capacity_per_sm is in bytes
    # Register usage per block (bytes) = reg_footprint * 4 * 32 * warps_per_block
    #   where 32 = threads_per_warp, 4 = bytes_per_register
    # However, in the existing code, reg_footprint units vary by op type:
    #   matmul: already divided by 32 and 4 (approximates the register count per thread)
    #   elementwise: similar
    # Use a conservative estimate: treat reg_footprint as the number of 4B registers per warp
    if reg_footprint > 0 and warps_per_block > 0:
        reg_per_block_bytes = reg_footprint * 4 * 32 * warps_per_block
        reg_limit = int(arch.register_capacity_per_sm / reg_per_block_bytes)
    else:
        reg_limit = max_blocks

    tiles_per_sm = min(smem_limit, reg_limit, max_blocks)

    # At least 1 (a block can still launch even if its footprint exceeds capacity)
    tiles_per_sm = max(tiles_per_sm, 1)

    # utcmma_cta2: 2-CTA cluster; each cluster occupies 2 SM slots
    if mma_type == "utcmma_cta2":
        tiles_per_sm = max(tiles_per_sm // 2, 1)

    log.info("occupancy: smem_limit=%d, reg_limit=%d, max_blocks=%d => tiles_per_sm=%d",
             smem_limit, reg_limit, max_blocks, tiles_per_sm)

    return tiles_per_sm
