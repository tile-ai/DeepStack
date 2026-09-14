"""Two-level SDCM cascade: L1.5 (per-group) + L2 (global).

Models the passive L1.5 cache where physically adjacent SMs (~8 per group)
share a small cache. Within a group, if a tile was recently accessed by
another SM in the same group, the data may still be in the L1.5.

Provides:
1. multilevel_hit_rate_general() — N-D grid, arbitrary tensors with reuse dims
2. multilevel_hit_rate_reuse_distance() — GEMM convenience wrapper
"""

import numpy as np
from numpy.random import permutation
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from functools import reduce
import operator

from tilesight.util.sdcm import sdcm
from tilesight.util.extract_blocks import extract_blocks


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class TensorAccess:
    """Describes how a tensor is accessed across the tile grid.

    A tensor is "reused" along certain grid dimensions — meaning multiple
    tiles along that dimension access the same data block.

    reuse_dims: indices into grid_shape that this tensor is shared across.
        e.g. for grid_shape=(gridM, gridN):
            reuse_dims=[1] → shared across N (same M-row reuses this tensor)
            reuse_dims=[0] → shared across M (same N-column reuses)
            reuse_dims=[]  → no reuse, unique per tile

    access_count: number of times the same data block is accessed per tile
        per inner-loop iteration. Default=1.
        Conv example: weight tensor accessed KH*KW times per C-iteration
        → access_count=KH*KW, because the same weight block at a given
        (m,n) position is accessed KH*KW times within one K-iter.

    Examples:
        GEMM grid (gridM, gridN), K inner loop:
            A: reuse_dims=[1], footprint=tb_m*tb_k*2  → shared across N
            B: reuse_dims=[0], footprint=tb_n*tb_k*2  → shared across M

        MLA decode grid (batch_tiles, head_num), KV-loop inner:
            KV:   reuse_dims=[1], footprint=block_n*dv*2   → shared across heads
            K_pe: reuse_dims=[1], footprint=block_n*dpe*2  → shared across heads

        Conv NCHW grid (gridN, gridF, gridH, gridW), C inner loop:
            input:  reuse_dims=[1], footprint=tb_n*c_step*tb_padh*tb_padw*2
            weight: reuse_dims=[0,2,3], footprint=tb_f*c_step*2,
                    access_count=KH*KW   → same C-block accessed KH*KW times

        Conv implicit_gemm grid (gridM, gridN), K=KH*KW*C inner loop:
            activations(A): reuse_dims=[1], footprint=tb_m*tb_k*2
            weights(B):     reuse_dims=[0], footprint=tb_n*tb_k*2,
                            access_count=1  (KH*KW folded into K dim)
    """
    name: str
    footprint_bytes: float          # per-tile data size in bytes, per K-iteration
    reuse_dims: List[int] = field(default_factory=list)
    ddr_flag: int = 1               # 1 if loaded from DDR, 0 if not
    access_count: int = 1           # accesses per tile per K-iter (for conv KH*KW)


@dataclass
class CacheConfig:
    """Cache level parameters."""
    capacity: float                 # bytes (for L1.5: per-group; for L2: total)
    associativity: int = 8
    cacheline_bytes: int = 128

    @property
    def num_cachelines(self):
        return self.capacity / (self.associativity * self.cacheline_bytes)


# =====================================================================
# N-D tile scheduling
# =====================================================================

def _nd_tile_schedule(grid_shape, SM_Count):
    """Generate a 1D tile schedule from an N-D grid.

    Returns:
        coords: (total_tiles, N) array, 0-indexed tile coordinates.
        stride_length: number of tiles per scheduling stride (≈ SM_Count).
    """
    ndim = len(grid_shape)
    total_tiles = reduce(operator.mul, grid_shape, 1)

    if ndim == 2:
        # Use existing 2D extract_blocks (1-indexed, row-panel scheduling)
        gridM, gridN = int(grid_shape[0]), int(grid_shape[1])
        Stride_N = max(SM_Count // gridM, 1)
        Stride_M = max(SM_Count // Stride_N, 1)
        coords_1indexed = extract_blocks(gridM, gridN, Stride_M, Stride_N)
        # Convert to 0-indexed
        coords = coords_1indexed - 1
        stride_length = Stride_M * Stride_N
    else:
        # N-D: row-major linearization, stride = SM_Count
        coords = np.zeros((total_tiles, ndim), dtype=int)
        for i in range(total_tiles):
            idx = i
            for d in range(ndim - 1, -1, -1):
                coords[i, d] = idx % grid_shape[d]
                idx //= grid_shape[d]
        stride_length = SM_Count

    return coords, stride_length


def _tensor_rd_key(coords_row, grid_shape, reuse_dims):
    """Compute the reuse-distance lookup key for a tensor.

    For a tensor with reuse_dims, the key is the tuple of non-reuse coordinates.
    Tiles with the same key access the same data block.

    Returns:
        int: linearized key into the RD array.
        int: size of the RD array (product of non-reuse dimensions).
    """
    ndim = len(grid_shape)
    non_reuse = [d for d in range(ndim) if d not in reuse_dims]

    if len(non_reuse) == 0:
        # Reused everywhere
        return 0, 1

    # Linearize non-reuse coordinates
    key = 0
    stride = 1
    for d in reversed(non_reuse):
        key += int(coords_row[d]) * stride
        stride *= int(grid_shape[d])

    return key, stride  # stride here is the total RD size


def _compute_rd_size(grid_shape, reuse_dims):
    """Compute the reuse-distance array size for a tensor."""
    ndim = len(grid_shape)
    non_reuse = [d for d in range(ndim) if d not in reuse_dims]
    if len(non_reuse) == 0:
        return 1
    return reduce(operator.mul, [int(grid_shape[d]) for d in non_reuse], 1)


# =====================================================================
# General multi-level cache model
# =====================================================================

def multilevel_hit_rate_general(grid_shape, tensors, SM_Count,
                                l2_config, l1_5_config=None,
                                l1_5_group_size=8,
                                row_panel=None,
                                k_iterations=1,
                                output_footprint_bytes=0):
    """General multi-level cache hit rate with arbitrary tensor access patterns.

    Args:
        grid_shape: tuple of spatial grid dimensions.
            (gridM, gridN) for GEMM.
            (batch_tiles, head_num) for MLA decode.
            (gridN, gridF, gridH, gridW) for Conv NCHW.
        tensors: list of TensorAccess describing each tensor's reuse pattern.
        SM_Count: number of SMs on the GPU.
        l2_config: CacheConfig for L2 cache.
        l1_5_config: CacheConfig for L1.5 (per-group). None = no L1.5.
        l1_5_group_size: SMs per L1.5 group.
        row_panel: scheduling stride (2D only). Default auto.
        k_iterations: number of inner-loop iterations.
        output_footprint_bytes: output tile footprint (for inter-stride aging).

    Returns:
        dict:
            'per_tensor': {name: {'l1_5_hit', 'l2_hit', 'ddr_miss'}}
            'l1_5_hit_rate': aggregate fraction served by L1.5
            'l2_hit_rate': aggregate fraction of L1.5 misses served by L2
            'ddr_miss_rate': aggregate fraction going to DDR
    """
    grid_shape = tuple(int(g) for g in grid_shape)
    ndim = len(grid_shape)
    total_tiles = reduce(operator.mul, grid_shape, 1)

    if total_tiles == 0:
        return {'per_tensor': {}, 'l1_5_hit_rate': 0, 'l2_hit_rate': 0, 'ddr_miss_rate': 1}

    # --- Scheduling ---
    if ndim == 2 and row_panel is not None:
        gridM, gridN = int(grid_shape[0]), int(grid_shape[1])
        Stride_N = int(row_panel)
        Stride_M = max(SM_Count // Stride_N, 1)
        coords_1indexed = extract_blocks(gridM, gridN, Stride_M, Stride_N)
        coords = coords_1indexed - 1  # 0-indexed
        stride_length = Stride_M * Stride_N
    else:
        coords, stride_length = _nd_tile_schedule(grid_shape, SM_Count)

    # --- Cache configs ---
    L2_cl = l2_config.cacheline_bytes
    L2_total_cl = l2_config.num_cachelines

    has_l15 = l1_5_config is not None and l1_5_group_size > 0
    num_groups = max(SM_Count // l1_5_group_size, 1) if has_l15 else 1
    L15_cl = l1_5_config.cacheline_bytes if has_l15 else L2_cl

    output_cl_l2 = output_footprint_bytes / L2_cl if L2_cl > 0 else 0

    # --- Build per-tensor tracking ---
    tensor_infos = []
    for t in tensors:
        if not t.ddr_flag:
            continue

        rd_size = _compute_rd_size(grid_shape, t.reuse_dims)
        fp_cl_l2 = t.footprint_bytes / L2_cl
        fp_cl_l15 = t.footprint_bytes / L15_cl if has_l15 else 0

        tensor_infos.append({
            'tensor': t,
            'rd_size': rd_size,
            'fp_cl_l2': fp_cl_l2,
            'fp_cl_l15': fp_cl_l15,
            'RD_l2': np.ones(rd_size) * 1e9,
            'RD_l15': [np.ones(rd_size) * 1e9 for _ in range(num_groups)] if has_l15 else None,
            'l15_hits': 0.0,
            'l2_hits': 0.0,
            'total_accesses': 0,
            'unique_per_stride': np.zeros(rd_size),
        })

    # Precompute total per-tile footprint across all tensors (for RD reset offset)
    # This models the perturbation that tensors are loaded sequentially within a tile,
    # so later tensors see the earlier tensors' data already in cache.
    # Original model: M_RD[m] = -0.5*MK, N_RD[n] = -MK - 0.5*NK
    # Generalized: tensor_i starts at -(sum of earlier tensors' footprints) - 0.5*own
    cumulative_fp_l2 = []
    cumulative_fp_l15 = []
    running_l2 = 0.0
    running_l15 = 0.0
    for ti in tensor_infos:
        cumulative_fp_l2.append(running_l2)
        cumulative_fp_l15.append(running_l15)
        running_l2 += ti['fp_cl_l2']
        running_l15 += ti['fp_cl_l15']
    total_fp_per_tile_l2 = running_l2
    total_fp_per_tile_l15 = running_l15

    # --- SM-to-group mapping ---
    # Physical GPU: SM 0..7 = group 0, SM 8..15 = group 1, etc.
    # This mapping is fixed — shuffle only changes which tile goes to which SM,
    # NOT which SM belongs to which group.
    sm_to_group = np.zeros(SM_Count, dtype=int)
    if has_l15:
        for s in range(SM_Count):
            sm_to_group[s] = min(s // l1_5_group_size, num_groups - 1)

    # --- Main simulation ---
    for count in range(0, total_tiles, stride_length):
        length = min(stride_length, total_tiles - count)
        shuffled_seq = permutation(length)

        for ti in tensor_infos:
            ti['unique_per_stride'][:] = 0

        for i in range(length):
            current_idx = count + shuffled_seq[i]
            tile_coord = coords[current_idx]
            # shuffled_seq[i] is the SM index within this wave;
            # map to its fixed physical group
            sm_idx = int(shuffled_seq[i]) % SM_Count
            group_id = int(sm_to_group[sm_idx]) if has_l15 else 0

            for t_idx, ti in enumerate(tensor_infos):
                t = ti['tensor']
                rd_idx, _ = _tensor_rd_key(tile_coord, grid_shape, t.reuse_dims)

                # Each access_count is a separate cache probe on the same block
                for _ in range(t.access_count):
                    ti['total_accesses'] += 1

                    # L1.5 check
                    l15_hit = 0.0
                    if has_l15:
                        l15_hit = sdcm(
                            ti['RD_l15'][group_id][rd_idx],
                            l1_5_config.associativity,
                            l1_5_config.num_cachelines * l1_5_config.associativity)
                        ti['l15_hits'] += l15_hit

                    # L2 check (for L1.5 misses)
                    l2_hit = sdcm(
                        ti['RD_l2'][rd_idx],
                        l2_config.associativity,
                        L2_total_cl * l2_config.associativity)
                    ti['l2_hits'] += (1 - l15_hit) * l2_hit

                # Update RDs with perturbation: loading is sequential across tensors,
                # so tensor_i's RD resets to -(earlier tensors' footprint) - 0.5*own.
                # All tensors' RDs age by the total per-tile footprint (all tensors
                # contribute to cache pressure).
                if has_l15:
                    ti['RD_l15'][group_id][rd_idx] = (
                        -cumulative_fp_l15[t_idx] - 0.5 * ti['fp_cl_l15'])
                    ti['RD_l15'][group_id] += total_fp_per_tile_l15

                ti['RD_l2'][rd_idx] = (
                    -cumulative_fp_l2[t_idx] - 0.5 * ti['fp_cl_l2'])
                ti['RD_l2'] += total_fp_per_tile_l2

                ti['unique_per_stride'][rd_idx] = 1

        # Inter-stride aging: account for ALL tensors' unique footprints,
        # not just the current tensor (matches original model behavior).
        total_unique_aging_l2 = sum(
            ti['fp_cl_l2'] * np.sum(ti['unique_per_stride'])
            for ti in tensor_infos)
        total_unique_aging_l15 = sum(
            ti['fp_cl_l15'] * np.sum(ti['unique_per_stride'])
            for ti in tensor_infos) if has_l15 else 0

        for ti in tensor_infos:
            aging_l2 = (stride_length * output_cl_l2
                        + (k_iterations - 1) * total_unique_aging_l2)
            ti['RD_l2'] += aging_l2

            if has_l15:
                output_cl_l15 = output_footprint_bytes / L15_cl if L15_cl > 0 else 0
                aging_l15 = (stride_length * output_cl_l15
                             + (k_iterations - 1) * total_unique_aging_l15)
                for g in range(num_groups):
                    ti['RD_l15'][g] += aging_l15

    # --- Compute hit rates ---
    per_tensor = {}
    total_l15_io = 0.0
    total_l2_io = 0.0
    total_ddr_io = 0.0

    for ti in tensor_infos:
        t = ti['tensor']
        n_acc = ti['total_accesses']
        if n_acc == 0:
            per_tensor[t.name] = {'l1_5_hit': 0.0, 'l2_hit': 0.0, 'ddr_miss': 1.0}
            continue

        l15_rate = ti['l15_hits'] / n_acc
        l2_rate = ti['l2_hits'] / n_acc
        ddr_rate = max(1.0 - l15_rate - l2_rate, 0.0)

        per_tensor[t.name] = {
            'l1_5_hit': float(l15_rate),
            'l2_hit': float(l2_rate),
            'ddr_miss': float(ddr_rate),
        }

        fp = t.footprint_bytes
        total_l15_io += ti['l15_hits'] * fp
        total_l2_io += ti['l2_hits'] * fp
        total_ddr_io += max(n_acc - ti['l15_hits'] - ti['l2_hits'], 0) * fp

    total_io = total_l15_io + total_l2_io + total_ddr_io
    agg_l15 = total_l15_io / total_io if total_io > 0 else 0.0
    agg_l2_of_miss = total_l2_io / (total_l2_io + total_ddr_io) if (total_l2_io + total_ddr_io) > 0 else 0.0
    agg_ddr = total_ddr_io / total_io if total_io > 0 else 1.0

    return {
        'per_tensor': per_tensor,
        'l1_5_hit_rate': float(agg_l15),
        'l2_hit_rate': float(agg_l2_of_miss),
        'ddr_miss_rate': float(agg_ddr),
    }


# =====================================================================
# GEMM convenience wrapper (backward compat)
# =====================================================================

def multilevel_hit_rate_reuse_distance(M, N, K, tb_m, tb_n, tb_k,
                                       L2_Cap, SM_Count, mem_levels, row_panel,
                                       l1_5_group_size=8,
                                       l1_5_capacity_per_group=180*1024,
                                       l1_5_associativity=8,
                                       l1_5_cacheline_bytes=128):
    """GEMM-specific two-level SDCM cascade.

    Convenience wrapper: A matrix reused across N, B matrix reused across M.
    """
    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    gridM = int(np.ceil(M / tb_m))
    gridN = int(np.ceil(N / tb_n))
    gridK = int(np.ceil(K / tb_k))

    tensors = [
        TensorAccess('A', footprint_bytes=tb_m * tb_k * in1_level[-1],
                     reuse_dims=[1], ddr_flag=in1_level[0]),
        TensorAccess('B', footprint_bytes=tb_n * tb_k * in2_level[-1],
                     reuse_dims=[0], ddr_flag=in2_level[0]),
    ]

    l2_config = CacheConfig(capacity=L2_Cap, associativity=8, cacheline_bytes=128)
    l1_5_config = CacheConfig(
        capacity=l1_5_capacity_per_group,
        associativity=l1_5_associativity,
        cacheline_bytes=l1_5_cacheline_bytes,
    ) if l1_5_group_size > 0 else None

    return multilevel_hit_rate_general(
        grid_shape=(gridM, gridN),
        tensors=tensors,
        SM_Count=SM_Count,
        l2_config=l2_config,
        l1_5_config=l1_5_config,
        l1_5_group_size=l1_5_group_size,
        row_panel=row_panel,
        k_iterations=gridK,
        output_footprint_bytes=tb_m * tb_n * out1_level[-1],
    )
