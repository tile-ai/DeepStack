"""Thin adapters from distributed phases to the existing local oracles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np

from ..gemm import (
    GemmBankResult,
    GemmLayoutSet,
    GemmProblem,
    make_layout_set,
    model_gemm_bank,
)
from ..layout import MatrixAccess, TensorLayout
from ..pipeline import BankAwareGemmPipelineResult, model_gemm_pipeline
from ..spec import DramBankSpec
from .common import LocalModelConfig


@dataclass(frozen=True)
class LocalPhaseResult:
    cluster: int
    problem: Optional[GemmProblem]
    bank: Optional[GemmBankResult]
    pipeline: Optional[BankAwareGemmPipelineResult]
    useful_flops: int
    modeled_flops: int

    @property
    def seconds(self) -> float:
        return 0.0 if self.pipeline is None else self.pipeline.total_seconds


def make_configured_layouts(
    problem: GemmProblem,
    config: LocalModelConfig,
    spec: DramBankSpec,
) -> GemmLayoutSet:
    return make_layout_set(
        problem,
        config.tiling,
        spec,
        preset=config.layout_preset,
        pitch_pad_sectors=config.pitch_pad_sectors,
        swizzle=config.swizzle,
        a_phase=config.a_phase,
        b_phase=config.b_phase,
        c_phase=config.c_phase,
        name="multi_cluster_%s" % config.layout_preset,
    )


def make_packed_shard_layouts(
    global_layouts: GemmLayoutSet,
    maximum_shard: GemmProblem,
    config: LocalModelConfig,
    spec: DramBankSpec,
) -> GemmLayoutSet:
    """Build owner-local shard strides while preserving global A/B/C anchors.

    Keeping the full allocation's three aligned base regions prevents a
    packed B or C shard from overlapping a canonical/global A allocation on
    an owner that happens to host both.  Only shape-dependent storage and
    batch strides become local to the maximum shard in the process grid.
    """

    local = make_configured_layouts(maximum_shard, config, spec)
    return replace(
        local,
        a=replace(
            local.a,
            base_offset_bytes=global_layouts.a.base_offset_bytes,
        ),
        b=replace(
            local.b,
            base_offset_bytes=global_layouts.b.base_offset_bytes,
        ),
        c=replace(
            local.c,
            base_offset_bytes=global_layouts.c.base_offset_bytes,
        ),
        name="%s:packed_shard" % global_layouts.name,
    )


def distributed_memory_outstanding_bytes(
    problem: GemmProblem,
    config: LocalModelConfig,
) -> int:
    """Working-set capacity exposed to the outer Little's-Law floor."""

    if config.bank_options.smem_buffer_bytes_per_cta is not None:
        per_cta = config.bank_options.smem_buffer_bytes_per_cta
    else:
        per_k = config.tiling.tb_k * (
            config.tiling.tb_m * problem.a_dtype_bytes
            + config.tiling.tb_n * problem.b_dtype_bytes
        )
        per_cta = config.tiling.pending_k_iterations * per_k
    return config.tiling.ctas_per_wave * per_cta


def _layout_for_batch(layout: TensorLayout, stride: int, batch: int) -> TensorLayout:
    if batch == 0 or stride == 0:
        return layout
    return replace(layout, base_offset_bytes=layout.base_offset_bytes + batch * stride)


def tensor_sectors(
    layouts: GemmLayoutSet,
    problem: GemmProblem,
    tensor: str,
    access: MatrixAccess,
    batch: int,
    spec: DramBankSpec,
) -> np.ndarray:
    if not 0 <= batch < problem.batch:
        raise ValueError("batch index is out of range")
    if tensor == "A":
        layout = _layout_for_batch(
            layouts.a, layouts.a_batch_stride_bytes, batch
        )
        shape = problem.a_shape
        dtype_bytes = problem.a_dtype_bytes
    elif tensor == "B":
        b_batch = 0 if problem.b_broadcast_across_batch else batch
        layout = _layout_for_batch(
            layouts.b, layouts.b_batch_stride_bytes, b_batch
        )
        shape = problem.b_shape
        dtype_bytes = problem.b_dtype_bytes
    elif tensor == "C":
        layout = _layout_for_batch(
            layouts.c, layouts.c_batch_stride_bytes, batch
        )
        shape = problem.c_shape
        dtype_bytes = problem.c_dtype_bytes
    else:
        raise ValueError("tensor must be A, B, or C")
    return layout.touched_sectors(
        shape,
        access,
        dtype_bytes,
        spec.sector_bytes,
    )


def tensor_block_sectors(
    layouts: GemmLayoutSet,
    tensor: str,
    shape: Tuple[int, int],
    dtype_bytes: int,
    batch: int,
    batch_count: int,
    spec: DramBankSpec,
    *,
    b_broadcast_across_batch: bool = False,
) -> np.ndarray:
    """Return sectors for a complete tensor block in a packed shard layout.

    ``layouts`` is normally built for the maximum block shape in the process
    grid.  Its aligned batch strides and A/B/C base regions are consequently
    stable across ragged ranks, while ``shape`` trims the actual final block.
    """

    if not 0 <= batch < batch_count:
        raise ValueError("batch index is out of range")
    if min(shape) <= 0 or dtype_bytes <= 0:
        raise ValueError("packed tensor shape and dtype must be positive")
    if tensor == "A":
        layout = _layout_for_batch(
            layouts.a, layouts.a_batch_stride_bytes, batch
        )
    elif tensor == "B":
        b_batch = 0 if b_broadcast_across_batch else batch
        layout = _layout_for_batch(
            layouts.b, layouts.b_batch_stride_bytes, b_batch
        )
    elif tensor == "C":
        layout = _layout_for_batch(
            layouts.c, layouts.c_batch_stride_bytes, batch
        )
    else:
        raise ValueError("tensor must be A, B, or C")
    return layout.touched_sectors(
        shape,
        MatrixAccess(0, shape[0], 0, shape[1]),
        dtype_bytes,
        spec.sector_bytes,
    )


def tensor_allocation_sector_interval(
    layouts: GemmLayoutSet,
    problem: GemmProblem,
    tensor: str,
    spec: DramBankSpec,
    *,
    sector_offset: int = 0,
) -> Tuple[int, int]:
    """Reserved half-open owner-sector range for one A/B/C allocation.

    A and B use the next aligned operand anchor as their end.  C has no next
    anchor, so its aligned per-batch stride determines the reserved extent.
    This intentionally includes allocator padding: another allocation cannot
    legally occupy an address merely because the current access skips it.
    """

    if tensor == "A":
        start_bytes = layouts.a.base_offset_bytes
        stop_bytes = layouts.b.base_offset_bytes
    elif tensor == "B":
        start_bytes = layouts.b.base_offset_bytes
        stop_bytes = layouts.c.base_offset_bytes
    elif tensor == "C":
        start_bytes = layouts.c.base_offset_bytes
        stop_bytes = (
            start_bytes + layouts.c_batch_stride_bytes * problem.batch
        )
    else:
        raise ValueError("tensor must be A, B, or C")
    if stop_bytes <= start_bytes or sector_offset < 0:
        raise ValueError("invalid tensor allocation interval")
    start = start_bytes // spec.sector_bytes + sector_offset
    stop = (
        (stop_bytes + spec.sector_bytes - 1) // spec.sector_bytes
        + sector_offset
    )
    return start, stop


def model_onchip_phase(
    cluster: int,
    problem: Optional[GemmProblem],
    config: LocalModelConfig,
    spec: DramBankSpec,
    *,
    useful_flops: int,
    charge_launch: bool,
    include_store_epilogue: bool = True,
) -> LocalPhaseResult:
    """Run compute/SMEM/L2 timing while distributed memory owns DDR reads.

    A/B are already materialized in a local or double-buffered working set;
    C is stored by the outer placement model.  Therefore this call suppresses
    all local DDR traffic but retains the existing compute and on-chip memory
    pipeline equations.
    """

    if problem is None:
        return LocalPhaseResult(cluster, None, None, None, useful_flops, 0)
    layouts = make_configured_layouts(problem, config, spec)
    options = replace(
        config.bank_options,
        a_miss_rate=0.0,
        b_miss_rate=0.0,
        c_write_rate=0.0,
        l2_cache=None,
    )
    bank = model_gemm_bank(
        problem,
        config.tiling,
        layouts,
        spec,
        options,
    )
    pipeline = model_gemm_pipeline(
        bank,
        config.arch,
        compute_cycles_per_k=config.compute_cycles_per_k,
        other_memory_cycles_per_k=config.other_memory_cycles_per_k,
        store_other_memory_cycles=(
            config.store_other_memory_cycles
            if include_store_epilogue
            else 0.0
        ),
        kernel_launch_seconds=(
            config.kernel_launch_seconds if charge_launch else 0.0
        ),
    )
    modeled_flops = 2 * problem.m * problem.n * problem.k * problem.batch
    return LocalPhaseResult(
        cluster,
        problem,
        bank,
        pipeline,
        useful_flops,
        modeled_flops,
    )
