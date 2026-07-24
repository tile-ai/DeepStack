"""Statistical GEMM-to-bank mapper built on representative memory waves.

The production path never enumerates tensor elements and never walks every
CTA/K iteration for a large GEMM.  It samples deterministic spatial/K bins,
builds exact sector histograms for those representatives, and weights the
results.  Small problems can set the sample limits to ``None`` to become an
exact tile/request oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tilesight.util.extract_blocks import extract_blocks

from .engine import (
    BankServiceResult,
    apply_littles_law_cycles,
    service_request_groups,
    service_requests,
)
from .layout import BankSwizzle, MatrixAccess, TensorLayout
from .l2_cache import (
    DeterministicFifoCache,
    GemmL2Result,
    L2CacheSpec,
    OperandStatsAccumulator,
)
from .parallel import ParallelBackend, map_jobs
from .spec import DramBankSpec


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, alignment: int) -> int:
    return _ceil_div(x, alignment) * alignment


@dataclass(frozen=True)
class GemmProblem:
    m: int
    n: int
    k: int
    batch: int = 1
    a_dtype_bytes: int = 2
    b_dtype_bytes: int = 2
    c_dtype_bytes: int = 4
    b_broadcast_across_batch: bool = False
    compute_dtype_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if min(self.m, self.n, self.k, self.batch) <= 0:
            raise ValueError("GEMM dimensions and batch must be positive")
        if min(self.a_dtype_bytes, self.b_dtype_bytes, self.c_dtype_bytes) <= 0:
            raise ValueError("dtype byte widths must be positive integers")
        if (
            self.a_dtype_bytes != self.b_dtype_bytes
            and self.compute_dtype_bytes is None
        ):
            raise ValueError(
                "mixed A/B precision requires explicit compute_dtype_bytes"
            )
        if (
            self.compute_dtype_bytes is not None
            and self.compute_dtype_bytes not in {1, 2, 4}
        ):
            raise ValueError("compute_dtype_bytes must be 1, 2, or 4")

    @property
    def a_shape(self) -> Tuple[int, int]:
        return self.m, self.k

    @property
    def b_shape(self) -> Tuple[int, int]:
        return self.k, self.n

    @property
    def c_shape(self) -> Tuple[int, int]:
        return self.m, self.n


@dataclass(frozen=True)
class GemmTiling:
    tb_m: int
    tb_n: int
    tb_k: int
    stage: int = 2
    ctas_per_wave: int = 8
    row_panel: int = 1
    column_panel: Optional[int] = None
    raster_axis: str = "legacy"

    def __post_init__(self) -> None:
        if min(self.tb_m, self.tb_n, self.tb_k, self.stage, self.ctas_per_wave) <= 0:
            raise ValueError("tile dimensions, stage, and ctas_per_wave must be positive")
        if self.row_panel <= 0:
            raise ValueError("row_panel must be positive")
        if self.column_panel is not None and self.column_panel <= 0:
            raise ValueError("column_panel must be positive")
        if self.raster_axis not in {"legacy", "along_m", "along_n"}:
            raise ValueError("invalid raster_axis")

    @property
    def pending_k_iterations(self) -> int:
        # TileSight defines software-pipeline depth as stage_num - 1.
        return max(1, self.stage - 1)


@dataclass(frozen=True)
class GemmLayoutSet:
    a: TensorLayout
    b: TensorLayout
    c: TensorLayout
    a_batch_stride_bytes: int
    b_batch_stride_bytes: int
    c_batch_stride_bytes: int
    name: str = ""


@dataclass(frozen=True)
class GemmBankOptions:
    connectivity: str = "direct"
    a_miss_rate: float = 1.0
    b_miss_rate: float = 1.0
    c_write_rate: float = 1.0
    coalesce_scope: str = "wave"
    # Four first/interior/last representatives are exact for the common
    # periodic GEMM patterns we validate and keep large sensitivity cases
    # inexpensive.
    max_spatial_samples: Optional[int] = 4
    max_k_samples: Optional[int] = 4
    sampling_mode: str = "representative"  # representative | phase_residue
    deterministic_seed: int = 0
    smem_buffer_bytes_per_cta: Optional[int] = None
    apply_littles_law: bool = True
    l2_cache: Optional[L2CacheSpec] = None
    l2_exact_profile_threshold: int = 4096

    def __post_init__(self) -> None:
        if self.connectivity not in {"direct", "interleaved"}:
            raise ValueError("connectivity must be direct or interleaved")
        if self.coalesce_scope not in {"request", "wave"}:
            raise ValueError("coalesce_scope must be request or wave")
        if self.sampling_mode not in {"representative", "phase_residue"}:
            raise ValueError(
                "sampling_mode must be representative or phase_residue"
            )
        if not 0.0 <= self.a_miss_rate <= 1.0:
            raise ValueError("a_miss_rate must be in [0, 1]")
        if not 0.0 <= self.b_miss_rate <= 1.0:
            raise ValueError("b_miss_rate must be in [0, 1]")
        if not 0.0 <= self.c_write_rate <= 1.0:
            raise ValueError("c_write_rate must be in [0, 1]")
        if self.l2_cache is not None and (
            self.a_miss_rate != 1.0 or self.b_miss_rate != 1.0
        ):
            raise ValueError(
                "deterministic L2 cannot be combined with a_miss_rate or "
                "b_miss_rate thinning"
            )
        if self.l2_exact_profile_threshold < 0:
            raise ValueError("l2_exact_profile_threshold must be non-negative")
        for limit in (self.max_spatial_samples, self.max_k_samples):
            if limit is not None and limit <= 0:
                raise ValueError("sample limits must be positive or None")


@dataclass(frozen=True)
class GemmWaveProfile:
    spatial_wave_index: int
    spatial_weight: int
    k_window_index: Optional[int]
    k_weight: int
    active_ctas: int
    k_iterations: int
    is_store: bool
    bank_cycles: float
    bank_lower_bound_cycles: float
    fixed_job_lower_bound_cycles: float
    final_cycles: float
    ideal_cycles: float
    ll_cycles: float
    ll_limited: bool
    transferred_bytes: int
    useful_bytes: int
    active_banks: int
    transaction_efficiency: float
    bank_balance_efficiency: float
    overall_efficiency: float


@dataclass(frozen=True)
class SpatialWaveSummary:
    spatial_wave_index: int
    spatial_weight: int
    active_ctas: int
    load_cycles_per_k: float
    ideal_load_cycles_per_k: float
    store_cycles: float
    ideal_store_cycles: float


@dataclass(frozen=True)
class GemmBankResult:
    problem: GemmProblem
    tiling: GemmTiling
    layouts: GemmLayoutSet
    options: GemmBankOptions
    spec: DramBankSpec
    spatial_waves: int
    k_windows: int
    sampled_profiles: Tuple[GemmWaveProfile, ...]
    spatial_summaries: Tuple[SpatialWaveSummary, ...]
    total_load_cycles: float
    total_store_cycles: float
    ideal_load_cycles: float
    ideal_store_cycles: float
    total_bank_load_cycles: float
    total_bank_store_cycles: float
    bank_load_lower_bound_cycles: float
    bank_store_lower_bound_cycles: float
    fixed_job_load_lower_bound_cycles: float
    fixed_job_store_lower_bound_cycles: float
    actual_read_bytes: float
    actual_write_bytes: float
    mean_transaction_efficiency: float
    mean_bank_balance_efficiency: float
    mean_active_bank_fraction: float
    runtime_seconds: float
    l2_result: Optional[GemmL2Result] = None

    @property
    def total_memory_cycles(self) -> float:
        return self.total_load_cycles + self.total_store_cycles

    @property
    def ideal_memory_cycles(self) -> float:
        return self.ideal_load_cycles + self.ideal_store_cycles

    @property
    def memory_attainment(self) -> float:
        if self.total_memory_cycles <= 0:
            return 1.0
        return self.ideal_memory_cycles / self.total_memory_cycles

    @property
    def bank_only_attainment(self) -> float:
        actual = self.total_bank_load_cycles + self.total_bank_store_cycles
        if actual <= 0:
            return 1.0
        lower_bound = (
            self.bank_load_lower_bound_cycles
            + self.bank_store_lower_bound_cycles
        )
        return lower_bound / actual

    @property
    def conflict_attainment(self) -> float:
        """Balance quality for the row jobs produced by this layout.

        Unlike raw bank balance, this does not penalize an operator merely
        because it has fewer independent jobs than banks.  Transaction-size
        efficiency remains a separate metric.
        """

        actual = self.total_bank_load_cycles + self.total_bank_store_cycles
        if actual <= 0:
            return 1.0
        lower_bound = (
            self.fixed_job_load_lower_bound_cycles
            + self.fixed_job_store_lower_bound_cycles
        )
        return lower_bound / actual

    @property
    def worst_sample_conflict_attainment(self) -> float:
        values = [
            profile.fixed_job_lower_bound_cycles / profile.bank_cycles
            for profile in self.sampled_profiles
            if profile.bank_cycles > 0
        ]
        return min(values, default=1.0)

    @property
    def total_memory_seconds(self) -> float:
        return self.spec.cycles_to_seconds(self.total_memory_cycles)

    @property
    def read_bandwidth_bytes_s(self) -> float:
        """DDR read bytes divided by the final (including LL) load time."""

        if self.total_load_cycles <= 0:
            return 0.0
        return self.actual_read_bytes / self.spec.cycles_to_seconds(
            self.total_load_cycles
        )

    @property
    def read_raw_port_utilization(self) -> float:
        """Effective raw-port utilization after bank service and Little's Law."""

        if self.total_load_cycles <= 0:
            return 0.0
        data_cycles = (
            self.actual_read_bytes
            / self.spec.sector_bytes
            * self.spec.sector_cycles
        )
        return data_cycles / (
            self.spec.port_count * self.total_load_cycles
        )

    @property
    def bank_read_raw_port_utilization(self) -> float:
        """Raw-port utilization of the bank scheduler before Little's Law."""

        if self.total_bank_load_cycles <= 0:
            return 0.0
        data_cycles = (
            self.actual_read_bytes
            / self.spec.sector_bytes
            * self.spec.sector_cycles
        )
        return data_cycles / (
            self.spec.port_count * self.total_bank_load_cycles
        )

    @property
    def read_topology_attainment(self) -> float:
        """Effective read utilization relative to the topology roof."""

        return self.read_raw_port_utilization / self.spec.connectivity_efficiency

    @property
    def bank_read_topology_attainment(self) -> float:
        """Bank-scheduler utilization relative to the topology roof."""

        return (
            self.bank_read_raw_port_utilization
            / self.spec.connectivity_efficiency
        )


@dataclass(frozen=True)
class _GemmL2Trace:
    """Full deterministic cache decisions; bank sampling reads selected cells."""

    result: GemmL2Result
    a_status: np.ndarray
    b_status: np.ndarray
    a_partial_misses: Dict[int, np.ndarray]
    b_partial_misses: Dict[int, np.ndarray]

    def filter_sectors(
        self,
        operand: str,
        cta_index: int,
        k_index: int,
        sectors: np.ndarray,
    ) -> np.ndarray:
        grid_k = int(self.a_status.shape[1])
        flat_index = cta_index * grid_k + k_index
        if operand == "a":
            status = int(self.a_status[cta_index, k_index])
            partial = self.a_partial_misses
        else:
            status = int(self.b_status[cta_index, k_index])
            partial = self.b_partial_misses
        if status == 0:
            return np.empty(0, dtype=np.int64)
        if status == 1:
            return sectors
        return partial[flat_index]


@dataclass(frozen=True)
class GemmLayoutSearchResult:
    best: GemmBankResult
    results: Tuple[GemmBankResult, ...]
    validated_results: Tuple[GemmBankResult, ...] = ()

    @property
    def best_conflict(self) -> GemmBankResult:
        """Candidate closest to the fixed-transaction conflict-free bound."""

        pool = self.validated_results or self.results
        return max(
            pool,
            key=lambda result: (
                result.conflict_attainment,
                result.worst_sample_conflict_attainment,
                -result.total_memory_cycles,
            ),
        )

    @property
    def validation_performed(self) -> bool:
        return bool(self.validated_results)


def _representative_bins(count: int, limit: Optional[int]) -> List[Tuple[int, int]]:
    """Return deterministic (representative index, integer weight) bins."""

    if count <= 0:
        return []
    if limit is None or count <= limit:
        return [(index, 1) for index in range(count)]
    if limit == 1:
        return [(count // 2, count)]
    if limit == 2:
        return [(0, 1), (count - 1, count - 1)]

    bins: List[Tuple[int, int]] = [(0, 1)]
    interior_count = count - 2
    interior_bins = limit - 2
    edges = np.linspace(0, interior_count, interior_bins + 1, dtype=np.int64)
    for left, right in zip(edges[:-1], edges[1:]):
        width = int(right - left)
        if width <= 0:
            continue
        # Translate [left, right) in the interior to [1, count-1).
        representative = 1 + int(left + (width - 1) // 2)
        bins.append((representative, width))
    bins.append((count - 1, 1))
    assert sum(weight for _, weight in bins) == count
    return bins


def _phase_residue_bins(
    count: int,
    limit: Optional[int],
    *,
    separate_last: bool,
) -> List[Tuple[int, int]]:
    """Cover consecutive modular phases, with a separate ragged tail.

    Uniformly spaced representative bins can alias a bank-conflict period.
    This mode instead visits the first ``limit`` consecutive residues and
    weights later indices by residue class.  A partial final CTA/K window is
    emitted separately because it need not match its full-window residue.
    """

    if count <= 0:
        return []
    if limit is None or count <= limit:
        return [(index, 1) for index in range(count)]
    period = int(limit)
    weights = [1 + (count - 1 - residue) // period for residue in range(period)]
    bins = [(residue, weight) for residue, weight in enumerate(weights)]
    last = count - 1
    if separate_last and last >= period:
        residue = last % period
        bins[residue] = (residue, bins[residue][1] - 1)
        bins = [item for item in bins if item[1] > 0]
        bins.append((last, 1))
    assert sum(weight for _, weight in bins) == count
    return bins


def _sampling_bins(
    count: int,
    limit: Optional[int],
    mode: str,
    *,
    separate_last: bool = False,
) -> List[Tuple[int, int]]:
    if mode == "phase_residue":
        return _phase_residue_bins(
            count, limit, separate_last=separate_last
        )
    return _representative_bins(count, limit)


def _stable_thin(sectors: np.ndarray, rate: float, salt: int) -> np.ndarray:
    """Apply a deterministic hash-threshold traffic sensitivity knob."""

    sectors = np.asarray(sectors, dtype=np.int64)
    if rate >= 1.0 or sectors.size == 0:
        return sectors
    if rate <= 0.0:
        return np.empty(0, dtype=np.int64)
    with np.errstate(over="ignore"):
        hashed = sectors.astype(np.uint64) ^ np.uint64(salt & ((1 << 64) - 1))
        hashed ^= hashed >> np.uint64(30)
        hashed *= np.uint64(0xBF58476D1CE4E5B9)
        hashed ^= hashed >> np.uint64(27)
        hashed *= np.uint64(0x94D049BB133111EB)
        hashed ^= hashed >> np.uint64(31)
    threshold = int(rate * (1 << 64))
    if threshold <= 0:
        return np.empty(0, dtype=np.int64)
    if threshold >= 1 << 64:
        return sectors
    return np.sort(sectors[hashed < np.uint64(threshold)])


def _cta_order(problem: GemmProblem, tiling: GemmTiling) -> np.ndarray:
    grid_m = _ceil_div(problem.m, tiling.tb_m)
    grid_n = _ceil_div(problem.n, tiling.tb_n)
    stride_m = (
        tiling.column_panel
        if tiling.column_panel is not None
        else max(tiling.ctas_per_wave // tiling.row_panel, 1)
    )
    coords = extract_blocks(
        grid_m,
        grid_n,
        int(stride_m),
        int(tiling.row_panel),
        raster_axis=tiling.raster_axis,
    ).astype(np.int64)
    coords -= 1  # TileSight's scheduler coordinates are one-based.
    batches = []
    for batch in range(problem.batch):
        batch_col = np.full((coords.shape[0], 1), batch, dtype=np.int64)
        batches.append(np.concatenate([batch_col, coords], axis=1))
    return np.concatenate(batches, axis=0)


def _layout_for_batch(layout: TensorLayout, stride: int, batch: int) -> TensorLayout:
    if stride == 0 or batch == 0:
        return layout
    return replace(layout, base_offset_bytes=layout.base_offset_bytes + stride * batch)


def make_layout_set(
    problem: GemmProblem,
    tiling: GemmTiling,
    spec: DramBankSpec,
    *,
    preset: str = "linear",
    a_order: str = "row_major",
    b_order: str = "row_major",
    c_order: str = "row_major",
    pitch_pad_sectors: int = 0,
    swizzle: BankSwizzle = BankSwizzle(),
    a_phase: int = 0,
    b_phase: int = 0,
    c_phase: int = 0,
    name: str = "",
) -> GemmLayoutSet:
    """Create non-overlapping A/B/C allocations for a named layout family."""

    if preset not in {"linear", "pitch_pad", "tile_major"}:
        raise ValueError("preset must be linear, pitch_pad, or tile_major")

    def operand_layout(
        order: str,
        tile_shape: Tuple[int, int],
        phase: int,
    ) -> TensorLayout:
        operand_swizzle = replace(swizzle, phase=(swizzle.phase + phase) % spec.bank_count)
        return TensorLayout(
            kind=preset,
            order=order,
            pitch_pad_sectors=pitch_pad_sectors,
            pitch_unit_bytes=spec.sector_bytes,
            tile_shape=tile_shape if preset == "tile_major" else None,
            swizzle=operand_swizzle,
        )

    a0 = operand_layout(a_order, (tiling.tb_m, tiling.tb_k), a_phase)
    b0 = operand_layout(b_order, (tiling.tb_k, tiling.tb_n), b_phase)
    c0 = operand_layout(c_order, (tiling.tb_m, tiling.tb_n), c_phase)

    allocation_alignment = spec.full_bank_wave_bytes
    a_stride = _align_up(
        a0.storage_nbytes(problem.a_shape, problem.a_dtype_bytes),
        allocation_alignment,
    )
    a_total = a_stride * problem.batch
    b_base = _align_up(a_total, allocation_alignment)
    b_stride = 0 if problem.b_broadcast_across_batch else _align_up(
        b0.storage_nbytes(problem.b_shape, problem.b_dtype_bytes),
        allocation_alignment,
    )
    b_total = (
        b0.storage_nbytes(problem.b_shape, problem.b_dtype_bytes)
        if b_stride == 0
        else b_stride * problem.batch
    )
    c_base = _align_up(b_base + b_total, allocation_alignment)
    c_stride = _align_up(
        c0.storage_nbytes(problem.c_shape, problem.c_dtype_bytes),
        allocation_alignment,
    )
    return GemmLayoutSet(
        a=replace(a0, base_offset_bytes=0),
        b=replace(b0, base_offset_bytes=b_base),
        c=replace(c0, base_offset_bytes=c_base),
        a_batch_stride_bytes=a_stride,
        b_batch_stride_bytes=b_stride,
        c_batch_stride_bytes=c_stride,
        name=name or preset,
    )


def default_layout_candidates(
    problem: GemmProblem,
    tiling: GemmTiling,
    spec: DramBankSpec,
) -> Tuple[GemmLayoutSet, ...]:
    """Small legal search set; identity is always included."""

    candidates = [
        make_layout_set(problem, tiling, spec, preset="linear", name="linear"),
        make_layout_set(
            problem, tiling, spec, preset="pitch_pad", pitch_pad_sectors=1,
            name="pitch_pad_1",
        ),
        make_layout_set(
            problem, tiling, spec, preset="pitch_pad", pitch_pad_sectors=3,
            name="pitch_pad_3",
        ),
        make_layout_set(problem, tiling, spec, preset="tile_major", name="tile_major"),
    ]
    if spec.bank_count & (spec.bank_count - 1) == 0:
        bank_bits = spec.bank_count.bit_length() - 1
        sector_bits = min(4, bank_bits)
        sector_shift = max(bank_bits - sector_bits, 0)
        candidates.extend(
            [
                make_layout_set(
                    problem,
                    tiling,
                    spec,
                    preset="linear",
                    swizzle=BankSwizzle(
                        kind="xor",
                        sector_bits=sector_bits,
                        sector_bank_shift=sector_shift,
                    ),
                    b_phase=spec.bank_count // 2,
                    c_phase=spec.bank_count // 4,
                    name="sector_xor",
                ),
                make_layout_set(
                    problem,
                    tiling,
                    spec,
                    preset="linear",
                    swizzle=BankSwizzle(kind="xor", row_bits=bank_bits),
                    b_phase=spec.bank_count // 2,
                    c_phase=spec.bank_count // 4,
                    name="row_xor",
                ),
                make_layout_set(
                    problem,
                    tiling,
                    spec,
                    preset="tile_major",
                    swizzle=BankSwizzle(
                        kind="xor",
                        sector_bits=sector_bits,
                        sector_bank_shift=sector_shift,
                        row_bits=bank_bits,
                    ),
                    b_phase=spec.bank_count // 2,
                    c_phase=spec.bank_count // 4,
                    name="tile_xor",
                ),
            ]
        )
        # Do not force sector and row hashes to be combined.  Their best
        # choice depends on the caller-supplied tile and row geometry.
        tile_sector_widths = sorted({min(3, bank_bits, 4), min(4, bank_bits)})
        for width in tile_sector_widths:
            if width <= 0:
                continue
            target_shift = min(2, bank_bits - width)
            candidates.append(
                make_layout_set(
                    problem,
                    tiling,
                    spec,
                    preset="tile_major",
                    swizzle=BankSwizzle(
                        kind="xor",
                        sector_bits=width,
                        sector_bank_shift=target_shift,
                    ),
                    b_phase=spec.bank_count // 2,
                    c_phase=spec.bank_count // 4,
                    name=f"tile_sector_xor_{width}",
                )
            )
            if width == min(4, bank_bits):
                candidates.append(
                    make_layout_set(
                        problem,
                        tiling,
                        spec,
                        preset="tile_major",
                        swizzle=BankSwizzle(
                            kind="xor",
                            sector_bits=width,
                            sector_bank_shift=target_shift,
                        ),
                        name="tile_sector_xor_4_phase0",
                    )
                )
        candidates.append(
            make_layout_set(
                problem,
                tiling,
                spec,
                preset="tile_major",
                swizzle=BankSwizzle(kind="xor", row_bits=bank_bits),
                b_phase=spec.bank_count // 2,
                c_phase=spec.bank_count // 4,
                name="tile_row_xor",
            )
        )
        # Compact stage/layout joint-search shortlist.  These remain static
        # address transforms; stage changes only the number of pending K
        # iterations.  Different pending windows expose different useful bank
        # bits, so keeping several legal hashes is better than tying stage to
        # an address formula.
        extra_tile_hashes = (
            (2, 3, 0, "tile_sector_xor_2_shift3"),
            (3, 3, 0, "tile_sector_xor_3_shift3"),
            (1, 5, bank_bits, "tile_sector_row_xor_1_5"),
        )
        for width, target_shift, row_width, candidate_name in extra_tile_hashes:
            if width > 4 or target_shift + width > bank_bits:
                continue
            candidates.append(
                make_layout_set(
                    problem,
                    tiling,
                    spec,
                    preset="tile_major",
                    swizzle=BankSwizzle(
                        kind="xor",
                        sector_bits=width,
                        sector_bank_shift=target_shift,
                        row_bits=row_width,
                    ),
                    b_phase=spec.bank_count // 2,
                    c_phase=spec.bank_count // 4,
                    name=candidate_name,
                )
            )
    candidates.append(
        make_layout_set(
            problem,
            tiling,
            spec,
            preset="linear",
            swizzle=BankSwizzle(
                kind="cyclic", cyclic_sector_alpha=1, cyclic_row_beta=3
            ),
            b_phase=spec.bank_count // 2,
            c_phase=spec.bank_count // 4,
            name="bank_cyclic",
        )
    )
    return tuple(candidates)


class _RequestBuilder:
    def __init__(
        self,
        problem: GemmProblem,
        tiling: GemmTiling,
        layouts: GemmLayoutSet,
        spec: DramBankSpec,
        options: GemmBankOptions,
    ) -> None:
        self.problem = problem
        self.tiling = tiling
        self.layouts = layouts
        self.spec = spec
        self.options = options
        self.cache: Dict[Tuple[object, ...], np.ndarray] = {}

    def _raw_sectors(
        self,
        operand: str,
        batch: int,
        access: MatrixAccess,
    ) -> np.ndarray:
        if operand == "a":
            layout = _layout_for_batch(
                self.layouts.a, self.layouts.a_batch_stride_bytes, batch
            )
            shape = self.problem.a_shape
            dtype_bytes = self.problem.a_dtype_bytes
        elif operand == "b":
            layout = _layout_for_batch(
                self.layouts.b, self.layouts.b_batch_stride_bytes, batch
            )
            shape = self.problem.b_shape
            dtype_bytes = self.problem.b_dtype_bytes
        else:
            layout = _layout_for_batch(
                self.layouts.c, self.layouts.c_batch_stride_bytes, batch
            )
            shape = self.problem.c_shape
            dtype_bytes = self.problem.c_dtype_bytes
        key = (
            operand,
            batch,
            access.row_start,
            access.row_stop,
            access.col_start,
            access.col_stop,
        )
        cached = self.cache.get(key)
        if cached is None:
            cached = layout.touched_sectors(
                shape, access, dtype_bytes, self.spec.sector_bytes
            )
            self.cache[key] = cached
        return cached

    def _sectors(
        self,
        operand: str,
        batch: int,
        access: MatrixAccess,
        request_salt: int,
        *,
        l2_miss: bool = True,
    ) -> np.ndarray:
        cached = self._raw_sectors(operand, batch, access)
        if not l2_miss:
            return np.empty(0, dtype=np.int64)
        if operand == "a":
            miss_rate = self.options.a_miss_rate
        elif operand == "b":
            miss_rate = self.options.b_miss_rate
        else:
            miss_rate = self.options.c_write_rate
        if miss_rate < 1.0:
            return _stable_thin(
                cached,
                miss_rate,
                self.options.deterministic_seed ^ request_salt,
            )
        return cached

    def load_requests(
        self,
        ctas: np.ndarray,
        k_start: int,
        k_stop: int,
        *,
        global_cta_start: int = 0,
        l2_trace: Optional[_GemmL2Trace] = None,
    ) -> Tuple[List[np.ndarray], int, int]:
        requests: List[np.ndarray] = []
        useful_bytes = 0
        tile = self.tiling
        for k_index in range(k_start, k_stop):
            k0 = k_index * tile.tb_k
            k1 = min(k0 + tile.tb_k, self.problem.k)
            for cta_index, (batch, m_index, n_index) in enumerate(ctas):
                m0 = int(m_index) * tile.tb_m
                m1 = min(m0 + tile.tb_m, self.problem.m)
                n0 = int(n_index) * tile.tb_n
                n1 = min(n0 + tile.tb_n, self.problem.n)
                a_access = MatrixAccess(m0, m1, k0, k1)
                b_access = MatrixAccess(k0, k1, n0, n1)
                salt_base = (
                    (int(batch) + 1) * 0x9E3779B1
                    ^ (int(m_index) + 3) * 0x85EBCA77
                    ^ (int(n_index) + 5) * 0xC2B2AE3D
                    ^ (k_index + 7) * 0x27D4EB2F
                    ^ cta_index
                )
                global_cta = global_cta_start + cta_index
                a_sectors = self._sectors(
                    "a", int(batch), a_access, salt_base
                )
                b_sectors = self._sectors(
                    "b",
                    int(batch),
                    b_access,
                    salt_base ^ 0xA5A5A5A5,
                )
                if l2_trace is not None:
                    a_sectors = l2_trace.filter_sectors(
                        "a", global_cta, k_index, a_sectors
                    )
                    b_sectors = l2_trace.filter_sectors(
                        "b", global_cta, k_index, b_sectors
                    )
                requests.append(a_sectors)
                requests.append(b_sectors)
                useful_bytes += (
                    (m1 - m0) * (k1 - k0) * self.problem.a_dtype_bytes
                    + (k1 - k0) * (n1 - n0) * self.problem.b_dtype_bytes
                )
        tile_bytes_per_cta = (
            tile.tb_m * tile.tb_k * self.problem.a_dtype_bytes
            + tile.tb_k * tile.tb_n * self.problem.b_dtype_bytes
        )
        return requests, useful_bytes, tile_bytes_per_cta

    def store_requests(self, ctas: np.ndarray) -> Tuple[List[np.ndarray], int, int]:
        requests: List[np.ndarray] = []
        useful_bytes = 0
        max_tile_bytes = self.tiling.tb_m * self.tiling.tb_n * self.problem.c_dtype_bytes
        for batch, m_index, n_index in ctas:
            m0 = int(m_index) * self.tiling.tb_m
            m1 = min(m0 + self.tiling.tb_m, self.problem.m)
            n0 = int(n_index) * self.tiling.tb_n
            n1 = min(n0 + self.tiling.tb_n, self.problem.n)
            access = MatrixAccess(m0, m1, n0, n1)
            requests.append(self._sectors("c", int(batch), access, 0))
            useful_bytes += (m1 - m0) * (n1 - n0) * self.problem.c_dtype_bytes
        return requests, useful_bytes, max_tile_bytes


def _fifo_sector_request(
    cache: DeterministicFifoCache,
    sectors: np.ndarray,
    stats: OperandStatsAccumulator,
    request_id: Tuple[object, ...],
    resident_memo: Dict[Tuple[object, ...], int],
) -> Tuple[int, Optional[np.ndarray]]:
    """Access one sector set and return 0=hit, 1=miss, 2=partial."""

    total = int(sectors.size)
    if total <= 0:
        raise ValueError("an L2 tile request must touch at least one sector")
    memo_sequence = resident_memo.get(request_id)
    if (
        memo_sequence is not None
        and memo_sequence >= cache.oldest_resident_sequence
    ):
        stats.add_line_request(total_lines=total, hit_lines=total)
        return 0, None

    hit_count = 0
    partial_misses: Optional[List[int]] = None
    minimum_sequence: Optional[int] = None
    for position, sector in enumerate(sectors):
        status = cache.access_line(int(sector))
        sequence = cache.last_line_sequence
        minimum_sequence = (
            sequence
            if minimum_sequence is None
            else min(minimum_sequence, sequence)
        )
        if status == 0:
            if hit_count == 0 and position:
                # All preceding lines missed; materialize them only after the
                # first hit proves this is a genuinely partial request.
                partial_misses = [int(value) for value in sectors[:position]]
            elif partial_misses is None:
                partial_misses = []
            hit_count += 1
        else:
            if hit_count:
                assert partial_misses is not None
                partial_misses.append(int(sector))

    stats.add_line_request(
        total_lines=total,
        hit_lines=hit_count,
    )
    assert minimum_sequence is not None
    if minimum_sequence >= cache.oldest_resident_sequence:
        resident_memo[request_id] = minimum_sequence
    else:
        resident_memo.pop(request_id, None)
    if hit_count == total:
        return 0, None
    if hit_count == 0:
        return 1, None
    assert partial_misses is not None
    return 2, np.asarray(partial_misses, dtype=np.int64)


def _build_gemm_l2_trace(
    problem: GemmProblem,
    tiling: GemmTiling,
    builder: _RequestBuilder,
    cta_order: np.ndarray,
    spec: DramBankSpec,
    cache_spec: L2CacheSpec,
) -> _GemmL2Trace:
    """Simulate the complete deterministic CTA/K FIFO trace.

    Bank timing may subsequently sample representative waves, but cache state
    is never advanced by weighted representatives.  Walking the full
    tile-level trace avoids the otherwise incorrect cold restart between
    sampled waves, especially for GEMV-like cross-CTA reuse.
    """

    if cache_spec.line_bytes != spec.sector_bytes:
        raise ValueError(
            "deterministic L2 line_bytes must match DRAM sector_bytes"
        )

    grid_k = _ceil_div(problem.k, tiling.tb_k)
    grid_m = _ceil_div(problem.m, tiling.tb_m)
    grid_n = _ceil_div(problem.n, tiling.tb_n)
    cta_count = int(cta_order.shape[0])
    a_status = np.empty((cta_count, grid_k), dtype=np.uint8)
    b_status = np.empty((cta_count, grid_k), dtype=np.uint8)
    a_partial_misses: Dict[int, np.ndarray] = {}
    b_partial_misses: Dict[int, np.ndarray] = {}
    cache = DeterministicFifoCache(cache_spec)
    a_stats = OperandStatsAccumulator()
    b_stats = OperandStatsAccumulator()
    output_stats = OperandStatsAccumulator()
    resident_memo: Dict[Tuple[object, ...], int] = {}

    # Sector enumeration is layout-aware but depends only on a unique logical
    # tile, not on every reuse event.  Precompute those footprints once so the
    # full FIFO walk remains O(tile accesses) with cheap array indexing.
    a_sector_blocks = np.empty(
        (problem.batch, grid_m, grid_k), dtype=object
    )
    b_batches = 1 if problem.b_broadcast_across_batch else problem.batch
    b_sector_blocks = np.empty((b_batches, grid_k, grid_n), dtype=object)
    c_sector_blocks = np.empty(
        (problem.batch, grid_m, grid_n), dtype=object
    )
    for batch_i in range(problem.batch):
        for m_i in range(grid_m):
            m0 = m_i * tiling.tb_m
            m1 = min(m0 + tiling.tb_m, problem.m)
            for k_index in range(grid_k):
                k0 = k_index * tiling.tb_k
                k1 = min(k0 + tiling.tb_k, problem.k)
                sectors = builder._raw_sectors(
                    "a", batch_i, MatrixAccess(m0, m1, k0, k1)
                )
                a_sector_blocks[batch_i, m_i, k_index] = sectors
        for m_i in range(grid_m):
            m0 = m_i * tiling.tb_m
            m1 = min(m0 + tiling.tb_m, problem.m)
            for n_i in range(grid_n):
                n0 = n_i * tiling.tb_n
                n1 = min(n0 + tiling.tb_n, problem.n)
                sectors = builder._raw_sectors(
                    "c", batch_i, MatrixAccess(m0, m1, n0, n1)
                )
                c_sector_blocks[batch_i, m_i, n_i] = sectors
    for batch_i in range(b_batches):
        for k_index in range(grid_k):
            k0 = k_index * tiling.tb_k
            k1 = min(k0 + tiling.tb_k, problem.k)
            for n_i in range(grid_n):
                n0 = n_i * tiling.tb_n
                n1 = min(n0 + tiling.tb_n, problem.n)
                sectors = builder._raw_sectors(
                    "b", batch_i, MatrixAccess(k0, k1, n0, n1)
                )
                b_sector_blocks[batch_i, k_index, n_i] = sectors

    for wave_start in range(0, cta_count, tiling.ctas_per_wave):
        wave_stop = min(wave_start + tiling.ctas_per_wave, cta_count)
        for k_index in range(grid_k):
            for cta_index in range(wave_start, wave_stop):
                batch, m_index, n_index = cta_order[cta_index]
                batch_i = int(batch)
                m_i = int(m_index)
                n_i = int(n_index)

                flat_index = cta_index * grid_k + k_index
                status, partial = _fifo_sector_request(
                    cache,
                    a_sector_blocks[batch_i, m_i, k_index],
                    a_stats,
                    ("a", batch_i, m_i, k_index),
                    resident_memo,
                )
                a_status[cta_index, k_index] = status
                if partial is not None:
                    a_partial_misses[flat_index] = partial

                # A broadcast B has one physical allocation and therefore one
                # precomputed address-set identity across batches.
                b_cache_batch = (
                    0 if problem.b_broadcast_across_batch else batch_i
                )
                status, partial = _fifo_sector_request(
                    cache,
                    b_sector_blocks[b_cache_batch, k_index, n_i],
                    b_stats,
                    ("b", b_cache_batch, k_index, n_i),
                    resident_memo,
                )
                b_status[cta_index, k_index] = status
                if partial is not None:
                    b_partial_misses[flat_index] = partial

        if cache_spec.output_pressure:
            for cta_index in range(wave_start, wave_stop):
                batch, m_index, n_index = cta_order[cta_index]
                batch_i = int(batch)
                m_i = int(m_index)
                n_i = int(n_index)
                _fifo_sector_request(
                    cache,
                    c_sector_blocks[batch_i, m_i, n_i],
                    output_stats,
                    ("c", batch_i, m_i, n_i),
                    resident_memo,
                )

    result = GemmL2Result(
        spec=cache_spec,
        a=a_stats.freeze(),
        b=b_stats.freeze(),
        output_pressure=output_stats.freeze(),
        peak_resident_lines=cache.peak_resident_lines,
        final_resident_lines=cache.resident_lines,
        trace_accesses=(
            a_stats.accesses + b_stats.accesses + output_stats.accesses
        ),
    )
    return _GemmL2Trace(
        result,
        a_status,
        b_status,
        a_partial_misses,
        b_partial_misses,
    )


def _l2_layout_signature(layouts: GemmLayoutSet) -> GemmLayoutSet:
    """Remove bank-only permutations from the logical L2 address signature."""

    def logical(layout: TensorLayout) -> TensorLayout:
        return replace(
            layout,
            swizzle=BankSwizzle(),
            name="",
        )

    return replace(
        layouts,
        a=logical(layouts.a),
        b=logical(layouts.b),
        c=logical(layouts.c),
        name="",
    )


def _l2_tiling_signature(tiling: GemmTiling) -> GemmTiling:
    """Pipeline depth groups bank requests but does not reorder L2 accesses."""

    return replace(tiling, stage=1)


@lru_cache(maxsize=8)
def _cached_gemm_l2_trace(
    problem: GemmProblem,
    tiling: GemmTiling,
    layouts: GemmLayoutSet,
    spec: DramBankSpec,
    cache_spec: L2CacheSpec,
) -> _GemmL2Trace:
    """Reuse a full FIFO trace across candidates differing only by swizzle."""

    raw_options = GemmBankOptions(apply_littles_law=False)
    builder = _RequestBuilder(problem, tiling, layouts, spec, raw_options)
    return _build_gemm_l2_trace(
        problem,
        tiling,
        builder,
        _cta_order(problem, tiling),
        spec,
        cache_spec,
    )


def _finish_profile_service(
    service: BankServiceResult,
    tile_bytes_per_cta: int,
    active_ctas: int,
    resident_k_iterations: int,
    spec: DramBankSpec,
    options: GemmBankOptions,
) -> Tuple[BankServiceResult, float, float, bool, float]:
    if options.smem_buffer_bytes_per_cta is None:
        per_cta_buffer = resident_k_iterations * tile_bytes_per_cta
    else:
        per_cta_buffer = options.smem_buffer_bytes_per_cta
    # Outstanding capacity is the smaller of the resident SMEM resource and
    # the unique DDR traffic in this pending window; the Little's-Law helper
    # applies the latter finite-window cap.
    outstanding = float(max(active_ctas, 1) * per_cta_buffer)
    if options.apply_littles_law:
        final, limited, ll_cycles = apply_littles_law_cycles(
            service.cycles,
            service.transferred_bytes,
            outstanding,
            spec,
        )
    else:
        final, limited, ll_cycles = service.cycles, False, 0.0
    ideal = max(service.sector_lower_bound_cycles, ll_cycles)
    return service, float(final), float(ideal), bool(limited), float(ll_cycles)


def _profile_service(
    requests: Sequence[np.ndarray],
    tile_bytes_per_cta: int,
    active_ctas: int,
    resident_k_iterations: int,
    spec: DramBankSpec,
    options: GemmBankOptions,
    swizzle: BankSwizzle,
) -> Tuple[BankServiceResult, float, float, bool, float]:
    service = service_requests(
        requests,
        spec,
        connectivity=options.connectivity,
        swizzle=swizzle,
        coalesce_scope=options.coalesce_scope,
    )
    return _finish_profile_service(
        service,
        tile_bytes_per_cta,
        active_ctas,
        resident_k_iterations,
        spec,
        options,
    )


def model_gemm_bank(
    problem: GemmProblem,
    tiling: GemmTiling,
    layouts: GemmLayoutSet,
    spec: DramBankSpec = DramBankSpec(),
    options: GemmBankOptions = GemmBankOptions(),
) -> GemmBankResult:
    started = time.perf_counter()
    grid_k = _ceil_div(problem.k, tiling.tb_k)
    k_window = tiling.pending_k_iterations
    k_windows = _ceil_div(grid_k, k_window)
    cta_order = _cta_order(problem, tiling)
    spatial_waves = _ceil_div(cta_order.shape[0], tiling.ctas_per_wave)
    exact_cache_profiles = (
        options.l2_cache is not None
        and options.l2_exact_profile_threshold > 0
        and spatial_waves * k_windows
        <= options.l2_exact_profile_threshold
    )
    # A one-bin cache-aware sample can select only a warm interior wave and
    # weight away the compulsory cold wave.  Preserve at least the first/last
    # cache states; larger limits retain the normal statistical policy.
    spatial_sample_limit = (
        None if exact_cache_profiles else options.max_spatial_samples
    )
    if (
        options.l2_cache is not None
        and spatial_waves > 1
        and spatial_sample_limit == 1
    ):
        spatial_sample_limit = 2
    spatial_bins = _sampling_bins(
        spatial_waves,
        spatial_sample_limit,
        options.sampling_mode,
        separate_last=(cta_order.shape[0] % tiling.ctas_per_wave != 0),
    )
    k_bins = _sampling_bins(
        k_windows,
        None if exact_cache_profiles else options.max_k_samples,
        options.sampling_mode,
        separate_last=(grid_k % k_window != 0),
    )
    builder = _RequestBuilder(problem, tiling, layouts, spec, options)
    l2_trace = (
        None
        if options.l2_cache is None
        else _cached_gemm_l2_trace(
            problem,
            _l2_tiling_signature(tiling),
            _l2_layout_signature(layouts),
            spec,
            options.l2_cache,
        )
    )

    profiles: List[GemmWaveProfile] = []
    summaries: List[SpatialWaveSummary] = []
    total_load_cycles = 0.0
    total_store_cycles = 0.0
    ideal_load_cycles = 0.0
    ideal_store_cycles = 0.0
    total_bank_load_cycles = 0.0
    total_bank_store_cycles = 0.0
    bank_load_lower_bound_cycles = 0.0
    bank_store_lower_bound_cycles = 0.0
    fixed_job_load_lower_bound_cycles = 0.0
    fixed_job_store_lower_bound_cycles = 0.0
    actual_read_bytes = 0.0
    actual_write_bytes = 0.0
    txn_weighted = 0.0
    balance_weighted = 0.0
    active_weighted = 0.0
    profile_weight_total = 0.0

    for spatial_index, spatial_weight in spatial_bins:
        cta_start = spatial_index * tiling.ctas_per_wave
        ctas = cta_order[cta_start:cta_start + tiling.ctas_per_wave]
        active_ctas = int(ctas.shape[0])
        spatial_load_cycles = 0.0
        spatial_ideal_load_cycles = 0.0

        for k_index, k_weight in k_bins:
            k_start = k_index * k_window
            k_stop = min(k_start + k_window, grid_k)
            requests, useful_bytes, tile_bytes_per_cta = builder.load_requests(
                ctas,
                k_start,
                k_stop,
                global_cta_start=cta_start,
                l2_trace=l2_trace,
            )
            # Requests are interleaved A,B.  Keep operands in distinct
            # coalescing domains even when their swizzles happen to match:
            # A and B are separate tensors, while all decoded jobs still
            # contend in one physical-bank/shared-port schedule.
            a_requests = requests[0::2]
            b_requests = requests[1::2]
            grouped_service = service_request_groups(
                (
                    (a_requests, layouts.a.swizzle),
                    (b_requests, layouts.b.swizzle),
                ),
                spec,
                connectivity=options.connectivity,
                coalesce_scope=options.coalesce_scope,
            )
            service, final, ideal, limited, ll_cycles = (
                _finish_profile_service(
                    grouped_service,
                    tile_bytes_per_cta,
                    active_ctas,
                    k_stop - k_start,
                    spec,
                    options,
                )
            )
            weighted = spatial_weight * k_weight
            total_load_cycles += final * weighted
            ideal_load_cycles += ideal * weighted
            total_bank_load_cycles += service.cycles * weighted
            bank_load_lower_bound_cycles += (
                service.sector_lower_bound_cycles * weighted
            )
            fixed_job_load_lower_bound_cycles += (
                service.fixed_job_lower_bound_cycles * weighted
            )
            actual_read_bytes += service.transferred_bytes * weighted
            spatial_load_cycles += final * k_weight
            spatial_ideal_load_cycles += ideal * k_weight
            txn_weighted += service.transaction_efficiency * weighted
            balance_weighted += service.bank_balance_efficiency * weighted
            active_weighted += service.active_banks / spec.bank_count * weighted
            profile_weight_total += weighted
            profiles.append(
                GemmWaveProfile(
                    spatial_wave_index=spatial_index,
                    spatial_weight=spatial_weight,
                    k_window_index=k_index,
                    k_weight=k_weight,
                    active_ctas=active_ctas,
                    k_iterations=k_stop - k_start,
                    is_store=False,
                    bank_cycles=service.cycles,
                    bank_lower_bound_cycles=service.sector_lower_bound_cycles,
                    fixed_job_lower_bound_cycles=service.fixed_job_lower_bound_cycles,
                    final_cycles=final,
                    ideal_cycles=ideal,
                    ll_cycles=ll_cycles,
                    ll_limited=limited,
                    transferred_bytes=service.transferred_bytes,
                    useful_bytes=useful_bytes,
                    active_banks=service.active_banks,
                    transaction_efficiency=service.transaction_efficiency,
                    bank_balance_efficiency=service.bank_balance_efficiency,
                    overall_efficiency=service.overall_efficiency,
                )
            )

        store_requests_, store_useful, store_tile_bytes = builder.store_requests(ctas)
        store_service, store_final, store_ideal, store_limited, store_ll = _profile_service(
            store_requests_,
            store_tile_bytes,
            active_ctas,
            1,
            spec,
            options,
            layouts.c.swizzle,
        )
        total_store_cycles += store_final * spatial_weight
        ideal_store_cycles += store_ideal * spatial_weight
        total_bank_store_cycles += store_service.cycles * spatial_weight
        bank_store_lower_bound_cycles += (
            store_service.sector_lower_bound_cycles * spatial_weight
        )
        fixed_job_store_lower_bound_cycles += (
            store_service.fixed_job_lower_bound_cycles * spatial_weight
        )
        actual_write_bytes += store_service.transferred_bytes * spatial_weight
        txn_weighted += store_service.transaction_efficiency * spatial_weight
        balance_weighted += store_service.bank_balance_efficiency * spatial_weight
        active_weighted += (
            store_service.active_banks / spec.bank_count * spatial_weight
        )
        profile_weight_total += spatial_weight
        profiles.append(
            GemmWaveProfile(
                spatial_wave_index=spatial_index,
                spatial_weight=spatial_weight,
                k_window_index=None,
                k_weight=1,
                active_ctas=active_ctas,
                k_iterations=0,
                is_store=True,
                bank_cycles=store_service.cycles,
                bank_lower_bound_cycles=store_service.sector_lower_bound_cycles,
                fixed_job_lower_bound_cycles=store_service.fixed_job_lower_bound_cycles,
                final_cycles=store_final,
                ideal_cycles=store_ideal,
                ll_cycles=store_ll,
                ll_limited=store_limited,
                transferred_bytes=store_service.transferred_bytes,
                useful_bytes=store_useful,
                active_banks=store_service.active_banks,
                transaction_efficiency=store_service.transaction_efficiency,
                bank_balance_efficiency=store_service.bank_balance_efficiency,
                overall_efficiency=store_service.overall_efficiency,
            )
        )
        summaries.append(
            SpatialWaveSummary(
                spatial_wave_index=spatial_index,
                spatial_weight=spatial_weight,
                active_ctas=active_ctas,
                load_cycles_per_k=spatial_load_cycles / grid_k,
                ideal_load_cycles_per_k=spatial_ideal_load_cycles / grid_k,
                store_cycles=store_final,
                ideal_store_cycles=store_ideal,
            )
        )

    runtime = time.perf_counter() - started
    denominator = max(profile_weight_total, 1.0)
    return GemmBankResult(
        problem=problem,
        tiling=tiling,
        layouts=layouts,
        options=options,
        spec=spec,
        spatial_waves=spatial_waves,
        k_windows=k_windows,
        sampled_profiles=tuple(profiles),
        spatial_summaries=tuple(summaries),
        total_load_cycles=total_load_cycles,
        total_store_cycles=total_store_cycles,
        ideal_load_cycles=ideal_load_cycles,
        ideal_store_cycles=ideal_store_cycles,
        total_bank_load_cycles=total_bank_load_cycles,
        total_bank_store_cycles=total_bank_store_cycles,
        bank_load_lower_bound_cycles=bank_load_lower_bound_cycles,
        bank_store_lower_bound_cycles=bank_store_lower_bound_cycles,
        fixed_job_load_lower_bound_cycles=fixed_job_load_lower_bound_cycles,
        fixed_job_store_lower_bound_cycles=fixed_job_store_lower_bound_cycles,
        actual_read_bytes=actual_read_bytes,
        actual_write_bytes=actual_write_bytes,
        mean_transaction_efficiency=txn_weighted / denominator,
        mean_bank_balance_efficiency=balance_weighted / denominator,
        mean_active_bank_fraction=active_weighted / denominator,
        runtime_seconds=runtime,
        l2_result=None if l2_trace is None else l2_trace.result,
    )


@dataclass(frozen=True)
class _GemmModelJob:
    """Pickleable candidate job for thread/process layout search."""

    problem: GemmProblem
    tiling: GemmTiling
    layouts: GemmLayoutSet
    spec: DramBankSpec
    options: GemmBankOptions


def _evaluate_gemm_model_job(job: _GemmModelJob) -> GemmBankResult:
    """Module-level process worker; keep closures out of ProcessPool jobs."""

    return model_gemm_bank(
        job.problem,
        job.tiling,
        job.layouts,
        job.spec,
        job.options,
    )


def _gemm_model_work_units(
    problem: GemmProblem,
    tiling: GemmTiling,
    options: GemmBankOptions,
) -> int:
    """Cheap estimate used only to amortize automatic process startup."""

    grid_m = _ceil_div(problem.m, tiling.tb_m)
    grid_n = _ceil_div(problem.n, tiling.tb_n)
    cta_count = problem.batch * grid_m * grid_n
    spatial_waves = _ceil_div(cta_count, tiling.ctas_per_wave)
    grid_k = _ceil_div(problem.k, tiling.tb_k)
    pending_k = tiling.pending_k_iterations
    k_windows = _ceil_div(grid_k, pending_k)
    exact_cache_profiles = (
        options.l2_cache is not None
        and options.l2_exact_profile_threshold > 0
        and spatial_waves * k_windows
        <= options.l2_exact_profile_threshold
    )
    spatial_samples = len(
        _sampling_bins(
            spatial_waves,
            None if exact_cache_profiles else options.max_spatial_samples,
            options.sampling_mode,
            separate_last=(cta_count % tiling.ctas_per_wave != 0),
        )
    )

    k_samples = len(
        _sampling_bins(
            k_windows,
            None if exact_cache_profiles else options.max_k_samples,
            options.sampling_mode,
            separate_last=(grid_k % pending_k != 0),
        )
    )
    active_ctas = min(cta_count, tiling.ctas_per_wave)
    # Every model materializes the CTA raster once.  Each sampled load profile
    # then visits active CTAs across a pending-K window; stores add one visit per
    # sampled spatial wave.  This need not predict wall time precisely—it only
    # separates tiny searches from work that can amortize process startup.
    profile_work = spatial_samples * active_ctas * (
        k_samples * pending_k + 1
    )
    cache_work = 0
    if options.l2_cache is not None:
        cache_work = 2 * cta_count * _ceil_div(problem.k, tiling.tb_k)
    return max(cta_count + profile_work + cache_work, 1)


def search_gemm_layouts(
    problem: GemmProblem,
    tiling: GemmTiling,
    spec: DramBankSpec = DramBankSpec(),
    options: GemmBankOptions = GemmBankOptions(),
    candidates: Optional[Sequence[GemmLayoutSet]] = None,
    workers: int = 0,
    validation_top_k: int = 0,
    validation_phase_samples: int = 64,
    backend: ParallelBackend = "auto",
) -> GemmLayoutSearchResult:
    """Search layouts with serial, thread, process, or automatic execution.

    ``auto`` preserves serial execution for small searches and selects a
    process pool only when ``workers > 1`` and the estimated mapper work can
    amortize startup.  It never auto-selects threads.
    """

    if candidates is None:
        candidates = default_layout_candidates(problem, tiling, spec)
    if not candidates:
        raise ValueError("at least one layout candidate is required")
    candidate_tuple = tuple(candidates)
    jobs = tuple(
        _GemmModelJob(problem, tiling, layouts, spec, options)
        for layouts in candidate_tuple
    )
    results = map_jobs(
        _evaluate_gemm_model_job,
        jobs,
        backend=backend,
        workers=workers,
        work_units=(
            _gemm_model_work_units(problem, tiling, options) * len(jobs)
        ),
    )
    best = min(results, key=lambda result: result.total_memory_cycles)
    validated_results: Tuple[GemmBankResult, ...] = ()
    if validation_top_k:
        if validation_top_k < 0 or validation_phase_samples <= 0:
            raise ValueError(
                "validation_top_k and validation_phase_samples must be positive"
            )
        count = min(validation_top_k, len(results))
        fastest = sorted(
            results, key=lambda result: result.total_memory_cycles
        )[:count]
        least_conflicted = sorted(
            results,
            key=lambda result: (
                -result.conflict_attainment,
                -result.worst_sample_conflict_attainment,
                result.total_memory_cycles,
            ),
        )[:count]
        selected_by_name = {
            result.layouts.name: result.layouts
            for result in (*fastest, *least_conflicted)
        }
        validation_options = replace(
            options,
            sampling_mode="phase_residue",
            max_spatial_samples=validation_phase_samples,
            max_k_samples=validation_phase_samples,
        )

        selected = tuple(selected_by_name.values())
        validation_jobs = tuple(
            _GemmModelJob(
                problem, tiling, layouts, spec, validation_options
            )
            for layouts in selected
        )
        validated_results = map_jobs(
            _evaluate_gemm_model_job,
            validation_jobs,
            backend=backend,
            workers=workers,
            work_units=(
                _gemm_model_work_units(problem, tiling, validation_options)
                * len(validation_jobs)
            ),
        )
        best = min(
            validated_results, key=lambda result: result.total_memory_cycles
        )
    return GemmLayoutSearchResult(
        best=best,
        results=results,
        validated_results=validated_results,
    )
