"""Pipeline overlap adapter for the isolated bank-aware GEMM prototype.

The equations intentionally match TileSight's
``fused_op_pipeline_wave.pipeline_overlap`` prologue/steady/epilogue model,
but DDR time comes from explicit bank service cycles.  The adapter never reads
coarse architecture bandwidth/utilization fields or a separately applied
architecture-level Little's-Law adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple

from .gemm import (
    GemmBankOptions,
    GemmBankResult,
    GemmLayoutSet,
    GemmProblem,
    GemmTiling,
    _gemm_model_work_units,
    default_layout_candidates,
    model_gemm_bank,
)
from .parallel import ParallelBackend, map_jobs
from .spec import DramBankSpec


@dataclass(frozen=True)
class PipelineBreakdown:
    prologue_cycles: float
    steady_cycles: float
    epilogue_cycles: float
    total_cycles: float
    steady_memory_cycles: float
    steady_compute_cycles: float


@dataclass(frozen=True)
class BankAwareGemmPipelineResult:
    bank_result: GemmBankResult
    total_cycles: float
    ideal_total_cycles: float
    total_seconds: float
    ideal_total_seconds: float
    compute_cycles_per_k: float
    mean_other_memory_cycles_per_k: float
    read_hidden_fraction: float
    pipeline_attainment: float
    breakdowns: Tuple[PipelineBreakdown, ...]
    ideal_breakdowns: Tuple[PipelineBreakdown, ...]

    @property
    def bank_only_attainment(self) -> float:
        return self.bank_result.bank_only_attainment


@dataclass(frozen=True)
class PipelineSearchResult:
    best: BankAwareGemmPipelineResult
    results: Tuple[BankAwareGemmPipelineResult, ...]
    validated_results: Tuple[BankAwareGemmPipelineResult, ...] = ()

    @property
    def validation_performed(self) -> bool:
        return bool(self.validated_results)


def pipeline_cycles(
    num_iterations: int,
    stage: int,
    memory_cycles_per_iteration: float,
    compute_cycles_per_iteration: float,
    store_cycles: float = 0.0,
) -> PipelineBreakdown:
    """Safe TileSight-compatible prologue/steady/epilogue equation."""

    if num_iterations < 0 or stage <= 0:
        raise ValueError("num_iterations must be non-negative and stage positive")
    mem = float(memory_cycles_per_iteration)
    comp = float(compute_cycles_per_iteration)
    store = float(store_cycles)
    if min(mem, comp, store) < 0:
        raise ValueError("pipeline times must be non-negative")
    if num_iterations == 0:
        return PipelineBreakdown(0.0, 0.0, store, store, mem, comp)
    if stage == 1:
        steady = num_iterations * (mem + comp)
        return PipelineBreakdown(
            prologue_cycles=0.0,
            steady_cycles=steady,
            epilogue_cycles=store,
            total_cycles=steady + store,
            steady_memory_cycles=mem,
            steady_compute_cycles=comp,
        )
    # Clamp the fill/drain depth so gridK < stage cannot create phantom work.
    depth = min(stage - 1, num_iterations)
    steady_iterations = num_iterations - depth
    prologue = depth * mem
    steady = steady_iterations * max(mem, comp)
    epilogue = depth * comp + store
    return PipelineBreakdown(
        prologue_cycles=prologue,
        steady_cycles=steady,
        epilogue_cycles=epilogue,
        total_cycles=prologue + steady + epilogue,
        steady_memory_cycles=mem,
        steady_compute_cycles=comp,
    )


def _flops_capacity(arch: Any, dtype_bytes: int) -> float:
    if dtype_bytes == 1:
        if hasattr(arch, "fp8_tensor_flops"):
            return float(arch.fp8_tensor_flops)
        return float(arch.int8_tensor_flops)
    if dtype_bytes == 2:
        return float(arch.fp16_tensor_flops)
    if dtype_bytes == 4:
        if hasattr(arch, "fp32_tensor_flops"):
            return float(arch.fp32_tensor_flops)
        return float(arch.fp32_cuda_core_flops)
    raise ValueError("pipeline prototype supports 1/2/4-byte GEMM inputs")


def _compute_cycles_per_k(
    bank_result: GemmBankResult,
    arch: Any,
) -> float:
    problem = bank_result.problem
    tiling = bank_result.tiling
    flops = 2 * tiling.tb_m * tiling.tb_n * tiling.tb_k
    compute_dtype = problem.compute_dtype_bytes or problem.a_dtype_bytes
    per_sm_flops_s = _flops_capacity(arch, compute_dtype) / arch.sm_count
    compute_util = float(getattr(arch, "compute_max_util", 1.0))
    seconds = flops / (per_sm_flops_s * compute_util)
    return bank_result.spec.seconds_to_cycles(seconds)


def _other_memory_cycles(
    bank_result: GemmBankResult,
    arch: Any,
    active_ctas: int,
    *,
    store: bool,
) -> float:
    """L2/SMEM portion of the existing TileSight per-iteration resource model."""

    problem = bank_result.problem
    tiling = bank_result.tiling
    if store:
        bytes_per_cta = tiling.tb_m * tiling.tb_n * problem.c_dtype_bytes
        l2_bytes = bytes_per_cta
        smem_bytes = bytes_per_cta
    else:
        bytes_per_cta = tiling.tb_k * (
            tiling.tb_m * problem.a_dtype_bytes
            + tiling.tb_n * problem.b_dtype_bytes
        )
        l2_bytes = bytes_per_cta
        # TileSight wgmma/utcmma path: L2->SMEM plus SMEM->tensor core.
        smem_bytes = 2 * bytes_per_cta

    l2_bw = float(getattr(arch, "l2_bandwidth", float("inf")))
    smem_bw = float(getattr(arch, "smem_bandwidth", float("inf")))
    l2_util = float(getattr(arch, "l2_max_util", 1.0))
    smem_util = float(getattr(arch, "l1_max_util", 1.0))
    l2_seconds = l2_bytes * active_ctas / (l2_bw * l2_util)
    # SMEM is private per SM; use one CTA's resource and per-SM bandwidth.
    per_sm_smem_bw = smem_bw / arch.sm_count
    smem_seconds = smem_bytes / (per_sm_smem_bw * smem_util)
    return bank_result.spec.seconds_to_cycles(max(l2_seconds, smem_seconds))


def model_gemm_pipeline(
    bank_result: GemmBankResult,
    arch: Optional[Any] = None,
    *,
    compute_cycles_per_k: Optional[float] = None,
    other_memory_cycles_per_k: Optional[float] = None,
    store_other_memory_cycles: Optional[float] = None,
    kernel_launch_seconds: float = 0.0,
) -> BankAwareGemmPipelineResult:
    """Compose sampled bank waves with software-pipeline overlap."""

    if arch is not None and bank_result.tiling.ctas_per_wave != arch.sm_count:
        raise ValueError(
            "the current pipeline adapter requires one CTA per SM and "
            "ctas_per_wave == arch.sm_count"
        )

    if compute_cycles_per_k is None:
        if arch is None:
            raise ValueError("arch or compute_cycles_per_k is required")
        compute_cycles_per_k = _compute_cycles_per_k(bank_result, arch)
    grid_k = (bank_result.problem.k + bank_result.tiling.tb_k - 1) // bank_result.tiling.tb_k
    breakdowns = []
    ideal_breakdowns = []
    total_cycles = 0.0
    ideal_total_cycles = 0.0
    hidden_read_cycles = 0.0
    total_read_cycles = 0.0
    total_weight = 0.0
    other_mem_weighted = 0.0

    for summary in bank_result.spatial_summaries:
        if other_memory_cycles_per_k is not None:
            other_mem = float(other_memory_cycles_per_k)
        elif arch is not None:
            other_mem = _other_memory_cycles(
                bank_result, arch, summary.active_ctas, store=False
            )
        else:
            other_mem = 0.0
        if store_other_memory_cycles is not None:
            store_other = float(store_other_memory_cycles)
        elif arch is not None:
            store_other = _other_memory_cycles(
                bank_result, arch, summary.active_ctas, store=True
            )
        else:
            store_other = 0.0
        mem = max(summary.load_cycles_per_k, other_mem)
        ideal_mem = max(summary.ideal_load_cycles_per_k, other_mem)
        store = max(summary.store_cycles, store_other)
        ideal_store = max(summary.ideal_store_cycles, store_other)
        actual = pipeline_cycles(
            grid_k,
            bank_result.tiling.stage,
            mem,
            compute_cycles_per_k,
            store,
        )
        ideal = pipeline_cycles(
            grid_k,
            bank_result.tiling.stage,
            ideal_mem,
            compute_cycles_per_k,
            ideal_store,
        )
        breakdowns.append(actual)
        ideal_breakdowns.append(ideal)
        total_cycles += actual.total_cycles * summary.spatial_weight
        ideal_total_cycles += ideal.total_cycles * summary.spatial_weight
        if bank_result.tiling.stage >= 2:
            depth = min(bank_result.tiling.stage - 1, grid_k)
            overlapping_iterations = grid_k - depth
        else:
            # A single stage serializes load and compute, so none of the read
            # service is hidden even though every K iteration is "steady".
            overlapping_iterations = 0
        total_read_cycles += grid_k * mem * summary.spatial_weight
        hidden_read_cycles += (
            overlapping_iterations
            * min(mem, compute_cycles_per_k)
            * summary.spatial_weight
        )
        total_weight += summary.spatial_weight
        other_mem_weighted += other_mem * summary.spatial_weight

    launch_cycles = bank_result.spec.seconds_to_cycles(kernel_launch_seconds)
    total_cycles += launch_cycles
    ideal_total_cycles += launch_cycles
    pipeline_attainment = (
        ideal_total_cycles / total_cycles if total_cycles > 0 else 1.0
    )
    return BankAwareGemmPipelineResult(
        bank_result=bank_result,
        total_cycles=total_cycles,
        ideal_total_cycles=ideal_total_cycles,
        total_seconds=bank_result.spec.cycles_to_seconds(total_cycles),
        ideal_total_seconds=bank_result.spec.cycles_to_seconds(ideal_total_cycles),
        compute_cycles_per_k=float(compute_cycles_per_k),
        mean_other_memory_cycles_per_k=(
            other_mem_weighted / total_weight if total_weight else 0.0
        ),
        read_hidden_fraction=(
            hidden_read_cycles / total_read_cycles
            if total_read_cycles else 1.0
        ),
        pipeline_attainment=pipeline_attainment,
        breakdowns=tuple(breakdowns),
        ideal_breakdowns=tuple(ideal_breakdowns),
    )


@dataclass(frozen=True)
class _PipelineModelJob:
    """Pickleable stage/layout job for candidate-level process search."""

    problem: GemmProblem
    tiling: GemmTiling
    layouts: GemmLayoutSet
    spec: DramBankSpec
    options: GemmBankOptions
    arch: Optional[Any]
    compute_cycles_per_k: Optional[float]


def _evaluate_pipeline_model_job(
    job: _PipelineModelJob,
) -> BankAwareGemmPipelineResult:
    """Module-level process worker; all job fields must be pickleable."""

    bank = model_gemm_bank(
        job.problem,
        job.tiling,
        job.layouts,
        job.spec,
        job.options,
    )
    return model_gemm_pipeline(
        bank,
        job.arch,
        compute_cycles_per_k=job.compute_cycles_per_k,
    )


def search_layouts_and_stages(
    problem: GemmProblem,
    base_tiling: GemmTiling,
    spec: DramBankSpec = DramBankSpec(),
    options: GemmBankOptions = GemmBankOptions(),
    *,
    arch: Optional[Any] = None,
    stages: Sequence[int] = (1, 2, 3, 4),
    row_panels: Optional[Sequence[int]] = None,
    raster_axes: Optional[Sequence[str]] = None,
    compute_cycles_per_k: Optional[float] = None,
    workers: int = 0,
    backend: ParallelBackend = "auto",
    validation_top_k_per_stage: int = 0,
    validation_phase_samples: int = 64,
) -> PipelineSearchResult:
    """Search legal layout/swizzle candidates together with stage 1..4.

    ``row_panels`` and ``raster_axes`` optionally add CTA access-grouping axes;
    by default each keeps the corresponding value from ``base_tiling``.  The
    optional second pass takes the fastest and least-conflicted Top-K from every
    (stage, row-panel, raster-axis) configuration and checks consecutive bank
    residues.  This catches common modular aliases but is not an exact proof
    for arbitrary strides.  ``auto`` uses serial execution for small searches
    and processes for sufficiently heavy jobs; it never auto-selects threads.
    """

    stage_values = tuple(dict.fromkeys(int(stage) for stage in stages))
    row_panel_values = tuple(dict.fromkeys(
        [base_tiling.row_panel]
        if row_panels is None
        else [int(row_panel) for row_panel in row_panels]
    ))
    raster_axis_values = tuple(dict.fromkeys(
        [base_tiling.raster_axis]
        if raster_axes is None
        else [str(raster_axis) for raster_axis in raster_axes]
    ))
    jobs = []
    for stage in stage_values:
        for row_panel in row_panel_values:
            for raster_axis in raster_axis_values:
                tiling = replace(
                    base_tiling,
                    stage=stage,
                    row_panel=row_panel,
                    raster_axis=raster_axis,
                )
                if arch is not None and hasattr(
                    arch, "configurable_smem_capacity"
                ):
                    if options.smem_buffer_bytes_per_cta is None:
                        per_stage_bytes = tiling.tb_k * (
                            tiling.tb_m * problem.a_dtype_bytes
                            + tiling.tb_n * problem.b_dtype_bytes
                        )
                        smem_footprint = tiling.stage * per_stage_bytes
                    else:
                        smem_footprint = options.smem_buffer_bytes_per_cta
                    if smem_footprint > float(
                        arch.configurable_smem_capacity
                    ):
                        continue
                for layouts in default_layout_candidates(
                    problem, tiling, spec
                ):
                    jobs.append((tiling, layouts))
    if not jobs:
        raise ValueError("no legal stage/layout jobs fit the SMEM capacity")

    model_jobs = tuple(
        _PipelineModelJob(
            problem,
            tiling,
            layouts,
            spec,
            options,
            arch,
            compute_cycles_per_k,
        )
        for tiling, layouts in jobs
    )
    results = map_jobs(
        _evaluate_pipeline_model_job,
        model_jobs,
        backend=backend,
        workers=workers,
        work_units=sum(
            _gemm_model_work_units(problem, job.tiling, options)
            for job in model_jobs
        ),
    )
    best = min(results, key=lambda result: result.total_cycles)
    validated_results: Tuple[BankAwareGemmPipelineResult, ...] = ()
    if validation_top_k_per_stage:
        if validation_top_k_per_stage < 0 or validation_phase_samples <= 0:
            raise ValueError(
                "validation counts must be positive"
            )
        selected = {}
        configurations = tuple(dict.fromkeys(
            (
                result.bank_result.tiling.stage,
                result.bank_result.tiling.row_panel,
                result.bank_result.tiling.raster_axis,
            )
            for result in results
        ))
        for stage, row_panel, raster_axis in configurations:
            configuration_results = [
                result
                for result in results
                if (
                    result.bank_result.tiling.stage,
                    result.bank_result.tiling.row_panel,
                    result.bank_result.tiling.raster_axis,
                ) == (stage, row_panel, raster_axis)
            ]
            count = min(
                validation_top_k_per_stage, len(configuration_results)
            )
            fastest = sorted(
                configuration_results,
                key=lambda result: result.total_cycles,
            )[:count]
            least_conflicted = sorted(
                configuration_results,
                key=lambda result: (
                    -result.bank_result.conflict_attainment,
                    -result.bank_result.worst_sample_conflict_attainment,
                    result.total_cycles,
                ),
            )[:count]
            for result in (*fastest, *least_conflicted):
                key = (
                    result.bank_result.tiling.stage,
                    result.bank_result.tiling.row_panel,
                    result.bank_result.tiling.raster_axis,
                    result.bank_result.layouts.name,
                )
                selected[key] = (
                    result.bank_result.tiling,
                    result.bank_result.layouts,
                )

        validation_options = replace(
            options,
            sampling_mode="phase_residue",
            max_spatial_samples=validation_phase_samples,
            max_k_samples=validation_phase_samples,
        )

        validation_jobs = tuple(
            _PipelineModelJob(
                problem,
                tiling,
                layouts,
                spec,
                validation_options,
                arch,
                compute_cycles_per_k,
            )
            for tiling, layouts in selected.values()
        )
        validated_results = map_jobs(
            _evaluate_pipeline_model_job,
            validation_jobs,
            backend=backend,
            workers=workers,
            work_units=sum(
                _gemm_model_work_units(
                    problem, job.tiling, validation_options
                )
                for job in validation_jobs
            ),
        )
        best = min(validated_results, key=lambda result: result.total_cycles)
    return PipelineSearchResult(
        best=best,
        results=results,
        validated_results=validated_results,
    )
