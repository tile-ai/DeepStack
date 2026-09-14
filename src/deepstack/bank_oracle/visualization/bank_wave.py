"""Export exact bank-row jobs into a dependency-free interactive HTML view.

The normal GEMM result intentionally stores compact statistical profiles, not
every decoded address.  This module reconstructs only user-selected waves so
the main model stays small while a debugging view can still inspect every
``(request, bank, row)`` job and its configurable-width sector mask.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..engine import (
    BankHistogram,
    _combine_histograms,
    build_bank_histogram,
    service_histogram,
)
from ..gemm import (
    GemmBankOptions,
    GemmLayoutSet,
    GemmProblem,
    GemmTiling,
    _RequestBuilder,
    _cta_order,
    make_layout_set,
)
from ..layout import BankSwizzle, MatrixAccess
from ..spec import DramBankSpec


@dataclass(frozen=True)
class RequestGroup:
    """Requests that share one operand-specific physical-bank swizzle."""

    name: str
    requests: Sequence[np.ndarray]
    swizzle: BankSwizzle = BankSwizzle()
    request_labels: Sequence[str] = ()
    operand: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("request group name must be non-empty")
        if self.request_labels and len(self.request_labels) != len(self.requests):
            raise ValueError("one request label is required per request")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _swizzle_payload(swizzle: BankSwizzle) -> Dict[str, Any]:
    return {
        "kind": swizzle.kind,
        "phase": swizzle.phase,
        "sector_source_shift": swizzle.sector_source_shift,
        "sector_bits": swizzle.sector_bits,
        "sector_bank_shift": swizzle.sector_bank_shift,
        "row_source_shift": swizzle.row_source_shift,
        "row_bits": swizzle.row_bits,
        "row_bank_shift": swizzle.row_bank_shift,
        "cyclic_sector_alpha": swizzle.cyclic_sector_alpha,
        "cyclic_row_beta": swizzle.cyclic_row_beta,
    }


def _spec_payload(spec: DramBankSpec) -> Dict[str, Any]:
    return {
        "total_layers": spec.total_layers,
        "connected_layers": spec.connected_layers,
        "banks_per_layer": spec.banks_per_layer,
        "physical_bank_count": spec.physical_bank_count,
        "port_count": spec.port_count,
        "row_bytes": spec.row_bytes,
        "sector_bytes": spec.sector_bytes,
        "sectors_per_row": spec.sectors_per_row,
        "sector_cycles": spec.sector_cycles,
        "recharge_cycles": spec.recharge_cycles,
        "row_read_cycles": spec.row_read_cycles,
        "full_row_cycles": spec.full_row_cycles,
        "data_rate_hz": spec.data_rate_hz,
    }


def _empty_histogram() -> BankHistogram:
    return BankHistogram(
        banks=np.empty(0, dtype=np.int64),
        rows=np.empty(0, dtype=np.int64),
        sector_masks=np.empty(0, dtype=np.uint64),
        request_ids=np.empty(0, dtype=np.int64),
    )


def _filter_histogram(histogram: BankHistogram, request_id: int) -> BankHistogram:
    keep = histogram.request_ids == request_id
    if not np.any(keep):
        return _empty_histogram()
    return BankHistogram(
        banks=histogram.banks[keep],
        rows=histogram.rows[keep],
        sector_masks=histogram.sector_masks[keep],
        request_ids=histogram.request_ids[keep],
    )


def _summary_payload(
    histogram: BankHistogram,
    spec: DramBankSpec,
) -> Tuple[Dict[str, Any], Any]:
    service = service_histogram(histogram, spec)
    actual = float(service.cycles)
    fixed = float(service.fixed_job_lower_bound_cycles)
    absolute = float(service.sector_lower_bound_cycles)
    fixed_attainment = 1.0 if actual == 0 else fixed / actual
    absolute_attainment = 1.0 if actual == 0 else absolute / actual
    epsilon = 1e-9
    summary = {
        "cycles": actual,
        "fixed_job_lower_bound_cycles": fixed,
        "sector_lower_bound_cycles": absolute,
        "fixed_job_attainment": fixed_attainment,
        "absolute_sector_attainment": absolute_attainment,
        "avoidable_fixed_job_conflict": actual > fixed + epsilon,
        "layout_packing_headroom": actual > absolute + epsilon,
        "active_banks": int(service.active_banks),
        "bank_count": spec.physical_bank_count,
        "row_job_count": histogram.job_count,
        "sector_count": int(service.sector_count),
        "transferred_bytes": int(service.transferred_bytes),
        "data_cycles_sum": int(service.sector_count * spec.sector_cycles),
        "recharge_cycles_sum": int(histogram.job_count * spec.recharge_cycles),
        "transaction_efficiency": float(service.transaction_efficiency),
        "bank_balance_efficiency": float(service.bank_balance_efficiency),
        "overall_efficiency": float(service.overall_efficiency),
    }
    return summary, service


def _job_payloads(
    histogram: BankHistogram,
    spec: DramBankSpec,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for index, (bank, row, mask, request_id) in enumerate(
        zip(
            histogram.banks,
            histogram.rows,
            histogram.sector_masks,
            histogram.request_ids,
        )
    ):
        mask_i = int(mask)
        sectors = [
            sector
            for sector in range(spec.sectors_per_row)
            if mask_i & (1 << sector)
        ]
        sector_count = len(sectors)
        jobs.append(
            {
                "job_id": index,
                "request_id": int(request_id),
                "bank": int(bank),
                "layer": int(bank) // spec.banks_per_layer,
                "bank_in_layer": int(bank) % spec.banks_per_layer,
                "row": int(row),
                "sector_mask_hex": f"0x{mask_i:0{_ceil_div(spec.sectors_per_row, 4)}x}",
                "active_sectors": sectors,
                "sector_count": sector_count,
                "data_cycles": sector_count * spec.sector_cycles,
                "recharge_cycles": spec.recharge_cycles,
                "total_cycles": (
                    sector_count * spec.sector_cycles + spec.recharge_cycles
                ),
            }
        )
    return jobs


def _bank_payloads(
    histogram: BankHistogram,
    service: Any,
    spec: DramBankSpec,
) -> List[Dict[str, Any]]:
    job_counts = np.bincount(
        histogram.banks,
        minlength=spec.physical_bank_count,
    )
    if histogram.job_count:
        sector_counts = np.bincount(
            histogram.banks,
            weights=np.fromiter(
                (int(mask).bit_count() for mask in histogram.sector_masks),
                dtype=np.int64,
                count=histogram.job_count,
            ),
            minlength=spec.physical_bank_count,
        )
    else:
        sector_counts = np.zeros(spec.physical_bank_count, dtype=np.int64)
    data_cycles = sector_counts * spec.sector_cycles
    recharge_cycles = job_counts * spec.recharge_cycles
    maximum = float(service.bank_cycles.max()) if service.bank_cycles.size else 0.0
    return [
        {
            "bank": bank,
            "layer": bank // spec.banks_per_layer,
            "bank_in_layer": bank % spec.banks_per_layer,
            "row_job_count": int(job_counts[bank]),
            "sector_count": int(sector_counts[bank]),
            "data_cycles": float(data_cycles[bank]),
            "recharge_cycles": float(recharge_cycles[bank]),
            "total_cycles": float(service.bank_cycles[bank]),
            "hotspot": bool(maximum > 0 and service.bank_cycles[bank] == maximum),
        }
        for bank in range(spec.physical_bank_count)
    ]


def _contiguous_sector_runs(sectors: np.ndarray) -> List[Tuple[int, int]]:
    """Return exact ``(start, length)`` runs for sorted unique sector IDs."""

    values = np.unique(np.asarray(sectors, dtype=np.int64))
    if values.size == 0:
        return []
    breaks = np.flatnonzero(values[1:] != values[:-1] + 1) + 1
    starts = np.concatenate((np.asarray([0]), breaks))
    stops = np.concatenate((breaks, np.asarray([values.size])))
    return [
        (int(values[start]), int(stop - start))
        for start, stop in zip(starts, stops)
    ]


def _flatten_sector_runs(sectors: np.ndarray) -> List[int]:
    return [
        value
        for start, length in _contiguous_sector_runs(sectors)
        for value in (start, length)
    ]


def build_logical_matrix_maps(
    problem: GemmProblem,
    tiling: GemmTiling,
    layouts: GemmLayoutSet,
    spec: DramBankSpec,
) -> Dict[str, Any]:
    """Encode logical A/B CTA blocks as exact physical-sector runs.

    Storage addresses do not change across reversible bank swizzles, so one
    scenario-level map can be decoded with whichever variant is selected in
    the browser.  Block order is row-major in the logical tile grid.
    """

    def build_operand(
        operand: str,
        shape: Tuple[int, int],
        tile_shape: Tuple[int, int],
        dtype_bytes: int,
    ) -> Dict[str, Any]:
        layout = layouts.a if operand == "A" else layouts.b
        grid_rows = _ceil_div(shape[0], tile_shape[0])
        grid_cols = _ceil_div(shape[1], tile_shape[1])
        blocks: List[List[int]] = []
        for block_row in range(grid_rows):
            row_start = block_row * tile_shape[0]
            row_stop = min(row_start + tile_shape[0], shape[0])
            for block_col in range(grid_cols):
                col_start = block_col * tile_shape[1]
                col_stop = min(col_start + tile_shape[1], shape[1])
                sectors = layout.touched_sectors(
                    shape,
                    MatrixAccess(row_start, row_stop, col_start, col_stop),
                    dtype_bytes,
                    spec.sector_bytes,
                )
                blocks.append(_flatten_sector_runs(sectors))
        return {
            "operand": operand,
            "logical_shape": list(shape),
            "tile_shape": list(tile_shape),
            "dtype_bytes": dtype_bytes,
            "grid_shape": [grid_rows, grid_cols],
            "blocks": blocks,
        }

    return {
        "A": build_operand(
            "A",
            problem.a_shape,
            (tiling.tb_m, tiling.tb_k),
            problem.a_dtype_bytes,
        ),
        "B": build_operand(
            "B",
            problem.b_shape,
            (tiling.tb_k, tiling.tb_n),
            problem.b_dtype_bytes,
        ),
    }


def build_bank_wave_snapshot(
    wave_id: str,
    label: str,
    groups: Sequence[RequestGroup],
    spec: DramBankSpec = DramBankSpec(),
    *,
    connectivity: str = "direct",
    coalesce_scope: str = "request",
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Decode one simultaneous bank wave into exact jobs and bank summaries."""

    if not wave_id or not label:
        raise ValueError("wave_id and label must be non-empty")
    if coalesce_scope not in {"request", "wave"}:
        raise ValueError("coalesce_scope must be request or wave")

    histograms: List[BankHistogram] = []
    request_entries: List[Dict[str, Any]] = []
    # Compact lossless schema: [request_id, group_index,
    # constituent_request, logical_sector_start, run_length].  The HTML uses
    # the same direct/interleaved decoder and swizzle formula to reconstruct
    # per-sector base/mapped coordinates on demand.  Keeping thousands of
    # repeated dict keys here made raw-access objects dominate the file size.
    access_runs: List[List[int]] = []
    raw_access_counts: Dict[int, int] = {}
    group_entries: List[Dict[str, Any]] = []
    request_offset = 0

    for group_index, group in enumerate(groups):
        requests = tuple(np.asarray(request, dtype=np.int64) for request in group.requests)
        labels = tuple(group.request_labels) or tuple(
            f"{group.name} request {index}" for index in range(len(requests))
        )
        histogram = build_bank_histogram(
            requests,
            spec,
            connectivity=connectivity,
            swizzle=group.swizzle,
            coalesce_scope=coalesce_scope,
        )
        if histogram.job_count:
            histogram = BankHistogram(
                banks=histogram.banks,
                rows=histogram.rows,
                sector_masks=histogram.sector_masks,
                request_ids=histogram.request_ids + request_offset,
            )
        histograms.append(histogram)

        if coalesce_scope == "request":
            global_ids = [request_offset + index for index in range(len(requests))]
            request_entries.extend(
                {
                    "request_id": global_id,
                    "label": labels[index],
                    "group": group.name,
                    "operand": group.operand,
                    "constituent_requests": 1,
                }
                for index, global_id in enumerate(global_ids)
            )
        else:
            global_ids = [request_offset] * len(requests)
            if requests:
                request_entries.append(
                    {
                        "request_id": request_offset,
                        "label": f"{group.name} (wave-coalesced)",
                        "group": group.name,
                        "operand": group.operand,
                        "constituent_requests": len(requests),
                    }
                )

        for local_request, (request, global_id) in enumerate(zip(requests, global_ids)):
            sectors = np.unique(request)
            if sectors.size == 0:
                continue
            raw_access_counts[int(global_id)] = (
                raw_access_counts.get(int(global_id), 0) + int(sectors.size)
            )
            access_runs.extend(
                [
                    int(global_id),
                    group_index,
                    local_request,
                    start,
                    length,
                ]
                for start, length in _contiguous_sector_runs(sectors)
            )

        group_entries.append(
            {
                "name": group.name,
                "operand": group.operand,
                "request_count": len(requests),
                "swizzle": _swizzle_payload(group.swizzle),
            }
        )
        request_offset += len(requests)

    histogram = _combine_histograms(histograms)
    summary, service = _summary_payload(histogram, spec)
    jobs = _job_payloads(histogram, spec)
    banks = _bank_payloads(histogram, service, spec)

    for entry in request_entries:
        request_histogram = _filter_histogram(histogram, entry["request_id"])
        request_summary, _ = _summary_payload(request_histogram, spec)
        entry["summary"] = request_summary
        entry["raw_access_count"] = raw_access_counts.get(entry["request_id"], 0)

    return {
        "wave_id": wave_id,
        "label": label,
        "context": dict(context or {}),
        "connectivity": connectivity,
        "coalesce_scope": coalesce_scope,
        "spec": _spec_payload(spec),
        "groups": group_entries,
        "summary": summary,
        "requests": request_entries,
        "access_runs": access_runs,
        "jobs": jobs,
        "banks": banks,
    }


def _gemm_request_labels(
    ctas: np.ndarray,
    global_cta_start: int,
    k_start: int,
    k_stop: int,
) -> Tuple[List[str], List[str]]:
    a_labels: List[str] = []
    b_labels: List[str] = []
    for k_index in range(k_start, k_stop):
        for local_cta, (batch, m_index, n_index) in enumerate(ctas):
            global_cta = global_cta_start + local_cta
            coordinate = f"CTA {global_cta} (b={batch}, m={m_index}, n={n_index})"
            a_labels.append(f"A | {coordinate} | K-tile {k_index}")
            b_labels.append(f"B | {coordinate} | K-tile {k_index}")
    return a_labels, b_labels


def build_gemm_wave_snapshot(
    problem: GemmProblem,
    tiling: GemmTiling,
    layouts: GemmLayoutSet,
    *,
    spatial_wave_index: int,
    k_window_index: Optional[int] = 0,
    store: bool = False,
    spec: DramBankSpec = DramBankSpec(),
    options: GemmBankOptions = GemmBankOptions(),
    wave_id: Optional[str] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Rebuild one deterministic GEMM load or store wave for inspection.

    L2-filtered waves are deliberately rejected here: the compact L2 trace has
    additional temporal state.  Use this view for the all-miss physical layout
    and bank-conflict oracle, or pass explicit already-filtered requests to
    :func:`build_bank_wave_snapshot`.
    """

    if options.l2_cache is not None:
        raise ValueError("GEMM wave visualization currently requires l2_cache=None")
    cta_order = _cta_order(problem, tiling)
    spatial_waves = _ceil_div(int(cta_order.shape[0]), tiling.ctas_per_wave)
    if not 0 <= spatial_wave_index < spatial_waves:
        raise IndexError("spatial_wave_index is outside the GEMM grid")
    cta_start = spatial_wave_index * tiling.ctas_per_wave
    ctas = cta_order[cta_start:cta_start + tiling.ctas_per_wave]
    builder = _RequestBuilder(problem, tiling, layouts, spec, options)

    if store:
        requests, useful_bytes, _ = builder.store_requests(ctas)
        labels = [
            f"C | CTA {cta_start + local} (b={b}, m={m}, n={n})"
            for local, (b, m, n) in enumerate(ctas)
        ]
        groups = (
            RequestGroup("C stores", requests, layouts.c.swizzle, labels, "C"),
        )
        kind = "store"
        k_context: Dict[str, Any] = {}
    else:
        if k_window_index is None:
            raise ValueError("a load wave requires k_window_index")
        grid_k = _ceil_div(problem.k, tiling.tb_k)
        k_windows = _ceil_div(grid_k, tiling.pending_k_iterations)
        if not 0 <= k_window_index < k_windows:
            raise IndexError("k_window_index is outside the GEMM K grid")
        k_start = k_window_index * tiling.pending_k_iterations
        k_stop = min(k_start + tiling.pending_k_iterations, grid_k)
        requests, useful_bytes, _ = builder.load_requests(
            ctas,
            k_start,
            k_stop,
            global_cta_start=cta_start,
        )
        a_labels, b_labels = _gemm_request_labels(
            ctas, cta_start, k_start, k_stop
        )
        groups = (
            RequestGroup("A loads", requests[0::2], layouts.a.swizzle, a_labels, "A"),
            RequestGroup("B loads", requests[1::2], layouts.b.swizzle, b_labels, "B"),
        )
        kind = "load"
        k_context = {
            "k_window_index": k_window_index,
            "k_window_count": k_windows,
            "k_tile_start": k_start,
            "k_tile_stop": k_stop,
        }

    context = {
        "kind": kind,
        "problem": [problem.m, problem.n, problem.k],
        "batch": problem.batch,
        "tile": [tiling.tb_m, tiling.tb_n, tiling.tb_k],
        "stage": tiling.stage,
        "pending_k_iterations": tiling.pending_k_iterations,
        "ctas_per_wave": tiling.ctas_per_wave,
        "row_panel": tiling.row_panel,
        "column_panel": tiling.column_panel,
        "raster_axis": tiling.raster_axis,
        "spatial_wave_index": spatial_wave_index,
        "spatial_wave_count": spatial_waves,
        "global_cta_start": cta_start,
        "active_ctas": int(ctas.shape[0]),
        "cta_coordinates": ctas.tolist(),
        "useful_bytes_before_coalescing": int(useful_bytes),
        "layout": layouts.name,
        **k_context,
    }
    default_id = (
        f"gemm-s{spatial_wave_index}-store"
        if store
        else f"gemm-s{spatial_wave_index}-k{k_window_index}"
    )
    default_label = (
        f"spatial wave {spatial_wave_index} | C store"
        if store
        else f"spatial wave {spatial_wave_index} | K window {k_window_index}"
    )
    return build_bank_wave_snapshot(
        wave_id or default_id,
        label or default_label,
        groups,
        spec,
        connectivity=options.connectivity,
        coalesce_scope=options.coalesce_scope,
        context=context,
    )


def _synthetic_scenarios(spec: DramBankSpec) -> List[Dict[str, Any]]:
    spread_count = min(spec.bank_count, spec.sectors_per_row)
    contiguous = np.arange(spread_count, dtype=np.int64)
    full_rows = np.concatenate(
        [
            np.arange(
                row * spec.bank_count * spec.sectors_per_row,
                row * spec.bank_count * spec.sectors_per_row
                + spec.sectors_per_row,
                dtype=np.int64,
            )
            for row in range(spec.bank_count)
        ]
    )
    xor_legal = (
        _is_power_of_two(spec.bank_count)
        and _is_power_of_two(spec.sectors_per_row)
    )
    sector_spread = (
        BankSwizzle(
            kind="xor",
            sector_bits=min(
                spec.bank_count.bit_length() - 1,
                spec.sectors_per_row.bit_length() - 1,
            ),
        )
        if xor_legal
        else BankSwizzle(kind="cyclic", cyclic_sector_alpha=1)
    )
    row_balance = (
        BankSwizzle(
            kind="xor",
            row_bits=spec.bank_count.bit_length() - 1,
        )
        if xor_legal
        else BankSwizzle(kind="cyclic", cyclic_row_beta=1)
    )
    permutation_label = "XOR" if xor_legal else "cyclic"

    def variant(
        variant_id: str,
        variant_label: str,
        explanation: str,
        sectors: np.ndarray,
        swizzle: BankSwizzle,
        wave_label: str,
    ) -> Dict[str, Any]:
        snapshot = build_bank_wave_snapshot(
            f"{variant_id}-wave0",
            wave_label,
            (RequestGroup("synthetic", (sectors,), swizzle, (wave_label,), "synthetic"),),
            spec,
            context={"kind": "synthetic", "logical_sector_count": int(sectors.size)},
        )
        return {
            "variant_id": variant_id,
            "label": variant_label,
            "explanation": explanation,
            "waves": [snapshot],
        }

    return [
        {
            "scenario_id": "synthetic-sector-spread",
            "label": "Synthetic contiguous-sector spread",
            "description": (
                "The linear layout keeps a contiguous logical run together. "
                "A reversible sector XOR spreads the same requests across "
                "independent banks. All geometry and timing are synthetic."
            ),
            "variants": [
                variant(
                    "contiguous-linear",
                    "Linear",
                    "The contiguous run remains one row job.",
                    contiguous,
                    BankSwizzle(),
                    f"logical sectors 0..{spread_count - 1}",
                ),
                variant(
                    "contiguous-sector-xor",
                    f"Sector {permutation_label}",
                    "Sector coordinates are reversibly distributed across banks.",
                    contiguous,
                    sector_spread,
                    f"logical sectors 0..{spread_count - 1}",
                ),
            ],
        },
        {
            "scenario_id": "synthetic-row-balance",
            "label": "Synthetic colliding-row balance",
            "description": (
                "One complete row per configured bank is deliberately mapped "
                "to one bank, then balanced with a reversible row XOR."
            ),
            "variants": [
                variant(
                    "rows-linear",
                    "Linear",
                    "One bank serializes every full-row job.",
                    full_rows,
                    BankSwizzle(),
                    f"{spec.bank_count} complete rows",
                ),
                variant(
                    "rows-row-xor",
                    f"Row {permutation_label}",
                    "Each bank receives one intact full-row job; locality is preserved.",
                    full_rows,
                    row_balance,
                    f"{spec.bank_count} complete rows",
                ),
            ],
        },
    ]


def _layout_variant_specs(spec: DramBankSpec) -> List[Dict[str, Any]]:
    bank_bits = spec.bank_count.bit_length() - 1
    sector_bits = min(
        4,
        bank_bits,
        spec.sectors_per_row.bit_length() - 1,
    )
    variants: List[Dict[str, Any]] = [
        {
            "variant_id": "tile_major_linear",
            "label": "Tile-major linear",
            "explanation": "No bank-field swizzle.",
            "swizzle": BankSwizzle(),
            "b_phase": 0,
            "c_phase": 0,
        }
    ]
    if (
        _is_power_of_two(spec.bank_count)
        and _is_power_of_two(spec.sectors_per_row)
    ):
        variants.append(
            {
                "variant_id": "tile_sector_xor",
                "label": "Sector XOR",
                "explanation": (
                    "A legal subset of sector bits feeds the configured bank field."
                ),
                "swizzle": BankSwizzle(
                    kind="xor",
                    sector_bits=sector_bits,
                    sector_bank_shift=bank_bits - sector_bits,
                ),
                "b_phase": 0,
                "c_phase": 0,
            }
        )
    else:
        variants.append(
            {
                "variant_id": "tile_sector_cyclic",
                "label": "Sector cyclic",
                "explanation": (
                    "A modular permutation supports non-power-of-two geometry."
                ),
                "swizzle": BankSwizzle(
                    kind="cyclic",
                    cyclic_sector_alpha=1,
                ),
                "b_phase": 0,
                "c_phase": 0,
            }
        )
    return variants


def _representative_wave_coordinates(
    problem: GemmProblem,
    tiling: GemmTiling,
    *,
    include_middle: bool,
) -> List[Tuple[int, int]]:
    spatial_waves = _ceil_div(
        int(_cta_order(problem, tiling).shape[0]), tiling.ctas_per_wave
    )
    grid_k = _ceil_div(problem.k, tiling.tb_k)
    k_windows = _ceil_div(grid_k, tiling.pending_k_iterations)
    candidates = [(0, 0)]
    if include_middle:
        candidates.append((spatial_waves // 2, k_windows // 2))
    candidates.append((spatial_waves - 1, k_windows - 1))
    wave_coordinates: List[Tuple[int, int]] = []
    for coordinate in candidates:
        if coordinate not in wave_coordinates:
            wave_coordinates.append(coordinate)
    return wave_coordinates


def _gemm_scenario(
    spec: DramBankSpec,
    *,
    scenario_id: str,
    label: str,
    description: str,
    problem: GemmProblem,
    tiling: GemmTiling,
    include_middle: bool,
) -> Dict[str, Any]:
    options = GemmBankOptions(max_spatial_samples=None, max_k_samples=None)
    variant_specs = _layout_variant_specs(spec)
    layouts = [
        make_layout_set(
            problem,
            tiling,
            spec,
            preset="tile_major",
            swizzle=variant["swizzle"],
            b_phase=variant["b_phase"],
            c_phase=variant["c_phase"],
            name=variant["variant_id"],
        )
        for variant in variant_specs
    ]
    wave_coordinates = _representative_wave_coordinates(
        problem, tiling, include_middle=include_middle
    )

    variants = []
    for variant_spec, layout in zip(variant_specs, layouts):
        waves = [
            build_gemm_wave_snapshot(
                problem,
                tiling,
                layout,
                spatial_wave_index=spatial,
                k_window_index=k_window,
                spec=spec,
                options=options,
                wave_id=(
                    f"{scenario_id}-{layout.name}-s{spatial}-k{k_window}"
                ),
            )
            for spatial, k_window in wave_coordinates
        ]
        variants.append(
            {
                "variant_id": variant_spec["variant_id"],
                "label": variant_spec["label"],
                "explanation": variant_spec["explanation"],
                "waves": waves,
            }
        )
    return {
        "scenario_id": scenario_id,
        "label": label,
        "description": description,
        "matrix_maps": build_logical_matrix_maps(
            problem, tiling, layouts[0], spec
        ),
        "case_spec": {
            "problem": [problem.m, problem.n, problem.k],
            "tile": [tiling.tb_m, tiling.tb_n, tiling.tb_k],
            "stage": tiling.stage,
            "pending_k_iterations": tiling.pending_k_iterations,
            "ctas_per_wave": tiling.ctas_per_wave,
            "row_panel": tiling.row_panel,
            "column_panel": tiling.column_panel,
            "raster_axis": tiling.raster_axis,
            "coalesce_scope": options.coalesce_scope,
        },
        "variants": variants,
    }


def _gemm_scenarios(spec: DramBankSpec) -> List[Dict[str, Any]]:
    common_note = (
        "The combined wave uses operand-wide coalescing. This is a synthetic "
        "bank-trace diagnostic, not a kernel-performance claim."
    )
    return [
        _gemm_scenario(
            spec,
            scenario_id="synthetic-ragged-gemm",
            label="Synthetic ragged GEMM",
            description=(
                "Ragged dimensions expose first and tail wave mappings. "
                + common_note
            ),
            problem=GemmProblem(73, 149, 211),
            tiling=GemmTiling(
                32, 64, 32, stage=3, ctas_per_wave=4, row_panel=2
            ),
            include_middle=True,
        ),
    ]


def build_demo_dashboard(
    *,
    include_gemm: bool = True,
    spec: Optional[DramBankSpec] = None,
) -> Dict[str, Any]:
    """Return a deterministic dashboard for a supplied or synthetic spec."""

    spec = DramBankSpec() if spec is None else spec
    scenarios = _synthetic_scenarios(spec)
    if include_gemm:
        scenarios.extend(_gemm_scenarios(spec))
    return {
        "schema_version": 2,
        "title": "Synthetic bank-wave conflict explorer",
        "subtitle": (
            f"Synthetic toy configuration | {spec.total_layers} layers × "
            f"{spec.banks_per_layer} banks/layer | exact row jobs, "
            "configurable sector masks, and swizzle comparison"
        ),
        "scenarios": scenarios,
    }


def render_dashboard_html(data: Mapping[str, Any]) -> str:
    """Embed a JSON payload into the zero-dependency HTML template."""

    serialized = json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = base64.b64encode(serialized).decode("ascii")
    template_path = Path(__file__).with_name("bank_wave_template.html")
    template = template_path.read_text(encoding="utf-8")
    marker = "__BANK_WAVE_BASE64_PAYLOAD__"
    if template.count(marker) != 1:
        raise RuntimeError("bank-wave template payload marker is missing or duplicated")
    return template.replace(marker, payload)


def write_dashboard_html(data: Mapping[str, Any], output: Path | str) -> Path:
    """Write one self-contained dashboard and return its resolved path."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard_html(data), encoding="utf-8")
    return path.resolve()
