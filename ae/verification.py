"""Acceptance checks and concise summaries for generated AE results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any

import pandas as pd

from .paths import DATA_DIR, RESULTS_DIR, ROOT
from .selected_decode import load_candidate_table, select_fig15_local_best


DECODE_POPULATION = DATA_DIR / "fig15" / "decode_dse_population_march2026.csv"
LEGACY_TOLERANCE_PCT = 1e-9
CORRECTED_RELATIVE_TOLERANCE = 1e-9
DERIVED_FIELD_TOLERANCE = 1e-9


@dataclass(frozen=True)
class _CsvContract:
    """Small, declarative contract for one generated CSV."""

    path: str
    rows: int
    required: tuple[str, ...]
    allowed: tuple[str, ...] = ()
    unique: tuple[str, ...] = ()
    finite: tuple[str, ...] = ()
    zero: tuple[str, ...] = ()
    false: tuple[str, ...] = ()
    exact_values: tuple[tuple[str, tuple[str, ...]], ...] = ()
    reject_fail_status: bool = False


_MODEL_VERSIONS = ("paper_legacy", "current_corrected")

_EXPECTED_PLOT_STEMS = frozenset(
    {
        "fig08_modeling_accuracy_h100",
        "fig09_b200_decode_validation",
        "fig10_mi325x_decode_validation",
        "fig12_deepseek_v32_dsa",
        "fig13_noc_model_validation",
        "fig15_decode_pareto",
        "fig16_3d_vs_2_5d",
        "fig17_dram_bandwidth",
        "fig18_dram_layer_throughput",
        "fig19_prefill_decode_dse_heatmaps",
        "fig20_dram_thermal_terrain",
        "fig21_noc_bw_latency_sensitivity",
        "fig22_noc_layer_decode",
        "fig22_noc_layer_prefill",
        "fig23_parallelism_search_space",
        "table04_ablation",
    }
)
_MIN_PNG_BYTES = 1_000
_MIN_PDF_BYTES = 500
_MIN_PLOT_DIMENSION = 100
_MAX_PLOT_DIMENSION = 50_000
_EXPECTED_STAGE_AUXILIARY_FILES = frozenset(
    {
        "fig08/verification.json",
        "fig09/PASS",
        "fig10/PASS.txt",
        "fig10/verification.json",
        "fig12/PASS",
        "fig18_20/current_corrected/PASS",
        "fig18_20/current_corrected/status.json",
        "fig18_20/paper_legacy/PASS",
        "fig18_20/paper_legacy/status.json",
        "fig22/PASS",
        "fig22/current_corrected/PASS",
        "fig22/paper_legacy/PASS",
        "fig23/current_corrected/PASS",
        "fig23/paper_legacy/PASS",
        "table4/PASS",
    }
)


_STAGE_CSV_CONTRACTS: tuple[tuple[str, tuple[_CsvContract, ...]], ...] = (
    (
        "fig08",
        (
            _CsvContract(
                "fig08/model_points.csv",
                52,
                (
                    "category",
                    "point_index",
                    "paper_model_ms",
                    "current_model_ms",
                    "gpu_reference_ms",
                ),
                unique=("category", "point_index"),
                finite=("paper_model_ms", "current_model_ms", "gpu_reference_ms"),
            ),
            _CsvContract(
                "fig08/summary.csv",
                6,
                ("scope", "points", "paper_mape_pct", "current_mape_pct"),
                unique=("scope",),
                finite=("points", "paper_mape_pct", "current_mape_pct"),
            ),
        ),
    ),
    (
        "fig09",
        (
            _CsvContract(
                "fig09/model_points.csv",
                40,
                (
                    "point_id",
                    "model",
                    "bs",
                    "paper_legacy_stps",
                    "current_corrected_stps",
                    "gpu_runs_performed",
                ),
                unique=("point_id",),
                finite=("paper_legacy_stps", "current_corrected_stps"),
                zero=("gpu_runs_performed",),
            ),
            _CsvContract(
                "fig09/summary.csv",
                29,
                ("metric", "value", "status"),
                unique=("metric",),
                reject_fail_status=True,
            ),
        ),
    ),
    (
        "fig10",
        (
            _CsvContract(
                "fig10/model_points.csv",
                28,
                (
                    "point_id",
                    "model",
                    "bs",
                    "paper_legacy_recomputed_tps",
                    "current_corrected_recomputed_tps",
                ),
                unique=("point_id",),
                finite=(
                    "paper_legacy_recomputed_tps",
                    "current_corrected_recomputed_tps",
                ),
            ),
            _CsvContract(
                "fig10/summary.csv",
                15,
                ("model_version", "scope", "n_points", "gpu_runs_executed"),
                finite=("n_points",),
                zero=("gpu_runs_executed",),
            ),
        ),
    ),
    (
        "fig12",
        (
            *tuple(
                _CsvContract(
                    f"fig12/{version}/model_points.csv",
                    40,
                    ("point_id", "model_version", "model_stps", "gpu_runs"),
                    unique=("point_id",),
                    finite=("model_stps",),
                    zero=("gpu_runs",),
                    exact_values=(("model_version", (version,)),),
                )
                for version in _MODEL_VERSIONS
            ),
            _CsvContract(
                "fig12/version_drift.csv",
                40,
                ("point_id", "paper_legacy_stps", "current_corrected_stps"),
                unique=("point_id",),
                finite=("paper_legacy_stps", "current_corrected_stps"),
            ),
            _CsvContract(
                "fig12/summary.csv",
                16,
                ("metric", "scope", "value", "status"),
                unique=("metric", "scope"),
                reject_fail_status=True,
            ),
        ),
    ),
    (
        "fig13",
        (
            _CsvContract(
                "fig13/model_points.csv",
                420,
                (
                    "topology",
                    "collective",
                    "source_file",
                    "row_index",
                    "recomputed_model_time_ns",
                    "ns3_time_ns",
                    "analytical_time_ns",
                    "gpu_runs",
                ),
                unique=("topology", "collective", "row_index"),
                finite=("recomputed_model_time_ns",),
                zero=("gpu_runs",),
            ),
            _CsvContract(
                "fig13/panel_metrics.csv",
                6,
                (
                    "topology",
                    "collective",
                    "deepstack_weighted_error_pct",
                ),
                unique=("topology", "collective"),
                finite=("deepstack_weighted_error_pct",),
            ),
            _CsvContract(
                "fig13/summary.csv",
                10,
                ("metric", "value", "status"),
                unique=("metric",),
                reject_fail_status=True,
            ),
        ),
    ),
    (
        "fig15",
        (
            *tuple(
                _CsvContract(
                    f"fig15/{version}/fig15_pareto_candidates.csv",
                    784,
                    (
                        "candidate_id",
                        "model",
                        "bs",
                        "model_version",
                        "current_stps_avg",
                        "current_utps_avg",
                    ),
                    unique=("candidate_id",),
                    finite=("current_stps_avg", "current_utps_avg"),
                    exact_values=(("model_version", (version,)),),
                )
                for version in _MODEL_VERSIONS
            ),
            _CsvContract(
                "fig15/version_drift_summary.csv",
                28,
                (
                    "model",
                    "bs",
                    "paper_candidate_id",
                    "paper_legacy_reproduced_stps",
                    "current_corrected_same_config_stps",
                    "paper_winner_rank_within_legacy_manifest_under_current",
                ),
                unique=("model", "bs"),
                finite=(
                    "paper_legacy_reproduced_stps",
                    "current_corrected_same_config_stps",
                    "paper_winner_rank_within_legacy_manifest_under_current",
                ),
            ),
        ),
    ),
    (
        "fig16",
        tuple(
            contract
            for version in _MODEL_VERSIONS
            for contract in (
                _CsvContract(
                    f"fig16/{version}/fixed_winners.csv",
                    80,
                    ("candidate_id", "model_version", "current_stps_avg"),
                    unique=("candidate_id",),
                    finite=("current_stps_avg",),
                    exact_values=(("model_version", (version,)),),
                ),
                _CsvContract(
                    f"fig16/{version}/ratios.csv",
                    28,
                    ("model", "bs", "DeepStack"),
                    unique=("model", "bs"),
                    finite=("DeepStack",),
                ),
                _CsvContract(
                    f"fig16/{version}/summary.csv",
                    5,
                    ("metric", "value", "status"),
                    unique=("metric",),
                    reject_fail_status=True,
                ),
            )
        ),
    ),
    (
        "fig17",
        (
            _CsvContract(
                "fig17/bw_analysis_connected_8x_2x3.csv",
                96,
                (
                    "n",
                    "smem_cap_KiB",
                    "l1_tp_Bpc",
                    "actual_bw_TBs",
                ),
                allowed=(
                    "n",
                    "smem_cap_KiB",
                    "l1_tp_Bpc",
                    "actual_bw_TBs",
                ),
                unique=("n", "smem_cap_KiB", "l1_tp_Bpc"),
                finite=("actual_bw_TBs",),
            ),
            _CsvContract(
                "fig17/summary.csv",
                7,
                ("scope", "peak_layer", "peak_actual_bw_TBs"),
                finite=("peak_layer", "peak_actual_bw_TBs"),
            ),
        ),
    ),
    (
        "fig18_20",
        (
            *tuple(
                contract
                for version, claim_rows in (
                    ("paper_legacy", 26),
                    ("current_corrected", 23),
                )
                for contract in (
                    _CsvContract(
                        f"fig18_20/{version}/model_points.csv",
                        114,
                        (
                            "candidate_id",
                            "terrain_reference_id",
                            "phase",
                            "selection_roles",
                            "model_version",
                            "bs",
                            "dram_total_layers",
                            "dram_active_layers",
                        ),
                        unique=("candidate_id",),
                        finite=(
                            "bs",
                            "dram_total_layers",
                            "dram_active_layers",
                        ),
                        exact_values=(("model_version", (version,)),),
                    ),
                    _CsvContract(
                        f"fig18_20/{version}/claim_summary.csv",
                        claim_rows,
                        ("figure", "metric", "value", "status"),
                        reject_fail_status=True,
                    ),
                    _CsvContract(
                        f"fig18_20/{version}/terrain_sample_validation.csv",
                        22,
                        (
                            "candidate_id",
                            "terrain_reference_id",
                            "model_version",
                            "bs",
                            "dram_total_layers",
                            "dram_active_layers",
                            "archived_raw_stps",
                            "recomputed_raw_stps",
                            "raw_stps_error_pct",
                            "archived_temperature_C",
                            "recomputed_temperature_C",
                            "temperature_error_C",
                        ),
                        unique=("candidate_id", "terrain_reference_id"),
                        finite=(
                            "bs",
                            "dram_total_layers",
                            "dram_active_layers",
                            "archived_raw_stps",
                            "recomputed_raw_stps",
                            "raw_stps_error_pct",
                            "archived_temperature_C",
                            "recomputed_temperature_C",
                            "temperature_error_C",
                        ),
                        exact_values=(("model_version", (version,)),),
                    ),
                )
            ),
            _CsvContract(
                "fig18_20/archive/fig18_throughput_curves.csv",
                110,
                ("phase", "bs", "m", "best_scaled_stps"),
                unique=("phase", "bs", "m"),
                finite=("best_scaled_stps",),
            ),
            _CsvContract(
                "fig18_20/archive/fig19_metric_grid.csv",
                108,
                (
                    "phase",
                    "bs",
                    "m",
                    "n",
                    "best_effective_stps",
                    "best_tokens_per_joule",
                ),
                unique=("phase", "bs", "m", "n"),
                finite=("best_effective_stps", "best_tokens_per_joule"),
            ),
            _CsvContract(
                "fig18_20/archive/fig20_decode_thermal_terrain.csv",
                19233,
                (
                    "terrain_reference_id",
                    "dram_total_layers",
                    "dram_active_layers",
                    "bs",
                    "temperature_C",
                    "raw_stps",
                ),
                unique=("terrain_reference_id",),
                finite=(
                    "dram_total_layers",
                    "dram_active_layers",
                    "bs",
                    "temperature_C",
                    "raw_stps",
                ),
            ),
            _CsvContract(
                "fig18_20/archive/archive_manifest.csv",
                3,
                (
                    "dataset",
                    "file",
                    "rows",
                    "sha256",
                    "exhaustive_search_rerun",
                    "status",
                ),
                unique=("dataset", "file", "sha256"),
                finite=("rows",),
                false=("exhaustive_search_rerun",),
                exact_values=(
                    ("status", ("ARCHIVED_DSE_PROJECTION",)),
                ),
            ),
            _CsvContract(
                "fig18_20/dual_path_drift.csv",
                114,
                ("candidate_id", "recomputed_scaled_stps_drift_pct"),
                unique=("candidate_id",),
                finite=("recomputed_scaled_stps_drift_pct",),
            ),
            _CsvContract(
                "fig18_20/dual_path_drift_summary.csv",
                14,
                ("phase", "bs", "configs", "max_abs_scaled_stps_drift_pct"),
                unique=("phase", "bs"),
                finite=("configs", "max_abs_scaled_stps_drift_pct"),
            ),
        ),
    ),
    (
        "fig21_paper_legacy",
        (
            _CsvContract(
                "fig21/paper_legacy/raw.csv",
                180,
                (
                    "candidate_id",
                    "model_version",
                    "bs",
                    "latency_multiplier",
                    "bw_multiplier",
                    "tp_transform_moe",
                    "reproduced_stps_avg",
                    "reproduced_utps_avg",
                    "delta_from_paper_stps_pct",
                    "delta_from_paper_utps_pct",
                ),
                allowed=(
                    "candidate_id",
                    "model_version",
                    "bs",
                    "latency_multiplier",
                    "bw_multiplier",
                    "tp_transform_moe",
                    "reproduced_stps_avg",
                    "reproduced_utps_avg",
                    "delta_from_paper_stps_pct",
                    "delta_from_paper_utps_pct",
                ),
                unique=("candidate_id",),
                finite=(
                    "bs",
                    "latency_multiplier",
                    "bw_multiplier",
                    "reproduced_stps_avg",
                    "reproduced_utps_avg",
                    "delta_from_paper_stps_pct",
                    "delta_from_paper_utps_pct",
                ),
                exact_values=(("model_version", ("paper_legacy",)),),
            ),
            _CsvContract(
                "fig21/paper_legacy/summary.csv",
                8,
                (
                    "metric",
                    "definition",
                    "paper_reported",
                    "paper_recomputed",
                    "reproduced",
                    "delta_from_paper_recomputed_pct",
                ),
                allowed=(
                    "metric",
                    "definition",
                    "paper_reported",
                    "paper_recomputed",
                    "reproduced",
                    "delta_from_paper_recomputed_pct",
                ),
                unique=("metric",),
                finite=(
                    "paper_reported",
                    "paper_recomputed",
                    "reproduced",
                    "delta_from_paper_recomputed_pct",
                ),
            ),
        ),
    ),
    (
        "fig21_current_corrected",
        (
            _CsvContract(
                "fig21/current_corrected/raw.csv",
                180,
                (
                    "candidate_id",
                    "model_version",
                    "bs",
                    "latency_multiplier",
                    "bw_multiplier",
                    "tp_transform_moe",
                    "reproduced_stps_avg",
                    "reproduced_utps_avg",
                    "delta_from_paper_stps_pct",
                    "delta_from_paper_utps_pct",
                ),
                allowed=(
                    "candidate_id",
                    "model_version",
                    "bs",
                    "latency_multiplier",
                    "bw_multiplier",
                    "tp_transform_moe",
                    "reproduced_stps_avg",
                    "reproduced_utps_avg",
                    "delta_from_paper_stps_pct",
                    "delta_from_paper_utps_pct",
                ),
                unique=("candidate_id",),
                finite=(
                    "bs",
                    "latency_multiplier",
                    "bw_multiplier",
                    "reproduced_stps_avg",
                    "reproduced_utps_avg",
                    "delta_from_paper_stps_pct",
                    "delta_from_paper_utps_pct",
                ),
                exact_values=(("model_version", ("current_corrected",)),),
            ),
            _CsvContract(
                "fig21/current_corrected/summary.csv",
                8,
                (
                    "metric",
                    "definition",
                    "paper_reported",
                    "paper_recomputed",
                    "reproduced",
                    "delta_from_paper_recomputed_pct",
                ),
                allowed=(
                    "metric",
                    "definition",
                    "paper_reported",
                    "paper_recomputed",
                    "reproduced",
                    "delta_from_paper_recomputed_pct",
                ),
                unique=("metric",),
                finite=(
                    "paper_reported",
                    "paper_recomputed",
                    "reproduced",
                    "delta_from_paper_recomputed_pct",
                ),
            ),
        ),
    ),
    (
        "fig22",
        (
            *tuple(
                contract
                for version in _MODEL_VERSIONS
                for contract in (
                    _CsvContract(
                        f"fig22/{version}/model_points.csv",
                        756,
                        (
                            "point_id",
                            "model_version",
                            "phase",
                            "baseline_noc",
                            "scaled_layer",
                            "bw_multiplier",
                            "bs",
                            "seq",
                            "reproduced_logic_stps",
                            "reproduced_dram_stps",
                        ),
                        allowed=(
                            "point_id",
                            "model_version",
                            "phase",
                            "baseline_noc",
                            "scaled_layer",
                            "bw_multiplier",
                            "bs",
                            "seq",
                            "reproduced_logic_stps",
                            "reproduced_dram_stps",
                        ),
                        unique=("point_id",),
                        finite=(
                            "bw_multiplier",
                            "bs",
                            "seq",
                            "reproduced_logic_stps",
                            "reproduced_dram_stps",
                        ),
                        exact_values=(("model_version", (version,)),),
                    ),
                    _CsvContract(
                        f"fig22/{version}/normalized_summary.csv",
                        84,
                        (
                            "model_version",
                            "phase",
                            "baseline_noc",
                            "scaled_layer",
                            "bw_multiplier",
                            "logic_norm_stps",
                            "dram_norm_stps",
                        ),
                        allowed=(
                            "model_version",
                            "phase",
                            "baseline_noc",
                            "scaled_layer",
                            "bw_multiplier",
                            "logic_norm_stps",
                            "dram_norm_stps",
                        ),
                        unique=(
                            "phase",
                            "baseline_noc",
                            "scaled_layer",
                            "bw_multiplier",
                        ),
                        finite=("logic_norm_stps", "dram_norm_stps"),
                        exact_values=(("model_version", (version,)),),
                    ),
                    _CsvContract(
                        f"fig22/{version}/summary.csv",
                        15,
                        (
                            "metric",
                            "scope",
                            "value",
                            "expected",
                            "tolerance",
                            "status",
                            "notes",
                        ),
                        allowed=(
                            "metric",
                            "scope",
                            "value",
                            "expected",
                            "tolerance",
                            "status",
                            "notes",
                        ),
                        unique=("metric",),
                        reject_fail_status=True,
                    ),
                )
            ),
            _CsvContract(
                "fig22/version_drift.csv",
                756,
                (
                    "point_id",
                    "phase",
                    "baseline_noc",
                    "scaled_layer",
                    "bw_multiplier",
                    "bs",
                    "seq",
                    "logic_corrected_vs_legacy_pct",
                    "dram_corrected_vs_legacy_pct",
                ),
                allowed=(
                    "point_id",
                    "phase",
                    "baseline_noc",
                    "scaled_layer",
                    "bw_multiplier",
                    "bs",
                    "seq",
                    "logic_corrected_vs_legacy_pct",
                    "dram_corrected_vs_legacy_pct",
                ),
                unique=("point_id",),
                finite=(
                    "bw_multiplier",
                    "bs",
                    "seq",
                    "logic_corrected_vs_legacy_pct",
                    "dram_corrected_vs_legacy_pct",
                ),
            ),
            _CsvContract(
                "fig22/version_drift_summary.csv",
                9,
                (
                    "phase",
                    "batch_scope",
                    "model_values",
                    "mean_absolute_drift_pct",
                    "maximum_absolute_drift_pct",
                ),
                allowed=(
                    "phase",
                    "batch_scope",
                    "model_values",
                    "mean_absolute_drift_pct",
                    "maximum_absolute_drift_pct",
                ),
                unique=("phase", "batch_scope"),
                finite=(
                    "model_values",
                    "mean_absolute_drift_pct",
                    "maximum_absolute_drift_pct",
                ),
            ),
        ),
    ),
    (
        "fig23",
        tuple(
            contract
            for version in _MODEL_VERSIONS
            for contract in (
                _CsvContract(
                    f"fig23/{version}/raw.csv",
                    84,
                    (
                        "candidate_id",
                        "evaluation_key",
                        "model_version",
                        "reproduced_stps",
                    ),
                    unique=("candidate_id",),
                    finite=("reproduced_stps",),
                    exact_values=(("model_version", (version,)),),
                ),
                _CsvContract(
                    f"fig23/{version}/summary.csv",
                    28,
                    ("phase", "model", "bs", "reproduced_flexible_over_astra"),
                    unique=("phase", "model", "bs"),
                    finite=("reproduced_flexible_over_astra",),
                ),
                _CsvContract(
                    f"fig23/{version}/headline_claims.csv",
                    6,
                    ("model_version", "model", "phase", "bs", "metric", "reproduced"),
                    unique=("model_version", "model", "phase", "bs", "metric"),
                    finite=("reproduced",),
                    exact_values=(("model_version", (version,)),),
                ),
            )
        ),
    ),
    (
        "table4",
        (
            _CsvContract(
                "table4/model_points.csv",
                28,
                (
                    "candidate_id",
                    "model_version",
                    "step",
                    "step_name",
                    "bs",
                    "recomputed_stps_avg",
                    "delta_from_paper_stps_pct",
                ),
                allowed=(
                    "candidate_id",
                    "model_version",
                    "step",
                    "step_name",
                    "bs",
                    "recomputed_stps_avg",
                    "delta_from_paper_stps_pct",
                ),
                unique=("candidate_id", "model_version"),
                finite=(
                    "step",
                    "bs",
                    "recomputed_stps_avg",
                    "delta_from_paper_stps_pct",
                ),
                exact_values=(("model_version", _MODEL_VERSIONS),),
            ),
            _CsvContract(
                "table4/summary.csv",
                4,
                (
                    "model_version",
                    "bs",
                    "step1_stps_avg",
                    "step7_stps_avg",
                    "speedup_x",
                    "rounded_speedup_x",
                    "paper_claim_rounded_speedup_x",
                    "interpretation",
                ),
                allowed=(
                    "model_version",
                    "bs",
                    "step1_stps_avg",
                    "step7_stps_avg",
                    "speedup_x",
                    "rounded_speedup_x",
                    "paper_claim_rounded_speedup_x",
                    "interpretation",
                ),
                unique=("model_version", "bs"),
                finite=(
                    "bs",
                    "step1_stps_avg",
                    "step7_stps_avg",
                    "speedup_x",
                    "rounded_speedup_x",
                    "paper_claim_rounded_speedup_x",
                ),
                exact_values=(("model_version", _MODEL_VERSIONS),),
            ),
        ),
    ),
    (
        "figures",
        (
            _CsvContract(
                "figures/plot_manifest.csv",
                16,
                (
                    "figure",
                    "stem",
                    "model_version",
                    "source_files",
                    "png",
                    "pdf",
                    "png_bytes",
                    "pdf_bytes",
                    "status",
                    "note",
                ),
                unique=("stem",),
                finite=("png_bytes", "pdf_bytes"),
                exact_values=(
                    ("model_version", ("paper_legacy",)),
                    ("status", ("PASS",)),
                ),
            ),
        ),
    ),
)


def _finite_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    if not columns:
        return True
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return bool(
        numeric.apply(lambda column: column.map(math.isfinite)).all().all()
    )


def _is_explicit_false(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    return str(value).strip().lower() in {"false", "0", "no"}


def _require_scoped_regular_file(path: Path, scope_root: Path) -> Path:
    """Require a real file whose full path stays inside one result scope."""

    resolved_root = scope_root.resolve(strict=True)
    try:
        relative = path.relative_to(resolved_root)
    except ValueError as exc:
        raise AssertionError(f"stage output is outside result scope: {path}") from exc

    cursor = resolved_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise AssertionError(f"stage output must not use a symbolic link: {cursor}")
    if not path.is_file():
        raise AssertionError(f"missing stage output: {path}")
    try:
        path.resolve(strict=True).relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise AssertionError(f"stage output resolves outside result scope: {path}") from exc
    return path


def _read_csv_contract(
    path: Path, contract: _CsvContract, *, scope_root: Path
) -> pd.DataFrame:
    _require_scoped_regular_file(path, scope_root)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise AssertionError(f"could not parse stage output CSV {path}: {exc}") from exc

    if len(frame) != contract.rows:
        raise AssertionError(
            f"{path}: expected {contract.rows} rows, got {len(frame)}"
        )
    missing = sorted(set(contract.required) - set(frame.columns))
    if missing:
        raise AssertionError(f"{path}: missing columns: {', '.join(missing)}")
    if contract.allowed:
        unexpected = sorted(set(frame.columns) - set(contract.allowed))
        if unexpected:
            raise AssertionError(
                f"{path}: unexpected columns outside the publish-safe schema: "
                f"{', '.join(unexpected)}"
            )

    if contract.unique:
        keys = list(contract.unique)
        if frame[keys].isna().any().any():
            raise AssertionError(f"{path}: null value in unique key {keys}")
        duplicate_mask = frame.duplicated(keys, keep=False)
        if duplicate_mask.any():
            examples = frame.loc[duplicate_mask, keys].head(3).to_dict("records")
            raise AssertionError(
                f"{path}: duplicate unique key {keys}; examples={examples}"
            )

    if not _finite_numeric(frame, contract.finite):
        raise AssertionError(
            f"{path}: non-finite or non-numeric values in {list(contract.finite)}"
        )
    for column in contract.zero:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.map(math.isfinite).all() or not values.eq(0).all():
            raise AssertionError(f"{path}: {column} must contain only numeric zero")
    for column in contract.false:
        if not frame[column].map(_is_explicit_false).all():
            raise AssertionError(f"{path}: {column} must contain only false values")
    for column, expected_values in contract.exact_values:
        actual = set(frame[column].astype(str))
        expected = set(expected_values)
        if actual != expected:
            raise AssertionError(
                f"{path}: expected {column} values {sorted(expected)}, "
                f"found {sorted(actual)}"
            )
    if contract.reject_fail_status:
        statuses = frame["status"].astype(str).str.strip().str.upper()
        failed = frame.loc[statuses.str.startswith("FAIL"), "status"]
        if not failed.empty:
            raise AssertionError(
                f"{path}: summary contains FAIL status values: "
                f"{failed.astype(str).tolist()[:5]}"
            )
        allowed = {"PASS", "INFO", "INFO_FIXED_CONFIG_DIAGNOSTIC"}
        unexpected = sorted(set(statuses) - allowed)
        if unexpected:
            raise AssertionError(
                f"{path}: summary contains unexpected status values: {unexpected}"
            )
    return frame


def _read_json_marker(
    path: Path, expected: dict[str, Any], *, scope_root: Path
) -> dict[str, Any]:
    _require_scoped_regular_file(path, scope_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(f"invalid JSON stage marker {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssertionError(f"{path}: JSON marker must be an object")
    for key, expected_value in expected.items():
        if key not in value:
            raise AssertionError(f"{path}: marker is missing {key!r}")
        actual = value[key]
        matches = (
            actual is expected_value
            if isinstance(expected_value, bool)
            else actual == expected_value
        )
        if not matches:
            raise AssertionError(
                f"{path}: expected {key}={expected_value!r}, found {actual!r}"
            )
    return value


def _read_text_marker(
    path: Path,
    *,
    scope_root: Path,
    first_line: str | None = None,
    first_line_contains: str | None = None,
    required_fragments: tuple[str, ...] = (),
) -> str:
    _require_scoped_regular_file(path, scope_root)
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"empty stage marker: {path}")
    if first_line is not None and lines[0] != first_line:
        raise AssertionError(
            f"{path}: expected first line {first_line!r}, found {lines[0]!r}"
        )
    if first_line_contains is not None and first_line_contains not in lines[0]:
        raise AssertionError(
            f"{path}: first line does not contain {first_line_contains!r}"
        )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise AssertionError(f"{path}: marker is missing fragments {missing}")
    return text


def _claimed_plot_size(path: Path, column: str, value: object) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{path}: {column} must be an integer byte count") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise AssertionError(f"{path}: {column} must be an integer byte count")
    return int(numeric)


def _plot_file_names(directory: Path, suffix: str) -> set[str]:
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == suffix
    }


def _verify_plot_outputs(figures_dir: Path, manifest: pd.DataFrame) -> int:
    """Validate the plotting stage's manifest and exact binary-file closure."""

    manifest_path = figures_dir / "plot_manifest.csv"
    if not figures_dir.is_dir():
        raise AssertionError(f"missing figures output directory: {figures_dir}")

    actual_stems = set(manifest["stem"].astype(str))
    if actual_stems != _EXPECTED_PLOT_STEMS:
        raise AssertionError(
            f"{manifest_path}: plot stem closure mismatch: "
            f"missing={sorted(_EXPECTED_PLOT_STEMS - actual_stems)} "
            f"unexpected={sorted(actual_stems - _EXPECTED_PLOT_STEMS)}"
        )

    for column in ("figure", "source_files"):
        values = manifest[column].fillna("").astype(str).str.strip()
        if values.eq("").any():
            rows = values.index[values.eq("")].tolist()[:5]
            raise AssertionError(
                f"{manifest_path}: blank {column} values at rows {rows}"
            )

    expected_png_names = {f"{stem}.png" for stem in _EXPECTED_PLOT_STEMS}
    expected_pdf_names = {f"{stem}.pdf" for stem in _EXPECTED_PLOT_STEMS}
    expected_all_names = expected_png_names | expected_pdf_names | {
        "PASS",
        "plot_manifest.csv",
    }
    actual_all_names = {
        path.relative_to(figures_dir).as_posix()
        for path in figures_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_all_names != expected_all_names:
        raise AssertionError(
            f"{figures_dir}: figure file closure mismatch: "
            f"missing={sorted(expected_all_names - actual_all_names)} "
            f"unexpected={sorted(actual_all_names - expected_all_names)}"
        )
    actual_png_names = _plot_file_names(figures_dir, ".png")
    actual_pdf_names = _plot_file_names(figures_dir, ".pdf")
    for kind, expected, actual in (
        ("PNG", expected_png_names, actual_png_names),
        ("PDF", expected_pdf_names, actual_pdf_names),
    ):
        if actual != expected:
            raise AssertionError(
                f"{figures_dir}: {kind} file closure mismatch: "
                f"missing={sorted(expected - actual)} "
                f"unexpected={sorted(actual - expected)}"
            )

    resolved_directory = figures_dir.resolve(strict=True)
    result_root = resolved_directory.parent
    plot_count = 0
    for row in manifest.itertuples(index=False):
        stem = str(row.stem)
        for source in str(row.source_files).split(";"):
            relative = Path(source.strip())
            if (
                not source.strip()
                or relative.is_absolute()
                or not relative.parts
                or relative.parts[0] != "stages"
                or ".." in relative.parts
                or "\\" in source
                or relative.suffix.lower() != ".csv"
            ):
                raise AssertionError(
                    f"{manifest_path}: source_files entries must be scope-relative "
                    f"stages/... CSV paths, found {source!r}"
                )
            _require_scoped_regular_file(result_root / relative, result_root)
        for kind, minimum_bytes in (("png", _MIN_PNG_BYTES), ("pdf", _MIN_PDF_BYTES)):
            name = str(getattr(row, kind)).strip()
            expected_name = f"{stem}.{kind}"
            if (
                name != expected_name
                or Path(name).name != name
                or "/" in name
                or "\\" in name
            ):
                raise AssertionError(
                    f"{manifest_path}: {kind} must be the basename "
                    f"{expected_name!r}, found {name!r}"
                )
            target = figures_dir / name
            if target.is_symlink():
                raise AssertionError(f"{target}: plot output must not be a symbolic link")
            if not target.is_file():
                raise AssertionError(f"missing plot output: {target}")
            try:
                resolved_target = target.resolve(strict=True)
            except OSError as exc:
                raise AssertionError(f"could not resolve plot output {target}: {exc}") from exc
            if resolved_target.parent != resolved_directory:
                raise AssertionError(
                    f"{target}: plot output resolves outside {figures_dir}"
                )

            claimed_size = _claimed_plot_size(
                manifest_path, f"{kind}_bytes", getattr(row, f"{kind}_bytes")
            )
            actual_size = target.stat().st_size
            if actual_size != claimed_size:
                raise AssertionError(
                    f"{target}: manifest records {claimed_size} bytes, "
                    f"actual size is {actual_size}"
                )
            if actual_size <= minimum_bytes:
                raise AssertionError(
                    f"{target}: implausibly small {kind.upper()} output "
                    f"({actual_size} bytes; must exceed {minimum_bytes})"
                )

            with target.open("rb") as handle:
                header = handle.read(24)
            if kind == "png":
                if header[:8] != b"\x89PNG\r\n\x1a\n":
                    raise AssertionError(f"{target}: invalid PNG signature")
                if len(header) < 24 or struct.unpack(">I", header[8:12])[0] != 13:
                    raise AssertionError(f"{target}: invalid PNG IHDR chunk length")
                if header[12:16] != b"IHDR":
                    raise AssertionError(f"{target}: PNG does not begin with IHDR")
                width, height = struct.unpack(">II", header[16:24])
                if not (
                    _MIN_PLOT_DIMENSION <= width <= _MAX_PLOT_DIMENSION
                    and _MIN_PLOT_DIMENSION <= height <= _MAX_PLOT_DIMENSION
                ):
                    raise AssertionError(
                        f"{target}: implausible PNG dimensions {width}x{height}"
                    )
            elif header[:4] != b"%PDF":
                raise AssertionError(f"{target}: invalid PDF header")
            plot_count += 1

    if plot_count != 32:
        raise AssertionError(f"expected 32 plot files, verified {plot_count}")
    return plot_count


def _verify_fig18_20_archive_and_samples(
    result_root: Path, frames: dict[str, pd.DataFrame]
) -> None:
    """Close the archived plot bundle and its model-rerun audit samples."""

    archive_files = {
        "fig18_throughput_curves.csv": 110,
        "fig19_metric_grid.csv": 108,
        "fig20_decode_thermal_terrain.csv": 19233,
    }
    manifest = frames["fig18_20/archive/archive_manifest.csv"].copy()
    manifest["file"] = manifest["file"].astype(str)
    if not manifest["file"].is_unique:
        raise AssertionError("Fig. 18--20 archive manifest file names are not unique")
    if set(manifest["file"]) != set(archive_files):
        raise AssertionError(
            "Fig. 18--20 archive manifest file set differs from the output contract"
        )
    indexed_manifest = manifest.set_index("file")
    for filename, expected_rows in archive_files.items():
        key = f"fig18_20/archive/{filename}"
        frame = frames[key]
        record = indexed_manifest.loc[filename]
        claimed_rows = float(record["rows"])
        if not claimed_rows.is_integer() or int(claimed_rows) != expected_rows:
            raise AssertionError(
                f"Fig. 18--20 archive manifest gives invalid row count for {filename}"
            )
        expected_dataset = filename.removesuffix(".csv")
        if str(record["dataset"]) != expected_dataset:
            raise AssertionError(
                f"Fig. 18--20 archive manifest gives invalid dataset for {filename}"
            )
        path = result_root / "stages" / key
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(record["sha256"]).lower() != actual_sha256:
            raise AssertionError(
                f"Fig. 18--20 archive manifest checksum mismatch for {filename}"
            )
        if len(frame) != expected_rows:
            raise AssertionError(
                f"Fig. 18--20 archive frame has wrong row count for {filename}"
            )

    expected_strata = {
        (bs, layer) for bs in (4, 1024) for layer in range(2, 13)
    }

    def strata(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return set(
            zip(
                pd.to_numeric(frame["bs"]).astype(int),
                pd.to_numeric(frame["dram_total_layers"]).astype(int),
                strict=True,
            )
        )

    terrain = frames["fig18_20/archive/fig20_decode_thermal_terrain.csv"]
    curves = frames["fig18_20/archive/fig18_throughput_curves.csv"]
    if "archived_winner_id" in curves.columns:
        raise AssertionError(
            "Fig. 18 archive must not expose point IDs shared with Fig. 20"
        )
    forbidden_terrain_columns = {
        "thermal_resistance_CpW",
        "chip_power_W",
        "noc_power_per_device_W",
        "total_power_per_device_W",
        "tokens_per_joule",
    }
    leaked_terrain_columns = sorted(
        forbidden_terrain_columns.intersection(terrain.columns)
    )
    if leaked_terrain_columns:
        raise AssertionError(
            "Fig. 20 terrain contains non-publishable model intermediates: "
            f"{leaked_terrain_columns}"
        )
    terrain_total = pd.to_numeric(terrain["dram_total_layers"]).astype(int)
    terrain_active = pd.to_numeric(terrain["dram_active_layers"]).astype(int)
    if not terrain_total.eq(terrain_active).all():
        raise AssertionError("Fig. 20 terrain must contain only fully connected stacks")
    if strata(terrain) != expected_strata:
        raise AssertionError("Fig. 20 terrain does not cover every displayed BS/layer stratum")

    identity_columns = [
        "candidate_id",
        "terrain_reference_id",
        "bs",
        "dram_total_layers",
        "dram_active_layers",
    ]

    def identities(frame: pd.DataFrame) -> set[tuple[str, str, int, int, int]]:
        return {
            (
                str(row.candidate_id),
                str(row.terrain_reference_id),
                int(row.bs),
                int(row.dram_total_layers),
                int(row.dram_active_layers),
            )
            for row in frame[identity_columns].itertuples(index=False)
        }

    for version in _MODEL_VERSIONS:
        points = frames[f"fig18_20/{version}/model_points.csv"]
        samples = frames[f"fig18_20/{version}/terrain_sample_validation.csv"]
        leaked_point_columns = sorted(
            column
            for column in points.columns
            if column.startswith(("paper_", "recomputed_"))
            or "temperature" in column.lower()
            or "power" in column.lower()
            or "joule" in column.lower()
            or column in {"effective_stps", "raw_stps", "scaled_stps"}
        )
        if leaked_point_columns:
            raise AssertionError(
                f"Figs. 18--20 {version} model points contain non-publishable "
                f"intermediates: {leaked_point_columns}"
            )
        if (
            not samples["candidate_id"].is_unique
            or not samples["terrain_reference_id"].is_unique
        ):
            raise AssertionError(f"Fig. 20 {version} terrain sample IDs are not unique")
        if "stats_run_id" in points.columns or "stats_run_id" in samples.columns:
            raise AssertionError(
                f"Fig. 20 {version} outputs expose source-run IDs"
            )
        sample_total = pd.to_numeric(samples["dram_total_layers"]).astype(int)
        sample_active = pd.to_numeric(samples["dram_active_layers"]).astype(int)
        if not sample_total.eq(sample_active).all() or strata(samples) != expected_strata:
            raise AssertionError(
                f"Fig. 20 {version} samples must contain one fully connected point "
                "per displayed BS/layer stratum"
            )
        selected = points[
            points["selection_roles"].astype(str).map(
                lambda value: "fig20_archive_sample" in value.split(";")
            )
        ]
        if identities(samples) != identities(selected):
            raise AssertionError(
                f"Fig. 20 {version} sample table differs from its model-point roles"
            )
        forbidden_sample_columns = sorted(
            column
            for column in samples.columns
            if "scaled" in column.lower()
            or "power" in column.lower()
            or "joule" in column.lower()
        )
        if forbidden_sample_columns:
            raise AssertionError(
                f"Fig. 20 {version} sample table contains composable "
                f"intermediates: {forbidden_sample_columns}"
            )

        terrain_lookup = terrain[
            [
                "terrain_reference_id",
                "bs",
                "dram_total_layers",
                "dram_active_layers",
                "temperature_C",
                "raw_stps",
            ]
        ]
        archive_comparison = samples[
            [
                "terrain_reference_id",
                "bs",
                "dram_total_layers",
                "dram_active_layers",
                "archived_raw_stps",
                "archived_temperature_C",
            ]
        ].merge(
            terrain_lookup,
            on="terrain_reference_id",
            validate="one_to_one",
        )
        if len(archive_comparison) != len(samples):
            raise AssertionError(f"Fig. 20 {version} sample IDs are absent from terrain")
        for column in ("bs", "dram_total_layers", "dram_active_layers"):
            if not pd.to_numeric(archive_comparison[f"{column}_x"]).astype(int).eq(
                pd.to_numeric(archive_comparison[f"{column}_y"]).astype(int)
            ).all():
                raise AssertionError(
                    f"Fig. 20 {version} sample {column} differs from terrain"
                )
        if not all(
            math.isclose(
                float(sample_value),
                float(archive_value),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            for sample_value, archive_value in zip(
                archive_comparison["archived_raw_stps"],
                archive_comparison["raw_stps"],
                strict=True,
            )
        ):
            raise AssertionError(f"Fig. 20 {version} sample raw STPS differs from terrain")
        if not all(
            math.isclose(
                float(sample_value),
                float(archive_value),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for sample_value, archive_value in zip(
                archive_comparison["archived_temperature_C"],
                archive_comparison["temperature_C"],
                strict=True,
            )
        ):
            raise AssertionError(
                f"Fig. 20 {version} sample temperature differs from terrain"
            )

    from .fig18_20 import (  # Imported lazily to keep verifier startup light.
        LEGACY_MAX_ERROR_PCT,
        LEGACY_MAX_TEMPERATURE_ERROR_C,
    )

    legacy_samples = frames[
        "fig18_20/paper_legacy/terrain_sample_validation.csv"
    ]
    if (
        float(legacy_samples["raw_stps_error_pct"].abs().max())
        > LEGACY_MAX_ERROR_PCT
    ):
        raise AssertionError(
            "Fig. 20 legacy sample raw_stps_error_pct exceeds tolerance"
        )
    if (
        float(legacy_samples["temperature_error_C"].abs().max())
        > LEGACY_MAX_TEMPERATURE_ERROR_C
    ):
        raise AssertionError("Fig. 20 legacy sample temperature error exceeds tolerance")

    dual_drift = frames["fig18_20/dual_path_drift.csv"]
    absolute_suffixes = ("_paper_legacy", "_current_corrected")
    leaked_dual_columns = sorted(
        column
        for column in dual_drift.columns
        if column.endswith(absolute_suffixes)
    )
    if leaked_dual_columns:
        raise AssertionError(
            "Figs. 18--20 dual drift contains absolute per-path values: "
            f"{leaked_dual_columns}"
        )


def verify_stage_outputs(
    result_root: Path,
) -> dict[str, object]:
    """Validate the complete, bounded 15-stage reproduction output closure.

    Numerical stage outputs are resolved only below ``result_root / "stages"``.
    Rendered figures and their manifest live below ``result_root / "figures"``.
    No legacy sibling under the repository-wide results directory participates
    in verification.  The numerical Fig. 15 regression checks remain in
    :func:`verify_reproduce`.
    """

    from .workflow import reproduce_stages

    result_root = result_root.resolve(strict=True)

    expected_stage_names = tuple(
        stage.name for stage in reproduce_stages(1, "both")
    )
    contract_stage_names = tuple(stage for stage, _ in _STAGE_CSV_CONTRACTS)
    if contract_stage_names != expected_stage_names:
        raise AssertionError(
            "internal output-contract registry differs from workflow stages: "
            f"contracts={contract_stage_names}, workflow={expected_stage_names}"
        )

    frames: dict[str, pd.DataFrame] = {}
    csv_count = 0
    stage_root = result_root / "stages"
    expected_stage_files = _EXPECTED_STAGE_AUXILIARY_FILES | {
        contract.path
        for stage, contracts in _STAGE_CSV_CONTRACTS
        if stage != "figures"
        for contract in contracts
    }
    actual_stage_files = {
        path.relative_to(stage_root).as_posix()
        for path in stage_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_stage_files != expected_stage_files:
        raise AssertionError(
            f"{stage_root}: stage file closure mismatch: "
            f"missing={sorted(expected_stage_files - actual_stage_files)} "
            f"unexpected={sorted(actual_stage_files - expected_stage_files)}"
        )
    for stage, contracts in _STAGE_CSV_CONTRACTS:
        for contract in contracts:
            base = result_root if stage == "figures" else stage_root
            path = base / contract.path
            frames[contract.path] = _read_csv_contract(
                path, contract, scope_root=result_root
            )
            csv_count += 1

    # Cross-file closure checks catch correctly shaped but mismatched outputs.
    fig12_legacy_ids = set(
        frames["fig12/paper_legacy/model_points.csv"].point_id.astype(str)
    )
    fig12_current_ids = set(
        frames["fig12/current_corrected/model_points.csv"].point_id.astype(str)
    )
    if fig12_legacy_ids != fig12_current_ids:
        raise AssertionError("Fig. 12 model-version point_id sets differ")

    fig15_legacy = frames["fig15/paper_legacy/fig15_pareto_candidates.csv"]
    fig15_current = frames["fig15/current_corrected/fig15_pareto_candidates.csv"]
    if set(fig15_legacy.candidate_id.astype(str)) != set(
        fig15_current.candidate_id.astype(str)
    ):
        raise AssertionError("Fig. 15 model-version candidate_id sets differ")
    for version, frame in (
        ("paper_legacy", fig15_legacy),
        ("current_corrected", fig15_current),
    ):
        if frame.groupby(["model", "bs"]).ngroups != 28:
            raise AssertionError(
                f"Fig. 15 {version}: expected 28 model/batch-size groups"
            )

    fig17_summary = frames["fig17/summary.csv"]
    if int(fig17_summary.scope.astype(str).eq("global").sum()) != 1:
        raise AssertionError("Fig. 17 summary must contain exactly one global row")

    _verify_fig18_20_archive_and_samples(result_root, frames)

    for version in _MODEL_VERSIONS:
        raw = frames[f"fig23/{version}/raw.csv"]
        if raw.evaluation_key.nunique(dropna=False) != 75:
            raise AssertionError(
                f"Fig. 23 {version}: expected 75 unique fixed configurations"
            )
    table4 = frames["table4/model_points.csv"]
    if table4.candidate_id.nunique(dropna=False) != 14:
        raise AssertionError("Table 4 must contain 14 fixed candidate IDs")

    plot_count = _verify_plot_outputs(
        result_root / "figures", frames["figures/plot_manifest.csv"]
    )

    marker_count = 0
    _read_json_marker(
        stage_root / "fig08" / "verification.json",
        {"status": "PASS", "points": 52, "gpu_execution": False},
        scope_root=result_root,
    )
    marker_count += 1
    _read_json_marker(
        stage_root / "fig09" / "PASS",
        {
            "status": "PASS",
            "model_points": 40,
            "model_evaluations": 80,
            "gpu_runs_performed": 0,
        },
        scope_root=result_root,
    )
    marker_count += 1
    _read_json_marker(
        stage_root / "fig10" / "verification.json",
        {
            "status": "PASS",
            "points": 28,
            "cpu_model_evaluations": 56,
            "gpu_runs_executed": 0,
        },
        scope_root=result_root,
    )
    _read_text_marker(
        stage_root / "fig10" / "PASS.txt",
        scope_root=result_root,
        first_line="PASS",
        required_fragments=("CPU model evaluations: 56", "GPU runs executed: 0"),
    )
    marker_count += 2
    _read_text_marker(
        stage_root / "fig12" / "PASS",
        scope_root=result_root,
        first_line="PASS",
        required_fragments=("points_per_path=40", "gpu_runs=0"),
    )
    marker_count += 1

    for version in _MODEL_VERSIONS:
        expected_status = {
            "status": "PASS",
            "model_version": version,
            "fixed_configurations": 114,
            "gpu_required": False,
            "search_performed": False,
        }
        _read_json_marker(
            stage_root / "fig18_20" / version / "status.json",
            expected_status,
            scope_root=result_root,
        )
        _read_json_marker(
            stage_root / "fig18_20" / version / "PASS",
            expected_status,
            scope_root=result_root,
        )
        marker_count += 2

    for version in _MODEL_VERSIONS:
        _read_text_marker(
            stage_root / "fig22" / version / "PASS",
            scope_root=result_root,
            first_line="PASS",
            required_fragments=(
                f"model_version={version}",
                "fixed_points=756",
                "model_calls=1930",
                "gpu_runs=0",
            ),
        )
        marker_count += 1
    _read_text_marker(
        stage_root / "fig22" / "PASS",
        scope_root=result_root,
        first_line="PASS",
        required_fragments=(
            "model_paths=paper_legacy,current_corrected",
            "fixed_points_per_path=756",
            "gpu_runs=0",
        ),
    )
    marker_count += 1

    for version in _MODEL_VERSIONS:
        _read_text_marker(
            stage_root / "fig23" / version / "PASS",
            scope_root=result_root,
            first_line="PASS",
            required_fragments=(
                f"model_version={version}",
                "curve_points=84",
                "unique_fixed_configurations=75",
                "paper_headline_claims=PASS",
                "paper_nested_search_space_trend=PASS",
            ),
        )
        marker_count += 1

    _read_text_marker(
        stage_root / "table4" / "PASS",
        scope_root=result_root,
        first_line_contains="PASS",
        required_fragments=(
            "model_version=both",
            "fixed_configs=14",
            "evaluated_points=28",
        ),
    )
    marker_count += 1

    _read_text_marker(
        result_root / "figures" / "PASS",
        scope_root=result_root,
        first_line="PASS",
        required_fragments=(
            "plots=16",
            "png_files=16",
            "pdf_files=16",
            "model_version=paper_legacy",
        ),
    )
    marker_count += 1

    return {
        "stages": len(contract_stage_names),
        "csv_files": csv_count,
        "marker_files": marker_count,
        "plot_files": plot_count,
        "stage_names": list(contract_stage_names),
    }


def verify_workflow_manifest(result_root: Path) -> dict[str, object]:
    """Validate the auditable stage ledger written by the top-level runner."""

    result_root = result_root.resolve()
    path = result_root / "run_manifest.csv"
    if not path.is_file():
        raise AssertionError(f"missing workflow manifest: {path}")
    frame = pd.read_csv(path)
    required = {
        "stage",
        "status",
        "return_code",
        "wall_time_s",
        "gpu_runs",
        "command",
        "note",
        "log",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"{path}: missing columns: {', '.join(missing)}")
    if frame.empty:
        raise AssertionError(f"{path}: no completed stages")
    if frame.stage.duplicated().any():
        duplicates = frame.loc[frame.stage.duplicated(), "stage"].tolist()
        raise AssertionError(f"{path}: duplicate stages: {duplicates}")
    failed = frame[(frame.status != "PASS") | (frame.return_code != 0)]
    if not failed.empty:
        raise AssertionError(
            f"{path}: failed stages: {failed.stage.astype(str).tolist()}"
        )
    gpu_runs = pd.to_numeric(frame.gpu_runs, errors="coerce")
    if not gpu_runs.map(math.isfinite).all() or not gpu_runs.eq(0).all():
        raise AssertionError(f"{path}: default CPU workflow recorded GPU execution")
    wall_times = pd.to_numeric(frame.wall_time_s, errors="coerce")
    if (
        not wall_times.map(math.isfinite).all()
        or not wall_times.ge(0).all()
    ):
        raise AssertionError(f"{path}: invalid negative or non-finite wall time")
    missing_logs: list[str] = []
    for value in frame.log:
        relative = Path(str(value))
        if relative.is_absolute():
            raise AssertionError(f"{path}: stage log must be scope-relative: {value}")
        target = (result_root / relative).resolve()
        try:
            target.relative_to(result_root)
        except ValueError as exc:
            raise AssertionError(f"{path}: stage log escapes result scope: {value}") from exc
        if not target.is_file():
            missing_logs.append(str(value))
    if missing_logs:
        raise AssertionError(f"{path}: missing stage logs: {missing_logs[:5]}")
    from .workflow import reproduce_stages

    required_stages = [stage.name for stage in reproduce_stages(1, "both")]
    actual_stages = frame.stage.astype(str).tolist()
    if actual_stages != required_stages:
        actual_set = set(actual_stages)
        required_set = set(required_stages)
        raise AssertionError(
            "workflow stage order/closure mismatch: "
            f"missing={sorted(required_set - actual_set)} "
            f"unexpected={sorted(actual_set - required_set)} "
            f"expected_order={required_stages} actual_order={actual_stages}"
        )
    return {
        "stages": len(frame),
        "wall_time_s_sum": float(frame.wall_time_s.sum()),
        "manifest_path": path,
    }


def _read_decode_result(path: Path, expected_version: str) -> pd.DataFrame:
    if not path.is_file():
        raise AssertionError(f"missing decode result: {path}")
    frame = pd.read_csv(path)
    required = {
        "candidate_id",
        "model",
        "bs",
        "paper_stps_avg",
        "paper_utps_avg",
        "model_version",
        "current_stps_avg",
        "current_utps_avg",
        "delta_from_paper_stps_pct",
        "delta_from_paper_utps_pct",
        "current_stps_rank",
        "current_utps_rank",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"{path}: missing columns: {', '.join(missing)}")
    if frame.candidate_id.duplicated().any():
        raise AssertionError(f"{path}: duplicate candidate_id values")
    versions = set(frame.model_version.astype(str))
    if versions != {expected_version}:
        raise AssertionError(
            f"{path}: expected model_version={expected_version!r}, found {sorted(versions)}"
        )
    numeric = frame[
        [
            "paper_stps_avg",
            "paper_utps_avg",
            "current_stps_avg",
            "current_utps_avg",
            "delta_from_paper_stps_pct",
            "delta_from_paper_utps_pct",
            "current_stps_rank",
            "current_utps_rank",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not numeric.apply(lambda column: column.map(math.isfinite)).all().all():
        raise AssertionError(f"{path}: non-finite or missing numeric result")
    return frame


def _check_manifest_closure(frame: pd.DataFrame, manifest: pd.DataFrame) -> None:
    expected_ids = set(manifest.candidate_id.astype(str))
    actual_ids = set(frame.candidate_id.astype(str))
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise AssertionError(
            "decode manifest mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    if len(frame) != 784:
        raise AssertionError(f"expected 784 Fig. 15 local-best points, got {len(frame)}")
    if frame.groupby(["model", "bs"]).ngroups != 28:
        raise AssertionError("expected 28 model/batch-size groups")

    canonical_columns = list(manifest.columns)
    missing_columns = sorted(set(canonical_columns) - set(frame.columns))
    if missing_columns:
        raise AssertionError(
            f"decode result is missing canonical columns: {missing_columns}"
        )
    expected = manifest[canonical_columns].sort_values("candidate_id").reset_index(
        drop=True
    )
    actual = frame[canonical_columns].sort_values("candidate_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise AssertionError(
            "decode fixed configuration or paper reference differs from the "
            "canonical DSE manifest"
        ) from exc


def _verify_decode_derived_columns(
    frame: pd.DataFrame, manifest: pd.DataFrame
) -> float:
    """Recompute deltas/ranks instead of trusting derived result columns."""

    paper = manifest[
        ["candidate_id", "paper_stps_avg", "paper_utps_avg"]
    ].rename(
        columns={
            "paper_stps_avg": "canonical_paper_stps_avg",
            "paper_utps_avg": "canonical_paper_utps_avg",
        }
    )
    comparison = frame.merge(paper, on="candidate_id", validate="one_to_one")
    expected_deltas = {
        "delta_from_paper_stps_pct": 100.0
        * (
            comparison.current_stps_avg
            / comparison.canonical_paper_stps_avg
            - 1.0
        ),
        "delta_from_paper_utps_pct": 100.0
        * (
            comparison.current_utps_avg
            / comparison.canonical_paper_utps_avg
            - 1.0
        ),
    }
    maximum_error = 0.0
    for column, expected in expected_deltas.items():
        differences = (comparison[column] - expected).abs()
        maximum_difference = float(differences.max())
        if (
            not math.isfinite(maximum_difference)
            or maximum_difference > DERIVED_FIELD_TOLERANCE
        ):
            raise AssertionError(
                f"{column} differs from independently recomputed values; "
                f"max_abs_delta={maximum_difference:.6g}"
            )
        maximum_error = max(maximum_error, float(expected.abs().max()))

    for value_column, rank_column in (
        ("current_stps_avg", "current_stps_rank"),
        ("current_utps_avg", "current_utps_rank"),
    ):
        expected_rank = frame.groupby(["model", "bs"])[value_column].rank(
            method="min", ascending=False
        )
        maximum_difference = float((frame[rank_column] - expected_rank).abs().max())
        if (
            not math.isfinite(maximum_difference)
            or maximum_difference > DERIVED_FIELD_TOLERANCE
        ):
            raise AssertionError(
                f"{rank_column} differs from independently recomputed values; "
                f"max_abs_delta={maximum_difference:.6g}"
            )
    if not math.isfinite(maximum_error):
        raise AssertionError("decode replay produced a non-finite paper-relative error")
    return maximum_error


def build_version_drift_summary(
    legacy: pd.DataFrame,
    corrected: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize fixed-config drift without claiming a corrected-model DSE."""

    paper_winner_indices = legacy.groupby(["model", "bs"])["paper_stps_avg"].idxmax()
    paper_winners = legacy.loc[paper_winner_indices].copy()
    corrected_by_id = corrected.set_index("candidate_id")

    rows: list[dict[str, object]] = []
    for paper_row in paper_winners.itertuples(index=False):
        current_row = corrected_by_id.loc[paper_row.candidate_id]
        group = corrected[
            (corrected.model == paper_row.model) & (corrected.bs == paper_row.bs)
        ]
        best_row = group.loc[group.current_stps_avg.idxmax()]
        rows.append(
            {
                "model": paper_row.model,
                "bs": int(paper_row.bs),
                "paper_candidate_id": paper_row.candidate_id,
                "arch": paper_row.arch,
                "noc": paper_row.noc,
                "tp": int(paper_row.tp),
                "ep": int(paper_row.ep),
                "dp": int(paper_row.dp),
                "pp": int(paper_row.pp),
                "paper_stps": float(paper_row.paper_stps_avg),
                "paper_legacy_reproduced_stps": float(paper_row.current_stps_avg),
                "paper_legacy_error_pct": float(paper_row.delta_from_paper_stps_pct),
                "current_corrected_same_config_stps": float(current_row.current_stps_avg),
                "current_corrected_same_config_delta_pct": float(
                    current_row.delta_from_paper_stps_pct
                ),
                "paper_winner_rank_within_legacy_manifest_under_current": int(
                    current_row.current_stps_rank
                ),
                "best_current_within_legacy_manifest_candidate_id": best_row.candidate_id,
                "best_current_within_legacy_manifest_stps": float(best_row.current_stps_avg),
                "best_manifest_gain_over_paper_config_pct": 100.0
                * (float(best_row.current_stps_avg) / float(current_row.current_stps_avg) - 1.0),
                "scope_note": (
                    "corrected values re-evaluate the legacy local-best manifest only; "
                    "the corrected design space was not re-searched"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "bs"]).reset_index(drop=True)


def verify_fig15(fig15_root: Path) -> dict[str, object]:
    """Verify a self-contained Figure 15 stage directory."""

    legacy_path = fig15_root / "paper_legacy" / "fig15_pareto_candidates.csv"
    corrected_path = fig15_root / "current_corrected" / "fig15_pareto_candidates.csv"
    legacy = _read_decode_result(legacy_path, "paper_legacy")
    corrected = _read_decode_result(corrected_path, "current_corrected")
    manifest = select_fig15_local_best(load_candidate_table(DECODE_POPULATION))
    _check_manifest_closure(legacy, manifest)
    _check_manifest_closure(corrected, manifest)
    if set(legacy.candidate_id) != set(corrected.candidate_id):
        raise AssertionError("paper_legacy and current_corrected candidate sets differ")

    legacy_max_error = _verify_decode_derived_columns(legacy, manifest)
    _verify_decode_derived_columns(corrected, manifest)
    if legacy_max_error > LEGACY_TOLERANCE_PCT:
        raise AssertionError(
            f"paper_legacy maximum error {legacy_max_error:.6g}% exceeds "
            f"{LEGACY_TOLERANCE_PCT:.6g}%"
        )

    expected_current = manifest[
        [
            "candidate_id",
            "expected_current_utps_avg",
            "expected_current_stps_avg",
        ]
    ]
    if expected_current.isna().any().any():
        raise AssertionError("current_corrected values are missing for Fig. 15 winners")
    comparison = corrected[
        ["candidate_id", "current_utps_avg", "current_stps_avg"]
    ].merge(expected_current, on="candidate_id", validate="one_to_one")
    if len(comparison) != len(corrected):
        raise AssertionError("current_corrected expected-value table is incomplete")
    relative_errors = pd.concat(
        [
            (
                comparison.current_utps_avg
                / comparison.expected_current_utps_avg
                - 1.0
            ).abs(),
            (
                comparison.current_stps_avg
                / comparison.expected_current_stps_avg
                - 1.0
            ).abs(),
        ],
        ignore_index=True,
    )
    corrected_max_relative_error = float(relative_errors.max())
    if corrected_max_relative_error > CORRECTED_RELATIVE_TOLERANCE:
        raise AssertionError(
            "current_corrected maximum relative error "
            f"{corrected_max_relative_error:.6g} exceeds "
            f"{CORRECTED_RELATIVE_TOLERANCE:.6g}"
        )

    summary = build_version_drift_summary(legacy, corrected)
    summary_path = fig15_root / "version_drift_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.15g")
    winners_retained = int(
        (summary.paper_winner_rank_within_legacy_manifest_under_current == 1).sum()
    )
    return {
        "legacy_rows": len(legacy),
        "legacy_max_error_pct": legacy_max_error,
        "corrected_rows": len(corrected),
        "corrected_max_relative_error": corrected_max_relative_error,
        "paper_winners_retained_within_legacy_manifest": winners_retained,
        "paper_winner_groups": len(summary),
        "summary_path": summary_path,
    }


def verify_reproduce(result_root: Path) -> dict[str, object]:
    """Verify Figure 15 within a complete workflow result directory."""

    return verify_fig15(result_root / "stages" / "fig15")


def verify_all(result_root: Path | None = None) -> int:
    root = (result_root or (RESULTS_DIR / "reproduce")).resolve()
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_release_checksums.py"), "--verify"],
        cwd=ROOT,
        check=True,
    )
    workflow = verify_workflow_manifest(root)
    output_contracts = verify_stage_outputs(root)
    report = verify_reproduce(root)
    machine_report = {
        "status": "PASS",
        "workflow": {
            "stages": workflow["stages"],
            "wall_time_s_sum": workflow["wall_time_s_sum"],
            "manifest": Path(workflow["manifest_path"])
            .resolve()
            .relative_to(root)
            .as_posix(),
            "output_contract_stages": output_contracts["stages"],
            "output_contract_csv_files": output_contracts["csv_files"],
            "output_contract_marker_files": output_contracts["marker_files"],
            "output_contract_plot_files": output_contracts["plot_files"],
        },
        "fig15": {
            key: value.resolve().relative_to(root).as_posix()
            if isinstance(value, Path)
            else value
            for key, value in report.items()
        },
    }
    report_path = root / "verification_summary.json"
    report_path.write_text(
        json.dumps(machine_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Default workflow manifest: PASS")
    print(
        f"  stages={workflow['stages']} "
        f"summed_stage_wall_time_s={workflow['wall_time_s_sum']:.3f}"
    )
    print(
        "Stage output contracts: PASS "
        f"({output_contracts['csv_files']} CSVs, "
        f"{output_contracts['marker_files']} markers, "
        f"{output_contracts['plot_files']} plot files)"
    )
    print("Fig. 15 paper_legacy: PASS")
    print(
        f"  rows={report['legacy_rows']} "
        f"max_abs_error_pct={report['legacy_max_error_pct']:.6g}"
    )
    print("Fig. 15 current_corrected diagnostic: PASS")
    print(
        f"  rows={report['corrected_rows']} "
        f"max_relative_error={report['corrected_max_relative_error']:.6g}"
    )
    print(
        "  paper winner remains best within the legacy local-best manifest for "
        f"{report['paper_winners_retained_within_legacy_manifest']}/"
        f"{report['paper_winner_groups']} model/BS groups"
    )
    print(f"  summary={report['summary_path']}")
    print(f"Overall verification: PASS ({report_path})")
    return 0
