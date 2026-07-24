"""Reproduce Fig. 17's 3D-DRAM layer/effective-bandwidth sweep.

The paper figure shows the all-connected case (``m = n``) for sixteen
stacking depths, two shared-memory capacities, and three L1 bandwidths.
This module evaluates all 96 model configurations, retains the full model rows
only in memory, writes a publish-safe projection needed to review the figure,
and verifies the panel/global peaks against the archived paper summary bundled
with the artifact.

Run from the artifact root with::

    python -m ae.fig17

All default paths are relative to the artifact root, not the caller's current
working directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .paths import display_path


ROOT = Path(__file__).resolve().parents[1]
DEEPSTACK_SRC = ROOT / "src" / "deepstack"
TILESIGHT_SRC = ROOT / "src" / "tilesight"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "reproduce"
    / "stages"
    / "fig17"
    / "bw_analysis_connected_8x_2x3.csv"
)
DEFAULT_SUMMARY = ROOT / "results" / "reproduce" / "stages" / "fig17" / "summary.csv"
DEFAULT_REFERENCE = ROOT / "data" / "fig17" / "paper_reference_summary.sha256"

MAX_LAYERS = 16
FIGURE_AREA_SCALE = 8
EXPECTED_CONFIGS = 96
PUBLISHED_FIELDS = (
    "n",
    "smem_cap_KiB",
    "l1_tp_Bpc",
    "actual_bw_TBs",
)


def _activate_vendored_source() -> None:
    for source_root in (TILESIGHT_SRC, DEEPSTACK_SRC):
        source = str(source_root)
        if source not in sys.path:
            sys.path.insert(0, source)


def _artifact_path(value: str | Path) -> Path:
    """Resolve relative CLI paths against the artifact root."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def analyze_configuration(
    layer: int,
    smem_capacity: int,
    l1_throughput: int,
) -> dict[str, Any]:
    """Evaluate one all-connected (``m = n = layer``) configuration."""

    _activate_vendored_source()
    from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
        _make_raw_arch,
        find_max_sm_count,
        get_actual_bw,
    )

    sm_count = find_max_sm_count(layer, layer, smem_capacity, l1_throughput)

    arch = _make_raw_arch(
        layer,
        layer,
        smem_capacity,
        l1_throughput,
        sm_count=sm_count,
    )

    ddr_peak_bw = arch.ddr_peak_bandwidth
    ddr_effective_bw = arch.ddr_bandwidth

    arch, _, _ = arch.with_littles_law()
    littles_law_bw = arch.ddr_bandwidth
    actual_bw = get_actual_bw(arch)
    is_l1_bound = actual_bw < littles_law_bw

    return {
        "n": layer,
        "sm_count": sm_count,
        "smem_cap_KiB": smem_capacity // 1024,
        "l1_tp_Bpc": l1_throughput,
        "ddr_peak_bw_TBs": ddr_peak_bw / 1e12,
        "ddr_eff_bw_TBs": ddr_effective_bw / 1e12,
        "littles_law_bw_TBs": littles_law_bw / 1e12,
        "actual_bw_TBs": actual_bw / 1e12,
        "is_l1_bound": is_l1_bound,
    }


def run_sweep() -> list[dict[str, Any]]:
    """Return the deterministic 16 x 2 x 3 Fig. 17 sweep."""

    _activate_vendored_source()
    from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
        L1_THROUGHPUTS,
        SMEM_CAPACITIES,
    )

    smem_capacities = [capacity for capacity in SMEM_CAPACITIES if capacity != 256 * 1024]
    l1_throughputs = [throughput for throughput in L1_THROUGHPUTS if throughput != 512]
    if smem_capacities != [128 * 1024, 512 * 1024]:
        raise RuntimeError(f"unexpected SMEM selection: {smem_capacities}")
    if l1_throughputs != [128, 256, 1024]:
        raise RuntimeError(f"unexpected L1-throughput selection: {l1_throughputs}")

    return [
        analyze_configuration(layer, smem_capacity, l1_throughput)
        for layer in range(1, MAX_LAYERS + 1)
        for smem_capacity in smem_capacities
        for l1_throughput in l1_throughputs
    ]


def publishable_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project full internal model rows onto the reviewer-facing CSV schema."""

    published: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = [field for field in PUBLISHED_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"Fig. 17 row {index} is missing publishable fields: "
                f"{', '.join(missing)}"
            )
        published.append({field: row[field] for field in PUBLISHED_FIELDS})
    return published


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each panel and the cross-panel envelope."""

    panels: dict[tuple[int, int], list[dict[str, Any]]] = {}
    layers: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        panel = (int(row["smem_cap_KiB"]), int(row["l1_tp_Bpc"]))
        panels.setdefault(panel, []).append(row)
        layers.setdefault(int(row["n"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (smem_kib, l1_bpc), panel_rows in sorted(panels.items()):
        peak = max(panel_rows, key=lambda row: float(row["actual_bw_TBs"]))
        layer_12 = next(row for row in panel_rows if int(row["n"]) == 12)
        summary.append(
            _summary_row(
                "panel",
                smem_kib,
                l1_bpc,
                peak_layer=int(peak["n"]),
                peak_bw=float(peak["actual_bw_TBs"]),
                layer_12_bw=float(layer_12["actual_bw_TBs"]),
            )
        )

    layer_envelope = {
        layer: max(layer_rows, key=lambda row: float(row["actual_bw_TBs"]))
        for layer, layer_rows in layers.items()
    }
    global_peak = max(layer_envelope.values(), key=lambda row: float(row["actual_bw_TBs"]))
    summary.append(
        _summary_row(
            "global",
            "",
            "",
            peak_layer=int(global_peak["n"]),
            peak_bw=float(global_peak["actual_bw_TBs"]),
            layer_12_bw=float(layer_envelope[12]["actual_bw_TBs"]),
        )
    )
    return summary


def _summary_row(
    scope: str,
    smem_kib: int | str,
    l1_bpc: int | str,
    *,
    peak_layer: int,
    peak_bw: float,
    layer_12_bw: float,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "smem_cap_KiB": smem_kib,
        "l1_tp_Bpc": l1_bpc,
        "peak_layer": peak_layer,
        "peak_actual_bw_TBs": peak_bw,
        "layer12_actual_bw_TBs": layer_12_bw,
        "drop_at_12_pct": 100.0 * (1.0 - layer_12_bw / peak_bw),
        "figure_area_scale": FIGURE_AREA_SCALE,
        "figure_peak_bw_TBs": peak_bw * FIGURE_AREA_SCALE,
        "figure_layer12_bw_TBs": layer_12_bw * FIGURE_AREA_SCALE,
    }


def verify(
    rows: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    reference_path: Path = DEFAULT_REFERENCE,
) -> None:
    """Check sweep closure and the archived paper-level numerical claims."""

    if len(rows) != EXPECTED_CONFIGS:
        raise AssertionError(f"expected {EXPECTED_CONFIGS} configurations, got {len(rows)}")

    keys = {
        (int(row["n"]), int(row["smem_cap_KiB"]), int(row["l1_tp_Bpc"]))
        for row in rows
    }
    if len(keys) != EXPECTED_CONFIGS:
        raise AssertionError(f"expected {EXPECTED_CONFIGS} unique configurations, got {len(keys)}")
    if {key[0] for key in keys} != set(range(1, MAX_LAYERS + 1)):
        raise AssertionError("layer sweep does not cover every layer from 1 through 16")

    expected_digests: dict[str, str] = {}
    for line in reference_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            expected_digests[fields[1]] = fields[0]
    required_digests = {"fig17_published_sweep_v2", "fig17_summary_v1"}
    if set(expected_digests) != required_digests:
        raise AssertionError(
            "Fig. 17 reference digest file has an unexpected schema"
        )
    if _published_sweep_digest(rows) != expected_digests["fig17_published_sweep_v2"]:
        raise AssertionError(
            "computed Fig. 17 sweep does not match the archived reference digest"
        )
    if _summary_digest(summary) != expected_digests["fig17_summary_v1"]:
        raise AssertionError(
            "computed Fig. 17 summary does not match the archived reference digest"
        )

    global_row = next(row for row in summary if row["scope"] == "global")
    if not math.isclose(float(global_row["drop_at_12_pct"]), 39.6, abs_tol=0.1):
        raise AssertionError("12-layer bandwidth is not approximately 39.6% below the peak")


def _published_sweep_digest(rows: Sequence[dict[str, Any]]) -> str:
    """Hash all reviewer-facing sweep rows in a stable representation."""

    normalized = []
    for row in sorted(
        rows,
        key=lambda item: (
            int(item["n"]),
            int(item["smem_cap_KiB"]),
            int(item["l1_tp_Bpc"]),
        ),
    ):
        normalized.append(
            {
                "n": int(row["n"]),
                "smem_cap_KiB": int(row["smem_cap_KiB"]),
                "l1_tp_Bpc": int(row["l1_tp_Bpc"]),
                "actual_bw_TBs": format(float(row["actual_bw_TBs"]), ".12g"),
            }
        )
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summary_digest(summary: Sequence[dict[str, Any]]) -> str:
    """Hash a stable, plot-precision representation of the seven-row summary."""

    normalized = []
    for row in sorted(
        summary,
        key=lambda item: (
            str(item["scope"]),
            str(item["smem_cap_KiB"]),
            str(item["l1_tp_Bpc"]),
        ),
    ):
        normalized.append(
            {
                "scope": str(row["scope"]),
                "smem_cap_KiB": str(row["smem_cap_KiB"]),
                "l1_tp_Bpc": str(row["l1_tp_Bpc"]),
                "peak_layer": int(row["peak_layer"]),
                "peak_actual_bw_TBs": format(
                    float(row["peak_actual_bw_TBs"]),
                    ".12g",
                ),
                "layer12_actual_bw_TBs": format(
                    float(row["layer12_actual_bw_TBs"]),
                    ".12g",
                ),
                "drop_at_12_pct": format(
                    float(row["drop_at_12_pct"]),
                    ".12g",
                ),
            }
        )
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "directory for bw_analysis_connected_8x_2x3.csv and summary.csv; "
            "overrides --output and --summary-output when supplied"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(ROOT)),
        help="publish-safe 96-row CSV path, relative to the artifact root by default",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY.relative_to(ROOT)),
        help="seven-row peak summary CSV path, relative to the artifact root by default",
    )
    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE.relative_to(ROOT)),
        help="paper reference summary used for verification",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="write results without checking the bundled paper summary",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir is not None:
        output_dir = _artifact_path(args.output_dir)
        output_path = output_dir / DEFAULT_OUTPUT.name
        summary_path = output_dir / DEFAULT_SUMMARY.name
    else:
        output_path = _artifact_path(args.output)
        summary_path = _artifact_path(args.summary_output)
    reference_path = _artifact_path(args.reference)

    started = time.perf_counter()
    full_rows = run_sweep()
    rows = publishable_rows(full_rows)
    summary = summarize(rows)
    _write_csv(output_path, rows)
    _write_csv(summary_path, summary)
    if not args.no_verify:
        verify(rows, summary, reference_path)

    global_row = next(row for row in summary if row["scope"] == "global")
    status = "PASS" if not args.no_verify else "SKIPPED"
    print(f"Fig. 17 verification: {status}")
    print(f"Configurations: {len(rows)}")
    print(
        "Global effective-bandwidth peak: "
        f"layer {global_row['peak_layer']}, {float(global_row['peak_actual_bw_TBs']):.9f} TB/s "
        f"({float(global_row['figure_peak_bw_TBs']):.9f} TB/s at the figure's 8x scale)"
    )
    print(
        "Layer 12: "
        f"{float(global_row['layer12_actual_bw_TBs']):.9f} TB/s, "
        f"{float(global_row['drop_at_12_pct']):.4f}% below the peak"
    )
    print(f"Publish-safe CSV: {display_path(output_path)}")
    print(f"Summary CSV: {display_path(summary_path)}")
    print(f"Wall time: {time.perf_counter() - started:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
