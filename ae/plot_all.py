"""Render every paper figure from one self-contained workflow result scope."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
from typing import Iterable, Sequence

from .paths import RESULTS_DIR, ROOT, display_path


MODEL_VERSIONS = ("paper_legacy", "current_corrected")
EXPECTED_STEMS = (
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
)
MANIFEST_FIELDS = (
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
)


def _resolve_path(path: Path) -> Path:
    """Resolve relative paths at the artifact root while allowing copied scopes."""

    return (path if path.is_absolute() else ROOT / path).resolve()


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _source_files(record: dict[str, object]) -> str:
    value = record.get("source_files", record.get("source", ""))
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


def _note(record: dict[str, object]) -> str:
    if "note" in record:
        return str(record["note"])
    figure = str(record["figure"])
    return f"{figure} redrawn from repository-local reproduced/reference CSVs"


def render_all_figures(
    *,
    result_root: Path = RESULTS_DIR / "reproduce",
    model_version: str = "paper_legacy",
) -> list[dict[str, object]]:
    """Render 16 figure/table groups and write a machine-checkable manifest."""

    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"unsupported model version: {model_version}")
    result_root = _resolve_path(result_root)
    output_dir = (result_root / "figures").resolve()
    try:
        output_dir.relative_to(result_root)
    except ValueError as exc:
        raise ValueError(
            f"figure output must stay inside {result_root}: {output_dir}"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import Matplotlib-backed renderers only after selecting a scope-local
    # cache.  This keeps a custom --result-dir from writing into another
    # workflow tree or depending on a reviewer's home-directory cache.
    matplotlib_cache = result_root / "logs" / ".matplotlib-cache"
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    from .plot_accuracy import render_accuracy_figures
    from .plot_codesign import render_codesign_figures

    raw_records = render_accuracy_figures(
        output_dir,
        model_version,
        result_root=result_root,
    )
    raw_records.extend(
        render_codesign_figures(
            output_dir,
            result_root=result_root,
            model_version=model_version,
        )
    )
    stems = tuple(str(record["stem"]) for record in raw_records)
    if len(stems) != len(set(stems)):
        raise AssertionError("plot renderers returned duplicate stems")
    if set(stems) != set(EXPECTED_STEMS):
        missing = sorted(set(EXPECTED_STEMS).difference(stems))
        extra = sorted(set(stems).difference(EXPECTED_STEMS))
        raise AssertionError(f"plot closure mismatch: missing={missing}, extra={extra}")

    by_stem = {str(record["stem"]): record for record in raw_records}
    manifest_rows: list[dict[str, object]] = []
    for stem in EXPECTED_STEMS:
        record = by_stem[stem]
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        if not png.is_file() or not pdf.is_file():
            raise FileNotFoundError(f"renderer did not create both outputs for {stem}")
        manifest_rows.append(
            {
                "figure": str(record["figure"]),
                "stem": stem,
                "model_version": model_version,
                "source_files": _source_files(record),
                "png": png.name,
                "pdf": pdf.name,
                "png_bytes": png.stat().st_size,
                "pdf_bytes": pdf.stat().st_size,
                "status": "PASS",
                "note": _note(record),
            }
        )

    _write_csv(output_dir / "plot_manifest.csv", manifest_rows)
    marker = output_dir / "PASS"
    marker.write_text(
        "PASS\n"
        f"plots={len(manifest_rows)}\n"
        f"png_files={len(manifest_rows)}\n"
        f"pdf_files={len(manifest_rows)}\n"
        f"model_version={model_version}\n",
        encoding="utf-8",
    )
    shutil.rmtree(matplotlib_cache, ignore_errors=True)
    return manifest_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULTS_DIR / "reproduce",
        help="workflow result directory (default: results/reproduce)",
    )
    parser.add_argument(
        "--model-version",
        choices=MODEL_VERSIONS,
        default="paper_legacy",
        help="paper_legacy redraws the accepted-paper path",
    )
    args = parser.parse_args(argv)
    rows = render_all_figures(
        result_root=args.result_root,
        model_version=args.model_version,
    )
    output_dir = _resolve_path(args.result_root) / "figures"
    print(f"plots: PASS ({len(rows)} PNG + {len(rows)} PDF)")
    print(f"manifest: {display_path(output_dir / 'plot_manifest.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
