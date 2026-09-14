#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = Path("results/dse-runs/prefill")

FIELD_CHOICES = {
    "total": ("chip_energy", "per_device", "total", "w"),
    "memory": ("chip_energy", "per_device", "memory", "w"),
    "compute": ("chip_energy", "per_device", "compute", "w"),
    "static": ("chip_energy", "per_device", "static", "w"),
    "noc": ("noc", "power_w"),
}


def _nested_get(payload: dict, path: tuple[str, ...]) -> float | None:
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    return float(cur)


def _normalize_text(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


def _matches_substring_filters(
    text: str,
    includes: list[str] | None,
    excludes: list[str] | None,
    case_sensitive: bool,
) -> bool:
    normalized = _normalize_text(text, case_sensitive)
    include_terms = [_normalize_text(item, case_sensitive) for item in (includes or [])]
    exclude_terms = [_normalize_text(item, case_sensitive) for item in (excludes or [])]

    if include_terms and not all(term in normalized for term in include_terms):
        return False
    if exclude_terms and any(term in normalized for term in exclude_terms):
        return False
    return True


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "filtered"


def _read_result_rows(run_dir: Path) -> dict[str, dict[str, str]]:
    result_rows: dict[str, dict[str, str]] = {}
    for csv_path in sorted(run_dir.glob("*_result.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats_run_id = row.get("stats_run_id")
                if stats_run_id:
                    result_rows[stats_run_id] = row
    return result_rows


def collect_records(
    run_dir: Path,
    field: str,
    include_bundle_substrings: list[str] | None = None,
    exclude_bundle_substrings: list[str] | None = None,
    case_sensitive: bool = False,
) -> list[dict[str, object]]:
    result_rows = _read_result_rows(run_dir)
    field_path = FIELD_CHOICES[field]
    records: list[dict[str, object]] = []

    for json_path in sorted(run_dir.glob("stats_bundles/*/model_stats.json")):
        stats_run_id = json_path.parent.name
        if not _matches_substring_filters(
            text=stats_run_id,
            includes=include_bundle_substrings,
            excludes=exclude_bundle_substrings,
            case_sensitive=case_sensitive,
        ):
            continue

        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        power_w = _nested_get(payload, field_path)
        if power_w is None:
            continue

        chip_total_w = _nested_get(payload, ("chip_energy", "total", "w"))
        num_devices = _nested_get(payload, ("chip_energy", "num_devices"))
        noc_power_w = _nested_get(payload, ("noc", "power_w"))
        timing_s = _nested_get(payload, ("timing", "total_time_s"))

        record: dict[str, object] = {
            "stats_run_id": stats_run_id,
            "json_path": str(json_path),
            "field": field,
            "per_device_power_w": power_w,
            "chip_total_power_w": chip_total_w,
            "noc_power_w": noc_power_w,
            "num_devices": int(num_devices) if num_devices is not None else "",
            "total_time_s": timing_s,
            "model_stats_name": payload.get("name", ""),
        }

        row = result_rows.get(stats_run_id, {})
        for key in (
            "arch",
            "noc",
            "model",
            "bs",
            "minibatch",
            "seq",
            "tp",
            "ep",
            "sp",
            "cp",
            "dp",
            "fsdp",
            "pp",
            "tp_transform_moe",
            "utps_avg",
            "stps_avg",
        ):
            record[key] = row.get(key, "")

        records.append(record)

    return records


def dump_csv(records: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("No records to dump.")

    fieldnames = list(records[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_histogram(records: list[dict[str, object]], field: str, bins: int, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    values = [float(record["per_device_power_w"]) for record in records]
    if not values:
        raise ValueError("No power values found to plot.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    ax.set_title(f"Per-device {field} power distribution")
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.25)

    mean_v = sum(values) / len(values)
    values_sorted = sorted(values)
    median_v = values_sorted[len(values_sorted) // 2]
    ax.axvline(mean_v, color="tab:red", linestyle="--", linewidth=1.5, label=f"mean={mean_v:.2f} W")
    ax.axvline(
        median_v,
        color="tab:green",
        linestyle=":",
        linewidth=1.5,
        label=f"median={median_v:.2f} W",
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def print_summary(records: list[dict[str, object]], field: str) -> None:
    values = [float(record["per_device_power_w"]) for record in records]
    if not values:
        print(f"[WARN] no `{field}` per-device power values found.")
        return

    values_sorted = sorted(values)
    count = len(values_sorted)
    mean_v = sum(values_sorted) / count
    median_v = values_sorted[count // 2]
    min_v = values_sorted[0]
    max_v = values_sorted[-1]

    print(f"[INFO] field={field}")
    print(f"[INFO] samples={count}")
    print(f"[INFO] min={min_v:.4f} W")
    print(f"[INFO] median={median_v:.4f} W")
    print(f"[INFO] mean={mean_v:.4f} W")
    print(f"[INFO] max={max_v:.4f} W")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read model_stats.json under a DSE run directory, extract per-device power, "
            "and plot its frequency distribution."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"DSE run directory. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--field",
        choices=sorted(FIELD_CHOICES),
        default="total",
        help="Which power field to analyze.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Histogram bin count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save outputs. Default: <run-dir>/analysis",
    )
    parser.add_argument(
        "--include-bundle-substring",
        action="append",
        default=[],
        help=(
            "Only keep stats bundle folder names containing this substring. "
            "Can be passed multiple times, e.g. --include-bundle-substring stacked_gpu"
        ),
    )
    parser.add_argument(
        "--exclude-bundle-substring",
        action="append",
        default=[],
        help=(
            "Drop stats bundle folder names containing this substring. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--match-case",
        action="store_true",
        help="Make bundle substring matching case-sensitive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "analysis"

    records = collect_records(
        run_dir=run_dir,
        field=args.field,
        include_bundle_substrings=args.include_bundle_substring,
        exclude_bundle_substrings=args.exclude_bundle_substring,
        case_sensitive=args.match_case,
    )
    if not records:
        raise SystemExit(f"No records found under {run_dir}")

    suffix_parts = [args.field]
    if args.include_bundle_substring:
        suffix_parts.append("include-" + _slugify("-".join(args.include_bundle_substring)))
    if args.exclude_bundle_substring:
        suffix_parts.append("exclude-" + _slugify("-".join(args.exclude_bundle_substring)))
    file_suffix = "_".join(suffix_parts)

    csv_path = output_dir / f"per_device_power_{file_suffix}.csv"
    png_path = output_dir / f"per_device_power_{file_suffix}_hist.png"

    dump_csv(records, csv_path)
    plot_histogram(records, field=args.field, bins=args.bins, output_png=png_path)
    print_summary(records, field=args.field)
    if args.include_bundle_substring:
        print(f"[INFO] include_bundle_substrings={args.include_bundle_substring}")
    if args.exclude_bundle_substring:
        print(f"[INFO] exclude_bundle_substrings={args.exclude_bundle_substring}")
    print(f"[INFO] csv saved to: {csv_path}")
    print(f"[INFO] histogram saved to: {png_path}")
    print("[INFO] Note: current dump stores averaged per-device power, not one sample per physical device.")


if __name__ == "__main__":
    main()
