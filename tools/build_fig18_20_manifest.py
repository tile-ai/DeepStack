#!/usr/bin/env python3
"""Build the compact data closure for paper Figs. 18--20.

This is a maintainer utility.  It reads the two archived full DRAM-layer DSE
CSVs, records why each retained row is needed, and writes a compact manifest.
The full (roughly 250 MiB each) search outputs are deliberately not copied into
the artifact.

Selection rules mirror the paper plotting code:

* Fig. 18: best scaled-throughput row at every fully connected layer count for
  BS=4 and BS=1024, for both prefill and decode.
* Fig. 19: global throughput- and tokens/J-optimal rows for every archived
  batch size on the even-(m,n) plotting grid.
* Fig. 20: for decode BS=4 and BS=1024, retain the hottest, raw-throughput-best,
  thermally-safe raw-throughput-best, and one deterministic hash-sampled row
  at every fully connected layer count.

The utility also writes small paper-facing projections of the full DSE: the
Fig. 18 throughput curves, the Fig. 19 even-grid heatmap, and the Fig. 20
decode thermal terrain.  These archived projections are plot inputs, not a
claim that the reviewer reran the exhaustive search.

Only display-precision values are written to those plotting projections.  The
fixed manifest contains model inputs and selection roles but no ``paper_*``
outputs; exact model regressions are maintained separately as non-reversible
SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

POWER_CAP_W = 100.0
NUM_DEVICES = 256
THERMAL_LIMIT_C = 85.0
ARCHIVE_SAMPLE_SALT = "deepstack-fig20-terrain-audit-v1"


def _prepare(path: Path, phase: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if phase == "decode":
        frame["paper_raw_utps"] = frame["raw_utps_avg"]
        frame["paper_raw_stps"] = frame["raw_stps_avg"]
        frame["paper_scaled_utps"] = frame["scaled_utps_avg"]
        frame["paper_scaled_stps"] = frame["scaled_stps_avg"]
    else:
        frame["paper_raw_utps"] = frame["raw_utps"]
        frame["paper_raw_stps"] = frame["raw_stps"]
        frame["paper_scaled_utps"] = frame["scaled_utps"]
        frame["paper_scaled_stps"] = frame["scaled_stps"]

    frame["phase"] = phase
    frame["paper_effective_power_W"] = frame["total_power_per_device_W"].clip(
        upper=POWER_CAP_W
    )
    frame["paper_effective_stps"] = frame["paper_raw_stps"].where(
        frame["total_power_per_device_W"] <= POWER_CAP_W,
        frame["paper_scaled_stps"],
    )
    frame["paper_tokens_per_joule"] = frame["paper_effective_stps"] / (
        frame["paper_effective_power_W"] * NUM_DEVICES
    )
    frame["paper_temperature_C"] = 35.0 + (
        frame["thermal_resistance_CpW"] * frame["total_power_per_device_W"]
    )
    return frame


def _record(
    selected: dict[tuple[str, int], set[str]],
    frame: pd.DataFrame,
    indices: object,
    role: str,
) -> None:
    for index in list(indices):
        selected.setdefault((str(frame.loc[index, "phase"]), int(index)), set()).add(
            role
        )


def _select(decode: pd.DataFrame, prefill: pd.DataFrame) -> pd.DataFrame:
    selected: dict[tuple[str, int], set[str]] = {}

    for frame in (decode, prefill):
        # Fig. 18 plots fully connected stacks through m=12.  The compact AE
        # keeps the two endpoint workloads used by every textual conclusion.
        curve = frame[
            (frame["dram_total_layers"] == frame["dram_active_layers"])
            & (frame["dram_total_layers"] <= 12)
            & frame["bs"].isin([4, 1024])
        ]
        indices = curve.groupby(["bs", "dram_total_layers"])[
            "paper_scaled_stps"
        ].idxmax()
        _record(selected, frame, indices, "fig18_curve_best")

        # Fig. 19 uses the even (m,n) grid and applies a 100 W/device cap.
        grid = frame[
            (frame["dram_total_layers"] % 2 == 0)
            & (frame["dram_active_layers"] % 2 == 0)
            & (frame["dram_total_layers"] <= 14)
        ]
        indices = grid.groupby("bs")["paper_effective_stps"].idxmax()
        _record(selected, frame, indices, "fig19_throughput_winner")
        indices = grid.groupby("bs")["paper_tokens_per_joule"].idxmax()
        _record(selected, frame, indices, "fig19_efficiency_winner")

    # Fig. 20 is the raw decode thermal terrain for BS=4 and 1024.  Retaining
    # extrema per layer is enough to audit the hot/slow and safe/fast trends.
    terrain = decode[
        (decode["dram_total_layers"] == decode["dram_active_layers"])
        & (decode["dram_total_layers"] <= 12)
        & decode["bs"].isin([4, 1024])
    ]
    groups = ["bs", "dram_total_layers"]
    _record(
        selected,
        decode,
        terrain.groupby(groups)["paper_temperature_C"].idxmax(),
        "fig20_hottest_per_layer",
    )
    _record(
        selected,
        decode,
        terrain.groupby(groups)["paper_raw_stps"].idxmax(),
        "fig20_raw_throughput_best_per_layer",
    )
    safe = terrain[terrain["paper_temperature_C"] <= THERMAL_LIMIT_C]
    _record(
        selected,
        decode,
        safe.groupby(groups)["paper_raw_stps"].idxmax(),
        "fig20_safe_throughput_best_per_layer",
    )

    # One stable, non-extremal sample per displayed (BS, layer) stratum gives
    # reviewers a genuinely random-looking audit of the archived terrain while
    # remaining deterministic across machines and pandas versions.
    already_selected = {
        index for phase, index in selected if phase == "decode"
    }
    sample_pool = terrain.loc[~terrain.index.isin(already_selected)].copy()
    sample_pool["_sample_hash"] = sample_pool["stats_run_id"].map(
        lambda value: hashlib.sha256(
            f"{ARCHIVE_SAMPLE_SALT}|{value}".encode("utf-8")
        ).hexdigest()
    )
    sample_indices = (
        sample_pool.sort_values(["bs", "dram_total_layers", "_sample_hash"])
        .groupby(groups, sort=True)
        .head(1)
        .index
    )
    _record(selected, decode, sample_indices, "fig20_archive_sample")
    terrain_reference_by_index = {
        int(index): f"terrain-{ordinal:06d}"
        for ordinal, index in enumerate(terrain.index, start=1)
    }

    rows = []
    by_phase = {"decode": decode, "prefill": prefill}
    for (phase, index), roles in selected.items():
        row = by_phase[phase].loc[index].copy()
        row["selection_roles"] = ";".join(sorted(roles))
        row["terrain_reference_id"] = (
            terrain_reference_by_index[int(index)]
            if "fig20_archive_sample" in roles
            else ""
        )
        rows.append(row)
    result = pd.DataFrame(rows)

    config_fields = [
        "phase",
        "dram_total_layers",
        "dram_active_layers",
        "smem_capacity_KiB",
        "l1_throughput_Bpc",
        "bs",
        "minibatch",
        "seq",
        "tp",
        "ep",
        "ep1",
        "ep2",
        "sp",
        "cp",
        "dp",
        "fsdp",
        "pp",
        "tp_transform_moe",
    ]

    def candidate_id(row: pd.Series) -> str:
        payload = "|".join(str(row[field]) for field in config_fields)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"dram-{row['phase']}-{digest}"

    result.insert(0, "candidate_id", result.apply(candidate_id, axis=1))
    if result["candidate_id"].duplicated().any():
        raise RuntimeError("candidate ID collision")

    columns = [
        "candidate_id",
        "phase",
        "selection_roles",
        "terrain_reference_id",
        "dram_total_layers",
        "dram_active_layers",
        "sm_count",
        "smem_capacity_KiB",
        "l1_throughput_Bpc",
        "arch",
        "noc",
        "model",
        "bs",
        "minibatch",
        "seq",
        "tp",
        "ep",
        "ep1",
        "ep2",
        "sp",
        "cp",
        "dp",
        "fsdp",
        "pp",
        "tp_transform_moe",
    ]
    if "kv_len_1" in result:
        for index in range(1, 5):
            columns.append(f"kv_len_{index}")
    else:
        # Decode rows have these columns while prefill rows do not; pandas
        # unioning supplies NaN for prefill.  This branch is mainly defensive.
        for index in range(1, 5):
            result[f"kv_len_{index}"] = pd.NA
            columns.append(f"kv_len_{index}")

    result = result[columns].sort_values(
        ["phase", "bs", "dram_total_layers", "dram_active_layers", "candidate_id"]
    )
    return result.reset_index(drop=True)


def _fig18_curves(decode: pd.DataFrame, prefill: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for frame in (decode, prefill):
        filtered = frame[
            (frame["dram_total_layers"] == frame["dram_active_layers"])
            & (frame["dram_total_layers"] <= 12)
            & (frame["bs"] >= 4)
            & (frame["bs"] <= 1024)
        ]
        indices = filtered.groupby(["dram_total_layers", "bs"])[
            "paper_scaled_stps"
        ].idxmax()
        selected = filtered.loc[
            indices,
            [
                "phase",
                "bs",
                "dram_total_layers",
                "paper_scaled_stps",
            ],
        ].copy()
        selected.rename(
            columns={
                "dram_total_layers": "m",
                "paper_scaled_stps": "best_scaled_stps",
            },
            inplace=True,
        )
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True).sort_values(["phase", "bs", "m"])
    expected_bs = {4, 16, 64, 256, 1024}
    if set(result["bs"]) != expected_bs or len(result) != 110:
        raise RuntimeError(
            f"unexpected Fig. 18 curve closure: rows={len(result)}, "
            f"bs={sorted(set(result['bs']))}"
        )
    return result.reset_index(drop=True)


def _fig19_grid(decode: pd.DataFrame, prefill: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for frame in (decode, prefill):
        grid = frame[
            (frame["dram_total_layers"] % 2 == 0)
            & (frame["dram_active_layers"] % 2 == 0)
            & (frame["dram_total_layers"] <= 14)
            & frame["bs"].isin([4, 1024])
        ]
        summary = (
            grid.groupby(
                ["phase", "bs", "dram_total_layers", "dram_active_layers"],
                as_index=False,
            )
            .agg(
                best_effective_stps=("paper_effective_stps", "max"),
                best_tokens_per_joule=("paper_tokens_per_joule", "max"),
            )
            .rename(
                columns={
                    "dram_total_layers": "m",
                    "dram_active_layers": "n",
                }
            )
        )
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["phase", "bs", "m", "n"]
    )
    if len(result) != 108:
        raise RuntimeError(f"expected 108 Fig. 19 grid cells, found {len(result)}")
    return result.reset_index(drop=True)


def _fig20_terrain(decode: pd.DataFrame) -> pd.DataFrame:
    terrain = decode[
        (decode["dram_total_layers"] == decode["dram_active_layers"])
        & (decode["dram_total_layers"] <= 12)
        & decode["bs"].isin([4, 1024])
    ][
        [
            "dram_total_layers",
            "dram_active_layers",
            "bs",
            "paper_temperature_C",
            "paper_raw_stps",
        ]
    ].copy()
    terrain.rename(
        columns={
            "paper_temperature_C": "temperature_C",
            "paper_raw_stps": "raw_stps",
        },
        inplace=True,
    )
    terrain.insert(
        0,
        "terrain_reference_id",
        [f"terrain-{index:06d}" for index in range(1, len(terrain) + 1)],
    )
    # Preserve original row order: duplicate (BS, layer, STPS) coordinates can
    # carry different temperatures, so sorting can change Delaunay tie breaks.
    if len(terrain) != 19233:
        raise RuntimeError(
            f"expected 19,233 Fig. 20 terrain points, found {len(terrain)}"
        )
    terrain["temperature_C"] = terrain["temperature_C"].round(3)
    return terrain.reset_index(drop=True)


def _write_archives(
    decode: pd.DataFrame, prefill: pd.DataFrame, archive_dir: Path
) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fig18_throughput_curves.csv": _fig18_curves(decode, prefill),
        "fig19_metric_grid.csv": _fig19_grid(decode, prefill),
        "fig20_decode_thermal_terrain.csv": _fig20_terrain(decode),
    }
    for name, frame in outputs.items():
        path = archive_dir / name
        # These files drive figures, not numerical regression.  Six
        # significant digits (and millidegree precision for Fig. 20) are well
        # below plot resolution while avoiding a second high-precision copy of
        # model intermediates.
        frame.to_csv(path, index=False, float_format="%.6g")
        print(f"wrote {len(frame):,} archived plot rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="paper-facing compact DSE projections (defaults to OUTPUT/../archive)",
    )
    args = parser.parse_args()

    decode = _prepare(args.decode, "decode")
    prefill = _prepare(args.prefill, "prefill")
    manifest = _select(decode, prefill)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False, float_format="%.15g")
    _write_archives(
        decode,
        prefill,
        args.archive_dir or (args.output.parent / "archive"),
    )

    print(f"wrote {len(manifest)} unique fixed configurations to {args.output}")
    print("regression digests are refreshed by ae.fig18_20 after model evaluation")
    print(manifest["phase"].value_counts().sort_index().to_string())
    roles = manifest["selection_roles"].str.split(";").explode().value_counts()
    print(roles.sort_index().to_string())


if __name__ == "__main__":
    main()
