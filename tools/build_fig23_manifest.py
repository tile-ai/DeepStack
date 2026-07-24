#!/usr/bin/env python3
"""Extract the fixed winners used by paper Figure 23.

The source CSVs are exhaustive DSE outputs and are intentionally not copied
into the artifact.  This script applies the exact three filters used by the
paper plotting script and records only one canonical winner per curve point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODELS = ("DeepSeekV3", "Qwen3_235b_a22b")
ARCH = "stacked_gpu_wgmma"
NOC = "torus_mesh_switch_1"
EXPECTED_BATCH_SIZES = {
    "decode": (1, 4, 16, 64, 128, 512, 1024),
    "prefill": (1, 4, 8, 16, 64, 128, 1024),
}
TIERS = ("astra_space", "expanded_parallel", "module_flexible")


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def _tier(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "astra_space":
        return frame[
            (frame.ep == 1)
            & (frame.sp == 1)
            & (frame.cp == 1)
            & (~_bool_series(frame.fsdp))
            & (frame.tp_transform_moe == "none")
        ]
    if name == "expanded_parallel":
        return frame[frame.tp_transform_moe == "none"]
    if name == "module_flexible":
        return frame
    raise ValueError(f"unknown tier: {name}")


def _runner_metadata(group: pd.DataFrame, metric: str) -> tuple[int, float]:
    values = group[metric].sort_values(ascending=False)
    best = float(values.iloc[0])
    tolerance = max(1.0e-12, abs(best) * 1.0e-12)
    tie_count = int((values.sub(best).abs() <= tolerance).sum())
    if len(values) == tie_count:
        return tie_count, float("nan")
    runner = float(values.iloc[tie_count])
    return tie_count, 100.0 * (1.0 - runner / best)


def _winner_rows(frame: pd.DataFrame, phase: str) -> pd.DataFrame:
    metric = "stps_avg" if phase == "decode" else "stps"
    utps_metric = "utps_avg" if phase == "decode" else "utps"
    base = frame[
        (frame.arch == ARCH)
        & (frame.noc == NOC)
        & (frame.model.isin(MODELS))
        & (frame.bs.isin(EXPECTED_BATCH_SIZES[phase]))
    ].copy()
    actual = tuple(sorted(int(value) for value in base.bs.unique()))
    if actual != EXPECTED_BATCH_SIZES[phase]:
        raise ValueError(
            f"{phase} batch-size mismatch: expected "
            f"{EXPECTED_BATCH_SIZES[phase]}, found {actual}"
        )

    rows: list[dict[str, object]] = []
    for tier_order, tier_name in enumerate(TIERS, start=1):
        candidates = _tier(base, tier_name)
        for model in MODELS:
            for bs in EXPECTED_BATCH_SIZES[phase]:
                group = candidates[
                    (candidates.model == model) & (candidates.bs == bs)
                ]
                if group.empty:
                    raise ValueError(
                        f"no {phase} candidates for {model}, BS={bs}, {tier_name}"
                    )
                # This exactly matches pandas groupby(...).idxmax() in the
                # plotting script, including its first-row tie break.
                source = group.loc[group[metric].idxmax()]
                tie_count, runner_gap = _runner_metadata(group, metric)
                model_slug = "deepseek" if model == "DeepSeekV3" else "qwen"
                row: dict[str, object] = {
                    "candidate_id": f"{phase}_{model_slug}_bs{bs}_{tier_name}",
                    "phase": phase,
                    "model": model,
                    "bs": int(bs),
                    "tier": tier_name,
                    "tier_order": tier_order,
                    "arch": str(source.arch),
                    "noc": str(source.noc),
                    "minibatch": int(source.minibatch),
                    "seq": int(source.seq),
                    "cached_kv": int(source.seq) if phase == "prefill" else "",
                    "tp": int(source.tp),
                    "ep": int(source.ep),
                    "ep1": int(source.ep1),
                    "ep2": int(source.ep2),
                    "sp": int(source.sp),
                    "cp": int(source.cp),
                    "dp": int(source.dp),
                    "fsdp": bool(_bool_series(pd.Series([source.fsdp])).iloc[0]),
                    "pp": int(source.pp),
                    "tp_transform_moe": str(source.tp_transform_moe),
                    "max_activation_gib": float(source["max_activation/GiB"]),
                    "mem_weight_gib": float(source["mem_weight/GiB"]),
                    "kv_cache_gib": float(source["kv_cache/GiB"]),
                    "paper_utps": float(source[utps_metric]),
                    "paper_stps": float(source[metric]),
                    "source_stats_run_id": str(source.stats_run_id),
                    "source_candidate_count": len(group),
                    "source_tie_count": tie_count,
                    "runner_up_gap_pct": runner_gap,
                }
                for index in range(1, 5):
                    name = f"kv_len_{index}"
                    row[name] = int(source[name]) if phase == "decode" else ""
                rows.append(row)

    result = pd.DataFrame(rows)
    if len(result) != 42 or result.candidate_id.duplicated().any():
        raise AssertionError(
            f"each Figure 23 phase must produce 42 unique points, found {len(result)}"
        )
    return result


def _paper_summary(winners: pd.DataFrame) -> pd.DataFrame:
    pivot = winners.pivot(
        index=["phase", "model", "bs"], columns="tier", values="paper_stps"
    ).reset_index()
    pivot["expanded_over_astra"] = (
        pivot.expanded_parallel / pivot.astra_space
    )
    pivot["flexible_over_expanded"] = (
        pivot.module_flexible / pivot.expanded_parallel
    )
    pivot["flexible_over_astra"] = pivot.module_flexible / pivot.astra_space
    pivot["paper_reported_flexible_over_astra"] = float("nan")
    mask_deepseek = (
        (pivot.phase == "decode")
        & (pivot.model == "DeepSeekV3")
        & (pivot.bs == 1024)
    )
    mask_qwen = (
        (pivot.phase == "decode")
        & (pivot.model == "Qwen3_235b_a22b")
        & (pivot.bs == 1024)
    )
    pivot.loc[mask_deepseek, "paper_reported_flexible_over_astra"] = 5.03
    pivot.loc[mask_qwen, "paper_reported_flexible_over_astra"] = 2.31
    return pivot


def _noc_area_supplement(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        scenario = path.stem.removeprefix("lat_bw_decode_search_")
        frame = pd.read_csv(path)
        frame.tp_transform_moe = frame.tp_transform_moe.fillna("none")
        base = frame[
            (frame.baseline_noc == NOC)
            & (frame.model == "DeepSeekV3")
            & (frame.latency_multiplier == 1.0)
            & (frame.bw_multiplier == 1.0)
        ].copy()
        for tier_name in TIERS:
            candidates = _tier(base, tier_name)
            indices = candidates.groupby(["bs", "bw_multiplier"])[
                "scaled_stps_avg"
            ].idxmax()
            for _, source in candidates.loc[indices].iterrows():
                rows.append(
                    {
                        "scenario": scenario,
                        "bs": int(source.bs),
                        "bw_multiplier": float(source.bw_multiplier),
                        "tier": tier_name,
                        "sm_count": int(source.sm_count),
                        "tp": int(source.tp),
                        "ep": int(source.ep),
                        "ep1": int(source.ep1),
                        "ep2": int(source.ep2),
                        "sp": int(source.sp),
                        "cp": int(source.cp),
                        "dp": int(source.dp),
                        "fsdp": bool(source.fsdp),
                        "pp": int(source.pp),
                        "tp_transform_moe": str(source.tp_transform_moe),
                        "paper_scaled_stps": float(source.scaled_stps_avg),
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 18:
        raise AssertionError(f"expected 18 supplemental winners, found {len(result)}")
    return result.sort_values(["scenario", "bs", "bw_multiplier", "tier"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-csv", type=Path, required=True)
    parser.add_argument("--prefill-csv", type=Path, required=True)
    parser.add_argument("--noc-area-csv", type=Path, action="append", default=[])
    parser.add_argument(
        "--current-raw",
        type=Path,
        help="optional verified current-corrected raw.csv used to freeze regression values",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    decode = _winner_rows(pd.read_csv(args.decode_csv), "decode")
    prefill = _winner_rows(pd.read_csv(args.prefill_csv), "prefill")
    winners = pd.concat([decode, prefill], ignore_index=True)
    winners.sort_values(
        ["phase", "model", "bs", "tier_order"], inplace=True
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    winners.to_csv(
        args.output_dir / "selected_winners_march2026.csv",
        index=False,
        float_format="%.15g",
    )
    _paper_summary(winners).to_csv(
        args.output_dir / "paper_reference_summary.csv",
        index=False,
        float_format="%.15g",
    )
    if args.noc_area_csv:
        if len(args.noc_area_csv) != 2:
            raise ValueError("pass both logic_die and noc_to_dram supplemental CSVs")
        _noc_area_supplement(args.noc_area_csv).to_csv(
            args.output_dir / "noc_area_parallelism_supplement.csv",
            index=False,
            float_format="%.15g",
        )
    if args.current_raw is not None:
        current = pd.read_csv(args.current_raw)
        if len(current) != 84 or set(current.candidate_id) != set(winners.candidate_id):
            raise ValueError("current-corrected raw result does not match the winner manifest")
        expected = current[
            ["candidate_id", "reproduced_utps", "reproduced_stps"]
        ].rename(
            columns={
                "reproduced_utps": "expected_utps",
                "reproduced_stps": "expected_stps",
            }
        )
        expected.sort_values("candidate_id", inplace=True)
        expected.to_csv(
            args.output_dir / "current_corrected_expected.csv",
            index=False,
            float_format="%.15g",
        )

    print(f"wrote {len(winners)} Figure 23 winners to {args.output_dir}")


if __name__ == "__main__":
    main()
