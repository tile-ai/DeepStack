"""Re-evaluate the 80 fixed winners used by decode Figure 16."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .paths import CONFIG_DIR, RESULTS_DIR
from .selected_decode import run_candidates


MANIFEST = CONFIG_DIR / "fig16_decode_winners_march2026.csv"
MODEL_VERSIONS = ("paper_legacy", "current_corrected")
EXPECTED_ROWS = 80
# DeepStack and H200* rows reproduce to floating precision. H100* traverses
# TileSight's permutation-based cache estimator; the AE pins seed 0 and allows
# a still-negligible 0.001% difference from the unseeded paper run.
LEGACY_TOLERANCE_PCT = 1e-3


def _ratios(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.pivot_table(
        index=["model", "bs"],
        columns="platform",
        values="current_stps_avg",
        aggfunc="first",
    ).reset_index()
    for baseline in ("H100_SCALED", "H200_SCALED"):
        values[f"deepstack_vs_{baseline.lower()}_x"] = (
            values.DeepStack / values[baseline]
        )
    return values.sort_values(["model", "bs"]).reset_index(drop=True)


def _write_claims(
    ratios: pd.DataFrame,
    *,
    model_version: str,
    max_error_pct: float,
    output: Path,
) -> None:
    bs1024 = ratios[ratios.bs == 1024].deepstack_vs_h200_scaled_x
    bs1 = ratios[ratios.bs == 1].deepstack_vs_h200_scaled_x.dropna()
    rows = [
        {
            "metric": "fixed_config_rows",
            "value": len(ratios) * 3 - 4,
            "expected": EXPECTED_ROWS,
            "tolerance": 0,
            "status": "PASS" if len(ratios) * 3 - 4 == EXPECTED_ROWS else "FAIL",
            "scope": "80 plotted platform/model/BS winners",
        },
        {
            "metric": "maximum_fixed_config_error_vs_paper",
            "value": max_error_pct,
            "expected": 0 if model_version == "paper_legacy" else "",
            "tolerance": LEGACY_TOLERANCE_PCT if model_version == "paper_legacy" else "",
            "status": (
                "PASS"
                if model_version == "paper_legacy" and max_error_pct <= LEGACY_TOLERANCE_PCT
                else "INFO"
            ),
            "scope": "paper values for legacy; diagnostic only for corrected",
        },
        {
            "metric": "bs1024_min_deepstack_vs_h200",
            "value": float(bs1024.min()),
            "expected": 1.30 if model_version == "paper_legacy" else "",
            "tolerance": 0.01 if model_version == "paper_legacy" else "",
            "status": (
                "INFO"
                if model_version != "paper_legacy"
                else ("PASS" if abs(bs1024.min() - 1.30) <= 0.01 else "FAIL")
            ),
            "scope": "four models at BS=1024",
        },
        {
            "metric": "bs1024_max_deepstack_vs_h200",
            "value": float(bs1024.max()),
            "expected": 1.48 if model_version == "paper_legacy" else "",
            "tolerance": 0.01 if model_version == "paper_legacy" else "",
            "status": (
                "INFO"
                if model_version != "paper_legacy"
                else ("PASS" if abs(bs1024.max() - 1.48) <= 0.01 else "FAIL")
            ),
            "scope": "four models at BS=1024",
        },
        {
            "metric": "bs1_max_deepstack_vs_h200",
            "value": float(bs1.max()),
            "expected": 2.79 if model_version == "paper_legacy" else "",
            "tolerance": 0.01 if model_version == "paper_legacy" else "",
            "status": (
                "INFO"
                if model_version != "paper_legacy"
                else ("PASS" if abs(bs1.max() - 2.79) <= 0.01 else "FAIL")
            ),
            "scope": "two BS=1 baseline points plotted by the paper script",
        },
    ]
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.15g")
    failures = [row for row in rows if row["status"] == "FAIL"]
    if failures:
        raise AssertionError(f"Fig. 16 claims failed: {failures}")


def run(
    *,
    model_version: str,
    workers: int,
    output_dir: Path = RESULTS_DIR / "reproduce" / "stages" / "fig16",
) -> dict[str, object]:
    version_output_dir = output_dir / model_version
    version_output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = version_output_dir / "fixed_winners.csv"
    started = time.perf_counter()
    frame = run_candidates(
        MANIFEST,
        raw_path,
        workers=workers,
        model_version=model_version,
    )
    if len(frame) != EXPECTED_ROWS or frame.candidate_id.nunique() != EXPECTED_ROWS:
        raise AssertionError(
            f"expected {EXPECTED_ROWS} unique Fig. 16 rows, got "
            f"{len(frame)}/{frame.candidate_id.nunique()}"
        )
    max_error_pct = float(
        frame[
            ["delta_from_paper_stps_pct", "delta_from_paper_utps_pct"]
        ].abs().to_numpy().max()
    )
    if model_version == "paper_legacy" and max_error_pct > LEGACY_TOLERANCE_PCT:
        raise AssertionError(
            f"paper_legacy Fig. 16 error {max_error_pct:.6g}% exceeds tolerance"
        )
    ratios = _ratios(frame)
    ratios_path = version_output_dir / "ratios.csv"
    ratios.to_csv(ratios_path, index=False, float_format="%.15g")
    claims_path = version_output_dir / "summary.csv"
    _write_claims(
        ratios,
        model_version=model_version,
        max_error_pct=max_error_pct,
        output=claims_path,
    )
    elapsed = time.perf_counter() - started
    print(
        f"Fig. 16 {model_version}: PASS; rows={len(frame)}; "
        f"max_error_pct={max_error_pct:.6g}; wall_time_s={elapsed:.3f}"
    )
    return {
        "rows": len(frame),
        "max_error_pct": max_error_pct,
        "wall_time_s": elapsed,
        "raw": raw_path,
        "ratios": ratios_path,
        "summary": claims_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version",
        choices=(*MODEL_VERSIONS, "both"),
        default="both",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "reproduce" / "stages" / "fig16",
        help="directory containing one generated subdirectory per model version",
    )
    args = parser.parse_args(argv)
    versions = MODEL_VERSIONS if args.model_version == "both" else (args.model_version,)
    for version in versions:
        run(
            model_version=version,
            workers=max(1, args.workers),
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
