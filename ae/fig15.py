"""Re-evaluate deterministic fixed configurations from the Figure 15 DSE.

The artifact ships the complete recorded stacked-GPU co-design population.  The
``reproduce`` scope deterministically derives its 784 paper local winners;
``quick`` hash-samples one non-winner for every model/target-BS stratum.  Both
model paths can be run without mixing their outputs.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

from .paths import DATA_DIR, RESULTS_DIR, ROOT, display_path
from .selected_decode import run_candidates


MODEL_VERSIONS = ("paper_legacy", "current_corrected")
POPULATION = DATA_DIR / "fig15" / "decode_dse_population_march2026.csv"
LEGACY_TOLERANCE_PCT = 1e-9


def run(
    *,
    scope: str,
    model_version: str,
    workers: int,
    output_root: Path | None = None,
) -> int:
    if scope not in {"quick", "reproduce"}:
        raise ValueError(f"unsupported scope: {scope}")
    versions = MODEL_VERSIONS if model_version == "both" else (model_version,)
    if not set(versions).issubset(MODEL_VERSIONS):
        raise ValueError(f"unsupported model version: {model_version}")

    root = output_root or (RESULTS_DIR / scope / "stages" / "fig15")
    for version in versions:
        filename = (
            "fig15_dse_sample.csv"
            if scope == "quick"
            else "fig15_pareto_candidates.csv"
        )
        output = root / version / filename
        started = time.perf_counter()
        result = run_candidates(
            POPULATION,
            output,
            workers=max(1, workers),
            model_version=version,
            fig15_scope=scope,
        )
        elapsed = time.perf_counter() - started
        max_error_pct = float(
            result[
                ["delta_from_paper_stps_pct", "delta_from_paper_utps_pct"]
            ].abs().to_numpy().max()
        )
        if not math.isfinite(max_error_pct):
            raise AssertionError(
                f"Fig. 15 {version} produced a non-finite paper-relative delta"
            )
        if version == "paper_legacy" and max_error_pct > LEGACY_TOLERANCE_PCT:
            raise AssertionError(
                f"Fig. 15 paper replay error {max_error_pct:.6g}% exceeds "
                f"{LEGACY_TOLERANCE_PCT:.6g}%"
            )
        comparison = (
            "paper replay max error"
            if version == "paper_legacy"
            else "maximum paper-relative drift"
        )
        print(
            f"[PASS] Fig. 15 {version}: {len(result)} fixed configurations, "
            f"GPU runs=0, {comparison}={max_error_pct:.6g}%, "
            f"wall={elapsed:.3f}s"
        )
        print(f"[ok] wrote {display_path(output)}")

    if scope == "reproduce" and set(versions) == set(MODEL_VERSIONS):
        from .verification import verify_fig15

        report = verify_fig15(root)
        print(
            "[PASS] Fig. 15 dual-path verification: "
            f"legacy max error={report['legacy_max_error_pct']:.6g}%, "
            "corrected regression max relative error="
            f"{report['corrected_max_relative_error']:.6g}"
        )
    return 0


def _artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("quick", "reproduce"), default="reproduce")
    parser.add_argument(
        "--model-version",
        choices=(*MODEL_VERSIONS, "both"),
        default="both",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--output-root",
        help="override the scope result root (relative paths use the artifact root)",
    )
    args = parser.parse_args()
    output_root = _artifact_path(args.output_root) if args.output_root else None
    return run(
        scope=args.scope,
        model_version=args.model_version,
        workers=args.workers,
        output_root=output_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
