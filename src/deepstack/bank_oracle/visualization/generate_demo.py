"""Generate the bundled self-contained bank-wave HTML dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bank_wave import build_demo_dashboard, write_dashboard_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("artifacts") / "bank_wave_demo.html",
        help="output HTML path",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="omit the synthetic ragged-GEMM wave snapshots",
    )
    args = parser.parse_args()
    data = build_demo_dashboard(include_gemm=not args.synthetic_only)
    output = write_dashboard_html(data, args.output)
    print(output)


if __name__ == "__main__":
    main()
