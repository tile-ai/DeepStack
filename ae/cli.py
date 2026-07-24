from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import os
import platform
import subprocess
import sys
from pathlib import Path

from .doctor import (
    ARCH_PROFILE_API_VERSION,
    ENERGY_PROFILE_API_VERSION,
    NOC_PROFILE_API_VERSION,
    PROPRIETARY_BINARY_LICENSE_ID,
    check_binary_license_sidecar,
    check_arch_reference_provider,
    check_noc_profile_provider,
    check_tilesight_noc_profile_provider,
)
from .paths import RESULTS_DIR, ROOT, activate_vendored_sources


REQUIRED_IMPORTS = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "torch",
    "mosaic",
    "mosaic.cost._capacity",
    "tilesight",
)
TOP_LEVEL_MODEL_VERSION_CHOICES = ("both",)


def doctor() -> int:
    activate_vendored_sources()
    failures: list[str] = []
    versions: list[tuple[str, str]] = []
    imported: dict[str, object] = {}
    if platform.python_implementation() != "CPython":
        failures.append("CPython is required by the bundled capacity extension")
    if sys.version_info[:2] != (3, 11):
        failures.append(
            "CPython 3.11 is required; use "
            "`conda run -n deepstack-ae ./run.sh ...` after ./setup.sh"
        )
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        failures.append("x86-64 Linux is required by the bundled capacity extension")
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        imported[name] = module
        versions.append((name, str(getattr(module, "__version__", "vendored"))))

    capacity_path: Path | None = None
    capacity_module = imported.get("mosaic.cost._capacity")
    if capacity_module is not None:
        raw_capacity_path = getattr(capacity_module, "__file__", None)
        if not raw_capacity_path:
            failures.append("mosaic.cost._capacity: imported module has no file path")
        else:
            capacity_path = Path(str(raw_capacity_path)).resolve()
            expected_parent = (ROOT / "src" / "deepstack" / "mosaic" / "cost").resolve()
            if capacity_path.parent != expected_parent:
                failures.append(
                    "mosaic.cost._capacity: loaded outside the bundled source tree: "
                    f"{capacity_path}"
                )
            if not any(
                capacity_path.name.endswith(suffix)
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            ):
                failures.append(
                    "mosaic.cost._capacity: expected the bundled CPython extension, "
                    f"got {capacity_path.name}"
                )

    noc_profile_path = check_noc_profile_provider(failures)
    tilesight_noc_profile_path = check_tilesight_noc_profile_provider(failures)
    arch_profile_path = check_arch_reference_provider(failures)
    binary_paths = tuple(
        path
        for path in (
            capacity_path,
            arch_profile_path,
            noc_profile_path,
            tilesight_noc_profile_path,
        )
        if path is not None
    )
    license_sidecars = tuple(
        check_binary_license_sidecar(path, failures)
        for path in binary_paths
    )

    required_paths = (
        ROOT / "src" / "deepstack" / "mosaic" / "__init__.py",
        ROOT / "src" / "tilesight" / "tilesight" / "__init__.py",
        ROOT / "src" / "deepstack" / "mosaic" / "data" / "aime_ds_r1" / "moe_activations_batch0.npz",
        ROOT / "src" / "deepstack" / "mosaic" / "data" / "aime_qwen_235b" / "qwen3_moe_activations_batch0.npz",
    )
    for path in required_paths:
        if not path.is_file():
            failures.append(f"missing file: {path.relative_to(ROOT)}")

    checksum_check = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_release_checksums.py"),
            "--verify",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if checksum_check.returncode:
        detail = (checksum_check.stderr or checksum_check.stdout).strip()
        failures.append(f"release checksums: {detail}")

    print(f"artifact_root={ROOT}")
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"cpu_count={os.cpu_count()}")
    for name, version in versions:
        print(f"import {name}: OK ({version})")
    if capacity_path is not None:
        print(f"capacity_provider=binary ({capacity_path.name})")
    if noc_profile_path is not None:
        print(
            "noc_energy_profile_provider=binary "
            f"({noc_profile_path.name}, noc_api={NOC_PROFILE_API_VERSION}, "
            f"energy_api={ENERGY_PROFILE_API_VERSION})"
        )
    if tilesight_noc_profile_path is not None:
        print(
            "tilesight_noc_profile_provider=binary "
            f"({tilesight_noc_profile_path.name})"
        )
    if arch_profile_path is not None:
        print(
            "arch_reference_provider=binary "
            f"({arch_profile_path.name}, api={ARCH_PROFILE_API_VERSION})"
        )
    if binary_paths and all(sidecar is not None for sidecar in license_sidecars):
        print(
            f"binary_license={PROPRIETARY_BINARY_LICENSE_ID} "
            f"({len(binary_paths)}/{len(binary_paths)} sidecars)"
        )
    if checksum_check.returncode == 0:
        print(checksum_check.stdout.strip())

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("doctor: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepStack AE workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="check the environment and bundled inputs")
    quick = subparsers.add_parser("quick", help="run the short smoke workflow")
    quick.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    quick.add_argument(
        "--model-version",
        choices=TOP_LEVEL_MODEL_VERSION_CHOICES,
        default="both",
        help=(
            "the official workflow always closes both model paths; run an "
            "individual ae.figXX module for a single-path diagnostic"
        ),
    )
    quick.add_argument(
        "--result-dir",
        type=Path,
        default=RESULTS_DIR / "quick",
        help="self-contained output directory (default: results/quick)",
    )
    reproduce = subparsers.add_parser("reproduce", help="run the default reproduction")
    reproduce.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    reproduce.add_argument(
        "--model-version",
        choices=TOP_LEVEL_MODEL_VERSION_CHOICES,
        default="both",
        help=(
            "the official workflow always closes both model paths; run an "
            "individual ae.figXX module for a single-path diagnostic"
        ),
    )
    reproduce.add_argument(
        "--result-dir",
        type=Path,
        default=RESULTS_DIR / "reproduce",
        help="self-contained output directory (default: results/reproduce)",
    )
    verify = subparsers.add_parser("verify", help="verify an existing result directory")
    verify.add_argument("result_dir", type=Path, nargs="?", default=RESULTS_DIR / "reproduce")
    plot = subparsers.add_parser(
        "plot", help="redraw all paper-facing figures from existing result CSVs"
    )
    plot.add_argument(
        "--result-dir", type=Path, default=RESULTS_DIR / "reproduce"
    )
    plot.add_argument(
        "--model-version",
        choices=("paper_legacy", "current_corrected"),
        default="paper_legacy",
    )
    return parser


def resolve_result_dir(path: Path) -> Path:
    """Anchor reviewer-supplied relative result paths at the artifact root."""

    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "quick":
        status = doctor()
        if status:
            return status
        from .workflow import run_workflow

        return run_workflow(
            scope="quick",
            workers=max(1, args.workers),
            model_version=args.model_version,
            result_root=resolve_result_dir(args.result_dir),
        )
    if args.command == "reproduce":
        status = doctor()
        if status:
            return status
        from .workflow import run_workflow

        result_root = resolve_result_dir(args.result_dir)
        status = run_workflow(
            scope="reproduce",
            workers=args.workers,
            model_version=args.model_version,
            result_root=result_root,
        )
        if status:
            return status
        from .verification import verify_all

        return verify_all(result_root)
    if args.command == "verify":
        from .verification import verify_all

        return verify_all(resolve_result_dir(args.result_dir))
    if args.command == "plot":
        from .plot_all import render_all_figures

        rows = render_all_figures(
            result_root=resolve_result_dir(args.result_dir),
            model_version=args.model_version,
        )
        print(f"plots: PASS ({len(rows)} PNG + {len(rows)} PDF)")
        print(f"output: {resolve_result_dir(args.result_dir) / 'figures'}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
