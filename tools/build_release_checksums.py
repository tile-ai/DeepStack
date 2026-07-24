#!/usr/bin/env python3
"""Write or verify the deterministic checksum manifest for release inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS"
ROOT_EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    ".ae-env",
    "build",
    "dist",
    "results",
}
ANYWHERE_EXCLUDED_DIRS = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".aux", ".fdb_latexmk", ".fls", ".log", ".out"}


def release_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in ROOT_EXCLUDED_DIRS:
            continue
        if any(
            part in ANYWHERE_EXCLUDED_DIRS or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path == MANIFEST or not path.is_file() or path.is_symlink():
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_manifest() -> None:
    lines = [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in release_files()
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST} ({len(lines)} files)")


def verify_manifest() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"missing {MANIFEST}")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            checksum, relative = line.split("  ", 1)
        except ValueError as exc:
            raise SystemExit(f"invalid SHA256SUMS line {line_number}") from exc
        expected[relative] = checksum
    actual_files = {
        path.relative_to(ROOT).as_posix(): path for path in release_files()
    }
    if set(expected) != set(actual_files):
        missing = sorted(set(expected) - set(actual_files))
        unexpected = sorted(set(actual_files) - set(expected))
        raise SystemExit(
            f"release file-set mismatch: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    mismatches = [
        relative
        for relative, path in actual_files.items()
        if digest(path) != expected[relative]
    ]
    if mismatches:
        raise SystemExit(f"checksum mismatches: {mismatches[:5]}")
    print(f"release checksums: PASS ({len(expected)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_manifest()
    else:
        verify_manifest()


if __name__ == "__main__":
    main()
