from __future__ import annotations

import hashlib
from pathlib import Path

from ae.paths import DATA_DIR, ROOT


def _resolve(manifest: Path, relative: str) -> Path:
    local = manifest.parent / relative
    return local if local.is_file() else ROOT / relative


def test_all_packaged_reference_checksum_manifests() -> None:
    manifests = sorted(DATA_DIR.rglob("SHA256SUMS"))
    assert manifests, "no per-result checksum manifests were packaged"
    checked = 0
    for manifest in manifests:
        for line_number, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            checksum, relative = line.split(maxsplit=1)
            path = _resolve(manifest, relative)
            assert path.is_file(), f"{manifest}:{line_number}: missing {relative}"
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == checksum, f"{manifest}:{line_number}: {relative}"
            checked += 1
    assert checked >= 50
