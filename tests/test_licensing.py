from __future__ import annotations

import tomllib
from pathlib import Path

from ae.doctor import (
    PROPRIETARY_BINARY_LICENSE_ID,
    PROPRIETARY_BINARY_LICENSE_PATH,
)
from ae.paths import ROOT


BINARY_GLOBS = (
    "src/deepstack/mosaic/arch/_reference_model*.so",
    "src/deepstack/mosaic/cost/_capacity*.so",
    "src/deepstack/mosaic/noc/_model_support*.so",
    "src/tilesight/tilesight/distributed/noc/_model_support*.so",
)


def _one_binary(pattern: str) -> Path:
    matches = tuple(ROOT.glob(pattern))
    assert len(matches) == 1, (pattern, matches)
    return matches[0]


def test_every_proprietary_binary_has_an_exact_spdx_sidecar() -> None:
    expected_identifier = (
        f"SPDX-License-Identifier: {PROPRIETARY_BINARY_LICENSE_ID}"
    )

    for pattern in BINARY_GLOBS:
        binary = _one_binary(pattern)
        sidecar = Path(f"{binary}.license")
        assert sidecar.is_file()
        content = sidecar.read_text(encoding="utf-8")
        assert expected_identifier in content
        assert "SPDX-FileCopyrightText: 2026 DeepStack Authors" in content


def test_binary_inventory_covers_every_distributed_shared_object() -> None:
    expected = {_one_binary(pattern) for pattern in BINARY_GLOBS}
    discovered = {
        path
        for path in ROOT.rglob("*.so")
        if ".git" not in path.parts
    }
    assert discovered == expected


def test_proprietary_binaries_do_not_embed_local_machine_identity() -> None:
    forbidden = (
        b"/data/",
        b"/home/",
        b"/homes/",
        b"/root/",
        b"/tmp/",
    )

    for pattern in BINARY_GLOBS:
        binary = _one_binary(pattern)
        payload = binary.read_bytes().lower()
        for marker in forbidden:
            assert marker not in payload, (binary, marker)


def test_mixed_license_scope_and_ae_archival_grant_are_explicit() -> None:
    root_scope = (ROOT / "LICENSE").read_text(encoding="utf-8")
    license_map = (ROOT / "LICENSING.md").read_text(encoding="utf-8")
    binary_terms = PROPRIETARY_BINARY_LICENSE_PATH.read_text(encoding="utf-8")
    normalized_binary_terms = " ".join(binary_terms.split())

    assert (ROOT / "LICENSES" / "Apache-2.0.txt").is_file()
    assert (ROOT / "NOTICE").is_file()
    assert "mixed-license distribution" in root_scope
    assert "Apache License, Version 2.0" in root_scope
    assert PROPRIETARY_BINARY_LICENSE_ID in root_scope
    for pattern in BINARY_GLOBS:
        assert pattern in root_scope
        assert pattern in license_map

    for required_term in (
        "worldwide",
        "perpetual",
        "irrespective of the recipient's employer affiliation",
        "public archival repositories",
        "exact, unmodified copies",
        "reverse engineer",
        "applicable law",
    ):
        assert required_term in normalized_binary_terms


def test_apache_package_metadata_selects_only_source_code() -> None:
    metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert metadata["project"]["license"] == "Apache-2.0"
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "ae*"
    ]
    assert "package-data" not in metadata["tool"]["setuptools"]
