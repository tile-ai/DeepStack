#!/usr/bin/env python3
"""Audit release binaries without loading proprietary model code.

The checks intentionally use only the Python standard library and GNU
binutils already required by the Linux artifact.  A private numeric manifest
may be supplied at release-build time to prove that selected calibration
values do not occur as plain IEEE-754 constants.  The manifest itself must
remain outside the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BINARY_LICENSE_ID = "LicenseRef-DeepStack-AE-Binary-1.0"
MAX_CAPACITY_BINARY_BYTES = 10_000_000
RELEASE_BINARY_PATTERNS = (
    "src/deepstack/mosaic/arch/_reference_model*.so",
    "src/deepstack/mosaic/cost/_capacity*.so",
    "src/deepstack/mosaic/noc/_model_support*.so",
    "src/tilesight/tilesight/distributed/noc/_model_support*.so",
)

FORBIDDEN_SECTIONS = frozenset(
    {
        ".comment",
        ".symtab",
        ".strtab",
    }
)

COMMON_FORBIDDEN_MARKERS = (
    b".pyx",
    b".pxd",
    b"/data/",
    b"/homes/",
    b"/tmp/",
    b"__pyx_unpickle",
    b"__reduce_cython__",
    b"__setstate_cython__",
    b"reference_power_wall",
    b"stacked_gpu_4x16",
    b"ReferenceChipEnergyModel",
    b"_reference_noc_energy_for_layer",
    b"_apply_standard",
    b"_efficiency",
    b"_require_reference",
    b"_update_derived",
    b"_update_dram",
    b"area_confidential",
    b"Making Strong ECC",
    b"Optional quadratic intra-cluster crossbar scaling",
    b"_dram_layers",
    b"_dram_bw_per_cluster",
    b"_dram_cap_per_cluster_gb",
    b"reference_noc_l1_bw",
    b"PURE_NOC_BASE_PER_SM_UM2",
    b"THERMAL_R_BASE",
)

CAPACITY_FORBIDDEN_MARKERS = (
    b"AreaModel",
    b"mosaic.cost._capacity.p_key",
    b"mosaic.cost._capacity.p_u",
    b"p_lookup",
    b"register_capacity_per_sm",
    b"tensor_core_shape",
    b"int32_cores_per_sm",
    b"fp32_cores_per_sm",
    b"sfu_cores_per_sm",
    b"tensor_cores_per_sm",
    b"xbar_quadratic_ref_sm",
    b"l1_smem_throughput_per_cycle",
    b"invalid protected architecture",
    b"invalid interconnect object",
    b"interconnect must contain three layers",
    b"interconnect bandwidths must be finite and positive",
    b"get_register_area",
    b"get_tensor_core_area",
    b"get_smem_area",
    b"get_cuda_core_area",
    b"get_sfu_core_area",
    b"get_sm_area",
    b"get_cluster_area",
    b"get_cluster_noc_area",
    b"get_mem_controller_area",
    b"get_dram_peripheral_area",
    b"get_l1_noc_area",
    b"get_die_area",
    b"get_thermal_resistance",
)


@dataclass(frozen=True)
class BinaryAudit:
    path: Path
    dynamic_symbols: tuple[str, ...]
    sections: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self, *, relative_to: Path | None = None) -> dict[str, object]:
        display_path = (
            self.path.relative_to(relative_to)
            if relative_to is not None
            else self.path
        )
        return {
            "path": str(display_path),
            "dynamic_symbols": list(self.dynamic_symbols),
            "sections": list(self.sections),
            "failures": list(self.failures),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PlainNumberMatch:
    value: float
    representation: str
    offset: int


def _run_binutil(*command: str) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def extension_module_name(path: Path) -> str:
    """Return the final Python module component encoded in an extension name."""

    name = path.name
    for marker in (".cpython-", ".abi3.so", ".so"):
        if marker in name:
            return name.split(marker, 1)[0]
    raise ValueError(f"not a Python extension filename: {path}")


def expected_init_symbol(path: Path) -> str:
    return f"PyInit_{extension_module_name(path)}"


def defined_dynamic_symbols(path: Path) -> tuple[str, ...]:
    output = _run_binutil(
        "nm",
        "-D",
        "--defined-only",
        "--format=posix",
        str(path),
    )
    names = [
        line.split()[0]
        for line in output.splitlines()
        if line.strip()
    ]
    return tuple(sorted(names))


def elf_sections(path: Path) -> tuple[str, ...]:
    output = _run_binutil("readelf", "-SW", str(path))
    names = re.findall(r"^\s*\[\s*\d+\]\s+(\S+)", output, re.MULTILINE)
    return tuple(names)


def _program_header_failures(path: Path) -> list[str]:
    failures: list[str] = []
    output = _run_binutil("readelf", "-lW", str(path))
    stack_lines = [
        line for line in output.splitlines() if line.lstrip().startswith("GNU_STACK")
    ]
    if len(stack_lines) != 1:
        failures.append("expected exactly one GNU_STACK program header")
    else:
        flags = stack_lines[0].split()[-2]
        if "E" in flags:
            failures.append("GNU_STACK is executable")
    if "GNU_RELRO" not in output:
        failures.append("GNU_RELRO program header is missing")
    return failures


def _dynamic_tag_failures(path: Path) -> list[str]:
    failures: list[str] = []
    output = _run_binutil("readelf", "-dW", str(path))
    if "BIND_NOW" not in output and "Flags: NOW" not in output:
        failures.append("full RELRO is incomplete: BIND_NOW/NOW is missing")
    if "(RPATH)" in output or "(RUNPATH)" in output:
        failures.append("binary contains RPATH or RUNPATH")
    return failures


def _property_note_failures(path: Path) -> list[str]:
    output = _run_binutil("readelf", "-nW", str(path))
    failures: list[str] = []
    for feature in ("IBT", "SHSTK"):
        if feature not in output:
            failures.append(f"GNU property {feature} is missing")
    return failures


def adjacent_license_failures(path: Path) -> list[str]:
    sidecar = path.with_name(path.name + ".license")
    if not sidecar.is_file():
        return [f"missing adjacent license sidecar: {sidecar.name}"]
    text = sidecar.read_text(encoding="utf-8")
    required = (
        "SPDX-FileCopyrightText:",
        f"SPDX-License-Identifier: {BINARY_LICENSE_ID}",
    )
    return [
        f"{sidecar.name} is missing {line!r}"
        for line in required
        if line not in text
    ]


def marker_failures(
    path: Path,
    markers: Sequence[bytes],
) -> list[str]:
    payload = path.read_bytes()
    return [
        f"payload contains forbidden marker {marker!r}"
        for marker in markers
        if marker in payload
    ]


def audit_binary(
    path: Path,
    *,
    forbidden_markers: Sequence[bytes] = COMMON_FORBIDDEN_MARKERS,
    require_cet: bool = True,
    require_full_relro: bool = True,
    max_size_bytes: int | None = None,
) -> BinaryAudit:
    path = path.resolve()
    failures: list[str] = []
    if not path.is_file():
        return BinaryAudit(path, (), (), ("binary does not exist",))
    if max_size_bytes is not None and path.stat().st_size > max_size_bytes:
        failures.append(
            f"binary is {path.stat().st_size} bytes; "
            f"maximum is {max_size_bytes} bytes"
        )

    symbols = defined_dynamic_symbols(path)
    expected = (expected_init_symbol(path),)
    if symbols != expected:
        failures.append(
            f"defined dynamic symbols are {symbols!r}, expected {expected!r}"
        )

    sections = elf_sections(path)
    forbidden_sections = sorted(
        section
        for section in sections
        if (
            section in FORBIDDEN_SECTIONS
            or section.startswith(".debug")
            or section.startswith(".gnu.lto")
        )
    )
    if forbidden_sections:
        failures.append(
            "forbidden ELF sections: " + ", ".join(forbidden_sections)
        )

    failures.extend(_program_header_failures(path))
    if require_full_relro:
        failures.extend(_dynamic_tag_failures(path))
    else:
        dynamic = _run_binutil("readelf", "-dW", str(path))
        if "(RPATH)" in dynamic or "(RUNPATH)" in dynamic:
            failures.append("binary contains RPATH or RUNPATH")
    if require_cet:
        failures.extend(_property_note_failures(path))
    failures.extend(adjacent_license_failures(path))
    failures.extend(marker_failures(path, forbidden_markers))
    return BinaryAudit(path, symbols, sections, tuple(failures))


def discover_release_binaries(
    root: Path = REPOSITORY_ROOT,
) -> tuple[Path, ...]:
    binaries: list[Path] = []
    for pattern in RELEASE_BINARY_PATTERNS:
        matches = sorted(root.glob(pattern))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one binary matching {pattern!r}; "
                f"found {len(matches)}"
            )
        binaries.append(matches[0])
    return tuple(binaries)


def load_private_number_manifest(path: Path) -> tuple[float, ...]:
    """Load a private JSON list of finite, non-trivial numeric fingerprints."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("private number manifest must be a JSON list")
    values: list[float] = []
    for index, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"manifest entry {index} is not numeric")
        value = float(item)
        if not math.isfinite(value):
            raise ValueError(f"manifest entry {index} is not finite")
        if value in {-1.0, 0.0, 1.0}:
            raise ValueError(
                f"manifest entry {index} is too common for a useful fingerprint"
            )
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError("private number manifest contains duplicates")
    return tuple(values)


def find_plain_number_matches(
    payload: bytes,
    values: Iterable[float],
) -> tuple[PlainNumberMatch, ...]:
    """Find plain float32/float64 encodings in an ELF payload."""

    matches: list[PlainNumberMatch] = []
    for value in values:
        representations: list[tuple[str, bytes]] = [
            ("float64-le", struct.pack("<d", value)),
            ("float64-be", struct.pack(">d", value)),
        ]
        try:
            packed32_le = struct.pack("<f", value)
            packed32_be = struct.pack(">f", value)
        except OverflowError:
            pass
        else:
            representations.extend(
                (
                    ("float32-le", packed32_le),
                    ("float32-be", packed32_be),
                )
            )
        for name, encoded in representations:
            start = 0
            while True:
                offset = payload.find(encoded, start)
                if offset < 0:
                    break
                matches.append(PlainNumberMatch(value, name, offset))
                start = offset + 1
    return tuple(matches)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="artifact root (default: repository containing this script)",
    )
    parser.add_argument(
        "--private-number-manifest",
        type=Path,
        help="untracked JSON list of sensitive numeric fingerprints",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable report",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    failures: list[str] = []
    try:
        binaries = discover_release_binaries(root)
    except ValueError as exc:
        failures.append(str(exc))
        binaries = ()

    audits: list[BinaryAudit] = []
    for binary in binaries:
        markers = COMMON_FORBIDDEN_MARKERS
        is_capacity = extension_module_name(binary).startswith("_capacity")
        if is_capacity:
            markers = markers + CAPACITY_FORBIDDEN_MARKERS
        audit = audit_binary(
            binary,
            forbidden_markers=markers,
            require_full_relro=is_capacity,
            max_size_bytes=(
                MAX_CAPACITY_BINARY_BYTES if is_capacity else None
            ),
        )
        audits.append(audit)
        failures.extend(
            f"{binary.relative_to(root)}: {failure}"
            for failure in audit.failures
        )

    number_matches: dict[str, list[dict[str, object]]] = {}
    if args.private_number_manifest is not None:
        values = load_private_number_manifest(args.private_number_manifest)
        manifest_indexes = {
            value: index for index, value in enumerate(values)
        }
        for binary in binaries:
            matches = find_plain_number_matches(binary.read_bytes(), values)
            if matches:
                relative = str(binary.relative_to(root))
                number_matches[relative] = [
                    {
                        "manifest_index": manifest_indexes[match.value],
                        "representation": match.representation,
                        "offset": match.offset,
                    }
                    for match in matches
                ]
                failures.append(
                    f"{relative}: {len(matches)} plain numeric fingerprints"
                )

    binary_license_path = (
        root
        / "LICENSES"
        / f"{BINARY_LICENSE_ID}.txt"
    )
    if not binary_license_path.is_file():
        failures.append(
            f"missing binary license text: "
            f"{binary_license_path.relative_to(root)}"
        )

    report = {
        "binaries": [
            audit.as_dict(relative_to=root) for audit in audits
        ],
        "plain_number_matches": number_matches,
        "failures": failures,
        "passed": not failures,
    }
    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        for audit in audits:
            state = "PASS" if audit.passed else "FAIL"
            print(f"{state}: {audit.path.relative_to(root)}")
            for failure in audit.failures:
                print(f"  - {failure}")
        print("binary hardening audit: " + ("PASS" if not failures else "FAIL"))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
