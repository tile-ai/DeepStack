from __future__ import annotations

import importlib
import importlib.machinery
from pathlib import Path

from .paths import ROOT


NOC_PROFILE_MODULE = "mosaic.noc._model_support"
TILESIGHT_NOC_PROFILE_MODULE = "tilesight.distributed.noc._model_support"
NOC_PROFILE_API_VERSION = 1
ENERGY_PROFILE_API_VERSION = 1
ARCH_PROFILE_MODULE = "mosaic.arch._reference_model"
ARCH_PROFILE_API_VERSION = 3
PROPRIETARY_BINARY_LICENSE_ID = "LicenseRef-DeepStack-AE-Binary-1.0"
PROPRIETARY_BINARY_LICENSE_PATH = (
    ROOT / "LICENSES" / f"{PROPRIETARY_BINARY_LICENSE_ID}.txt"
)
ARCH_PROFILE_REQUIRED_CALLABLES = (
    "q01",
    "q02",
    "q03",
    "q04",
    "q05",
    "q06",
    "q07",
    "q08",
    "q09",
    "q11",
    "q12",
    "q13",
)
NOC_PROFILE_FACTORY = "p00"
NOC_PROFILE_VERSION = "p98"
NOC_ENERGY_VERSION = "p97"
NOC_ENERGY_REQUIRED_CALLABLES = ("p93", "p94", "p95", "p96")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def check_binary_license_sidecar(
    binary_path: Path,
    failures: list[str],
) -> Path | None:
    """Check that a bundled extension retains its proprietary scope marker."""

    sidecar = Path(f"{binary_path}.license")
    if not PROPRIETARY_BINARY_LICENSE_PATH.is_file():
        failures.append(
            "missing proprietary binary license text: "
            f"{_display_path(PROPRIETARY_BINARY_LICENSE_PATH)}"
        )
    if not sidecar.is_file():
        failures.append(
            f"missing binary license sidecar: {_display_path(sidecar)}"
        )
        return None
    try:
        content = sidecar.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - diagnostic path
        failures.append(
            f"binary license sidecar unreadable: "
            f"{_display_path(sidecar)} ({type(exc).__name__})"
        )
        return None
    expected = f"SPDX-License-Identifier: {PROPRIETARY_BINARY_LICENSE_ID}"
    if expected not in content:
        failures.append(
            f"binary license sidecar has wrong SPDX identifier: "
            f"{_display_path(sidecar)}"
        )
    return sidecar


def _check_binary_profile_provider(
    module_name: str,
    expected_parent: Path,
    failures: list[str],
    *,
    require_version: bool,
) -> Path | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        failures.append(
            f"{module_name}: import failed ({type(exc).__name__})"
        )
        return None

    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        failures.append(f"{module_name}: imported module has no file path")
        return None

    module_path = Path(str(raw_path)).resolve()
    if module_path.parent != expected_parent.resolve():
        failures.append(
            f"{module_name}: loaded outside the bundled source tree"
        )
    if not any(
        module_path.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        failures.append(
            f"{module_name}: expected a bundled CPython extension"
        )

    if require_version:
        api_version = getattr(module, NOC_PROFILE_VERSION, None)
        if not callable(api_version):
            failures.append(f"{module_name}: missing API version function")
        else:
            try:
                version = api_version()
            except Exception as exc:  # pragma: no cover - diagnostic path
                failures.append(
                    f"{module_name}: API version check failed "
                    f"({type(exc).__name__})"
                )
            else:
                if version != NOC_PROFILE_API_VERSION:
                    failures.append(f"{module_name}: incompatible API version")

    factory = getattr(module, NOC_PROFILE_FACTORY, None)
    if not callable(factory):
        failures.append(f"{module_name}: missing reference profile factory")
    else:
        try:
            profile = factory(0)
            layers = getattr(profile, "layers")
            devices = getattr(profile, "num_devices")
            if not layers or int(devices) <= 0:
                raise ValueError("invalid profile shape")
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(
                f"{module_name}: profile construction failed "
                f"({type(exc).__name__})"
            )

    return module_path


def check_noc_profile_provider(failures: list[str]) -> Path | None:
    """Validate the DeepStack NoC/energy provider without exposing its data."""

    path = _check_binary_profile_provider(
        NOC_PROFILE_MODULE,
        ROOT / "src" / "deepstack" / "mosaic" / "noc",
        failures,
        require_version=True,
    )
    if path is None:
        return None

    module = importlib.import_module(NOC_PROFILE_MODULE)
    energy_version = getattr(module, NOC_ENERGY_VERSION, None)
    if not callable(energy_version):
        failures.append(
            f"{NOC_PROFILE_MODULE}: missing energy API version function"
        )
    else:
        try:
            version = energy_version()
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(
                f"{NOC_PROFILE_MODULE}: energy API version check failed "
                f"({type(exc).__name__})"
            )
        else:
            if version != ENERGY_PROFILE_API_VERSION:
                failures.append(
                    f"{NOC_PROFILE_MODULE}: incompatible energy API version"
                )

    for name in NOC_ENERGY_REQUIRED_CALLABLES:
        if not callable(getattr(module, name, None)):
            failures.append(
                f"{NOC_PROFILE_MODULE}: missing required energy interface {name}"
            )
    return path


def check_tilesight_noc_profile_provider(failures: list[str]) -> Path | None:
    """Validate the TileSight provider without exposing its profile data."""

    return _check_binary_profile_provider(
        TILESIGHT_NOC_PROFILE_MODULE,
        ROOT / "src" / "tilesight" / "tilesight" / "distributed" / "noc",
        failures,
        require_version=False,
    )


def check_arch_reference_provider(failures: list[str]) -> Path | None:
    """Validate the bundled architecture provider without reading its values."""

    try:
        module = importlib.import_module(ARCH_PROFILE_MODULE)
    except Exception as exc:  # pragma: no cover - diagnostic path
        failures.append(
            f"{ARCH_PROFILE_MODULE}: import failed ({type(exc).__name__})"
        )
        return None

    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        failures.append(f"{ARCH_PROFILE_MODULE}: imported module has no file path")
        return None

    module_path = Path(str(raw_path)).resolve()
    expected_parent = (
        ROOT / "src" / "deepstack" / "mosaic" / "arch"
    ).resolve()
    if module_path.parent != expected_parent:
        failures.append(
            f"{ARCH_PROFILE_MODULE}: loaded outside the bundled source tree"
        )
    if not any(
        module_path.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        failures.append(
            f"{ARCH_PROFILE_MODULE}: expected a bundled CPython extension"
        )

    api_version = getattr(module, "model_support_api_version", None)
    if not callable(api_version):
        failures.append(f"{ARCH_PROFILE_MODULE}: missing API version function")
    else:
        try:
            version = api_version()
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(
                f"{ARCH_PROFILE_MODULE}: API version check failed "
                f"({type(exc).__name__})"
            )
        else:
            if version != ARCH_PROFILE_API_VERSION:
                failures.append(
                    f"{ARCH_PROFILE_MODULE}: incompatible API version"
                )

    reference_arch_type = getattr(module, "ReferenceArch", None)
    if not isinstance(reference_arch_type, type):
        failures.append(
            f"{ARCH_PROFILE_MODULE}: missing opaque ReferenceArch type"
        )

    for name in ARCH_PROFILE_REQUIRED_CALLABLES:
        if not callable(getattr(module, name, None)):
            failures.append(
                f"{ARCH_PROFILE_MODULE}: missing required function {name}"
            )

    return module_path
