from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEEPSTACK_SRC = ROOT / "src" / "deepstack"
TILESIGHT_SRC = ROOT / "src" / "tilesight"
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def display_path(path: str | Path) -> Path:
    """Return a repository-relative path when possible for portable logs."""

    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else ROOT / candidate
    try:
        return resolved.resolve().relative_to(ROOT)
    except ValueError:
        return candidate


def activate_vendored_sources() -> None:
    """Put the two vendored source roots first on ``sys.path``."""

    for path in (TILESIGHT_SRC, DEEPSTACK_SRC):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
