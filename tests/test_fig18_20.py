from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ae.fig18_20 import (
    EXPECTED_CONFIGS,
    EXPECTED_ROLE_COUNTS,
    PUBLISHED_MODEL_COLUMNS,
    REGRESSION_COLUMNS,
    REGRESSION_DIGEST_FORMAT,
    REGRESSION_SIGNIFICANT_DIGITS,
    _check_regression_digest,
    _write_regression_digest,
)
from ae.paths import DATA_DIR


def test_fig18_20_manifest_closure() -> None:
    frame = pd.read_csv(DATA_DIR / "fig18_20" / "fixed_configs.csv")
    assert len(frame) == EXPECTED_CONFIGS
    assert frame.candidate_id.is_unique
    assert set(frame.phase) == {"decode", "prefill"}
    roles = frame.selection_roles.str.split(";").explode().value_counts().to_dict()
    assert roles == EXPECTED_ROLE_COUNTS


def test_fig18_20_non_reversible_regression_baselines() -> None:
    manifest = pd.read_csv(DATA_DIR / "fig18_20" / "fixed_configs.csv")
    assert len(manifest) == EXPECTED_CONFIGS
    for version in ("paper_legacy", "current_corrected"):
        record = json.loads(
            (
                DATA_DIR / "fig18_20" / f"{version}_regression.sha256"
            ).read_text(encoding="utf-8")
        )
        assert record["format"] == REGRESSION_DIGEST_FORMAT
        assert record["model_version"] == version
        assert record["rows"] == EXPECTED_CONFIGS
        assert record["significant_digits"] == REGRESSION_SIGNIFICANT_DIGITS
        assert record["columns"] == list(REGRESSION_COLUMNS)
        assert len(record["sha256"]) == 64
        int(record["sha256"], 16)

    assert not (DATA_DIR / "fig18_20" / "paper_legacy_expected.csv").exists()
    assert not (DATA_DIR / "fig18_20" / "current_corrected_expected.csv").exists()


def test_fig18_20_publish_safe_input_schema() -> None:
    root = DATA_DIR / "fig18_20"
    manifest = pd.read_csv(root / "fixed_configs.csv")
    assert "total_power_per_device_W" not in manifest
    assert "chip_power_W" not in manifest
    assert "noc_power_per_device_W" not in manifest
    assert not any(column.startswith("paper_") for column in manifest)

    terrain = pd.read_csv(root / "archive" / "fig20_decode_thermal_terrain.csv")
    assert list(terrain.columns) == [
        "terrain_reference_id",
        "dram_total_layers",
        "dram_active_layers",
        "bs",
        "temperature_C",
        "raw_stps",
    ]

    assert not any(
        column.startswith(("paper_", "recomputed_"))
        or "temperature" in column.lower()
        or "power" in column.lower()
        or "joule" in column.lower()
        for column in PUBLISHED_MODEL_COLUMNS
    )
    assert {
        "candidate_id",
        "terrain_reference_id",
        "phase",
        "selection_roles",
        "model_version",
        "bs",
        "dram_total_layers",
        "dram_active_layers",
    } <= set(PUBLISHED_MODEL_COLUMNS)

    curves = pd.read_csv(root / "archive" / "fig18_throughput_curves.csv")
    assert "archived_winner_id" not in curves


def test_fig18_20_digest_detects_internal_numeric_drift(tmp_path: Path) -> None:
    rows = []
    for index in range(EXPECTED_CONFIGS):
        row = {"candidate_id": f"candidate-{index:03d}"}
        for column_index, column in enumerate(REGRESSION_COLUMNS[1:], start=1):
            row[column] = (
                bool(index % 2)
                if column == "hit_power_wall"
                else float(index + column_index / 100.0)
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    path = tmp_path / "paper_legacy_regression.sha256"
    _write_regression_digest(frame, path, "paper_legacy")
    assert _check_regression_digest(frame, path, "paper_legacy")[0]

    changed = frame.copy()
    changed.loc[0, "temperature_C"] += 0.01
    assert not _check_regression_digest(
        changed, path, "paper_legacy"
    )[0]


def test_fig18_20_archived_plot_projection_closure() -> None:
    archive = DATA_DIR / "fig18_20" / "archive"
    expected = {
        "fig18_throughput_curves.csv": 110,
        "fig19_metric_grid.csv": 108,
        "fig20_decode_thermal_terrain.csv": 19233,
    }
    for filename, rows in expected.items():
        assert len(pd.read_csv(archive / filename)) == rows

    terrain = pd.read_csv(archive / "fig20_decode_thermal_terrain.csv")
    expected_strata = {
        (bs, layer) for bs in (4, 1024) for layer in range(2, 13)
    }
    assert terrain.terrain_reference_id.is_unique
    assert terrain.dram_total_layers.eq(terrain.dram_active_layers).all()
    assert set(zip(terrain.bs, terrain.dram_total_layers, strict=True)) == expected_strata

    manifest = pd.read_csv(DATA_DIR / "fig18_20" / "fixed_configs.csv")
    samples = manifest[
        manifest.selection_roles.str.split(";").map(
            lambda roles: "fig20_archive_sample" in roles
        )
    ]
    assert len(samples) == 22
    assert set(samples.bs) == {4, 1024}
    assert samples.groupby(["bs", "dram_total_layers"]).size().eq(1).all()
