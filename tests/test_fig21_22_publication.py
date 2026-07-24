from __future__ import annotations

import pandas as pd

from ae.fig21 import PUBLISHED_RAW_COLUMNS, _published_raw
from ae.fig22 import (
    PUBLISHED_MODEL_POINT_COLUMNS,
    PUBLISHED_VERSION_DRIFT_COLUMNS,
    _published_model_points,
    _published_version_drift,
)


def test_fig21_publication_projection_drops_reversible_intermediates() -> None:
    full = pd.DataFrame(
        {
            "candidate_id": ["fig21-0"],
            "model_version": ["paper_legacy"],
            "bs": [4],
            "latency_multiplier": [1.0],
            "bw_multiplier": [1.0],
            "tp_transform_moe": ["none"],
            "reproduced_stps_avg": [10.0],
            "reproduced_utps_avg": [2.0],
            "delta_from_paper_stps_pct": [0.0],
            "delta_from_paper_utps_pct": [0.0],
            "raw_stps_avg": [10.0],
            "raw_utps_avg": [2.0],
            "chip_power_per_device_w": [80.0],
            "noc_power_per_device_w": [5.0],
            "hit_power_wall": [False],
            "freq_scale_power": [1.0],
            "sm_count": [8],
            "model_runtime_s": [1.5],
            "tp": [8],
        }
    )

    published = _published_raw(full)

    assert tuple(published.columns) == PUBLISHED_RAW_COLUMNS
    assert set(full.columns) > set(published.columns)
    assert not {
        "raw_stps_avg",
        "raw_utps_avg",
        "chip_power_per_device_w",
        "noc_power_per_device_w",
        "hit_power_wall",
        "freq_scale_power",
        "sm_count",
        "model_runtime_s",
        "tp",
    }.intersection(published.columns)


def test_fig22_publication_projection_drops_model_internals() -> None:
    full = pd.DataFrame(
        {
            "point_id": ["fig22-0"],
            "model_version": ["paper_legacy"],
            "phase": ["decode"],
            "baseline_noc": ["torus_mesh_switch_1"],
            "scaled_layer": ["L2"],
            "bw_multiplier": [1.0],
            "bs": [4],
            "seq": [1],
            "reproduced_logic_stps": [10.0],
            "reproduced_dram_stps": [11.0],
            "logic_raw_stps": [12.0],
            "dram_raw_stps": [13.0],
            "logic_chip_power_w": [80.0],
            "logic_noc_power_w": [5.0],
            "logic_hit_power_wall": [True],
            "logic_frequency_scale": [0.9],
            "orig_sm_count": [6],
            "dram_tp": [8],
            "model_calls": [2],
            "model_runtime_s": [1.5],
            "gpu_runs": [0],
        }
    )

    published = _published_model_points(full)

    assert tuple(published.columns) == PUBLISHED_MODEL_POINT_COLUMNS
    assert set(full.columns) > set(published.columns)
    assert not {
        "logic_raw_stps",
        "dram_raw_stps",
        "logic_chip_power_w",
        "logic_noc_power_w",
        "logic_hit_power_wall",
        "logic_frequency_scale",
        "orig_sm_count",
        "dram_tp",
        "model_calls",
        "model_runtime_s",
        "gpu_runs",
    }.intersection(published.columns)


def test_fig22_version_drift_publishes_percentages_without_absolute_paths() -> None:
    full = pd.DataFrame(
        {
            "point_id": ["fig22-0"],
            "phase": ["decode"],
            "baseline_noc": ["torus_mesh_switch_1"],
            "scaled_layer": ["L2"],
            "bw_multiplier": [1.0],
            "bs": [4],
            "seq": [1],
            "reproduced_logic_stps_paper_legacy": [10.0],
            "reproduced_dram_stps_paper_legacy": [11.0],
            "reproduced_logic_stps_current_corrected": [10.5],
            "reproduced_dram_stps_current_corrected": [11.5],
            "logic_corrected_vs_legacy_pct": [5.0],
            "dram_corrected_vs_legacy_pct": [100.0 * (11.5 / 11.0 - 1.0)],
        }
    )

    published = _published_version_drift(full)

    assert tuple(published.columns) == PUBLISHED_VERSION_DRIFT_COLUMNS
    assert not any(
        column.endswith(("_paper_legacy", "_current_corrected"))
        for column in published.columns
    )
