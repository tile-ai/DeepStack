from __future__ import annotations

import math

import pandas as pd

from ae.paths import DATA_DIR
from ae.table4 import _published_model_points, _validate_manifest


PUBLISHED_MODEL_POINT_COLUMNS = (
    "candidate_id",
    "model_version",
    "step",
    "step_name",
    "bs",
    "recomputed_stps_avg",
    "delta_from_paper_stps_pct",
)
STEP6_REFERENCE_COLUMNS = (
    "dram_total_layers",
    "dram_active_layers",
    "sm_count",
    "smem_capacity_KiB",
    "l1_throughput_Bpc",
    "arch",
    "noc",
    "model",
    "bs",
    "minibatch",
    "seq",
    "tp",
    "ep",
    "ep1",
    "ep2",
    "sp",
    "cp",
    "dp",
    "fsdp",
    "pp",
    "tp_transform_moe",
    "kv_len_1",
    "kv_len_2",
    "kv_len_3",
    "kv_len_4",
    "scaled_stps_avg",
)
STEP7_REFERENCE_COLUMNS = (
    "hw_idx",
    "m",
    "n",
    "L1_mult",
    "L2_mult",
    "L3_mult",
    "sm_count",
    "tp",
    "ep",
    "dp",
    "pp",
    "fsdp",
    "tp_transform_moe",
    "scaled_stps_avg",
)


def test_table4_fixed_manifest_closure() -> None:
    manifest = pd.read_csv(DATA_DIR / "table4" / "fixed_configs.csv")
    _validate_manifest(manifest)
    assert len(manifest) == 14
    assert manifest.candidate_id.is_unique
    assert set(zip(manifest.step, manifest.bs, strict=True)) == {
        (step, bs) for step in range(1, 8) for bs in (4, 1024)
    }


def test_table4_paper_values_match_archived_summary() -> None:
    manifest = pd.read_csv(DATA_DIR / "table4" / "fixed_configs.csv")
    archived = pd.read_csv(
        DATA_DIR / "table4" / "search_reference" / "ablation_summary.csv"
    )
    merged = manifest.merge(
        archived[["step", "bs", "stps_avg"]],
        on=["step", "bs"],
        validate="one_to_one",
    )
    assert all(
        # Both files store the same IEEE-754 value; allow only the final CSV
        # decimal-parser round-trip bit at the largest (54k STPS) magnitude.
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-10)
        for actual, expected in zip(
            merged.paper_stps_avg, merged.stps_avg, strict=True
        )
    )


def test_table4_corrected_reference_matches_manifest() -> None:
    manifest = pd.read_csv(DATA_DIR / "table4" / "fixed_configs.csv")
    expected = pd.read_csv(
        DATA_DIR / "table4" / "current_corrected_expected.csv"
    )
    assert len(expected) == len(manifest)
    assert expected.candidate_id.is_unique
    assert set(expected.candidate_id) == set(manifest.candidate_id)


def test_table4_model_points_publish_projection() -> None:
    full = pd.DataFrame(
        [
            {
                "candidate_id": "table4_s01_bs4",
                "model_version": "paper_legacy",
                "step": 1,
                "step_name": "AstraSim (tp/pp/dp)",
                "bs": 4,
                "recomputed_stps_avg": 177.0,
                "delta_from_paper_stps_pct": 0.0,
                "raw_stps_avg": 178.0,
                "chip_power_W": 40.0,
                "tokens_per_joule": 1.0,
                "freq_scale_power": 0.9,
                "hit_power_wall": True,
                "stats_run_id": "private-source-id",
            }
        ]
    )
    published = _published_model_points(full)
    assert tuple(published.columns) == PUBLISHED_MODEL_POINT_COLUMNS
    assert published.iloc[0].candidate_id == "table4_s01_bs4"


def test_table4_search_references_use_publish_safe_schemas() -> None:
    reference = DATA_DIR / "table4" / "search_reference"
    ablation = pd.read_csv(reference / "ablation_summary.csv")
    assert tuple(ablation.columns) == (
        "step",
        "step_name",
        "bs",
        "stps_avg",
        "heating_feasible",
        "config",
    )
    assert len(ablation) == 14
    assert ablation.heating_feasible.map(str).str.lower().eq("true").all()

    for bs in (4, 1024):
        step6 = pd.read_csv(reference / f"step6_candidates_bs{bs}.csv")
        assert tuple(step6.columns) == STEP6_REFERENCE_COLUMNS
        assert step6.scaled_stps_avg.notna().all()

        for resolution in ("coarse", "fine"):
            step7 = pd.read_csv(
                reference / f"step7_{resolution}_bs{bs}.csv"
            )
            assert tuple(step7.columns) == STEP7_REFERENCE_COLUMNS
            assert step7.scaled_stps_avg.notna().all()
