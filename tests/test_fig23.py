from __future__ import annotations

import pandas as pd

from ae.fig23 import _validate_manifest
from ae.paths import DATA_DIR


def test_fig23_manifest_and_headline_claims() -> None:
    root = DATA_DIR / "fig23"
    winners = pd.read_csv(root / "selected_winners_march2026.csv")
    _validate_manifest(winners)

    summary = pd.read_csv(root / "paper_reference_summary.csv")
    headline = summary[(summary.phase == "decode") & (summary.bs == 1024)].set_index(
        "model"
    )
    assert round(headline.loc["DeepSeekV3", "flexible_over_astra"], 2) == 5.03
    assert round(
        headline.loc["Qwen3_235b_a22b", "flexible_over_astra"], 2
    ) == 2.31
    assert (summary.expanded_over_astra >= 1.0).all()
    assert (summary.flexible_over_expanded >= 1.0).all()

    expected = pd.read_csv(root / "current_corrected_expected.csv")
    assert len(expected) == 84
    assert set(expected.candidate_id) == set(winners.candidate_id)


def test_fig23_noc_area_supplement_is_bw1_cross_check() -> None:
    supplement = pd.read_csv(
        DATA_DIR / "fig23" / "noc_area_parallelism_supplement.csv"
    )
    assert len(supplement) == 18
    assert set(supplement.scenario) == {"logic_die", "noc_to_dram"}
    assert set(supplement.bw_multiplier) == {1.0}

    pivot = supplement.pivot(
        index=["bs", "tier"], columns="scenario", values="paper_scaled_stps"
    )
    assert (pivot.logic_die == pivot.noc_to_dram).all()
