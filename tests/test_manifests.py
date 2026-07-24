from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from ae.paths import CONFIG_DIR, DATA_DIR
from ae.selected_decode import (
    load_candidate_table,
    select_fig15_local_best,
    select_fig15_quick,
)
from ae.verification import _check_manifest_closure, _verify_decode_derived_columns


@pytest.fixture(scope="module")
def fig15_population() -> pd.DataFrame:
    return load_candidate_table(
        DATA_DIR / "fig15" / "decode_dse_population_march2026.csv"
    )


def _id_digest(frame: pd.DataFrame) -> str:
    lines = "\n".join(sorted(frame.candidate_id.astype(str))) + "\n"
    return hashlib.sha256(lines.encode()).hexdigest()


def test_fig15_population_and_local_best_closure(
    fig15_population: pd.DataFrame,
) -> None:
    assert len(fig15_population) == 301_728
    assert fig15_population.candidate_id.is_unique
    assert fig15_population.stats_run_id.is_unique
    assert (fig15_population.candidate_id == fig15_population.stats_run_id).all()
    assert fig15_population.arch.str.startswith("stacked_gpu_").all()

    local = select_fig15_local_best(fig15_population)
    assert len(local) == 784
    assert local.candidate_id.is_unique
    assert local.groupby(["model", "bs"]).ngroups == 28
    assert _id_digest(local) == (
        "c6cbc02c29410205017edbc4db018cbe244ae9b2816ee9c00033bc96b39aad66"
    )
    assert local.expected_current_utps_avg.notna().all()
    assert local.expected_current_stps_avg.notna().all()
    assert fig15_population.expected_current_stps_avg.notna().sum() == 784


def test_fig15_quick_is_deterministic_non_winner_sample(
    fig15_population: pd.DataFrame,
) -> None:
    winners = select_fig15_local_best(fig15_population)
    quick = select_fig15_quick(fig15_population)
    assert len(quick) == 28
    assert (quick.groupby(["model", "bs"]).size() == 1).all()
    assert set(quick.candidate_id).isdisjoint(winners.candidate_id)
    assert _id_digest(quick) == (
        "3c70dd0b591e77970097d566ddb5b30b3d5704c486c4fb2fb9515a09910512ed"
    )


def test_fig15_verifier_rejects_canonical_field_tampering(
    fig15_population: pd.DataFrame,
) -> None:
    manifest = select_fig15_local_best(fig15_population)
    tampered = manifest.copy()
    tampered.loc[0, "paper_stps_avg"] *= 1.01

    with pytest.raises(AssertionError, match="canonical DSE manifest"):
        _check_manifest_closure(tampered, manifest)


def test_fig15_verifier_recomputes_derived_columns(
    fig15_population: pd.DataFrame,
) -> None:
    manifest = select_fig15_quick(fig15_population)
    replay = manifest.copy()
    replay["current_stps_avg"] = replay.paper_stps_avg * 1.1
    replay["current_utps_avg"] = replay.paper_utps_avg * 1.1
    replay["delta_from_paper_stps_pct"] = 10.0
    replay["delta_from_paper_utps_pct"] = 10.0
    replay["current_stps_rank"] = replay.groupby(["model", "bs"])[
        "current_stps_avg"
    ].rank(method="min", ascending=False)
    replay["current_utps_rank"] = replay.groupby(["model", "bs"])[
        "current_utps_avg"
    ].rank(method="min", ascending=False)
    assert _verify_decode_derived_columns(replay, manifest) == pytest.approx(10.0)

    replay.loc[0, "delta_from_paper_stps_pct"] = 0.0
    with pytest.raises(AssertionError, match="independently recomputed"):
        _verify_decode_derived_columns(replay, manifest)


def test_fig16_manifest_closure() -> None:
    frame = pd.read_csv(CONFIG_DIR / "fig16_decode_winners_march2026.csv")
    assert len(frame) == 80
    assert frame.candidate_id.is_unique
    assert set(frame.platform) == {"DeepStack", "H100_SCALED", "H200_SCALED"}
    assert len(frame[frame.platform == "DeepStack"]) == 28
    assert len(frame[frame.platform == "H100_SCALED"]) == 26
    assert len(frame[frame.platform == "H200_SCALED"]) == 26


def test_fig21_manifest_grid() -> None:
    frame = pd.read_csv(DATA_DIR / "fig21" / "candidates_march2026.csv")
    assert len(frame) == 180
    assert frame.candidate_id.is_unique
    assert set(frame.bs) == {4, 64, 1024}
    assert set(frame.latency_multiplier) == {0.25, 0.5, 1.0, 2.0, 4.0}
    assert set(frame.bw_multiplier) == {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}
    assert set(frame.tp_transform_moe) == {"none", "replace_only"}
