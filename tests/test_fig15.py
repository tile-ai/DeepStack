from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ae import fig15


def test_quick_uses_distinct_sample_name_and_population(
    tmp_path: Path, monkeypatch
) -> None:
    observed: dict[str, object] = {}

    def fake_run_candidates(
        candidate_csv: Path,
        output_csv: Path,
        *,
        workers: int,
        model_version: str,
        fig15_scope: str | None = None,
    ) -> pd.DataFrame:
        observed.update(
            candidate_csv=candidate_csv,
            output_csv=output_csv,
            workers=workers,
            model_version=model_version,
            fig15_scope=fig15_scope,
        )
        return pd.DataFrame(
            {
                "delta_from_paper_stps_pct": [0.0],
                "delta_from_paper_utps_pct": [0.0],
            }
        )

    monkeypatch.setattr(fig15, "run_candidates", fake_run_candidates)
    assert (
        fig15.run(
            scope="quick",
            model_version="paper_legacy",
            workers=3,
            output_root=tmp_path,
        )
        == 0
    )
    assert observed == {
        "candidate_csv": fig15.POPULATION,
        "output_csv": tmp_path / "paper_legacy" / "fig15_dse_sample.csv",
        "workers": 3,
        "model_version": "paper_legacy",
        "fig15_scope": "quick",
    }


def test_default_root_is_scoped_stage(monkeypatch) -> None:
    observed: dict[str, Path] = {}

    def fake_run_candidates(
        _candidate_csv: Path,
        output_csv: Path,
        **_kwargs,
    ) -> pd.DataFrame:
        observed["output"] = output_csv
        return pd.DataFrame(
            {
                "delta_from_paper_stps_pct": [0.0],
                "delta_from_paper_utps_pct": [0.0],
            }
        )

    monkeypatch.setattr(fig15, "run_candidates", fake_run_candidates)
    fig15.run(
        scope="quick",
        model_version="paper_legacy",
        workers=1,
    )
    assert observed["output"] == (
        fig15.RESULTS_DIR
        / "quick"
        / "stages"
        / "fig15"
        / "paper_legacy"
        / "fig15_dse_sample.csv"
    )


def test_quick_rejects_non_finite_model_delta(tmp_path: Path, monkeypatch) -> None:
    def fake_run_candidates(*_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "delta_from_paper_stps_pct": [np.nan],
                "delta_from_paper_utps_pct": [0.0],
            }
        )

    monkeypatch.setattr(fig15, "run_candidates", fake_run_candidates)
    with pytest.raises(AssertionError, match="non-finite"):
        fig15.run(
            scope="quick",
            model_version="paper_legacy",
            workers=1,
            output_root=tmp_path,
        )
