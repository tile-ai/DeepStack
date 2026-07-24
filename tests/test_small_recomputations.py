from __future__ import annotations

import math

from ae.fig17 import PUBLISHED_FIELDS, publishable_rows, run_sweep, summarize, verify


def test_fig17_bandwidth_claim() -> None:
    full_rows = run_sweep()
    rows = publishable_rows(full_rows)
    summary = summarize(rows)
    global_row = next(row for row in summary if row["scope"] == "global")
    assert set(rows[0]) == set(PUBLISHED_FIELDS)
    assert {
        "ddr_peak_bw_TBs",
        "ddr_eff_bw_TBs",
        "littles_law_bw_TBs",
        "is_l1_bound",
        "die_area_mm2",
        "sm_count",
    }.isdisjoint(rows[0])
    assert len(rows) == 96
    assert global_row["peak_layer"] == 9
    assert round(global_row["drop_at_12_pct"], 2) == 39.62
    verify(rows, summary)
