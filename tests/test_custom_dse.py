import math
import runpy

import pytest

from conftest import ROOT
from mosaic.noc.noc_config_set import h200x32
from mosaic.noc import noc_config_set


example = runpy.run_path(str(ROOT / "examples/custom_dse.py"))


def test_h200_topology_exposes_the_four_node_eight_gpu_layout():
    assert noc_config_set.h200x32 is h200x32
    topology = h200x32()
    assert topology.num_devices == 32
    outer, inter_node, intra_node = topology.layers
    assert [layer.shape for layer in topology.layers] == [(1, 1), (1, 4), (1, 8)]
    assert intra_node.link_bandwidth == 370e9
    assert intra_node.hop_latency == pytest.approx(1.25e-6)
    assert inter_node.link_bandwidth == 50e9
    assert inter_node.hop_latency == pytest.approx(5e-6)
    assert outer.hop_latency == 0
    changed = h200x32(ib_bw=25e9)
    assert changed.layers[1].link_bandwidth == 25e9
    assert changed.layers[1].switch_center_in_bw == 25e9 * 4


def test_decode_example_uses_the_driver_microbatch_convention():
    row = example["evaluate"](tp=8, pp=4, dp=1, batch_size=64)
    assert row["status"] == "OK"
    assert row["microbatch"] == 16
    assert 0 < row["memory_gib"] < 141
    assert row["stps"] == pytest.approx(16 / (row["latency_ms"] / 1000))
    with pytest.raises(ValueError, match=r"TP \* PP \* DP"):
        example["evaluate"](tp=8, pp=4, dp=2)


def test_example_reports_oom_before_modeling_an_impossible_batch():
    row = example["evaluate"](batch_size=1_000_000)
    assert row["status"] == "OOM"
    assert row["memory_gib"] > 141
    assert row["latency_ms"] is None and row["stps"] is None


def test_dse_scans_distinct_fixed_budget_candidates_and_ranks_them():
    rows = example["search"]()
    assert len(rows) == 15
    assert len({(row["tp"], row["pp"], row["dp"]) for row in rows}) == 15
    assert all(row["tp"] * row["pp"] * row["dp"] == 32 for row in rows)
    feasible = [row for row in rows if row["status"] == "OK"]
    assert feasible
    assert all(math.isfinite(row["stps"]) and row["stps"] > 0 for row in feasible)
    assert [row["stps"] for row in feasible] == sorted(
        [row["stps"] for row in feasible], reverse=True
    )
