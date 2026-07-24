import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest

from bank_oracle.gemm import (
    GemmBankOptions,
    GemmProblem,
    GemmTiling,
    make_layout_set,
    model_gemm_bank,
)
from bank_oracle.layout import BankSwizzle
from bank_oracle.spec import DramBankSpec
from bank_oracle.visualization import (
    RequestGroup,
    build_bank_wave_snapshot,
    build_demo_dashboard,
    build_gemm_wave_snapshot,
    build_logical_matrix_maps,
    render_dashboard_html,
)


def _snapshot(sectors, swizzle=BankSwizzle(), spec=None):
    return build_bank_wave_snapshot(
        "wave",
        "wave",
        (RequestGroup("trace", (np.asarray(sectors, dtype=np.int64),), swizzle),),
        spec=DramBankSpec() if spec is None else spec,
    )


def _bank(snapshot, bank):
    return snapshot["banks"][bank]


def _expand_runs(flat_runs):
    return np.asarray(
        [
            sector
            for start, length in zip(flat_runs[0::2], flat_runs[1::2])
            for sector in range(start, start + length)
        ],
        dtype=np.int64,
    )


def test_contiguous_sector_visualization_explains_parallelism_tradeoff():
    spec = DramBankSpec()
    sectors = np.arange(spec.bank_count)
    linear = _snapshot(sectors, spec=spec)
    bank_bits = spec.bank_count.bit_length() - 1
    sector_bits = min(
        bank_bits,
        spec.sectors_per_row.bit_length() - 1,
    )
    xor = _snapshot(
        sectors,
        BankSwizzle(kind="xor", sector_bits=sector_bits),
        spec,
    )

    assert linear["summary"]["cycles"] == spec.full_row_cycles
    assert linear["summary"]["active_banks"] == 1
    assert (
        linear["summary"]["fixed_job_lower_bound_cycles"]
        == spec.full_row_cycles
    )
    expected_lower_bound = spec.recharge_cycles + spec.sector_cycles
    assert (
        linear["summary"]["sector_lower_bound_cycles"]
        == expected_lower_bound
    )
    assert not linear["summary"]["avoidable_fixed_job_conflict"]
    assert linear["summary"]["layout_packing_headroom"]
    assert _bank(linear, 0)["row_job_count"] == 1
    assert _bank(linear, 0)["sector_count"] == spec.sectors_per_row
    assert _bank(linear, 0)["total_cycles"] == spec.full_row_cycles

    assert xor["summary"]["cycles"] == expected_lower_bound
    assert xor["summary"]["active_banks"] == spec.bank_count
    assert (
        xor["summary"]["fixed_job_lower_bound_cycles"]
        == expected_lower_bound
    )
    assert xor["summary"]["sector_lower_bound_cycles"] == expected_lower_bound
    for bank in xor["banks"]:
        assert bank["row_job_count"] == 1
        assert bank["sector_count"] == 1
        assert bank["total_cycles"] == expected_lower_bound


def test_colliding_full_rows_visualization_is_a_true_bank_conflict():
    spec = DramBankSpec()
    rows = np.concatenate(
        [
            np.arange(
                row * spec.bank_count * spec.sectors_per_row,
                row * spec.bank_count * spec.sectors_per_row
                + spec.sectors_per_row,
            )
            for row in range(spec.bank_count)
        ]
    )
    linear = _snapshot(rows, spec=spec)
    xor = _snapshot(
        rows,
        BankSwizzle(
            kind="xor",
            row_bits=spec.bank_count.bit_length() - 1,
        ),
        spec,
    )

    assert (
        linear["summary"]["sector_count"]
        == spec.bank_count * spec.sectors_per_row
    )
    assert (
        linear["summary"]["cycles"]
        == spec.bank_count * spec.full_row_cycles
    )
    assert (
        linear["summary"]["fixed_job_lower_bound_cycles"]
        == spec.full_row_cycles
    )
    assert linear["summary"]["avoidable_fixed_job_conflict"]
    assert _bank(linear, 0)["row_job_count"] == spec.bank_count
    assert (
        _bank(linear, 0)["sector_count"]
        == spec.bank_count * spec.sectors_per_row
    )
    assert all(
        _bank(linear, bank)["row_job_count"] == 0
        for bank in range(1, spec.bank_count)
    )

    assert xor["summary"]["cycles"] == spec.full_row_cycles
    assert xor["summary"]["active_banks"] == spec.bank_count
    assert not xor["summary"]["avoidable_fixed_job_conflict"]
    for bank in xor["banks"]:
        assert bank["row_job_count"] == 1
        assert bank["sector_count"] == spec.sectors_per_row
        assert bank["total_cycles"] == spec.full_row_cycles


@pytest.mark.parametrize("swizzle_kind", ["none", "row_xor"])
def test_serialized_jobs_and_bank_summaries_are_internally_exact(swizzle_kind):
    dram = DramBankSpec()
    swizzle = (
        BankSwizzle()
        if swizzle_kind == "none"
        else BankSwizzle(
            kind="xor",
            row_bits=dram.bank_count.bit_length() - 1,
        )
    )
    snapshot = _snapshot(
        np.arange(dram.bank_count * dram.sectors_per_row),
        swizzle,
        dram,
    )
    spec = snapshot["spec"]
    assert sum(bank["sector_count"] for bank in snapshot["banks"]) == snapshot["summary"]["sector_count"]
    assert sum(bank["row_job_count"] for bank in snapshot["banks"]) == snapshot["summary"]["row_job_count"]
    assert sum(bank["row_job_count"] > 0 for bank in snapshot["banks"]) == snapshot["summary"]["active_banks"]
    for job in snapshot["jobs"]:
        assert job["total_cycles"] == (
            spec["sector_cycles"] * job["sector_count"]
            + spec["recharge_cycles"]
        )
    for bank in snapshot["banks"]:
        assert bank["total_cycles"] == bank["data_cycles"] + bank["recharge_cycles"]


def test_selected_gemm_load_and_store_snapshots_match_model_profiles():
    spec = DramBankSpec()
    problem = GemmProblem(64, 128, 64)
    tiling = GemmTiling(32, 64, 32, stage=2, ctas_per_wave=2, row_panel=1)
    layouts = make_layout_set(
        problem,
        tiling,
        spec,
        preset="tile_major",
        swizzle=BankSwizzle(
            kind="xor",
            sector_bits=4,
            sector_bank_shift=(
                spec.bank_count.bit_length() - 1 - 4
            ),
        ),
    )
    options = GemmBankOptions(max_spatial_samples=None, max_k_samples=None)
    result = model_gemm_bank(problem, tiling, layouts, spec, options)

    load = build_gemm_wave_snapshot(
        problem,
        tiling,
        layouts,
        spatial_wave_index=0,
        k_window_index=0,
        spec=spec,
        options=options,
    )
    load_profile = next(
        profile
        for profile in result.sampled_profiles
        if not profile.is_store
        and profile.spatial_wave_index == 0
        and profile.k_window_index == 0
    )
    assert load["summary"]["cycles"] == load_profile.bank_cycles
    assert load["summary"]["transferred_bytes"] == load_profile.transferred_bytes
    assert load["summary"]["active_banks"] == load_profile.active_banks

    store = build_gemm_wave_snapshot(
        problem,
        tiling,
        layouts,
        spatial_wave_index=0,
        k_window_index=None,
        store=True,
        spec=spec,
        options=options,
    )
    store_profile = next(
        profile
        for profile in result.sampled_profiles
        if profile.is_store and profile.spatial_wave_index == 0
    )
    assert store["summary"]["cycles"] == store_profile.bank_cycles
    assert store["summary"]["transferred_bytes"] == store_profile.transferred_bytes
    assert store["summary"]["active_banks"] == store_profile.active_banks


def test_gemm_wave_visualization_rejects_implicit_l2_state():
    from bank_oracle.l2_cache import L2CacheSpec

    spec = DramBankSpec()
    problem = GemmProblem(32, 32, 32)
    tiling = GemmTiling(32, 32, 32)
    layouts = make_layout_set(problem, tiling, spec)
    with pytest.raises(ValueError, match="l2_cache=None"):
        build_gemm_wave_snapshot(
            problem,
            tiling,
            layouts,
            spatial_wave_index=0,
            spec=spec,
            options=GemmBankOptions(l2_cache=L2CacheSpec(1 << 20)),
        )


def test_html_is_self_contained_deterministic_and_payload_safe():
    hostile = "</script><img src=x onerror=alert(1)>&\u2028"
    data = {"title": hostile, "subtitle": "offline", "scenarios": []}
    first = render_dashboard_html(data)
    second = render_dashboard_html(data)
    assert first == second
    assert hostile not in first
    assert not re.search(r"<script\s+[^>]*src=", first, re.IGNORECASE)
    assert "fetch(" not in first

    match = re.search(
        r'<script id="bank-wave-payload" type="application/octet-stream">([^<]+)</script>',
        first,
    )
    assert match is not None
    recovered = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))
    assert recovered == data


def test_access_run_encoding_is_lossless_per_request():
    requests = (
        np.asarray([0, 1, 2, 7, 8, 8, 31]),
        np.asarray([4, 5, 20, 21, 22]),
    )
    snapshot = build_bank_wave_snapshot(
        "rle",
        "rle",
        (
            RequestGroup(
                "requests",
                requests,
                BankSwizzle(kind="cyclic", cyclic_sector_alpha=1, cyclic_row_beta=3),
                ("first", "second"),
            ),
        ),
        connectivity="interleaved",
        coalesce_scope="request",
    )
    assert "accesses" not in snapshot
    for request_id, expected in enumerate(requests):
        runs = [
            run
            for encoded in snapshot["access_runs"]
            if encoded[0] == request_id
            for run in (encoded[3], encoded[4])
        ]
        assert np.array_equal(_expand_runs(runs), np.unique(expected))


def test_logical_matrix_maps_preserve_exact_a_and_b_tile_sector_sets():
    from bank_oracle.layout import MatrixAccess

    spec = DramBankSpec()
    problem = GemmProblem(65, 130, 67)
    tiling = GemmTiling(32, 64, 32, stage=2, ctas_per_wave=2)
    layouts = make_layout_set(problem, tiling, spec, preset="tile_major")
    maps = build_logical_matrix_maps(problem, tiling, layouts, spec)

    assert maps["A"]["grid_shape"] == [3, 3]
    assert maps["B"]["grid_shape"] == [3, 3]
    assert maps["A"]["dtype_bytes"] == problem.a_dtype_bytes
    assert maps["B"]["dtype_bytes"] == problem.b_dtype_bytes
    # A block (2, 2) and B block (2, 2) are ragged on both dimensions.
    a_index = 2 * 3 + 2
    b_index = 2 * 3 + 2
    expected_a = layouts.a.touched_sectors(
        problem.a_shape,
        MatrixAccess(64, 65, 64, 67),
        problem.a_dtype_bytes,
        spec.sector_bytes,
    )
    expected_b = layouts.b.touched_sectors(
        problem.b_shape,
        MatrixAccess(64, 67, 128, 130),
        problem.b_dtype_bytes,
        spec.sector_bytes,
    )
    assert np.array_equal(_expand_runs(maps["A"]["blocks"][a_index]), expected_a)
    assert np.array_equal(_expand_runs(maps["B"]["blocks"][b_index]), expected_b)


def test_bundled_preview_and_generated_catalog_are_synthetic():
    artifact = (
        Path(__file__).parents[1]
        / "visualization"
        / "artifacts"
        / "bank_wave_demo.html"
    )
    html = artifact.read_text(encoding="utf-8")
    assert "Synthetic bank-wave witness" in html
    assert "not a hardware calibration" in html
    assert not re.search(r"<script\s+[^>]*src=", html, re.IGNORECASE)
    assert "fetch(" not in html
    assert artifact.stat().st_size < 256 * 1024

    dashboard = build_demo_dashboard()

    expected_cases = {
        "synthetic-ragged-gemm": ([73, 149, 211], [32, 64, 32], 3),
    }
    scenarios = {item["scenario_id"]: item for item in dashboard["scenarios"]}
    assert expected_cases.keys() <= scenarios.keys()
    assert "Synthetic" in dashboard["title"]
    assert "synthetic-sector-spread" in scenarios
    assert "synthetic-row-balance" in scenarios
    all_wave_ids = []
    for scenario_id, (problem, tile, stage) in expected_cases.items():
        scenario = scenarios[scenario_id]
        assert scenario["case_spec"]["problem"] == problem
        assert scenario["case_spec"]["tile"] == tile
        assert scenario["case_spec"]["stage"] == stage
        assert set(scenario["matrix_maps"]) == {"A", "B"}
        for matrix_map in scenario["matrix_maps"].values():
            assert len(matrix_map["blocks"]) == np.prod(matrix_map["grid_shape"])
            assert all(len(block) % 2 == 0 for block in matrix_map["blocks"])

        reference_keys = None
        for variant in scenario["variants"]:
            keys = [
                (
                    wave["context"]["kind"],
                    wave["context"]["spatial_wave_index"],
                    wave["context"].get("k_window_index"),
                )
                for wave in variant["waves"]
            ]
            if reference_keys is None:
                reference_keys = keys
            else:
                assert keys == reference_keys
            all_wave_ids.extend(wave["wave_id"] for wave in variant["waves"])
    assert len(all_wave_ids) == len(set(all_wave_ids))


def test_dashboard_accepts_non_power_of_two_custom_geometry():
    spec = DramBankSpec(
        total_layers=3,
        connected_layers=2,
        banks_per_layer=5,
        row_bytes=7 * 96,
        sector_bytes=96,
        sector_cycles=5,
        recharge_cycles=9,
    )
    dashboard = build_demo_dashboard(include_gemm=False, spec=spec)
    assert "3 layers" in dashboard["subtitle"]
    scenarios = {
        item["scenario_id"]: item
        for item in dashboard["scenarios"]
    }
    for scenario in scenarios.values():
        assert {
            variant["waves"][0]["spec"]["sectors_per_row"]
            for variant in scenario["variants"]
        } == {7}
        assert any(
            variant["waves"][0]["groups"][0]["swizzle"]["kind"]
            == "cyclic"
            for variant in scenario["variants"]
        )
