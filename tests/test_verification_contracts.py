from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pandas as pd
import pytest

from ae.verification import (
    _CsvContract,
    _EXPECTED_PLOT_STEMS,
    _MODEL_VERSIONS,
    _STAGE_CSV_CONTRACTS,
    _read_csv_contract,
    verify_stage_outputs,
    verify_workflow_manifest,
)
from ae.workflow import reproduce_stages


EXPECTED_PLOT_STEMS = (
    "fig08_modeling_accuracy_h100",
    "fig09_b200_decode_validation",
    "fig10_mi325x_decode_validation",
    "fig12_deepseek_v32_dsa",
    "fig13_noc_model_validation",
    "fig15_decode_pareto",
    "fig16_3d_vs_2_5d",
    "fig17_dram_bandwidth",
    "fig18_dram_layer_throughput",
    "fig19_prefill_decode_dse_heatmaps",
    "fig20_dram_thermal_terrain",
    "fig21_noc_bw_latency_sensitivity",
    "fig22_noc_layer_decode",
    "fig22_noc_layer_prefill",
    "fig23_parallelism_search_space",
    "table04_ablation",
)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _synthetic_frame(contract: _CsvContract) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            (
                *contract.required,
                *contract.unique,
                *contract.finite,
                *contract.zero,
                *contract.false,
                *(column for column, _ in contract.exact_values),
            )
        )
    )
    frame = pd.DataFrame(
        {column: [f"{column}-value"] * contract.rows for column in columns}
    )
    exact_columns = {column for column, _ in contract.exact_values}
    for column in contract.finite:
        frame[column] = list(range(1, contract.rows + 1))
    for column in contract.zero:
        frame[column] = 0
    for column in contract.false:
        frame[column] = False
    for column in contract.unique:
        if column in exact_columns or column in contract.zero or column in contract.false:
            continue
        if column not in contract.finite:
            frame[column] = [f"{column}-{index}" for index in range(contract.rows)]
    for column, values in contract.exact_values:
        frame[column] = [values[index % len(values)] for index in range(contract.rows)]
    if "status" in frame and "status" not in exact_columns:
        frame["status"] = "PASS"
    return frame


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _synthetic_png() -> bytes:
    header = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", 640, 480)
    )
    return header + b"\x00" * (1_024 - len(header))


def _synthetic_pdf() -> bytes:
    header = b"%PDF-1.7\n"
    return header + b"%" * (640 - len(header))


def _build_synthetic_outputs(tmp_path: Path) -> tuple[Path, Path]:
    results_dir = tmp_path / "results"
    result_root = results_dir / "reproduce"
    frames: dict[str, pd.DataFrame] = {}
    stages: dict[str, str] = {}
    for stage, stage_contracts in _STAGE_CSV_CONTRACTS:
        for contract in stage_contracts:
            frame = _synthetic_frame(contract)
            frames[contract.path] = frame
            stages[contract.path] = stage

    # Cross-file identities and fixed cardinalities checked by the verifier.
    for version in _MODEL_VERSIONS:
        fig12 = frames[f"fig12/{version}/model_points.csv"]
        fig12["point_id"] = [f"fig12-{index}" for index in range(40)]

        fig15 = frames[f"fig15/{version}/fig15_pareto_candidates.csv"]
        fig15["candidate_id"] = [f"fig15-{index}" for index in range(784)]
        fig15["model"] = [f"model-{index % 4}" for index in range(784)]
        batch_sizes = (1, 4, 16, 64, 128, 256, 1024)
        fig15["bs"] = [batch_sizes[(index // 4) % 7] for index in range(784)]

        fig23 = frames[f"fig23/{version}/raw.csv"]
        fig23["evaluation_key"] = [f"eval-{index % 75}" for index in range(84)]

    fig17 = frames["fig17/summary.csv"]
    fig17["scope"] = ["global", *("panel" for _ in range(6))]

    table4 = frames["table4/model_points.csv"]
    table4["candidate_id"] = [f"table4-{index // 2}" for index in range(28)]
    table4["model_version"] = [
        _MODEL_VERSIONS[index % 2] for index in range(28)
    ]

    terrain_strata = [
        (bs, layer) for bs in (4, 1024) for layer in range(2, 13)
    ]
    terrain = frames["fig18_20/archive/fig20_decode_thermal_terrain.csv"]
    terrain["terrain_reference_id"] = [
        f"terrain-{index:06d}"
        for index in range(len(terrain))
    ]
    terrain["bs"] = [
        terrain_strata[index % len(terrain_strata)][0]
        for index in range(len(terrain))
    ]
    terrain["dram_total_layers"] = [
        terrain_strata[index % len(terrain_strata)][1]
        for index in range(len(terrain))
    ]
    terrain["dram_active_layers"] = terrain["dram_total_layers"]

    for version in _MODEL_VERSIONS:
        points = frames[f"fig18_20/{version}/model_points.csv"]
        points["terrain_reference_id"] = [
            f"terrain-{index:06d}" if index < 22 else ""
            for index in range(len(points))
        ]
        points["selection_roles"] = "synthetic_non_sample"
        points.loc[:21, "selection_roles"] = "fig20_archive_sample"

        samples = frames[f"fig18_20/{version}/terrain_sample_validation.csv"]
        for index, (bs, layer) in enumerate(terrain_strata):
            raw_stps = float(terrain.loc[index, "raw_stps"])
            terrain.loc[index, "temperature_C"] = 50.0
            candidate_id = str(points.loc[index, "candidate_id"])
            terrain_reference_id = str(
                points.loc[index, "terrain_reference_id"]
            )
            for frame in (points, samples):
                frame.loc[index, "candidate_id"] = candidate_id
                frame.loc[index, "terrain_reference_id"] = (
                    terrain_reference_id
                )
                frame.loc[index, "bs"] = bs
                frame.loc[index, "dram_total_layers"] = layer
                frame.loc[index, "dram_active_layers"] = layer
            samples.loc[index, "archived_raw_stps"] = raw_stps
            samples.loc[index, "recomputed_raw_stps"] = raw_stps
            samples.loc[index, "archived_temperature_C"] = 50.0
            samples.loc[index, "recomputed_temperature_C"] = 50.0
            samples.loc[index, "raw_stps_error_pct"] = 0.0
            samples.loc[index, "temperature_error_C"] = 0.0

    png_data = _synthetic_png()
    pdf_data = _synthetic_pdf()
    frames["figures/plot_manifest.csv"] = pd.DataFrame(
        [
            {
                "figure": f"reproduced plot {index + 1}",
                "stem": stem,
                "model_version": "paper_legacy",
                "source_files": "stages/fig08/model_points.csv",
                "png": f"{stem}.png",
                "pdf": f"{stem}.pdf",
                "png_bytes": len(png_data),
                "pdf_bytes": len(pdf_data),
                "status": "PASS",
                "note": "synthetic verifier fixture",
            }
            for index, stem in enumerate(EXPECTED_PLOT_STEMS)
        ]
    )

    for path_string, frame in frames.items():
        base = (
            result_root
            if stages[path_string] == "figures"
            else result_root / "stages"
        )
        _write_csv(base / path_string, frame)

    archive_manifest = frames["fig18_20/archive/archive_manifest.csv"].copy()
    archive_files = (
        "fig18_throughput_curves.csv",
        "fig19_metric_grid.csv",
        "fig20_decode_thermal_terrain.csv",
    )
    for index, filename in enumerate(archive_files):
        relative = f"fig18_20/archive/{filename}"
        archive_path = result_root / "stages" / relative
        archive_manifest.loc[index, "dataset"] = filename.removesuffix(".csv")
        archive_manifest.loc[index, "file"] = filename
        archive_manifest.loc[index, "rows"] = len(frames[relative])
        archive_manifest.loc[index, "sha256"] = hashlib.sha256(
            archive_path.read_bytes()
        ).hexdigest()
    _write_csv(
        result_root / "stages/fig18_20/archive/archive_manifest.csv",
        archive_manifest,
    )

    figures_dir = result_root / "figures"
    stage_root = result_root / "stages"
    for stem in EXPECTED_PLOT_STEMS:
        (figures_dir / f"{stem}.png").write_bytes(png_data)
        (figures_dir / f"{stem}.pdf").write_bytes(pdf_data)

    _write_json(
        stage_root / "fig08/verification.json",
        {"status": "PASS", "points": 52, "gpu_execution": False},
    )
    _write_json(
        stage_root / "fig09/PASS",
        {
            "status": "PASS",
            "model_points": 40,
            "model_evaluations": 80,
            "gpu_runs_performed": 0,
        },
    )
    _write_json(
        stage_root / "fig10/verification.json",
        {
            "status": "PASS",
            "points": 28,
            "cpu_model_evaluations": 56,
            "gpu_runs_executed": 0,
        },
    )
    _write_text(
        stage_root / "fig10/PASS.txt",
        "PASS\nCPU model evaluations: 56\nGPU runs executed: 0\n",
    )
    _write_text(
        stage_root / "fig12/PASS",
        "PASS\npoints_per_path=40\ngpu_runs=0\n",
    )
    for version in _MODEL_VERSIONS:
        status = {
            "status": "PASS",
            "model_version": version,
            "fixed_configurations": 114,
            "gpu_required": False,
            "search_performed": False,
        }
        _write_json(stage_root / f"fig18_20/{version}/status.json", status)
        _write_json(stage_root / f"fig18_20/{version}/PASS", status)
        _write_text(
            stage_root / f"fig22/{version}/PASS",
            "PASS\n"
            f"model_version={version}\n"
            "fixed_points=756\nmodel_calls=1930\ngpu_runs=0\n",
        )
        _write_text(
            stage_root / f"fig23/{version}/PASS",
            "PASS\n"
            f"model_version={version}\n"
            "curve_points=84\nunique_fixed_configurations=75\n"
            "paper_headline_claims=PASS\n"
            "paper_nested_search_space_trend=PASS\n",
        )
    _write_text(
        stage_root / "fig22/PASS",
        "PASS\nmodel_paths=paper_legacy,current_corrected\n"
        "fixed_points_per_path=756\ngpu_runs=0\n",
    )
    _write_text(
        stage_root / "table4/PASS",
        "Table 4 selected-configuration verification: PASS\n"
        "model_version=both\nfixed_configs=14\nevaluated_points=28\n",
    )
    _write_text(
        result_root / "figures/PASS",
        "PASS\nplots=16\npng_files=16\npdf_files=16\n"
        "model_version=paper_legacy\n",
    )
    return result_root, results_dir


def test_complete_stage_output_contracts(tmp_path: Path) -> None:
    result_root, _results_dir = _build_synthetic_outputs(tmp_path)
    report = verify_stage_outputs(result_root)
    assert report["stages"] == 15
    assert report["csv_files"] == 57
    assert report["marker_files"] == 16
    assert report["plot_files"] == 32
    assert _EXPECTED_PLOT_STEMS == set(EXPECTED_PLOT_STEMS)


def test_stage_outputs_ignore_poisoned_legacy_sibling(tmp_path: Path) -> None:
    result_root, results_dir = _build_synthetic_outputs(tmp_path)
    _write_text(results_dir / "fig10/model_points.csv", "poisoned,legacy\n1,2\n")
    _write_json(
        results_dir / "fig10/verification.json",
        {"status": "FAIL", "points": 0, "gpu_execution": True},
    )

    report = verify_stage_outputs(result_root)
    assert report["stages"] == 15


def test_fig18_20_archive_manifest_rejects_tampering(tmp_path: Path) -> None:
    result_root, _results_dir = _build_synthetic_outputs(tmp_path)
    path = result_root / "stages/fig18_20/archive/fig18_throughput_curves.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "best_scaled_stps"] = (
        float(frame.loc[0, "best_scaled_stps"]) + 1.0
    )
    _write_csv(path, frame)

    with pytest.raises(AssertionError, match="archive manifest checksum mismatch"):
        verify_stage_outputs(result_root)


def test_fig20_legacy_sample_rejects_raw_error(tmp_path: Path) -> None:
    result_root, _results_dir = _build_synthetic_outputs(tmp_path)
    path = (
        result_root
        / "stages/fig18_20/paper_legacy/terrain_sample_validation.csv"
    )
    frame = pd.read_csv(path)
    frame.loc[0, "raw_stps_error_pct"] = 1.0
    _write_csv(path, frame)

    with pytest.raises(AssertionError, match="raw_stps_error_pct exceeds tolerance"):
        verify_stage_outputs(result_root)


def test_fig18_20_rejects_published_power_intermediate(tmp_path: Path) -> None:
    result_root, _results_dir = _build_synthetic_outputs(tmp_path)
    path = (
        result_root
        / "stages/fig18_20/paper_legacy/model_points.csv"
    )
    frame = pd.read_csv(path)
    frame["total_power_per_device_W"] = 100.0
    _write_csv(path, frame)

    with pytest.raises(AssertionError, match="non-publishable intermediates"):
        verify_stage_outputs(result_root)


@pytest.mark.parametrize("filename", ("model_points.csv", "summary.csv"))
def test_table4_rejects_non_publishable_intermediates(
    tmp_path: Path, filename: str
) -> None:
    result_root, _results_dir = _build_synthetic_outputs(tmp_path)
    path = result_root / "stages/table4" / filename
    frame = pd.read_csv(path)
    for column in (
        "raw_stps_avg",
        "raw_utps_avg",
        "chip_power_W",
        "total_power_per_device_W",
        "tokens_per_joule",
        "freq_scale_power",
        "hit_power_wall",
        "stats_run_id",
    ):
        frame[column] = 1.0
    _write_csv(path, frame)

    with pytest.raises(AssertionError, match="publish-safe schema"):
        verify_stage_outputs(result_root)


def test_legacy_sibling_cannot_fill_missing_scoped_output(tmp_path: Path) -> None:
    result_root, results_dir = _build_synthetic_outputs(tmp_path)
    scoped = result_root / "stages/fig10/model_points.csv"
    legacy = results_dir / "fig10/model_points.csv"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(scoped.read_bytes())
    scoped.unlink()

    with pytest.raises(AssertionError, match="fig10/model_points.csv"):
        verify_stage_outputs(result_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate unique key"),
        ("infinite", "non-finite"),
        ("gpu", "numeric zero"),
        ("version", "expected model_version values"),
        ("summary", "summary contains FAIL"),
    ],
)
def test_csv_contract_rejects_corrupt_outputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    contract = _CsvContract(
        "sample.csv",
        2,
        ("id", "value", "gpu_runs", "model_version", "status"),
        unique=("id",),
        finite=("value",),
        zero=("gpu_runs",),
        exact_values=(("model_version", ("paper_legacy",)),),
        reject_fail_status=True,
    )
    frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "value": [1.0, 2.0],
            "gpu_runs": [0, 0],
            "model_version": ["paper_legacy", "paper_legacy"],
            "status": ["PASS", "INFO"],
        }
    )
    if mutation == "duplicate":
        frame.loc[1, "id"] = "a"
    elif mutation == "infinite":
        frame.loc[1, "value"] = float("inf")
    elif mutation == "gpu":
        frame.loc[1, "gpu_runs"] = 1
    elif mutation == "version":
        frame.loc[1, "model_version"] = "current_corrected"
    elif mutation == "summary":
        frame.loc[1, "status"] = "FAIL"
    path = tmp_path / contract.path
    _write_csv(path, frame)
    with pytest.raises(AssertionError, match=message):
        _read_csv_contract(path, contract, scope_root=tmp_path)


def test_csv_contract_rejects_columns_outside_allowed_schema(
    tmp_path: Path,
) -> None:
    contract = _CsvContract(
        "publish_safe.csv",
        1,
        ("selector", "displayed_value"),
        allowed=("selector", "displayed_value"),
    )
    path = tmp_path / contract.path
    _write_csv(
        path,
        pd.DataFrame(
            {
                "selector": ["case-0"],
                "displayed_value": [1.0],
                "calibration_intermediate": [2.0],
            }
        ),
    )

    with pytest.raises(AssertionError, match="publish-safe schema"):
        _read_csv_contract(path, contract, scope_root=tmp_path)


@pytest.mark.parametrize(
    ("contract_path", "leaked_column"),
    (
        ("fig21/paper_legacy/raw.csv", "chip_power_per_device_w"),
        ("fig22/paper_legacy/model_points.csv", "logic_raw_stps"),
        ("fig22/paper_legacy/normalized_summary.csv", "logic_chip_power_w"),
        ("fig22/paper_legacy/summary.csv", "absolute_power_w"),
        (
            "fig22/version_drift.csv",
            "reproduced_logic_stps_paper_legacy",
        ),
    ),
)
def test_fig21_fig22_contracts_reject_nonpublished_columns(
    tmp_path: Path,
    contract_path: str,
    leaked_column: str,
) -> None:
    contract = next(
        contract
        for _stage, contracts in _STAGE_CSV_CONTRACTS
        for contract in contracts
        if contract.path == contract_path
    )
    assert contract.allowed
    assert set(contract.allowed) == set(contract.required)
    frame = _synthetic_frame(contract)
    frame[leaked_column] = 1.0
    path = tmp_path / contract.path
    _write_csv(path, frame)

    with pytest.raises(AssertionError, match="publish-safe schema"):
        _read_csv_contract(path, contract, scope_root=tmp_path)


def test_stage_output_contract_rejects_external_csv_symlink(tmp_path: Path) -> None:
    result_root, _results_dir = _build_synthetic_outputs(tmp_path)
    scoped = result_root / "stages/fig10/model_points.csv"
    outside = tmp_path / "outside.csv"
    outside.write_bytes(scoped.read_bytes())
    scoped.unlink()
    scoped.symlink_to(outside)

    with pytest.raises(AssertionError, match="symbolic link"):
        verify_stage_outputs(result_root)


def test_stage_output_contract_rejects_unexpected_file(tmp_path: Path) -> None:
    result_root, _results_dir = _build_synthetic_outputs(tmp_path)
    _write_text(result_root / "stages/fig15/stale.csv", "stale\n1\n")

    with pytest.raises(AssertionError, match="stage file closure mismatch"):
        verify_stage_outputs(result_root)


def test_stage_output_contract_rejects_marker_content(tmp_path: Path) -> None:
    result_root, _results_dir = _build_synthetic_outputs(tmp_path)
    marker = result_root / "stages/fig10/verification.json"
    value = json.loads(marker.read_text(encoding="utf-8"))
    value["status"] = "FAIL"
    _write_json(marker, value)
    with pytest.raises(AssertionError, match="expected status='PASS'"):
        verify_stage_outputs(result_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("stem", "plot stem closure mismatch"),
        ("traversal", "must be the basename"),
        ("size", "manifest records"),
        ("small", "implausibly small PNG"),
        ("png_signature", "invalid PNG signature"),
        ("png_dimensions", "implausible PNG dimensions"),
        ("pdf_header", "invalid PDF header"),
        ("source", "scope-relative stages"),
        ("extra", "figure file closure mismatch"),
        ("symlink", "must not be a symbolic link"),
    ],
)
def test_plot_contract_rejects_corrupt_outputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    result_root, _results_dir = _build_synthetic_outputs(tmp_path)
    figures_dir = result_root / "figures"
    manifest_path = figures_dir / "plot_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    stem = str(manifest.loc[0, "stem"])
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"

    if mutation == "stem":
        manifest.loc[0, "stem"] = "unexpected_plot"
    elif mutation == "traversal":
        manifest.loc[0, "png"] = f"../{stem}.png"
    elif mutation == "size":
        manifest.loc[0, "png_bytes"] = int(manifest.loc[0, "png_bytes"]) + 1
    elif mutation == "small":
        png_path.write_bytes(png_path.read_bytes()[:100])
        manifest.loc[0, "png_bytes"] = 100
    elif mutation == "png_signature":
        data = bytearray(png_path.read_bytes())
        data[:8] = b"BADPNG!!"
        png_path.write_bytes(data)
    elif mutation == "png_dimensions":
        data = bytearray(png_path.read_bytes())
        data[16:24] = struct.pack(">II", 1, 1)
        png_path.write_bytes(data)
    elif mutation == "pdf_header":
        data = bytearray(pdf_path.read_bytes())
        data[:4] = b"NOPE"
        pdf_path.write_bytes(data)
    elif mutation == "source":
        manifest.loc[0, "source_files"] = "/tmp/external.csv"
    elif mutation == "extra":
        (figures_dir / "stale.png").write_bytes(_synthetic_png())
    elif mutation == "symlink":
        outside = tmp_path / "outside.png"
        outside.write_bytes(_synthetic_png())
        png_path.unlink()
        png_path.symlink_to(outside)
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(f"unknown mutation: {mutation}")

    _write_csv(manifest_path, manifest)
    with pytest.raises(AssertionError, match=message):
        verify_stage_outputs(result_root)


def _write_workflow_manifest(result_root: Path, stage_names: list[str]) -> None:
    rows: list[dict[str, object]] = []
    for index, stage in enumerate(stage_names):
        log = result_root / "logs" / f"{index:02d}_{stage}.log"
        _write_text(log, "PASS\n")
        rows.append(
            {
                "stage": stage,
                "status": "PASS",
                "return_code": 0,
                "wall_time_s": 1.0,
                "gpu_runs": 0,
                "command": f"python -m ae.{stage}",
                "note": "test",
                "log": log.relative_to(result_root).as_posix(),
            }
        )
    _write_csv(result_root / "run_manifest.csv", pd.DataFrame(rows))


def test_workflow_manifest_requires_exact_stage_order(tmp_path: Path) -> None:
    result_root = tmp_path / "results/reproduce"
    expected = [stage.name for stage in reproduce_stages(1, "both")]
    _write_workflow_manifest(result_root, expected)
    assert verify_workflow_manifest(result_root)["stages"] == 15

    reordered = expected.copy()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    _write_workflow_manifest(result_root, reordered)
    with pytest.raises(AssertionError, match="stage order/closure mismatch"):
        verify_workflow_manifest(result_root)


@pytest.mark.parametrize("bad_log", ["../outside.log", "/tmp/outside.log"])
def test_workflow_manifest_rejects_logs_outside_scope(
    tmp_path: Path, bad_log: str
) -> None:
    result_root = tmp_path / "results/reproduce"
    expected = [stage.name for stage in reproduce_stages(1, "both")]
    _write_workflow_manifest(result_root, expected)
    manifest_path = result_root / "run_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest.loc[0, "log"] = bad_log
    _write_csv(manifest_path, manifest)

    with pytest.raises(AssertionError, match="stage log (escapes|must be)"):
        verify_workflow_manifest(result_root)
