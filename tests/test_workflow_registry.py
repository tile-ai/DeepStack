from __future__ import annotations

import importlib.util

import pytest

from ae.cli import build_parser, resolve_result_dir
from ae.paths import ROOT
from ae.workflow import _command, _display_command, quick_stages, reproduce_stages


def test_default_workflow_registry_is_closed_and_importable() -> None:
    stages = reproduce_stages(32, "both")
    names = [stage.name for stage in stages]
    assert len(names) == len(set(names)) == 15
    assert names[0] == "fig08"
    assert names[-1] == "figures"
    for stage in stages:
        assert importlib.util.find_spec(stage.module) is not None, stage.module


def test_quick_workflow_is_a_strict_short_subset() -> None:
    quick = quick_stages(8, "both")
    assert [stage.name for stage in quick] == [
        "fig08",
        "fig13",
        "fig17",
        "fig15_quick",
    ]
    fig15 = quick[-1]
    assert "full recorded decode DSE" in fig15.note


def test_every_workflow_stage_is_scoped_to_one_result_tree() -> None:
    result_root = ROOT / "results" / "isolated-review"
    for stage in (*quick_stages(8, "both"), *reproduce_stages(32, "both")):
        output = stage.output_path(result_root).resolve()
        assert output == result_root.resolve() or result_root.resolve() in output.parents
        if stage.name == "figures":
            assert output == result_root.resolve()
        else:
            assert result_root.resolve() / "stages" in (output, *output.parents)

        command = _command(stage, result_root)
        assert command[3] == stage.output_option
        assert command[4] == str(output.relative_to(ROOT))
        assert "/data/" not in _display_command(command)


@pytest.mark.parametrize(
    ("stage_name", "expected_output"),
    [
        ("fig15", "stages/fig15"),
        ("fig21_paper_legacy", "stages/fig21"),
        ("fig21_current_corrected", "stages/fig21"),
        ("figures", "."),
    ],
)
def test_shared_stage_output_locations(stage_name: str, expected_output: str) -> None:
    result_root = ROOT / "results" / "reproduce"
    stage = next(item for item in reproduce_stages(1, "both") if item.name == stage_name)
    assert stage.output_path(result_root).resolve() == (result_root / expected_output).resolve()


@pytest.mark.parametrize("command", ["quick", "reproduce"])
def test_official_workflow_requires_dual_model_paths(command: str) -> None:
    parser = build_parser()
    args = parser.parse_args([command, "--model-version", "both"])
    assert args.model_version == "both"
    with pytest.raises(SystemExit):
        parser.parse_args([command, "--model-version", "paper_legacy"])


def test_verify_relative_path_is_anchored_at_artifact_root() -> None:
    args = build_parser().parse_args(["verify", "results/reproduce"])
    assert resolve_result_dir(args.result_dir) == ROOT / "results/reproduce"


@pytest.mark.parametrize("command", ["quick", "reproduce"])
def test_workflow_result_dir_is_configurable_and_repo_anchored(command: str) -> None:
    args = build_parser().parse_args([command, "--result-dir", "results/reviewer-copy"])
    assert resolve_result_dir(args.result_dir) == ROOT / "results/reviewer-copy"
