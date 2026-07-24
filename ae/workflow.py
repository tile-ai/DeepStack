"""Top-level quick and default reproduction workflows."""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import RESULTS_DIR, ROOT, display_path


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    args: tuple[str, ...] = ()
    note: str = "CPU analytical-model recomputation"
    output_option: str = "--output-dir"
    output_subdir: str | None = None
    output_from_stage_root: bool = True

    def output_path(self, result_root: Path) -> Path:
        """Return this stage's only writable result directory."""

        base = result_root / "stages" if self.output_from_stage_root else result_root
        return base / (self.output_subdir or self.name)


def _workers(value: int, ceiling: int | None = None) -> str:
    bounded = max(1, value)
    if ceiling is not None:
        bounded = min(bounded, ceiling)
    return str(bounded)


def quick_stages(workers: int, model_version: str) -> tuple[Stage, ...]:
    """Short smoke path covering model, reference, and dual-version checks."""

    return (
        Stage("fig08", "ae.fig08", ("--no-plot",)),
        Stage("fig13", "ae.fig13", note="420-point NoC model rerun plus bundled NS-3 audit"),
        Stage("fig17", "ae.fig17"),
        Stage(
            "fig15_quick",
            "ae.fig15",
            (
                "--scope",
                "quick",
                "--model-version",
                model_version,
                "--workers",
                _workers(workers, 8),
            ),
            "28 deterministic non-winner samples from the full recorded decode DSE",
            output_option="--output-root",
            output_subdir="fig15",
        ),
    )


def reproduce_stages(workers: int, model_version: str) -> tuple[Stage, ...]:
    """Default bounded reproduction; exhaustive DSE and external tools excluded."""

    both = model_version
    stages: list[Stage] = [
        Stage("fig08", "ae.fig08", ("--no-plot",)),
        Stage("fig09", "ae.fig09", ("--workers", _workers(workers))),
        Stage("fig10", "ae.fig10", ("--workers", _workers(workers, 16))),
        Stage("fig12", "ae.fig12"),
        Stage("fig13", "ae.fig13", note="420-point NoC model rerun plus bundled NS-3 audit"),
        Stage(
            "fig15",
            "ae.fig15",
            (
                "--scope",
                "reproduce",
                "--model-version",
                both,
                "--workers",
                _workers(workers),
            ),
            "784 local winners derived from the complete recorded decode DSE",
            output_option="--output-root",
        ),
        Stage(
            "fig16",
            "ae.fig16",
            ("--model-version", both, "--workers", _workers(workers)),
            "80 recorded Figure 16 winners",
        ),
        Stage("fig17", "ae.fig17"),
        Stage(
            "fig18_20",
            "ae.fig18_20",
            ("--model-version", both, "--workers", _workers(workers)),
            "114 fixed DRAM-layer configurations plus archived DSE plot projections",
            output_option="--output",
        ),
    ]

    stages.append(
        Stage(
            "fig21_paper_legacy",
            "ae.fig21",
            ("--model-version", "paper_legacy", "--workers", _workers(workers)),
            "90-point fixed NoC surface",
            output_option="--output-root",
            output_subdir="fig21",
        )
    )
    if model_version in {"both", "current_corrected"}:
        stages.append(
            Stage(
                "fig21_current_corrected",
                "ae.fig21",
                (
                    "--model-version",
                    "current_corrected",
                    "--workers",
                    _workers(workers),
                ),
                "same fixed NoC configurations after MoE bug fix",
                output_option="--output-root",
                output_subdir="fig21",
            )
        )
    for name, module, note, output_option in (
        (
            "fig22",
            "ae.fig22",
            "selected per-layer NoC sensitivity points",
            "--output-dir",
        ),
        (
            "fig23",
            "ae.fig23",
            "84 main-figure parallel-space winners",
            "--output-root",
        ),
        (
            "table4",
            "ae.table4",
            "14 selected ablation configurations",
            "--output-dir",
        ),
    ):
        stages.append(
            Stage(
                name,
                module,
                ("--model-version", both, "--workers", _workers(workers)),
                note,
                output_option=output_option,
            )
        )
    stages.append(
        Stage(
            "figures",
            "ae.plot_all",
            (
                "--model-version",
                "paper_legacy",
            ),
            "16 paper-facing figure/table groups rendered as PNG and PDF",
            output_option="--result-root",
            output_subdir=".",
            output_from_stage_root=False,
        )
    )
    return tuple(stages)


def _portable_path(path: Path) -> str:
    """Prefer repository-relative paths in commands and manifests."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _command(stage: Stage, result_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        stage.module,
        stage.output_option,
        _portable_path(stage.output_path(result_root)),
        *stage.args,
    ]


def _display_command(command: list[str]) -> str:
    """Hide the machine-specific interpreter path in persisted metadata."""

    return shlex.join(["python", *command[1:]])


def _write_manifest(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "stage",
        "status",
        "return_code",
        "wall_time_s",
        "gpu_runs",
        "command",
        "note",
        "log",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_workflow(
    *,
    scope: str,
    workers: int,
    model_version: str,
    stages: Iterable[Stage] | None = None,
    result_root: Path | None = None,
) -> int:
    selected = tuple(
        stages
        if stages is not None
        else (
            quick_stages(workers, model_version)
            if scope == "quick"
            else reproduce_stages(workers, model_version)
        )
    )
    result_root = (result_root or (RESULTS_DIR / scope)).resolve()
    log_root = result_root / "logs"
    manifest_path = result_root / "run_manifest.csv"
    log_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    workflow_started = time.perf_counter()

    env = os.environ.copy()
    source_paths = [
        str(ROOT),
        str(ROOT / "src" / "deepstack"),
        str(ROOT / "src" / "tilesight"),
    ]
    if env.get("PYTHONPATH"):
        source_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    for index, stage in enumerate(selected, start=1):
        command = _command(stage, result_root)
        command_text = _display_command(command)
        log_path = log_root / f"{index:02d}_{stage.name}.log"
        print(f"\n[{index}/{len(selected)}] {stage.name}", flush=True)
        print(f"$ {command_text}", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        elapsed = time.perf_counter() - started
        status = "PASS" if return_code == 0 else "FAIL"
        rows.append(
            {
                "stage": stage.name,
                "status": status,
                "return_code": return_code,
                "wall_time_s": f"{elapsed:.6f}",
                "gpu_runs": 0,
                "command": command_text,
                "note": stage.note,
                "log": log_path.relative_to(result_root).as_posix(),
            }
        )
        _write_manifest(manifest_path, rows)
        print(f"[{status}] {stage.name}: {elapsed:.3f}s", flush=True)
        if return_code:
            print(f"workflow stopped; see {display_path(log_path)}", file=sys.stderr)
            return return_code

    elapsed = time.perf_counter() - workflow_started
    print(f"\nworkflow: PASS ({len(rows)} stages, {elapsed:.3f}s)")
    print(f"manifest: {display_path(manifest_path)}")
    return 0
