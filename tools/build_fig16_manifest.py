#!/usr/bin/env python3
"""Extract the 80 fixed winners plotted in the paper's decode Fig. 16."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


MODELS = ("DeepSeekV3", "Qwen3_235b_a22b", "Llama3_405b", "Llama3_70b")
BATCH_SIZES = (1, 16, 64, 128, 256, 512, 1024)
SCALED_ARCHITECTURES = ("H100_SCALED", "H200_SCALED")
SCHEME_COLUMNS = ("tp", "ep", "ep1", "ep2", "sp", "cp", "dp", "fsdp", "pp")
USECOLS = (
    "arch",
    "noc",
    "model",
    "bs",
    "minibatch",
    "seq",
    *SCHEME_COLUMNS,
    "tp_transform_moe",
    "max_activation/GiB",
    "mem_weight/GiB",
    "kv_cache/GiB",
    "kv_len_1",
    "kv_len_2",
    "kv_len_3",
    "kv_len_4",
    "utps_avg",
    "stps_avg",
    "stats_run_id",
)


def _stable_id(row: pd.Series) -> str:
    values = [
        str(row[column])
        for column in ("arch", "noc", "model", "bs", *SCHEME_COLUMNS, "tp_transform_moe")
    ]
    return hashlib.sha256("|".join(values).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.source, usecols=list(USECOLS))
    frame = frame[
        frame.model.isin(MODELS)
        & frame.bs.isin(BATCH_SIZES)
        & (frame.kv_len_4 == 2048)
    ].copy()

    chosen: list[pd.Series] = []
    strata = frame[~frame.arch.isin(SCALED_ARCHITECTURES)]
    for _, group in strata.groupby(["model", "bs"], sort=True):
        row = group.loc[group.stps_avg.idxmax()].copy()
        row["platform"] = "DeepStack"
        row["selection_reason"] = "global_best_deepstack"
        chosen.append(row)

    scaled = frame[frame.arch.isin(SCALED_ARCHITECTURES)]
    for (arch, model, bs), group in scaled.groupby(["arch", "model", "bs"], sort=True):
        # The paper plotting script intentionally omits these two BS=1
        # baselines even though the full search CSV contains candidates.
        if bs == 1 and model in ("Llama3_70b", "Qwen3_235b_a22b"):
            continue
        row = group.loc[group.stps_avg.idxmax()].copy()
        row["platform"] = arch
        row["selection_reason"] = f"global_best_{str(arch).lower()}"
        chosen.append(row)

    output = pd.DataFrame(chosen)
    output.insert(0, "candidate_id", output.apply(_stable_id, axis=1))
    if len(output) != 80 or output.candidate_id.duplicated().any():
        raise RuntimeError(
            f"expected 80 unique Fig. 16 candidates, got {len(output)} rows and "
            f"{output.candidate_id.nunique()} IDs"
        )
    output.rename(
        columns={"utps_avg": "paper_utps_avg", "stps_avg": "paper_stps_avg"},
        inplace=True,
    )
    columns = [
        "candidate_id",
        "selection_reason",
        "platform",
        *[column for column in USECOLS if column not in ("utps_avg", "stps_avg")],
        "paper_utps_avg",
        "paper_stps_avg",
    ]
    # Keep stats_run_id at the end, matching the Fig. 15 manifest convention.
    columns.remove("stats_run_id")
    columns.append("stats_run_id")
    output = output[columns]
    output.sort_values(["model", "bs", "platform"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.15g")
    print(f"wrote {len(output)} Fig. 16 candidates to {args.output}")


if __name__ == "__main__":
    main()
