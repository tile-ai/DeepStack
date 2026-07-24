#!/usr/bin/env python3
"""Extract the compact Fig. 21 fixed-configuration sweep from the paper CSV."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


BASELINE = "torus_mesh_switch_1"
COLUMNS = [
    "baseline_noc",
    "latency_multiplier",
    "bw_multiplier",
    "bs",
    "minibatch",
    "seq",
    "tp",
    "ep",
    "ep1",
    "ep2",
    "sp",
    "cp",
    "dp",
    "fsdp",
    "pp",
    "tp_transform_moe",
    "kv_len_1",
    "kv_len_2",
    "kv_len_3",
    "kv_len_4",
    "scaled_utps_avg",
    "scaled_stps_avg",
]


def _candidate_id(row: pd.Series) -> str:
    key = "|".join(str(row[column]) for column in COLUMNS[:-2])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.source, usecols=COLUMNS)
    frame = frame[frame.baseline_noc == BASELINE].copy()
    frame.tp_transform_moe = frame.tp_transform_moe.fillna("none")
    if len(frame) != 180:
        raise RuntimeError(f"expected 180 {BASELINE} rows, found {len(frame)}")
    frame.insert(0, "candidate_id", frame.apply(_candidate_id, axis=1))
    if frame.candidate_id.duplicated().any():
        raise RuntimeError("candidate IDs are not unique")
    frame.rename(
        columns={
            "scaled_utps_avg": "paper_utps_avg",
            "scaled_stps_avg": "paper_stps_avg",
        },
        inplace=True,
    )
    frame.sort_values(
        ["bs", "latency_multiplier", "bw_multiplier", "tp_transform_moe"],
        inplace=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.15g")
    print(f"wrote {len(frame)} Fig. 21 candidates to {args.output}")


if __name__ == "__main__":
    main()
