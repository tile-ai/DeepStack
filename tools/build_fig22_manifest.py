#!/usr/bin/env python3
"""Build the compact fixed-configuration manifest for AE Figure 22.

This is a packaging-time provenance helper.  It joins the March 2026 DSE
winners with the June 2026 fixed-point re-evaluation CSVs.  Runtime AE code
uses only the resulting artifact-local manifest.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


KEYS = {
    "decode": ["baseline_noc", "scaled_layer", "bw_multiplier", "bs"],
    "prefill": ["baseline_noc", "scaled_layer", "bw_multiplier", "bs", "seq"],
}


def _load_onex(source: pd.DataFrame, phase: str) -> dict[tuple[object, ...], tuple[int, ...]]:
    workload = ["baseline_noc", "bs"]
    if phase == "prefill":
        workload.append("seq")
    # Preserve source-row order: this exactly matches the original runner's
    # drop_duplicates call when the three bw=1 layer rows are equivalent/tied.
    one = source[source.bw_multiplier == 1.0].drop_duplicates(workload)
    return {
        tuple(getattr(row, name) for name in workload): (
            int(row.tp),
            int(row.ep),
            int(row.dp),
            int(row.pp),
        )
        for row in one.itertuples(index=False)
    }


def _point_id(phase: str, index: int) -> str:
    return f"{phase}_{index:04d}"


def build_phase(source_path: Path, reference_path: Path, phase: str) -> pd.DataFrame:
    source = pd.read_csv(source_path)
    reference = pd.read_csv(reference_path)
    keys = KEYS[phase]
    if source.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError(f"{phase}: duplicate fixed-point keys")
    if len(source) != len(reference):
        raise ValueError(f"{phase}: source/reference row-count mismatch")

    source_columns = keys + ["tp", "ep", "dp", "pp", "scaled_stps_avg" if phase == "decode" else "scaled_stps"]
    paper_dse_column = source_columns[-1]
    selected = source[source_columns].rename(
        columns={
            "tp": "source_tp",
            "ep": "source_ep",
            "dp": "source_dp",
            "pp": "source_pp",
            paper_dse_column: "source_dse_stps",
        }
    )
    joined = reference.merge(selected, on=keys, how="inner", validate="one_to_one")
    if len(joined) != len(source):
        raise ValueError(f"{phase}: join did not close")
    for name in ("tp", "ep", "dp", "pp"):
        if not (joined[name].astype(int) == joined[f"source_{name}"].astype(int)).all():
            raise ValueError(f"{phase}: {name} differs between source and reference")
    source_delta = (joined.source_dse_stps / joined.orig_stps_from_csv - 1.0).abs().max()
    if source_delta > 1.0e-12:
        raise ValueError(f"{phase}: archived DSE STPS mismatch ({source_delta})")

    onex = _load_onex(source, phase)
    workload = ["baseline_noc", "bs"] + (["seq"] if phase == "prefill" else [])
    rows: list[dict[str, object]] = []
    ordered = joined.sort_values(keys).reset_index(drop=True)
    for index, row in enumerate(ordered.itertuples(index=False), start=1):
        key = tuple(getattr(row, name) for name in workload)
        onex_tp, onex_ep, onex_dp, onex_pp = onex[key]
        bs = int(row.bs)
        pp = int(row.pp)
        if phase == "decode":
            seq = 1
            sp, cp = 1, 1
            kv = (1024, 1365, 1707, 2048)
            recon = 0.0
        else:
            seq = int(row.seq)
            sp, cp = int(row.sp), int(row.cp)
            kv = (1, 1, 1, 1)
            recon = float(row.recon_rel_err)
        rows.append(
            {
                "point_id": _point_id(phase, index),
                "phase": phase,
                "model": "DeepSeekV3",
                "baseline_noc": row.baseline_noc,
                "scaled_layer": row.scaled_layer,
                "bw_multiplier": float(row.bw_multiplier),
                "bs": bs,
                "minibatch": math.ceil(bs / pp),
                "seq": seq,
                "orig_sm_count": int(row.orig_sm_count),
                "baseline_sm_count": int(row.baseline_sm_count),
                "num_devices": 256,
                "tp": int(row.tp),
                "ep": int(row.ep),
                "dp": int(row.dp),
                "pp": pp,
                "sp": sp,
                "cp": cp,
                "onex_tp": onex_tp,
                "onex_ep": onex_ep,
                "onex_dp": onex_dp,
                "onex_pp": onex_pp,
                "kv_len_1": kv[0],
                "kv_len_2": kv[1],
                "kv_len_3": kv[2],
                "kv_len_4": kv[3],
                "tp_transform_moe": "replace_only",
                "paper_logic_stps": float(row.stps_logic_die),
                "paper_dram_stps": float(row.stps_noc_to_dram),
                "paper_dse_stps": float(row.orig_stps_from_csv),
                "paper_recon_rel_err": recon,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-source", type=Path, required=True)
    parser.add_argument("--prefill-source", type=Path, required=True)
    parser.add_argument("--decode-reference", type=Path, required=True)
    parser.add_argument("--prefill-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.concat(
        [
            build_phase(args.decode_source, args.decode_reference, "decode"),
            build_phase(args.prefill_source, args.prefill_reference, "prefill"),
        ],
        ignore_index=True,
    )
    if len(frame) != 756 or frame.point_id.nunique() != 756:
        raise ValueError(f"expected 756 unique points, found {len(frame)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.15g")
    print(f"wrote {args.output}: {len(frame)} fixed points")


if __name__ == "__main__":
    main()
