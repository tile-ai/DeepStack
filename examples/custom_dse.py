"""Single-configuration and DSE examples using an NVIDIA H200 reference system."""

import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/deepstack"), str(ROOT / "src/tilesight")]

import numpy as np
import torch

from mosaic.arch import H200
from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
    get_max_footprint_decode,
    modeling_decode,
)
from mosaic.llm_arch import Llama3_70b
from mosaic.noc.noc_config_set import h200x32
from mosaic.parallelism import ParallelScheme
from mosaic.utils import Modeling_Granularity


def evaluate(*, tp=8, pp=4, dp=1, batch_size=64, kv_length=2048,
             hbm_bandwidth_gbyteps=4800.0, nvlink_bandwidth_gbyteps=370.0,
             ib_bandwidth_gbyteps=50.0):
    if min(tp, pp, dp) < 1 or tp * pp * dp != 32:
        raise ValueError("this example uses 32 GPUs: TP * PP * DP must equal 32")
    if batch_size < 1 or kv_length < 1:
        raise ValueError("batch size and KV length must be positive")
    if not math.isfinite(hbm_bandwidth_gbyteps) or hbm_bandwidth_gbyteps <= 0:
        raise ValueError("HBM bandwidth must be finite and positive")
    np.random.seed(0)
    torch.set_num_threads(1)
    model = Llama3_70b()
    chip = H200()
    chip.ddr_bandwidth = hbm_bandwidth_gbyteps * 1e9
    noc = h200x32(nvlink_bw=nvlink_bandwidth_gbyteps * 1e9, ib_bw=ib_bandwidth_gbyteps * 1e9)
    parallel = ParallelScheme(tp=tp, pp=pp, dp=dp)
    # Follow the existing decode DSE driver's global-batch convention.
    microbatch = math.ceil(batch_size / pp)
    footprint = get_max_footprint_decode(
        model, microbatch, 1, kv_length, parallel, parallel,
        tp_transform_moe="none",
    )
    row = dict(tp=tp, pp=pp, dp=dp, batch_size=batch_size, microbatch=microbatch,
               memory_gib=sum(footprint) / 1024**3, status="OOM",
               latency_ms=None, stps=None)
    if sum(footprint) > chip.ddr_capacity:
        return row
    times, _, _ = modeling_decode(
        model_arch=model, bs=microbatch, seq=1, cached_kv_list=[kv_length],
        moe_parallel=parallel, non_moe_parallel=parallel,
        single_chip=chip, noc_hierarchy=noc,
        granularity=Modeling_Granularity("coarse", True, False, True),
        routing_array=None, tp_transform_moe="none",
    )
    latency_s = float(times[0][-1])
    row.update(status="OK", latency_ms=latency_s * 1000, stps=microbatch / latency_s)
    return row


def search(**settings):
    """Evaluate 15 fixed parallelism candidates, without kernel autotuning."""
    rows = []
    for tp in (1, 2, 4, 8, 16, 32):
        for pp in (1, 2, 4):
            if 32 % (tp * pp) == 0:
                rows.append(evaluate(tp=tp, pp=pp, dp=32 // (tp * pp), **settings))
    return sorted(rows, key=lambda row: row["stps"] or 0.0, reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dse", action="store_true", help="scan TP/PP/DP instead of one configuration")
    parser.add_argument("--tp", type=int, default=8, help="single-configuration tensor parallelism")
    parser.add_argument("--pp", type=int, default=4, help="single-configuration pipeline parallelism")
    parser.add_argument("--dp", type=int, default=1, help="single-configuration data parallelism")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kv-length", type=int, default=2048)
    parser.add_argument("--hbm-bandwidth-gbyteps", type=float, default=4800.0)
    parser.add_argument("--nvlink-bandwidth-gbyteps", type=float, default=370.0)
    parser.add_argument("--ib-bandwidth-gbyteps", type=float, default=50.0)
    parser.add_argument("--output", type=Path, help="optional CSV output")
    args = parser.parse_args()
    settings = {name: getattr(args, name) for name in (
        "batch_size", "kv_length", "hbm_bandwidth_gbyteps",
        "nvlink_bandwidth_gbyteps", "ib_bandwidth_gbyteps",
    )}
    rows = search(**settings) if args.dse else [evaluate(tp=args.tp, pp=args.pp, dp=args.dp, **settings)]
    print("32 x H200 (4 nodes x 8 GPUs), Llama3-70B decode")
    print(f"NVLink={args.nvlink_bandwidth_gbyteps:g} GB/s, IB={args.ib_bandwidth_gbyteps:g} GB/s, HBM={args.hbm_bandwidth_gbyteps:g} GB/s")
    print(" TP  PP  DP  microbatch  memory/GiB  latency/ms       STPS  status")
    for row in rows:
        latency = f"{row['latency_ms']:.3f}" if row["latency_ms"] is not None else "-"
        stps = f"{row['stps']:.3f}" if row["stps"] is not None else "-"
        print(f"{row['tp']:3d} {row['pp']:3d} {row['dp']:3d} {row['microbatch']:11d} {row['memory_gib']:11.2f} {latency:>11} {stps:>10}  {row['status']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
