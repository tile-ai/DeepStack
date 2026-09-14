"""Custom accelerator example using an NVIDIA H100 reference configuration."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/deepstack"), str(ROOT / "src/tilesight")]

import numpy as np
import torch

from mosaic.arch import H100
from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import modeling_decode
from mosaic.llm_arch import Llama3_70b
from mosaic.noc.custom_profile import make_custom_profile
from mosaic.noc.energy_config import NocEnergyConfig
from mosaic.parallelism import ParallelScheme
from mosaic.utils import Modeling_Granularity


def evaluate(*, hbm_bandwidth_gbyteps=3350.0, noc_bandwidth_gbyteps=450.0,
             noc_latency_ns=1000.0, tp=8, dp=1, batch_size=64):
    if tp < 1 or dp < 1 or tp * dp != 8:
        raise ValueError("this example uses eight GPUs, so TP * DP must equal 8")
    if batch_size < 1 or not np.isfinite(hbm_bandwidth_gbyteps) or hbm_bandwidth_gbyteps <= 0:
        raise ValueError("batch size and HBM bandwidth must be positive")
    np.random.seed(0)
    torch.set_num_threads(1)
    chip = H100()
    chip.ddr_bandwidth = hbm_bandwidth_gbyteps * 1e9  # decimal GB/s -> bytes/s
    # An illustrative switch fabric, not a detailed NVLink topology calibration.
    noc = make_custom_profile(
        name="custom_eight_gpu_switch",
        layers={
            level: {
                "kind": "switch", "shape": (1, 8 if level == "L1" else 1),
                "hop_latency_ns": noc_latency_ns,
                "link_bandwidth_gbytes_per_s": noc_bandwidth_gbyteps,
            }
            for level in ("L3", "L2", "L1")
        },
        # Illustrative energy inputs; this example reports throughput only.
        energy_config=NocEnergyConfig(l1_pj_per_bit=1.0, l2_pj_per_bit=1.0, l3_pj_per_bit=1.0),
    )
    parallel = ParallelScheme(tp=tp, dp=dp)
    times, _, _ = modeling_decode(
        model_arch=Llama3_70b(), bs=batch_size, seq=1, cached_kv_list=[2048],
        moe_parallel=parallel, non_moe_parallel=parallel,
        single_chip=chip, noc_hierarchy=noc,
        granularity=Modeling_Granularity("coarse", True, False, True),
        routing_array=None, tp_transform_moe="none",  # dense model
    )
    latency_s = float(times[0][-1])
    return latency_s, batch_size / latency_s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hbm-bandwidth-gbyteps", type=float, default=3350.0)
    parser.add_argument("--noc-bandwidth-gbyteps", type=float, default=450.0)
    parser.add_argument("--noc-latency-ns", type=float, default=1000.0)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    latency, stps = evaluate(**vars(args))
    print(f"8 x H100, Llama3-70B, BS={args.batch_size}, KV=2048, TP={args.tp}, DP={args.dp}")
    print(f"HBM={args.hbm_bandwidth_gbyteps:g} GB/s, NoC={args.noc_bandwidth_gbyteps:g} GB/s, hop={args.noc_latency_ns:g} ns")
    print(f"Decode step: {latency * 1000:.3f} ms; system throughput: {stps:.3f} tokens/s")


if __name__ == "__main__":
    main()
