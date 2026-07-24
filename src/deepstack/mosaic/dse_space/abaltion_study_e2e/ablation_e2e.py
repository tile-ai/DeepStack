#!/usr/bin/env python3
"""
ablation_e2e.py — End-to-end ablation study for DSE framework.

Steps 1-4: Filter merged CSV → find best config → re-evaluate with overlap=False
Step 5:    Use existing CSV data (overlap=True) for best Step4 config
Step 6:    DRAM layer specialization from dram_layer CSV
Step 7:    NoC BW optimization (coarse + fine search) — new DSE
"""

import os
import sys
import math
import logging
import dataclasses
import csv
from pathlib import Path
from itertools import product as iter_product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project imports ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # source root
sys.path.insert(0, str(PROJECT_ROOT))

from mosaic.arch.stacked_gpu_base import stacked_gpu_base
from mosaic.arch.stacked_gpu_wgmma import stacked_gpu_wgmma
from mosaic.noc.noc_config_set import torus_mesh_switch_1
from mosaic.parallelism import ParallelScheme
from mosaic.utils import Modeling_Granularity
from mosaic.utils.allocate_ep import allocate_ep
from mosaic.llm_arch import DeepSeekV3
from mosaic.dse_space.dse_framework_multi_process_v4_decode_dump_stats import (
    modeling_decode, get_max_footprint_decode, _get_effective_moe_parallel,
)
from mosaic.dse_space.case_study_dram_layer.dram_layer_config import (
    make_arch_for_config, compute_power_wall, compute_thermal_freq_scale,
)

logging.basicConfig(level=logging.ERROR)

# ── Paths ────────────────────────────────────────────────────────────────
DSE_SPACE = PROJECT_ROOT / "mosaic" / "dse_space"
MERGED_CSV = DSE_SPACE / "lfs" / "merged_stacked_gpu_h200_decode_all_20260319.csv"
DRAM_CSV = DSE_SPACE / "case_study_dram_layer" / "lfs" / "dram_layer_decode_result_dpsk_mn1to12.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
NPZ_TRACE = PROJECT_ROOT / "mosaic" / "data" / "aime_ds_r1" / "moe_activations_batch0.npz"

BATCH_SIZES = [4, 1024]
NUM_DEVICES = 256
INPUT_SEQ = 1024
TASK_MAX_SEQ = 2048
NUM_KV_POINT = 4

# ── Step Filters ─────────────────────────────────────────────────────────
STEP_FILTERS = {
    1: {"arch": ["stacked_gpu_base"], "noc": ["torus_mesh_switch_1"],
        "ep": [1], "sp": [1], "cp": [1], "fsdp": [False],
        "tp_transform_moe": ["none"]},
    2: {"arch": ["stacked_gpu_base"], "noc": ["torus_mesh_switch_1"],
        "tp_transform_moe": ["none"]},
    3: {"arch": ["stacked_gpu_base"], "noc": ["torus_mesh_switch_1"],
        "tp_transform_moe": ["none", "replace_only"]},
    4: {"arch": ["stacked_gpu_base", "stacked_gpu_wgmma"], "noc": ["torus_mesh_switch_1"],
        "tp_transform_moe": ["none", "replace_only"]},
}

STEP_NAMES = {
    1: "AstraSim (tp/pp/dp)",
    2: "+ Full Parallel",
    3: "+ Flexible Parallelism Across Module",
    4: "+ Search On-chip Architecture",
    5: "+ Comm-Comp Overlap",
    6: "+ Stacking DSE",
    7: "+ NoC DSE",
}

# ── Routing array (loaded once) ─────────────────────────────────────────
_ROUTING_ARRAY = None

def _load_routing_array():
    global _ROUTING_ARRAY
    if _ROUTING_ARRAY is None:
        from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
        _, decode_array = load_npz_routing_keep_shape(str(NPZ_TRACE), as_list=False)
        _ROUTING_ARRAY = decode_array
    return _ROUTING_ARRAY


# ═════════════════════════════════════════════════════════════════════════
#  CSV Loading & Filtering
# ═════════════════════════════════════════════════════════════════════════

def load_merged_csv():
    """Load and pre-filter merged CSV for DeepSeekV3, 256 devices."""
    print("Loading merged CSV...")
    df = pd.read_csv(MERGED_CSV)
    df = df[df["model"] == "DeepSeekV3"].copy()
    df["num_devices"] = df["tp"] * df["ep"] * df["sp"] * df["cp"] * df["dp"] * df["pp"]
    df = df[df["num_devices"] == NUM_DEVICES].copy()
    print(f"  DeepSeekV3 256-device rows: {len(df)}")
    return df


def filter_and_find_best(df, filters, bs):
    """Apply step-specific filters and return the row with best stps_avg."""
    sub = df[df["bs"] == bs].copy()
    for col, values in filters.items():
        sub = sub[sub[col].isin(values)]
    if sub.empty:
        print(f"  WARNING: no data for bs={bs} with filters {filters}")
        return None
    best_idx = sub["stps_avg"].idxmax()
    return sub.loc[best_idx]


# ═════════════════════════════════════════════════════════════════════════
#  Single-point modeling_decode evaluation
# ═════════════════════════════════════════════════════════════════════════

def _build_arch(arch_name):
    """Construct arch object from name."""
    if arch_name == "stacked_gpu_wgmma":
        return stacked_gpu_wgmma()
    else:
        return stacked_gpu_base()


def _compute_sampled_kv_len():
    """Compute the 4 sampled KV lengths."""
    model = DeepSeekV3()
    max_kv = min(TASK_MAX_SEQ, model.max_seq_len)
    step = (max_kv - INPUT_SEQ) / (NUM_KV_POINT - 1)
    return tuple(int(round(INPUT_SEQ + i * step)) for i in range(NUM_KV_POINT))


def evaluate_config(config_row, comp_comm_overlap, arch_override=None, noc_override=None):
    """Evaluate a single config point with specified overlap setting.

    Args:
        config_row: pandas Series from CSV with parallel scheme + arch info
        comp_comm_overlap: bool
        arch_override: optional arch object (for Step 7 with custom DRAM/BW)
        noc_override: optional noc object

    Returns:
        dict with stps, power, area, etc.
    """
    routing_array = _load_routing_array()
    model_arch = DeepSeekV3()

    # Build arch & noc
    if arch_override is not None:
        arch = arch_override
    else:
        arch = _build_arch(config_row["arch"])
    noc = noc_override if noc_override is not None else torus_mesh_switch_1()

    # Build parallel scheme
    tp = int(config_row["tp"])
    ep = int(config_row["ep"])
    sp = int(config_row["sp"])
    cp = int(config_row["cp"])
    dp = int(config_row["dp"])
    pp = int(config_row["pp"])
    fsdp = bool(config_row["fsdp"])
    tp_transform_moe = str(config_row["tp_transform_moe"])

    moe_parallel = ParallelScheme(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp, fsdp=fsdp)
    minibatch = math.ceil(int(config_row["bs"]) / pp)
    allocate_ep(parallel=moe_parallel, bs=minibatch, seq=1)

    non_moe_parallel = dataclasses.replace(
        moe_parallel, ep=1, ep1=1, ep2=1, dp=dp * ep
    )

    granularity = Modeling_Granularity(
        mode="coarse", comp_comm_overlap=comp_comm_overlap,
        auto_tune=False, dump_perf_log=True,
    )

    sampled_kv_len = _compute_sampled_kv_len()

    time_list, model_stats, representative_kv_len = modeling_decode(
        model_arch=model_arch, bs=minibatch, seq=1,
        cached_kv_list=sampled_kv_len,
        moe_parallel=moe_parallel, non_moe_parallel=non_moe_parallel,
        single_chip=arch, noc_hierarchy=noc, granularity=granularity,
        routing_array=routing_array, tp_transform_moe=tp_transform_moe,
    )

    # Compute metrics
    utps_all, stps_all = 0.0, 0.0
    for i, kv_len in enumerate(sampled_kv_len):
        _, _, _, _, _, _, _, time_total = time_list[i]
        stps_all += minibatch / time_total
        utps_all += 1 / time_total / pp
    stps_avg = stps_all / len(sampled_kv_len)
    utps_avg = utps_all / len(sampled_kv_len)

    # Power from model_stats
    # Use representative time for power calculation
    _, _, _, _, _, _, _, repr_time = time_list[len(sampled_kv_len) // 2]
    e2e_time = repr_time
    num_dev = noc.num_devices
    chip_power = model_stats.chip_total_energy_j / e2e_time / num_dev if e2e_time > 0 else 0
    noc_power = model_stats.noc_total_energy_j / e2e_time / num_dev if e2e_time > 0 else 0
    total_power = chip_power + noc_power

    # Power wall / heating
    hit_wall, freq_scale = compute_power_wall(chip_power, noc_power)
    heating_feasible = not hit_wall

    # Token efficiency
    tokens_j = stps_avg / (num_dev * total_power) if total_power > 0 else 0

    return {
        "stps_avg": stps_avg,
        "utps_avg": utps_avg,
        "tokens_per_joule": tokens_j,
        "total_power_W": total_power,
        "chip_power_W": chip_power,
        "noc_power_W": noc_power,
        "hit_power_wall": hit_wall,
        "freq_scale_power": freq_scale,
        "heating_feasible": heating_feasible,
        "tp": tp, "ep": ep, "sp": sp, "cp": cp, "dp": dp, "pp": pp,
        "fsdp": fsdp, "tp_transform_moe": tp_transform_moe,
        "arch": config_row.get("arch", "stacked_gpu_wgmma"),
    }


# ═════════════════════════════════════════════════════════════════════════
#  Steps 1-5
# ═════════════════════════════════════════════════════════════════════════

def run_steps_1_to_5(merged_df):
    """Run ablation Steps 1-5.

    Steps 1-4: filter CSV → find best → re-evaluate with overlap=False
    Step 5: use existing CSV data (overlap=True) for Step4 best config
    """
    results = {}

    for step in [1, 2, 3, 4]:
        results[step] = {}
        for bs in BATCH_SIZES:
            print(f"\n=== Step {step}: {STEP_NAMES[step]}, bs={bs} ===")
            best_row = filter_and_find_best(merged_df, STEP_FILTERS[step], bs)
            if best_row is None:
                continue
            print(f"  Best CSV config: arch={best_row['arch']}, tp={int(best_row['tp'])}, "
                  f"ep={int(best_row['ep'])}, dp={int(best_row['dp'])}, pp={int(best_row['pp'])}, "
                  f"fsdp={best_row['fsdp']}, moe={best_row['tp_transform_moe']}, "
                  f"stps_avg(overlap=True)={best_row['stps_avg']:.1f}")

            # Re-evaluate with overlap=False
            result = evaluate_config(best_row, comp_comm_overlap=False)
            print(f"  Re-evaluated (overlap=False): stps={result['stps_avg']:.1f}, "
                  f"power={result['total_power_W']:.1f}W, "
                  f"tok/J={result['tokens_per_joule']:.4f}, heat_ok={result['heating_feasible']}")
            results[step][bs] = result

    # Step 5: overlap=True — use existing CSV data for Step 4 best config
    results[5] = {}
    for bs in BATCH_SIZES:
        print(f"\n=== Step 5: {STEP_NAMES[5]}, bs={bs} ===")
        best_row = filter_and_find_best(merged_df, STEP_FILTERS[4], bs)
        if best_row is None:
            continue
        # Re-evaluate with overlap=True to get power/area consistently
        result = evaluate_config(best_row, comp_comm_overlap=True)
        print(f"  With overlap=True: stps={result['stps_avg']:.1f}, "
              f"power={result['total_power_W']:.1f}W, "
              f"tok/J={result['tokens_per_joule']:.4f}, heat_ok={result['heating_feasible']}")
        results[5][bs] = result

    return results


# ═════════════════════════════════════════════════════════════════════════
#  Step 6: DRAM Layer Specialization
# ═════════════════════════════════════════════════════════════════════════

def run_step_6():
    """Step 6: Find optimal DRAM config for each BS, show specialization."""
    print("\n" + "=" * 70)
    print("Step 6: DRAM Layer Specialization")
    print("=" * 70)

    dram_df = pd.read_csv(DRAM_CSV)
    dram_df = dram_df[dram_df["model"] == "DeepSeekV3"].copy()

    results = {}
    candidates = {}

    for bs in BATCH_SIZES:
        print(f"\n--- bs={bs} ---")
        sub = dram_df[dram_df["bs"] == bs].copy()
        if sub.empty:
            print(f"  WARNING: no DRAM data for bs={bs}")
            continue

        # Find best by scaled_stps_avg
        best_idx = sub["scaled_stps_avg"].idxmax()
        best = sub.loc[best_idx]
        best_stps = best["scaled_stps_avg"]

        # Baseline: m=4, n=4
        baseline = sub[(sub["dram_total_layers"] == 4) & (sub["dram_active_layers"] == 4)]
        if not baseline.empty:
            baseline_best_stps = baseline["scaled_stps_avg"].max()
            print(f"  Baseline (m=4,n=4) best stps: {baseline_best_stps:.1f}")
        else:
            baseline_best_stps = 0
            print(f"  Baseline (m=4,n=4): no data")

        print(f"  Best DRAM config: m={int(best['dram_total_layers'])}, n={int(best['dram_active_layers'])}, "
              f"smem={int(best['smem_capacity_KiB'])}KiB, l1_tp={int(best['l1_throughput_Bpc'])}B/cyc, "
              f"SM={int(best['sm_count'])}")
        print(f"  Best stps: {best_stps:.1f} (vs baseline {baseline_best_stps:.1f}, "
              f"+{(best_stps/baseline_best_stps-1)*100:.1f}%)" if baseline_best_stps > 0 else "")
        print(
            f"  Power: {best['total_power_per_device_W']:.1f}W, "
            f"heat_wall: {best['hit_power_wall']}"
        )

        # Keep candidates within 5% of best
        threshold = best_stps * 0.95
        near_best = sub[sub["scaled_stps_avg"] >= threshold].copy()
        print(f"  Candidates within 5%: {len(near_best)} configs")

        # Compute tokens/J for best
        total_power = best["total_power_per_device_W"]
        tokens_j = best_stps / (NUM_DEVICES * total_power) if total_power > 0 else 0

        results[bs] = {
            "stps_avg": best_stps,
            "tokens_per_joule": tokens_j,
            "total_power_W": total_power,
            "heating_feasible": not best["hit_power_wall"],
            "tp": int(best["tp"]), "ep": int(best["ep"]),
            "sp": int(best["sp"]), "cp": int(best["cp"]),
            "dp": int(best["dp"]), "pp": int(best["pp"]),
            "fsdp": bool(best["fsdp"]),
            "tp_transform_moe": best["tp_transform_moe"],
            "arch": "stacked_gpu_wgmma",
            "dram_m": int(best["dram_total_layers"]),
            "dram_n": int(best["dram_active_layers"]),
            "smem_KiB": int(best["smem_capacity_KiB"]),
            "l1_tp_Bpc": int(best["l1_throughput_Bpc"]),
            "sm_count": int(best["sm_count"]),
        }
        candidates[bs] = near_best

    # Show specialization
    if 4 in results and 1024 in results:
        r4, r1024 = results[4], results[1024]
        same_hw = (r4.get("dram_m") == r1024.get("dram_m") and
                   r4.get("dram_n") == r1024.get("dram_n") and
                   r4.get("smem_KiB") == r1024.get("smem_KiB") and
                   r4.get("l1_tp_Bpc") == r1024.get("l1_tp_Bpc"))
        if same_hw:
            print("\n  >> Same HW optimal for bs=4 and bs=1024 (no specialization needed)")
        else:
            print(f"\n  >> Different HW optimal! Specialization/disaggregation valuable:")
            print(f"     bs=4:    m={r4.get('dram_m')}, n={r4.get('dram_n')}, smem={r4.get('smem_KiB')}KiB, l1={r4.get('l1_tp_Bpc')}B/c")
            print(f"     bs=1024: m={r1024.get('dram_m')}, n={r1024.get('dram_n')}, smem={r1024.get('smem_KiB')}KiB, l1={r1024.get('l1_tp_Bpc')}B/c")

    # Save candidates
    os.makedirs(OUT_DIR, exist_ok=True)
    for bs in BATCH_SIZES:
        if bs in candidates:
            cand_path = OUT_DIR / f"step6_candidates_bs{bs}.csv"
            candidates[bs].to_csv(cand_path, index=False)
            print(f"  Saved candidates: {cand_path}")

    return results, candidates


# ═════════════════════════════════════════════════════════════════════════
#  Step 7: NoC BW Optimization
# ═════════════════════════════════════════════════════════════════════════

def run_step_7(step6_results, step6_candidates):
    """Step 7: Search optimal NoC BW for each layer on Step 6 candidates."""
    from mosaic.dse_space.abaltion_study_e2e.noc_multi_layer_bw import (
        make_noc_multi, make_arch_for_noc,
    )

    print("\n" + "=" * 70)
    print("Step 7: NoC Bandwidth Optimization")
    print("=" * 70)

    routing_array = _load_routing_array()
    model_arch = DeepSeekV3()
    sampled_kv_len = _compute_sampled_kv_len()

    COARSE_MULTS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = {}

    for bs in BATCH_SIZES:
        print(f"\n--- bs={bs} ---")
        if bs not in step6_candidates or step6_candidates[bs].empty:
            print(f"  No Step6 candidates for bs={bs}, skipping")
            continue

        cands = step6_candidates[bs]
        # Take top N unique HW configs to limit search space
        hw_cols = ["dram_total_layers", "dram_active_layers", "smem_capacity_KiB", "l1_throughput_Bpc"]
        unique_hw = cands.drop_duplicates(subset=hw_cols)
        # For each unique HW, take the best parallel config
        best_per_hw = []
        for _, hw in unique_hw.iterrows():
            mask = True
            for c in hw_cols:
                mask = mask & (cands[c] == hw[c])
            hw_sub = cands[mask]
            best_idx = hw_sub["scaled_stps_avg"].idxmax()
            best_per_hw.append(cands.loc[best_idx])

        # Limit to top 3 HW configs to keep search tractable
        best_per_hw.sort(key=lambda r: r["scaled_stps_avg"], reverse=True)
        best_per_hw = best_per_hw[:3]
        print(f"  Searching {len(best_per_hw)} HW configs × {len(COARSE_MULTS)}^3 = "
              f"{len(best_per_hw) * len(COARSE_MULTS)**3} BW combos")

        coarse_results = []

        for hw_idx, cand_row in enumerate(best_per_hw):
            m = int(cand_row["dram_total_layers"])
            n = int(cand_row["dram_active_layers"])
            smem = int(cand_row["smem_capacity_KiB"]) * 1024
            l1_tp = int(cand_row["l1_throughput_Bpc"])

            tp = int(cand_row["tp"])
            ep = int(cand_row["ep"])
            sp = int(cand_row["sp"])
            cp = int(cand_row["cp"])
            dp = int(cand_row["dp"])
            pp = int(cand_row["pp"])
            fsdp = bool(cand_row["fsdp"])
            tp_transform_moe = str(cand_row["tp_transform_moe"])

            moe_parallel = ParallelScheme(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp, fsdp=fsdp)
            minibatch = math.ceil(bs / pp)
            allocate_ep(parallel=moe_parallel, bs=minibatch, seq=1)
            non_moe_parallel = dataclasses.replace(moe_parallel, ep=1, ep1=1, ep2=1, dp=dp * ep)

            granularity = Modeling_Granularity(
                mode="coarse", comp_comm_overlap=True,
                auto_tune=False, dump_perf_log=True,
            )

            for l1m, l2m, l3m in iter_product(COARSE_MULTS, repeat=3):
                bw_mults = {"L1": l1m, "L2": l2m, "L3": l3m}
                noc = make_noc_multi("torus_mesh_switch_1", bw_mults)
                arch, sm_count = make_arch_for_noc(
                    noc, m=m, n=n, smem_cap=smem, l1_throughput=l1_tp,
                )

                # Check memory fit
                max_kv = min(TASK_MAX_SEQ, model_arch.max_seq_len)
                max_act, total_weight, kv_cache = get_max_footprint_decode(
                    model_arch=model_arch, bs=minibatch, seq=1, cached_kv=max_kv,
                    moe_parallel=moe_parallel, non_moe_parallel=non_moe_parallel,
                    tp_transform_moe=tp_transform_moe,
                )
                if (max_act + total_weight + kv_cache) > arch.ddr_capacity:
                    continue

                try:
                    time_list, model_stats, _ = modeling_decode(
                        model_arch=model_arch, bs=minibatch, seq=1,
                        cached_kv_list=sampled_kv_len,
                        moe_parallel=moe_parallel, non_moe_parallel=non_moe_parallel,
                        single_chip=arch, noc_hierarchy=noc, granularity=granularity,
                        routing_array=routing_array, tp_transform_moe=tp_transform_moe,
                    )
                except Exception as e:
                    continue

                stps_all = sum(minibatch / time_list[i][-1] for i in range(len(sampled_kv_len)))
                stps_avg = stps_all / len(sampled_kv_len)

                repr_time = time_list[len(sampled_kv_len) // 2][-1]
                num_dev = noc.num_devices
                chip_power = model_stats.chip_total_energy_j / repr_time / num_dev if repr_time > 0 else 0
                noc_power = model_stats.noc_total_energy_j / repr_time / num_dev if repr_time > 0 else 0
                total_power = chip_power + noc_power
                hit_wall, freq_scale = compute_power_wall(chip_power, noc_power)

                # Apply freq scaling to stps if power wall hit
                scaled_stps = stps_avg * freq_scale if hit_wall else stps_avg
                tokens_j = scaled_stps / (num_dev * total_power) if total_power > 0 else 0

                coarse_results.append({
                    "hw_idx": hw_idx, "m": m, "n": n,
                    "L1_mult": l1m, "L2_mult": l2m, "L3_mult": l3m,
                    "sm_count": sm_count,
                    "raw_stps_avg": stps_avg,
                    "scaled_stps_avg": scaled_stps,
                    "tokens_per_joule": tokens_j,
                    "total_power_W": total_power,
                    "hit_power_wall": hit_wall,
                    "freq_scale": freq_scale,
                    "tp": tp, "ep": ep, "dp": dp, "pp": pp,
                    "fsdp": fsdp, "tp_transform_moe": tp_transform_moe,
                })

        if not coarse_results:
            print(f"  No valid coarse results for bs={bs}")
            continue

        coarse_df = pd.DataFrame(coarse_results)
        coarse_path = OUT_DIR / f"step7_coarse_bs{bs}.csv"
        coarse_df.to_csv(coarse_path, index=False)
        print(f"  Saved coarse results: {coarse_path} ({len(coarse_df)} rows)")

        # Find best coarse
        best_coarse_idx = coarse_df["scaled_stps_avg"].idxmax()
        best_coarse = coarse_df.loc[best_coarse_idx]
        print(f"  Best coarse: L1={best_coarse['L1_mult']}, L2={best_coarse['L2_mult']}, "
              f"L3={best_coarse['L3_mult']}, stps={best_coarse['scaled_stps_avg']:.1f}, "
              f"SM={int(best_coarse['sm_count'])}")

        # Fine search around best coarse
        fine_results = _fine_search(
            best_coarse, best_per_hw, model_arch, sampled_kv_len, routing_array, bs,
        )

        if fine_results:
            fine_df = pd.DataFrame(fine_results)
            fine_path = OUT_DIR / f"step7_fine_bs{bs}.csv"
            fine_df.to_csv(fine_path, index=False)
            print(f"  Saved fine results: {fine_path} ({len(fine_df)} rows)")

            best_fine_idx = fine_df["scaled_stps_avg"].idxmax()
            best_fine = fine_df.loc[best_fine_idx]
            print(f"  Best fine: L1={best_fine['L1_mult']:.2f}, L2={best_fine['L2_mult']:.2f}, "
                  f"L3={best_fine['L3_mult']:.2f}, stps={best_fine['scaled_stps_avg']:.1f}")

            best_row = best_fine
        else:
            best_row = best_coarse

        total_power = best_row["total_power_W"]
        tokens_j = best_row["scaled_stps_avg"] / (NUM_DEVICES * total_power) if total_power > 0 else 0

        results[bs] = {
            "stps_avg": best_row["scaled_stps_avg"],
            "tokens_per_joule": tokens_j,
            "total_power_W": total_power,
            "heating_feasible": not best_row["hit_power_wall"],
            "tp": int(best_row["tp"]), "ep": int(best_row["ep"]),
            "dp": int(best_row["dp"]), "pp": int(best_row["pp"]),
            "fsdp": bool(best_row["fsdp"]),
            "tp_transform_moe": best_row["tp_transform_moe"],
            "arch": "stacked_gpu_wgmma",
            "L1_mult": best_row["L1_mult"],
            "L2_mult": best_row["L2_mult"],
            "L3_mult": best_row["L3_mult"],
        }

    return results


def _fine_search(best_coarse, best_per_hw, model_arch, sampled_kv_len, routing_array, bs):
    """Fine-grained BW search around the best coarse result."""
    from mosaic.dse_space.abaltion_study_e2e.noc_multi_layer_bw import (
        make_noc_multi, make_arch_for_noc,
    )

    hw_idx = int(best_coarse["hw_idx"])
    cand_row = best_per_hw[hw_idx]
    m = int(cand_row["dram_total_layers"])
    n = int(cand_row["dram_active_layers"])
    smem = int(cand_row["smem_capacity_KiB"]) * 1024
    l1_tp = int(cand_row["l1_throughput_Bpc"])

    tp = int(cand_row["tp"])
    ep = int(cand_row["ep"])
    sp = int(cand_row["sp"])
    cp = int(cand_row["cp"])
    dp = int(cand_row["dp"])
    pp = int(cand_row["pp"])
    fsdp = bool(cand_row["fsdp"])
    tp_transform_moe = str(cand_row["tp_transform_moe"])

    moe_parallel = ParallelScheme(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp, fsdp=fsdp)
    minibatch = math.ceil(bs / pp)
    allocate_ep(parallel=moe_parallel, bs=minibatch, seq=1)
    non_moe_parallel = dataclasses.replace(moe_parallel, ep=1, ep1=1, ep2=1, dp=dp * ep)

    granularity = Modeling_Granularity(
        mode="coarse", comp_comm_overlap=True,
        auto_tune=False, dump_perf_log=True,
    )

    # Generate fine steps around each best mult
    def _fine_range(center, step=0.1, count=3):
        vals = set()
        for i in range(-count, count + 1):
            v = round(center + i * step, 2)
            if v > 0.1:
                vals.add(v)
        return sorted(vals)

    l1_center = float(best_coarse["L1_mult"])
    l2_center = float(best_coarse["L2_mult"])
    l3_center = float(best_coarse["L3_mult"])

    l1_fine = _fine_range(l1_center)
    l2_fine = _fine_range(l2_center)
    l3_fine = _fine_range(l3_center)

    print(f"  Fine search: L1∈{l1_fine}, L2∈{l2_fine}, L3∈{l3_fine} "
          f"({len(l1_fine)*len(l2_fine)*len(l3_fine)} combos)")

    fine_results = []
    for l1m, l2m, l3m in iter_product(l1_fine, l2_fine, l3_fine):
        bw_mults = {"L1": l1m, "L2": l2m, "L3": l3m}
        noc = make_noc_multi("torus_mesh_switch_1", bw_mults)
        arch, sm_count = make_arch_for_noc(
            noc, m=m, n=n, smem_cap=smem, l1_throughput=l1_tp,
        )

        max_kv = min(TASK_MAX_SEQ, model_arch.max_seq_len)
        max_act, total_weight, kv_cache = get_max_footprint_decode(
            model_arch=model_arch, bs=minibatch, seq=1, cached_kv=max_kv,
            moe_parallel=moe_parallel, non_moe_parallel=non_moe_parallel,
            tp_transform_moe=tp_transform_moe,
        )
        if (max_act + total_weight + kv_cache) > arch.ddr_capacity:
            continue

        try:
            time_list, model_stats, _ = modeling_decode(
                model_arch=model_arch, bs=minibatch, seq=1,
                cached_kv_list=sampled_kv_len,
                moe_parallel=moe_parallel, non_moe_parallel=non_moe_parallel,
                single_chip=arch, noc_hierarchy=noc, granularity=granularity,
                routing_array=routing_array, tp_transform_moe=tp_transform_moe,
            )
        except Exception:
            continue

        stps_all = sum(minibatch / time_list[i][-1] for i in range(len(sampled_kv_len)))
        stps_avg = stps_all / len(sampled_kv_len)

        repr_time = time_list[len(sampled_kv_len) // 2][-1]
        num_dev = noc.num_devices
        chip_power = model_stats.chip_total_energy_j / repr_time / num_dev if repr_time > 0 else 0
        noc_power = model_stats.noc_total_energy_j / repr_time / num_dev if repr_time > 0 else 0
        total_power = chip_power + noc_power
        hit_wall, freq_scale = compute_power_wall(chip_power, noc_power)
        scaled_stps = stps_avg * freq_scale if hit_wall else stps_avg
        tokens_j = scaled_stps / (num_dev * total_power) if total_power > 0 else 0

        fine_results.append({
            "hw_idx": hw_idx, "m": m, "n": n,
            "L1_mult": l1m, "L2_mult": l2m, "L3_mult": l3m,
            "sm_count": sm_count,
            "raw_stps_avg": stps_avg,
            "scaled_stps_avg": scaled_stps,
            "tokens_per_joule": tokens_j,
            "total_power_W": total_power,
            "hit_power_wall": hit_wall,
            "freq_scale": freq_scale,
            "tp": tp, "ep": ep, "dp": dp, "pp": pp,
            "fsdp": fsdp, "tp_transform_moe": tp_transform_moe,
        })

    return fine_results


# ═════════════════════════════════════════════════════════════════════════
#  Summary & Visualization
# ═════════════════════════════════════════════════════════════════════════

def build_summary(all_results):
    """Build summary table from all step results."""
    rows = []
    for step in range(1, 8):
        for bs in BATCH_SIZES:
            r = all_results.get(step, {}).get(bs, {})
            if not r:
                continue

            config_str = f"tp={r.get('tp','?')},ep={r.get('ep','?')},dp={r.get('dp','?')},pp={r.get('pp','?')}"
            if r.get("tp_transform_moe") == "replace_only":
                config_str += ",moe=repl"
            if step == 6:
                config_str += f",m={r.get('dram_m','?')},n={r.get('dram_n','?')}"
                config_str += f",smem={r.get('smem_KiB','?')}K,l1={r.get('l1_tp_Bpc','?')}B"
            if step == 7:
                config_str += f",L1={r.get('L1_mult','?')},L2={r.get('L2_mult','?')},L3={r.get('L3_mult','?')}"

            rows.append({
                "step": step,
                "step_name": STEP_NAMES[step],
                "bs": bs,
                "stps_avg": r.get("stps_avg", 0),
                "tokens_per_joule": r.get("tokens_per_joule", 0),
                "total_power_W": r.get("total_power_W", 0),
                "heating_feasible": r.get("heating_feasible", True),
                "config": config_str,
            })

    summary_df = pd.DataFrame(rows)
    return summary_df


def make_ablation_table(summary_df):
    """Print and save ablation table — one table per BS."""
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = OUT_DIR / "ablation_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    for bs in BATCH_SIZES:
        print(f"\n{'='*100}")
        print(f"  ABLATION TABLE — DeepSeekV3 Decode, BS={bs}, 256 Devices")
        print(f"{'='*100}")
        header = (
            f"{'#':<3} {'Technique':<42} {'STPS':>10} {'Gain':>8} "
            f"{'Tok/J':>10} {'Heat':>6} {'Config':<30}"
        )
        print(header)
        print("-" * 130)

        sub = summary_df[summary_df["bs"] == bs].sort_values("step")
        prev_stps = None
        baseline_stps = None
        for _, row in sub.iterrows():
            step = int(row["step"])
            stps = row["stps_avg"]
            if baseline_stps is None:
                baseline_stps = stps
            gain = f"+{(stps / prev_stps - 1) * 100:.1f}%" if prev_stps else "-"
            prev_stps = stps
            tokj = f"{row['tokens_per_joule']:.4f}"
            heat = "Y" if row["heating_feasible"] else "N"
            cfg = row["config"][:30]
            print(f"{step:<3} {STEP_NAMES[step]:<42} {stps:>10.1f} {gain:>8} "
                  f"{tokj:>10} {heat:>6} {cfg:<30}")

        total_gain = prev_stps / baseline_stps if baseline_stps else 1
        print(f"\n  Total speedup: {total_gain:.2f}x (Step 1 -> Step 7)")

    print(f"\nSaved: {csv_path}")


def _plot_single_bs(summary_df, bs, steps):
    """Generate a 2-subplot figure (STPS + Tokens/J) for a single BS."""
    sub = summary_df[summary_df["bs"] == bs].sort_values("step")
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Ablation Study — DeepSeekV3 Decode, BS={bs}, 256 Devices", fontsize=14, y=1.02)

    x = np.arange(len(steps))
    step_labels = [STEP_NAMES[s] for s in steps]
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(steps)))

    for ax, metric, ylabel, fmt in [
        (ax1, "stps_avg", "System Tokens Per Second (STPS)", ".0f"),
        (ax2, "tokens_per_joule", "Tokens / Joule", ".4f"),
    ]:
        vals = []
        heats = []
        for step in steps:
            r = sub[sub["step"] == step]
            vals.append(r[metric].values[0] if len(r) > 0 else 0)
            heats.append(r["heating_feasible"].values[0] if len(r) > 0 else True)

        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)

        for i, (h, v) in enumerate(zip(heats, vals)):
            if not h:
                bars[i].set_edgecolor("red")
                bars[i].set_linewidth(2.5)
            # value label on top
            ax.text(i, v * 1.01, f"{v:{fmt}}", ha="center", va="bottom", fontsize=7.5)

        # Percentage gain annotations
        for i in range(1, len(vals)):
            if vals[i - 1] > 0:
                pct = (vals[i] / vals[i - 1] - 1) * 100
                if abs(pct) > 0.5:
                    ax.annotate(f"+{pct:.1f}%", xy=(i, vals[i]),
                                xytext=(i - 0.3, vals[i] * 0.85),
                                fontsize=6.5, color="green" if pct > 0 else "gray",
                                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_xlim(-0.6, len(steps) - 0.4)

    plt.tight_layout()
    chart_path = OUT_DIR / f"ablation_bs{bs}.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    print(f"Saved chart: {chart_path}")
    plt.close()


def plot_ablation(summary_df):
    """Generate one chart per BS (bs=4 and bs=1024), each with STPS + Tokens/J."""
    os.makedirs(OUT_DIR, exist_ok=True)
    steps = list(range(1, 8))
    for bs in BATCH_SIZES:
        _plot_single_bs(summary_df, bs, steps)


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.chdir(str(PROJECT_ROOT))

    # Phase 1: Load data
    merged_df = load_merged_csv()

    # Phase 2: Steps 1-5
    results_1_to_5 = run_steps_1_to_5(merged_df)

    # Phase 3: Step 6
    step6_results, step6_candidates = run_step_6()

    # Phase 4: Step 7
    step7_results = run_step_7(step6_results, step6_candidates)

    # Merge all results
    all_results = {}
    for step in range(1, 6):
        all_results[step] = results_1_to_5.get(step, {})
    all_results[6] = step6_results
    all_results[7] = step7_results

    # Phase 5: Summary & visualization
    summary_df = build_summary(all_results)
    make_ablation_table(summary_df)
    plot_ablation(summary_df)

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
