"""
quick_test.py — 最小化测试: 1 个 DRAM config × 1 个 prefill task + 1 个 decode task
验证代码逻辑和 CSV 输出是否正确.

用法:  python -m mosaic.dse_space.case_study_dram_layer.quick_test
"""

import logging
import os
import sys
import time
import multiprocessing as mp

# ── monkey-patch: 缩小搜索空间 ──────────────────────────────────────────
import mosaic.dse_space.case_study_dram_layer.dram_layer_config as _cfg

_ORIG_GENERATE = _cfg.generate_dram_layer_configs


def _tiny_configs():
    """只返回 baseline m=4,n=4,smem=256K,l1=256."""
    return [(4, 4, 256 * 1024, 256)]


_cfg.generate_dram_layer_configs = _tiny_configs

# ── monkey-patch prefill tasks ───────────────────────────────────────────
import mosaic.dse_space.case_study_dram_layer.dram_layer_dse_prefill as _pf


def _tiny_prefill_tasks():
    from mosaic.llm_arch import DeepSeekV3
    return [(DeepSeekV3(), 16, 1024, 1024, 1024)]


_pf.dram_layer_prefill_tasks = _tiny_prefill_tasks

# ── monkey-patch decode tasks ────────────────────────────────────────────
import mosaic.dse_space.case_study_dram_layer.dram_layer_dse_decode as _dc


def _tiny_decode_tasks():
    from mosaic.llm_arch import DeepSeekV3
    return [(DeepSeekV3(), 16, 1024, 2048, 1)]


_dc.dram_layer_decode_tasks = _tiny_decode_tasks

# ── run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import datetime

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_quicktest"
    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs", run_tag
    )
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "dse.log")
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    mp.set_start_method("spawn", force=True)
    num_workers = min(4, (os.cpu_count() or 2) // 2)

    # ── Prefill ──
    print("=" * 60)
    print("Running PREFILL quick test ...")
    print("=" * 60)
    t0 = time.time()
    _pf.dse_1(run_dir, num_workers, enable_proc_log=False)
    print(f"Prefill done in {time.time() - t0:.1f}s")

    # ── Decode ──
    print("=" * 60)
    print("Running DECODE quick test ...")
    print("=" * 60)
    t0 = time.time()
    _dc.dse_1(run_dir, num_workers, enable_proc_log=False)
    print(f"Decode done in {time.time() - t0:.1f}s")

    # ── 打印 CSV 结果 ──
    print("\n" + "=" * 60)
    print(f"Results in: {run_dir}")
    print("=" * 60)
    for fname in sorted(os.listdir(run_dir)):
        if fname.endswith(".csv"):
            fpath = os.path.join(run_dir, fname)
            size = os.path.getsize(fpath)
            with open(fpath) as f:
                lines = f.readlines()
            print(f"\n--- {fname} ({len(lines)} lines, {size} bytes) ---")
            for line in lines[:5]:  # header + first few rows
                print(line.rstrip())
            if len(lines) > 5:
                print(f"  ... ({len(lines) - 5} more rows)")
