# Qwen3-Omni 全组件 DSE 入口 (v1, Phase 1 同集群串行模式)。
#
# 对每个组件枚举并行方案 (tp/ep/dp 因子分解, world_size 固定), 逐 plan 调
# modeling_omni_e2e, 输出 CSV (按 first_audio_latency 排序)。
# 用法: python -m mosaic.dse_space.dse_framework_omni_v1 [num_workers]
import csv
import itertools
import logging
import math
import multiprocessing as mp
import os
import sys

from mosaic.llm_arch import Qwen3_Omni_30b_a3b
from mosaic.parallelism import ParallelScheme
from mosaic.noc.noc_config_set import torus_mesh_switch_1
from mosaic.noc.noc_topo import Hierarchy
from mosaic.utils import Modeling_Granularity
from mosaic.arch import stacked_gpu_base

from mosaic.dse_space.modeling_omni import OmniParallelPlan, OmniWorkload, modeling_omni_e2e

log = logging.getLogger(__name__)

WORLD_SIZE = 8

MODEL_REG = {
    "Qwen3_Omni_30b_a3b": Qwen3_Omni_30b_a3b,
}


def make_noc_hierarchy() -> Hierarchy:
    return torus_mesh_switch_1()


def dense_scheme_candidates(world: int):
    """dense 组件 (encoder / attention / c2w): tp x dp = world, sp=1 (decode 兼容)。"""
    out = []
    for tp in [1, 2, 4, 8, 16]:
        if tp > world or world % tp:
            continue
        dp = world // tp
        out.append(ParallelScheme(tp=tp, ep=1, sp=1, cp=1, dp=dp, pp=1, fsdp=False))
    return out


def moe_scheme_candidates(world: int):
    """MoE 组件: tp x ep = world。"""
    out = []
    for ep in [1, 2, 4, 8, 16]:
        if ep > world or world % ep:
            continue
        tp = world // ep
        out.append(ParallelScheme(tp=tp, ep=ep, sp=1, cp=1, dp=1, pp=1, fsdp=False))
    return out


def _scheme_str(p: ParallelScheme) -> str:
    return f"tp{p.tp}ep{p.ep}sp{p.sp}dp{p.dp}pp{p.pp}"


def _eval_one(args):
    (model_key, thinker_scheme, thinker_moe, talker_scheme, talker_moe, enc_scheme, c2w_scheme) = args

    omni = MODEL_REG[model_key]()
    noc_hierarchy = make_noc_hierarchy()
    single_chip = stacked_gpu_base()
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False, dump_perf_log=False)

    plan = OmniParallelPlan(
        vision=enc_scheme, audio=enc_scheme,
        thinker_prefill=thinker_scheme, thinker_prefill_atten=thinker_scheme, thinker_moe=thinker_moe,
        thinker_decode=thinker_scheme,
        talker_prefill=talker_scheme, talker_decode=talker_scheme, talker_moe=talker_moe,
        code_predictor=talker_scheme, code2wav=c2w_scheme,
    )
    wl = OmniWorkload(
        bs=1, text_in_tokens=128,
        num_images=1, image_height=1080, image_width=1920,
        audio_in_seconds=30.0,
        text_out_tokens=256, audio_out_seconds=20.0,
    )
    try:
        m = modeling_omni_e2e(omni, wl, plan, granularity, single_chip, noc_hierarchy)
    except Exception as e:
        log.warning("plan failed: %s (%s)", args, e)
        return None
    return {
        "thinker": _scheme_str(thinker_scheme), "thinker_moe": _scheme_str(thinker_moe),
        "talker": _scheme_str(talker_scheme), "talker_moe": _scheme_str(talker_moe),
        "encoder": _scheme_str(enc_scheme), "code2wav": _scheme_str(c2w_scheme),
        "TTFT": m["TTFT"], "text_TPOT": m["text_TPOT"],
        "first_audio_latency": m.get("first_audio_latency"),
        "audio_RTF": m.get("audio_RTF"), "audio_RTF_colocated": m.get("audio_RTF_colocated"),
        "t_vision_encoder": m["t_vision_encoder"], "t_audio_encoder": m["t_audio_encoder"],
        "t_thinker_prefill": m["t_thinker_prefill"],
    }


def dse_1(run_dir: str = "omni_dse_out", num_workers: int = 8, model_key: str = "Qwen3_Omni_30b_a3b"):
    os.makedirs(run_dir, exist_ok=True)

    thinker_dense = dense_scheme_candidates(WORLD_SIZE)
    thinker_moe = moe_scheme_candidates(WORLD_SIZE)
    talker_dense = dense_scheme_candidates(WORLD_SIZE)
    talker_moe = moe_scheme_candidates(WORLD_SIZE)
    enc = dense_scheme_candidates(WORLD_SIZE)
    c2w = dense_scheme_candidates(WORLD_SIZE)

    # 组合裁剪: talker/cp/c2w 较小, encoder 与 c2w 用同一候选集; 全笛卡尔积会爆炸,
    # 按 (thinker, thinker_moe) x (talker, talker_moe) x encoder x c2w 组合
    tasks = [(model_key, a, b, c, d, e, f)
             for a, b, c, d, e, f in itertools.product(thinker_dense, thinker_moe, talker_dense, talker_moe, enc, c2w)]
    log.info("total %s plans", len(tasks))

    if num_workers <= 1:
        results = [_eval_one(t) for t in tasks]
    else:
        with mp.Pool(num_workers) as pool:
            results = pool.map(_eval_one, tasks)
    results = [r for r in results if r is not None]
    results.sort(key=lambda r: (r["first_audio_latency"] if r["first_audio_latency"] is not None else float("inf")))

    out_csv = os.path.join(run_dir, "omni_dse_v1.csv")
    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    log.info("wrote %s rows to %s", len(results), out_csv)
    if results:
        best = results[0]
        log.info("best plan: %s", best)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 8) - 2)
    dse_1(num_workers=num_workers)
