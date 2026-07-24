import logging
from mosaic.parallelism import ParallelScheme
from mosaic.noc.noc_topo import Hierarchy
from mosaic.utils import Modeling_Granularity
from mosaic.llm_arch import LLM_Arch
from tilesight.arch import *

from .dsa_decode_coarse import dsa_mla_decode_coarse, dsa_mla_decode_kv_list_coarse
from .dsa_prefill_coarse import dsa_mla_prefill_coarse

log = logging.getLogger(__name__)

# DeepSeek-V3.2 / GLM-5.1 等 DSA (DeepSeek Sparse Attention) 模型的 attention block 入口,
# 接口与 mla_decode_top / mla_prefill_top 对齐。
# 适用于 model_arch.mla_arch 和 model_arch.dsa_arch 都不为 None 的模型。


def dsa_mla_decode_top(bs:int, seq:int, cached_kv:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    assert parallel.ep == 1

    if seq == 1:
        assert parallel.sp == 1

    assert model_arch.mla_arch is not None
    assert model_arch.dsa_arch is not None

    if granularity.get_mode() == "coarse":
        return dsa_mla_decode_coarse(bs, seq, cached_kv, model_arch, parallel, atten_parallel, next_parallel, granularity, single_chip, noc_hierarchy)
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass


def dsa_mla_decode_kv_list_top(bs:int, seq:int, cached_kv_list:list[int], model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    assert parallel.ep == 1

    if seq == 1:
        assert parallel.sp == 1

    assert model_arch.mla_arch is not None
    assert model_arch.dsa_arch is not None

    if granularity.get_mode() == "coarse":
        return dsa_mla_decode_kv_list_coarse(bs, seq, cached_kv_list, model_arch, parallel, atten_parallel, next_parallel, granularity, single_chip, noc_hierarchy)
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass


def dsa_mla_prefill_top(bs:int, seq:int, model_arch:LLM_Arch, parallel:ParallelScheme, atten_parallel:ParallelScheme,
     next_parallel:ParallelScheme, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):

    assert parallel.ep == 1

    if seq == 1:
        assert parallel.sp == 1

    assert model_arch.mla_arch is not None
    assert model_arch.dsa_arch is not None

    if granularity.get_mode() == "coarse":
        return dsa_mla_prefill_coarse(bs, seq, model_arch, parallel, atten_parallel, next_parallel, granularity, single_chip, noc_hierarchy)
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from mosaic.llm_arch.deepseek_v3_2 import DeepSeekV3_2
    from mosaic.llm_arch.deepseek_v3 import DeepSeekV3
    from mosaic.dse_space.arch_noc_combinations import b200_8x1_8_arch_noc_combinations

    combinations = b200_8x1_8_arch_noc_combinations()
    single_chip = combinations[0][0]
    noc_hierarchy = combinations[0][1]

    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False, dump_perf_log=False)

    model_arch = DeepSeekV3_2()
    model_arch_v3 = DeepSeekV3()

    # ----------------- decode -----------------
    parallel =       ParallelScheme(tp=8, ep=1, sp=1, cp=1, dp=1, pp=1, fsdp=False)
    bs = 64
    seq = 1
    cached_kv_list = [4*1024, 32*1024, 128*1024]

    from mosaic.llm.mla.mla_decode_top import mla_decode_kv_list_top

    time_list, stats_list = dsa_mla_decode_kv_list_top(bs=bs, seq=seq, cached_kv_list=cached_kv_list, model_arch=model_arch, parallel=parallel, atten_parallel=parallel, next_parallel=parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    time_list_v3, _ = mla_decode_kv_list_top(bs=bs, seq=seq, cached_kv_list=cached_kv_list, model_arch=model_arch_v3, parallel=parallel, atten_parallel=parallel, next_parallel=parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    for kv, t, t_v3 in zip(cached_kv_list, time_list, time_list_v3):
        log.info("decode bs=%s kv=%s: DSA (v3.2) %.6f s vs dense MLA (v3) %.6f s, speedup %.2fx", bs, kv, t, t_v3, t_v3/t)

    # ----------------- prefill -----------------
    parallel =       ParallelScheme(tp=8, ep=1, sp=8, cp=1, dp=1, pp=1, fsdp=False)
    bs = 4
    seq = 32*1024

    time_prefill, _ = dsa_mla_prefill_top(bs=bs, seq=seq, model_arch=model_arch, parallel=parallel, atten_parallel=parallel, next_parallel=parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)
    from mosaic.llm.mla.mla_prefill_top import mla_prefill_top
    time_prefill_v3, _ = mla_prefill_top(bs=bs, seq=seq, model_arch=model_arch_v3, parallel=parallel, atten_parallel=parallel, next_parallel=parallel, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy)

    log.info("prefill bs=%s seq=%s: DSA (v3.2) %.6f s vs dense MLA (v3) %.6f s, speedup %.2fx", bs, seq, time_prefill, time_prefill_v3, time_prefill_v3/time_prefill)

    # ----------------- footprint -----------------
    from mosaic.llm.dsa.dsa_footprint import get_dsa_mla_absorb_and_no_absorb_footprint
    max_act, mem_w, mem_kv = get_dsa_mla_absorb_and_no_absorb_footprint(bs=64, seq=1, cached_kv=128*1024, model_arch=model_arch, parallel=ParallelScheme(tp=8, ep=1, sp=1, cp=1, dp=1, pp=1, fsdp=False), atten_parallel=ParallelScheme(tp=8, ep=1, sp=1, cp=1, dp=1, pp=1, fsdp=False))
    log.info("dsa+mla decode footprint per layer: max_activation %.3f GiB, mem_weight %.3f GiB, mem_kv_cache(+index cache) %.3f GiB", max_act/(1024**3), mem_w/(1024**3), mem_kv/(1024**3))
