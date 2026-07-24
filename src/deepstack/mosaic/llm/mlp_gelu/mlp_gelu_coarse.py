# 非门控 2-GEMM MLP (encoder 用): y = W2 @ act(W1 @ x)
# 对应 Qwen3OmniMoeVisionMLP / AudioEncoderLayer fc1-fc2 / TalkerResizeMLP 等 gelu MLP。
# 结构照 swiglu_coarse: stage1 = x@W1 + gelu (smem 融合), stage2 = @W2 + tp 归约。
import torch
import math
import numpy as np
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
from mosaic.op_dtype.element_wrapper import element_wrapper

from tilesight.arch import *
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
import logging
log = logging.getLogger(__name__)
from mosaic.collectives import all_reduce_wrapper
from mosaic.collectives import reduce_scatter_wrapper
from mosaic.utils import get_comp_comm_e2e_time
from mosaic.cost.op_perf_stats import OpPerfStats


def mlp_gelu_coarse_stage1(bs:int, seq:int, hidden:int, up_hidden:int, parallel:ParallelScheme, mlp_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    assert granularity.get_mode() == "coarse"

    in_bytes, weight_bytes, out_bytes = mlp_bytes.get_dtype_bytes()
    shard_up_hidden = math.ceil(up_hidden / parallel.tp)
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    # x: [bs/dp, seq/sp, hidden]
    # W1 [hidden, up_hidden/tp]
    # x@W1 -> [bs/dp, seq/sp, up_hidden/tp], smem
    # gelu -> [bs/dp, seq/sp, up_hidden/tp], ddr

    if (parallel.fsdp == False or parallel.dp == 1):
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1 = 0, 0, 0
    else:
        tm = TrafficMatrix(parallel.world_size())
        tm.add_intra_group_traffic("dp", hidden*shard_up_hidden*weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1), tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=noc_hop_time_1, link_time_s=noc_ext_max_1)
        log.info("mlp_gelu stage 1 fsdp, noc_overall_time: %s", noc_overall_time_1)

    smem_fusion_list = []

    gemm1_bytes = OpBytes(
        input1=Tensor_Loc(mlp_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(mlp_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(mlp_bytes.output.dtype, 'smem'),
    )
    hete_post_data_gemm, gemm_smem_fusion_post_data, gemm_tiling_config = gemm_wrapper(M=shard_bs*shard_seq, N=shard_up_hidden, K=hidden, gemm_bytes=gemm1_bytes, granularity=granularity, single_chip=single_chip)
    grids = [shard_bs * shard_seq / gemm_tiling_config[0], shard_up_hidden / gemm_tiling_config[1]]
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gemm], single_chip))

    if (granularity.get_comp_comm_overlap() == True):
        additional_time = max(gemm_smem_fusion_post_data[0], noc_overall_time_1) - gemm_smem_fusion_post_data[0]
    else:
        additional_time = noc_overall_time_1

    # gelu: tanh 近似 = 常数乘加(cuda) + tanh(sfu) + 乘(cuda), 按 sfu 1 pass + cuda 2 pass 建模
    gelu_tb_tile = (gemm_tiling_config[0], gemm_tiling_config[1])
    gelu_bytes_sfu = OpBytes(
        input1=Tensor_Loc(mlp_bytes.input1.dtype, 'smem', [shard_bs, shard_seq, shard_up_hidden]),
        input2=None,
        output=Tensor_Loc(mlp_bytes.output.dtype, 'reg', [shard_bs, shard_seq, shard_up_hidden]),
    )
    hete_post_data_gelu_1, _ = element_wrapper(element_op_bytes=gelu_bytes_sfu, granularity=granularity, single_chip=single_chip, batch=1, type="sfu_core", tb_tiling_config=gelu_tb_tile)

    gelu_bytes_cuda = OpBytes(
        input1=Tensor_Loc(mlp_bytes.input1.dtype, 'reg', [shard_bs, shard_seq, shard_up_hidden]),
        input2=None,
        output=Tensor_Loc(mlp_bytes.output.dtype, 'ddr', [shard_bs, shard_seq, shard_up_hidden]),
    )
    hete_post_data_gelu_2, _ = element_wrapper(element_op_bytes=gelu_bytes_cuda, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=gelu_tb_tile)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_gelu_1, hete_post_data_gelu_2], single_chip))

    smem_fusion_post_data = hete_smem_fusion(smem_fusion_list, grids, single_chip)
    single_chip_time = smem_fusion_post_data[0]
    log.info("mlp_gelu stage 1 single chip time: %s s, grids: %s", single_chip_time, grids)

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    return single_chip_time + additional_time


def mlp_gelu_coarse_stage2(bs:int, seq:int, hidden:int, up_hidden:int, out_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, mlp_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, stats:"OpPerfStats | None" = None):
    assert granularity.get_mode() == "coarse"

    in_bytes, weight_bytes, out_bytes = mlp_bytes.get_dtype_bytes()
    shard_up_hidden = math.ceil(up_hidden / parallel.tp)
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    # act[bs/dp, seq/sp, up_hidden/tp] @ W2[up_hidden/tp, out_hidden] -> [bs/dp, seq/sp, out_hidden]
    # tp-level 归约

    if (parallel.fsdp == False or parallel.dp == 1):
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1 = 0, 0, 0
    else:
        tm = TrafficMatrix(parallel.world_size())
        tm.add_intra_group_traffic("dp", shard_up_hidden*out_hidden*weight_bytes * (parallel.dp-1)/parallel.dp/(parallel.dp-1), tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        if stats is not None:
            stats.append_traffic(noc_traffic_1, hop_time_s=noc_hop_time_1, link_time_s=noc_ext_max_1)
        log.info("mlp_gelu stage 2 fsdp, noc_overall_time: %s", noc_overall_time_1)

    gemm2_bytes = OpBytes(
        input1=Tensor_Loc(mlp_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(mlp_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(mlp_bytes.output.dtype, 'ddr'),
    )
    hete_post_data, smem_fusion_post_data, tiling_config = gemm_wrapper(M=shard_bs*shard_seq, N=out_hidden, K=shard_up_hidden, gemm_bytes=gemm2_bytes, granularity=granularity, single_chip=single_chip)
    log.info("mlp_gelu stage 2 single device time: %s s", smem_fusion_post_data[0])
    if (granularity.get_comp_comm_overlap() == True):
        time_stage2 = max(smem_fusion_post_data[0], noc_overall_time_1)
    else:
        time_stage2 = smem_fusion_post_data[0] + noc_overall_time_1

    if parallel.tp == 1:
        all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = 0, 0, None
    elif parallel.tp > 1 and ((next_parallel == parallel) or (parallel.tp * parallel.dp * parallel.sp == next_parallel.tp * next_parallel.dp * next_parallel.sp)):
        all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = reduce_scatter_wrapper(
            all_reduce_op_bytes=mlp_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy,
            granularity=granularity, dim_to_process="tp", bytes=shard_bs*shard_seq*out_hidden*weight_bytes)
    elif parallel.tp > 1 and (next_parallel.tp == 1) and parallel.dp == next_parallel.dp and parallel.sp == next_parallel.sp:
        all_reduce_hop, all_reduce_ext_max, all_reduce_traffic = all_reduce_wrapper(
            all_reduce_op_bytes=mlp_bytes, parallel=parallel, noc_hierarchy=noc_hierarchy,
            granularity=granularity, dim_to_process="tp", bytes=shard_bs*shard_seq*out_hidden*weight_bytes)
    else:
        log.error("Invalid parallel scheme between current and next. Current Parallel: %s, Next Parallel: %s", parallel, next_parallel)
        raise ValueError("Invalid parallel scheme", parallel, next_parallel)

    if stats is not None and all_reduce_traffic is not None:
        stats.append_traffic(all_reduce_traffic, hop_time_s=all_reduce_hop, link_time_s=all_reduce_ext_max)

    grids = [shard_bs * shard_seq / tiling_config[0], out_hidden / tiling_config[1]]
    waves = np.prod(grids) / single_chip.sm_count
    e2e_time = get_comp_comm_e2e_time(compute_time=time_stage2, network_hop_latency=all_reduce_hop, network_link_time=all_reduce_ext_max, waves=waves, overlap=granularity.get_comp_comm_overlap())
    log.info("mlp_gelu stage 2 comp+comm e2e time: %s s", e2e_time)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    return e2e_time


def mlp_gelu_coarse(bs:int, seq:int, hidden:int, up_hidden:int, out_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, mlp_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    assert granularity.get_mode() == "coarse"

    stats = OpPerfStats(op_name="mlp_gelu", dump_perf_log=granularity.dump_perf_log) if granularity.dump_perf_log else None

    time_stage1 = mlp_gelu_coarse_stage1(bs=bs, seq=seq, hidden=hidden, up_hidden=up_hidden, parallel=parallel, mlp_bytes=mlp_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)
    time_stage2 = mlp_gelu_coarse_stage2(bs=bs, seq=seq, hidden=hidden, up_hidden=up_hidden, out_hidden=out_hidden, parallel=parallel, next_parallel=next_parallel, mlp_bytes=mlp_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, stats=stats)

    total_time = time_stage1 + time_stage2
    log.info("mlp_gelu total time: %s s (stage1: %s, stage2: %s)", total_time, time_stage1, time_stage2)
    return total_time, stats
