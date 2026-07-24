# rabenseifner 的前半部分其实可以被视作是一种reduce scatter
# 而且其实正好，all reduce 就是tp 维度，然后scatter也是在这个tp维度


from mosaic.utils import OpBytes, Modeling_Granularity,Tensor_Loc
from tilesight.arch import Arch
import math
# from tilesight.fused_op_dtype.matmul_fused_op_new_api import calculate_matmul_resource_utilization_new
from tilesight.fused_op_dtype_wave.N_0_element_wise_fused_op_wave import calculate_N_0_elementwise_resource_utilization
from tilesight.fused_op_dtype_wave.N_1_element_wise_fused_op_wave import calculate_N_1_elementwise_resource_utilization
from tilesight.fused_op_dtype_wave.N_N_element_wise_fused_op_wave import calculate_N_N_elementwise_resource_utilization

from tilesight.fusion_support.hete_post_process_single_op import hete_post_process_tensor_core_op,hete_post_process_sfu_core_op,hete_post_process_cuda_core_op

from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
# tiling_configs
# tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages
from functools import lru_cache
import torch
import numpy as np
import logging
log = logging.getLogger(__name__)

from mosaic.parallelism import ParallelScheme
from numpy import uint64
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, make_mesh_or_torus, make_switch, TopoKind, PortSpread, get_extend_max_routes, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity



def reduce_scatter_rabenseifner(all_reduce_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    
    dim_to_process = dim_to_process.lower()
    if dim_to_process not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
        raise ValueError("dim_to_process 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

    dim_size_map = {"tp": parallel.tp, "ep": parallel.ep, "sp": parallel.sp, "cp": parallel.cp, "dp": parallel.dp, "pp": parallel.pp}
    
    gsize = int(dim_size_map[dim_to_process])  # which -> 该维度的组大小
    
    # assert gsize is power of 2
    assert gsize & (gsize - 1) == 0, log.error("gsize must be power of 2, gsize: %s", gsize)

    stages = int(math.log2(gsize))

    bytes_each_device = bytes 
    # 逐 stage 的递归加倍交换，stride = 2^stage
    total_noc_hop_time = 0
    total_noc_ext_max = 0

    tm = TrafficMatrix(parallel.world_size())

    total_traffic = None
    for stage in range(stages):
        stride = 1 << stage
        # rabensiefer, many stages, less data
        bytes_each_iter = bytes_each_device / (1<<(stage+1))


        traffic_pairs = []
        for i in range(gsize):
            partner = i ^ stride
            traffic_pairs.append([bytes_each_iter, i, partner])

        tm.add_intra_group_traffic_pair_bulk(
            dim_to_process,
            traffic_pairs,
            tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
        )

        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        total_noc_hop_time += noc_hop_time_1
        total_noc_ext_max += noc_ext_max_1
        if total_traffic is None:
            total_traffic = noc_traffic_1.copy()
        else:
            total_traffic += noc_traffic_1

        # tm.export_json(f"reduce_scatter_rabenseifner_stage_{stage}.json")
        # tm.export_json(f"reduce_scatter_rabenseifner.json")
        # tm.save_heatmap(f"reduce_scatter_rabenseifner.png")

        tm.reset()

    # no broadcast process, so only half is enough
    log.info(
        "reduce_scatter_rabenseifner, stages: %s, total_noc_hop_time: %s, total_noc_ext_max: %s",
        stages,
        total_noc_hop_time,
        total_noc_ext_max,
    )

    return total_noc_hop_time, total_noc_ext_max, total_traffic




def reduce_scatter_wrapper(all_reduce_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    bytes = uint64(bytes)

    # 统一走 auto-tune 评估多种算法

    latency_rab, link_rab, traffic_rab = reduce_scatter_rabenseifner(all_reduce_op_bytes, parallel, noc_hierarchy, granularity, dim_to_process, bytes)



    total_rab = latency_rab + link_rab


    best_algo = "rabenseifner"
    best_latency = latency_rab
    best_link = link_rab
    best_traffic = traffic_rab
    best_total = total_rab


    log.info(
        "all_reduce_select: algo=%s, latency=%s, link=%s, total=%s; candidates: rabenseifner(total=%s), ",
        best_algo,
        best_latency,
        best_link,
        best_total,
        total_rab,
    )

    return best_latency, best_link, best_traffic

    # methods: recursive doubling, ring, rabenseifner

    # return max_latency, max_link_time

if __name__ == "__main__":
    stages=3
    gsize=8
    bytes_each_device=1024
    # parallel = ParallelScheme(tp=8, ep=8, sp=8, cp=8, dp=8, pp=8)

    for stage in range(stages):
        stride = 1 << stage
        # tm = TrafficMatrix(parallel.world_size())
        tm = TrafficMatrix(256)

        traffic_pairs = []
        for i in range(gsize):
            partner = i ^ stride
            traffic_pairs.append([bytes_each_device, i, partner])
    
        print(traffic_pairs)

    for stage in range(stages):
        stride = 1 << stage

        bytes_each_iter = int(bytes_each_device / (1<<(stage+1)))

        print(bytes_each_iter)

    # gsize = int(dim_size_map[dim_to_process])  # which -> 该维度的组大小

    gsize = 32
    
    # assert gsize is power of 2
    assert gsize & (gsize - 1) == 0, log.error("gsize must be power of 2, gsize: %s", gsize)

    stages = int(math.log2(gsize))

    # 双二叉树：A 树朝右根 (gsize-1)，B 树朝左根 (0)
    total_noc_hop_time = 0
    total_noc_ext_max = 0

    # tm = TrafficMatrix(parallel.world_size())
    tm = TrafficMatrix(256)

    # 每棵树承担一半数据量
    bytes_each_iter = bytes_each_device /2

    # 构建 A 树各层（从偶数叶子到右根）
    a_levels = []  # 每层若干 [src, dst]
    current_nodes = list(range(0, gsize, 2))  # 偶数
    while len(current_nodes) >= 2:
        next_nodes = []
        level_pairs = []
        for j in range(0, len(current_nodes), 2):
            left = current_nodes[j]
            right = current_nodes[j + 1]
            parent = (left + right) // 2  # 奇数
            level_pairs.append([left, parent])
            level_pairs.append([right, parent])
            next_nodes.append(parent)
        a_levels.append(level_pairs)
        current_nodes = next_nodes
    if len(current_nodes) == 1:
        # 最后一跳到右根
        a_levels.append([[current_nodes[0], gsize - 1]])

    # 构建 B 树各层（从奇数叶子到左根）
    b_levels = []
    current_nodes = list(range(1, gsize, 2))  # 奇数
    while len(current_nodes) >= 2:
        next_nodes = []
        level_pairs = []
        for j in range(0, len(current_nodes), 2):
            left = current_nodes[j]
            right = current_nodes[j + 1]
            parent = (left + right) // 2  # 偶数
            level_pairs.append([left, parent])
            level_pairs.append([right, parent])
            next_nodes.append(parent)
        b_levels.append(level_pairs)
        current_nodes = next_nodes
    if len(current_nodes) == 1:
        # 最后一跳到左根
        b_levels.append([[current_nodes[0], 0]])

    assert len(a_levels) == stages and len(b_levels) == stages, \
        f"double_tree levels mismatch, a: {len(a_levels)}, b: {len(b_levels)}, stages: {stages}"

    # 向上规约阶段（A、B 同步进行，每一层为一 stage）
    for lvl in range(stages):
        traffic_pairs = []
        for src, dst in a_levels[lvl]:
            traffic_pairs.append([bytes_each_iter, src, dst])
        for src, dst in b_levels[lvl]:
            traffic_pairs.append([bytes_each_iter, src, dst])

        print(f"stage {lvl}")
        print(traffic_pairs)
        tm.reset()

    # 反向广播阶段（从根向叶，反向边，层序反转）
    for lvl in range(stages - 1, -1, -1):
        traffic_pairs = []
        for src, dst in a_levels[lvl]:
            traffic_pairs.append([bytes_each_iter, dst, src])
        for src, dst in b_levels[lvl]:
            traffic_pairs.append([bytes_each_iter, dst, src])

        print(f"stage {lvl}")
        print(traffic_pairs)


        tm.reset()