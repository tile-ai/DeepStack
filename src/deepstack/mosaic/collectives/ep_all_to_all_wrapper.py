from mosaic.utils import OpBytes, Modeling_Granularity,Tensor_Loc
from tilesight.arch import Arch
import math
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
from pathlib import Path



# def ep_all_to_all_wrapper(
#     parallel: ParallelScheme,
#     noc_hierarchy: Hierarchy,
#     granularity: Modeling_Granularity,
#     bytes_each_token: int,
#     # routing_list: list,
#     routing_array: np.ndarray,
#     bs: int,
#     seq: int,
#     num_routed_experts: int,
#     num_activated_experts: int,
#     imbalance_overhead_ratio: float,
# ):
#     """
#     # routing_list: list[list[int]]
#     routing_array: np.ndarray,
#         第 t 个 token 的路由专家 id 列表（如 [6, 19, 98, ...]），只使用前 num_activated_experts 个。
#         专家 id 的取值范围为 [0, num_routed_experts).
#         正反完全对偶,互为逆操作,即src dst互换, 不改变负载、延迟与总时间
#     """

#     tm = TrafficMatrix(parallel.world_size())

#     if (bs * seq >= 1024):
#         # 规模较大，使用平均 + 不平衡因子
#         bytes_per_pair = math.ceil(bs * seq / parallel.dp / parallel.sp / parallel.ep) \
#                          * num_activated_experts / parallel.ep * bytes_each_token * imbalance_overhead_ratio
#         log.info("bytes_per_pair: %s", bytes_per_pair)

#         # 直接用组内 all-to-all（平均），底层实现在同一并行维度内生成成对通信
#         tm.add_intra_group_traffic(
#             "ep",
#             bytes_per_pair,  # 基础单位字节；内部按平均/组合扩展
#             tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp
#         )

#         noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
#         log.info("noc_hop_time_1: %s, noc_ext_max_1: %s, noc_overall_time_1: %s",
#                  noc_hop_time_1, noc_ext_max_1, noc_overall_time_1)
        
#     else:
#         # -----------------------------
#         # 细粒度仿真：按 routing_list 真实分发
#         # -----------------------------
#         total_tokens = bs * seq

#         # 检查shape
#         shape_of_array = routing_array.shape
#         # print(f"routing_array.shape (before reshape): {shape_of_array}")
#         log.info("routing_array.shape (before reshape): %s", shape_of_array)
#         if routing_array.ndim >= 2:
#             routing_array = routing_array.reshape(-1, routing_array.shape[-1])
#         else:
#             raise ValueError("routing_array 的维度应≥2，形如 [*, num_activated_experts]")

#         # 检查是否有足够的 token 行数
#         tokens_available, experts_per_token = routing_array.shape
#         assert tokens_available >= total_tokens, (
#             f"routing_array 行数不足! 需要至少 {total_tokens} 行, 实际 {tokens_available}."
#         )

#         group_size = parallel.ep
#         ep_groups = math.ceil(parallel.world_size() / group_size)

#         # -----------------------------------------------
#         # ✅ 新增：定义“数据窗口”的数量（用于循环复用 token）
#         # 每 dp*sp 个 EP 组共享一份 token 数据
#         # -----------------------------------------------
#         groups_per_data = parallel.dp * parallel.sp

#         # 每个数据窗口负责 tokens_each_ep_group 个 token
#         tokens_each_ep_group = math.ceil(total_tokens / groups_per_data)

#         # =========================================
#         # 从外到内：
#         # 0,.... ep_groups-1
#         # 每个 ep_group 代表一个独立的 Expert Parallel 组。
#         # 每个组内包含 ep 个进程（rank 0 ~ ep-1）。
#         #
#         # ✅ 修改：
#         #   当 ep_group_index >= groups_per_data 时，循环复用之前的数据窗口。
#         #   即 ep_group_index 8 → 使用 data_slot 0，9→1，依此类推。
#         # =========================================
#         for ep_group_index in range(ep_groups):

#             # ✅ 关键修改：循环复用数据窗口
#             data_slot = ep_group_index % groups_per_data

#             # 当前 EP 组对应的 routing_list 的 token 区间：
#             # [group_token_start, group_token_end)
#             group_token_start = data_slot * tokens_each_ep_group
#             group_token_end = min(group_token_start + tokens_each_ep_group, total_tokens)

#             # 若没有可用 token（例如组数很多而 token 用完），跳过
#             if group_token_start >= group_token_end:
#                 continue

#             local_tokens = group_token_end - group_token_start

#             # 每个 src（EP rank）持有的 token 数量大致相等，前 rem 个多一个
#             base = local_tokens // group_size
#             rem = local_tokens % group_size

#             # =========================================
#             # 从外到内：
#             # 0,.... ep_groups-1
#             # 每个 ep_group 代表一个独立的 Expert Parallel 组。
#             # 每个组内包含 ep 个进程（rank 0 ~ ep-1），负责 tokens_each_ep_group 个 token。
#             # 对于每个组：
#             #   - 按 src (EP rank) 划分输入 token 段
#             #   - 对每个 token，从 routing_list 读取它的专家列表
#             #   - 根据专家 id 决定目标 dst rank（同组内）
#             #   - 如果 src == dst，则认为是本地，不产生通信
#             # =========================================
#             for src_index in range(group_size):
#                 # ----------------------------------
#                 # 当前 src_index 表示组内哪个 EP rank
#                 # 例如 ep=8，则 src_index ∈ [0..7]
#                 #
#                 # 该 src 持有 tokens 的数量：base 或 base+1
#                 # 例如 tokens_each_ep_group=100, ep=8 =>
#                 #   base=12, rem=4 → 前4个 src 各13个，后4个 src 各12个
#                 # ----------------------------------
#                 count = base + (1 if src_index < rem else 0)
#                 start = src_index * base + min(src_index, rem)
#                 abs_start = group_token_start + start

#                 if count == 0:
#                     continue

#                 # 初始化一个数组来统计发往每个 dst 的总字节
#                 bytes_to_each_dst = [0] * group_size

#                 # 遍历该 src 负责的 tokens
#                 for local_t in range(count):
#                     t = abs_start + local_t
#                     if t >= group_token_end:
#                         break
#                     # 从 routing_array 取第 t 个 token 的专家列表
#                     rl_entry = routing_array[t]

#                     # 实际使用前 num_activated_experts 个专家
#                     use_k = min(num_activated_experts, len(rl_entry))

#                     # =======================================================
#                     # FIX: EP给同一个gpu的多个expert发送数据时去重。使用 set 对当前 Token 的目标 rank 进行去重
#                     # =======================================================

#                     # ------------------------------
#                     # 举例：
#                     # list[i] = [6, 19, 98, 109, 142, 156, 204, 216]
#                     # num_routed_experts = 256, ep = 8
#                     # → 每 32 个专家对应一个 dst
#                     # → [0~31]→dst=0, [32~63]→dst=1, ...
#                     # ------------------------------
#                     token_target_ranks = set()
#                     for k in range(use_k):
#                         expert_id = rl_entry[k]
#                         dst_index = (expert_id * group_size) // num_routed_experts
#                         dst_index = min(max(dst_index, 0), group_size - 1)

#                         # 如果 src==dst，则不计入路由
#                         if dst_index == src_index:
#                             continue

#                         token_target_ranks.add(dst_index)

#                         # bytes_to_each_dst[dst_index] += bytes_each_token
#                     for dst in token_target_ranks:
#                         bytes_to_each_dst[dst] += bytes_each_token

#                 # 将所有非零通信对打包成 triples
#                 triples = [
#                     [bytes_to_each_dst[dst], src_index, dst]
#                     for dst in range(group_size)
#                     if bytes_to_each_dst[dst] > 0
#                 ]

#                 # log.info("triples for ep_group_index: %s (data_slot=%s), src_index: %s, triples: %s", ep_group_index, data_slot, src_index, triples)

#                 # 写入当前组的通信矩阵
#                 if triples:
#                     tm.add_intra_group_traffic_pair_at_group_bulk(
#                         "ep",
#                         triples=triples,
#                         group_index=ep_group_index,
#                         tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
#                         cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
#                     )

#         noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
#         log.info(
#             "noc_hop_time_1: %s, noc_ext_max_1: %s, noc_overall_time_1: %s",
#             noc_hop_time_1, noc_ext_max_1, noc_overall_time_1
#         )
#         # tm.save_heatmap("tm_ep_all_to_all.png")
#     # log.info("routing_array type: %s", type(routing_array))
#     return noc_hop_time_1, noc_ext_max_1

def ep_all_to_all_wrapper(
    parallel: ParallelScheme,
    noc_hierarchy: Hierarchy,
    granularity: Modeling_Granularity,
    bytes_each_token: int,
    routing_array: np.ndarray,
    bs: int,
    seq: int,
    num_routed_experts: int,
    num_activated_experts: int,
    imbalance_overhead_ratio: float,
):
    tm = TrafficMatrix(parallel.world_size())

    if parallel.ep == 1:
        return 0.0, 0.0, None

    # -------------------------------------------------------------
    # 路径 1: 估算模式 (大 Batch)
    # -------------------------------------------------------------
    if (bs * seq >= 8192):
        bytes_per_pair = math.ceil(bs * seq / parallel.dp / parallel.sp / parallel.ep) \
                         * num_activated_experts / parallel.ep * bytes_each_token * imbalance_overhead_ratio
        log.info("bytes_per_pair: %s", bytes_per_pair)

        tm.add_intra_group_traffic(
            "ep",
            bytes_per_pair,
            tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp
        )
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
        log.info("noc_hop_time_1: %s, noc_ext_max_1: %s, noc_overall_time_1: %s",
                 noc_hop_time_1, noc_ext_max_1, noc_overall_time_1)
        return noc_hop_time_1, noc_ext_max_1, noc_traffic_1

    # -------------------------------------------------------------
    # 路径 2: 从trace dump, 额外向量化
    # -------------------------------------------------------------
    
    total_tokens = bs * seq
    
    # 1. 预处理 Routing Array
    # 确保维度正确
    if routing_array.ndim >= 2:
        routing_array = routing_array.reshape(-1, routing_array.shape[-1])
    else:
        raise ValueError("routing_array 的维度应≥2")

    tokens_available, _ = routing_array.shape
    assert tokens_available >= total_tokens, "routing_array 行数不足"

    group_size = parallel.ep
    ep_groups = math.ceil(parallel.world_size() / group_size)
    groups_per_data = parallel.dp * parallel.sp
    tokens_each_ep_group = math.ceil(total_tokens / groups_per_data)

    # =================================================================
    # [向量化步骤 1] 预计算全局 Destination Rank 矩阵
    # =================================================================
    # 只取前 num_activated_experts 列
    # shape: (total_tokens, num_activated_experts)
    active_routing = routing_array[:total_tokens, :num_activated_experts]
    
    # 利用 numpy 广播计算所有 token 所有 expert 对应的 rank
    # 运算: (expert_id * group_size) // num_routed_experts
    # shape: (total_tokens, num_activated_experts)
    global_dst_ranks = (active_routing * group_size) // num_routed_experts
    
    # 防止越界 (clip 只是为了安全，逻辑正确的话不需要)
    np.clip(global_dst_ranks, 0, group_size - 1, out=global_dst_ranks)

    # 准备索引数组，用于后续构造 mask
    # range_k: [0, 1, ..., num_activated - 1]
    # 我们将在循环内部利用这个生成列索引
    cols_count = num_activated_experts

    for ep_group_index in range(ep_groups):
        
        # 确定数据窗口
        data_slot = ep_group_index % groups_per_data
        group_token_start = data_slot * tokens_each_ep_group
        group_token_end = min(group_token_start + tokens_each_ep_group, total_tokens)

        if group_token_start >= group_token_end:
            continue
        
        # 获取当前组负责的 tokens 的 Destination Ranks
        # Shape: (num_local_tokens, num_activated_experts)
        local_dst_ranks = global_dst_ranks[group_token_start:group_token_end, :]
        num_local_tokens = local_dst_ranks.shape[0]

        # =================================================================
        # [向量化步骤 2] 构建去重后的流量掩码 (Boolean Mask)
        # =================================================================
        # 我们想要一个矩阵 M，shape = (num_local_tokens, group_size)
        # 如果 token i 发往 rank j，则 M[i, j] = True
        
        # 使用 flat索引技巧：
        # 行索引：每个 token 重复 k 次 -> [0,0,..,0, 1,1,..,1, ...]
        row_indices = np.repeat(np.arange(num_local_tokens), cols_count)
        # 列索引：直接展平 local_dst_ranks -> [rank_a, rank_b, ..., rank_x, ...]
        col_indices = local_dst_ranks.flatten()

        # 创建布尔矩阵 (初始化全为 False)
        # dtype=bool 非常节省内存
        traffic_mask = np.zeros((num_local_tokens, group_size), dtype=bool)
        
        # 赋值 True。这也天然解决了 "Duplication" 问题
        # 即使同一个 token 对同一个 dst rank 赋值多次 True，结果还是 True
        traffic_mask[row_indices, col_indices] = True
        
        # =================================================================
        # [向量化步骤 3] 按 Source Rank 切片并聚合
        # =================================================================
        
        # 计算每个 src 分配到的 token 数量
        base = num_local_tokens // group_size
        rem = num_local_tokens % group_size
        
        current_row_ptr = 0 # 追踪在 traffic_mask 中的行位置

        # 初始化 triples 列表
        group_triples = []

        # 遍历组内每个 Source Rank
        for src_index in range(group_size):
            # 计算该 src 拥有的 token 数量
            count = base + (1 if src_index < rem else 0)
            
            if count == 0:
                continue
            
            # 切片：取该 src 负责的那部分 token 的 mask
            # shape: (count, group_size)
            src_mask = traffic_mask[current_row_ptr : current_row_ptr + count, :]
            
            # 按列求和：得到发往每个 dst 的 token 总数 (已去重)
            # shape: (group_size,)
            token_counts = src_mask.sum(axis=0)
            
            # 排除自己发给自己 (intra-gpu)
            token_counts[src_index] = 0
            
            # 找到非零流量的目标
            # np.nonzero 返回的是 tuple，取 [0]
            dst_indices = np.nonzero(token_counts)[0]
            
            for dst in dst_indices:
                vol_bytes = int(token_counts[dst]) * bytes_each_token
                group_triples.append([vol_bytes, src_index, int(dst)])
            
            # 移动指针
            current_row_ptr += count

        # =================================================================
        # 写入 Traffic Matrix
        # =================================================================
        if group_triples:
            tm.add_intra_group_traffic_pair_at_group_bulk(
                "ep",
                triples=group_triples,
                group_index=ep_group_index,
                tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
                cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
            )

    noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
    log.info(
        "noc_hop_time_1: %s, noc_ext_max_1: %s, noc_overall_time_1: %s",
        noc_hop_time_1, noc_ext_max_1, noc_overall_time_1
    )
    return noc_hop_time_1, noc_ext_max_1, noc_traffic_1
