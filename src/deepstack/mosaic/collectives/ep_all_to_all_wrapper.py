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
#     """

#     tm = TrafficMatrix(parallel.world_size())

#     if (bs * seq >= 1024):
#         bytes_per_pair = math.ceil(bs * seq / parallel.dp / parallel.sp / parallel.ep) \
#                          * num_activated_experts / parallel.ep * bytes_each_token * imbalance_overhead_ratio
#         log.info("bytes_per_pair: %s", bytes_per_pair)

#         tm.add_intra_group_traffic(
#             "ep",
#             tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp
#         )

#         noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
#         log.info("noc_hop_time_1: %s, noc_ext_max_1: %s, noc_overall_time_1: %s",
#                  noc_hop_time_1, noc_ext_max_1, noc_overall_time_1)
        
#     else:
#         # -----------------------------
#         # -----------------------------
#         total_tokens = bs * seq

#         shape_of_array = routing_array.shape
#         # print(f"routing_array.shape (before reshape): {shape_of_array}")
#         log.info("routing_array.shape (before reshape): %s", shape_of_array)
#         if routing_array.ndim >= 2:
#             routing_array = routing_array.reshape(-1, routing_array.shape[-1])
#         else:

#         tokens_available, experts_per_token = routing_array.shape
#         assert tokens_available >= total_tokens, (
#         )

#         group_size = parallel.ep
#         ep_groups = math.ceil(parallel.world_size() / group_size)

#         # -----------------------------------------------
#         # -----------------------------------------------
#         groups_per_data = parallel.dp * parallel.sp

#         tokens_each_ep_group = math.ceil(total_tokens / groups_per_data)

#         # =========================================
#         # 0,.... ep_groups-1
#         #
#         # =========================================
#         for ep_group_index in range(ep_groups):

#             data_slot = ep_group_index % groups_per_data

#             # [group_token_start, group_token_end)
#             group_token_start = data_slot * tokens_each_ep_group
#             group_token_end = min(group_token_start + tokens_each_ep_group, total_tokens)

#             if group_token_start >= group_token_end:
#                 continue

#             local_tokens = group_token_end - group_token_start

#             base = local_tokens // group_size
#             rem = local_tokens % group_size

#             # =========================================
#             # 0,.... ep_groups-1
#             # =========================================
#             for src_index in range(group_size):
#                 # ----------------------------------
#                 #
#                 # ----------------------------------
#                 count = base + (1 if src_index < rem else 0)
#                 start = src_index * base + min(src_index, rem)
#                 abs_start = group_token_start + start

#                 if count == 0:
#                     continue

#                 bytes_to_each_dst = [0] * group_size

#                 for local_t in range(count):
#                     t = abs_start + local_t
#                     if t >= group_token_end:
#                         break
#                     rl_entry = routing_array[t]

#                     use_k = min(num_activated_experts, len(rl_entry))

#                     # =======================================================
#                     # =======================================================

#                     # ------------------------------
#                     # list[i] = [6, 19, 98, 109, 142, 156, 204, 216]
#                     # num_routed_experts = 256, ep = 8
#                     # → [0~31]→dst=0, [32~63]→dst=1, ...
#                     # ------------------------------
#                     token_target_ranks = set()
#                     for k in range(use_k):
#                         expert_id = rl_entry[k]
#                         dst_index = (expert_id * group_size) // num_routed_experts
#                         dst_index = min(max(dst_index, 0), group_size - 1)

#                         if dst_index == src_index:
#                             continue

#                         token_target_ranks.add(dst_index)

#                         # bytes_to_each_dst[dst_index] += bytes_each_token
#                     for dst in token_target_ranks:
#                         bytes_to_each_dst[dst] += bytes_each_token

#                 triples = [
#                     [bytes_to_each_dst[dst], src_index, dst]
#                     for dst in range(group_size)
#                     if bytes_to_each_dst[dst] > 0
#                 ]

#                 # log.info("triples for ep_group_index: %s (data_slot=%s), src_index: %s, triples: %s", ep_group_index, data_slot, src_index, triples)

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
    # -------------------------------------------------------------
    
    total_tokens = bs * seq
    
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
    # =================================================================
    # shape: (total_tokens, num_activated_experts)
    active_routing = routing_array[:total_tokens, :num_activated_experts]
    
    # shape: (total_tokens, num_activated_experts)
    global_dst_ranks = (active_routing * group_size) // num_routed_experts
    
    np.clip(global_dst_ranks, 0, group_size - 1, out=global_dst_ranks)

    # range_k: [0, 1, ..., num_activated - 1]
    cols_count = num_activated_experts

    for ep_group_index in range(ep_groups):
        
        data_slot = ep_group_index % groups_per_data
        group_token_start = data_slot * tokens_each_ep_group
        group_token_end = min(group_token_start + tokens_each_ep_group, total_tokens)

        if group_token_start >= group_token_end:
            continue
        
        # Shape: (num_local_tokens, num_activated_experts)
        local_dst_ranks = global_dst_ranks[group_token_start:group_token_end, :]
        num_local_tokens = local_dst_ranks.shape[0]

        # =================================================================
        # =================================================================
        
        row_indices = np.repeat(np.arange(num_local_tokens), cols_count)
        col_indices = local_dst_ranks.flatten()

        traffic_mask = np.zeros((num_local_tokens, group_size), dtype=bool)
        
        traffic_mask[row_indices, col_indices] = True
        
        # =================================================================
        # =================================================================
        
        base = num_local_tokens // group_size
        rem = num_local_tokens % group_size
        
        current_row_ptr = 0

        group_triples = []

        for src_index in range(group_size):
            count = base + (1 if src_index < rem else 0)
            
            if count == 0:
                continue
            
            # shape: (count, group_size)
            src_mask = traffic_mask[current_row_ptr : current_row_ptr + count, :]
            
            # shape: (group_size,)
            token_counts = src_mask.sum(axis=0)
            
            token_counts[src_index] = 0
            
            dst_indices = np.nonzero(token_counts)[0]
            
            for dst in dst_indices:
                vol_bytes = int(token_counts[dst]) * bytes_each_token
                group_triples.append([vol_bytes, src_index, int(dst)])
            
            current_row_ptr += count

        # =================================================================
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
