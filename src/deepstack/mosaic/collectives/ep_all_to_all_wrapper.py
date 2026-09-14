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
#         List of expert IDs routed to by token t (e.g., [6, 19, 98, ...]); use only the first num_activated_experts.
#         Expert IDs lie in [0, num_routed_experts).
#         Forward and reverse are exact duals and inverse operations: swap src and dst without changing load, latency, or total time.
#     """

#     tm = TrafficMatrix(parallel.world_size())

#     if (bs * seq >= 1024):
#         # At larger scales, use the average plus an imbalance factor.
#         bytes_per_pair = math.ceil(bs * seq / parallel.dp / parallel.sp / parallel.ep) \
#                          * num_activated_experts / parallel.ep * bytes_each_token * imbalance_overhead_ratio
#         log.info("bytes_per_pair: %s", bytes_per_pair)

#         # Use intragroup all-to-all directly (averaged); the underlying implementation generates pairwise communication within the same parallel dimension.
#         tm.add_intra_group_traffic(
#             "ep",
#             bytes_per_pair,  # Base quantity in bytes; expanded internally by averaging/combinations.
#             tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp
#         )

#         noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)
#         log.info("noc_hop_time_1: %s, noc_ext_max_1: %s, noc_overall_time_1: %s",
#                  noc_hop_time_1, noc_ext_max_1, noc_overall_time_1)
        
#     else:
#         # -----------------------------
#         # Fine-grained simulation: dispatch according to routing_list.
#         # -----------------------------
#         total_tokens = bs * seq

#         # Check the shape.
#         shape_of_array = routing_array.shape
#         # print(f"routing_array.shape (before reshape): {shape_of_array}")
#         log.info("routing_array.shape (before reshape): %s", shape_of_array)
#         if routing_array.ndim >= 2:
#             routing_array = routing_array.reshape(-1, routing_array.shape[-1])
#         else:
#             raise ValueError("routing_array must have ≥2 dimensions, with shape [*, num_activated_experts]")

#         # Check that enough token rows are available.
#         tokens_available, experts_per_token = routing_array.shape
#         assert tokens_available >= total_tokens, (
#             f"Insufficient routing_array rows! Need at least {total_tokens}, got {tokens_available}."
#         )

#         group_size = parallel.ep
#         ep_groups = math.ceil(parallel.world_size() / group_size)

#         # -----------------------------------------------
#         # ✅ Added: define the number of data windows (for cyclic token reuse).
#         # Every dp*sp EP groups share one copy of the token data.
#         # -----------------------------------------------
#         groups_per_data = parallel.dp * parallel.sp

#         # Each data window covers tokens_each_ep_group tokens.
#         tokens_each_ep_group = math.ceil(total_tokens / groups_per_data)

#         # =========================================
#         # Outer to inner:
#         # 0,.... ep_groups-1
#         # Each ep_group represents an independent Expert Parallel group.
#         # Each group contains ep processes (ranks 0 ~ ep-1).
#         #
#         # ✅ Change:
#         #   When ep_group_index >= groups_per_data, cyclically reuse earlier data windows.
#         #   For example, ep_group_index 8 -> data_slot 0, 9->1, and so on.
#         # =========================================
#         for ep_group_index in range(ep_groups):

#             # ✅ Key change: cyclically reuse data windows.
#             data_slot = ep_group_index % groups_per_data

#             # Token range in routing_list for the current EP group:
#             # [group_token_start, group_token_end)
#             group_token_start = data_slot * tokens_each_ep_group
#             group_token_end = min(group_token_start + tokens_each_ep_group, total_tokens)

#             # Skip if no tokens are available (e.g., many groups have exhausted the tokens).
#             if group_token_start >= group_token_end:
#                 continue

#             local_tokens = group_token_end - group_token_start

#             # Each src (EP rank) holds roughly the same number of tokens; the first rem hold one extra.
#             base = local_tokens // group_size
#             rem = local_tokens % group_size

#             # =========================================
#             # Outer to inner:
#             # 0,.... ep_groups-1
#             # Each ep_group represents an independent Expert Parallel group.
#             # Each group contains ep processes (ranks 0 ~ ep-1), handling tokens_each_ep_group tokens.
#             # For each group:
#             #   - Partition input token ranges by src (EP rank).
#             #   - Read each token's expert list from routing_list.
#             #   - Determine the destination dst rank within the same group from the expert ID.
#             #   - If src == dst, treat the transfer as local with no communication.
#             # =========================================
#             for src_index in range(group_size):
#                 # ----------------------------------
#                 # The current src_index identifies the EP rank within the group.
#                 # For example, ep=8 gives src_index ∈ [0..7].
#                 #
#                 # Number of tokens held by this src: base or base+1.
#                 # For example, tokens_each_ep_group=100, ep=8 =>
#                 #   base=12, rem=4 -> the first 4 src ranks hold 13 tokens each, the last 4 hold 12 each.
#                 # ----------------------------------
#                 count = base + (1 if src_index < rem else 0)
#                 start = src_index * base + min(src_index, rem)
#                 abs_start = group_token_start + start

#                 if count == 0:
#                     continue

#                 # Initialize an array to count total bytes sent to each dst.
#                 bytes_to_each_dst = [0] * group_size

#                 # Iterate over the tokens assigned to this src.
#                 for local_t in range(count):
#                     t = abs_start + local_t
#                     if t >= group_token_end:
#                         break
#                     # Get token t's expert list from routing_array.
#                     rl_entry = routing_array[t]

#                     # Use only the first num_activated_experts experts.
#                     use_k = min(num_activated_experts, len(rl_entry))

#                     # =======================================================
#                     # FIX: deduplicate EP transfers to multiple experts on the same GPU. Use a set to deduplicate destination ranks for the current token.
#                     # =======================================================

#                     # ------------------------------
#                     # Example:
#                     # list[i] = [6, 19, 98, 109, 142, 156, 204, 216]
#                     # num_routed_experts = 256, ep = 8
#                     # -> Every 32 experts map to one dst.
#                     # → [0~31]→dst=0, [32~63]→dst=1, ...
#                     # ------------------------------
#                     token_target_ranks = set()
#                     for k in range(use_k):
#                         expert_id = rl_entry[k]
#                         dst_index = (expert_id * group_size) // num_routed_experts
#                         dst_index = min(max(dst_index, 0), group_size - 1)

#                         # If src==dst, exclude it from routing.
#                         if dst_index == src_index:
#                             continue

#                         token_target_ranks.add(dst_index)

#                         # bytes_to_each_dst[dst_index] += bytes_each_token
#                     for dst in token_target_ranks:
#                         bytes_to_each_dst[dst] += bytes_each_token

#                 # Pack all nonzero communication pairs into triples.
#                 triples = [
#                     [bytes_to_each_dst[dst], src_index, dst]
#                     for dst in range(group_size)
#                     if bytes_to_each_dst[dst] > 0
#                 ]

#                 # log.info("triples for ep_group_index: %s (data_slot=%s), src_index: %s, triples: %s", ep_group_index, data_slot, src_index, triples)

#                 # Write the communication matrix for the current group.
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
    # Path 1: estimation mode (large batch).
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
    # Path 2: use a trace dump, with additional vectorization.
    # -------------------------------------------------------------
    
    total_tokens = bs * seq
    
    # 1. Preprocess the Routing Array.
    # Ensure the dimensions are correct.
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
    # [Vectorization step 1] Precompute the global Destination Rank matrix.
    # =================================================================
    # Take only the first num_activated_experts columns.
    # shape: (total_tokens, num_activated_experts)
    active_routing = routing_array[:total_tokens, :num_activated_experts]
    
    # Use NumPy broadcasting to compute the rank for every expert of every token.
    # Operation: (expert_id * group_size) // num_routed_experts.
    # shape: (total_tokens, num_activated_experts)
    global_dst_ranks = (active_routing * group_size) // num_routed_experts
    
    # Prevent out-of-bounds values (clip is only a safeguard and is unnecessary if the logic is correct).
    np.clip(global_dst_ranks, 0, group_size - 1, out=global_dst_ranks)

    # Prepare the index array for constructing masks later.
    # range_k: [0, 1, ..., num_activated - 1]
    # Use this to generate column indices inside the loop.
    cols_count = num_activated_experts

    for ep_group_index in range(ep_groups):
        
        # Determine the data window.
        data_slot = ep_group_index % groups_per_data
        group_token_start = data_slot * tokens_each_ep_group
        group_token_end = min(group_token_start + tokens_each_ep_group, total_tokens)

        if group_token_start >= group_token_end:
            continue
        
        # Get Destination Ranks for the tokens assigned to the current group.
        # Shape: (num_local_tokens, num_activated_experts)
        local_dst_ranks = global_dst_ranks[group_token_start:group_token_end, :]
        num_local_tokens = local_dst_ranks.shape[0]

        # =================================================================
        # [Vectorization step 2] Build the deduplicated traffic mask (Boolean Mask).
        # =================================================================
        # Construct a matrix M with shape = (num_local_tokens, group_size).
        # If token i is sent to rank j, M[i, j] = True.
        
        # Use flat indexing:
        # Row indices: repeat each token k times -> [0,0,..,0, 1,1,..,1, ...].
        row_indices = np.repeat(np.arange(num_local_tokens), cols_count)
        # Column indices: flatten local_dst_ranks directly -> [rank_a, rank_b, ..., rank_x, ...].
        col_indices = local_dst_ranks.flatten()

        # Create a Boolean matrix (initialized to False).
        # dtype=bool is very memory-efficient.
        traffic_mask = np.zeros((num_local_tokens, group_size), dtype=bool)
        
        # Assign True, which also handles duplication naturally.
        # Assigning True repeatedly for the same token and dst rank still yields True.
        traffic_mask[row_indices, col_indices] = True
        
        # =================================================================
        # [Vectorization step 3] Slice and aggregate by Source Rank.
        # =================================================================
        
        # Compute the number of tokens assigned to each src.
        base = num_local_tokens // group_size
        rem = num_local_tokens % group_size
        
        current_row_ptr = 0 # Track the row position in traffic_mask.

        # Initialize the triples list.
        group_triples = []

        # Iterate over each Source Rank in the group.
        for src_index in range(group_size):
            # Compute the number of tokens owned by this src.
            count = base + (1 if src_index < rem else 0)
            
            if count == 0:
                continue
            
            # Slice the mask for the tokens assigned to this src.
            # shape: (count, group_size)
            src_mask = traffic_mask[current_row_ptr : current_row_ptr + count, :]
            
            # Sum by column to get the total number of tokens sent to each dst (deduplicated).
            # shape: (group_size,)
            token_counts = src_mask.sum(axis=0)
            
            # Exclude self-transfers (intra-gpu).
            token_counts[src_index] = 0
            
            # Find destinations with nonzero traffic.
            # np.nonzero returns a tuple; take [0].
            dst_indices = np.nonzero(token_counts)[0]
            
            for dst in dst_indices:
                vol_bytes = int(token_counts[dst]) * bytes_each_token
                group_triples.append([vol_bytes, src_index, int(dst)])
            
            # Advance the pointer.
            current_row_ptr += count

        # =================================================================
        # Write to the Traffic Matrix.
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
