
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

def all_gather_sp_fission(parallel:ParallelScheme, noc_hierarchy:Hierarchy, bytes_needed_per_device:int, next_virtual_sp:int):

    all_gather_bytes = bytes_needed_per_device
    log.info("all_gather_bytes: %s", all_gather_bytes)
    if parallel.sp == 1:
        all_gather_hop_latency, all_gather_ext_max = 0, 0
        all_gather_traffic = None
        return all_gather_hop_latency, all_gather_ext_max, all_gather_traffic
    all_gather_bytes_per_pair = all_gather_bytes / (parallel.sp -1)
    log.info("all_gather_bytes_per_pair: %s", all_gather_bytes_per_pair)

    new_virtual_parallel = ParallelScheme(tp=parallel.tp, ep=parallel.ep, sp=int(parallel.sp/next_virtual_sp), cp=1, dp=1, pp=int(parallel.world_size()/parallel.tp/parallel.ep/(parallel.sp/next_virtual_sp)))
    
    new_virtual_all_gather_byte_per_pair=all_gather_bytes/new_virtual_parallel.sp
    log.info("new_virtual_all_gather_byte_per_pair: %s", new_virtual_all_gather_byte_per_pair)
    # reconstruct the weight matrix
    # tm = TrafficMatrix(parallel.world_size())
    # tm.add_intra_group_traffic("sp", all_gather_bytes_per_pair, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
    tm = TrafficMatrix(new_virtual_parallel.world_size())
    tm.add_intra_group_traffic("sp", new_virtual_all_gather_byte_per_pair, tp=new_virtual_parallel.tp, ep=new_virtual_parallel.ep, sp=new_virtual_parallel.sp, cp=new_virtual_parallel.cp, dp=new_virtual_parallel.dp, pp=new_virtual_parallel.pp)
    # tm.save_heatmap(path='./sp_all_gather.png')
    # tm.add_intra_group_traffic("dp", 1000/3, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
    # all_gather_hop_latency, all_gather_ext_max, all_gather_overall_time= get_extend_max_routes_with_traffic(tm, noc_hierarchy)
    all_gather_hop_latency, all_gather_ext_max, all_gather_overall_time, all_gather_traffic= get_extend_max_routes_with_traffic(tm, noc_hierarchy)

    log.info("all_gather_sp_fission, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", all_gather_hop_latency, all_gather_ext_max, all_gather_overall_time)

    return all_gather_hop_latency, all_gather_ext_max, all_gather_traffic

def _all_gather_group_size(parallel:ParallelScheme, dim_to_process:str) -> int:
    dim_to_process = dim_to_process.lower()
    if dim_to_process not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
        raise ValueError("dim_to_process 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")
    dim_size_map = {"tp": parallel.tp, "ep": parallel.ep, "sp": parallel.sp, "cp": parallel.cp, "dp": parallel.dp, "pp": parallel.pp}
    return int(dim_size_map[dim_to_process])


def all_gather_ring(all_gather_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    """Ring all-gather: gsize-1 steps; in each step every device forwards one
    shard (bytes / gsize) to its ring successor."""
    gsize = _all_gather_group_size(parallel, dim_to_process)
    dim_to_process = dim_to_process.lower()
    if gsize == 1:
        return 0, 0, None

    stages = gsize - 1
    # bytes means the gathered tensor on each device; each device starts with one shard
    bytes_each_iter = bytes / gsize

    tm = TrafficMatrix(parallel.world_size())
    traffic_pairs = []
    for i in range(gsize):
        traffic_pairs.append([bytes_each_iter, i, (i + 1) % gsize])

    tm.add_intra_group_traffic_pair_bulk(
        dim_to_process,
        traffic_pairs,
        tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
    )

    noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)

    log.info("all_gather_ring, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", noc_hop_time_1*stages, noc_ext_max_1*stages, noc_overall_time_1*stages)

    return noc_hop_time_1*stages, noc_ext_max_1*stages, noc_traffic_1*stages


def all_gather_recursive_doubling(all_gather_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    """Recursive-doubling all-gather for a power-of-two group: log2(gsize)
    stages; in stage k every device exchanges everything gathered so far,
    (bytes / gsize) * 2**k, with the partner whose index differs in bit k."""
    gsize = _all_gather_group_size(parallel, dim_to_process)
    dim_to_process = dim_to_process.lower()
    if gsize == 1:
        return 0, 0, None

    # assert gsize is power of 2
    assert gsize & (gsize - 1) == 0, log.error("gsize must be power of 2, gsize: %s", gsize)

    stages = int(math.log2(gsize))

    total_noc_hop_time = 0
    total_noc_ext_max = 0

    tm = TrafficMatrix(parallel.world_size())

    total_traffic = None
    for stage in range(stages):
        stride = 1 << stage
        # every stage doubles the data each device holds
        bytes_each_iter = bytes / gsize * stride

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

        tm.reset()

    log.info(
        "all_gather_recursive_doubling, stages: %s, total_noc_hop_time: %s, total_noc_ext_max: %s",
        stages,
        total_noc_hop_time,
        total_noc_ext_max,
    )

    return total_noc_hop_time, total_noc_ext_max, total_traffic


def all_gather_all_to_all(all_gather_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    """Direct all-gather: one stage in which every device sends its shard
    (bytes / gsize) to each other device of its group."""
    gsize = _all_gather_group_size(parallel, dim_to_process)
    dim_to_process = dim_to_process.lower()
    if gsize == 1:
        return 0, 0, None

    tm = TrafficMatrix(parallel.world_size())
    tm.add_intra_group_traffic(dim_to_process, bytes / gsize, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)

    noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)

    log.info("all_gather_all_to_all, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", noc_hop_time_1, noc_ext_max_1, noc_overall_time_1)

    return noc_hop_time_1, noc_ext_max_1, noc_traffic_1


def all_gather_wrapper(all_gather_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    """All-gather along ``dim_to_process``; returns ``(hop_latency_s, link_time_s, traffic)``
    of the fastest algorithm, like ``all_reduce_wrapper``.

    ``bytes`` is the gathered tensor held by every device afterwards; each device
    contributes ``bytes / group_size``. It is the same quantity as the per-device
    buffer of ``all_reduce_wrapper`` and the input of ``reduce_scatter_wrapper``, so
    a ring all-reduce of ``bytes`` costs a ring reduce-scatter plus a ring
    all-gather of ``bytes``. Candidates are ring, recursive doubling (power-of-two
    groups only) and direct all-to-all. ``traffic`` is ``None`` for a group of one.
    ``all_gather_op_bytes`` and ``granularity`` are accepted for symmetry with the
    other collective wrappers and are not used.

    Example::

        from mosaic.collectives import all_gather_wrapper
        from mosaic.noc.noc_topo import Hierarchy, PortSpread, make_switch
        from mosaic.parallelism import ParallelScheme
        from mosaic.utils import Modeling_Granularity

        # One switched node of eight devices (illustrative bandwidth and latency).
        node = Hierarchy(
            layers=[make_switch(1, hop_latency=0.0, link_bandwidth=450e9),
                    make_switch(1, hop_latency=0.0, link_bandwidth=450e9),
                    make_switch(8, hop_latency=1e-6, link_bandwidth=450e9)],
            port_spread=PortSpread.EVEN, name="node8",
        )
        hop_s, link_s, traffic = all_gather_wrapper(
            None, ParallelScheme(tp=8), node,
            Modeling_Granularity("coarse", True, False, True),
            dim_to_process="tp", bytes=64 * 2**20,
        )
        total_s = hop_s + link_s
    """
    gsize = _all_gather_group_size(parallel, dim_to_process)
    if gsize == 1:
        return 0, 0, None

    bytes = uint64(bytes)

    candidates = [("ring", all_gather_ring)]
    if gsize & (gsize - 1) == 0:
        candidates.append(("recursive_doubling", all_gather_recursive_doubling))
    candidates.append(("all_to_all", all_gather_all_to_all))

    results = {}
    for name, algo in candidates:
        results[name] = algo(all_gather_op_bytes, parallel, noc_hierarchy, granularity, dim_to_process, bytes)

    # keep the first fastest candidate, as all_reduce_wrapper does
    best_algo = min(results, key=lambda name: results[name][0] + results[name][1])
    best_latency, best_link, best_traffic = results[best_algo]

    log.info(
        "all_gather_select: algo=%s, latency=%s, link=%s, total=%s; candidates: %s",
        best_algo,
        best_latency,
        best_link,
        best_latency + best_link,
        ", ".join(f"{name}(total={latency + link})" for name, (latency, link, _) in results.items()),
    )

    return best_latency, best_link, best_traffic


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Illustrative topologies (bytes/s, seconds).
    def make_switch_topo(n, bw=450e9, lat=1e-6):
        return Hierarchy(layers=[make_switch(1, hop_latency=0, link_bandwidth=bw),
                                 make_switch(1, hop_latency=0, link_bandwidth=bw),
                                 make_switch(n, hop_latency=lat, link_bandwidth=bw)],
                         port_spread=PortSpread.EVEN, node_mapper=None, name=f"switch{n}")

    def make_nvlink_ib_topo(n_nvlink, n_ib, nvlink_bw=450e9, nvlink_lat=1e-6, ib_bw=50e9, ib_lat=5e-6):
        return Hierarchy(layers=[make_switch(1, hop_latency=0, link_bandwidth=ib_bw),
                                 make_switch(n_ib, hop_latency=ib_lat, link_bandwidth=ib_bw),
                                 make_switch(n_nvlink, hop_latency=nvlink_lat, link_bandwidth=nvlink_bw)],
                         port_spread=PortSpread.EVEN, node_mapper=None, name=f"nvlink{n_nvlink}+ib{n_ib}")

    def make_torus_topo(dim_x, dim_y, bw=200e9, lat=5e-6):
        return Hierarchy(layers=[make_switch(1, hop_latency=0, link_bandwidth=bw),
                                 make_switch(1, hop_latency=0, link_bandwidth=bw),
                                 make_mesh_or_torus(dim_x, dim_y, TopoKind.TORUS2D, hop_latency=lat, link_bandwidth=bw)],
                         port_spread=PortSpread.EVEN, node_mapper=None, name=f"torus{dim_x}x{dim_y}")

    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False)
    CASES = [
        # (label, parallel, dim, topo)
        ("switch8_tp8", ParallelScheme(tp=8), "tp", make_switch_topo(8)),
        ("nvlink8_ib4_tp8", ParallelScheme(tp=8, dp=4), "tp", make_nvlink_ib_topo(8, 4)),
        ("nvlink8_ib4_dp4", ParallelScheme(tp=8, dp=4), "dp", make_nvlink_ib_topo(8, 4)),
        ("torus4x4_tp4", ParallelScheme(tp=4, dp=4), "tp", make_torus_topo(4, 4)),
    ]
    ALGOS = [
        ("ring", all_gather_ring),
        ("recursive_doubling", all_gather_recursive_doubling),
        ("all_to_all", all_gather_all_to_all),
    ]

    hdr = f"{'case':<18} {'algo':<20} {'MB':>6} {'lat_us':>9} {'link_us':>9} {'total_us':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, parallel, dim, topo in CASES:
        for raw_bytes in (1 * 2**20, 64 * 2**20):
            for algo_name, algo_fn in ALGOS:
                lat, link, _ = algo_fn(None, parallel, topo, granularity, dim, raw_bytes)
                print(f"{label:<18} {algo_name:<20} {raw_bytes / 2**20:>6.0f} {lat*1e6:>9.2f} {link*1e6:>9.2f} {(lat+link)*1e6:>9.2f}")
            lat, link, _ = all_gather_wrapper(None, parallel, topo, granularity, dim, raw_bytes)
            print(f"{label:<18} {'[wrapper->best]':<20} {raw_bytes / 2**20:>6.0f} {lat*1e6:>9.2f} {link*1e6:>9.2f} {(lat+link)*1e6:>9.2f}")
        print()
