
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

def all_reduce_recursive_doubling(all_reduce_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    
    dim_to_process = dim_to_process.lower()
    if dim_to_process not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
        raise ValueError("dim_to_process 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

    dim_size_map = {"tp": parallel.tp, "ep": parallel.ep, "sp": parallel.sp, "cp": parallel.cp, "dp": parallel.dp, "pp": parallel.pp}
    
    gsize = int(dim_size_map[dim_to_process])  # which -> 该维度的组大小
    
    # assert gsize is power of 2
    assert gsize & (gsize - 1) == 0, log.error("gsize must be power of 2, gsize: %s", gsize)

    stages = int(math.log2(gsize))

    bytes_each_iter = bytes 
    # 逐 stage 的递归加倍交换，stride = 2^stage
    total_noc_hop_time = 0
    total_noc_ext_max = 0

    tm = TrafficMatrix(parallel.world_size())

    total_traffic = None
    for stage in range(stages):
        stride = 1 << stage


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

        # tm.export_json(f"all_reduce_recursive_doubling_{stage}.json")

        tm.reset()

    log.info(
        "all_reduce_recursive_doubling, stages: %s, total_noc_hop_time: %s, total_noc_ext_max: %s",
        stages,
        total_noc_hop_time,
        total_noc_ext_max,
    )

    return total_noc_hop_time, total_noc_ext_max, total_traffic





def all_reduce_ring(all_reduce_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    dim_to_process = dim_to_process.lower()
    if dim_to_process not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
        raise ValueError("dim_to_process 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

    dim_size_map = {"tp": parallel.tp, "ep": parallel.ep, "sp": parallel.sp, "cp": parallel.cp, "dp": parallel.dp, "pp": parallel.pp}
    
    gsize = int(dim_size_map[dim_to_process])  # which -> 该维度的组大小

    stages = (gsize-1)*2
    # bytes means each device holding each tensor size
    
    bytes_each_iter = bytes / gsize

    tm = TrafficMatrix(parallel.world_size())

    # let's construct the ring pair along with the bytes_each_iter
    traffic_pairs = []
    for i in range(gsize):
        traffic_pairs.append([bytes_each_iter, i, (i+1)%gsize])

    tm.add_intra_group_traffic_pair_bulk(
        dim_to_process,
        traffic_pairs,
        tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
    )

    noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)

    
    log.info("all_reduce_ring, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", noc_hop_time_1*stages, noc_ext_max_1*stages, noc_overall_time_1*stages)

    return noc_hop_time_1*stages, noc_ext_max_1*stages, noc_traffic_1*stages
        
    


def all_reduce_rabenseifner(all_reduce_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    
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

        tm.reset()

    total_noc_hop_time*=2
    total_noc_ext_max*=2
    total_traffic*=2
    stages_doubling=stages*2
    log.info(
        "all_reduce_rabenseifner, stages: %s, total_noc_hop_time: %s, total_noc_ext_max: %s",
        stages_doubling,
        total_noc_hop_time,
        total_noc_ext_max,
    )

    return total_noc_hop_time, total_noc_ext_max, total_traffic


def all_reduce_double_tree(all_reduce_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    
    dim_to_process = dim_to_process.lower()
    if dim_to_process not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
        raise ValueError("dim_to_process 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

    dim_size_map = {"tp": parallel.tp, "ep": parallel.ep, "sp": parallel.sp, "cp": parallel.cp, "dp": parallel.dp, "pp": parallel.pp}
    
    gsize = int(dim_size_map[dim_to_process])  # which -> 该维度的组大小
    
    # assert gsize is power of 2
    assert gsize & (gsize - 1) == 0, log.error("gsize must be power of 2, gsize: %s", gsize)

    stages = int(math.log2(gsize))

    bytes_each_device = bytes 
    # 双二叉树：A 树朝右根 (gsize-1)，B 树朝左根 (0)
    total_noc_hop_time = 0
    total_noc_ext_max = 0
    total_traffic = None

    tm = TrafficMatrix(parallel.world_size())

    # 每棵树承担一半数据量
    bytes_each_iter = bytes_each_device / 2

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

    # 反向广播阶段（从根向叶，反向边，层序反转）
    for lvl in range(stages - 1, -1, -1):
        traffic_pairs = []
        for src, dst in a_levels[lvl]:
            traffic_pairs.append([bytes_each_iter, dst, src])
        for src, dst in b_levels[lvl]:
            traffic_pairs.append([bytes_each_iter, dst, src])

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
        "all_reduce_double_tree, stages_total: %s, total_noc_hop_time: %s, total_noc_ext_max: %s",
        stages * 2,
        total_noc_hop_time,
        total_noc_ext_max,
    )

    return total_noc_hop_time, total_noc_ext_max, total_traffic

def all_reduce_all_to_all(all_reduce_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    dim_to_process = dim_to_process.lower()
    if dim_to_process not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
        raise ValueError("dim_to_process 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

    dim_size_map = {"tp": parallel.tp, "ep": parallel.ep, "sp": parallel.sp, "cp": parallel.cp, "dp": parallel.dp, "pp": parallel.pp}
    
    gsize = int(dim_size_map[dim_to_process])  # which -> 该维度的组大小

    stages = 1
    # bytes means each device holding each tensor size
    
    bytes_each_iter = bytes

    tm = TrafficMatrix(parallel.world_size())
    tm.add_intra_group_traffic(dim_to_process, bytes_each_iter, tp=parallel.tp, ep=parallel.ep, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
    # tm.add_intra_group_traffic("dp", 1000/3, sp=parallel.sp, cp=parallel.cp, dp=parallel.dp, pp=parallel.pp)
    # hop_time_1, ext_max_1, overall_time_1, noc_traffic_1 = get_extend_max_routes_with_traffic(tm, noc_hierarchy)

    noc_hop_time_1, noc_ext_max_1, noc_overall_time_1, noc_traffic_1= get_extend_max_routes_with_traffic(tm, noc_hierarchy)

    
    log.info("all_reduce_all_to_all, noc_hop_time: %s, noc_ext_max: %s, noc_overall_time: %s", noc_hop_time_1*stages, noc_ext_max_1*stages, noc_overall_time_1*stages)

    return noc_hop_time_1, noc_ext_max_1, noc_traffic_1


def all_reduce_wrapper(all_reduce_op_bytes:OpBytes, parallel:ParallelScheme, noc_hierarchy:Hierarchy, granularity:Modeling_Granularity, dim_to_process:str, bytes:int):
    bytes = uint64(bytes)

    # 统一走 auto-tune 评估多种算法
    latency_rd, link_rd, traffic_rd = all_reduce_recursive_doubling(all_reduce_op_bytes, parallel, noc_hierarchy, granularity, dim_to_process, bytes)
    latency_ring, link_ring, traffic_ring = all_reduce_ring(all_reduce_op_bytes, parallel, noc_hierarchy, granularity, dim_to_process, bytes)
    latency_rab, link_rab, traffic_rab = all_reduce_rabenseifner(all_reduce_op_bytes, parallel, noc_hierarchy, granularity, dim_to_process, bytes)
    latency_tree, link_tree, traffic_tree = all_reduce_double_tree(all_reduce_op_bytes, parallel, noc_hierarchy, granularity, dim_to_process, bytes)
    latency_all2all, link_all2all, traffic_all2all = all_reduce_all_to_all(all_reduce_op_bytes, parallel, noc_hierarchy, granularity, dim_to_process, bytes)

    total_rd = latency_rd + link_rd
    total_ring = latency_ring + link_ring
    total_rab = latency_rab + link_rab
    total_tree = latency_tree + link_tree
    total_all2all = latency_all2all + link_all2all

    best_algo = "recursive_doubling"
    best_latency = latency_rd
    best_link = link_rd
    best_traffic = traffic_rd
    best_total = total_rd

    if total_ring < best_total:
        best_algo = "ring"
        best_latency = latency_ring
        best_link = link_ring
        best_traffic = traffic_ring
        best_total = total_ring

    if total_rab < best_total:
        best_algo = "rabenseifner"
        best_latency = latency_rab
        best_link = link_rab
        best_traffic = traffic_rab
        best_total = total_rab

    if total_tree < best_total:
        best_algo = "double_tree"
        best_latency = latency_tree
        best_link = link_tree
        best_traffic = traffic_tree
        best_total = total_tree

    if total_all2all < best_total:
        best_algo = "all_to_all"
        best_latency = latency_all2all
        best_link = link_all2all
        best_traffic = traffic_all2all
        best_total = total_all2all

    log.info(
        "all_reduce_select: algo=%s, latency=%s, link=%s, total=%s; candidates: recursive_doubling(total=%s), ring(total=%s), rabenseifner(total=%s), double_tree(total=%s), all_to_all(total=%s)",
        best_algo,
        best_latency,
        best_link,
        best_total,
        total_rd,
        total_ring,
        total_rab,
        total_tree,
        total_all2all,
    )

    return best_latency, best_link, best_traffic

    # methods: recursive doubling, ring, rabenseifner

    # return max_latency, max_link_time

if __name__ == "__main__":
    import numpy as np
    from mosaic.utils import Tensor_Loc
    from mosaic.noc.noc_topo import make_mesh_or_torus, make_switch, TopoKind, PortSpread

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ----------------------------------------------------------------
    # Topology builders
    # ----------------------------------------------------------------
    def make_switch_topo(n, bw=450e9, lat=1e-6):
        L3 = make_switch(1, hop_latency=0, link_bandwidth=10*bw,
                         switch_center_in_bw=10*bw, switch_center_out_bw=10*bw)
        L2 = make_switch(1, hop_latency=0, link_bandwidth=10*bw,
                         switch_center_in_bw=10*bw, switch_center_out_bw=10*bw)
        L1 = make_switch(n, hop_latency=lat, link_bandwidth=bw,
                         switch_center_in_bw=n*bw, switch_center_out_bw=n*bw)
        return Hierarchy(layers=[L3, L2, L1], port_spread=PortSpread.EVEN,
                         node_mapper=None, name=f"switch{n}")

    def make_nvlink_ib_topo(n_nvlink, n_ib, nvlink_bw=450e9, nvlink_lat=1e-6,
                            ib_bw=200e9, ib_lat=5e-6):
        L3 = make_switch(1, hop_latency=0, link_bandwidth=10*ib_bw,
                         switch_center_in_bw=10*ib_bw, switch_center_out_bw=10*ib_bw)
        L2 = make_switch(n_ib, hop_latency=ib_lat, link_bandwidth=ib_bw,
                         switch_center_in_bw=n_ib*ib_bw, switch_center_out_bw=n_ib*ib_bw)
        L1 = make_switch(n_nvlink, hop_latency=nvlink_lat, link_bandwidth=nvlink_bw,
                         switch_center_in_bw=n_nvlink*nvlink_bw,
                         switch_center_out_bw=n_nvlink*nvlink_bw)
        return Hierarchy(layers=[L3, L2, L1], port_spread=PortSpread.EVEN,
                         node_mapper=None, name=f"nvlink{n_nvlink}+ib{n_ib}")

    def make_torus_topo(dim_x, dim_y, bw=200e9, lat=5e-6):
        L3 = make_switch(1, hop_latency=0, link_bandwidth=10*bw,
                         switch_center_in_bw=10*bw, switch_center_out_bw=10*bw)
        L2 = make_switch(1, hop_latency=0, link_bandwidth=10*bw,
                         switch_center_in_bw=10*bw, switch_center_out_bw=10*bw)
        L1 = make_mesh_or_torus(dim_x, dim_y, TopoKind.TORUS2D,
                                hop_latency=lat, link_bandwidth=bw)
        return Hierarchy(layers=[L3, L2, L1], port_spread=PortSpread.EVEN,
                         node_mapper=None, name=f"torus{dim_x}x{dim_y}")

    # ----------------------------------------------------------------
    # Shared config
    # ----------------------------------------------------------------
    op_bytes = OpBytes(
        input1=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
        input2=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
        output=Tensor_Loc(dtype=torch.bfloat16, loc="ddr"),
    )
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=False)

    CASES = [
        # (label, parallel, dim, topo)
        ("switch8_tp8",    ParallelScheme(tp=8, ep=1, sp=1, cp=1, dp=1, pp=1), "tp", make_switch_topo(8)),
        ("switch8_tp4dp2", ParallelScheme(tp=4, ep=1, sp=1, cp=1, dp=2, pp=1), "tp", make_switch_topo(8)),
        ("nvlink8_ib4_tp8",ParallelScheme(tp=8, ep=1, sp=1, cp=1, dp=4, pp=1), "tp", make_nvlink_ib_topo(8, 4)),
        ("torus4x4_tp4",   ParallelScheme(tp=4, ep=1, sp=1, cp=1, dp=4, pp=1), "tp", make_torus_topo(4, 4)),
    ]

    BYTES_LIST = [128*1024, 1*1024*1024, 16*1024*1024, 256*1024*1024]

    ALGOS = [
        ("recursive_doubling", all_reduce_recursive_doubling),
        ("ring",               all_reduce_ring),
        ("rabenseifner",       all_reduce_rabenseifner),
        ("double_tree",        all_reduce_double_tree),
        ("all_to_all",         all_reduce_all_to_all),
    ]

    def check(lat, link, traffic, world_size, label):
        assert isinstance(lat,  (int, float, np.floating)) and lat  >= 0, f"[{label}] bad latency {lat}"
        assert isinstance(link, (int, float, np.floating)) and link >= 0, f"[{label}] bad link {link}"
        assert isinstance(traffic, np.ndarray),                            f"[{label}] traffic not ndarray"
        assert traffic.ndim == 2 and traffic.shape[0] == traffic.shape[1], f"[{label}] traffic not square"
        assert traffic.shape[0] >= world_size,                             f"[{label}] traffic dim too small"
        assert np.all(traffic >= 0),                                       f"[{label}] traffic has negatives"

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    col_w = 22
    hdr = (f"{'case':<{col_w}} {'algo':<20} {'dim':<4} {'bytes_MB':>8} "
           f"{'lat_ns':>10} {'link_ns':>10} {'total_ns':>10} {'traffic_GB':>11}")
    print(hdr)
    print("-" * len(hdr))

    for label, parallel, dim, topo in CASES:
        world = parallel.world_size()
        for raw_bytes in BYTES_LIST:
            mb = raw_bytes / 1024 / 1024

            # --- individual algorithms ---
            algo_results = {}
            for algo_name, algo_fn in ALGOS:
                lat, link, traffic = algo_fn(op_bytes, parallel, topo, granularity, dim, raw_bytes)
                check(lat, link, traffic, world, f"{label}/{algo_name}/{mb:.0f}MB")
                algo_results[algo_name] = (lat, link, traffic)
                print(f"{label:<{col_w}} {algo_name:<20} {dim:<4} {mb:>8.1f} "
                      f"{lat*1e9:>10.2f} {link*1e9:>10.2f} {(lat+link)*1e9:>10.2f} "
                      f"{traffic.sum()/1e9:>11.3f}")

            # --- wrapper (auto-tune) ---
            lat_w, link_w, traffic_w = all_reduce_wrapper(op_bytes, parallel, topo, granularity, dim, raw_bytes)
            check(lat_w, link_w, traffic_w, world, f"{label}/wrapper/{mb:.0f}MB")
            # wrapper must select one of the known algos
            assert any(
                lat_w == r[0] and link_w == r[1]
                for r in algo_results.values()
            ), f"[{label}] wrapper picked unknown result"
            print(f"{label:<{col_w}} {'[wrapper→best]':<20} {dim:<4} {mb:>8.1f} "
                  f"{lat_w*1e9:>10.2f} {link_w*1e9:>10.2f} {(lat_w+link_w)*1e9:>10.2f} "
                  f"{traffic_w.sum()/1e9:>11.3f}")
            print()

    print("All checks passed.")