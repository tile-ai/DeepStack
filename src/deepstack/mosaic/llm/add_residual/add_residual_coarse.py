import torch
import math
import numpy as np
from mosaic.parallelism import ParallelScheme
from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.noc_topo import Hierarchy, make_mesh_or_torus, make_switch, TopoKind, PortSpread, get_extend_max_routes, get_extend_max_routes_with_traffic
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
from mosaic.op_dtype.element_wrapper import element_wrapper
import cProfile

from tilesight.arch import *
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
import logging
log = logging.getLogger(__name__) 
from mosaic.collectives import all_reduce_wrapper
from mosaic.collectives import reduce_scatter_wrapper
from mosaic.utils import get_comp_comm_e2e_time
from mosaic.cost.energy_record import EnergyRecord
from mosaic.cost.op_perf_stats import OpPerfStats

def add_residual_coarse(bs:int, seq:int, hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, add_residual_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy) -> tuple[float, OpPerfStats]:
    assert granularity.get_mode() == "coarse"
    # print(parallel.world_size())


    in_bytes, weight_bytes, out_bytes = add_residual_bytes.get_dtype_bytes()
    shard_hidden = math.ceil(hidden / parallel.tp)
    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    # x: [bs/dp, seq/sp, hidden]
    # x_residual: [bs/dp, seq/sp, hidden]
    # y: [bs/dp, seq/sp, hidden]
    # 一般我们认为不会去切这个hidden，因为下一个算子的时候就还需要一个all-reduce的重建


    # ------------------------------------- stage 4 -------------------------------------
    # add residual

    smem_fusion_list=[]

    local_residual_bytes=OpBytes(
        input1=Tensor_Loc(add_residual_bytes.input1.dtype, add_residual_bytes.input1.loc,[shard_bs, shard_seq, shard_hidden]),
        input2=Tensor_Loc(add_residual_bytes.input2.dtype, add_residual_bytes.input2.loc,[shard_bs, shard_seq, shard_hidden]),
        output=Tensor_Loc(add_residual_bytes.output.dtype, add_residual_bytes.output.loc,[shard_bs, shard_seq, shard_hidden]),
    )

    hete_post_data_residual, tb_residual_config=element_wrapper(element_op_bytes=local_residual_bytes, granularity=granularity, single_chip=single_chip, batch=2, type="cuda_core", tb_tiling_config=None)
    smem_fusion_list.append(hete_reg_fusion([hete_post_data_residual], single_chip))

    grids = [shard_bs/tb_residual_config[0], shard_seq/tb_residual_config[1], shard_hidden/tb_residual_config[2]]

    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)

    single_chip_time=smem_fusion_post_data[0]

    log.info("add_residual stage 4 (Residual Add) single chip time: %s s", single_chip_time)

    stats = OpPerfStats(op_name="add_residual", dump_perf_log=granularity.dump_perf_log)
    stats.append_hete(smem_fusion_post_data)
    stats.finalize(total_time_s=single_chip_time, h=noc_hierarchy, arch=single_chip)

    return single_chip_time, stats
