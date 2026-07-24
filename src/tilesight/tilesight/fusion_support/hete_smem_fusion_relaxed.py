import math
import numpy as np
from tilesight.arch import Arch
import logging
log = logging.getLogger(__name__)

def find_min_multiple(x, wave_bytes):

    # 计算 x 除以 wave_bytes 的结果
    division_result = x / wave_bytes
    # 找到大于等于 division_result 的最小整数
    min_multiple = math.ceil(division_result)
    # 返回 wave_bytes 乘以这个最小整数
    return min_multiple * wave_bytes


def hete_smem_fusion(smem_fusion_list,grids,arch_spec:Arch):
    # print("smem fusion list")
    # print("bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util,ddr_read_io, l2_read_io")
    # print(smem_fusion_list)

    time_added, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, ddr_read_io, l2_read_io, tensor_util, cuda_util, sfu_util=0,0,0,0,0,0,0,0,0,0,0,0
    for row in smem_fusion_list:

        time_added+=row[0]
        ddr_util+=row[0]*row[1]
        l2_util+=row[0]*row[3]
        smem_footprint=max(smem_footprint,row[4])
        smem_l1_util+=row[0]*row[5]
        reg_footprint=max(reg_footprint,row[6])
        ddr_read_io+=row[7]
        l2_read_io+=row[8]
        tensor_util+=row[0]*row[9]
        cuda_util+=row[0]*row[10]
        sfu_util+=row[0]*row[11]
        

    # l2_hit_rate/=time_added
    # l2_hit_rate=l2_hit_rate/l2_util
    # print("time_added",time_added)
    # print("after fusion")
    l2_hit_rate=(1-(max(ddr_read_io,1)/max(l2_read_io,1)))

    waves=np.prod(grids)/arch_spec.sm_count



    # bound_time=max(ddr_util/arch_spec.ddr_max_util,l2_util/arch_spec.l2_max_util,smem_l1_util/arch_spec.l1_max_util,tensor_util/arch_spec.compute_max_util,cuda_util/arch_spec.compute_max_util,sfu_util/arch_spec.compute_max_util)*math.ceil(waves)/waves
    # bound_time=max(ddr_util/arch_spec.ddr_max_util,l2_util/arch_spec.l2_max_util,smem_l1_util/arch_spec.l1_max_util,(tensor_util + cuda_util + sfu_util)/arch_spec.compute_max_util)*math.ceil(waves)/waves

    bound_time_1=max(ddr_util/arch_spec.ddr_max_util,l2_util/arch_spec.l2_max_util)
    
    bound_time_2=max(smem_l1_util/arch_spec.l1_max_util,(tensor_util + cuda_util + sfu_util)/arch_spec.compute_max_util)*math.ceil(waves)/waves
    

    bound_time=max(bound_time_1,bound_time_2)

    # ------------kernel launch time with cuda graph------------
    log.info("fused kernel time without kernel launch inside gpu: %s", bound_time)
    kernel_launch_time = 2e-6
    bound_time += kernel_launch_time
    log.info("add kernel launch time: %s, total kernel time: %s", kernel_launch_time, bound_time)
    # ------------kernel launch time with cuda graph------------

    # ddr_footprint=0 # TODO
    ddr_util=ddr_util/bound_time
    l2_util=l2_util/bound_time
    smem_l1_util=smem_l1_util/bound_time

    tensor_util=tensor_util/bound_time
    cuda_util=cuda_util/bound_time
    sfu_util=sfu_util/bound_time

    posted=[bound_time, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, ddr_read_io, l2_read_io, tensor_util, cuda_util, sfu_util]

    # return time_added,posted
    return posted
