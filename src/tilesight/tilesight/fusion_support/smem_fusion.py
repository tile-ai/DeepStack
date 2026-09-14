import math
import numpy as np
from tilesight.arch import Arch

def smem_fusion(smem_fusion_list,grids,arch_spec:Arch):
    # print("smem fusion list")
    # print("bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util,ddr_read_io, l2_read_io")
    # print(smem_fusion_list)
    time_added, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0,0


    for row in smem_fusion_list:
        # print(row)
        ddr_util+=row[0]*row[2]
        l2_util+=row[0]*row[4]
        smem_l1_util+=row[0]*row[6]
        compute_util+=row[0]*row[8]
        time_added+=row[0]
        # l2_hit_rate+=row[0]*row[3]
        # l2_hit_rate+=row[0]*row[4]*row[3]
        smem_footprint=max(smem_footprint,row[5])
        reg_footprint=max(reg_footprint,row[7])
        ddr_read_io+=row[9]
        l2_read_io+=row[10]

    # l2_hit_rate/=time_added
    # l2_hit_rate=l2_hit_rate/l2_util
    # print("time_added",time_added)
    # print("after fusion")
    l2_hit_rate=(1-(max(ddr_read_io,1)/max(l2_read_io,1)))

    waves=np.prod(grids)/arch_spec.sm_count

    # bound_time=max(ddr_util/arch_spec.ddr_max_util, l2_util/arch_spec.l2_max_util, smem_l1_util/waves*math.ceil(waves)/arch_spec.l1_max_util,compute_util/waves*math.ceil(waves)/arch_spec.compute_max_util)
    bound_time=max(ddr_util/arch_spec.ddr_max_util,l2_util/arch_spec.l2_max_util,smem_l1_util/arch_spec.l1_max_util,compute_util/arch_spec.compute_max_util)*math.ceil(waves)/waves
    # ddr_footprint=0 # TODO
    ddr_util=ddr_util/bound_time
    l2_util=l2_util/bound_time
    smem_l1_util=smem_l1_util/bound_time
    compute_util=compute_util/bound_time
    # print(bound_time,ddr_footprint,ddr_util,l2_hit_rate_added,l2_util,smem_footprint_added,smem_l1_util,reg_footprint_added,compute_util)

    # smem_l1_util=smem_l1_util/waves*math.ceil(waves)
    compute_util=compute_util/waves*math.ceil(waves)

    # current_max_util=max(ddr_util,l2_util,smem_l1_util*waves/math.ceil(waves),compute_util*waves/math.ceil(waves))
    # bound_time=bound_time*0.9/current_max_util


    posted=[bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util]
    

    return time_added,posted


# def smem_fuse_analyze(time_list,smem_fusion_list,grids,row_id,csv_path):
#     # print("Mutli single op added time:/ms",sum(time_list)*1000)

#     reg_fusion_time_added,post_data=smem_fusion(smem_fusion_list,grids)
#     # print("Mutli reg fused op added time:/ms",reg_fusion_time_added*1000)
#     modeling_metrics=get_modeling_metrics(post_data)

#     ncu_metrics = extract_metrics_from_row_by_id(csv_path, row_id)
#     # print_compare(ncu_metrics,modeling_metrics)
#     return ncu_metrics,modeling_metrics