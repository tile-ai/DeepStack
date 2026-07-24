
from tilesight.arch import Arch

def reg_fusion(reg_array,arch_spec:Arch):
    # print("bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util,ddr_read_io, l2_read_io")
    # (0.0002442525040764034, 0, 0.0, 0.0, 0.0, 512, 0.9, 44, 0.30000000000000004, 0, 0)
    time_added, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io=0,0,0,0,0,0,0,0,0,0
    # print(reg_array)

    for row in reg_array:
        # print(row)
        ddr_util+=row[0]*row[2]
        l2_util+=row[0]*row[4]
        smem_l1_util+=row[0]*row[6]
        compute_util+=row[0]*row[8]
        time_added+=row[0]
        # l2_hit_rate+=row[0]*row[3]
        # l2_hit_rate+=row[0]*row[4]*row[3]
        smem_footprint+=row[5]
        # reg_footprint+=row[7]
        reg_footprint=max(reg_footprint,row[7])
        ddr_read_io+=row[9]
        l2_read_io+=row[10]

    # for row in ret_array:
    #     print(row)
    #     sum[0]+=row[0]*row[2]
    #     sum[1]+=row[0]*row[4]
    #     sum[2]+=row[0]*row[6]
    #     sum[3]+=row[0]*row[8]
    #     time_added+=row[0]
    #     l2_hit_rate_added+=row[0]*row[3]
    #     smem_footprint_added+=row[5]
    #     reg_footprint_added+=row[7]

    l2_hit_rate=(1-(max(ddr_read_io,1)/max(l2_read_io,1)))
    # l2_hit_rate/=time_added
    # l2_hit_rate=l2_hit_rate/l2_util
    # print("time_added",time_added)
    # print("after fusion")
    bound_time=max(ddr_util/arch_spec.ddr_max_util,l2_util/arch_spec.l2_max_util,smem_l1_util/arch_spec.l1_max_util,compute_util/arch_spec.compute_max_util)
    # ddr_footprint=0 # TODO
    ddr_util=ddr_util/bound_time
    l2_util=l2_util/bound_time
    smem_l1_util=smem_l1_util/bound_time
    compute_util=compute_util/bound_time
    # print(bound_time,ddr_footprint,ddr_util,l2_hit_rate_added,l2_util,smem_footprint_added,smem_l1_util,reg_footprint_added,compute_util)


    posted=[bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io]


    return time_added,posted