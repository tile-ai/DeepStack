# 文件名: matmul_op.py
from ..util import *
from tilesight.arch import uses_gpu_resource_model
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, stage_num, block_per_sm
import numpy as np
import math

# def calculate_conv_implicit_gemm_resource_utilization(m,n,k,tb_m,tb_n,tb_k,wp_m,wp_n,wp_k,arch.l2_capacity,arch.sm_count, bytes_per_num, stage_num,arch):
def calculate_conv_implicit_gemm_sdp_resource_utilization(op_shape,tb_shape,wp_shape, bytes_per_num, stage_num,arch,mem_levels, stride=1,dialation=1,padding=0):
    # dialation is always 1 in most cases
    # actually, we update stride and dialation

    # conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw,conv_s,conv_d,conv_p=op_shape
    conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw=op_shape
    tb_m,tb_n,tb_k=tb_shape
    wp_m,wp_n,wp_k=wp_shape

    # inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    # inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    # padh = inh + 2 * p
    # padw = inw + 2 * p

    
    # m,n,k,tb_m,tb_n,tb_k,wp_m,wp_n,wp_k,arch.l2_capacity,arch.sm_count, bytes_per_num, stage_num,arch
    m=conv_n*conv_h*conv_w
    n=conv_f
    k=conv_kh*conv_kw*conv_c

    DDR_non_ideal_para=1.1 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    compute_flops=2*m*n*k
    # print("ideal time",compute_flops/arch.fp16_tensor_flops/0.9)
    # implicit_gemm_l2_hitrate_reuse_distance(M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, CONV_C, CONV_KH, CONV_KW):
    l2_hit_rate=implicit_gemm_l2_hitrate_reuse_distance(m, n, k, tb_m, tb_n, tb_k, arch.l2_capacity, arch.sm_count, bytes_per_num, conv_c, conv_kh, conv_kw)
    
    # print("l2_hit_rate",l2_hit_rate)

    # l2_hit_rate= 0.9

    gridM=math.ceil(m/tb_m)
    gridN=math.ceil(n/tb_n)

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    l2_read_io=gridM*gridN*k*(tb_m*in1_level[0]+tb_n*in2_level[0])*bytes_per_num
    l2_store_io=gridM*gridN*(tb_m*tb_n)*bytes_per_num*out1_level[0]

    ddr_io=l2_read_io*(1-l2_hit_rate)+l2_store_io
    ddr_io=ddr_io*DDR_non_ideal_para

    active_warp_per_tb=(tb_m/wp_m)*(tb_n/wp_n)

    if (arch.core=="A100" or arch.core=="H100" or arch.core=="B200"):
        #a100 with special two-part l2 cache structure
        l2_io=l2_read_io*(l2_hit_rate)+l2_read_io*(1-l2_hit_rate)*2+l2_store_io*2
    else:
        l2_io=l2_read_io+l2_store_io

    smem_footprint=(tb_m+tb_n)*tb_k*bytes_per_num*stage_num

    if uses_gpu_resource_model(arch):
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / bytes_per_num)) + math.ceil(wp_m * 32 / 32 / (4 / bytes_per_num)) + math.ceil(wp_n * 32 / 32 / (4 / bytes_per_num)))
        reg_footprint=reg_footprint*REG_spill_para
        reg_footprint=math.ceil(reg_footprint)
    else:#with dup
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / bytes_per_num)) + 1*math.ceil(wp_m * 32 / 32 / (4 / bytes_per_num)) + 1*math.ceil(wp_n * 32 / 32 / (4 / bytes_per_num)))
        reg_footprint=reg_footprint*REG_spill_para
        reg_footprint=math.ceil(reg_footprint)


    if uses_gpu_resource_model(arch):
        #somehow 0.5 or 0.25 * ldgsts
        smem_io=0.5*(gridM*gridN*k*(tb_m*in1_level[0]+tb_n*in2_level[0])*bytes_per_num)  #derived in1&in2 level[0]
        #shared load
        smem_io+=(gridM*gridN*k*(tb_m+tb_n)*bytes_per_num)*active_warp_per_tb*(wp_m*in1_level[1]+wp_n*in2_level[1])/(tb_m+tb_n)
        #epilogue, shared store
        # maybe no epilogue? by cutlass with epilogue, but by welder no epilogue 
        # -----------------------
        # smem_io+=l2_store_io
        #epilogue, shared load
        # -----------------------
        # smem_io+=l2_store_io
        #store global
        l1_io=(gridM*gridN*(tb_m*tb_n)*bytes_per_num)*out1_level[1]
        #add together
        smem_l1_io=l1_io+smem_io
    else:#noldgsts
        #load global
        l1_io=(gridM*gridN*k*(tb_m*in1_level[0]+tb_n*in2_level[0])*bytes_per_num)
        #store shared
        smem_io=(gridM*gridN*k*(tb_m*in1_level[0]+tb_n*in2_level[0])*bytes_per_num)
        #shared load
        smem_io+=(gridM*gridN*k*(tb_m+tb_n)*bytes_per_num)*active_warp_per_tb*(wp_m*in1_level[1]+wp_n*in2_level[1])/(tb_m+tb_n)
        #epilogue, shared store
        # -----------------------
        # smem_io+=l2_store_io
        #epilogue, shared load
        # -----------------------
        # smem_io+=l2_store_io
        #store global
        l1_io=(gridM*gridN*(tb_m*tb_n)*bytes_per_num)*out1_level[1]
        #add together
        smem_l1_io=l1_io+smem_io
    
    ddr_read_io=l2_read_io*(1-l2_hit_rate)

    overheads=(4 / active_warp_per_tb if active_warp_per_tb <= 4 else
               8 / active_warp_per_tb if active_warp_per_tb <= 8 else
               12 / active_warp_per_tb if active_warp_per_tb <= 12 else
               16 / active_warp_per_tb if active_warp_per_tb <= 16 else
               1)
    
    compute_flops=compute_flops*overheads
    smem_l1_io=smem_l1_io*overheads

    # return 1, 1, 1, 1, 1, 1, 1, 1, 1

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        
