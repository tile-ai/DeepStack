import numpy as np
from math import gcd
from numpy.random import permutation
# from .extract_blocks import extract_blocks
from tilesight.util.extract_blocks import *
from tilesight.util.sdcm import sdcm

# 确保已经定义了之前转换的 extract_blocks 和 sdcm 函数
def L2_hit_rate_flow_sim_reuse_distance_triton_swizzle(M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, mem_levels, group_m):
    gridM = np.ceil(M / tb_m)
    gridN = np.ceil(N / tb_n)
    gridK = np.ceil(K / tb_k)

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    # Stride_N=row_panel
    # Stride_M = SM_Count // Stride_N
    # print(Stride_M)


    # Stride_M = SM_Count
    # Stride_N = SM_Count // Stride_M

    Num_Associative = 8
    Bytes_per_cacheline = 128
    Num_Cachelines = L2_Cap / (Num_Associative * Bytes_per_cacheline)

    MN_Num_Cachelines = tb_m * tb_n * out1_level[-1] / Bytes_per_cacheline
    MK_Num_Cachelines = tb_m * tb_k * in1_level[-1] / Bytes_per_cacheline
    NK_Num_Cachelines = tb_n * tb_k * in2_level[-1] / Bytes_per_cacheline

    # gcd1 = gcd(int(MN_Num_Cachelines), int(MK_Num_Cachelines))
    # gcd2 = gcd(int(NK_Num_Cachelines), int(Num_Cachelines))
    # gcd_all = gcd(gcd1, gcd2)

    # Num_Cachelines = Num_Cachelines / gcd_all
    # MN_Num_Cachelines = MN_Num_Cachelines / gcd_all
    # MK_Num_Cachelines = MK_Num_Cachelines / gcd_all
    # NK_Num_Cachelines = NK_Num_Cachelines / gcd_all

    coords = extract_blocks_triton_swizzle_column_major(int(gridM), int(gridN), int(group_m))
    # print(coords)

    M_RD = np.ones(int(gridM)) * 1e9
    N_RD = np.ones(int(gridN)) * 1e9
    hit_prob_mk = 0
    hit_prob_nk = 0

    for count in range(0, int(gridM * gridN), SM_Count):
        length = min(SM_Count, int(gridM * gridN) - count)
        shuffled_seq = permutation(length)
        
        M_Unique_per_It = np.zeros(int(gridM))
        N_Unique_per_It = np.zeros(int(gridN))
        
        for i in range(length):
            current_idx = count + shuffled_seq[i]
            m = int(coords[current_idx, 0])
            n = int(coords[current_idx, 1])
            
            hit_prob_mk += sdcm(M_RD[m - 1], Num_Associative, Num_Cachelines * Num_Associative)
            hit_prob_nk += sdcm(N_RD[n - 1], Num_Associative, Num_Cachelines * Num_Associative)
            
            M_RD[m - 1] = -0.5 * MK_Num_Cachelines
            N_RD[n - 1] = -MK_Num_Cachelines - 0.5 * NK_Num_Cachelines
            
            M_RD += MK_Num_Cachelines + NK_Num_Cachelines
            N_RD += MK_Num_Cachelines + NK_Num_Cachelines
            
            M_Unique_per_It[m - 1] = 1
            N_Unique_per_It[n - 1] = 1
        
        M_RD += SM_Count * MN_Num_Cachelines + (gridK - 1) * (MK_Num_Cachelines * np.sum(M_Unique_per_It) + NK_Num_Cachelines * np.sum(N_Unique_per_It))
        N_RD += SM_Count * MN_Num_Cachelines + (gridK - 1) * (MK_Num_Cachelines * np.sum(M_Unique_per_It) + NK_Num_Cachelines * np.sum(N_Unique_per_It))

    L2_IO = gridM * gridN * (MK_Num_Cachelines + NK_Num_Cachelines)
    DDR_IO = (gridM * gridN - hit_prob_mk) * MK_Num_Cachelines + (gridM * gridN - hit_prob_nk) * NK_Num_Cachelines

    hit_rate = 1 - DDR_IO / L2_IO

    return hit_rate

# # 使用实际参数调用函数81024
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, stage_num, block_per_sm=8192,8192,8192,128,128,32,8102431457280,108,2,3,1
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count = 8192,8192,8192,128,128,32,720*1024*1024,128
# mem_levels = {
#     'in1': [1,1,1,2], 
#     'in2': [1,1,1,2], 
#     'out1':[1,1,1,2],  
# }
# row_panel=int(SM_Count/8)
# hit_rate=L2_hit_rate_flow_sim_reuse_distance(M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, mem_levels, row_panel)
# print(f"Hit Rate: {hit_rate}")
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count = 16384,16384,16384,128,128,32,72*1024*1024,128
# mem_levels = {
#     'in1': [1,1,1,2], 
#     'in2': [1,1,1,2], 
#     'out1':[1,1,1,2],  
# }
# row_panel=int(SM_Count/8)
# hit_rate=L2_hit_rate_flow_sim_reuse_distance(M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, mem_levels, row_panel)
# print(f"Hit Rate: {hit_rate}")
