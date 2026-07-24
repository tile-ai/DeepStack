import numpy as np
from math import gcd
from .extract_blocks import extract_blocks
from .sdcm import sdcm

def implicit_gemm_l2_hitrate_reuse_distance(M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, CONV_C, CONV_KH, CONV_KW):
    gridM = np.ceil(M / tb_m)
    gridN = np.ceil(N / tb_n)
    gridK = np.ceil(K / tb_k)
    grid_noduplication_K = np.ceil(CONV_C / tb_k)

    Stride_M = SM_Count
    Stride_N = SM_Count // Stride_M

    # Cache parameters
    Num_Associative = 8
    Bytes_per_cacheline = 128
    Num_Cachelines = L2_Cap / Num_Associative / Bytes_per_cacheline

    MN_Num_Cachelines = tb_m * tb_n * BYTE_per_num / Bytes_per_cacheline
    MK_Num_Cachelines = tb_m * tb_k * BYTE_per_num / Bytes_per_cacheline
    NK_Num_Cachelines = tb_n * tb_k * BYTE_per_num / Bytes_per_cacheline

    gcd1 = gcd(int(MN_Num_Cachelines), int(MK_Num_Cachelines))
    gcd2 = gcd(int(NK_Num_Cachelines), int(Num_Cachelines))
    gcd_all = gcd(gcd1, gcd2)

    Num_Cachelines //= gcd_all
    MN_Num_Cachelines //= gcd_all
    MK_Num_Cachelines //= gcd_all
    NK_Num_Cachelines //= gcd_all

    coords = extract_blocks(int(gridM), int(gridN), Stride_M, Stride_N)

    M_RD = np.full(int(gridM), 1e9)
    N_RD = np.full(int(gridN), 1e9)
    hit_prob_mk = 0
    hit_prob_nk = 0

    for count in range(0, int(gridM * gridN), Stride_M * Stride_N):
        length = min(Stride_M * Stride_N, int(gridM * gridN) - count)
        shuffled_seq = np.random.permutation(length) + 1  # +1 to mimic MATLAB's 1-based indexing
        M_Unique_per_It = np.zeros(int(gridM))
        N_Unique_per_It = np.zeros(int(gridN))

        for i in range(length):
            current_idx = count + shuffled_seq[i] - 1  # -1 to adjust for 0-based indexing
            m, n = coords[current_idx]  # Adjusted for 0-based indexing
            
            # Adjusting calculations for 0-based indexing
            m -= 1
            n -= 1

            tmp_mk_rd = max(M_RD[m], tb_m * CONV_C * BYTE_per_num / Bytes_per_cacheline / gcd_all)
            hit_prob_mk += sdcm(tmp_mk_rd, Num_Associative, Num_Cachelines * Num_Associative)

            tmp_nk_rd = max(N_RD[n], tb_n * CONV_C * BYTE_per_num / Bytes_per_cacheline / gcd_all)
            hit_prob_nk += CONV_KH * CONV_KW * sdcm(tmp_nk_rd, Num_Associative, Num_Cachelines * Num_Associative)

            M_RD[m] = 0
            N_RD[n] = 0

            M_Unique_per_It[m] = 1
            N_Unique_per_It[n] = 1

        if CONV_KH * CONV_KW != 1:
            tmp_mk_rd = grid_noduplication_K * (MK_Num_Cachelines * np.sum(M_Unique_per_It) + NK_Num_Cachelines * np.sum(N_Unique_per_It))
            hit_prob_mk += length * (CONV_KH * CONV_KW - 1) * sdcm(tmp_mk_rd, Num_Associative, Num_Cachelines * Num_Associative)

        M_RD += Stride_M * Stride_N * MN_Num_Cachelines + (grid_noduplication_K) * MK_Num_Cachelines * np.sum(M_Unique_per_It) + gridK * NK_Num_Cachelines * np.sum(N_Unique_per_It)
        N_RD += Stride_M * Stride_N * MN_Num_Cachelines + (grid_noduplication_K) * MK_Num_Cachelines * np.sum(M_Unique_per_It) + gridK * NK_Num_Cachelines * np.sum(N_Unique_per_It)

    L2_IO = gridM * gridN * CONV_KH * CONV_KW * (MK_Num_Cachelines + NK_Num_Cachelines)
    DDR_IO = (gridM * gridN * CONV_KH * CONV_KW - hit_prob_mk) * MK_Num_Cachelines + (gridM * gridN * CONV_KH * CONV_KW - hit_prob_nk) * NK_Num_Cachelines

    hit_rate = 1 - DDR_IO / L2_IO

    return hit_rate


# # 使用实际参数调用函数
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, CONV_C, CONV_KH, CONV_KW=100352,128,1152,128,64,32,31457280,108,2,128,3,3
# hit_rate = implicit_gemm_l2_hitrate_reuse_distance(M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, CONV_C, CONV_KH, CONV_KW)
# print(f"Hit Rate: {hit_rate}")
