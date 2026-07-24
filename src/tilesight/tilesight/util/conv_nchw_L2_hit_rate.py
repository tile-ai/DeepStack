import numpy as np
from math import gcd
from .extract_blocks import extract_blocks
from .sdcm import sdcm

def nchw_conv_l2_hitrate(N, F, H, W, C, KH, KW, S, D, P, tb_n, tb_f, tb_h, tb_w, c_rstep, L2_Cap, SM_Count, BYTE_per_num):
    Num_Associative = 8
    Bytes_per_cacheline = 128
    Num_Cachelines = L2_Cap / Bytes_per_cacheline / Num_Associative

    gridN = int(np.ceil(N / tb_n))
    gridF = int(np.ceil(F / tb_f))
    gridH = int(np.ceil(H / tb_h))
    gridW = int(np.ceil(W / tb_w))

    input_activation_RD = np.full((gridN, gridH, gridW), 1e8)
    kernel_RD = np.full(gridF, 1e8)
    

    input_block_Num_Cachelines = tb_n * C * tb_h * tb_w * BYTE_per_num / Bytes_per_cacheline
    kernel_block_Num_Cachelines = tb_f * C * KH * KW * BYTE_per_num / Bytes_per_cacheline
    output_block_Num_Cachelines = tb_n * tb_f * tb_h * tb_w * BYTE_per_num / Bytes_per_cacheline

    gcd1 = gcd(int(input_block_Num_Cachelines), int(kernel_block_Num_Cachelines))
    gcd2 = gcd(int(output_block_Num_Cachelines), int(Num_Cachelines))
    gcd_all = gcd(gcd1, gcd2)

    Num_Cachelines //= gcd_all
    input_block_Num_Cachelines //= gcd_all
    kernel_block_Num_Cachelines //= gcd_all
    output_block_Num_Cachelines //= gcd_all

    coords = np.zeros((int(gridN * gridF * gridH * gridW), 4), dtype=int)
    index = 0

    hit_prob_ia = 0
    hit_prob_kn = 0

    for n in range(int(gridN)):
        for f in range(int(gridF)):
            for h in range(int(gridH)):
                for w in range(int(gridW)):
                    coords[index] = [n, f, h, w]
                    index += 1

    for count in range(0, int(gridN * gridF * gridH * gridW), SM_Count):
        length = min(SM_Count, int(gridN * gridF * gridH * gridW) - count)
        shuffled_seq = np.random.permutation(length)

        IA_Unique_Per_IT = np.zeros((gridN, gridH, gridW))
        # Kernel_Unique_Per_IT = np.zeros((gridF, gridH, gridW))
        Kernel_Unique_Per_IT = np.zeros((gridF))

        for i in range(length):
            current_idx = count + shuffled_seq[i]
            n, f, h, w = coords[current_idx]

            tmp_ia_RD = max(input_activation_RD[n, h, w], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
            hit_prob_ia += sdcm(tmp_ia_RD, Num_Associative, Num_Associative * Num_Cachelines)
            # Implement or import the `sdcm` function here

            tmp_kn_RD = max(kernel_RD[f], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
            hit_prob_kn += sdcm(tmp_kn_RD, Num_Associative, Num_Associative * Num_Cachelines)
            # Implement or import the `sdcm` function here

            input_activation_RD[n, h, w] = 0
            kernel_RD[f] = 0

            IA_Unique_Per_IT[n, h, w] = 1
            # Kernel_Unique_Per_IT[f, h, w] = 1
            Kernel_Unique_Per_IT[f] = 1

        input_activation_RD += output_block_Num_Cachelines * length + np.sum(IA_Unique_Per_IT) * input_block_Num_Cachelines + np.sum(Kernel_Unique_Per_IT) * kernel_block_Num_Cachelines
        kernel_RD += output_block_Num_Cachelines * length + np.sum(IA_Unique_Per_IT) * input_block_Num_Cachelines + np.sum(Kernel_Unique_Per_IT) * kernel_block_Num_Cachelines

    DDR_IO = (gridN * gridF * gridH * gridW - hit_prob_ia) * input_block_Num_Cachelines + (gridN * gridF * gridH * gridW - hit_prob_kn) * kernel_block_Num_Cachelines
    L2_IO = gridN * gridF * gridH * gridW * (input_block_Num_Cachelines + kernel_block_Num_Cachelines)
    hitrate = 1 - DDR_IO / L2_IO

    return hitrate


# N =128
# F =128
# H =28
# W =28
# C =128
# KH =3
# KW =3
# S =1
# D =1
# P =1
# tb_n =1
# tb_f =16
# tb_h =14
# tb_w =28
# c_rstep =8
# L2_Cap =31457280
# SM_Count =108
# BYTE_per_num =2

# hit_rate=nchw_conv_l2_hitrate(N, F, H, W, C, KH, KW, S, D, P, tb_n, tb_f, tb_h, tb_w, c_rstep, L2_Cap, SM_Count, BYTE_per_num)
# print(hit_rate)