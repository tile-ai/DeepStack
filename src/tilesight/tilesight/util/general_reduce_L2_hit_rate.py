# import numpy as np
# import math
# from math import gcd
# from .sdcm import sdcm

# # def general_reduce_l2_hitrate(N, F, H, W, C, KH, KW, S, D, P, tb_n, tb_f, tb_h, tb_w, c_rstep, L2_Cap, SM_Count, bytes_per_num):
# def general_reduce_l2_hitrate(out_axis_mapping, out_tb_shape, spatial_grids, in1_tb_spatial_shape, in2_tb_spatial_shape, in1_tb_reduction_shape, in2_tb_reduction_shape,L2_Cap, SM_Count,bytes_per_num,mem_levels):
    
#     # 可以考虑把mem_levels信息就在这里加入，使得hitrate计算更精准
#     # example:
#     #     mem_levels = {
#     #     'in1': '[1,1,1]', for ddr
#     #     'in2': '[0,1,1]', for smem
#     #     'out1': [0,0,1],  for reg
#     # }

#     in1_level = mem_levels['in1']
#     in2_level = mem_levels['in2']
#     out1_level = mem_levels['out1']

#     Num_Associative = 8
#     Bytes_per_cacheline = 128
#     Num_Cachelines = L2_Cap / Bytes_per_cacheline / Num_Associative

#     # 前提：每个所需要的数据不是很大,即每个的reduction_shape乘起来不大

#     # out_axis_mapping=[0,1,0,0]
#     #     [0],  # 输出轴 'n' 来自 input1 的第 0 个轴，是空间轴
#     #     [1],  # 输出轴 'f' 来自 input2 的第 0 个轴，是空间轴
#     #     [0],  # 输出轴 'h' 来自 input1 的第 2 个轴，是空间轴
#     #     [0]   # 输出轴 'w' 来自 input1 的第 3 个轴，是空间轴
#     # ]

#     # gridN = int(np.ceil(N / tb_n))
#     # gridF = int(np.ceil(F / tb_f))
#     # gridH = int(np.ceil(H / tb_h))
#     # gridW = int(np.ceil(W / tb_w))

#     # now it's
#     # spatial_grids=[gridN,gridF,gridH,gridW]
#     # 
#     in1_spatial_grids=[]# n h w
#     in2_spatial_grids=[]# f

#     in1_slice_index=[]
#     in2_slice_index=[]

#     for i, mapping in enumerate(out_axis_mapping):
#         dim_size=spatial_grids[i]
#         if (mapping==0):
#             in1_spatial_grids.append(dim_size)
#             in1_slice_index.append(i)
#         elif(mapping==1):
#             in2_spatial_grids.append(dim_size)
#             in2_slice_index.append(i)
#         elif(mapping==2):
#             in1_spatial_grids.append(dim_size)
#             in1_slice_index.append(i)
#             in2_spatial_grids.append(dim_size)
#             in2_slice_index.append(i)
#     # print("in1_spatial_grids",in1_spatial_grids)
#     # print("in1_slice_index",in1_slice_index)
#     # print("in1_tb_reduction_shape",in1_tb_reduction_shape)
#     # print(np.prod(in1_tb_reduction_shape)) # np.prod([])=0
#     # print("in2_spatial_grids",in2_spatial_grids)
#     # print("in2_slice_index",in2_slice_index)
#     # print("in2_tb_reduction_shape",in2_tb_reduction_shape)
            
    

#     # input_activation_RD = np.full((gridN, gridH, gridW), 1e8)
#     input_activation_RD = np.full(tuple(in1_spatial_grids), 1e8)
#     # kernel_RD = np.full(gridF, 1e8)
#     kernel_RD = np.full(tuple(in2_spatial_grids), 1e8)
    

#     input_block_Num_Cachelines = np.prod(in1_tb_spatial_shape) * np.prod(in1_tb_reduction_shape) * bytes_per_num / Bytes_per_cacheline
#     kernel_block_Num_Cachelines = np.prod(in2_tb_spatial_shape) * np.prod(in2_tb_reduction_shape) * bytes_per_num / Bytes_per_cacheline
#     output_block_Num_Cachelines = np.prod(out_tb_shape) * bytes_per_num / Bytes_per_cacheline
#     # print("input_block_Num_Cachelines",input_block_Num_Cachelines)
#     # print("kernel_block_Num_Cachelines",int(kernel_block_Num_Cachelines))
#     # print("output_block_Num_Cachelines",output_block_Num_Cachelines)

    
#     # gcd1 = gcd(int(input_block_Num_Cachelines), int(kernel_block_Num_Cachelines))
#     # gcd2 = gcd(int(output_block_Num_Cachelines), int(Num_Cachelines))
#     # gcd_all = gcd(gcd1, gcd2)

#     #向下取整
#     # Num_Cachelines //= gcd_all 
#     # input_block_Num_Cachelines //= gcd_all
#     # kernel_block_Num_Cachelines //= gcd_all
#     # output_block_Num_Cachelines //= gcd_all

#     #向上取整
#     # Num_Cachelines = math.ceil(Num_Cachelines/gcd_all)
#     # input_block_Num_Cachelines = math.ceil(input_block_Num_Cachelines/gcd_all)
#     # kernel_block_Num_Cachelines = math.ceil(kernel_block_Num_Cachelines/gcd_all)
#     # output_block_Num_Cachelines = math.ceil(output_block_Num_Cachelines/gcd_all)

#     # coords = np.zeros((int(gridN * gridF * gridH * gridW), 4), dtype=int)
#     # index = 0

#     # for n in range(int(gridN)):
#     #     for f in range(int(gridF)):
#     #         for h in range(int(gridH)):
#     #             for w in range(int(gridW)):
#     #                 coords[index] = [n, f, h, w]
#     #                 index += 1
#     coords=[]
#     for index in np.ndindex(tuple(spatial_grids)):
#         coords.append(index)

#     coords=np.array(coords)
#     in1_coords=coords[:,in1_slice_index]
#     # print(in1_coords)
#     in2_coords=coords[:,in2_slice_index]
#     # print(in2_coords)

#     hit_prob_ia = 0
#     hit_prob_kn = 0

#     for count in range(0, np.prod(spatial_grids), SM_Count):
#         length = min(SM_Count, np.prod(spatial_grids) - count)
#         shuffled_seq = np.random.permutation(length)

#         # IA_Unique_Per_IT = np.zeros((gridN, gridH, gridW))
#         # Kernel_Unique_Per_IT = np.zeros((gridF, gridH, gridW))
#         IA_Unique_Per_IT = np.zeros(tuple(in1_spatial_grids))
#         Kernel_Unique_Per_IT = np.zeros(tuple(in2_spatial_grids))

#         for i in range(length):
#             current_idx = count + shuffled_seq[i]
#             # n, f, h, w = coords[current_idx]

#             in1_coord=in1_coords[current_idx]
#             in2_coord=in2_coords[current_idx]

#             # tmp_ia_RD = max(input_activation_RD[n, h, w], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
#             # hit_prob_ia += sdcm(tmp_ia_RD, Num_Associative, Num_Associative * Num_Cachelines)
#             # # Implement or import the `sdcm` function here

#             # tmp_kn_RD = max(kernel_RD[f], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
#             # hit_prob_kn += sdcm(tmp_kn_RD, Num_Associative, Num_Associative * Num_Cachelines)
#             # # Implement or import the `sdcm` function here

#             tmp_ia_RD = max(input_activation_RD[tuple(in1_coord)], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
#             hit_prob_ia += sdcm(tmp_ia_RD, Num_Associative, Num_Associative * Num_Cachelines)
#             # Implement or import the `sdcm` function here

#             tmp_kn_RD = max(kernel_RD[tuple(in2_coord)], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
#             hit_prob_kn += sdcm(tmp_kn_RD, Num_Associative, Num_Associative * Num_Cachelines)
#             # Implement or import the `sdcm` function here

#             input_activation_RD[tuple(in1_coord)] = 0
#             kernel_RD[tuple(in2_coord)] = 0

#             IA_Unique_Per_IT[tuple(in1_coord)] = 1
#             Kernel_Unique_Per_IT[tuple(in2_coord)] = 1

#         input_activation_RD += output_block_Num_Cachelines * length + np.sum(IA_Unique_Per_IT) * input_block_Num_Cachelines + np.sum(Kernel_Unique_Per_IT) * kernel_block_Num_Cachelines
#         kernel_RD += output_block_Num_Cachelines * length + np.sum(IA_Unique_Per_IT) * input_block_Num_Cachelines + np.sum(Kernel_Unique_Per_IT) * kernel_block_Num_Cachelines

#     # DDR_IO = (gridN * gridF * gridH * gridW - hit_prob_ia) * input_block_Num_Cachelines + (gridN * gridF * gridH * gridW - hit_prob_kn) * kernel_block_Num_Cachelines
#     # L2_IO = gridN * gridF * gridH * gridW * (input_block_Num_Cachelines + kernel_block_Num_Cachelines)
#     # print("spatial_grids",spatial_grids)
#     # print("hit_prob_ia",hit_prob_ia)
#     # print("hit_prob_kn",hit_prob_kn)
#     # print("input_block_Num_Cachelines",input_block_Num_Cachelines)
#     # print("kernel_block_Num_Cachelines",kernel_block_Num_Cachelines)
#     DDR_IO = (np.prod(spatial_grids) - hit_prob_ia) * input_block_Num_Cachelines + (np.prod(spatial_grids) - hit_prob_kn) * kernel_block_Num_Cachelines
#     L2_IO = np.prod(spatial_grids) * (input_block_Num_Cachelines + kernel_block_Num_Cachelines)
#     hitrate = 1 - DDR_IO / L2_IO

#     return hitrate


import numpy as np
import math
from math import gcd
from .sdcm import sdcm

# def general_reduce_l2_hitrate(N, F, H, W, C, KH, KW, S, D, P, tb_n, tb_f, tb_h, tb_w, c_rstep, L2_Cap, SM_Count, bytes_per_num):
def general_reduce_l2_hitrate(out_axis_mapping, out_tb_shape, spatial_grids, in1_tb_spatial_shape, in2_tb_spatial_shape, in1_tb_reduction_shape, in2_tb_reduction_shape,L2_Cap, SM_Count,mem_levels):
    
    # 可以考虑把mem_levels信息就在这里加入，使得hitrate计算更精准
    # example:
    #     mem_levels = {
    #     'in1': '[1,1,1]', for ddr
    #     'in2': '[0,1,1]', for smem
    #     'out1': [0,0,1],  for reg
    # }

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']
    # print(mem_levels)
    
    # print("in1_level",in1_level)
    in1_ddr_flag=in1_level[0]
    in2_ddr_flag=in2_level[0]
    out1_ddr_flag=out1_level[0]
    # print(in1_ddr_flag,in2_ddr_flag,out1_ddr_flag)
    # out1_ddr_flag=1

    Num_Associative = 8
    Bytes_per_cacheline = 128
    Num_Cachelines = L2_Cap / Bytes_per_cacheline / Num_Associative

    # 前提：每个所需要的数据不是很大,即每个的reduction_shape乘起来不大

    # out_axis_mapping=[0,1,0,0]
    #     [0],  # 输出轴 'n' 来自 input1 的第 0 个轴，是空间轴
    #     [1],  # 输出轴 'f' 来自 input2 的第 0 个轴，是空间轴
    #     [0],  # 输出轴 'h' 来自 input1 的第 2 个轴，是空间轴
    #     [0]   # 输出轴 'w' 来自 input1 的第 3 个轴，是空间轴
    # ]

    # gridN = int(np.ceil(N / tb_n))
    # gridF = int(np.ceil(F / tb_f))
    # gridH = int(np.ceil(H / tb_h))
    # gridW = int(np.ceil(W / tb_w))

    # now it's
    # spatial_grids=[gridN,gridF,gridH,gridW]
    # 

    # for high performance, if-else should be used the least time as possible
    
    if ((in1_ddr_flag==1) and (in2_ddr_flag==1)):
        # print("in1_ddr_flag==1&in2_ddr_flag==1")
        in1_spatial_grids=[]# n h w
        in2_spatial_grids=[]# f

        in1_slice_index=[]
        in2_slice_index=[]

        for i, mapping in enumerate(out_axis_mapping):
            dim_size=spatial_grids[i]
            if (mapping==0):
                in1_spatial_grids.append(dim_size)
                in1_slice_index.append(i)
            elif(mapping==1):
                in2_spatial_grids.append(dim_size)
                in2_slice_index.append(i)
            elif(mapping==2):
                in1_spatial_grids.append(dim_size)
                in1_slice_index.append(i)
                in2_spatial_grids.append(dim_size)
                in2_slice_index.append(i)
        # print("in1_spatial_grids",in1_spatial_grids)
        # print("in1_slice_index",in1_slice_index)
        # print("in1_tb_reduction_shape",in1_tb_reduction_shape)
        # print(np.prod(in1_tb_reduction_shape)) # np.prod([])=0
        # print("in2_spatial_grids",in2_spatial_grids)
        # print("in2_slice_index",in2_slice_index)
        # print("in2_tb_reduction_shape",in2_tb_reduction_shape)
                
        

        # input_activation_RD = np.full((gridN, gridH, gridW), 1e8)
        input_activation_RD = np.full(tuple(in1_spatial_grids), 1e8)
        # kernel_RD = np.full(gridF, 1e8)
        kernel_RD = np.full(tuple(in2_spatial_grids), 1e8)
        

        input_block_Num_Cachelines = np.prod(in1_tb_spatial_shape) * np.prod(in1_tb_reduction_shape) * in1_level[-1] / Bytes_per_cacheline
        kernel_block_Num_Cachelines = np.prod(in2_tb_spatial_shape) * np.prod(in2_tb_reduction_shape) * in2_level[-1] / Bytes_per_cacheline
        output_block_Num_Cachelines = np.prod(out_tb_shape) * out1_level[-1] / Bytes_per_cacheline
        # print("input_block_Num_Cachelines",input_block_Num_Cachelines)
        # print("kernel_block_Num_Cachelines",int(kernel_block_Num_Cachelines))
        # print("output_block_Num_Cachelines",output_block_Num_Cachelines)

        
        # gcd1 = gcd(int(input_block_Num_Cachelines), int(kernel_block_Num_Cachelines))
        # gcd2 = gcd(int(output_block_Num_Cachelines), int(Num_Cachelines))
        # gcd_all = gcd(gcd1, gcd2)

        #向下取整
        # Num_Cachelines //= gcd_all 
        # input_block_Num_Cachelines //= gcd_all
        # kernel_block_Num_Cachelines //= gcd_all
        # output_block_Num_Cachelines //= gcd_all

        #向上取整
        # Num_Cachelines = math.ceil(Num_Cachelines/gcd_all)
        # input_block_Num_Cachelines = math.ceil(input_block_Num_Cachelines/gcd_all)
        # kernel_block_Num_Cachelines = math.ceil(kernel_block_Num_Cachelines/gcd_all)
        # output_block_Num_Cachelines = math.ceil(output_block_Num_Cachelines/gcd_all)

        # coords = np.zeros((int(gridN * gridF * gridH * gridW), 4), dtype=int)
        # index = 0

        # for n in range(int(gridN)):
        #     for f in range(int(gridF)):
        #         for h in range(int(gridH)):
        #             for w in range(int(gridW)):
        #                 coords[index] = [n, f, h, w]
        #                 index += 1
        coords=[]
        for index in np.ndindex(tuple(spatial_grids)):
            coords.append(index)

        coords=np.array(coords)
        in1_coords=coords[:,in1_slice_index]
        # print(in1_coords)
        in2_coords=coords[:,in2_slice_index]
        # print(in2_coords)

        hit_prob_ia = 0
        hit_prob_kn = 0

        for count in range(0, np.prod(spatial_grids), SM_Count):
            length = min(SM_Count, np.prod(spatial_grids) - count)
            shuffled_seq = np.random.permutation(length)

            # IA_Unique_Per_IT = np.zeros((gridN, gridH, gridW))
            # Kernel_Unique_Per_IT = np.zeros((gridF, gridH, gridW))
            IA_Unique_Per_IT = np.zeros(tuple(in1_spatial_grids))
            Kernel_Unique_Per_IT = np.zeros(tuple(in2_spatial_grids))

            for i in range(length):
                current_idx = count + shuffled_seq[i]
                # n, f, h, w = coords[current_idx]

                in1_coord=in1_coords[current_idx]
                in2_coord=in2_coords[current_idx]

                # tmp_ia_RD = max(input_activation_RD[n, h, w], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
                # hit_prob_ia += sdcm(tmp_ia_RD, Num_Associative, Num_Associative * Num_Cachelines)
                # # Implement or import the `sdcm` function here

                # tmp_kn_RD = max(kernel_RD[f], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
                # hit_prob_kn += sdcm(tmp_kn_RD, Num_Associative, Num_Associative * Num_Cachelines)
                # # Implement or import the `sdcm` function here

                tmp_ia_RD = max(input_activation_RD[tuple(in1_coord)], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
                hit_prob_ia += sdcm(tmp_ia_RD, Num_Associative, Num_Associative * Num_Cachelines)
                # Implement or import the `sdcm` function here

                tmp_kn_RD = max(kernel_RD[tuple(in2_coord)], (input_block_Num_Cachelines + kernel_block_Num_Cachelines) / 2)
                hit_prob_kn += sdcm(tmp_kn_RD, Num_Associative, Num_Associative * Num_Cachelines)
                # Implement or import the `sdcm` function here

                input_activation_RD[tuple(in1_coord)] = 0
                kernel_RD[tuple(in2_coord)] = 0

                IA_Unique_Per_IT[tuple(in1_coord)] = 1
                Kernel_Unique_Per_IT[tuple(in2_coord)] = 1

            input_activation_RD += output_block_Num_Cachelines * length * out1_ddr_flag + np.sum(IA_Unique_Per_IT) * input_block_Num_Cachelines + np.sum(Kernel_Unique_Per_IT) * kernel_block_Num_Cachelines
            kernel_RD += output_block_Num_Cachelines * length * out1_ddr_flag + np.sum(IA_Unique_Per_IT) * input_block_Num_Cachelines + np.sum(Kernel_Unique_Per_IT) * kernel_block_Num_Cachelines

        # DDR_IO = (gridN * gridF * gridH * gridW - hit_prob_ia) * input_block_Num_Cachelines + (gridN * gridF * gridH * gridW - hit_prob_kn) * kernel_block_Num_Cachelines
        # L2_IO = gridN * gridF * gridH * gridW * (input_block_Num_Cachelines + kernel_block_Num_Cachelines)
        # print("spatial_grids",spatial_grids)
        # print("hit_prob_ia",hit_prob_ia)
        # print("hit_prob_kn",hit_prob_kn)
        # print("input_block_Num_Cachelines",input_block_Num_Cachelines)
        # print("kernel_block_Num_Cachelines",kernel_block_Num_Cachelines)
        DDR_IO = (np.prod(spatial_grids) - hit_prob_ia) * input_block_Num_Cachelines  + (np.prod(spatial_grids) - hit_prob_kn) * kernel_block_Num_Cachelines 
        L2_IO = np.prod(spatial_grids) * (input_block_Num_Cachelines  + kernel_block_Num_Cachelines )
        hitrate = 1 - DDR_IO / max(L2_IO,1)
    elif ((in1_ddr_flag==1) and (in2_ddr_flag==0)):
        # print("in1_ddr_flag==1&in2_ddr_flag==0")
        in1_spatial_grids=[]# n h w

        in1_slice_index=[]

        for i, mapping in enumerate(out_axis_mapping):
            dim_size=spatial_grids[i]
            if (mapping==0):
                in1_spatial_grids.append(dim_size)
                in1_slice_index.append(i)
            elif(mapping==2):
                in1_spatial_grids.append(dim_size)
                in1_slice_index.append(i)

        input_activation_RD = np.full(tuple(in1_spatial_grids), 1e8)

        

        input_block_Num_Cachelines = np.prod(in1_tb_spatial_shape) * np.prod(in1_tb_reduction_shape) * in1_level[-1] / Bytes_per_cacheline
        output_block_Num_Cachelines = np.prod(out_tb_shape) * out1_level[-1] / Bytes_per_cacheline

        coords=[]
        for index in np.ndindex(tuple(spatial_grids)):
            coords.append(index)

        coords=np.array(coords)
        in1_coords=coords[:,in1_slice_index]

        hit_prob_ia = 0

        for count in range(0, np.prod(spatial_grids), SM_Count):
            length = min(SM_Count, np.prod(spatial_grids) - count)
            shuffled_seq = np.random.permutation(length)

            # IA_Unique_Per_IT = np.zeros((gridN, gridH, gridW))
            # Kernel_Unique_Per_IT = np.zeros((gridF, gridH, gridW))
            IA_Unique_Per_IT = np.zeros(tuple(in1_spatial_grids))

            for i in range(length):
                current_idx = count + shuffled_seq[i]
                # n, f, h, w = coords[current_idx]

                in1_coord=in1_coords[current_idx]

                tmp_ia_RD = max(input_activation_RD[tuple(in1_coord)], (input_block_Num_Cachelines ) / 2)
                hit_prob_ia += sdcm(tmp_ia_RD, Num_Associative, Num_Associative * Num_Cachelines)

                input_activation_RD[tuple(in1_coord)] = 0

                IA_Unique_Per_IT[tuple(in1_coord)] = 1

            input_activation_RD += output_block_Num_Cachelines * length * out1_ddr_flag + np.sum(IA_Unique_Per_IT) * input_block_Num_Cachelines 


        DDR_IO = (np.prod(spatial_grids) - hit_prob_ia) * input_block_Num_Cachelines 
        L2_IO = np.prod(spatial_grids) * (input_block_Num_Cachelines)
        hitrate = 1 - DDR_IO / max(L2_IO,1)
    elif ((in1_ddr_flag==0) and (in2_ddr_flag==1)):
        # print("in1_ddr_flag==0&in2_ddr_flag==1")
        # print(out_axis_mapping)
        # print(in2_tb_reduction_shape)
        in2_spatial_grids=[]# f

        in2_slice_index=[]

        for i, mapping in enumerate(out_axis_mapping):
            dim_size=spatial_grids[i]
            if(mapping==1):
                in2_spatial_grids.append(dim_size)
                in2_slice_index.append(i)
            elif(mapping==2):
                in2_spatial_grids.append(dim_size)
                in2_slice_index.append(i)

        kernel_RD = np.full(tuple(in2_spatial_grids), 1e8)
        # print(in2_spatial_grids,kernel_RD)
        
        kernel_block_Num_Cachelines = np.prod(in2_tb_spatial_shape) * np.prod(in2_tb_reduction_shape) * in2_level[-1] / Bytes_per_cacheline
        output_block_Num_Cachelines = np.prod(out_tb_shape) * out1_level[-1] / Bytes_per_cacheline

        coords=[]
        for index in np.ndindex(tuple(spatial_grids)):
            coords.append(index)

        coords=np.array(coords)
        in2_coords=coords[:,in2_slice_index]
        # print(in2_coords)

        hit_prob_kn = 0

        for count in range(0, np.prod(spatial_grids), SM_Count):
            length = min(SM_Count, np.prod(spatial_grids) - count)
            shuffled_seq = np.random.permutation(length)

            Kernel_Unique_Per_IT = np.zeros(tuple(in2_spatial_grids))

            for i in range(length):
                current_idx = count + shuffled_seq[i]
                # n, f, h, w = coords[current_idx]
                in2_coord=in2_coords[current_idx]

                tmp_kn_RD = max(kernel_RD[tuple(in2_coord)], (kernel_block_Num_Cachelines) / 2)
                hit_prob_kn += sdcm(tmp_kn_RD, Num_Associative, Num_Associative * Num_Cachelines)
                # Implement or import the `sdcm` function here

                kernel_RD[tuple(in2_coord)] = 0

                Kernel_Unique_Per_IT[tuple(in2_coord)] = 1

            kernel_RD += output_block_Num_Cachelines * length * out1_ddr_flag + np.sum(Kernel_Unique_Per_IT) * kernel_block_Num_Cachelines

        DDR_IO = (np.prod(spatial_grids) - hit_prob_kn) * kernel_block_Num_Cachelines 
        L2_IO = np.prod(spatial_grids) * ( kernel_block_Num_Cachelines )
        hitrate = 1 - DDR_IO / max(L2_IO,1)
    elif ((in1_ddr_flag==0) and (in2_ddr_flag==0)):
        # print("in1_ddr_flag==0&in2_ddr_flag==0")
        hitrate=0

    return hitrate

