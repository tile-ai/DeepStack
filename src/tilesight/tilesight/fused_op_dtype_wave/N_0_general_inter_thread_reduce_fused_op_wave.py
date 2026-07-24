# 文件名: N_1_element_wise_op.py
import numpy as np
import math
from ..util import *
from tilesight.arch import uses_dram_wave_quantization

import timeit

import logging
log = logging.getLogger(__name__) 

def calculate_N_0_general_ruduce_inter_thread_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch,mem_levels,compute_at=1):
# ret=calculate_N_N_elementwise_resource_utilization([20480,20480],[10,160],[10,10], 4,chip,conv_levels)
    # example: direct conv 
    # n   f   h   w [128,512,28,28]
    # n   c   h   w [128,1024,28,28]
    # f   c   kh  kw[512,1024,3,3]


    # out_shape=[128,512,28,28]
    
    # reduction_shape=[1024,3,3]
        
    # out_axis_mapping=[0,1,0,0]
    #     [0],  # 输出轴 'n' 来自 input1 的第 0 个轴，是空间轴
    #     [1],  # 输出轴 'f' 来自 input2 的第 0 个轴，是空间轴
    #     [0],  # 输出轴 'h' 来自 input1 的第 2 个轴，是空间轴
    #     [0]   # 输出轴 'w' 来自 input1 的第 3 个轴，是空间轴
    # ]

    # reduction_axis_mapping=
    # [
    #     [2, 8],  # 规约轴的第一个'c' 是公共归约轴，current_step是8
    #     [1, 3],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
    #     [1, 3],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
    # ]

    DDR_non_ideal_para=1.0 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    spatial_grids = [math.ceil(dim / tb_dim) for dim, tb_dim in zip(out_shape, out_tb_shape)]
    #   n   f   h   w   [128,512,28,28]
    #   tn  tf  th  tw  [1,32,7,28]
    #spatial_grids[128,16,4,1]

    thread_per_tb=np.prod(dim_threads)*reduce_threads_info[1]
    out_thread_shape=[math.ceil(tb_dim / thread_dim) for tb_dim, thread_dim in zip(out_tb_shape, dim_threads)]


    # example:
    #     mem_levels = {
    #     'in1': '[1,1,1]', for ddr
    #     'in2': '[0,1,1]', for smem
    #     'out1': [0,0,1],  for reg
    # }
    

    in1_level = mem_levels['in1']

    out1_level = mem_levels['out1']

    # 初始化输入tensor block的大小
    in1_tb_spatial_shape = []


    in1_tb_reduction_shape = []


    in1_tb_current_step_shape = []


    
    reduction_grids=[]

    reduction_step=[]
    # [8,3,3]

    in1_thread_shape=[]


        # 处理输出轴
    for i, dim_size in enumerate(out_tb_shape):
        input_index = out_axis_mapping[i]  # 确定是input1还是input2
        if input_index == 0:
            in1_tb_spatial_shape.append(dim_size)
            in1_thread_shape.append(out_thread_shape[i])

    
    in1_tb_spatial_index=len(in1_tb_spatial_shape)-1


    

    # 处理归约轴
    for i, dim_size in enumerate(reduction_shape):
        input_index, current_step = reduction_axis_mapping[i]
        reduction_grids.append(math.ceil(dim_size/current_step))
        reduction_step.append(current_step)



        if input_index == 1:  # input2的私有归约轴

            in1_tb_spatial_shape[in1_tb_spatial_index]+=current_step-1 # e.g. kh=3, pad_h+=2
            in1_tb_spatial_index-=1

        
        elif input_index == 0:  # input1的私有归约轴
            in1_tb_reduction_shape.append(dim_size)
            in1_tb_current_step_shape.append(current_step)


    # print(f"Input 1 Tensor Block Size: {in1_tb_spatial_shape}")
    # print(f"Input 2 Tensor Block Size: {in2_tb_spatial_shape}")
    # l2_hit_rate=0.88952535
    # l2_hit_rate=general_reduce_l2_hitrate(out_axis_mapping, out_tb_shape, spatial_grids, in1_tb_spatial_shape, in2_tb_spatial_shape, in1_tb_reduction_shape, in2_tb_reduction_shape,arch.l2_capacity,arch.sm_count,bytes_per_num)
    l2_hit_rate = 0
    # test_code = '''
    # ret = calculate_general_ruduce_resource_utilization(
    #     [n, f, h, w], 
    #     [c, kh, kw], 
    #     out_axis_mapping, 
    #     reduction_axis_mapping, 
    #     [1, 64, 4, 28], 
    #     [1, 32, 1, 4], 
    #     4, 
    #     chip, 
    #     mem_levels, 
    #     compute_at=1
    # )
    # '''
    # elapsed_time = timeit.timeit(test_code, globals=globals(), number=100)

    # # 计算平均时间
    # average_time = elapsed_time / 100
    # print(f"Average execution time: {average_time} seconds")
    # # Average execution time: 0.19484585613012315 seconds

    l2_read_io=np.prod(spatial_grids)*np.prod(reduction_grids)*(
        np.prod(in1_tb_spatial_shape)*np.prod(in1_tb_current_step_shape)*in1_level[0]*in1_level[-1])
    
    l2_store_io=np.prod(spatial_grids)*np.prod(out_tb_shape)*out1_level[0]*out1_level[-1]
    # print(spatial_grids,out_tb_shape)

    # ----------------- related to waves -----------------
    assert hasattr(arch, 'ddr_wave_bytes'), "arch.ddr_wave_bytes does not exist"
    if l2_read_io > 0 and uses_dram_wave_quantization(arch):
        l2_read_waves = l2_read_io / arch.ddr_wave_bytes
        log.info("l2 read waves: %s, original l2 read io: %s", l2_read_waves, l2_read_io)
        l2_read_io = l2_read_io * math.ceil(l2_read_waves) / l2_read_waves

    if l2_store_io > 0 and uses_dram_wave_quantization(arch):
        l2_store_waves = l2_store_io / arch.ddr_wave_bytes
        log.info("l2 store waves: %s, original l2 store io: %s", l2_store_waves, l2_store_io)
        l2_store_io = l2_store_io * math.ceil(l2_store_waves) / l2_store_waves
    # ----------------- related to waves -----------------    

    ddr_io=l2_read_io*(1-l2_hit_rate)+l2_store_io

    ddr_io=ddr_io*DDR_non_ideal_para

    if (arch.core=="A100" or arch.core=="H100" or arch.core=="B200"):
        #a100 with special two-part l2 cache structure
        l2_io=l2_read_io*(l2_hit_rate)+l2_read_io*(1-l2_hit_rate)*2+l2_store_io*2
    else:
        l2_io=l2_read_io+l2_store_io

    # now let's think about the compute stage
    # take the conv as example, local reg could be kept at different stage
    # spatial_grid*reduction_grid -> now it's like: tb_n tb_f tb_h tb_w; tb_n, current_step_c, tb_padh, tb_padw; tb_f current_step_c current_step_kh current_step_kw
    # compute here with no local reg is all data loaded from smem, which is tb_n*tb_f*tb_h*tb_w*(current_step_c*current_step_kh*current_step_kw)*2
    # if with limited local keep, e.g., 128 threads with thread_n, thread_f, thread_h, thread_w,
    # then it's 128 *  thread_n*thread_f*thread_h*thread_w*



    

    smem_footprint=(np.prod(in1_tb_spatial_shape)*np.prod(in1_tb_current_step_shape)*(1-(in1_level[1]-in1_level[0]))*in1_level[-1]+
                    np.prod(out_tb_shape)*(out1_level[1]-out1_level[0])*out1_level[-1])
    # load in1 and in2, ldg
    smem_l1_io=in1_level[-1]*np.prod(spatial_grids)*np.prod(reduction_grids)*(
        np.prod(in1_tb_spatial_shape)*np.prod(in1_tb_current_step_shape)*in1_level[0])
    
    # load in1 and in2, sts
    smem_l1_io+=in1_level[-1]*np.prod(spatial_grids)*np.prod(reduction_grids)*(
        np.prod(in1_tb_spatial_shape)*np.prod(in1_tb_current_step_shape)*in1_level[0])
        
    # 接下来开始考虑一个rstep被切开的事情
    # 唯一的不同是多了reduce_threads_info
    # 这个放在哪里呢？应该放在一个归约轴上
    # 要从rstep某一条轴的最里面塞
    
    # reduce_threads_info=[0,2] 
    # reduce thread at reduce axis 0, reduce thread is 2
    # 对应reduction_shape以及reduction_threads的维度都要除以reduce_threads_info
    # reduction_axis_mapping[reduce_threads_info[0],1]=math.ceil(reduction_axis_mapping[reduce_threads_info[0],1]/reduce_threads_info[1])
    # reduction_shape[reduce_threads_info[0]]=math.ceil(reduction_shape[reduce_threads_info[0]]/reduce_threads_info[1])
    # ----------------------------------------------------------------------------------------------
    reduction_step[reduce_threads_info[0]]=math.ceil(reduction_step[reduce_threads_info[0]]/reduce_threads_info[1])
    # ----------------------------------------------------------------------------------------------

    # load in1 and in2, lds
    if(compute_at==-1): # all data from smem
        reg_footprint=np.prod(out_thread_shape)*out1_level[-1]/4+2+len(reduction_shape)
        smem_l1_io+=in1_level[-1]*np.prod(spatial_grids)*np.prod(reduction_grids)*thread_per_tb*np.prod(out_thread_shape)*np.prod(reduction_step)*(in1_level[1])
        # every output with 2 inputs,like 1*2*7*7* (1+1)
    elif(compute_at==0):# keep at innermost loop，like (thread_n*thread_h*thread_w,thread_f)
        reg_footprint=np.prod(out_thread_shape)*out1_level[-1]/4+np.prod(in1_thread_shape*in1_level[1])*in1_level[-1]/4+len(reduction_shape)+2
        smem_l1_io+=in1_level[-1]*np.prod(spatial_grids)*np.prod(reduction_grids)*thread_per_tb*np.prod(reduction_step)*(
            np.prod(in1_thread_shape)*in1_level[1])
        # every thread tiles with some shape of input1 and 2,like 1*7*7+2
    else:# keep at some level of outer loop 
        assert(len(reduction_step)>=compute_at)
        j=0
        for i in range(compute_at):
            last_row=reduction_axis_mapping[-1]
            reduction_axis_mapping = np.delete(reduction_axis_mapping, -1, axis=0)
            last_step=reduction_step.pop()
            if(last_row[0]==2):
                in1_thread_shape.insert(0,last_step)
            elif(last_row[0]==1):#input2 的归约轴
                in1_thread_shape[-1-j]+=last_step-1
                j+=1
            elif(last_row[0]==0):#input1 的归约轴
                in1_thread_shape.insert(0,last_step)
                j+=1
        reg_footprint=np.prod(out_thread_shape)*out1_level[-1]/4+np.prod(in1_thread_shape*in1_level[1])*in1_level[-1]/4+len(reduction_shape)+2
        smem_l1_io+=in1_level[-1]*np.prod(spatial_grids)*np.prod(reduction_grids)*thread_per_tb*np.prod(reduction_step)*(
            np.prod(in1_thread_shape)*in1_level[1])

    smem_l1_io+=(np.prod(spatial_grids)*thread_per_tb*np.prod(out_thread_shape)*out1_level[1])*out1_level[-1]/reduce_threads_info[1]

    
    # print(out_thread_shape)
    # print(in2_thread_shape)

    # a+b, FADD also uses FMA pipe, which will also occupy one cycle
    
    

    # 在tb的最后，需要进行归约
    # 归约是一个无论如何都单独的stage，前后、中间都插入很多的sync
    # 或许其实可以考虑一下sync的代价？
    if(reduce_threads_info[1]>32):
        smem_footprint+=128*out1_level[-1]*np.prod(out_thread_shape)
        # smem_l1_io+=np.prod(spatial_grids)*bytes_per_num
        current_ruduce=reduce_threads_info[1]
        smem_l1_io+=np.prod(spatial_grids)*thread_per_tb/reduce_threads_info[1]*np.prod(out_thread_shape)*out1_level[-1]*current_ruduce
        
        while(current_ruduce>32):
            smem_l1_io+=np.prod(spatial_grids)*thread_per_tb/reduce_threads_info[1]*np.prod(out_thread_shape)*out1_level[-1]*current_ruduce*3/2
            current_ruduce/=2
        
        # smem_l1_io+=math.log2(32)*np.prod(spatial_grids)*thread_per_tb/reduce_threads_info[1]*np.prod(out_thread_shape)*bytes_per_num*32*3/2
        smem_l1_io+=math.log2(32)*np.prod(spatial_grids)*thread_per_tb/reduce_threads_info[1]*np.prod(out_thread_shape)*out1_level[-1]*32*3

        # load shared and stg.e to global (repeated)
        smem_l1_io+=np.prod(spatial_grids)*reduce_threads_info[1]*np.prod(out_thread_shape)*out1_level[-1]*2
        # print(np.prod(spatial_grids),reduce_threads_info[1],np.prod(out_thread_shape),out1_level[-1],2,thread_per_tb)
        reg_footprint+=math.log2(32)

    else:
        # using shfl & sync 
        reg_footprint+=3
        pass

    overheads=(32 / thread_per_tb if thread_per_tb <= 32 else
          128 / thread_per_tb if thread_per_tb <= 128 else
          256 / thread_per_tb if thread_per_tb <= 256 else
          384 / thread_per_tb if thread_per_tb <= 384 else
          1)
    
    compute_flops=2*np.prod(out_shape)*np.prod(reduction_shape)
    compute_flops+=2*np.prod(out_shape)*math.ceil(math.log2(reduce_threads_info[1]))
    compute_flops*=overheads
    smem_l1_io*=overheads
    reg_footprint=math.ceil(reg_footprint*REG_spill_para)

    ddr_read_io=l2_read_io*(1-l2_hit_rate)

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
