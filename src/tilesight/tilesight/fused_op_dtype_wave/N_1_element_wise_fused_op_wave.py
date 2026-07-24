# 文件名: N_1_element_wise_op.py
import numpy as np
import math
import logging
from tilesight.arch import uses_dram_wave_quantization

log = logging.getLogger(__name__) 

def calculate_N_1_elementwise_resource_utilization(in1_shape,in2_shape,in1_tb_shape,dim_threads, arch,mem_levels,num_ops=1):
    # ret=calculate_N_N_elementwise_resource_utilization([20480,20480],[10,160],[10,10], 4,chip,conv_levels)

    # m,n=in1_shape
    # tb_m,tb_n=in1_tb_shape
    # m_thread,n_thread=dim_threads

    DDR_non_ideal_para=1.0 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    thread_per_tb=np.prod(dim_threads)
    in1_thread_shape=[math.ceil(tb_dim / thread_dim) for tb_dim, thread_dim in zip(in1_tb_shape, dim_threads)]
    # thread_m=math.ceil(tb_m/m_thread)
    # thread_n=math.ceil(tb_n/n_thread)
    # in2_tb_shape=[math.ceil(float(in2_dim)/float(in1_dim)*float(tb_dim)) for in1_dim, in2_dim,tb_dim in zip(in1_shape, in2_shape,in1_tb_shape)]
    in2_tb_shape=[math.ceil(float(in2_dim/in1_dim*tb_dim)) for in1_dim, in2_dim,tb_dim in zip(in1_shape, in2_shape,in1_tb_shape)]
    in2_thread_shape=[math.ceil(float(in2_dim/in1_dim*tb_dim)) for in1_dim, in2_dim,tb_dim in zip(in1_tb_shape, in2_tb_shape,in1_thread_shape)]

    # print(in2_tb_shape)
    # print(in2_thread_shape)

    
    grids = [math.ceil(dim / tb_dim) for dim, tb_dim in zip(in1_shape, in1_tb_shape)]
    # example:
    #     mem_levels = {
    #     'in1': '[1,1,1]', for ddr
    #     'in2': '[0,1,1]', for smem
    #     'out1': [0,0,1],  for reg
    # }

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']
    
    l2_read_io=(np.prod(in1_shape)*in1_level[0]*in1_level[-1]+np.prod(grids)*np.prod(in2_tb_shape)*in2_level[0]*in2_level[-1])
    l2_store_io=np.prod(in1_shape)*out1_level[-1]*out1_level[0]
    # print("in2_tb_shape",in2_tb_shape)
    # print("grids",grids)
    # print("in2_level",in2_level)
    # print("l2_read_io",l2_read_io)
    ddr_read_io=(np.prod(in1_shape)*in1_level[0]*in1_level[-1]+np.prod(in2_shape)*in2_level[0]*in2_level[-1])
    ddr_store_io=np.prod(in1_shape)*out1_level[-1]*out1_level[0]


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

    if ddr_read_io > 0 and uses_dram_wave_quantization(arch):
        ddr_read_waves = ddr_read_io / arch.ddr_wave_bytes
        log.info("ddr read waves: %s, original ddr read io: %s", ddr_read_waves, ddr_read_io)
        ddr_read_io = ddr_read_io * math.ceil(ddr_read_waves) / ddr_read_waves

    if ddr_store_io > 0 and uses_dram_wave_quantization(arch):
        ddr_store_waves = ddr_store_io / arch.ddr_wave_bytes
        log.info("ddr store waves: %s, original ddr store io: %s", ddr_store_waves, ddr_store_io)
        ddr_store_io = ddr_store_io * math.ceil(ddr_store_waves) / ddr_store_waves

    # ----------------- related to waves -----------------

    l2_hit_rate=max((l2_read_io-ddr_read_io)/max(l2_read_io,1),0)

    ddr_io=ddr_read_io+ddr_store_io
    ddr_io=ddr_io*DDR_non_ideal_para

    if (arch.core=="A100" or arch.core=="H100" or arch.core=="B200"):
        #a100 with special two-part l2 cache structure
        l2_io=l2_read_io*(l2_hit_rate)+l2_read_io*(1-l2_hit_rate)*2+l2_store_io*2
    else:
        l2_io=l2_read_io+l2_store_io

    smem_footprint=(np.prod(in2_tb_shape)*(1-(in2_level[1]-in2_level[0]))*in2_level[-1]+
                           np.prod(in1_tb_shape)*(out1_level[1]-out1_level[0])*out1_level[-1])
    # load in1
    smem_l1_io=(np.prod(in1_shape)*in1_level[1])*in1_level[-1]
    # load in2 , ldg + sts
    smem_l1_io+=(np.prod(grids)*np.prod(in2_tb_shape)*in2_level[0])*in2_level[-1]*2
    # load in2 , lds
    smem_l1_io+=(np.prod(grids)*thread_per_tb*np.prod(in2_thread_shape)*in2_level[1])*in2_level[-1]
    # smem_l1_io=(np.prod(in1_shape)*in1_level[1]+np.prod(in1_shape)*in2_level[1])*bytes_per_num
    smem_l1_io+=(np.prod(in1_shape)*out1_level[1])*out1_level[-1]

    # 3 means 2 input + 1 ouput 
    reg_footprint=math.ceil((np.prod(in1_thread_shape)*(in1_level[1]*in1_level[-1]/4+1*out1_level[-1]/4)+np.prod(in2_thread_shape)*in2_level[1]*in2_level[-1]/4)*REG_spill_para)
    # print(in1_thread_shape)
    # print(in2_thread_shape)
    overheads=(32 / thread_per_tb if thread_per_tb <= 32 else
          128 / thread_per_tb if thread_per_tb <= 128 else
          256 / thread_per_tb if thread_per_tb <= 256 else
          384 / thread_per_tb if thread_per_tb <= 384 else
          1)
    # a+b, FADD also uses FMA pipe, which will also occupy one cyle
    compute_flops=2*np.prod(in1_shape)*overheads*num_ops
    smem_l1_io=smem_l1_io*overheads

    # print("ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io")
    # print(ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io)

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
