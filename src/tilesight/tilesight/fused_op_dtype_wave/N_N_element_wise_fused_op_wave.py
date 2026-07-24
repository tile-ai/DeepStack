# 文件名: N_N_element_wise_op.py
import numpy as np
import math
import logging
from tilesight.arch import uses_dram_wave_quantization

log = logging.getLogger(__name__) 

def calculate_N_N_elementwise_resource_utilization(op_shape,tb_shape,dim_threads, arch,mem_levels,num_ops=1):
    # ret=calculate_N_N_elementwise_resource_utilization([20480,20480],[10,160],[10,10], 4,chip,conv_levels)

    # m,n=op_shape
    # tb_m,tb_n=tb_shape
    # m_thread,n_thread=dim_threads

    DDR_non_ideal_para=1.0 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    thread_per_tb=np.prod(dim_threads)
    thread_tiles=[math.ceil(tb_dim / thread_dim) for tb_dim, thread_dim in zip(tb_shape, dim_threads)]
    # thread_m=math.ceil(tb_m/m_thread)
    # thread_n=math.ceil(tb_n/n_thread)

    l2_hit_rate=0
    grids = [math.ceil(dim / tb_dim) for dim, tb_dim in zip(op_shape, tb_shape)]
    # example:
    #     mem_levels = {
    #     'in1': '[1,1,1]', for ddr
    #     'in2': '[0,1,1]', for smem
    #     'out1': [0,0,1],  for reg
    # }

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    l2_read_io=(np.prod(op_shape)*in1_level[0]*in1_level[-1]+np.prod(op_shape)*in2_level[0]*in2_level[-1])
    l2_store_io=np.prod(op_shape)*out1_level[-1]*out1_level[0]

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

    smem_footprint=np.prod(tb_shape)*(out1_level[1]-out1_level[0])*out1_level[-1]

    smem_l1_io=(np.prod(op_shape)*in1_level[1]*in1_level[-1]+np.prod(op_shape)*in2_level[1]*in2_level[-1])
    smem_l1_io+=(np.prod(op_shape)*out1_level[1])*out1_level[-1]

    # 3 means 2 input + 1 ouput 
    reg_footprint=math.ceil(np.prod(thread_tiles)*(1*out1_level[-1]/4+in1_level[1]*in1_level[-1]/4+in2_level[1]*in2_level[-1]/4)*REG_spill_para)

    overheads=(32 / thread_per_tb if thread_per_tb <= 32 else
          128 / thread_per_tb if thread_per_tb <= 128 else
          256 / thread_per_tb if thread_per_tb <= 256 else
          384 / thread_per_tb if thread_per_tb <= 384 else
          1)
    # a+b, FADD also uses FMA pipe, which will also occupy one cyle
    compute_flops=2*np.prod(op_shape)*overheads*num_ops
    smem_l1_io=smem_l1_io*overheads

    ddr_read_io=l2_read_io*(1-l2_hit_rate)

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
