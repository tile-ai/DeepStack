from ..util import *
from tilesight.arch import uses_dram_wave_quantization, uses_gpu_resource_model
from tilesight.model_version import get_model_version
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, stage_num, block_per_sm
import numpy as np
import math
import os
import logging
log = logging.getLogger(__name__) 

def calculate_matmul_resource_utilization_new(op_shape,tb_shape,wp_shape,stage_num,arch,mem_levels,row_panel=1,batch=1,mma_type="wmma"):
# def calculate_matmul_resource_utilization_new(op_shape,cluster_shape,tb_shape,wp_shape,stage_num,arch,mem_levels,row_panel=1,batch=1,mma_type="utcmma_cta1"):
    # mma_type:wmma, wgmma, utcmma_cta1, utcmma_cta2
    m,n,k=op_shape
    # assuming sharing data intra-cluster
    # cluster_m,cluster_n,cluster_k=cluster_shape
    tb_m,tb_n,tb_k=tb_shape
    wp_m,wp_n,wp_k=wp_shape

    DDR_non_ideal_para=1.0 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    
    l2_hit_rate=L2_hit_rate_flow_sim_reuse_distance(m, n, k, tb_m, tb_n, tb_k, arch.l2_capacity, arch.sm_count, mem_levels, row_panel)
    # l2_hit_rate=0.5959
    gridM=math.ceil(m/tb_m)
    gridN=math.ceil(n/tb_n)
    gridK=math.ceil(k/tb_k)
    # gridM=m/tb_m
    # gridN=n/tb_n
    # gridK=k/tb_k
    # example:
    #     mem_levels = {
    #     'in1': '[1,1,1,2]', for ddr, 2byte
    #     'in2': '[0,1,1,2]', for smem, 2byte
    #     'out1': [0,0,1,4],  for reg, 4byte
    # }

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    useful_compute_flops=m*n*k*2

    # get_tensor_core_minimum_ptx
    # assert arch has this function arch.ge 
    assert hasattr(arch, 'get_tensor_core_minimum_ptx'), "arch.get_tensor_core_minimum_ptx does not exist"
    minimum_tc_ptx_shape = arch.get_tensor_core_minimum_ptx(bytes = in1_level[-1])
    padding_scope = os.environ.get(
        "TILESIGHT_TENSOR_PADDING_SCOPE", "collective"
    ).strip().lower()
    if padding_scope == "legacy":
        # Exact submitted-artifact arithmetic.  This intentionally retains the
        # historical min(1, ...) behavior for paper_legacy replay only.
        compute_overheads = min(
            1,
            math.ceil(wp_m) / minimum_tc_ptx_shape[0]
            * math.ceil(wp_n) / minimum_tc_ptx_shape[1]
            * math.ceil(wp_k) / minimum_tc_ptx_shape[2],
        )
    else:
        if padding_scope == "warp":
            execution_shape = (wp_m, wp_n, wp_k)
        elif padding_scope == "collective":
            execution_shape = (
                (tb_m, tb_n, tb_k)
                if mma_type in {"wgmma", "utcmma_cta1", "utcmma_cta2"}
                else (wp_m, wp_n, wp_k)
            )
        else:
            raise ValueError(
                "TILESIGHT_TENSOR_PADDING_SCOPE must be 'legacy', 'warp', "
                f"or 'collective', got {padding_scope!r}"
            )
        padded_execution_shape = tuple(
            math.ceil(extent / instruction_extent) * instruction_extent
            for extent, instruction_extent in zip(
                execution_shape, minimum_tc_ptx_shape
            )
        )
        compute_overheads = (
            np.prod(padded_execution_shape) / np.prod(execution_shape)
        )
    if compute_overheads > 1:
        # log.info(" warp shape wp, wn ,wk, arch name, arch need minimum ptx shape, compute_overheads: %s", (wp_m, wp_n, wp_k), arch.name, minimum_tc_ptx_shape, compute_overheads))
        # log.info(" warp shape wp, wn ,wk, arch name, arch need minimum ptx shape, compute_overheads: %s", (wp_m, wp_n, wp_k), arch.core, minimum_tc_ptx_shape, compute_overheads)
        log.debug(
            "tensor padding scope=%s warp=%s arch=%s minimum=%s overhead=%s",
            padding_scope, (wp_m, wp_n, wp_k), arch.core,
            minimum_tc_ptx_shape, compute_overheads,
        )

    compute_flops = useful_compute_flops * compute_overheads
    padding_compute_flops = max(0.0, compute_flops - useful_compute_flops)



    # print(in1_level)
    # print(in2_level)
    # print(out1_level)

    # if (arch.core=="B200" and mma_type=="utcmma_cta2"):
    if (mma_type=="utcmma_cta2"):
        l2_hit_rate=L2_hit_rate_flow_sim_reuse_distance(m, n, k, tb_m*2, tb_n, tb_k, arch.l2_capacity, arch.sm_count//2, mem_levels, max(row_panel//2, 1))
        l2_read_io=gridM/2*gridN*k*(tb_m*2*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1])
        # l2_store_io=gridM/2*gridN*(tb_m*2*tb_n)*out1_level[-1]*out1_level[0]
        l2_store_io=m*n*4
        # print("act    io:",gridM*gridN*k*tb_m*in1_level[0]*in1_level[-1])
        # print("weight io:",gridM*gridN*k*tb_n*in2_level[0]*in2_level[-1])
        # print("output io:",gridM*gridN*(tb_m*tb_n)*out1_level[-1]*out1_level[0])

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
    
        active_warp_per_tb=(tb_m/wp_m)*(tb_n/wp_n)

    else:
        l2_read_io=gridM*gridN*k*(tb_m*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1])
        l2_store_io=gridM*gridN*(tb_m*tb_n)*out1_level[-1]*out1_level[0]
        # print("act    io:",gridM*gridN*k*tb_m*in1_level[0]*in1_level[-1])
        # print("weight io:",gridM*gridN*k*tb_n*in2_level[0]*in2_level[-1])
        # print("output io:",gridM*gridN*(tb_m*tb_n)*out1_level[-1]*out1_level[0])

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

        active_warp_per_tb=(tb_m/wp_m)*(tb_n/wp_n)


    

    if (arch.core=="A100" or arch.core=="H100" or arch.core=="B200" or arch.core=="A100_LUT"):
        #a100 with special two-part l2 cache structure
        l2_io=l2_read_io*(l2_hit_rate)+l2_read_io*(1-l2_hit_rate)*2+l2_store_io*2
        # print ("l2_read_io/GiB",l2_read_io/(1024**3))
        # print ("l2_hit_rate",l2_hit_rate)
        # print ("l2_store_io/GiB",l2_store_io/(1024**3))
        # print ("l2_io/GiB",l2_io/(1024**3))
    else:
        l2_io=l2_read_io+l2_store_io

    if get_model_version() == "paper_legacy":
        if (arch.core=="B200" and mma_type=="utcmma_cta2"):
            smem_footprint=(tb_m*tb_k*(1-(in1_level[1]-in1_level[0]))*in1_level[-1]+
                        tb_n/2*tb_k*(1-(in2_level[1]-in2_level[0]))*in2_level[-1]+
                        tb_m*tb_n*(out1_level[1]-out1_level[0])*out1_level[-1])*stage_num
        else:
            smem_footprint=(tb_m*tb_k*(1-(in1_level[1]-in1_level[0]))*in1_level[-1]+
                        tb_n*tb_k*(1-(in2_level[1]-in2_level[0]))*in2_level[-1]+
                        tb_m*tb_n*(out1_level[1]-out1_level[0])*out1_level[-1])*stage_num
    else:
        # DDR inputs have one buffer per pipeline stage.  Fused SMEM inputs
        # and outputs remain resident once; register inputs occupy no SMEM.
        def input_smem_copies(level):
            if level[0] == 1:
                return stage_num
            if level[1] == 1:
                return 1
            return 0

        in2_tb_n = tb_n/2 if (arch.core=="B200" and mma_type=="utcmma_cta2") else tb_n
        smem_footprint=(
            tb_m*tb_k*in1_level[-1]*input_smem_copies(in1_level)
            + in2_tb_n*tb_k*in2_level[-1]*input_smem_copies(in2_level)
            + tb_m*tb_n*(out1_level[1]-out1_level[0])*out1_level[-1]
        )
    
    # print("arch.core",arch.core)

    smem_io=0
    
    if (arch.core=="B200" and (mma_type=="utcmma_cta1" or mma_type=="utcmma_cta2")):
        # reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / out1_level[-1])) + math.ceil(wp_m * wp_k / 32 / (4 / in1_level[-1]))*in1_level[1] + math.ceil(wp_n * wp_k / 32 / (4 / in2_level[-1])))*in2_level[1]
        tmem_footprint = math.ceil(tb_m * tb_n * out1_level[-1])
        reg_footprint = math.ceil(tb_m * tb_n * out1_level[-1] / 4 ) /256
        
        # tmem_footprint 
    elif uses_gpu_resource_model(arch, include_lut=True):
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / out1_level[-1])) + math.ceil(wp_m * wp_k / 32 / (4 / in1_level[-1]))*in1_level[1] + math.ceil(wp_n * wp_k / 32 / (4 / in2_level[-1])))*in2_level[1]
        # reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / 2)) + math.ceil(wp_m * wp_k / 32 / (4 / in1_level[-1]))*in1_level[1] + math.ceil(wp_n * wp_k / 32 / (4 / in2_level[-1])))*in2_level[1]
        reg_footprint=reg_footprint*REG_spill_para
        reg_footprint=math.ceil(reg_footprint)
    else:#with dup
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / out1_level[-1])) + 1*math.ceil(wp_m * wp_k / 32 / (4 / in1_level[-1]))*in1_level[1] + 1*math.ceil(wp_n * wp_k / 32 / (4 / in2_level[-1])))*in2_level[1]
        reg_footprint=reg_footprint*REG_spill_para
        reg_footprint=math.ceil(reg_footprint)

    # if (arch.core=="B200" and mma_type=="utcmma_cta1") or (arch.core == "H100" and mma_type == "wgmma"):
    if (mma_type == "wgmma" or mma_type == "utcmma_cta1"):   
        smem_io += (gridM*gridN*k*(tb_m*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1])) # l2 -> smem
        smem_io += (gridM*gridN*k*(tb_m*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1])) # smem -> reg/tcgen05

        l1_io=(gridM*gridN*(tb_m*tb_n)*out1_level[-1])*out1_level[1]
        smem_l1_io=l1_io+smem_io

    elif (mma_type=="utcmma_cta2"):
        # in this case, cta2 means m is additionally grouped with another thread block
        # print("trigger utcmma_cta2")
        
        smem_io += (gridM/2*gridN*k*(tb_m*2*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1])) # l2 -> smem
        smem_io += (gridM/2*gridN*k*(tb_m*2*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1])) # smem -> reg/tcgen05

        l1_io=(gridM*gridN*(tb_m*tb_n)*out1_level[-1])*out1_level[1]
        smem_l1_io=l1_io+smem_io

    elif (arch.core=="V100"):#noldgsts
        #load global
        l1_io=gridM*gridN*k*(tb_m*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1])
        #store shared
        smem_io=gridM*gridN*k*(tb_m*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1])
        #shared load
        smem_io+=(gridM*gridN*k*(tb_m*in1_level[-1]+tb_n*in2_level[-1]))*active_warp_per_tb*(wp_m*in1_level[1]*in1_level[-1]+wp_n*in2_level[1]*in2_level[-1])/(tb_m*in1_level[-1]+tb_n*in2_level[-1])
        #epilogue, shared store
        # -----------------------
        # smem_io+=l2_store_io
        #epilogue, shared load
        # -----------------------
        # smem_io+=l2_store_io
        #store global
        l1_io=(gridM*gridN*(tb_m*tb_n)*out1_level[-1])*out1_level[1]
        #add together
        smem_l1_io=l1_io+smem_io

    # elif (arch.core=="A100" or arch.core=="H100"):
    elif ((arch.core=="A100" or arch.core=="H100") and mma_type=="wmma"):
        #somehow 0.5 or 0.25 * ldgsts
        smem_io=0.5*(gridM*gridN*k*(tb_m*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1]))  #derived in1&in2 level[0]
        #shared load
        smem_io+=(gridM*gridN*k*(tb_m*in1_level[-1]+tb_n*in2_level[-1]))*active_warp_per_tb*(wp_m*in1_level[1]*in1_level[-1]+wp_n*in2_level[1]*in2_level[-1])/(tb_m*in1_level[-1]+tb_n*in2_level[-1])
        #epilogue, shared store
        # maybe no epilogue? by cutlass with epilogue, but by welder no epilogue 
        # -----------------------
        # smem_io+=l2_store_io
        #epilogue, shared load
        # -----------------------
        # smem_io+=l2_store_io
        #store global
        l1_io=(gridM*gridN*(tb_m*tb_n)*out1_level[-1])*out1_level[1]
        #add together
        smem_l1_io=l1_io+smem_io        
    elif (mma_type=="wmma"):
        #1* ldgsts
        smem_io=(gridM*gridN*k*(tb_m*in1_level[0]*in1_level[-1]+tb_n*in2_level[0]*in2_level[-1]))  #derived in1&in2 level[0]
        #shared load
        smem_io+=(gridM*gridN*k*(tb_m*in1_level[-1]+tb_n*in2_level[-1]))*active_warp_per_tb*(wp_m*in1_level[1]*in1_level[-1]+wp_n*in2_level[1]*in2_level[-1])/(tb_m*in1_level[-1]+tb_n*in2_level[-1])
        #epilogue, shared store
        # maybe no epilogue? by cutlass with epilogue, but by welder no epilogue 
        # -----------------------
        # smem_io+=l2_store_io
        #epilogue, shared load
        # -----------------------
        # smem_io+=l2_store_io
        #store global
        l1_io=(gridM*gridN*(tb_m*tb_n)*out1_level[-1])*out1_level[1]
        #add together
        smem_l1_io=l1_io+smem_io
    
    ddr_read_io=l2_read_io*(1-l2_hit_rate)



    


    return (
        ddr_io*batch,
        l2_hit_rate,
        l2_io*batch,
        smem_footprint,
        smem_l1_io*batch,
        reg_footprint,
        compute_flops*batch,
        ddr_read_io*batch,
        l2_read_io*batch,
        useful_compute_flops*batch,
        padding_compute_flops*batch,
    )
        
