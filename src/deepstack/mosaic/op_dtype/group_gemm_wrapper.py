from mosaic.utils import OpBytes, Modeling_Granularity, shrink_tiling_by_waves, find_max_power_of_two
from tilesight.arch import Arch
import math
# from tilesight.fused_op_dtype.matmul_fused_op_new_api import calculate_matmul_resource_utilization_new
from tilesight.fused_op_dtype_wave.matmul_fused_op_new_api_wave import calculate_matmul_resource_utilization_new
# from tilesight.fusion_support.post_process_single_op import hyper_post_process_tensor_core_op,hyper_post_process_sfu_core_op,hyper_post_process_cuda_core_op
from tilesight.fusion_support.hete_post_process_single_op import hete_post_process_tensor_core_op,hete_post_process_sfu_core_op,hete_post_process_cuda_core_op
# from tilesight.fusion_support.reg_fusion import reg_fusion
# from tilesight.fusion_support.smem_fusion import smem_fusion
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
from mosaic.cost.op_perf_stats import OpPerfStats
# tiling_configs
# tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages
from functools import lru_cache
import numpy as np
import logging
log = logging.getLogger(__name__) 

def get_modeling_time(M:int,N:int,K:int,gemm_bytes:OpBytes,single_chip:Arch,tiling_config:tuple, batch=1):

    tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, row_panel = tiling_config
    reg_array=[]
    # time_list=[]
    smem_fusion_list=[]
    # matmul_levels={'in1': [1,1,1,2], 'in2': [1,1,1,2],'out1': [1,1,1,4]}
    input_bytes, weight_bytes, output_bytes=gemm_bytes.get_dtype_bytes()
    # print("input_bytes, weight_bytes, output_bytes",input_bytes, weight_bytes, output_bytes)
    matmul_levels=gemm_bytes.to_mem_levels()
    # ret=calculate_matmul_resource_utilization_new([M,N,K],[128,256,64],[64,128,64],5,chip,matmul_levels,row_panel=108,batch=1,mma_type="utcmma_cta1")
    if single_chip.support_utcmma == True:
        mma_type="utcmma_cta2"
    elif single_chip.support_wgmma == True:
        mma_type="wgmma"
    else:
        mma_type="wmma"
    ret=calculate_matmul_resource_utilization_new([M,N,K],[tb_m,tb_n,tb_k],[wp_m,wp_n,wp_k],int(stages),single_chip,matmul_levels,row_panel=row_panel,batch=batch,mma_type=mma_type)
    # hyper_post_data=hyper_post_process_tensor_core_op(ret, single_chip, min(input_bytes,weight_bytes))
    # reg_array.append(hyper_post_data)

    # _, post_data=reg_fusion(reg_array, single_chip)
    # smem_fusion_list.append(post_data)
    # grids=[M/tb_m, N/tb_n]

    # _,smem_fusion_post_data=smem_fusion(smem_fusion_list, grids, single_chip)
    # # posted=[bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util]
    
    # from tilesight.compare_with_ncu.extract_metrcis_from_modeling_data import extract_metrcis_from_modeling_data
            
    # log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
    # log.info("smem_fusion_post_data: %s, tiling_config: %s", extract_metrcis_from_modeling_data(smem_fusion_post_data), tiling_config)
    
    # # return smem_fusion_post_data,tiling_config
    # return hyper_post_data, smem_fusion_post_data,tiling_config
    # hete_post_data=hete_post_process_tensor_core_op(ret, single_chip, min(input_bytes,weight_bytes))
    hete_post_data=hete_post_process_tensor_core_op(ret, single_chip, weight_bytes)
    reg_array.append(hete_post_data)
    smem_fusion_list.append(hete_reg_fusion(reg_array, single_chip))
    grids=[M/tb_m, N/tb_n, batch]
    smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
    log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
    log.info("hete modeling metrics: %s, tiling_config: %s", get_hete_metrics(smem_fusion_post_data), tiling_config)

    return hete_post_data,smem_fusion_post_data,tiling_config



def group_gemm_wrapper(shape_list:list, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, batch=1, stats:"OpPerfStats | None" = None):
    '''
    shape_list: list of tuples, each tuple includes: bs, M, N, K
    '''

    # each shape includes: bs, M, N, K
    # assert mode in ["coarse", "fine","roof"]
    
    mode=granularity.get_mode()
    tune_flag=granularity.get_auto_tune()
    if mode == "coarse":

        smem_fusion_list=[]
        
        grid=0

        for shape in shape_list:
            batch, M, N, K = shape
            # log.info("shape: %s", shape)
            log.info("M: %s, N: %s, K: %s, batch: %s", M, N, K, batch)
            reg_array=[]
            hete_post_data,smem_fusion_post_data, tiling_config = gemm_wrapper(M=M, N=N, K=K, gemm_bytes=gemm_bytes, granularity=granularity, single_chip=single_chip, batch=batch)

            reg_array.append(hete_post_data)
            smem_fusion_list.append(hete_reg_fusion(reg_array, single_chip))

            grid+=M/tiling_config[0]*N/tiling_config[1]*batch
        
        # hack and consrtuct the number of threadblocks of the whole group gemm
        grids=[grid]
        smem_fusion_post_data=hete_smem_fusion(smem_fusion_list, grids, single_chip)
        log.info("grids: %s, waves: %s", grids, np.prod(grids)/single_chip.sm_count)
        log.info("hete modeling metrics: %s", get_hete_metrics(smem_fusion_post_data))

        if stats is not None:
            stats.append_hete(smem_fusion_post_data)
        return smem_fusion_list, smem_fusion_post_data, grids

            # each gemm_wrapper will return this:
            # hete_post_data,smem_fusion_post_data,best_config
        

        
    elif mode == "fine":
        # schedule_list=GEMMSchedule(M, N, K, chip)
        pass
    elif mode == "roof":
        pass

