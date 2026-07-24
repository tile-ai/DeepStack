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
from mosaic.cost.op_perf_stats import OpPerfStats
# tiling_configs
# tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages
from functools import lru_cache
import numpy as np
import logging
log = logging.getLogger(__name__) 

@lru_cache(maxsize=1)
def get_tiling_configs_max():
    tb_m_values = [1, 4, 16, 32, 64, 128, 256]
    tb_n_values = [1, 4, 16, 32, 64, 128, 256]
    tb_k_values = [16, 32, 64]
    stages_values = [1, 2, 3, 4]

    configs = []
    for tb_m in tb_m_values:
        for tb_n in tb_n_values:
            for tb_k in tb_k_values:
                wp_m = tb_m // 2 if tb_m != 1 else 1
                wp_n = tb_n // 2 if tb_n != 1 else 1
                wp_k = tb_k
                for stages in stages_values:
                    configs.append((tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages))
    return configs

@lru_cache(maxsize=1)
def get_tiling_configs_medium():
    tb_m_values = [1, 16, 32, 64, 128, 256]
    tb_n_values = [1, 16, 32, 64, 128, 256]
    tb_k_values = [32]
    stages_values = [1, 2, 3]

    configs = []
    for tb_m in tb_m_values:
        for tb_n in tb_n_values:
            for tb_k in tb_k_values:
                wp_m = tb_m // 2 if tb_m != 1 else 1
                wp_n = tb_n // 2 if tb_n != 1 else 1
                wp_k = tb_k
                for stages in stages_values:
                    configs.append((tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages))
    return configs




def find_best_l2_swizzle_row_panel(M:int, N:int, tb_m:int, tb_n:int, gemm_bytes:OpBytes, single_chip:Arch):
    grid_m=math.ceil(M/tb_m)
    grid_n=math.ceil(N/tb_n)
    input_bytes, weight_bytes, output_bytes=gemm_bytes.get_dtype_bytes()

    input_tile_bytes=input_bytes*tb_m
    weight_tile_bytes=weight_bytes*tb_n
    # best swizzle pattern means making the whole wave's tile got more overlap, which means square-like pattern

    # m_wave * n_wave = sm_count
    # m_wave * input_tile_bytes ~= n_wave * weight_tile_bytes
    # to solve this, m_wave = sm_count/n_wave
    # -> sm_count/n_wave * input_tile_bytes ~= n_wave * weight_tile_bytes
    # -> n_wave ~= sqrt(sm_count * input_tile_bytes / weight_tile_bytes)

    # row_panel means n_wave
    best_row_panel=math.sqrt(single_chip.sm_count * input_tile_bytes / weight_tile_bytes)
    # 四舍五入并确保至少为 1
    best_row_panel=max(1, int(round(best_row_panel)))

    # 不超过 grid_n
    row_panel = min(best_row_panel, grid_n)

    return row_panel



def get_default_tiling(M:int, N:int, K:int, gemm_bytes:OpBytes, single_chip:Arch):
    
    tb_m = find_max_power_of_two(256, M)
    tb_n = find_max_power_of_two(256, N)
    tb_k = find_max_power_of_two(32, K)

    # also need to make sure waves saturate the whole gpu
    grids = [math.ceil(M/tb_m), math.ceil(N/tb_n)]
    waves = np.prod(grids)/single_chip.sm_count
    
    
    tb_m, tb_n = shrink_tiling_by_waves(waves, [tb_m, tb_n])

    # wp_m=tb_m//2
    # wp_n=tb_n//2
    wp_m=math.ceil(tb_m/2)
    wp_n=math.ceil(tb_n/2)
    wp_k=tb_k

    input_bytes, weight_bytes, output_bytes=gemm_bytes.get_dtype_bytes()

    stages= min (3, single_chip.configurable_smem_capacity/(input_bytes*tb_m*tb_k + weight_bytes*tb_n*tb_k))
    row_panel=find_best_l2_swizzle_row_panel(M, N, tb_m, tb_n, gemm_bytes, single_chip)

    return tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, row_panel

def get_auto_tune_tiling(M:int, N:int, K:int, gemm_bytes:OpBytes, single_chip:Arch):


    max_tb_m, max_tb_n, max_tb_k, max_wp_m, max_wp_n, max_wp_k, max_stages, max_row_panel=get_default_tiling(M, N, K, gemm_bytes, single_chip)
    # extend?
    all_configs=get_tiling_configs_medium()
    filtered_configs=[]
    for config in all_configs:
        tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages=config

        if tb_m <= max_tb_m and tb_n <= max_tb_n and stages <= max_stages:
            row_panel=find_best_l2_swizzle_row_panel(M, N, tb_m, tb_n, gemm_bytes, single_chip)
            filtered_configs.append((tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, row_panel))
    
    return filtered_configs

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



def gemm_wrapper(M:int, N:int, K:int, gemm_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, batch=1, tiling_config=None):

    # 如果M,N是奇数,则将其调整为偶数
    if M % 2 != 0:
        M = M + 1
    if N % 2 != 0:
        N = N + 1

    # assert mode in ["coarse", "fine","roof"]
    
    mode=granularity.get_mode()
    tune_flag=granularity.get_auto_tune()
    if mode == "coarse":

        if tiling_config is not None:
            tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, row_panel = tiling_config
            hete_post_data,smem_fusion_post_data,tiling_config=get_modeling_time(M,N,K,gemm_bytes,single_chip,tiling_config,batch)
            return hete_post_data,smem_fusion_post_data,tiling_config

        elif tune_flag==False:
            tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, row_panel = get_default_tiling(M, N, K, gemm_bytes, single_chip)
            
            tiling_config =(tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, row_panel)

            hete_post_data,smem_fusion_post_data,tiling_config=get_modeling_time(M,N,K,gemm_bytes,single_chip,tiling_config,batch)
            # posted=[bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util]
            
            # hete_post is not processed
            # smem_fusion_post is wave-processed
            # tiling config is for debugging
            return hete_post_data,smem_fusion_post_data,tiling_config
            

        else:
            filtered_configs = get_auto_tune_tiling(M, N, K, gemm_bytes, single_chip)
            min_time = float('inf')
            best_config = None
            best_smem_fusion_post_data = None
            for config in filtered_configs:
                hete_post_data,smem_fusion_post_data,tiling_config=get_modeling_time(M,N,K,gemm_bytes,single_chip,config,batch)
                time=smem_fusion_post_data[0]
                if time < min_time:
                    min_time = time
                    best_config = config
                    best_smem_fusion_post_data = smem_fusion_post_data
            if best_config is None:
                # 兜底使用默认配置
                # tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, row_panel = get_default_tiling(M, N, K, gemm_bytes, single_chip)
                # default_config = (tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, row_panel)
                # best_smem_fusion_post_data, best_config = get_modeling_time(M,N,K,gemm_bytes,single_chip,default_config)
                raise ValueError("best_config is None")
            
            return hete_post_data,smem_fusion_post_data,best_config

        
    elif mode == "fine":
        # schedule_list=GEMMSchedule(M, N, K, chip)
        pass
    elif mode == "roof":
        pass

