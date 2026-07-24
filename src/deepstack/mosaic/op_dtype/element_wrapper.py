from mosaic.utils import OpBytes, Modeling_Granularity,Tensor_Loc,find_max_power_of_two
from tilesight.arch import Arch
import math
# from tilesight.fused_op_dtype.matmul_fused_op_new_api import calculate_matmul_resource_utilization_new
from tilesight.fused_op_dtype_wave.N_0_element_wise_fused_op_wave import calculate_N_0_elementwise_resource_utilization
from tilesight.fused_op_dtype_wave.N_1_element_wise_fused_op_wave import calculate_N_1_elementwise_resource_utilization
from tilesight.fused_op_dtype_wave.N_N_element_wise_fused_op_wave import calculate_N_N_elementwise_resource_utilization

from tilesight.fusion_support.hete_post_process_single_op import hete_post_process_tensor_core_op,hete_post_process_sfu_core_op,hete_post_process_cuda_core_op

from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
from mosaic.cost.op_perf_stats import OpPerfStats
# tiling_configs
# tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages
from functools import lru_cache
import torch
import numpy as np
import logging

log = logging.getLogger(__name__) 


def get_dim_threads(tb_shape:list):
    shape_len = len(tb_shape)
    dim_threads= [1] * shape_len

    cascade_dim=1

    # assert np.prod(tb_shape) >= 128
    if np.prod(tb_shape) < 128:
        # log.warning("tb_shape %s is too small, not saturating minimal threads >=128 size", tb_shape)
        log.info("tb_shape %s is too small, not saturating minimal threads >=128 size", tb_shape)
        total_threads=np.prod(tb_shape)
    else:
        total_threads=128


    # for i in range(shape_len):

    #     if cascade_dim >= 128:
    #         break

    #     current_dim = find_max_power_of_two(128/cascade_dim, tb_shape[i])
    #     dim_threads[i] = current_dim
    #     cascade_dim *= current_dim
    for i in range(shape_len):

        if cascade_dim >= total_threads:
            break

        current_dim = find_max_power_of_two(total_threads/cascade_dim, tb_shape[i])
        dim_threads[i] = current_dim
        cascade_dim *= current_dim

def get_default_tiling(input1_shape:tuple, element_op_bytes:OpBytes, single_chip:Arch):
    shape_len = len(input1_shape)
    trasaction_bytes = single_chip.ddr_transaction_size
    transaction_size = trasaction_bytes / element_op_bytes.input1.num_bytes

    current_stride = 1
    # initialize tb_shape and dim_threads with all 1, shape_len's 1
    tb_shape = [1] * shape_len
    dim_threads= [1] * shape_len

    # assert np.prod(input1_shape) >= 128

    if np.prod(input1_shape) < 128:
        # log.warning("input1_shape %s is too small, not saturating minimal threads >=128 size", input1_shape)
        log.info("input1_shape %s is too small, not saturating minimal threads >=128 size", input1_shape)
        total_threads=np.prod(input1_shape)
    else:
        total_threads=128

    

    if (np.prod(input1_shape) < transaction_size):
        # log.warning("input1_shape %s is too small, not saturating minimal ddr wave size %s", input1_shape, transaction_size)
        log.info("input1_shape %s is too small, not saturating minimal ddr wave size %s", input1_shape, transaction_size)
        transaction_size = find_max_power_of_two(max_const=transaction_size, N=np.prod(input1_shape))
        # log.warning("set transaction_size to : %s", transaction_size)
        log.info("set transaction_size to : %s", transaction_size)

    # 倒序访问 input_shape 的每个值
    for i in range(shape_len - 1, -1, -1):
        
        if current_stride >= transaction_size:
            break

        current_tile = find_max_power_of_two(max_const=transaction_size/current_stride, N=input1_shape[i])
        current_stride *= current_tile
        tb_shape [i] = current_tile

    
    # now we get tb_shape
    # now we are going to assign this to threads
    # 正序访问 tb_shape 的每个值
    cascade_dim=1
    for i in range(shape_len):

        if cascade_dim >= total_threads:
            break

        current_dim = find_max_power_of_two(total_threads/cascade_dim, tb_shape[i])
        dim_threads[i] = current_dim
        cascade_dim *= current_dim
        # current_tile = tb_shape[i]
        # current_stride *= current_tile
        # dim_threads[i] = current_tile
    
    # return tb_shape, dim_threads

    return tb_shape, dim_threads


def get_transformed_tiling(input1_shape:tuple, element_op_bytes:OpBytes, single_chip:Arch, tb_tiling_config:tuple):
    shape_len = len(input1_shape)
    # trasaction_bytes = single_chip.ddr_transaction_size
    # transaction_size = trasaction_bytes / element_op_bytes.input1.num_bytes
    transaction_size = np.prod(tb_tiling_config)

    current_stride = 1
    # initialize tb_shape and dim_threads with all 1, shape_len's 1
    tb_shape = [1] * shape_len
    dim_threads= [1] * shape_len

    # assert np.prod(tb_tiling_config) >= 128
    if np.prod(tb_tiling_config) < 128:
        # log.warning("tb_tiling_config %s is too small, not saturating minimal threads >=128 size", tb_tiling_config)
        log.info("tb_tiling_config %s is too small, not saturating minimal threads >=128 size", tb_tiling_config)

    # if (np.prod(input1_shape) < transaction_size):
    #     log.warning("input1_shape %s is too small, not saturating minimal ddr wave size %s", input1_shape, transaction_size)
    #     transaction_size = find_max_power_of_two(max_const=transaction_size, N=np.prod(input1_shape))
    #     log.warning("set transaction_size to : %s", transaction_size)

    # 倒序访问 input_shape 的每个值
    for i in range(shape_len - 1, -1, -1):
        
        if current_stride >= transaction_size:
            break

        current_tile = find_max_power_of_two(max_const=transaction_size/current_stride, N=input1_shape[i])
        current_stride *= current_tile
        tb_shape [i] = current_tile

    
    # now we get tb_shape
    # now we are going to assign this to threads
    # 正序访问 tb_shape 的每个值
    cascade_dim=1
    for i in range(shape_len):

        if cascade_dim >= 128:
            break

        current_dim = find_max_power_of_two(128/cascade_dim, tb_shape[i])
        dim_threads[i] = current_dim
        cascade_dim *= current_dim
        # current_tile = tb_shape[i]
        # current_stride *= current_tile
        # dim_threads[i] = current_tile
    
    # return tb_shape, dim_threads

    return tb_shape, dim_threads

def get_auto_tune_tiling(input1_shape:tuple, element_op_bytes:OpBytes, single_chip:Arch):
    pass

# def element_wrapper(op_shape,tb_shape,dim_threads, arch,mem_levels,num_ops=1):


def element_wrapper(element_op_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, batch=1, type="cuda_core", tb_tiling_config=None):
    assert type in ["cuda_core", "sfu_core"]
    # tb_tiling_config make this tuple

    mode=granularity.get_mode()
    tune_flag=granularity.get_auto_tune()

    # input1_shape = element_op_bytes.input1.shape
    
    # input2_shape = element_op_bytes.input2.shape
    # output_shape = element_op_bytes.output.shape
    input1_shape, input2_shape, output_shape = element_op_bytes.get_shapes()
    
    if mode == "coarse":
        if tune_flag==False:
            pass
        elif tune_flag==True:
            # to be honest, I don't think element-wise op needs exhausted auto tuning
            # log.warning("element-wise op does not support / need auto tuning")
            # log.warning("Fall back to use default tiling")
            log.info("element-wise op does not support / need auto tuning")
            log.info("Fall back to use default tiling")

        if tb_tiling_config is None:
            # then we need to generate config ourselves, otherwise we need to use the config from the input (cascaded tiling)
            tb_shape, dim_threads = get_default_tiling(input1_shape, element_op_bytes, single_chip)
            bytes_per_num = element_op_bytes.output.num_bytes

        elif tb_tiling_config is not None:
            tb_tiling_config = tuple(tb_tiling_config)
            # re-arrange the tiling shape (layout transformation)
            # dim_threads = get_dim_threads(tb_tiling_config)
            # tb_shape = tb_tiling_config
            tb_shape, dim_threads = get_transformed_tiling(input1_shape, element_op_bytes, single_chip, tb_tiling_config)
        
        if input2_shape is None:
            ret=calculate_N_0_elementwise_resource_utilization(
                op_shape=output_shape,tb_shape=tb_shape,dim_threads=dim_threads,
                arch=single_chip,mem_levels=element_op_bytes.to_mem_levels(),num_ops=batch)
        elif input2_shape != input1_shape:
            ret=calculate_N_1_elementwise_resource_utilization(
                in1_shape=input1_shape,in2_shape=input2_shape,in1_tb_shape=tb_shape,dim_threads=dim_threads,
                arch=single_chip,mem_levels=element_op_bytes.to_mem_levels(),num_ops=batch)
        elif input2_shape == input1_shape:
            ret=calculate_N_N_elementwise_resource_utilization(
                op_shape=output_shape,tb_shape=tb_shape,dim_threads=dim_threads,
                arch=single_chip,mem_levels=element_op_bytes.to_mem_levels(),num_ops=batch)
        
        if type == "cuda_core":
            hete_post_data=hete_post_process_cuda_core_op(ret, single_chip, element_op_bytes.output.num_bytes)
        elif type == "sfu_core":
            hete_post_data=hete_post_process_sfu_core_op(ret, single_chip)
            
        return hete_post_data, tb_shape

    elif mode == "fine":
        pass
    elif mode == "roof":
        pass
    pass

if __name__ == "__main__":
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=True)
    silu_bytes=OpBytes(
        input1=Tensor_Loc(torch.float16, 'smem',[2, 64, 4096]),
        input2=Tensor_Loc(torch.float16, 'ddr',[2, 64, 4096]),
        output=Tensor_Loc(torch.float16, 'smem',[2, 64, 4096]),
    )

    input1_shape=(2, 64, 4096)
    from mosaic.arch import stacked_gpu_base
    single_chip = stacked_gpu_base()


    tb_shape, dim_threads=get_default_tiling(input1_shape, silu_bytes, single_chip)
    print(tb_shape, dim_threads)
