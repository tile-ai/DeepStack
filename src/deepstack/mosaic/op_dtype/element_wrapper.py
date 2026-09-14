from mosaic.utils import OpBytes, Modeling_Granularity,Tensor_Loc,find_max_power_of_two
from tilesight.arch import Arch
import math
import os
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
    return dim_threads

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

    for i in range(shape_len - 1, -1, -1):
        
        if current_stride >= transaction_size:
            break

        current_tile = find_max_power_of_two(max_const=transaction_size/current_stride, N=input1_shape[i])
        current_stride *= current_tile
        tb_shape [i] = current_tile

    
    # now we get tb_shape
    # now we are going to assign this to threads
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


def _get_transformed_tiling_legacy(input1_shape:tuple, element_op_bytes:OpBytes, single_chip:Arch, tb_tiling_config:tuple):
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

    for i in range(shape_len - 1, -1, -1):
        
        if current_stride >= transaction_size:
            break

        current_tile = find_max_power_of_two(max_const=transaction_size/current_stride, N=input1_shape[i])
        current_stride *= current_tile
        tb_shape [i] = current_tile

    
    # now we get tb_shape
    # now we are going to assign this to threads
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


def _map_flat_tile_extent(extent:int, shape:tuple, axes:tuple, tb_shape:list):
    """Map a flattened producer extent onto exact consumer axes.

    An outer physical tile may exceed the logical shape after padding; those
    lanes are predicated by the real kernel and must remain in the tile cost.
    """
    remaining = extent
    for axis in reversed(axes):
        axis_size = shape[axis]
        if remaining <= axis_size:
            tb_shape[axis] = remaining
            remaining = 1
            break
        if remaining % axis_size:
            raise ValueError(
                f"flattened tile extent {extent} is not rectangular over "
                f"axes {axes} of shape {shape}"
            )
        tb_shape[axis] = axis_size
        remaining //= axis_size
    if remaining != 1:
        tb_shape[axes[0]] *= remaining


def get_transformed_tiling(
    input1_shape:tuple,
    element_op_bytes:OpBytes,
    single_chip:Arch,
    tb_tiling_config:tuple,
    tb_axis_groups=None,
):
    if os.environ.get("TILESIGHT_TENSOR_PADDING_SCOPE") == "legacy":
        return _get_transformed_tiling_legacy(
            input1_shape, element_op_bytes, single_chip, tb_tiling_config
        )
    if input1_shape is None:
        raise ValueError("input1 shape is required for a transformed fused tile")
    if not tb_tiling_config:
        raise ValueError("tb_tiling_config must not be empty")
    if any(value <= 0 or int(value) != value for value in tb_tiling_config):
        raise ValueError(
            f"fused producer tile values must be positive integers, got {tb_tiling_config}"
        )
    shape = tuple(int(value) for value in input1_shape)
    if any(value <= 0 for value in shape):
        raise ValueError(f"consumer shape values must be positive, got {shape}")
    config = tuple(int(value) for value in tb_tiling_config)
    shape_len = len(shape)

    if len(config) == shape_len and tb_axis_groups is None:
        tb_shape = list(config)
    else:
        if tb_axis_groups is None:
            if len(config) != 2 or shape_len < 2:
                raise ValueError(
                    "non-full-rank fused tiles require a two-dimensional "
                    "(M, N) config or explicit tb_axis_groups"
                )
            tb_axis_groups = (tuple(range(shape_len - 1)), (shape_len - 1,))
        groups = tuple(tuple(group) for group in tb_axis_groups)
        if len(groups) != len(config):
            raise ValueError(
                f"tile {config} has {len(config)} extents but axis mapping "
                f"{groups} has {len(groups)} groups"
            )
        flat_axes = tuple(axis for group in groups for axis in group)
        if (
            len(set(flat_axes)) != len(flat_axes)
            or any(axis < 0 or axis >= shape_len for axis in flat_axes)
        ):
            raise ValueError(f"invalid/overlapping fused axis groups {groups}")
        tb_shape = [1] * shape_len
        for extent, axes in zip(config, groups):
            if not axes:
                raise ValueError("fused axis groups must not be empty")
            _map_flat_tile_extent(extent, shape, axes, tb_shape)
    if int(np.prod(tb_shape)) != int(np.prod(config)):
        raise AssertionError(
            f"internal fused tile mapping error: {config} -> {tb_shape}"
        )
    return tb_shape, get_dim_threads(tb_shape)

def get_auto_tune_tiling(input1_shape:tuple, element_op_bytes:OpBytes, single_chip:Arch):
    pass

# def element_wrapper(op_shape,tb_shape,dim_threads, arch,mem_levels,num_ops=1):


def element_wrapper(element_op_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, batch=1, type="cuda_core", tb_tiling_config=None, tb_axis_groups=None):
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
            tb_shape, dim_threads = get_transformed_tiling(
                input1_shape,
                element_op_bytes,
                single_chip,
                tb_tiling_config,
                tb_axis_groups=tb_axis_groups,
            )
        
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
