from mosaic.utils import OpBytes, Modeling_Granularity,Tensor_Loc
from tilesight.arch import Arch
import math
# from tilesight.fused_op_dtype.matmul_fused_op_new_api import calculate_matmul_resource_utilization_new
from tilesight.fused_op_dtype_wave.N_0_general_reduce_fused_op_wave import calculate_N_0_general_ruduce_resource_utilization
from tilesight.fused_op_dtype_wave.N_0_general_inter_thread_reduce_fused_op_wave import calculate_N_0_general_ruduce_inter_thread_resource_utilization

from tilesight.fusion_support.hete_post_process_single_op import hete_post_process_tensor_core_op,hete_post_process_sfu_core_op,hete_post_process_cuda_core_op

from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
from tilesight.fusion_support.get_hete_metrics import get_hete_metrics
from mosaic.cost.op_perf_stats import OpPerfStats
# tiling_configs
# tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages
from functools import lru_cache
from mosaic.utils import find_max_power_of_two
import torch
import numpy as np
import logging

log = logging.getLogger(__name__) 


def allocate_tiling(output_shape:tuple, out_tb_prod:int):

    def find_proper_factors(n: int) -> list:
        factors = []
        divisor = 2
        while n > 1 and divisor < 100:
            while n % divisor == 0:
                factors.append(divisor)
                n //= divisor
            divisor += 1
        # 若没有找到因数且 n>1，说明 n 本身是质数，将其作为因子加入
        if not factors and n > 1:
            factors.append(n)
        return factors
        
    factors = find_proper_factors(out_tb_prod)
    tiling = list(output_shape)

    # 从左到右优先，为每个因子寻找可整除的维度并分配（做除法）
    for f in factors:
        placed = False
        for i in range(len(tiling)):
            if tiling[i] % f == 0:
                tiling[i] //= f
                placed = True
                break
        if not placed:
            # 无法分配该因子，跳过并记录日志
            # log.warning("allocate_tiling -> factor %s cannot be allocated for output_shape=%s", f, output_shape)
            log.info("allocate_tiling -> factor %s cannot be allocated for output_shape=%s", f, output_shape)

    log.info("allocate_tiling -> output_shape=%s, factors=%s, tiling=%s", output_shape, factors, tiling)
    return tiling    


def get_default_tiling(reduce_op_bytes:OpBytes, single_chip:Arch, tb_tiling_config=None):
    if tb_tiling_config is None:
    
        input1_shape, _, output_shape = reduce_op_bytes.get_shapes()
        if (len(input1_shape) == (len(output_shape))) and output_shape[-1] == 1:
            # remove the last dimension of  output
            output_shape = output_shape[:-1]
            

        input_shape_len = len(input1_shape)
        output_shape_len = len(output_shape)

        reduction_shape = [input1_shape[-1]/output_shape[-1]]
        out_axis_mapping = [0] * output_shape_len
        # reduce_step = max(128, min(1024,reduction_shape[0]))
        reduce_step = min(max(128, min(1024,reduction_shape[0])), reduction_shape[0])

        if (reduce_step < 128):
            reduce_step = find_max_power_of_two(128, reduce_step)
            reduction_axis_mapping=[[0,reduce_step]]
            out_tb_shape = allocate_tiling(output_shape, 128//reduce_step)
            dim_threads= out_tb_shape
            reduce_threads_info=[0,reduce_step]
        else:
            reduction_axis_mapping=[[0,reduce_step]]
            out_tb_shape=[1] * output_shape_len
            dim_threads=[1] * output_shape_len
            reduce_threads_info=[0,128]

        # return     
        # shape_len = len(input1_shape)
        # reduction_shape = [input1_shape[-1]]
        # out_axis_mapping = [0] * shape_len
        # reduce_step = max(128, min(1024,reduction_shape))
        # reduction_axis_mapping=[[0,reduce_step]]
        # out_tb_shape=[1] * shape_len
        # dim_threads=[1] * shape_len
        # reduce_threads_info=[0,128]
        return output_shape,reduction_shape, out_axis_mapping, reduction_axis_mapping, out_tb_shape, dim_threads, reduce_threads_info
    elif tb_tiling_config is not None:
        input1_shape, _, output_shape = reduce_op_bytes.get_shapes()
        if (len(input1_shape) == (len(output_shape))) and output_shape[-1] == 1:
            # remove the last dimension of  output
            output_shape = output_shape[:-1]
        
        if (len(input1_shape) == (len(tb_tiling_config))) and tb_tiling_config[-1] == 1:
            # remove the last dimension of  output
            tb_tiling_config = tb_tiling_config[:-1]

            

        print("tb_tiling_config: ", tb_tiling_config)
            
        
        out_tb_shape = tb_tiling_config
        reduce_threads_info = [0,128]

        input_shape_len = len(input1_shape)
        output_shape_len = len(output_shape)

        reduction_shape = [input1_shape[-1]/output_shape[-1]]
        out_axis_mapping = [0] * output_shape_len
        reduce_step = max(128, min(1024,reduction_shape[0]))
        reduction_axis_mapping=[[0,reduce_step]]
        # out_tb_shape=[1] * output_shape_len
        dim_threads=out_tb_shape
        if np.prod(tb_tiling_config) >= 128:
            reduce_threads_info=[0,1]
        else:            
            reduce_threads_info=[0,int(128/np.prod(tb_tiling_config))]

        # return     
        # shape_len = len(input1_shape)
        # reduction_shape = [input1_shape[-1]]
        # out_axis_mapping = [0] * shape_len
        # reduce_step = max(128, min(1024,reduction_shape))
        # reduction_axis_mapping=[[0,reduce_step]]
        # out_tb_shape=[1] * shape_len
        # dim_threads=[1] * shape_len
        # reduce_threads_info=[0,128]
        return output_shape,reduction_shape, out_axis_mapping, reduction_axis_mapping, out_tb_shape, dim_threads, reduce_threads_info        


def reduce_wrapper(reduce_op_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, tb_tiling_config=None):
    # we assuem only one axis for reduce, e.g., rms_norm, softmax, etc.......
    # rms_norm: mean(x^2, last_dim)
    # softmax, last_dim (bs,head,seq,seq)
    mode=granularity.get_mode()
    tune_flag=granularity.get_auto_tune()

    input1_shape, input2_shape, output_shape = reduce_op_bytes.get_shapes()

    if output_shape == input1_shape:
        # time = 0
        # return (time), None
        bound_time    = 0.0  # 相当于 time
        ddr_util      = 0.0
        l2_hit_rate   = 0.0
        l2_util       = 0.0
        smem_footprint= 0.0
        smem_l1_util  = 0.0
        reg_footprint = 0.0
        ddr_read_io   = 0.0
        l2_read_io    = 0.0
        tensor_util   = 0.0
        cuda_util     = 0.0
        sfu_util      = 0.0

        hete_post_data = (
            bound_time,
            ddr_util,
            l2_hit_rate,
            l2_util,
            smem_footprint,
            smem_l1_util,
            reg_footprint,
            ddr_read_io,
            l2_read_io,
            tensor_util,
            cuda_util,
            sfu_util,
        )

        # out_tb_shape 对这个 degenerate reduce 没有意义，保持 None 即可
        return hete_post_data, None

    assert input1_shape[:-1] == output_shape[:-1] and input1_shape[-1] != output_shape[-1], "input1_shape should only differ from output_shape in the last dimension"
    

    if mode == "coarse":
        if tune_flag==False:
            pass
        elif tune_flag==True:
            log.warning("reduce op does not support / need auto tuning")
            log.warning("Fall back to use default tiling")


        # sum_levels={'in1': [1,1,1,2],'out1': [0,0,1,2]}
        # reduce_step = max(128, min(4096,cached_seq))
        # reduction_axis_mapping=[[0,reduce_step]]

        # matmul_levels={'in1': [0,1,1,2], 'in2': [1,1,1,2],'out1': [1,1,1,2]}
        # out_axis_mapping=[2,2,0,1]
        # reduction_axis_mapping=[[2,128]]
        # reduce_threads_info=[0,16]
        # ret=calculate_general_ruduce_inter_thread_resource_utilization([bs,num_attention_heads,seq,seq],[hidden_size/num_attention_heads],out_axis_mapping,reduction_axis_mapping
        #                                                                ,[8,1,1,1],[8,1,1,1],reduce_threads_info,chip,matmul_levels,compute_at=1)
        # ret=calculate_N_0_general_ruduce_inter_thread_resource_utilization([bs,seq],[hidden_size],[0,0],reduction_axis_mapping,[1,1],[1,1],[0,128],chip,mean_levels,compute_at=0)
        # post_data=bare_post_process_cuda_core_op(ret, chip, MAX_UTIL)
        # reg_array.append(post_data)

        # thread_per_tb=np.prod(dim_threads)*reduce_threads_info[1]
        # out_thread_shape=[math.ceil(tb_dim / thread_dim) for tb_dim, thread_dim in zip(out_tb_shape, dim_threads)]

        # reduce_threads_info=[0,2] 
        # reduce thread at reduce axis 0, reduce thread is 2
        # 对应reduction_shape以及reduction_threads的维度都要除以reduce_threads_info
        # reduction_axis_mapping[reduce_threads_info[0],1]=math.ceil(reduction_axis_mapping[reduce_threads_info[0],1]/reduce_threads_info[1])
        # reduction_shape[reduce_threads_info[0]]=math.ceil(reduction_shape[reduce_threads_info[0]]/reduce_threads_info[1])
        # ----------------------------------------------------------------------------------------------
        # reduction_step[reduce_threads_info[0]]=math.ceil(reduction_step[reduce_threads_info[0]]/reduce_threads_info[1])
        # ----------------------------------------------------------------------------------------------


        # shape_len = len(input1_shape)
        # reduction_shape = [input1_shape[-1]]
        # out_axis_mapping = [0] * shape_len
        # reduce_step = max(128, min(1024,reduction_shape))
        # reduction_axis_mapping=[[0,reduce_step]]
        # out_tb_shape=[1] * shape_len
        # dim_threads=[1] * shape_len
        # reduce_threads_info=[0,128]



        # ret=calculate_N_0_general_ruduce_resource_utilization([bs,num_attention_heads,seq],[cached_seq],[0,0,0],reduction_axis_mapping,[1,128,1],[1,128,1],chip,sum_levels,compute_at=0)
        
        # ret=calculate_N_0_general_ruduce_inter_thread_resource_utilization([bs,seq],[hidden_size],[0,0],reduction_axis_mapping,[1,1],[1,1],[0,128],chip,mean_levels,compute_at=0)
        

        
        out_shape,reduction_shape, out_axis_mapping, reduction_axis_mapping, out_tb_shape, dim_threads, reduce_threads_info = get_default_tiling(reduce_op_bytes, single_chip, tb_tiling_config)
        log.info("out_shape: %s, reduction_shape: %s, out_axis_mapping: %s, reduction_axis_mapping: %s, out_tb_shape: %s, dim_threads: %s, reduce_threads_info: %s", out_shape, reduction_shape, out_axis_mapping, reduction_axis_mapping, out_tb_shape, dim_threads, reduce_threads_info)

        ret=calculate_N_0_general_ruduce_inter_thread_resource_utilization(out_shape=out_shape, 
        reduction_shape=reduction_shape, out_axis_mapping=out_axis_mapping, reduction_axis_mapping=reduction_axis_mapping, 
        out_tb_shape=out_tb_shape, dim_threads=dim_threads, reduce_threads_info=reduce_threads_info, arch=single_chip, 
        mem_levels=reduce_op_bytes.to_mem_levels(), compute_at=0)
        



        hete_post_data=hete_post_process_cuda_core_op(ret, single_chip, reduce_op_bytes.output.num_bytes)

        return hete_post_data, out_tb_shape
        

    elif mode == "fine":
        pass
    elif mode == "roof":
        pass

            

if __name__ == "__main__":
    granularity = Modeling_Granularity(mode="coarse", comp_comm_overlap=True, auto_tune=True)
    silu_bytes=OpBytes(
        input1=Tensor_Loc(torch.float16, 'smem',[1, 64, 4096]),
        # input2=Tensor_Loc(torch.float16, 'ddr',[2, 64, 4096]),
        input2=None,
        output=Tensor_Loc(torch.float16, 'smem',[1, 64, 4096]),
    )

    input1_shape=(2, 64, 4096)
    from mosaic.arch import stacked_gpu_base
    single_chip = stacked_gpu_base()

    tiling = allocate_tiling(output_shape=[1, 64, 4096], out_tb_prod=16)
    print(tiling)


    # tb_shape, dim_threads=get_default_tiling(input1_shape, silu_bytes, single_chip)
    # print(tb_shape, dim_threads)
