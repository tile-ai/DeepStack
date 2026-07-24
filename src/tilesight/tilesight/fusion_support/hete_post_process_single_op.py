from tilesight.arch import Arch

def hete_post_process_tensor_core_op(ret, arch:Arch , data_bytes):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret

    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    if data_bytes==4:
        compute_time = compute_flops/arch.fp32_tensor_flops
    elif  data_bytes==2:
        compute_time = compute_flops/arch.fp16_tensor_flops
    elif data_bytes==1:
        # if (fp8_tensor_flops exist)
        if hasattr(arch, 'fp8_tensor_flops'):
            compute_time = compute_flops/arch.fp8_tensor_flops
        elif hasattr(arch, 'int8_tensor_flops'):
            compute_time = compute_flops/arch.int8_tensor_flops
        else:
            raise ValueError("fp8_tensor_flops or int8_tensor_flops not found")
    # compute_time = compute_flops/arch.fp16_tensor_flops
    bound_time = max(ddr_time/arch.ddr_max_util, l2_time/arch.l2_max_util, smem_l1_time/arch.l1_max_util, compute_time/arch.compute_max_util)
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    tensor_util = compute_time / bound_time

    cuda_util = 0
    sfu_util = 0
    
    # result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    result = bound_time, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, ddr_read_io, l2_read_io, tensor_util, cuda_util, sfu_util
    return result

def hete_post_process_cuda_core_op(ret, arch:Arch, data_bytes):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    if  data_bytes==8:
        compute_time = compute_flops/arch.fp64_cuda_core_flops
    elif data_bytes==4:
        compute_time = compute_flops/arch.fp32_cuda_core_flops
    elif data_bytes ==2:
        compute_time = compute_flops/arch.fp16_cuda_core_flops
    elif data_bytes ==1:
        compute_time = compute_flops/arch.fp8_cuda_core_flops

    bound_time = max(ddr_time/arch.ddr_max_util, l2_time/arch.l2_max_util, smem_l1_time/arch.l1_max_util, compute_time/arch.compute_max_util)
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    cuda_util = compute_time / bound_time

    tensor_util = 0
    sfu_util = 0
    
    # result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    result = bound_time, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, ddr_read_io, l2_read_io, tensor_util, cuda_util, sfu_util
    return result


def hete_post_process_sfu_core_op(ret, arch:Arch):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # we will halve the compute flops for sfu core because defaultly we double the flops for cuda cores (FMA)
    compute_flops/=2

    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.sfu_flops
    bound_time = max(ddr_time/arch.ddr_max_util, l2_time/arch.l2_max_util, smem_l1_time/arch.l1_max_util, compute_time/arch.compute_max_util)
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    sfu_util = compute_time / bound_time

    tensor_util = 0
    cuda_util = 0
    # result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    result = bound_time, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, ddr_read_io, l2_read_io, tensor_util, cuda_util, sfu_util
    return result
