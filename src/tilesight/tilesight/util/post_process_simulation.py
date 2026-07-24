from tilesight.arch import Arch

def bare_post_process_tensor_core_op(ret, arch:Arch, MAX_UTIL):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.fp16_tensor_flops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    return result



def bare_post_process_int8_tensor_core_op(ret, arch:Arch, MAX_UTIL):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.int8_tensor_flops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    return result

def bare_post_process_int8_int2_tensor_core_op(ret, arch:Arch, MAX_UTIL):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.int8_int2_flops
    # print("arch.int8_int2_flops",arch.int8_int2_flops)
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    return result

def bare_post_process_int8_int1_tensor_core_op(ret, arch:Arch, MAX_UTIL):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.int8_int1_flops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    return result

def bare_post_process_cuda_core_op(ret, arch:Arch, MAX_UTIL):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.fp32_cuda_core_flops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    if bound_time == 0:
        raise ValueError("bound_time is 0","ddr_io, l2_io, smem_l1_io, compute_flops",ddr_io, l2_io, smem_l1_io, compute_flops)
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    return result

def bare_post_process_fp64_cuda_core_op(ret, arch:Arch, MAX_UTIL):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.fp64_cuda_core_flops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    return result

def bare_post_process_fp64_divide_core_op(ret, arch:Arch, MAX_UTIL):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.fp64_divide_tflops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    return result

def bare_post_process_fp16_cuda_core_op(ret, arch:Arch, MAX_UTIL):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.fp16_cuda_core_flops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    return result

def bare_post_process_sfu_op(ret, arch:Arch, MAX_UTIL):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.sfu_flops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    
    return result


def process_matmul_op_result(ret, arch:Arch, MAX_UTIL, op_type, dims, count):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.fp16_tensor_flops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    # 构建结果字典
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util
    op_statistics = {
        'op_type': op_type,
        'dims': dims,
        'times': count,
        'result': result
    }
    
    return op_statistics


def process_element_op_result(ret, arch:Arch, MAX_UTIL, op_type, input_size, output_size, count):
    # 解析返回值
    ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops = ret
    
    # 计算各种时间和利用率
    ddr_time = ddr_io/arch.ddr_bandwidth
    l2_time = l2_io/arch.l2_bandwidth
    smem_l1_time = smem_l1_io/arch.smem_bandwidth
    compute_time = compute_flops/arch.fp32_cuda_core_flops
    bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    
    # 构建结果字典
    result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util
    op_statistics = {
        'op_type': op_type,
        'input_size': input_size,
        'output_size': output_size,
        'times': count,
        'result': result
    }
    
    return op_statistics


def dram_stack_post_process(ret, arch:Arch, MAX_UTIL=0.9):
    # 解析返回值
    dram_3d_cached_traffic, dram_3d_uncached_traffic, smem_l2_traffic,smem_reg_traffic, smem_footprint, tensor_flops, vector_flops, sfu_flops, reg_footprint, l2_noc_traffic_per_node_in, l2_noc_traffic_per_node_out, l1_noc_traffic_per_node_in, l1_noc_traffic_per_node_out, l2_hit_rate = ret

    effective_dram_traffic = dram_3d_cached_traffic + dram_3d_uncached_traffic/arch.dram_3d_uncached_max_util

    ddr_time = effective_dram_traffic/arch.ddr_bandwidth
    smem_l1_time = max(smem_l2_traffic/arch.smem_l2_bandwidth, smem_reg_traffic/arch.smem_register_bandwidth)
    compute_time = (tensor_flops/arch.fp16_tensor_flops + vector_flops/arch.fp16_cuda_core_flops + sfu_flops/arch.sfu_flops)
    
    l2_noc_in_time = l2_noc_traffic_per_node_in/arch.layer2_noc_single_direction_bw
    l2_noc_out_time = l2_noc_traffic_per_node_out/arch.layer2_noc_single_direction_bw
    l1_noc_in_time = l1_noc_traffic_per_node_in/arch.layer1_noc_single_direction_bw
    l1_noc_out_time = l1_noc_traffic_per_node_out/arch.layer1_noc_single_direction_bw

    l2_noc_time = max (l2_noc_in_time, l2_noc_out_time)
    l1_noc_time = max (l1_noc_in_time, l1_noc_out_time)

    # 计算各种时间和利用率
    # ddr_time = ddr_io/arch.ddr_bandwidth
    # l2_time = l2_io/arch.l2_bandwidth
    # smem_l1_time = smem_l1_io/arch.smem_bandwidth
    # compute_time = compute_flops/arch.fp16_tensor_flops
    bound_time = max(ddr_time, smem_l1_time, compute_time, l2_noc_time, l1_noc_time) / MAX_UTIL
    # bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    # l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    l2_noc_util = l2_noc_time / bound_time
    l1_noc_util = l1_noc_time / bound_time
    l2_noc_in_util = l2_noc_in_time / bound_time
    l2_noc_out_util = l2_noc_out_time / bound_time
    l1_noc_in_util = l1_noc_in_time / bound_time
    l1_noc_out_util = l1_noc_out_time / bound_time

    
    # result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    result = bound_time, 0, ddr_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, l2_noc_in_util, l2_noc_out_util, l1_noc_in_util, l1_noc_out_util, l2_hit_rate
    return result

def dram_stack_post_process_v2(ret, arch:Arch, MAX_UTIL=0.9):
    # 解析返回值
    dram_3d_cached_traffic, dram_3d_uncached_traffic, l2_to_smem_traffic, smem_to_l2_traffic, smem_reg_traffic, smem_footprint, tensor_flops, vector_flops, sfu_flops, reg_footprint, l2_noc_traffic_per_node_in, l2_noc_traffic_per_node_out, l1_noc_traffic_per_node_in, l1_noc_traffic_per_node_out, l2_hit_rate = ret

    effective_dram_traffic = dram_3d_cached_traffic + dram_3d_uncached_traffic/arch.dram_3d_uncached_max_util

    ddr_time = effective_dram_traffic/arch.ddr_bandwidth
    # smem_l1_time = max(smem_l2_traffic/arch.smem_l2_bandwidth, smem_reg_traffic/arch.smem_register_bandwidth)
    l2_to_smem_time = l2_to_smem_traffic/arch.l2_to_smem_bandwidth
    smem_to_l2_time = smem_to_l2_traffic/arch.smem_to_l2_bandwidth
    smem_reg_time = smem_reg_traffic/arch.smem_register_bandwidth
    smem_l1_time = max(l2_to_smem_time, smem_to_l2_time, smem_reg_time)
    
    compute_time = (tensor_flops/arch.fp16_tensor_flops + vector_flops/arch.fp16_cuda_core_flops + sfu_flops/arch.sfu_flops)
    
    l2_noc_in_time = l2_noc_traffic_per_node_in/arch.layer2_noc_single_direction_bw
    l2_noc_out_time = l2_noc_traffic_per_node_out/arch.layer2_noc_single_direction_bw
    l1_noc_in_time = l1_noc_traffic_per_node_in/arch.layer1_noc_single_direction_bw
    l1_noc_out_time = l1_noc_traffic_per_node_out/arch.layer1_noc_single_direction_bw

    l2_noc_time = max (l2_noc_in_time, l2_noc_out_time)
    l1_noc_time = max (l1_noc_in_time, l1_noc_out_time)

    # 计算各种时间和利用率
    # ddr_time = ddr_io/arch.ddr_bandwidth
    # l2_time = l2_io/arch.l2_bandwidth
    # smem_l1_time = smem_l1_io/arch.smem_bandwidth
    # compute_time = compute_flops/arch.fp16_tensor_flops
    bound_time = max(ddr_time, smem_l1_time, compute_time, l2_noc_time, l1_noc_time) / MAX_UTIL
    # bound_time = max(ddr_time, l2_time, smem_l1_time, compute_time) / MAX_UTIL
    ddr_util = ddr_time / bound_time
    # l2_util = l2_time / bound_time
    smem_l1_util = smem_l1_time / bound_time
    compute_util = compute_time / bound_time
    l2_noc_util = l2_noc_time / bound_time
    l1_noc_util = l1_noc_time / bound_time
    l2_noc_in_util = l2_noc_in_time / bound_time
    l2_noc_out_util = l2_noc_out_time / bound_time
    l1_noc_in_util = l1_noc_in_time / bound_time
    l1_noc_out_util = l1_noc_out_time / bound_time
    l2_to_smem_util = l2_to_smem_time / bound_time
    smem_to_l2_util = smem_to_l2_time / bound_time
    smem_reg_util = smem_reg_time / bound_time


    
    # result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util, ddr_read_io, l2_read_io
    result = bound_time, 0, ddr_util, smem_footprint, smem_l1_util, l2_to_smem_util, smem_to_l2_util, smem_reg_util, reg_footprint, compute_util, l2_noc_in_util, l2_noc_out_util, l1_noc_in_util, l1_noc_out_util, l2_hit_rate
    return result