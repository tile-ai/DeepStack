def get_hete_metrics(post_data):
    # bound_time, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, ddr_read_io, l2_read_io, tensor_util, cuda_util, sfu_util

    modeling_metrics = {
    "Overall time /s": post_data[0],
    "DDR util": post_data[1]*100,
    "L2 Read Hit Rate": post_data[2]*100,
    "L2 Util": post_data[3]*100,
    "Smem Footprint per Thread Block/Bytes": post_data[4],  # 假定这是对应的
    "Smem/L1 Util": post_data[5]*100,
    "Reg Footprint per Thread": post_data[6],

    "Compute Util - Tensor": post_data[9]*100,  # 
    "Compute Util - CUDA": post_data[10]*100,  # 
    "Compute Util - SFU": post_data[11]*100,  #
}
    return modeling_metrics