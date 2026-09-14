from tilesight.arch.arch_base import Arch

class H100(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "H100"
        self.sm_count = 132
        self.base_freq = 1.83 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.max_freq = 1.83 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        # self.base_freq = 1.44 * 1e9 # 1.44 * 1e9 for tensorcore
        # self.max_freq = 1.44 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        # self.base_freq = 1.41 * 1e9 # 1.44 * 1e9 for tensorcore
        # self.max_freq = 1.41 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 16) # M,N.K
        self.tensor_core_flops = 1024
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64
        self.ddr_bandwidth = 3352.32 *1e9
        self.ddr_capacity = 80 * (1024**3)
        self.l2_bandwidth = 8748e9 # 96* self.max_freq * 2 * 32 # no compression, 2 sectors/cycle?, 95???
        self.l2_capacity = 50 * (1024**2) # H100 with dup
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 228 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        # self.int8_int2_flops = self.fp16_tensor_flops * 6
        # self.int8_int1_flops = self.fp16_tensor_flops * 12
        
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2
        self.int32_cuda_core_flops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        # 111
        # self.ddr_max_util=1
        # self.l2_max_util= 0.93 
        # self.l1_max_util=1 
        # self.compute_max_util=1

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

        # L1.5: passive per-group cache (Hopper)
        self.l1_5_group_size = 8
        self.l1_5_capacity_per_group = 180 * 1024
        self.l1_5_bandwidth = 12758e9
        self.l1_5_associativity = 8
        self.l1_5_cacheline_bytes = 128
        self.l1_5_max_util = 0.9

if __name__ == "__main__":
    arch = H100()
    print("fp16_tensor_flops: ", arch.fp16_tensor_flops)
    print("int8_tensor_flops: ", arch.int8_tensor_flops)
    print("fp32_cuda_core_flops: ", arch.fp32_cuda_core_flops)
    print("fp16_cuda_core_flops: ", arch.fp16_cuda_core_flops)
    print("int32_cuda_core_flops: ", arch.int32_cuda_core_flops)
    print("sfu_flops: ", arch.sfu_flops)
    print("smem_bandwidth: ", arch.smem_bandwidth)
    print("register_bandwidth: ", arch.register_bandwidth)