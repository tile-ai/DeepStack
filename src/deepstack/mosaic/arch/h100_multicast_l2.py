from tilesight.arch.arch_base import Arch

class H100(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "H100"
        self.sm_count = 132
        self.core_freq = 1.83 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.memory_freq = 0.4 * 1e9
        self.noc_freq = 1.6 * 1e9
        self.base_freq = self.core_freq
        self.max_freq = self.core_freq
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 16) # M,N.K
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64
        self.ddr_bandwidth = 3350 * 1e9
        self.ddr_capacity = 80 * (1024**3)
        self.l2_bandwidth = 16790.417380 * 1e9 # scaling down by 1.4 for memory freq
        self.l2_capacity = 50 * (1024**2) # H100 with dup
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 256 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16


        self.ddr_transaction_size = 128
        self.ddr_stack = 1
        self.ddr_wave_bytes = self.ddr_transaction_size * self.ddr_stack
        
        
        self.dram_3d_uncached_max_util = 0.9

        
        # Calculate the derived properties
        self.update_derived()

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

        self.support_wgmma = True
        self.support_utcmma = False

    def update_derived(self):

        self.layer1_noc_size = self.sm_count
        self.layer1_noc = "xbar"
        self.layer1_noc_single_direction_bw = 128 * self.noc_freq 


        self.tensor_core_flops = self.tensor_core_shape[0] * self.tensor_core_shape[1] * self.tensor_core_shape[2] * 2
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.fp32_tensor_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2
        self.int8_tensor_flops = self.fp16_tensor_flops * 2
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        # self.fp8_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        self.int32_cuda_core_flops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle * 1
        self.smem_l2_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle * 1
        self.l2_to_smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle * 0.5
        self.smem_to_l2_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle * 0.5
        self.smem_register_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle * 1
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

    def set_to_spec(self):
        self

    def set_to_microbench(self):

        return self

    def set_to_ncu(self):

        return self
    
    # def self.fp16_tensor_core_minimum_ptx = (8, 8, 16)
    def get_tensor_core_minimum_ptx(self, bytes = 2):
        if bytes == 2:
            return (8, 8, 16)
        elif bytes == 4:
            return (8, 8, 8)
        elif bytes == 1:
            return (8, 8, 32)
        elif bytes == 0.5:
            return (8, 8, 64)
        else:
            raise ValueError("bytes must be 2, 4, 1, or 0.5")
            
# arch_config 2,3,4,5 still needs to be added......


if __name__ == "__main__":
    arch_instance = H100()
    print("sm_count: ", arch_instance.sm_count)
    print("base_freq: ", arch_instance.base_freq/1e9,"GHz")
    print("max_freq: ", arch_instance.max_freq/1e9,"GHz")
    print("tensor_cores_per_sm: ", arch_instance.tensor_cores_per_sm)
    print("tensor_core_shape: ", arch_instance.tensor_core_shape)
    print("tensor_core_flops: ", arch_instance.tensor_core_flops)
    print("fp32_cores_per_sm: ", arch_instance.fp32_cores_per_sm)
    print("ddr_bandwidth: ", arch_instance.ddr_bandwidth/1e9,"GB/s")
    print("ddr_capacity: ", arch_instance.ddr_capacity/(1024**3),"GiB")
    print("l2_bandwidth: ", arch_instance.l2_bandwidth/1e9,"GB/s")
    print("l2_capacity: ", arch_instance.l2_capacity/(1024**2),"MiB")
    print("smem_bandwidth: ", arch_instance.smem_bandwidth/1e9,"GB/s")
    print("register_bandwidth: ", arch_instance.register_bandwidth/1e9,"GB/s")
    print("layer1_noc_single_direction_bw: ", arch_instance.layer1_noc_single_direction_bw/1e9,"GB/s")
    print("layer1_noc_single_direction_bw_overall: ", arch_instance.layer1_noc_single_direction_bw/1e9 * arch_instance.layer1_noc_size ,"GB/s")
    print("fp16_tensor_flops: ", arch_instance.fp16_tensor_flops/1e12,"TFlops")
    print("fp32_cuda_core_flops: ", arch_instance.fp32_cuda_core_flops/1e12,"TFlops")
    print("fp16_cuda_core_flops: ", arch_instance.fp16_cuda_core_flops/1e12,"TFlops")
    print("fp64_cuda_core_flops: ", arch_instance.fp64_cuda_core_flops/1e12,"TFlops")
    print("int32_cuda_core_flops: ", arch_instance.int32_cuda_core_flops/1e12,"TFlops")
    print("sfu_flops: ", arch_instance.sfu_flops/1e12,"TFlops")
