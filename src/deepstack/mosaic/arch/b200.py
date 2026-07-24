from tilesight.arch.arch_base import Arch

class B200(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "B200"
        self.sm_count = 148 # 132 // 1.5
        self.core_freq = 1.965 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        # self.memory_freq = 0.4 * 1e9
        self.noc_freq = 1.6 * 1e9
        self.base_freq = self.core_freq
        self.max_freq = self.core_freq
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 32) # M,N.K
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64
        self.ddr_bandwidth = 8000 * 1e9
        self.ddr_capacity = 192 * (1024**3)
        self.l2_bandwidth = 20160.86 * 1e9 / 0.9  # this it the value from benchmark, while as it will be later scaled down by 0.9 utilization, so we need to divide by 0.9 here
        self.l2_capacity = 126.5 * (1024**2) # H100 with dup
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 228 * (1024**1)
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
        self.support_utcmma = True

        self._init_tmem()

    def _init_tmem(self):
        """Tensor Memory (Blackwell SM100) tcgen05 ld/st datapath.

        Calibrated with an SM100 TMEM microbenchmark (tcgen05.32dp32b,
        single-SM <<<1,256>>>), B/cyc/SM:
          write-only 1023.88 | read-only 466.23 | combined R+W 1007.97 (REP=32,
          avoiding the local-mem spill that mismeasured it at 252 with REP=128).
        FA4 steady state interleaves R(S,O)+W(P,O), so combined R+W is the
        effective bandwidth for a single lumped tmem channel. Freq-independent
        per-cycle rate; bandwidth scales with whatever freq is active.
        """
        self.tmem_capacity_per_sm = 256 * 1024
        self.tmem_write_bytes_per_cycle = 1023.88
        self.tmem_read_bytes_per_cycle = 466.228
        self.tmem_throughput_per_cycle = 1007.97   # combined R+W (measured, no spill)
        self.tmem_bandwidth = self.sm_count * self.max_freq * self.tmem_throughput_per_cycle
        self.tmem_max_util = 0.9

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
        self.fp8_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
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
        return self

    def set_to_microbench(self):
        self.base_freq = 1.8 * 1e9
        self.max_freq = 1.8 * 1e9
        self.update_derived()
        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0

        self.ddr_bandwidth = 6954.75 * 1e9
        self.l2_bandwidth= 20160.86 *1e9
        # self.smem_bandwidth = 37699.2 * 1e9
        self.smem_bandwidth = 37224.96 * 1e9


        self.fp32_cuda_core_flops = 57.72 * 1e12
        self.fp16_cuda_core_flops = 55.18 * 1e12
        self.fp64_cuda_core_flops = 30.32 * 1e12
        self.fp16_tensor_flops = 2184.91 * 1e12
        self.sfu_flops = 4.108416 * 1e12

        self.int8_tensor_flops = self.fp16_tensor_flops * 2
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2
        self._init_tmem()   # TMEM bw scales with this freq
        return self

    def set_to_ncu(self):
        self.base_freq = 1.05 * 1e9 # ncu
        self.max_freq = 1.05 * 1e9 # ncu
        self.update_derived()
        self._init_tmem()   # TMEM bw scales with this freq
        return self
    
    # def self.fp16_tensor_core_minimum_ptx = (8, 8, 16)
    def get_tensor_core_minimum_ptx(self, bytes = 2):
        if bytes == 2:
            return (64, 8, 16)
        elif bytes == 4:
            return (64, 8, 8)
        elif bytes == 1:
            return (64, 8, 32)
        elif bytes == 0.5:
            return (64, 8, 64)
        else:
            raise ValueError("bytes must be 2, 4, 1, or 0.5")
            
# arch_config 2,3,4,5 still needs to be added......


if __name__ == "__main__":
    arch_instance = B200()
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
    print("fp8_tensor_flops: ", arch_instance.fp8_tensor_flops/1e12,"TFlops")
    print("fp32_cuda_core_flops: ", arch_instance.fp32_cuda_core_flops/1e12,"TFlops")
    print("fp16_cuda_core_flops: ", arch_instance.fp16_cuda_core_flops/1e12,"TFlops")
    print("fp64_cuda_core_flops: ", arch_instance.fp64_cuda_core_flops/1e12,"TFlops")
    print("int32_cuda_core_flops: ", arch_instance.int32_cuda_core_flops/1e12,"TFlops")
    print("sfu_flops: ", arch_instance.sfu_flops/1e12,"TFlops")
