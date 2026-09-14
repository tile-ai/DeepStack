# Filename: a100.py
from .arch_base import Arch

class A100(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "A100"
        self.sm_count = 108
        self.base_freq = 1.41 * 1e9
        self.max_freq = 1.41 * 1e9
        # self.base_freq = 1.06 * 1e9
        # self.max_freq = 1.06 * 1e9
        # self.base_freq = 1.386 * 1e9
        # self.max_freq = 1.386 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 8)
        self.tensor_core_flops = 512
        self.fp32_cores_per_sm = 64
        #self.ddr_freq=1.512*1e9
        # self.ddr_bus_width = 5120
        self.ddr_bandwidth = 1935 * 1e9
        self.ddr_capacity = 80 * (1024**3)
        self.l2_bandwidth = 5288 * 1e9
        self.l2_capacity = 30 * (1024**2) # effective cap here; A100 with dup
        self.sm_sub_partitions = 4
        # self.l1_smem_throughput_per_cycle = 128 / 1.33
        self.l1_smem_throughput_per_cycle = 128 
        self.configurable_smem_capacity = 164 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.fp16_tensor_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        self.int8_int2_flops = self.fp16_tensor_flops * 4
        self.int8_int1_flops = self.fp16_tensor_flops * 2
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 4
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util=0.9
        self.l2_max_util=0.75
        self.l1_max_util=0.9
        self.compute_max_util=0.9


        # self.fp16_tensor_flops = 298.951 * 1e12
        # self.l2_bandwidth=3234*1e9
        # self.l2_bandwidth=self.l2_bandwidth * 0.75
        # self.smem_bandwidth= 19491 * 1e9

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

    # def set_to_analytical_upper_bounds(self):
    #     # Set upper bounds for analysis
    #     self.max_freq = 1.5 * 1e9  # Assume frequency can increase to 1.5 GHz
    #     self.ddr_bandwidth = 2100 * 1e9  # Assume DDR bandwidth can increase
    #     self.l2_bandwidth = 6000 * 1e9  # Increase L2 cache bandwidth
    #     self.fp16_tensor_flops = 350 * 1e12  # Assume increased theoretical FP16 mixed-precision performance
    #     self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1.5  # Increase FP32 performance
    #     self.smem_bandwidth = 22000 * 1e9  # Increase shared-memory bandwidth

    #     # Recalculate all derived attributes that depend on these values
    #     self.calculate_derived_properties()
    def set_to_microbench(self):
        # Values from the project A100 calibration microbenchmark.
        # gpu_frequency_fmac: 1386.869376 MHz (realistic under compute load)
        self.base_freq = 1.386869 * 1e9
        self.max_freq = 1.386869 * 1e9

        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        # dram_bandwidth.cu: read 1654.812 GB/s, write 1692.901 GB/s, copy 1472.928 GB/s
        self.ddr_bandwidth = 1654.812 * 1e9
        # l2cache_bandwidth.cu: 3234.773 GB/s
        self.l2_bandwidth = 3234.773 * 1e9
        # smem_bandwidth.cu: whole chip 19491.839844 GB/s (theoretical from per-SM measurement)
        self.smem_bandwidth = 19491.839844 * 1e9

        # fp32_cuda_core_tflops.cu: 19.017181 TFLOPS
        self.fp32_cuda_core_flops = 19.017181 * 1e12
        self.fp16_cuda_core_flops = self.fp32_cuda_core_flops * 4
        self.fp64_cuda_core_flops = self.fp32_cuda_core_flops * 0.5
        # fp16_tensor_core_tflops.cu: 298.951 TFLOPS
        self.fp16_tensor_flops = 298.951 * 1e12
        self.int8_tensor_flops = self.fp16_tensor_flops * 2
        self.int8_int2_flops = self.fp16_tensor_flops * 4
        self.int8_int1_flops = self.fp16_tensor_flops * 2

        return self

    def set_to_spec(self):
        # Set upper bounds for analysis
        # self.max_freq = 1.5 * 1e9
        self.base_freq = 1.41 * 1e9
        self.max_freq = 1.41 * 1e9
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0
        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9

        self.ddr_bandwidth = 1935 * 1e9
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.l2_bandwidth=3234*1e9
        # self.l2_bandwidth=5288 * 1e9 *0.75
        # self.l2_bandwidth=5288 * 1e9
        self.l2_bandwidth=5288 * 1e9 * 1.0
        # self.l2_capacity = 40 * (1024**2) # effective cap here; A100 with dup
        self.l2_capacity = 40 * (1024**2) # effective cap here; A100 with dup

        self.smem_bandwidth= self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle

        return self
