from tilesight.arch.arch_base import Arch


class B6000(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments

        # After calling the base class constructor, set the properties
        self.core = "RTX5090"
        self.sm_count = 188
        self.core_freq = 2.43 * 1e9
        self.memory_freq = 0.4 * 1e9
        self.noc_freq = 1.6 * 1e9
        self.base_freq = self.core_freq
        self.max_freq = self.core_freq
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 4)  # M,N,K
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64
        self.ddr_bandwidth = 1792 * 1e9
        self.ddr_capacity = 96 * (1024**3)
        self.l2_bandwidth = 7602.59 * 1e9
        self.l2_capacity = 128 * (1024**2)
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 100 * (1024**1)  # 128 in total
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm = 16

        self.ddr_transaction_size = 128
        self.ddr_stack = 1
        self.ddr_wave_bytes = self.ddr_transaction_size * self.ddr_stack

        self.dram_3d_uncached_max_util = 0.9

        # Calculate the derived properties
        self.update_derived()

        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

        self.support_wgmma = False
        self.support_utcmma = False

    def update_derived(self):

        self.layer1_noc_size = self.sm_count
        self.layer1_noc = "xbar"
        self.layer1_noc_single_direction_bw = 128 * self.noc_freq

        self.tensor_core_flops = (
            self.tensor_core_shape[0]
            * self.tensor_core_shape[1]
            * self.tensor_core_shape[2]
            * 2
        )
        self.fp16_tensor_flops = (
            self.sm_count
            * self.max_freq
            * self.tensor_cores_per_sm
            * self.tensor_core_flops
        )
        self.fp32_tensor_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2
        self.int8_tensor_flops = self.fp16_tensor_flops * 2
        self.int4_tflops = self.fp16_tensor_flops * 4
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp8_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp64_cuda_core_flops = (
            self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 / 64
        )
        self.int32_cuda_core_flops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.smem_l2_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.l2_to_smem_bandwidth = (
            self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle * 0.5
        )
        self.smem_to_l2_bandwidth = (
            self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle * 0.5
        )
        self.smem_register_bandwidth = (
            self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        )
        self.register_bandwidth = (
            self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4
        )

    def set_to_spec(self):
        self.core_freq = 2.43 * 1e9
        self.base_freq = self.core_freq
        self.max_freq = self.core_freq
        self.update_derived()

        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        self.ddr_bandwidth = 1792 * 1e9
        self.l2_bandwidth = 7602.59 * 1e9
        self.l2_capacity = 128 * (1024**2)

        return self

    def set_to_microbench(self):
        self.core_freq = 2.132 * 1e9
        self.base_freq = self.core_freq
        self.max_freq = self.core_freq
        self.update_derived()

        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        self.ddr_bandwidth = 1442.90 * 1e9
        self.l2_bandwidth = 7602.59 * 1e9
        self.smem_bandwidth = 58475.52 * 1e9

        self.fp32_cuda_core_flops = 88.58 * 1e12
        self.fp16_cuda_core_flops = 103.27 * 1e12
        self.fp64_cuda_core_flops = 1.448851 * 1e12
        self.fp16_tensor_flops = 432.86 * 1e12
        self.int8_tensor_flops = self.fp16_tensor_flops * 2
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2
        self.int4_tflops = self.fp16_tensor_flops * 4

        return self

    def set_to_ncu(self):

        return self

    def get_tensor_core_minimum_ptx(self, bytes=2):
        if bytes == 2:
            return (8, 8, 4)
        elif bytes == 4:
            return (8, 8, 4)
        elif bytes == 1:
            return (8, 8, 16)
        elif bytes == 0.5:
            return (8, 8, 32)
        else:
            raise ValueError("bytes must be 2, 4, 1, or 0.5")
