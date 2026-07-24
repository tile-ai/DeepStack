from tilesight.arch.arch_base import Arch


class MI325X(Arch):
    """AMD Instinct MI325X model using calibrated microbenchmark values.

    The defaults mirror the standard spec with 0.9 utilization.  The vLLM E2E
    drivers call set_to_microbench() to use measured calibration values for
    compute and memory bandwidth.
    """

    def __init__(self):
        super().__init__()

        self.core = "MI325X"
        self.sm_count = 304
        self.core_freq = 2.1 * 1e9
        self.noc_freq = 1.6 * 1e9
        self.base_freq = self.core_freq
        self.max_freq = self.core_freq

        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (16, 16, 16)
        self.tensor_core_flops = 512
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64

        self.ddr_bandwidth = 6000 * 1e9
        self.ddr_capacity = 256 * (1024**3)
        # self.l2_bandwidth = 16750.6 * 1e9 / 0.9
        self.l2_bandwidth = 25000 * 1e9
        self.l2_capacity = 256 * (1024**2)

        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 128 * 1024
        self.register_capacity_per_sm = 256 * 1024
        self.warp_schedulers_per_sm = 1
        self.sfu_cores_per_sm = 16

        self.ddr_transaction_size = 128
        self.ddr_stack = 1
        self.ddr_wave_bytes = self.ddr_transaction_size * self.ddr_stack
        self.dram_3d_uncached_max_util = 0.9

        self.update_derived()

        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

        self.support_wgmma = True
        self.support_utcmma = False

    def update_derived(self):
        self.layer1_noc_size = self.sm_count
        self.layer1_noc = "xbar"
        self.layer1_noc_single_direction_bw = 128 * self.noc_freq

        self.fp16_tensor_flops = (
            self.sm_count
            * self.max_freq
            * self.tensor_cores_per_sm
            * self.tensor_core_flops
        )
        self.fp32_tensor_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp8_tensor_flops = self.fp16_tensor_flops
        self.int8_tensor_flops = self.fp16_tensor_flops
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp8_cuda_core_flops = self.fp16_cuda_core_flops
        self.fp64_cuda_core_flops = self.fp32_cuda_core_flops * 0.5
        self.int32_cuda_core_flops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.smem_l2_bandwidth = self.smem_bandwidth
        self.l2_to_smem_bandwidth = self.smem_bandwidth * 0.5
        self.smem_to_l2_bandwidth = self.smem_bandwidth * 0.5
        self.smem_register_bandwidth = self.smem_bandwidth
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

    def set_to_spec(self):
        self.base_freq = 2.1 * 1e9
        self.max_freq = 2.1 * 1e9
        self.update_derived()
        self.ddr_bandwidth = 6000 * 1e9
        # self.l2_bandwidth = 16750.6 * 1e9
        self.l2_bandwidth = 25000 * 1e9
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0
        return self

    def set_to_microbench(self):
        self.base_freq = 2.094 * 1e9
        self.max_freq = 2.094 * 1e9
        self.update_derived()

        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        self.ddr_bandwidth = 3972.6 * 1e9
        self.l2_bandwidth = 16750.6 * 1e9
        self.smem_bandwidth = 50160.0 * 1e9
        self.smem_l2_bandwidth = self.smem_bandwidth
        self.l2_to_smem_bandwidth = self.smem_bandwidth * 0.5
        self.smem_to_l2_bandwidth = self.smem_bandwidth * 0.5
        self.smem_register_bandwidth = self.smem_bandwidth

        self.fp32_cuda_core_flops = 120.19 * 1e12
        self.fp16_cuda_core_flops = 138.22 * 1e12
        self.fp8_cuda_core_flops = self.fp16_cuda_core_flops
        self.fp64_cuda_core_flops = 63.59 * 1e12
        self.fp16_tensor_flops = 1241.0 * 1e12
        self.fp8_tensor_flops = self.fp16_tensor_flops
        self.int8_tensor_flops = self.fp16_tensor_flops
        self.sfu_flops = 4.51 * 1e12
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4
        return self

    def set_to_ncu(self):
        return self

    def get_tensor_core_minimum_ptx(self, bytes=2):
        if bytes == 2:
            return (16, 16, 16)
        if bytes == 4:
            return (16, 16, 8)
        if bytes == 1:
            return (16, 16, 32)
        if bytes == 0.5:
            return (16, 16, 64)
        raise ValueError("bytes must be 2, 4, 1, or 0.5")


class MI325XSpec(MI325X):
    """MI325X default-spec variant for DSE combo indexing.

    specify_decode_v4 rebuilds arch/noc in child processes by class name and
    NoC name.  This subclass lets us compare the default constructor path
    against MI325X().set_to_microbench() without ambiguity.
    """

    pass
