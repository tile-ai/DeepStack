
from .arch_base import Arch

class B6000(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "RTX5090"
        self.sm_count = 188
        self.base_freq = 2.43 * 1e9
        self.max_freq = 2.43 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 4)
        self.tensor_core_flops = 256
        self.fp32_cores_per_sm = 128
        self.ddr_bandwidth = 1792 * 1e9
        self.ddr_capacity = 96 * (1024**3)
        self.l2_bandwidth = 7602.59 * 1e9 *self.base_freq/2.43
        self.l2_capacity = 128 * (1024**2) # 
        self.sm_sub_partitions = 4
        # self.l1_smem_throughput_per_cycle = 128 / 1.33
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 100 * (1024**1) #128 in total
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.fp16_tensor_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        self.int4_tflops = self.fp16_tensor_flops * 4 
        # self.int8_int2_flops = self.fp16_tensor_flops * 6
        # self.int8_int1_flops = self.fp16_tensor_flops * 12
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 / 64
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        # self.ddr_max_util=0.9
        # self.l2_max_util=0.75
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

        # GB202 (sm_120): Ada 风格 mma, 无 Hopper wgmma / Blackwell DC tcgen05(UTCMMA)
        self.support_wgmma = False
        self.support_utcmma = False

        # L2<->DRAM wave 粒度: H100 (5120-bit HBM) 校准值 8196, GB202 512-bit GDDR7 按总线比例 ~4096
        self.ddr_wave_bytes = 4096
        # element op 的 DDR 事务/tile 粒度: GDDR7 32B/channel x 32 channels
        self.ddr_transaction_size = 1024

        # 其他 dtype 的 tensor core 吞吐 (GB202: tf32 = fp16/2, fp8 = fp16x2)
        self.fp32_tensor_flops = self.fp16_tensor_flops / 2
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2
        self.int32_cores_per_sm = 64
        self.int32_cuda_core_flops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2

    def set_to_spec(self):
        # 设定分析上限
        # self.max_freq = 1.5 * 1e9
        self.base_freq = 2.43 * 1e9
        self.max_freq = 2.43 * 1e9
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0
        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9

        self.ddr_bandwidth = 1792 * 1e9
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.l2_bandwidth=3234*1e9
        # self.l2_bandwidth=5288 * 1e9 *0.75
        self.l2_bandwidth=7602.59 * 1e9
        # self.l2_capacity = 40 * (1024**2) # effective cap here; A100 with dup
        self.l2_capacity = 128 * (1024**2) # effective cap here; A100 with dup

        self.smem_bandwidth= self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle

        return self

    def get_tensor_core_minimum_ptx(self, bytes = 2):
        # sm_120 mma 最小形状 (同 Ampere/Ada 风格)
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

    def set_to_microbench(self):
        self.base_freq =  2.132 * 1e9
        self.max_freq =  2.132 * 1e9
        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0

        self.ddr_bandwidth = 1442.90 * 1e9
        self.l2_bandwidth= 7602.59 *1e9
        self.smem_bandwidth = 58475.52 * 1e9

        self.fp32_cuda_core_flops = 88.58 * 1e12
        self.fp16_cuda_core_flops = 103.27 * 1e12
        self.fp64_cuda_core_flops = 1.448851 * 1e12
        self.fp16_tensor_flops = 432.86 * 1e12
        
        return self
