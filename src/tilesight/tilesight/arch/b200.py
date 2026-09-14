from tilesight.arch.arch_base import Arch

class B200(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "B200"
        self.sm_count = 148
        self.base_freq = 1.965 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.max_freq = 1.965 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        # self.base_freq = 1.05 * 1e9 # 1.44 * 1e9 for tensorcore
        # self.max_freq = 1.05 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 32) # M,N.K
        self.tensor_core_flops = 2048
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64
        self.ddr_bandwidth = 8000 *1e9
        self.ddr_capacity = 192 * (1024**3)
        # self.l2_bandwidth = 15425.23e9 # 96* self.max_freq * 2 * 32 # no compression, 2 sectors/cycle?, 95???
        self.l2_bandwidth = 20160.86 * 1e9 
        self.l2_capacity = 126.5 * (1024**2) # H100 with dup
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 228 * (1024**1)
        self.register_capacity_per_sm = (256+256) * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2 
        
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        self.int32_cuda_core_flops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

        self.max_blocks_per_sm = 24  # Blackwell SM limit

        # L1.5: passive per-group cache (Blackwell)
        # From microbench CSV: peak BW 22,778 GB/s at 1MB, plateau ~20,560 GB/s at 3MB+
        self.l1_5_group_size = 8
        self.l1_5_capacity_per_group = 190 * 1024  # ~190KB per group
        self.l1_5_bandwidth = 22778e9              # 22,778 GB/s peak from CSV
        self.l1_5_associativity = 8
        self.l1_5_cacheline_bytes = 128
        self.l1_5_max_util = 0.9

        # --- Tensor Memory (TMEM) ---
        self._init_tmem()

    def _init_tmem(self):
        """Tensor Memory (Blackwell SM100) tcgen05 ld/st datapath. Call after max_freq is set."""
        # SM100 TMEM = 128 lanes x 512 cols x 4B = 256KB/SM. Used by softmax t2r/r2t
        # (read S / write P) and correction O-rescale (read/write O); the MMA accumulator
        # writeback is internal to the tensor core, NOT on this ld/st datapath.
        # Calibrated with an SM100 TMEM microbenchmark (tcgen05.32dp32b,
        # single-SM <<<1,256>>>), per-SM Bytes/Cycle:
        #   write-only    : 1023.88  (TEST_MODE=6)
        #   read-only     :  466.23  (TEST_MODE=7)
        #   combined R+W  : 1007.97  (TEST_MODE=8, REP=32/N_ITERS=512 to avoid local-mem
        #                             spills; REP=128 spilled and mismeasured this at 252.32)
        # FA4 steady state interleaves R(S,O)+W(P,O), so combined R+W is the effective
        # bandwidth for a single lumped tmem channel.
        self.tmem_capacity_per_sm = 256 * 1024
        self.tmem_write_bytes_per_cycle = 1023.88
        self.tmem_read_bytes_per_cycle = 466.228
        self.tmem_throughput_per_cycle = 1007.97   # combined R+W (measured, no spill), B/cyc/SM
        self.tmem_bandwidth = self.sm_count * self.max_freq * self.tmem_throughput_per_cycle
        self.tmem_max_util = 0.9

    def set_to_spec(self):
        return self

    def set_to_microbench(self):
        # self.base_freq = 1.52 * 1e9
        # self.max_freq = 1.52 * 1e9
        self.base_freq = 1.8 * 1e9
        self.max_freq = 1.8 * 1e9
        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0

        self.ddr_bandwidth = 6954.75 * 1e9
        # self.l2_bandwidth= 15425.23 *1e9
        self.l2_bandwidth= 20160.86 * 1e9 
        # self.smem_bandwidth = 37699.2 * 1e9
        self.smem_bandwidth = 37224.96 * 1e9


        self.fp32_cuda_core_flops = 57.72 * 1e12
        self.fp16_cuda_core_flops = 55.18 * 1e12
        self.fp64_cuda_core_flops = 30.32 * 1e12
        self.fp16_tensor_flops = 2184.91 * 1e12
        # sfu_flops not separately measured here; recompute at this freq for consistency
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm

        # TMEM bandwidth scales with this config's freq (B/cyc is freq-independent)
        self._init_tmem()

        return self

    def set_to_ncu(self):
        # After calling the base class constructor, set the properties
        self.core = "B200"
        self.sm_count = 148
        # self.base_freq = 1.965 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        # self.max_freq = 1.965 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.base_freq = 1.05 * 1e9 # ncu
        self.max_freq = 1.05 * 1e9 # ncu
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (16, 4, 16) # M,N.K
        self.tensor_core_flops = 2048
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64
        self.ddr_bandwidth = 8000 *1e9
        self.ddr_capacity = 192 * (1024**3)
        self.l2_bandwidth = 15425.23e9 # 96* self.max_freq * 2 * 32 # no compression, 2 sectors/cycle?, 95???
        self.l2_capacity = 126.5 * (1024**2) # H100 with dup
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 228 * (1024**1)
        self.register_capacity_per_sm = (256+256) * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2 
        
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        self.int32_cuda_core_flops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        # --- Tensor Memory (TMEM) ---
        self._init_tmem()

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

        return self


if __name__ == "__main__":
    chip=B200()
    # print(chip.l2_bandwidth)
    # print all the metrics
    print("chip.l2_bandwidth: TB/s",chip.l2_bandwidth/1e12)
    print("chip.ddr_bandwidth: TB/s",chip.ddr_bandwidth/1e12)
    print("chip.smem_bandwidth: TB/s",chip.smem_bandwidth/1e12)
    print("chip.register_bandwidth: TB/s",chip.register_bandwidth/1e12)
    print("chip.fp32_cuda_core_flops: TFLOPS/s",chip.fp32_cuda_core_flops/1e12)
    print("chip.fp16_cuda_core_flops: TFLOPS/s",chip.fp16_cuda_core_flops/1e12)
    print("chip.fp64_cuda_core_flops: TFLOPS/s",chip.fp64_cuda_core_flops/1e12)
    print("chip.int32_cuda_core_flops: TFLOPS/s",chip.int32_cuda_core_flops/1e12)
    print("chip.sfu_flops: TFLOPS/s",chip.sfu_flops/1e12)
    print("chip.fp16_tensor_flops: TFLOPS/s",chip.fp16_tensor_flops/1e12)
    print("chip.fp8_tensor_flops: TFLOPS/s",chip.fp8_tensor_flops/1e12)
    print("chip.sm_count",chip.sm_count)
    print("chip.tensor_cores_per_sm",chip.tensor_cores_per_sm)
    print("chip.tensor_core_shape",chip.tensor_core_shape)
    print("chip.tensor_core_flops: per SM per cycle",chip.tensor_core_flops)
    print("chip.fp32_cores_per_sm",chip.fp32_cores_per_sm)
    print("chip.int32_cores_per_sm",chip.int32_cores_per_sm)
    print("chip.ddr_capacity: GiB",chip.ddr_capacity/(1024**3))
