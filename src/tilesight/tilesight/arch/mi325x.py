from .arch_base import Arch

# =============================================================================
# AMD Instinct MI325X (CDNA3, gfx942, NPS1 unified partition)
#
# Three modes (mirrors mi210.py):
#   __init__()           -> default STANDARD SPEC (derived peaks, util=0.9)
#   set_to_spec()        -> theoretical peak, util=1.0
#   set_to_microbench()  -> measured microbench values, util=1.0
#
# SPEC source: AMD Instinct MI325X product page + CDNA3 whitepaper:
#   Compute Units            304 CU (4 XCDs × 76 CU)
#   Stream processors        19,456          (= 304 × 64)
#   Matrix Cores             1,216           (= 304 × 4)
#   Peak FP16/BF16 matrix    ~1,307 TF       (= 304 × 4 × 512 flops/cycle × 2.1 GHz)
#   Peak FP32 vector         ~163 TF         (= 304 × 128 × 2 × 2.1 GHz, packed)
#   Peak FP64 vector         ~81.7 TF        (= 0.5 × FP32)
#   Memory                   256 GiB HBM3E, ~6.0 TB/s peak bandwidth
#   Max sclk                 2.1 GHz
#   L2 (per XCD)             ~32 MiB; ~256 MiB total (L3 in HSA topology)
#   L2 channels (per part.)  16 slices × 256 B/cycle → ~8.96 TB/s peak per XCD
#   LDS per CU               64 KiB, 32 banks × 4 B = 128 B/cycle hardware peak
#
# MICROBENCH provenance: project calibration measurements for MI325X
#   (ROCm/gfx942).
#
# KEY DIFF vs MI210:
#   tensor_core_flops = 512  (CDNA3 doubled MFMA throughput vs CDNA2's 256)
#   sm_count = 304           (vs 104 for MI210)
#   max_freq = 2.1 GHz       (vs 1.7 GHz)
#   HBM3E ~6000 GB/s         (vs HBM2e 1638 GB/s)
# =============================================================================
class MI325X(Arch):
    def __init__(self):
        super().__init__()

        self.core = "MI325X"
        self.sm_count = 304                # CU count
        self.base_freq = 2.1 * 1e9         # max sclk 2100 MHz
        self.max_freq = 2.1 * 1e9

        # MFMA matrix cores: 4 per CU, CDNA3 doubled throughput to 512 flops/core/cycle.
        # 304 × 4 × 512 × 2.1 GHz = 1,307 TF  (matches datasheet ~1307 TF FP16 dense)
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (16, 16, 16)  # mfma_f32_16x16x16f16 tile
        self.tensor_core_flops = 512

        # Packed FP32: model as 128 effective cores/CU (same trick as MI210).
        # 304 × 128 × 2 × 2.1 GHz = 163.5 TF (matches AMD FP32 vector spec ~163 TF).
        self.fp32_cores_per_sm = 128

        # HBM3E peak bandwidth: ~6.0 TB/s
        self.ddr_bandwidth = 6000 * 1e9
        self.ddr_capacity = 256 * (1024**3)  # 256 GiB

        # L2: 16 slices/XCD × 256 B/cycle (CDNA3) × 2.1 GHz = 8.96 TB/s per XCD.
        # 4 XCDs unified → 4 × 8.96 = ~35.8 TB/s theoretical; but HSA presents one
        # partition so effective inter-XCD L2 BW is lower. Measured 16.75 TB/s.
        # Use 16 slices (as reported by omniperf for gfx942 NPS1) × 256 × freq:
        self.l2_bandwidth = 16 * 256 * self.max_freq   # ~8.96 TB/s (per-partition)
        self.l2_capacity = 256 * (1024**2)              # 256 MiB total (L3 in HSA)

        self.sm_sub_partitions = 4          # 4 SIMD16 units per CU
        self.l1_smem_throughput_per_cycle = 128  # 32 banks × 4 B = 128 B/cycle/CU
        self.configurable_smem_capacity = 64 * (1024**1)   # 64 KiB LDS per workgroup
        # CDNA3: 256 KiB VGPRs + 256 KiB AGPRs per CU
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 1
        self.sfu_cores_per_sm = 16

        # ---- derived standard-spec peaks ----
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops  # ~1307 TF
        self.int8_tensor_flops = self.fp16_tensor_flops          # INT8 = FP16 rate on CDNA3
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2          # ~163 TF
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2          # same as FP32 on CDNA3 vector
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5    # ~81.7 TF
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle          # ~81.7 TB/s
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

    def set_to_spec(self):
        """Theoretical peak (datasheet) with utilization = 1.0."""
        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        self.base_freq = 2.1 * 1e9
        self.max_freq = 2.1 * 1e9

        self.ddr_bandwidth = 6000 * 1e9                                                                    # 6.0 TB/s
        self.l2_bandwidth = 16 * 256 * self.max_freq                                                      # ~28 TB/s
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle            # ~81.7 TB/s

        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops  # ~1307 TF
        self.int8_tensor_flops = self.fp16_tensor_flops
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2             # ~163 TF
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2             # ~163 TF
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5       # ~81.7 TF
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2

        return self

    def set_to_microbench(self):
        """Apply the measured MI325X calibration values."""
        # measured effective clock: ~2094 MHz (nop) / ~2063 MHz (fma pressure)
        self.base_freq = 2.094 * 1e9
        self.max_freq = 2.094 * 1e9

        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        # DRAM: read=3972.6 / write=3967.6 / copy=4243.9 GB/s.
        # Use sustained read as the effective HBM bandwidth (bandwidth-bound inference).
        self.ddr_bandwidth = 3972.6 * 1e9

        # L2: dedicated stream test 16750.6 GB/s.
        self.l2_bandwidth = 16750.6 * 1e9

        # LDS/shared memory: whole-chip 50160 GB/s (hardware peak 81715 GB/s, 61.4%).
        self.smem_bandwidth = 50160.0 * 1e9

        # CUDA-core (vector) throughput.
        self.fp32_cuda_core_flops = 120.19 * 1e12   # FP32 FMAC min
        self.fp16_cuda_core_flops = 138.22 * 1e12   # FP16 FMAC
        self.fp64_cuda_core_flops = 63.59 * 1e12    # FP64 FMAC

        # MFMA fp16: 4-wave/CU bench (blocks=2432) = 1224.2 TF min.
        # Full-occupancy sweep (32 waves/CU, blocks=9728) peaks at 1241 TF.
        # Use sweep peak as the model value (matches real GEMM occupancy better).
        self.fp16_tensor_flops = 1241.0 * 1e12
        self.int8_tensor_flops = self.fp16_tensor_flops    # INT8 = FP16 rate on CDNA3

        # SFU: exp2f throughput = 4.51 TOPS
        self.sfu_flops = 4.51 * 1e12

        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        return self


if __name__ == "__main__":
    arch = MI325X()
    print("== default standard spec ==")
    print(f"fp16_tensor_flops  (TF):  {arch.fp16_tensor_flops / 1e12:.1f}")
    print(f"int8_tensor_flops  (TOPS):{arch.int8_tensor_flops / 1e12:.1f}")
    print(f"fp32_cuda_core (TF):      {arch.fp32_cuda_core_flops / 1e12:.1f}")
    print(f"fp16_cuda_core (TF):      {arch.fp16_cuda_core_flops / 1e12:.1f}")
    print(f"fp64_cuda_core (TF):      {arch.fp64_cuda_core_flops / 1e12:.1f}")
    print(f"l2_bandwidth   (TB/s):    {arch.l2_bandwidth / 1e12:.2f}")
    print(f"smem_bandwidth (TB/s):    {arch.smem_bandwidth / 1e12:.1f}")
    print(f"ddr_bandwidth  (TB/s):    {arch.ddr_bandwidth / 1e12:.3f}")

    arch.set_to_microbench()
    print("== microbench ==")
    print(f"fp16_tensor_flops (TF):   {arch.fp16_tensor_flops / 1e12:.1f}")
    print(f"fp32_cuda_core (TF):      {arch.fp32_cuda_core_flops / 1e12:.2f}")
    print(f"fp64_cuda_core (TF):      {arch.fp64_cuda_core_flops / 1e12:.2f}")
    print(f"fp16_cuda_core (TF):      {arch.fp16_cuda_core_flops / 1e12:.2f}")
    print(f"ddr_bandwidth  (TB/s):    {arch.ddr_bandwidth / 1e12:.3f}")
    print(f"l2_bandwidth   (TB/s):    {arch.l2_bandwidth / 1e12:.3f}")
    print(f"smem_bandwidth (TB/s):    {arch.smem_bandwidth / 1e12:.1f}")


# =============================================================================
# REFERENCE: representative MI325X topology summary (gfx942)
#
# Representative device properties:
#   name:                    AMD Instinct MI325X
#   clockRate:               2100 MHz
#   memoryClockRate:         1500 MHz
#   memoryBusWidth:          8192 bits
#   totalGlobalMem:          255.984 GiB
#   l2CacheSize:             4 MiB  (per hipDeviceProp; full chip ~256 MiB L3)
#   sharedMemPerBlock:       64 KiB
#   sharedMemPerMultiprocessor: 19456 KiB
#   multiProcessorCount:     304
#   maxThreadsPerMultiProcessor: 2048
#   warpSize:                64
#
# Omniperf topology (gfx942, NPS1):
#   Shader Engines:          32
#   Total L2 Channels:       16
#   Max Waves Per CU:        32
#   L1 cache:                32 KiB (vL1D per CU)
#   L2 cache:                4096 KiB  (per hipDeviceProp partition)
#   L3 cache:                262144 KiB = 256 MiB (total HBM3E-backed cache)
#   LDS Banks per CU:        32
# =============================================================================
