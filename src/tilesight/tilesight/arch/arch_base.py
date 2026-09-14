"""Common architecture state and source-visible modeling capabilities."""


_GPU_RESOURCE_MODEL_CORES = frozenset(("A100", "H100", "B200"))


def uses_gpu_resource_model(arch, *, include_lut=False):
    """Return whether ``arch`` uses the GPU tensor-resource equations.

    Caller-defined architectures opt in through
    ``use_tensor_core_resource_model``. Public baseline GPU profiles retain
    their existing behavior.
    """

    if bool(getattr(arch, "use_tensor_core_resource_model", False)):
        return True
    core = getattr(arch, "core", None)
    if core in _GPU_RESOURCE_MODEL_CORES:
        return True
    return bool(include_lut and core == "A100_LUT")


def uses_dram_wave_quantization(arch):
    """Return whether DRAM transfers are rounded to architecture wave size."""

    return bool(getattr(arch, "apply_dram_wave_quantization", False))


class Arch:
    def __init__(self):
        self.core = None
        self.sm_count = 0
        self.freq = 0.0
        self.tensor_cores_per_sm = 0
        self.tensor_core_shape = (0, 0, 0)
        self.tensor_core_flops = 0
        self.fp32_cores_per_sm = 0
        self.fp64_cores_per_sm = 0
        self.fp16_cores_per_sm = 0
        self.ddr_bandwidth = 0.0
        self.ddr_capacity = 0
        self.l2_bandwidth = 0.0
        self.l2_capacity = 0
        self.sm_sub_partitions = 0
        self.l1_smem_throughput_per_cycle = 0
        self.configurable_smem_capacity = 0
        self.register_capacity_per_sm = 0
        self.warp_schedulers_per_sm = 0
        self.fp16_tensor_flops = 0.0
        self.fp32_cuda_core_flops = 0.0
        self.smem_bandwidth = 0
        self.register_bandwidth = 0
        # self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        # self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

        self.max_blocks_per_sm = 24  # hardware limit, overridden per arch
        self.use_tensor_core_resource_model = False
        self.apply_dram_wave_quantization = False

        # L1.5 cache parameters (passive per-group cache between SMEM and L2)
        # l1_5_group_size=0 means L1.5 is not present (backward compatible)
        self.l1_5_group_size = 0           # SMs per L1.5 group (0=disabled)
        self.l1_5_capacity_per_group = 0   # bytes, per group
        self.l1_5_bandwidth = 0.0          # bytes/s, whole chip peak
        self.l1_5_associativity = 8        # same as L2 default
        self.l1_5_cacheline_bytes = 128    # same as L2 default
        self.l1_5_max_util = 0.9

        # Tensor Memory (TMEM) parameters (Blackwell SM100+ tcgen05 ld/st datapath).
        # tmem_bandwidth=0 means TMEM is not present (backward compatible: pre-Blackwell
        # archs have no TMEM channel, so tmem_io contributes 0 time).
        self.tmem_capacity_per_sm = 0      # bytes per SM (256KB on SM100: 128 lanes x 512 cols x 4B)
        self.tmem_bandwidth = 0.0          # bytes/s, whole chip peak (tcgen05.ld/st datapath)
        self.tmem_max_util = 0.9



    # ... other methods as needed
