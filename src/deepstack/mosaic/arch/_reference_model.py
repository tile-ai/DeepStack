"""Source reconstruction of the reference provider; integration under validation."""
import copy
import json
from pathlib import Path
from . import recovered_reference_formulas as f

_CANONICAL = json.loads(Path(__file__).with_name('capacity_canonical_profiles.json').read_text())['architectures']
_PROFILE_ARCH = {'standard':'stacked_gpu_base', **{n:'stacked_gpu_'+n for n in ('large_matrix','large_vector','high_l1','high_l2','high_noc','low_noc','reduced_sm','wgmma')}, 'h100_scaled':'H100_SCALED','h200_scaled':'H200_SCALED'}

def model_support_api_version(): return 3

def is_reference_arch(value): return isinstance(value, ReferenceArch)

def _apply(arch, name):
    state = f.reference_profile(name)
    state.update({k:v for k,v in _CANONICAL.get(_PROFILE_ARCH[name], {}).items() if v is not None})
    arch.__dict__.update(state)
    arch._profile_name = name

def q01(arch, profile_name): _apply(arch, profile_name)
def q02(arch, profile_name): _apply(arch, profile_name)
def q03(arch): pass
q04 = f.q04

def q05(arch, total_layers, active_layers=None):
    arch.__dict__.update(f.dram_layer_rates(total_layers, active_layers))

def q06(arch, cluster_width):
    changes = f.cluster_scale(cluster_width)
    ratio = changes['sm_count'] / arch.sm_count
    for key in tuple(arch.__dict__):
        if key.endswith('_flops') or key in ('smem_bandwidth', 'register_bandwidth', 'smem_l2_bandwidth', 'smem_register_bandwidth', 'smem_to_l2_bandwidth', 'l2_to_smem_bandwidth', 'ddr_bandwidth', 'ddr_peak_bandwidth', 'l2_bandwidth'):
            arch.__dict__[key] *= ratio
    scale = changes.pop('aggregate_scale')
    for key in ('ddr_bandwidth', 'ddr_peak_bandwidth', 'l2_bandwidth'):
        changes[key] = f.dram_layer_rates(arch.dram_layers_per_cluster, arch.dram_active_layers)[key] * scale
    changes['ddr_wave_bytes'] = arch.ddr_transaction_size * 2 * changes['sm_count']
    arch.__dict__.update(changes)
    arch.layer1_noc_size = arch.sm_count

def q07(arch, total_layers, active_layers):
    q05(arch, total_layers, active_layers)
    arch.ddr_capacity = float(2**31) * int(total_layers)

def q08(arch):
    required = f.little_required_buffer(arch.ddr_bandwidth, arch.sm_count)
    return required > arch.configurable_smem_capacity, required

q09 = f.q09_thermal_scale
q11 = f.q11_power_wall
q12 = f.q12
q13 = f.q13

class ReferenceArch:
    def __init__(self, profile_name='standard', *, public_name=None, scaled=False):
        _apply(self, profile_name)
        self._public_name = public_name or type(self).__name__

    def get_tensor_core_minimum_ptx(self, width_bytes=2):
        return q04(self._profile_name, width_bytes)

    row_reduce_time = staticmethod(f.row_reduce_time)

    def update_derived(self): return q03(self)
    def update_ddr(self, total_layers, active_layers=None): return q05(self, total_layers, active_layers)

    def with_sm_count(self, sm_count):
        clone = copy.copy(self)
        n = int(sm_count)
        if n <= 0: raise ValueError('sm_count must be positive')
        ratio = n / clone.sm_count
        for key in tuple(clone.__dict__):
            if key.endswith('_flops') or key in ('smem_bandwidth', 'register_bandwidth', 'smem_l2_bandwidth', 'smem_register_bandwidth', 'smem_to_l2_bandwidth', 'l2_to_smem_bandwidth'):
                clone.__dict__[key] *= ratio
        clone.sm_count = n
        clone.layer1_noc_size = n
        return clone

    def with_cluster_width(self, cluster_width):
        clone = copy.copy(self)
        q06(clone, cluster_width)
        return clone

    def with_dram(self, total_layers, active_layers=None):
        clone = copy.copy(self)
        q07(clone, total_layers, total_layers if active_layers is None else active_layers)
        return clone

    def with_thermal_scale(self, thermal_scale):
        if float(thermal_scale) <= 0: raise ValueError('thermal_scale must be positive')
        return copy.copy(self)

    def with_littles_law(self):
        clone = copy.copy(self)
        _, _, clone.ddr_bandwidth = f.little_law(clone.ddr_bandwidth, clone.sm_count, clone.configurable_smem_capacity)
        return clone, *q08(clone)

    def with_design_point(self, *, sm_count=None, total_layers=None, active_layers=None, smem_capacity=None, l1_throughput_bpc=None, thermal_scale=None):
        clone = copy.copy(self)
        if sm_count is not None: clone = clone.with_sm_count(sm_count)
        if total_layers is not None: q07(clone, total_layers, total_layers if active_layers is None else active_layers)
        elif active_layers is not None: raise ValueError('active_layers requires total_layers')
        for value, field in ((smem_capacity, 'configurable_smem_capacity'), (l1_throughput_bpc, 'l1_smem_throughput_per_cycle')):
            if value is not None:
                if int(value) <= 0: raise ValueError(field + ' must be positive')
                if field == 'l1_smem_throughput_per_cycle':
                    ratio = int(value) / clone.l1_smem_throughput_per_cycle
                    for bandwidth in ('smem_bandwidth', 'smem_l2_bandwidth', 'smem_register_bandwidth', 'smem_to_l2_bandwidth', 'l2_to_smem_bandwidth'):
                        clone.__dict__[bandwidth] *= ratio
                setattr(clone, field, int(value))
        if thermal_scale is not None: clone = clone.with_thermal_scale(thermal_scale)
        return clone
