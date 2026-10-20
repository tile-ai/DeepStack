"""Independent recovery of the _capacity canonical 19-word key.

Takes an ordinary architecture object exposing the named raw hardware fields,
and a hierarchy with exactly three layers. Does not load any native provider.
"""
import math
import struct

MAX_EXACT = 2**53 - 1
ERROR = 'invalid protected configuration'

def positive_integer(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(ERROR) from None
    if not math.isfinite(number) or not 1 <= number <= MAX_EXACT or math.floor(number) != number:
        raise ValueError(ERROR)
    return int(number)

def bandwidth_integer(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(ERROR) from None
    if not math.isfinite(number) or not 0 < number <= MAX_EXACT:
        raise ValueError(ERROR)
    rounded = math.floor(number + 0.5)
    # The binary accepts an integer rounded within one upward ULP.
    if abs(number - rounded) > math.nextafter(number, number + number) - number:
        raise ValueError(ERROR)
    return rounded

def canonical_words(architecture, hierarchy):
    a = architecture
    try:
        shape = a.tensor_core_shape
        layers = hierarchy.layers
    except (AttributeError, TypeError):
        raise TypeError(ERROR) from None
    if len(shape) != 3 or len(layers) != 3:
        raise ValueError(ERROR)
    optional = getattr(a, 'xbar_quadratic_ref_sm', None)
    values = [positive_integer(a.register_capacity_per_sm)]
    values.extend(positive_integer(shape[i]) for i in range(3))
    values.append(int(bool(a.support_wgmma)))
    for name in ['configurable_smem_capacity', 'l1_smem_throughput_per_cycle',
                 'int32_cores_per_sm', 'fp32_cores_per_sm', 'sfu_cores_per_sm',
                 'tensor_cores_per_sm']:
        values.append(positive_integer(getattr(a, name)))
    values.append(positive_integer(optional) if optional else 0)
    for name in ['dram_layers_per_cluster', 'dram_active_layers', 'l2_capacity', 'l2_bandwidth']:
        values.append(positive_integer(getattr(a, name)))
    values.extend(bandwidth_integer(layers[-i].link_bandwidth) for i in range(1, 4))
    return tuple(values)

def canonical_bytes(architecture, hierarchy, domain=0x44534341504c5551):
    return struct.pack('<20Q', domain, *canonical_words(architecture, hierarchy))
