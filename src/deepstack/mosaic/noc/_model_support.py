"""Source provider backed by the recovered numerical contracts."""
from . import recovered_model_support as recovered
from .noc_topo import Hierarchy, PortSpread, TopoKind, make_switch, make_ring, make_chain, make_all2all, make_mesh_or_torus

def _routing(value):
    layers = []
    for layer in value.layers:
        kw = dict(hop_latency=layer.hop_latency, link_bandwidth=layer.link_bandwidth)
        m, n = layer.shape
        if layer.kind == 'switch':
            obj = make_switch(n, **kw)
            obj.shape = (m, n)
            obj.switch_center_in_bw = layer.switch_center_in_bw
            obj.switch_center_out_bw = layer.switch_center_out_bw
        elif layer.kind == 'ring':
            obj = make_ring(m*n, **kw)
        elif layer.kind == 'chain':
            obj = make_chain(m*n, **kw)
        elif layer.kind == 'all_to_all':
            obj = make_all2all(m, n, **kw)
        else:
            obj = make_mesh_or_torus(m, n, TopoKind(layer.kind), **kw)
        layers.append(obj)
    return Hierarchy(layers, PortSpread(value.port_spread), value.name or 'default_noc_hierarchy')

def p00(profile_id):
    return _routing(recovered.p00(profile_id))

def p01(variant, num_nodes):
    return _routing(recovered.p01(variant, num_nodes))

p93 = recovered.p93
p94 = recovered.p94
p95 = recovered.p95
p96 = recovered.p96
p97 = recovered.p97
p98 = recovered.p98
