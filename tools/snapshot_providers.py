"""Export comparable provider integration snapshots without GPU dependencies."""
import sys, json, math
from pathlib import Path
root=Path(sys.argv[1])
sys.path[:0]=[str(root/'src/deepstack'),str(root/'src/tilesight')]
from mosaic.arch import _reference_model as r
from mosaic.arch.stacked_gpu_base import stacked_gpu_base
from mosaic.noc import _model_support as n
from mosaic.cost import _capacity as c
fields=json.loads((Path(__file__).parents[1]/'src/deepstack/mosaic/arch/recovered_reference_profiles.json').read_text())['standard']['fields']

def snapshot(arch): return {k:getattr(arch,k) for k in fields}
results={}
for profile in ['standard','large_matrix','large_vector','high_l1','high_l2','high_noc','low_noc','reduced_sm','wgmma','h100_scaled','h200_scaled']:
    a=stacked_gpu_base()
    (r.q02 if profile.endswith('scaled') else r.q01)(a,profile)
    results['profile/'+profile]=snapshot(a)
    for width in [1,4,8,16,32]:
        results['profile_sm/'+profile+'/'+str(width)]=snapshot(a.with_sm_count(width))
        results['profile_cluster/'+profile+'/'+str(width)]=snapshot(a.with_cluster_width(width))
    for total in [1,4,8]:
        results['profile_dram/'+profile+'/'+str(total)]=snapshot(a.with_dram(total))
        results['profile_design/'+profile+'/'+str(total)]=snapshot(a.with_design_point(sm_count=4,total_layers=total,active_layers=1,smem_capacity=131072,l1_throughput_bpc=128))
for width in [1,4,8,16,32]:
    a=stacked_gpu_base()
    results['sm/'+str(width)]=snapshot(a.with_sm_count(width))
    results['cluster/'+str(width)]=snapshot(a.with_cluster_width(width))
for total in [1,2,4,8,16]:
    a=stacked_gpu_base()
    results['dram/'+str(total)]=snapshot(a.with_dram(total))
    results['design/'+str(total)]=snapshot(a.with_design_point(sm_count=4,total_layers=total,active_layers=1,smem_capacity=131072,l1_throughput_bpc=128))
for i in range(49):
    try:
        h=n.p00(i)
        results['noc/'+str(i)]=[{'kind':l.kind.value,'shape':l.shape,'hop_latency':l.hop_latency,'link_bandwidth':l.link_bandwidth,'in':l.switch_center_in_bw,'out':l.switch_center_out_bw} for l in h.layers]
        try: results['capacity/'+str(i)]=c.p00(stacked_gpu_base(),h)
        except Exception as e: results['capacity/'+str(i)]={'error':type(e).__name__,'message':str(e)}
    except Exception as e: results['noc/'+str(i)]={'error':type(e).__name__,'message':str(e)}
Path(sys.argv[2]).write_text(json.dumps(results,indent=2))
