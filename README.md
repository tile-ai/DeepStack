# DeepStack

<p align="center">
  <img src="docs/badges/acm-artifacts-available-v1.1.jpg" height="110" alt="ACM Artifacts Available">
  <img src="docs/badges/acm-artifacts-evaluated-functional-v1.1.jpg" height="110" alt="ACM Artifacts Evaluated — Functional">
  <img src="docs/badges/acm-results-reproduced-v1.1.jpg" height="110" alt="ACM Results Reproduced">
</p>

Analytical modeling for GPU and 3D-stacked LLM inference.

Paper: [DeepStack: Facilitating Co-Design Exploration of 3D DRAM-Stacked Accelerators for Distributed LLM Inference](https://arxiv.org/abs/2604.04750).

`main` provides the latest version of DeepStack, with ongoing bug fixes and
model updates, such as improved energy modeling for computations on zero-padded
inputs. We recommend `main` for using and extending DeepStack. These updates
may produce results that differ from those reported in the paper.

The [`ae`](https://github.com/tile-ai/DeepStack/tree/ae) branch preserves the
model snapshot and evaluation workflow from the paper submission. Use it to
reproduce the paper's results. The badges above refer to this evaluated artifact.

## Install

Use CPython 3.11 on x86-64 Linux. Modeling runs on CPU; no GPU or model weights
are needed.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Custom accelerator configuration

[`examples/custom_accelerator.py`](examples/custom_accelerator.py) shows how
to change accelerator memory bandwidth, interconnect settings and parallelism.
The examples use NVIDIA GPU configurations as a reference. The concrete
hardware and workload are defined in the script. Each run prints latency
and system throughput.

1. Run the baseline:
   ```bash
   python examples/custom_accelerator.py
   ```
2. Change per-GPU HBM bandwidth (decimal GB/s):
   ```bash
   python examples/custom_accelerator.py --hbm-bandwidth-gbyteps 2680
   ```
3. Change interconnect bandwidth and per-hop latency:
   ```bash
   python examples/custom_accelerator.py --noc-bandwidth-gbyteps 225 --noc-latency-ns 2000
   ```
4. Compare tensor/data parallelism on the same eight GPUs:
   ```bash
   python examples/custom_accelerator.py --tp 4 --dp 2
   ```

Keep `TP × DP = 8` for this example. `--batch-size` changes the global batch
size. In the script,
`chip.ddr_bandwidth` controls memory bandwidth, `make_custom_profile` builds
the NoC, and `ParallelScheme` selects parallelism.

For hypothetical 3D DRAM designs, use `DramInterfaceConfig` and
`StackedGpuConfig` in [`mosaic.arch.custom_profile`](src/deepstack/mosaic/arch/custom_profile.py).
`total_layers`, `connected_layers`, and `capacity_per_layer_bytes` describe
a custom stacked-memory architecture.
A complete caller-defined configuration is in
[`tests/test_custom_arch_profile.py`](tests/test_custom_arch_profile.py).

## Single-configuration runs and DSE

[`examples/custom_dse.py`](examples/custom_dse.py) runs a single configuration
or a small parallelism search. Its [network configuration](src/deepstack/mosaic/noc/noc_config_set.py)
is ordinary Python, with four nodes and eight devices per node. Edit that
function or override its bandwidth arguments to try another setup.

```bash
# Run one configuration on the example's 32-device system.
python examples/custom_dse.py --tp 8 --pp 4 --dp 1
# Change the inter-node bandwidth.
python examples/custom_dse.py --ib-bandwidth-gbyteps 25
# Search 15 TP/PP/DP combinations and save all results.
python examples/custom_dse.py --dse --output results/custom_dse.csv
```

The single run and search use the same `modeling_decode` entry point as the
decode DSE driver. `TP * PP * DP` stays at 32. `--batch-size` is the global
batch, and microbatch size is `ceil(batch_size / PP)`, following that driver.
STPS is microbatch size divided by the modeled step time. The search reports
estimated per-GPU memory, skips OOM configurations, and sorts feasible results
by STPS. It does not perform kernel tile autotuning or hardware benchmarking.

## Custom NoC topologies

The H200 switch configuration is [`h200x32()`](src/deepstack/mosaic/noc/noc_config_set.py).
[`examples/custom_noc.py`](examples/custom_noc.py) adds ring and mixed
torus/mesh/switch examples using the existing `make_*` constructors:

| Configuration | L3 (outermost) | L2 | L1 (innermost) | Devices |
|---|---|---|---|---|
| `h200x32()` | Singleton switch | 4-port switch | 8-port switch | 32 |
| `ring_switch_32()` | Singleton switch | 4-node ring | 8-port switch | 32 |
| `torus_mesh_switch_64()` | 2 × 2 torus | 2 × 2 mesh | 4-port switch | 64 |

```bash
python examples/custom_noc.py
```

Each function returns `Hierarchy(layers=[L3, L2, L1], ...)`. The device count
is the product of all layer sizes. `make_mesh_or_torus(..., TopoKind.MESH2D)`
builds a mesh; `TopoKind.TORUS2D` enables wraparound links. Other constructors
include `make_ring`, `make_chain`, `make_switch` and `make_all2all`, defined
in [`mosaic.noc.noc_topo`](src/deepstack/mosaic/noc/noc_topo.py).

Edit the functions to change shapes, per-link bandwidth (bytes/s) and
per-hop latency (seconds). The ring and torus examples use illustrative
bandwidth, latency and energy inputs. Pass the returned hierarchy as
`noc_hierarchy` to `modeling_decode` or `modeling_prefill`, with parallelism
matching its device count. The DSE example uses `h200x32()` by default.

## Collectives

The wrappers in [`mosaic.collectives`](src/deepstack/mosaic/collectives) price
one collective on any `Hierarchy` and return `(hop_latency_s, link_time_s,
traffic)` for the fastest algorithm they model. `bytes` is the full tensor on
each device: the buffer an all-reduce reduces, the input a reduce-scatter
splits, or the output an all-gather assembles. Ranks are grouped TP, EP, SP,
CP, DP, PP from the innermost NoC level outwards, so a TP group shares the
innermost switch when it fits there.

```python
from mosaic.collectives import all_gather_wrapper, all_reduce_wrapper, reduce_scatter_wrapper
from mosaic.noc.noc_topo import Hierarchy, PortSpread, make_switch
from mosaic.parallelism import ParallelScheme
from mosaic.utils import Modeling_Granularity

# Four switched nodes of eight devices; bandwidth (bytes/s) and latency (s) are illustrative.
nodes = Hierarchy(
    layers=[make_switch(1, hop_latency=0.0, link_bandwidth=50e9),
            make_switch(4, hop_latency=5e-6, link_bandwidth=50e9),
            make_switch(8, hop_latency=1e-6, link_bandwidth=450e9)],
    port_spread=PortSpread.EVEN, name="4x8",
)
parallel = ParallelScheme(tp=8, dp=4)
granularity = Modeling_Granularity("coarse", True, False)
for wrapper in (all_reduce_wrapper, reduce_scatter_wrapper, all_gather_wrapper):
    hop_s, link_s, _ = wrapper(None, parallel, nodes, granularity, "tp", 64 * 2**20)
    print(f"{wrapper.__name__}: {(hop_s + link_s) * 1e6:.1f} us")
```

## Routing data

The selected routing arrays are packaged with the model:

```python
from mosaic.data.routing import load_routing
prefill, decode = load_routing("qwen3_235b")  # or "deepseek_v3"
```

## Licensing

Source code is Apache-2.0. Three bundled native model libraries have a separate
license; see [LICENSING.md](LICENSING.md).
