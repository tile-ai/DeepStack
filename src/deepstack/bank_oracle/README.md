# Standalone bank oracle

This package is a source-visible validation oracle for bank-aware memory
scheduling.  It is intentionally separate from the coarse performance model:
its purpose is to construct and inspect schedules, lower bounds, and
layout/swizzle witnesses.

The bundled defaults and demos use an explicitly **synthetic toy
configuration**.  They are deterministic test inputs, not a hardware
calibration and not a claim about a particular product.  Reviewers can replace
every physical field through `DramBankSpec`.

## Reviewer value

The oracle answers a narrow but useful question: is a topology-level analytic
limit merely a formula, or can an explicit bank/row/port schedule attain it?

For a supplied configuration, the implementation:

- decodes every requested sector into `(bank, row, sector-in-row)`;
- coalesces duplicate sectors into configurable-width row masks;
- schedules row-data jobs on physical banks and shared transfer ports;
- computes fixed-job and freely-repacked sector lower bounds; and
- reports whether the explicit schedule attains those modeled bounds.

The synthetic full-row and swizzled traces are constructive witnesses: the
dashboard exposes every row job, and the tests independently check that their
scheduled time equals the applicable model bound.  This establishes
reachability **inside the stated analytical model**.  It does not claim that a
kernel, controller, or physical device must attain the same value.

## Configure the physical model

```python
from bank_oracle import DramBankSpec

spec = DramBankSpec(
    total_layers=6,
    connected_layers=3,
    banks_per_layer=5,
    row_bytes=1536,
    sector_bytes=96,
    sector_cycles=5,
    recharge_cycles=9,
    data_rate_hz=1.25e9,
    round_trip_latency_cycles=23,
    latency_clock_hz=1.75e9,
)
```

These numbers are another synthetic example.  The interface accepts:

| Field | Meaning |
|---|---|
| `total_layers` | physical row-buffer layers |
| `connected_layers` | independently transferring layers per bank column |
| `banks_per_layer` | bank columns in each layer |
| `row_bytes`, `sector_bytes` | row geometry; 1--64 sectors per row |
| `sector_cycles`, `recharge_cycles` | data and row-recharge service time |
| `data_rate_hz` | clock domain used by bank service cycles |
| `round_trip_latency_cycles`, `latency_clock_hz` | independent latency domain |

`DramBankSpec.from_arch()` copies only matching `dram_*` physical attributes
and accepts explicit overrides for every field.  It does not infer a data rate
from another clock or consume a pre-scaled effective-bandwidth value.

Sector masks use unsigned 64-bit storage, so arbitrary row geometries from one
through 64 sectors are supported.  XOR swizzles still require power-of-two
bank and sector fields because they are bit permutations.  Cyclic swizzles
cover non-power-of-two bank counts.

## Model components

| Mechanism | Implementation |
|---|---|
| Transaction cost | exact row mask and `recharge + touched_sectors * sector_cycles` |
| Finite bank parallelism | physical-bank histogram and critical path |
| Partial connectivity | separate physical-bank and transfer-port counts |
| Layout conflict | linear/padded/tile-major layouts and reversible XOR/cyclic swizzles |
| Latency window | operator-level Little's-Law floor from explicit outstanding bytes |
| Cache filtering | deterministic fully-associative FIFO before bank decode |
| Pipeline | prologue/steady/epilogue overlap composition |
| Multi-cluster | configurable placement, owner-private L2/DRAM, and NoC accounting |

The optional multi-cluster wrapper uses an exact
`ClusterGrid(P_M, P_N, P_K)` and caller-supplied `ClusterUnitSpec`. It supports
sharded, replicated, and canonical tensor placement; owner-private L2/DRAM;
configurable NoC service; generic distributed GEMM; and square-grid Cannon.
`P_K > 1` ends at explicitly marked partial C unless the caller models the
required reduction. The default overlap is a stage-level analytical bound,
not a transaction-timeline or coherence simulator. All cluster, cache, NoC,
placement, and physical-address inputs remain caller controlled.

## Run the focused validation

From the repository root:

```bash
PYTHONPATH="$PWD/src/deepstack:$PWD/src/tilesight:$PWD" \
python -m pytest -q src/deepstack/bank_oracle/tests

PYTHONPATH="$PWD/src/deepstack:$PWD/src/tilesight:$PWD" \
python -m bank_oracle.validate_conflict_free
```

Generate the self-contained dashboard:

```bash
PYTHONPATH="$PWD/src/deepstack:$PWD/src/tilesight:$PWD" \
python -m bank_oracle.visualization.generate_demo \
  --output results/bank_oracle/bank_wave_demo.html
```

The committed HTML is a convenience preview.  It has no CDN, network request,
or runtime package dependency. It exposes row jobs, bank/row/sector mappings,
variable-width masks, topology views, and scheduled-time versus lower-bound
checks. Its title and payload label the configuration as synthetic.

## Boundaries

- The oracle validates its own configured analytical assumptions; it is not a
  cycle-accurate controller simulator.
- The ready/LPT shared-port scheduler is deterministic.
- Cache sets, associativity, dirty writeback, and coherence are outside the
  single-cluster FIFO abstraction.
- Representative sampling is a deterministic estimate.  Set spatial and
  K-sample limits to `None` when an exact enumeration is required.
- A bound-attaining row schedule is evidence that the modeled upper limit is
  constructively reachable, not that end-to-end execution has no other limit.
