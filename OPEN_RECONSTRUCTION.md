# DeepStack source reconstruction ? work in progress

This local fork replaces the three provider extensions with Python modules.
No original `.so` is present in this fork. The original repository is unchanged.
The recovered reference values and capacity data are included with provenance;
this work does not assign a new permissive license to recovered upstream data.
Existing license and notice documents are retained. No public release is made.

## Current evidence

- 11,877 integrated comparisons against a separate native-provider process: zero differences within 2e-14 relative tolerance.
- Upstream tests: 162 passed under Python 3.12, Windows, CPU-only PyTorch.
- Fig08 subset baseline: 44 archived prediction/measurement pairs, MAPE 9.04248%, maximum APE 60.0310%. This excludes the EP dispatch column variant and is not the full paper aggregate.
- Full 52-point Fig08 source-model rerun passed paper-statistics verification: MAPE 8.366985%; AG-GEMM 3.974068%. See results/reproduce/stages/fig08/verification.json.
- Exploratory leave-one-out per-category latency offsets achieved 7.710054% MAPE, so this correction was rejected and is not part of the predictor.

The 5% hardware-accuracy target is not achieved yet. Provider equivalence is not hardware accuracy. A successful reproduction of paper statistics does not mean MAPE < 5%.

## Replay

```text
python -m pytest
python tools/evaluate_paper_baseline.py
python tools/snapshot_providers.py <original-root> validation/native_snapshots.json
python tools/snapshot_providers.py . validation/source_snapshots.json
python tools/compare_snapshots.py
python -m ae.fig08 --no-plot
```

The original-root snapshot needs Linux CPython 3.11 for the unmodified native oracle. The source implementation does not need those extensions.

The pure Python wheel builds and installs; installed capacity smoke returns 8. Wheel inspection confirms zero .so/.pyd files.

Remaining gates: exhaustive clone/API integration, reproducing the remaining paper figures beyond Fig08, held-out accuracy below 5%, and source/data licensing review before any release.
