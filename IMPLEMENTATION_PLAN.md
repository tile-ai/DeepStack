# Completion gates

User objective: complete a source-based DeepStack and reproduce results below 5% error. Treat full 52-point Fig08 hardware-reference MAPE as the initial accuracy target pending user clarification. Do not narrow this to AG-GEMM alone or source/native output parity.

Completed:
- Isolated local fork, original repository unchanged.
- Three source adapters, recovered profile/NoC tables and capacity LUT packaged.
- Actual upstream routing Topology/Hierarchy objects reconstructed.
- 11,877 integrated native/source comparisons, zero mismatches.
- 162 upstream tests passed.
- Full Fig08 workload recomputation (52 points), paper-statistic verification passed.
- Pure Python wheel build, zero native binaries; installed capacity smoke passed before latest cluster-only correction.

Incomplete:
1. Broaden native/source differential coverage to API exceptions, mutation/copy semantics, state sequences, and all DSE entry points. Update source wrappers as differences are discovered.
2. Reproduce other available AE figures using source providers. Record missing workloads/data explicitly.
3. Achieve and independently validate full hardware-reference MAPE <5%; current Fig08 MAPE 8.366985%, maximum error >60%. AG-GEMM alone <5% is insufficient.
4. Investigate RS GEMM low-wave performance and Ulysses communication/computation resource sharing. A per-category constant latency correction was rejected (leave-one-out MAPE 7.710054%). No fitted corrections are shipped.
5. Freeze calibration and use workload-group holdout/new hardware measurement. Existing paper-data exploration is not blind validation.
6. Verify rebuilt wheel after final corrections and record SHA256. Source/data license provenance must remain explicit; local reconstruction is not proof of permissive redistribution rights.

Artifacts: OPEN_RECONSTRUCTION.md, validation/*.json, results/reproduce/stages/fig08/*.csv and verification.json. No commit/push/publication requested or performed.
