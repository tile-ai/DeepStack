# TileSight source snapshot

This directory contains the source-visible TileSight components used by the
DeepStack analytical model. The artifact imports them directly from
`src/tilesight/tilesight/`; no external TileSight checkout is required.

The main artifact-facing areas are:

- `arch/`: NVIDIA, AMD, and other architecture descriptions;
- `distributed/`: devices, parallelism, traffic, topology, and distributed
  operator composition;
- `fused_op_pipeline_wave/`: wave/resource models for fused operators; and
- `tile_cache/`: deterministic cache and reuse-distance estimators.

The release is pinned as `tilesight-release-2026-07`. The proprietary
`distributed/noc/_model_support*.so` supplies only the reference NoC
topology/profile boundary described in the root `LICENSING.md`; the surrounding
Python source remains visible. Reviewer commands and accepted outputs are
documented in the root `README.md` and `docs/result_matrix.md`.
