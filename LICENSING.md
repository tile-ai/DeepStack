# Licensing map

This repository is an open-source framework with proprietary binary
components. It is a mixed-license reproducibility artifact, not a wholly
open-source distribution and not a dual-license offer.

This file is a scope map. The controlling license texts are linked below.

| Component | Scope | License |
|---|---|---|
| Artifact source, runners, scripts, docs, manifests, and data | All repository paths except the rows below, licensing materials, or files with their own headers | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| Scope notices and license metadata | `LICENSE`, `LICENSING.md`, `LICENSES/*`, and adjacent `.so.license` files | Verbatim copying under their respective controlling terms |
| Reference architecture/DSE-policy provider | `src/deepstack/mosaic/arch/_reference_model*.so` | [LicenseRef-DeepStack-AE-Binary-1.0](LICENSES/LicenseRef-DeepStack-AE-Binary-1.0.txt) |
| Reference SM-capacity provider | `src/deepstack/mosaic/cost/_capacity*.so` | [LicenseRef-DeepStack-AE-Binary-1.0](LICENSES/LicenseRef-DeepStack-AE-Binary-1.0.txt) |
| DeepStack NoC-profile and chip/NoC-energy provider | `src/deepstack/mosaic/noc/_model_support*.so` | [LicenseRef-DeepStack-AE-Binary-1.0](LICENSES/LicenseRef-DeepStack-AE-Binary-1.0.txt) |
| TileSight NoC-topology/profile provider | `src/tilesight/tilesight/distributed/noc/_model_support*.so` | [LicenseRef-DeepStack-AE-Binary-1.0](LICENSES/LicenseRef-DeepStack-AE-Binary-1.0.txt) |
| Upstream model-description files | Paths listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Their retained Apache-2.0 headers |
| Environment dependencies | Installed by Conda/pip and not redistributed in this repository | Their respective licenses |

The Apache-2.0 metadata in `pyproject.toml` applies only to the source-only
`ae*` Python package selected there. The four proprietary shared objects are
not part of that wheel or source distribution.

The proprietary binary license permits public archival hosting, download,
execution, benchmarking, paper reproduction, and redistribution of exact,
unmodified binary copies as part of the complete artifact. Those permissions
are intended to support artifact evaluation and long-term archival, including
reviewers employed by commercial organizations. The same license prohibits
reverse engineering, extraction of embedded proprietary material,
modification, and commercial product/design-service use, subject to mandatory
rights under applicable law.

## Binary packaging and ABI

The four binary globs in the table have distinct roles:

- `arch/_reference_model*.so` supplies the paper-reference architecture and
  DSE policy;
- `cost/_capacity*.so` supplies only the feasible reference SM count for a
  configured design point; it does not expose area estimates or breakdowns;
- DeepStack `noc/_model_support*.so` supplies reference NoC profiles plus
  chip/NoC energy calibration; and
- TileSight `distributed/noc/_model_support*.so` supplies NoC topology and
  profile support.

They target CPython 3.11 on 64-bit x86 Linux and load locally without Cython,
a compiler, online activation, expiration, machine binding, or an access
token. Adjacent `.so.license` files provide machine-readable SPDX scope
markers. Binary packaging reduces incidental exposure; it is not encryption
or a guarantee against reverse engineering.

Source-visible replacements accept caller-owned values through
`mosaic.arch.custom_profile`, `mosaic.noc.custom_profile`,
`ChipEnergyConfig`, `NocEnergyConfig`, `DramDsePolicy`, and
`bank_oracle.DramBankSpec`. These interfaces cover architecture, DRAM,
topology, per-level latency and bandwidth, energy, bank timing, thermal,
caller-defined capacity/area policy, and power inputs without exposing the
bundled reference calibration.

## Modified forks and public release packaging

An exact official release or archival deposition may be mirrored under the
binary-license conditions above. A modified fork is not a Complete Artifact
under that license. When publishing a modified source fork, remove the four
proprietary `.so` files and their adjacent `.so.license` files unless separate
permission has been obtained; the Apache-2.0 source subset may be modified and
redistributed under its own terms.

For long-term public release, describe the complete deposition as
mixed/custom licensed rather than assigning Apache-2.0 to the deposition as a
whole. The root `LICENSE`, this map, both controlling license texts, all four
binary sidecars, notices, and `SHA256SUMS` should remain together in the
archived artifact.
