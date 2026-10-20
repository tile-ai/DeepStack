# Reproducibility, evidence, and provenance

This is the single detailed reviewer index for the artifact. The root
`README.md` gives the commands; this document maps each paper result to the
recomputed evidence, recorded inputs, model version, and provenance. Exact
released bytes are covered by the root `SHA256SUMS`; selected data directories
also contain local manifests.

## Figure-number convention

The current AE paper removed two earlier figures, while artifact stage IDs
remain stable for scripts and archived evidence:

| Current AE paper | Artifact stage |
|---|---|
| Figs. 8–10 | `fig08`–`fig10` |
| Fig. 11 (DeepSeek-V3.2 sparse attention) | `fig12` |
| Fig. 12 (ASTRA-Sim/NS-3) | `fig13` |
| Figs. 13–21 | `fig15`–`fig23` |

References below give the current paper number first and the stable artifact
stage second. Paths, commands, output filenames, and preview assets continue
to use the stage ID.

## Shared replay semantics

- The artifact reruns DeepStack analytical modeling. GPU measurements and
  ASTRA-Sim/NS-3 outputs are immutable comparison inputs and are never
  presented as fresh runs.
- `paper_legacy` preserves the historical sparse-MoE activated-expert count
  used for submitted values. Most paper-value reruns close directly. Paper
  Fig. 9 is the explicit source-version exception: the final-paper CSV metric
  is audited separately from the released legacy rerun.
  `current_corrected` evaluates the same fixed configurations after the July
  2026 fix. It is version-drift evidence, not a corrected exhaustive DSE.
- Expected-value CSVs are deterministic regression references. Each stage
  first calls the model and only then compares the fresh output.
- The original combinatorial search is not rerun. Paper Fig. 13 (stage
  `fig15`) retains its complete 301,728-row stacked-GPU population; other DSE
  figures use complete selected configurations plus compact, checksummed plot
  projections.
- Publish-safe result CSVs omit unnecessary internal power, energy, raw
  throughput, machine path, and run-identity fields. Acceptance checks still
  operate on complete in-memory model results.
- Die-area breakdown data are not distributed. Reference feasibility calls
  return only an SM-capacity count from a finite compiled table.

## Result and evidence matrix

The full workflow has 15 stages: 14 numerical stages followed by one
repository-local plotting stage that emits 16 PNG and 16 PDF outputs. The
exact command/runtime ledger is `results/reproduce/run_manifest.csv`.
Times below are final stage wall times from the packaged reproduction;
slash-separated Fig. 19 times are its separately invoked model paths.

| Current paper result / artifact stage | Default AE treatment | Required model-side evidence | External reference treatment | Verified observation |
|---|---|---|---|---|
| Paper Fig. 8 / stage `fig08`, H100 kernels | all 52 published points | recompute DeepStack/TileSight latency | bundled measured CSV | 1.75 s; 8.366% weighted MAPE |
| Paper Fig. 9 / stage `fig09`, B200 vLLM | all 40 common plotted points | dual-path end-to-end STPS rerun | bundled vLLM CSV | 2.33 s; 80 analytical evaluations; final-paper CSV MAPE 12.918976%; legacy rerun MAPE 12.724500%; historical three-panel audit 12.18% |
| Paper Fig. 10 / stage `fig10`, MI325X | all 28 plotted 4xMI325X points | dual-path decode STPS rerun | bundled vLLM CSV | 9.80 s; paper equal-model mean 13.095% |
| Paper Fig. 11 / stage `fig12`, DSA | complete 40-point memory-feasible context/BS grid | dual-path DSA STPS rerun | bundled vLLM CSV | 4.20 s; legacy wMAPE 2.612% |
| Paper Fig. 12 / stage `fig13`, ASTRA-Sim | all 420 points in six panels | rerun pinned paper-time NoC closure | bundled NS-3/analytical CSV and reported runtime endpoints | 3.44 s; model max relative delta 4.45e-16; reported-endpoint ratio 108,000x |
| Paper Fig. 13 / stage `fig15`, Pareto | 784 local winners re-derived from 301,728 recorded DSE rows | dual-path fixed-config UTPS/STPS rerun; quick uses 28 deterministic non-winners | complete reduced March DSE CSV | 186.95 s; legacy max error 4.0e-13% |
| Paper Fig. 14 / stage `fig16`, 3D vs 2.5D | all 80 plotted winners | dual-path values and ratios | compact baseline configs | 19.96 s; 1.30–1.48x and 2.79x |
| Paper Fig. 15 / stage `fig17`, 3D DRAM bandwidth | complete 96-point analytical sweep | recompute bandwidth and peaks | digests over the publish-safe projection and summary | 0.70 s; 9-layer peak; 39.6226% drop |
| Paper Figs. 16–18 / stage `fig18_20`, DRAM TPS/energy/thermal | 114 unique complete fixed configs, including 22 terrain samples | dual-path TPS/power/temperature rerun | 110 curve rows, 108 metric cells, and all 19,233 plotted terrain points | 79.78 s; sample errors below 0.000242% STPS and 0.000491 C |
| Paper Fig. 19 / stage `fig21`, NoC latency/BW | all 180 recorded fixed configs | dual-path rerun and 90-point reduced surface | none | 6.62/6.43 s; legacy max error 2.4e-13% |
| Paper Fig. 20 / stage `fig22`, per-layer NoC | 756 fixed decode/prefill points per path | dual-path rerun, 1,930 model calls/path | archived paper model column for regression only | 364.05 s at 32 workers; legacy max error 4.77e-13% |
| Paper Fig. 21 / stage `fig23`, parallelism | all 84 curve winners, 75 unique configs | dual-path decode/prefill rerun across three search spaces | compact full-DSE summaries | 22.17 s; legacy 5.0343x and 2.3104x |
| Paper Table 4 / stage `table4`, ablation | fourteen selected configs | dual-path rerun of seven steps at BS=4/1024 | compact full-DSE summaries | 9.89 s; 2.7906x/9.4744x |
| Retained result plots / stage `figures` | 16 paper-facing groups | redraw PNG/PDF from generated and explicitly archived CSVs | no external path | 11.82 s; 32 images plus `plot_manifest.csv` |

## Result-specific audits

- **Paper Fig. 8 / stage `fig08`:** the five measured CSVs cover
  8+8+12+12+12 points. The paper
  weighted MAPE is 8.366124% and AG-GEMM MAPE is 3.971760%. NumPy seed 0
  removes only reuse-distance Monte Carlo ordering noise.
- **Paper Fig. 9 / stage `fig09`:** the final four-panel paper CSV has
  12.918976% pointwise MAPE, reported as 12.92% in the paper. Rerunning the
  same 40 configurations through `paper_legacy` gives 12.724500%. Qwen3-235B
  and Llama3-405B close directly; the final Llama3-70B and DeepSeek-V3 panels
  pin older `0222`/`0311` source states, while the released path closes against
  adjacent `0223`/`0313` records without hidden per-BS scale factors. The
  12.177422% value, rounded to 12.18%, is retained only as provenance for an
  earlier three-model plot and is not the current paper claim. Point evidence
  is in `legacy_source_version_audit.csv` and
  `historical_source_matrix.csv`.
- **Paper Fig. 10 / stage `fig10`:** 13.095363% is the mean of the three
  per-model weighted MAPEs
  (5.154283%, 13.335579%, and 20.796228%). The alternative all-28-point
  aggregate is 16.276045% and is labeled separately.
- **Paper Fig. 11 / stage `fig12`:** the complete grid has 10+9+8+7+6 points
  for prompt lengths 1,024 through 31,744. The paper metric is
  `sum(abs(model-measured))/sum(measured)*100 = 2.6120978817%`; the ordinary
  point MAPE is 3.878250%. Mean corrected-path drift is 24.29% at BS <= 8
  versus 1.87% at BS >= 64.
- **Paper Fig. 12 / stage `fig13`:** the archived DeepStack column is not
  copied into the result. The pinned closure re-executes all 420 points. The
  0.1 s and 3 h values are reported endpoints used only to audit the 108,000x
  arithmetic, not fresh wall-clock measurements.
- **Paper Fig. 13 / stage `fig15`:** the source has 313,409 rows; removing only
  non-stacked H100/H200-scaled rows leaves all 301,728 stacked-GPU rows.
  Reproduction
  derives 784 bucket winners. Quick mode excludes them and selects 28
  non-winners by stable SHA-256 ordering.
- **Paper Figs. 16–18 / stage `fig18_20`:** the fixed set represents 44
  stage-`fig18` roles, 28 stage-`fig19` winner roles, 66 stage-`fig20`
  extremal roles, and 22 non-extremal terrain audit roles; duplicates are
  stored once. Stage `fig19` retains all 108 valid heatmap cells in the
  paper's two-row, four-column layout. Stage `fig20` retains all 19,233
  plotted terrain points, while the model rerun checks 22 stratified
  non-extremal samples and complete non-published vector digests.
- **Paper Fig. 19 / stage `fig21`:** the 180 rows cover three batch sizes,
  five latency multipliers, six bandwidth multipliers, and two mappings. The
  renderer uses three normalized surfaces with no selected-point star.
- **Paper Fig. 20 / stage `fig22`:** 294 decode plus 462 prefill points are
  rerun per model path. Logic-die and NoC-in-DRAM accounting are both evaluated
  before the 84 normalized curve points are emitted.
- **Paper Fig. 21 / stage `fig23`:** winners are selected for ASTRA-style,
  expanded-parallel, and module-flexible spaces. Decode BS=1024 closes at
  5.034325x for DeepSeek-V3 and 2.310412x for Qwen3-235B-A22B.
- **Paper Table 4 / stage `table4`:** steps 1–5 follow the cumulative ablation
  filters; Step 6 selects the DRAM-layer winner; Step 7 joins the fine-NoC
  winner back to its complete Step 6 hardware/parallel row.

## Release and model-version provenance

The released tree plus root `SHA256SUMS` is the authoritative byte-exact
snapshot. External source-control history, ignored run directories, GPU
microbenchmarks, and unrelated development outputs are not dependencies.

| Component | Release state | State date | Reviewer-visible location |
|---|---|---|---|
| DeepStack model | `deepstack-release-2026-07` | 2026-07 | `src/deepstack/` |
| TileSight model | `tilesight-release-2026-07` | 2026-07 | `src/tilesight/` |
| Paper Fig. 12 / stage `fig13` paper-time closure | `fig13-closure-2025-11` | 2025-11 | `ae/fig13_paper_model/` |
| Full-DSE projections | `paper-dse-2026-03` | 2026-03 | result-specific `data/` directories |

The March projections predate the activated-expert correction. The two paths
share one released source tree except for the explicit compatibility selector;
`paper_legacy` is not a rollback of every historical source file. Binary
roles, ABI support, custom replacements, and redistribution terms are in
`LICENSING.md`.

## Data and provenance matrix

| Result | Repository-local evidence | Source/selection boundary | Integrity |
|---|---|---|---|
| Paper Fig. 8 / stage `fig08` | `data/fig08/reference/paper_csv/` | five immutable 8xH100 measurement projections; all 52 model points rerun | root manifest and generated `verification.json` |
| Paper Fig. 9 / stage `fig09` | `data/fig09/` | four vLLM CSVs; configs declare TP8/EP1, while `replace_only` maps the MoE path to effective EP8 before best-per-BS selection | local `SHA256SUMS` and `SOURCE_PROVENANCE.csv` |
| Paper Fig. 10 / stage `fig10` | `data/fig10/reference/`, `paper_legacy/` | six byte-exact 4xMI325X CSV copies; 8x exploratory data excluded | local `SHA256SUMS` |
| Paper Fig. 11 / stage `fig12` | `data/fig12/` | privacy-minimized 40-point 8xB200 projection; run names, paths, and timestamps removed | local `SHA256SUMS` and `SOURCE_PROVENANCE.csv` |
| Paper Fig. 12 / stage `fig13` | `data/fig13/`, `ae/fig13_paper_model/` | six ASTRA/NS-3 CSVs plus a three-file pinned analytical closure | local `SHA256SUMS` |
| Paper Fig. 13 / stage `fig15` | `data/fig15/decode_dse_population_march2026.csv` | complete stacked-GPU population from the March decode search | root manifest |
| Paper Fig. 14 / stage `fig16` | `data/fig16/configs/` | 80 plot-selected 3D and area-normalized baseline winners | root manifest |
| Paper Fig. 15 / stage `fig17` | `data/fig17/paper_reference_summary.sha256` | digests over the complete 96-point projection and seven-row summary | root manifest plus digest check |
| Paper Figs. 16–18 / stage `fig18_20` | `data/fig18_20/fixed_configs.csv`, `archive/` | 114 fixed configs plus compact curve, heatmap, and terrain views | local `SHA256SUMS` and vector digests |
| Paper Fig. 19 / stage `fig21` | `data/fig21/candidates_march2026.csv` | lossless 180-row TMS1 projection; best-of-two-mappings reduction happens after rerun | root manifest |
| Paper Fig. 20 / stage `fig22` | `data/fig22/` | 756 March-selected layer-sensitivity configs per path | local `SHA256SUMS` and `SOURCE_PROVENANCE.csv` |
| Paper Fig. 21 / stage `fig23` | `data/fig23/` | 84 plotted winners, 75 unique configs, selected by the paper's three nested search-space filters | local `SHA256SUMS` |
| Paper Table 4 / stage `table4` | `data/table4/` | fourteen complete fixed rows and publish-safe Step 6/7 candidate projections | local `SHA256SUMS` |

Machine-readable manifests, expected-value CSVs, source matrices, and
regression `.sha256` files remain beside their inputs. The per-directory prose
was intentionally consolidated here so each result has one authoritative
description.

## Recorded source digests

These digests identify source-only inputs audited during packaging. They are
not runtime paths.

| Scope | Recorded source and SHA-256 |
|---|---|
| Paper Fig. 8 / stage `fig08` plotting/model scripts | `combine_figures.py` `9c3bc0d8003b9d4c113e9b789dfb2e11af9f01b2bc7682474ad9aa392ffd1071`; MoE reduce+AR `e664d3a8b22f8e9d2ebc1950700ed8c36d504cbfbf01fccff168e3c9569bbf73`; EP A2A `42708c0487bdfd42e4c33034676a08d629a507eeaf33d298efd9e2b211a2c735`; Ulysses `b59b1d19e3b4941a315f9701cbfef0ce7af0ccc688c5ff7868f168c159072479`; AG-GEMM `a65b761b4230b15199a993f2a23a1ab6d3b181bddca3f4718e86bd0e00305101`; GEMM-RS `6dc5c06b335b35c28bbeeed7b46dc187d8588822ce0e9f76fcc1954d34bfd763` |
| Paper Fig. 10 / stage `fig10` submitted PDF | `195fc9eeb415de0e11d7bfd7866c47182c20b359761a55850fea188120aadbd8` |
| Paper Fig. 11 / stage `fig12` plot input / metric script | `f19318d2fd3c95dd8a2f90f98dec2598dc553dc346517672bdfd0b65e768b130`; `3188d0a4b6680286189c49ec69af1ed2ec2710cf63896138910020d11b0a307b` |
| Paper Fig. 12 / stage `fig13` closure | `noc_topo.py` `c072744b4dcc1e6c1b48c96a947c04c86a73571f8cce6c0fc43dc72eb8289c05`; `noc_topo_switch_only.py` `4e509f32adf40f65930cdc7ed11ac13fe6c3425b1685b82d35f5d6051f9a8961`; `traffic_matrix.py` `bcdeedfdb1df8a3c64a46dc4545b47d0c6db4d21ae287469f538a40ad9c15beb` |
| Paper Figs. 16–18 / stages `fig18`–`fig20` plotting scripts | Stage `fig18` `b6102fbabfdf3fa609257fd668c0f8e6b99da88c70907e347d350c960e0cb001`; stage `fig19` `f0368ccfd4e9c32242967e9653068ab25c2a1cc45f901188fb0d65e4efb8df82`; stage `fig20` `61f15ebc9f1508c01caad4d22a5d752d1d60e5dca8693b39c1ba75d7afa5a82c` |
| Paper Fig. 20 / stage `fig22` runners/plotter | decode `fe8cd07c9410a5864e527467868cfa2d9479462b0d1d9b14919e1eec8806600f`; prefill `6ed0dcd2d902c245b38dbcc82af51072e96a8c4b5b764c15ac66da80b1245e7c`; plot `449004486ed30ce2f2ee538c68ff9696ef28999a14c2dfb2588092e8d8275c47` |
| Paper Fig. 21 / stage `fig23` plot/PDF | script `3d0abfe148888fa20bea5d85e7873b514112e222c3c648104ccb6385ff9fe309`; PDF `3cb1a53f4bf2149a37f25183631c3550542f2e83adac2615a471b9422950fb58` |
| Paper Table 4 / stage `table4` selectors | `ablation_e2e.py` `9abb757ef959d48c5f4230c96b8de24a03137092895130086165fc156c0bee32`; `noc_multi_layer_bw.py` `73443a05163f488920104e2bc55963aae224b48b94d690b7ef94d8aeae859745` |

The largest source tables intentionally excluded from the release are:

| Source table | Bytes | SHA-256 |
|---|---:|---|
| March stacked-GPU decode search | 172,331,027 | `9e96696f107dad300731527804daff34cf3c4063ca3c627860689b3cf2e3144f` |
| DRAM-layer decode search | 265,422,287 | `0d5b6fb281df8ec6617773a9f9c7bec5bbb39d7e33e5ab0278212609a3860e74` |
| DRAM-layer prefill search | 288,292,845 | `958ed53c46776cd35d093db244511cefa0d5b0ef9bbccf15d7eac1d4e4c0b897` |
| Expanded-parallel prefill search | not bundled | `3e1ae2034c316fb807381fd4dd0aa401d63f8427e86cf67743a089c2abc1d570` |
| NoC logic-die / NoC-in-DRAM searches | not bundled | `3bdb4eb9f7adfb29675ebfd935b82ba5dd8657356487ae7cff9cce3f5b116215` / `0935e532ed98942755b99f768d68c694608c8689bf8b0e39a3bcd3ac19ce3193` |

## Bundled routing traces

The four JSONs under `src/deepstack/mosaic/data/routing_outputs/` contain
selected expert/group indices and generator metadata only: no prompts, model
weights, or hidden-state values.

| Trace | Tokens | Routing rule |
|---|---:|---|
| `routing_iter128_N256_G1_TG1_E8.json` | 128 | global top-8 |
| `routing_iter128_N256_G8_TG4_E8.json` | 128 | top-4 of 8 groups, 2 experts/group |
| `routing_iter10000_N256_G1_TG1_E8.json` | 10,000 | global top-8 |
| `routing_iter10000_N256_G8_TG4_E8.json` | 10,000 | top-4 of 8 groups, 2 experts/group |

The 128-token files are prefixes of the corresponding 10,000-token traces.
The artifact never resolves the preserved `output_dir` generator string.
AIME/MMLU activation NPZs in neighboring directories retain per-layer
prefill/decode expert selections used by the model.

## Explicit exclusion

The paper's hardware-emulation cross-validation uses a non-distributable 3D
design reference. Those data are unavailable in the artifact, so that
validation is outside the Results Reproduced scope.

## Optional reusability evidence

| Interface | Caller-controlled inputs | Evidence |
|---|---|---|
| `mosaic.arch.custom_profile` and `ChipEnergyConfig` | compute/cache resources, clocks, stacked-DRAM geometry, memory/compute energy, static power | `tests/test_custom_arch_profile.py`, `tests/test_energy_configuration.py` |
| `mosaic.noc.custom_profile` and `NocEnergyConfig` | L3/L2/L1 topology, shape, hop latency, link/switch bandwidth, port spread, per-level energy | `tests/test_noc_profile_provider.py`, `tests/test_energy_configuration.py` |
| `DramDsePolicy` and DSE callbacks | DRAM capacity/latency, thermal, power, buffering, architecture, NoC, capacity provider or area estimator | `tests/test_custom_dram_dse_policy.py` |
| `bank_oracle.DramBankSpec` | layer/bank counts, row/sector geometry, service cycles, independent data/latency clocks | `src/deepstack/bank_oracle/tests/` and the synthetic HTML witness |

These source-visible contract suites use synthetic caller inputs and neither
expose nor assert the built-in calibration.
