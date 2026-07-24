# Third-party notices

The artifact's own source, runner, manifests, documentation, data, distributed
DeepStack source, and TileSight source are licensed under Apache-2.0 unless a
path is one of the proprietary binaries listed in [LICENSING.md](LICENSING.md)
or a file states otherwise.

The four proprietary binary components are governed by
[LicenseRef-DeepStack-AE-Binary-1.0](LICENSES/LicenseRef-DeepStack-AE-Binary-1.0.txt).
Their inventory, roles, packaging, and compatibility are described in
[LICENSING.md](LICENSING.md).

The supplementary `src/deepstack/bank_oracle/` source, tests, and
visualization are part of the pinned DeepStack source snapshot and are
licensed under Apache-2.0. They do not include the reference-capacity source
or the reference providers' source.

The following model-description files retain their upstream copyright and
Apache License 2.0 headers:

- `src/deepstack/mosaic/llm_arch/transformers_modeling/modeling_llama.py`
- `src/deepstack/mosaic/llm_arch/transformers_modeling/modeling_qwen3.py`
- `src/deepstack/mosaic/llm_arch/transformers_modeling/modeling_qwen3_moe.py`
- `src/deepstack/mosaic/llm_arch/transformers_modeling/modeling_qwen3_omni_moe.py`
- `src/deepstack/mosaic/llm_arch/transformers_modeling/modeling_gpt_oss.py`
- `src/deepstack/mosaic/llm_arch/transformers_modeling/modeling_glm_moe_dsa.py`
- `src/deepstack/mosaic/llm_arch/transformers_modeling/modeling_deepseek_v32.py`

The complete Apache License 2.0 text is included at
`LICENSES/Apache-2.0.txt`. Package dependencies installed into the Conda
environment are not redistributed in this repository and retain their own
licenses.
