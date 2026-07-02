# Metrics and Status Fields

PreQ-BMC writes per-run JSON summaries and aggregate CSV tables. Important fields include:

- `contract_status`: formal status for the layer or run contract. Typical values are `VERIFIED`, `FAILED`, `TIMEOUT`, `MEMOUT`, `UNKNOWN`, or `VERIFIED_BY_SYNTHESIS`.
- `contract_verified`: boolean view of whether the formal contract was verified.
- `no_saturation_status`: formal status for no-saturation checking. Values include `VERIFIED`, `FAILED`, `TIMEOUT`, `MEMOUT`, `UNKNOWN`, and `SKIPPED`.
- `no_saturation_verified`: boolean view of whether formal no-saturation checks verified.
- `deployment_quality_accepted`: whether empirical deployment-quality checks passed for the candidate.
- `python_c_exact_match`: whether Python fixed-point outputs and generated-C outputs matched exactly on the evaluated samples.
- `final_status`: combined status used by article tables. `VERIFIED` means contract, deployment quality, and required no-saturation evidence passed. `PARTIAL_VERIFIED` means contract and deployment quality passed while optional or inconclusive no-saturation evidence did not fully verify. `FAILED` records a definite failure.
- `saturation_rate`: fraction of fixed-point activations that saturated during empirical diagnostics.
- `mismatch_rate`: fraction of samples where fixed-point outputs differ from the quantized Keras reference.
- `accuracy_drop`: difference between a source accuracy and target implementation accuracy.
- `MRR`: maximum robustness radius estimate produced from epsilon sweeps in article aggregation.
- `SMT syntactic complexity`: optional syntactic measurements over generated SMT formulas, such as file size, operation counts, and approximate parenthesis depth.
- `approximate AST depth`: a syntax-level proxy derived from parenthesis nesting in exported SMT text. It is not a semantic proof metric.

Block-wise fields such as `total_blocks`, `verified_blocks`, and `skipped_blocks_due_to_fail_fast` describe ESBMC query decomposition. They do not imply mixed precision.
