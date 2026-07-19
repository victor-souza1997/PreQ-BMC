# Expected Results and Metrics

This document explains what a reviewer should expect after running the artifact commands and how to interpret the main status fields. Exact runtimes and selected bit-widths can vary with ESBMC version, solver backend, host CPU, and resource limits.

## Cached Demo Outputs

Command:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

Expected inputs:

```text
examples/preimage_cache/iris_15x2__2blk_15_15__sample27__eps0.05__milp__e904e4f9a08a7a5f/metadata.json
examples/preimage_cache/iris_15x2__2blk_15_15__sample27__eps0.05__milp__e904e4f9a08a7a5f/preimage.npz
```

Expected key outputs:

```text
output/demo_run/command.txt
output/demo_run/run_stdout.log
output/demo_run/run_stderr.log
output/demo_run/reports/pipeline_summary.json
output/demo_run/reports/experiment_summary.json
output/demo_run/reports/qnn_vs_keras_metrics.json
output/demo_run/reports/refinement_history.json
output/demo_run/layers/
output/demo_run/c_export/qnn_model.c
```

The cached Iris demo is intended as a smoke test. It should be much smaller than the full article run and does not require Gurobi.

## Per-Run Reports

Important files under each run's `reports/` directory:

- `pipeline_summary.json`: complete run configuration, solver controls, ESBMC call records, block-wise verification summary, no-saturation block summary, chaining status, fixed-point semantics, and artifact paths.
- `experiment_summary.json`: paper-oriented summary separating `formal_only` and `quality_refined` results.
- `qnn_vs_keras_metrics.json`: comparison among quantized Keras, Python fixed-point, and generated-C fixed-point execution.
- `qnn_vs_keras_mismatches.csv`: sample-level mismatches, when any are recorded.
- `refinement_history.json`: quality-refinement attempts and reasons.
- `quantization_config.json`: selected layer-wise `<Q,I,F>` configuration.
- `table_formal_vs_refined.csv`, `table_deployment_metrics.csv`, `table_resource_metrics.csv`: per-run CSV tables.

Generated verification and deployment artifacts:

- `layers/*.c`: ESBMC contract and no-saturation harnesses.
- `layers/**/*.stdout.log` and `layers/**/*.stderr.log`: streamed ESBMC logs for each harness.
- `c_export/qnn_model.c`: generated fixed-point C backend.
- `c_export/qnn_model.so`: compiled shared library when C backend compilation is enabled.

## Aggregate Article Outputs

After:

```bash
preqbmc aggregate --input-root output/article_runs --output-root output/article_results --plots
```

Expected aggregate outputs include:

```text
output/article_results/all_experiments.csv
output/article_results/all_experiments.json
output/article_results/article_summary.md
output/article_results/failed_runs.csv
output/article_results/table_quality_metrics.csv
output/article_results/table_bitwidths.csv
output/article_results/table_success_failure.csv
output/article_results/table_scalability.csv
output/article_results/table_esbmc_status_counts.csv
output/article_results/table_mrr.csv
output/article_results/table_implementation_gap.csv
output/article_results/table_ablation.csv
output/article_results/latex/
output/article_results/plots/
```

Some SMT complexity tables or plots are populated only when SMT formula generation was enabled during the run.

## Core Status Fields

Use these fields when deciding what claim a run supports:

- `contract_status`: ESBMC contract status. Typical values are `VERIFIED`, `FAILED`, `TIMEOUT`, `MEMOUT`, `UNKNOWN`, `SKIPPED`, or `VERIFIED_BY_SYNTHESIS`.
- `contract_verified`: boolean view of whether the formal layer or run contract verified.
- `no_saturation_status`: formal no-saturation status. Values include `VERIFIED`, `FAILED`, `TIMEOUT`, `MEMOUT`, `UNKNOWN`, and `SKIPPED`.
- `no_saturation_verified`: boolean view of formal no-saturation verification.
- `deployment_quality_accepted`: whether empirical deployment-quality gates passed.
- `python_c_exact_match`: whether Python fixed-point outputs and generated-C outputs matched exactly over the evaluated samples.
- `final_status`: combined status used by article tables.
- `guarantee_level`: claim-strength field in `experiment_summary.json`.

`final_status` values:

- `VERIFIED`: contract, deployment quality, and required no-saturation evidence passed.
- `PARTIAL_VERIFIED`: contract and deployment quality passed, but optional or inconclusive no-saturation evidence did not fully verify.
- `FAILED`: a definite formal or deployment-quality failure.
- `UNKNOWN`: insufficient conclusive evidence.

`guarantee_level` values:

- `deployed-transfer`: all recorded transfer preconditions are satisfied.
- `harness-verified`: contracts verified, but at least one transfer precondition is missing or a diagnostic/legacy mode weakens the claim.
- `failed`: a definite failure.
- `unknown`: insufficient conclusive evidence.



## Deployment Metrics

Common quality fields:

- `quantized_keras_accuracy`: accuracy of the quantized-weight Keras model.
- `python_fixed_accuracy`: accuracy of the Python integer fixed-point model.
- `c_fixed_accuracy`: accuracy of the generated C fixed-point backend.
- `mismatch_rate_vs_keras`: fraction of predictions that differ from quantized Keras.
- `max_abs_logit_error` and `mean_abs_logit_error`: logit-level implementation gap measures.
- `max_saturation_rate` and `mean_saturation_rate`: empirical saturation rates.
- `python_c_exact_match`: exact integer agreement between Python fixed-point and generated C.

Empirical metrics are dataset-split evidence, not formal proofs over the whole perturbation region.

## Block-Wise Verification Fields

Block-wise fields describe ESBMC query decomposition only. They do not imply mixed precision.

Important fields:

- `blockwise_verification.enabled`;
- `blockwise_verification.block_size`;
- `blockwise_verification.policy`, expected to be `shared_layer_qif`;
- `total_blocks`, `verified_blocks`, `failed_blocks`, `timeout_blocks`, `memout_blocks`, `unknown_blocks`;
- `skipped_blocks_due_to_fail_fast`;
- `no_saturation_blocks_*` counters.

A layer is accepted only when all required blocks pass with the same layer `<Q,I,F>`.

## Interpreting Failures And Unknowns

Do not hide `TIMEOUT`, `MEMOUT`, or `UNKNOWN` runs. They are important evidence for scalability and solver-resource limits.

Recommended wording:

> A timeout, memory-out, or UNKNOWN status is not a counterexample. It means the configured solver resources did not produce a proof for that harness.

For formal no-saturation:

> A failed no-saturation interval check indicates that saturation is possible under the interval abstraction. Because the abstraction can be conservative, this may act as a refinement trigger unless a concrete counterexample is inspected and confirmed.

For empirical saturation:

> Empirical saturation diagnostics are deployment-quality evidence over evaluated samples, not formal no-saturation proofs.
