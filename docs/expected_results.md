# Expected Results and Metrics
## Main Output Artifacts
A successful single pipeline run generates the following key files in its reports/ directory:

- `pipeline_summary.json`: The complete run-level artifact containing exact parameters, ESBMC status counts, formal saturation summaries, and deployment comparison metrics.
- `experiment_summary.json`: A paper-ready structured summary separating formal_only results and quality_refined results.
- `qnn_vs_keras_metrics.json`: Evaluates if the synthesized QNN behaves like the quantized Keras model and the generated C backend.
- `refinement_history.json`: Documents each quality-refinement attempt, showing why a configuration changed from formal-only to quality-refined.

For a batch paper run, aggregation produces files like all_experiments.csv, failed_runs.csv, and organized `tables/figures` under the `output/paper_results/` directory.

## Core Metrics and Status Fields
When analyzing the aggregate CSVs, the following fields determine the success and validity of the synthesized network:

- **contract_status**: The formal status. Values include `VERIFIED`, `FAILED`, `TIMEOUT`, `MEMOUT`, `UNKNOWN`.
- **final_status**: Combined status used for article tables.
  - `VERIFIED`: Contract, deployment quality, and required no-saturation evidence passed.
- `PARTIAL_VERIFIED`: Contract and deployment quality passed; optional no-saturation evidence did not fully verify.
- `FAILED`: A definite failure.
- `python_c_exact_match`: A boolean indicating if Python fixed-point outputs and generated C outputs matched exactly.
- `saturation_rate`: The fraction of fixed-point activations that saturated during empirical diagnostics.
- `mismatch_rate`: The fraction of samples where fixed-point outputs differ from the quantized Keras reference.
- `Interpreting Failed` or `Unknown` Runs
Do not hide failed runs (`TIMEOUT`, `UNKNOWN`, `MEMOUT`, or memory exhaustion); they are critical evidence for characterizing scalability limits.
