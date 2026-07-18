# PreQ-BMC Experiment and Reproducibility Guide

## 1. Command used for paper/master experiments

Run from the repository root:

```bash
python tool/scripts/run_paper_experiments.py \
  --config experiments/paper_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

This command executes all enabled runs in `experiments/paper_experiments.json`, stores per-run outputs under `output/paper_runs/`, aggregates results under `output/paper_results/`, and generates plots.

## 2. Requirements

Required:

- Python environment with the repository dependencies installed.
- ESBMC available on `PATH`.
- Gurobi and `gurobipy` when using `preimg_mode = "milp"` without `--no-gurobi`.
- GCC or another configured compiler when C backend compilation is enabled.
- Matplotlib for plotting.

Recommended preflight checks:

```bash
which esbmc
python -c "import gurobipy; print('gurobi ok')"
gcc --version
python tool/scripts/run_paper_experiments.py \
  --config experiments/paper_experiments.json \
  --dry-run
```

## 3. How the batch runner works

The batch runner performs these steps:

1. Reads the JSON config.
2. Merges top-level `defaults` into each run.
3. Builds a command for `tool/scripts/run_robustness_pipeline.py`.
4. Executes the command in the repository root.
5. Writes logs and status files per run.
6. Continues after failures if `--continue-on-error` is enabled.
7. Runs aggregation if `--aggregate` is enabled.
8. Runs plot generation if `--plots` is enabled.

Each run creates:

```text
output/paper_runs/<run_name>/command.txt
output/paper_runs/<run_name>/run_config.json
output/paper_runs/<run_name>/run_status.json
output/paper_runs/<run_name>/run_stdout.log
output/paper_runs/<run_name>/run_stderr.log
output/paper_runs/<run_name>/reports/
```

## 4. Important configuration warning

The current `experiments/paper_experiments.json` contains this in `defaults`:

```json
"formal_saturation_check": false,
"empirical_saturation_check": false
```

Because the runner merges defaults into every run, the paper command currently disables both formal no-saturation verification and empirical saturation gating unless a specific run overrides those fields.

This is probably not what you want if the dissertation/article claims that saturation is part of the main method. There are two clean ways to handle it.

### Option A: Make saturation part of the main method

Change the defaults to:

```json
"formal_saturation_check": true,
"empirical_saturation_check": true
```

Then rerun:

```bash
python tool/scripts/run_paper_experiments.py \
  --config experiments/paper_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

This makes the paper results match the methodology claim that saturation is checked.

### Option B: Treat saturation as an ablation

Keep the current defaults as the `formal_only_no_saturation` ablation, but create a second config:

```text
experiments/paper_experiments_saturation_enabled.json
```

with:

```json
"formal_saturation_check": true,
"empirical_saturation_check": true
```

Then compare both result sets in the paper.

This is scientifically stronger because it shows why saturation checking is needed.

## 5. Single-run command

Example:

```bash
python tool/scripts/run_robustness_pipeline.py \
  --dataset iris_15x2 \
  --arch 2blk_15_15 \
  --sample-id 0 \
  --eps 0.1 \
  --bit-lb 2 \
  --bit-ub 40 \
  --preimage-mode milp \
  --verify-mode esbmc \
  --formal-saturation-check \
  --empirical-saturation-check \
  --accuracy-drop-threshold 0.05 \
  --saturation-threshold 0.01 \
  --mismatch-threshold 0.05 \
  --max-quality-refinement-steps 10 \
  --esbmc-layer-block-size 10 \
  --output-dir output/manual_iris_15x2_sample0_eps01
```

## 6. Useful run modes

### 6.1 Main proposed method

```bash
--verify-mode esbmc \
--formal-saturation-check \
--require-formal-no-saturation \
--empirical-saturation-check \
--max-quality-refinement-steps 10
```

Use this for the strongest claim.

### 6.2 Formal contract only

```bash
--verify-mode esbmc \
--no-formal-saturation-check \
--no-empirical-saturation-check \
--max-quality-refinement-steps 0
```

Use this as a baseline/ablation to show that preimage-contract preservation alone does not necessarily preserve deployment quality.

### 6.3 Empirical saturation only

```bash
--no-formal-saturation-check \
--empirical-saturation-check \
--saturation-threshold 0.01
```

Use this to show how empirical saturation diagnostics affect accuracy and bit-width refinement.

### 6.4 Formal no-saturation optional evidence

```bash
--formal-saturation-check \
--no-require-formal-no-saturation
```

Use this if ESBMC saturation checks are expensive and you want to record them without rejecting the candidate.

### 6.5 Block-wise ESBMC verification

```bash
--esbmc-layer-block-size 10
```

Use `0` for full-layer verification:

```bash
--esbmc-layer-block-size 0
```

For diagnostics:

```bash
--blockwise-run-all-blocks-on-failure
```

## 7. Interpreting the most important report files

### 7.1 `pipeline_summary.json`

This is the most complete run-level artifact. It contains:

- dataset, architecture, sample id, epsilon;
- selected label and predicted label;
- formal synthesis result;
- final synthesis result after quality refinement;
- ESBMC status counts;
- timing metrics;
- block-wise verification summary;
- formal saturation summary;
- fixed-point semantics;
- accumulator range analysis;
- deployment comparison metrics;
- artifact paths.

### 7.2 `experiment_summary.json`

This is the paper-ready structured summary. It separates:

- `formal_only`: the initial bit-width configuration found by formal synthesis;
- `quality_refined`: the final configuration after quality gates and refinement;
- reference metrics;
- deployment metrics;
- resource metrics;
- formal saturation controls;
- block-wise controls;
- external baselines, if provided.

### 7.3 `qnn_vs_keras_metrics.json`

This file answers: “Does the synthesized QNN behave like the quantized Keras model and the generated C backend?”

Important fields:

```text
keras_quantized_accuracy
python_qnn_accuracy
c_qnn_accuracy
python_qnn_mismatch_rate_vs_keras
c_qnn_mismatch_rate_vs_keras
python_qnn_max_abs_error
python_qnn_mean_abs_error
python_c_integer_comparison.exact_match
fixed_point_diagnostics.python.layers[*].saturation_rate
warnings
```

### 7.4 `refinement_history.json`

This file shows each quality-refinement attempt:

- bit-width configuration;
- quality failures;
- ESBMC status;
- refinement action;
- final reason.

Use it in the paper to explain why a configuration changed from formal-only to quality-refined.

## 8. Interpreting aggregate CSVs

### 8.1 `all_experiments.csv`

This is the main table for analysis. Each run appears once per method:

```text
method = formal_only
method = quality_refined
```

Important columns:

```text
run_name
dataset
arch
eps
method
status
verified
accepted
Q
I
F
full_precision_keras_accuracy
quantized_keras_accuracy
python_fixed_accuracy
c_fixed_accuracy
python_c_exact_match
mismatch_rate_vs_keras
max_saturation_rate
mean_saturation_rate
max_abs_logit_error
mean_abs_logit_error
compression_ratio_vs_float32
backward_time
forward_time
total_time
esbmc_calls
refinement_steps
final_reason
no_saturation_verified_all_layers
accumulator_fits_int64_all_layers
```

### 8.2 `failed_runs.csv`

Use this to report scalability limits honestly:

- timeouts;
- memory exhaustion;
- ESBMC unknown;
- misclassified selected samples;
- missing benchmark weights;
- command/runtime failures.

Do not hide failed runs. They are evidence for the scalability discussion.

### 8.3 `all_formal_vs_refined.csv`

Use this table to compare the original formal-only configuration against the final refined configuration.

Good paper plots:

- total bits before vs after refinement;
- accuracy before vs after refinement;
- max saturation rate before vs after refinement;
- mismatch rate before vs after refinement.

### 8.4 `all_resource_metrics.csv`

Use this for embedded-systems arguments:

- number of parameters;
- fixed memory estimate;
- float32 memory estimate;
- compression ratio;
- activation memory estimate;
- C source lines;
- shared library size.

## 9. Recommended experiment matrix for the dissertation

A strong dissertation experiment design would include these groups.

### Group A: Correctness and quality

Compare:

1. floating-point Keras;
2. quantized Keras;
3. formal-only Python fixed-point;
4. quality-refined Python fixed-point;
5. generated C fixed-point.

Report:

- accuracy;
- mismatch rate;
- max/mean logit error;
- Python/C exact match;
- saturation rate.

### Group B: Formal-method scalability

Vary architecture size:

- Iris/Seeds small networks;
- MNIST 1-layer networks;
- MNIST 2-layer networks;
- larger manual MNIST runs if available, such as `1blk_50`, `1blk_100`, `2blk_50_50`, `2blk_100_100`.

Report:

- ESBMC calls;
- verified/failed/timeout/memout/unknown counts;
- max/mean query time;
- block size;
- largest estimated MACs per query.

### Group C: Saturation ablation

Compare:

1. no saturation checks;
2. empirical saturation only;
3. formal no-saturation only;
4. empirical + formal no-saturation;
5. empirical + formal + quality refinement.

This directly supports the thesis that deployment-level fixed-point properties must be checked in addition to preimage preservation.

### Group D: Block-wise verification

Compare:

```text
full layer
block size 25
block size 10
block size 5
```

Report:

- total ESBMC runtime;
- timeout rate;
- number of queries;
- largest query size;
- whether all blocks verified.

## 10. Recommended paper tables

### Table 1 — Benchmark summary

```text
Dataset | Architecture | #Layers | #Parameters | eps | Sample ID | Reference Accuracy
```

### Table 2 — Formal synthesis results

```text
Dataset | Arch | Q/I/F | Contract Status | ESBMC Calls | Time | Timeout/Memout/Unknown
```

### Table 3 — Deployment quality

```text
Dataset | Arch | Method | Keras-Q Acc | Python-QNN Acc | C-QNN Acc | Mismatch | Max Sat. | Python=C
```

### Table 4 — Resource metrics

```text
Dataset | Arch | Method | Avg Bits/Param | Fixed Bytes | Float32 Bytes | Compression Ratio
```

### Table 5 — Ablation

```text
Dataset | Arch | No Sat. | Empirical Sat. | Formal Sat. | Refined | Accuracy | Max Sat. | Bits
```

## 11. Recommended wording for failed or unknown runs

Use wording like:

> A timeout or UNKNOWN result is not a counterexample to the property. It means that ESBMC did not complete the proof under the selected resource limits. We report these cases separately to characterize scalability.

For formal no-saturation:

> A failed no-saturation interval check indicates that saturation is possible under the interval abstraction. Because the abstraction can be conservative, this should be interpreted as a refinement trigger unless ESBMC provides a concrete counterexample that is inspected and confirmed.

For empirical saturation:

> Empirical saturation diagnostics are not formal proofs. They measure the deployed fixed-point behavior over the selected evaluation split and are used as a deployment-quality criterion.

## 12. Immediate checklist before using results in the article

- [ ] Decide whether saturation is part of the main method or an ablation.
- [ ] Fix `paper_experiments.json` if saturation should be enabled by default.
- [ ] Ensure README and experiment config agree about the MNIST cap.
- [ ] Keep `formal_only` and `quality_refined` as separate methods in tables.
- [ ] Report timeouts/UNKNOWN/MEMOUT explicitly.
- [ ] Report Python-vs-C exact match.
- [ ] Include resource metrics to support the embedded-systems motivation.
- [ ] Explain that empirical saturation is complementary evidence, not formal proof.
- [ ] Explain that formal no-saturation is interval-based and can be conservative.
