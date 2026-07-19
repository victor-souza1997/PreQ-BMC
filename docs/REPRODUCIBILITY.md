# Experiment and Reproducibility Guide

This guide gives the reviewer-facing commands for the SBSeg artifact. Run commands from the repository root after completing [installation](installation.md).

## 1. Environment Check

```bash
preqbmc verify-environment
gcc --version
```

If ESBMC is missing and network access is available, use:

```bash
preqbmc verify-environment --install-missing-esbmc
```

Missing Gurobi is acceptable for the default CBC pipeline and for the cached demo. Missing TensorFlow, h5py, or scikit-learn must be fixed before running full benchmark experiments.

## 2. Quickstart Artifact Demo

The cached Iris demo avoids Gurobi and is the recommended first smoke test:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

This should produce `reports/`, `layers/`, and `c_export/` under `output/demo_run/`. See [expected_results.md](expected_results.md) for the file checklist.

## 3. Article Runner

Dry run one Iris case:

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --only iris \
  --max-runs 1 \
  --dry-run
```

Run configured article experiments and aggregate results:

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

Run only a small real article subset:

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --only iris \
  --max-runs 1 \
  --continue-on-error \
  --aggregate \
  --plots
```

If article outputs already exist:

```bash
preqbmc aggregate \
  --input-root output/article_runs \
  --output-root output/article_results \
  --plots
```

## 4. Manual Single Run

Use the public CLI when possible. For lower-level debugging, call the current script path under `src/scripts/`:

```bash
python src/scripts/run_robustness_pipeline.py \
  --dataset iris_15x2 \
  --arch 2blk_15_15 \
  --sample-id 27 \
  --eps 0.05 \
  --bit-lb 3 \
  --bit-ub 40 \
  --preimage-mode milp \
  --verify-mode esbmc \
  --solver cbc \
  --formal-saturation-check \
  --no-require-formal-no-saturation \
  --empirical-saturation-check \
  --accuracy-drop-threshold 0.05 \
  --saturation-threshold 0.01 \
  --mismatch-threshold 0.05 \
  --max-quality-refinement-steps 10 \
  --esbmc-layer-block-size 10 \
  --output-dir output/manual_iris_15x2_sample27_eps005
```

For a cached preimage run:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

`--no-gurobi` means "load cached preimage bounds"; it is not the same as choosing a solver backend.

## 5. Solver Backends

Default CBC run:

```bash
preqbmc demo --output output/demo_cbc
```

Gurobi reference run:

```bash
preqbmc demo --solver gurobi --output output/demo_gurobi
```

The Gurobi command requires an importable `gurobipy` package and a valid license. CBC and Gurobi may differ in performance and small numerical tolerances; the formal transfer claim rests on ESBMC verification of generated C harnesses, not on solver identity.

## 6. Soundness-Sensitive Modes

The article configuration may include compatibility or diagnostic defaults, such as legacy non-zero contract tolerance or disabled chaining enforcement. Those modes can be useful for engineering experiments, but they should not be described as the strongest deployed-transfer claim unless `experiment_summary.json` reports:

```text
guarantee_level = deployed-transfer
```

For a stricter soundness-oriented article run, override the relevant settings:

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --only iris \
  --max-runs 1 \
  --no-unsound-contract-tolerance \
  --enforce-contract-chaining \
  --continue-on-error \
  --aggregate
```

For legacy diagnostic replay:

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --only iris \
  --max-runs 1 \
  --unsound-contract-tolerance \
  --no-enforce-contract-chaining \
  --continue-on-error \
  --aggregate
```

Interpret the resulting claim through `guarantee_level` and `transfer_preconditions` in `reports/experiment_summary.json`.

## 7. Useful Run Flags

Main ESBMC path:

```bash
--verify-mode esbmc
```

Block-wise hidden-layer verification:

```bash
--esbmc-layer-block-size 10
```

Full-layer verification:

```bash
--esbmc-layer-block-size 0
```

Formal no-saturation as optional evidence:

```bash
--formal-saturation-check
--no-require-formal-no-saturation
```

Formal no-saturation as an acceptance requirement:

```bash
--formal-saturation-check
--require-formal-no-saturation
```

Disable formal no-saturation:

```bash
--no-formal-saturation-check
```

## 8. Output Inspection

For a single run, inspect:

```text
<run-output>/reports/pipeline_summary.json
<run-output>/reports/experiment_summary.json
<run-output>/reports/qnn_vs_keras_metrics.json
<run-output>/reports/refinement_history.json
<run-output>/layers/
<run-output>/c_export/qnn_model.c
```

For aggregated article results, inspect:

```text
output/article_results/all_experiments.csv
output/article_results/table_quality_metrics.csv
output/article_results/table_scalability.csv
output/article_results/table_esbmc_status_counts.csv
output/article_results/table_mrr.csv
output/article_results/table_implementation_gap.csv
output/article_results/article_summary.md
```
