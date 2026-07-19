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

## 4. Experiment JSON Configuration Reference

Article runs are configured in `experiments/article_experiments.json`. The file has three top-level areas:

- `metadata`: descriptive fields and default output roots used by the runner.
- `defaults`: values inherited by every run unless a run overrides them.
- `runs`: benchmark entries. A single entry can expand into multiple concrete runs through `sample_ids` and `eps_sweep`.

Useful `metadata` keys:

- `description`: human-readable description.
- `output_root`: directory for per-run outputs, typically `output/article_runs`.
- `aggregate_output_root`: directory for aggregate tables and plots, typically `output/article_results`.
- `notes`: list of caveats that should be visible to reviewers.

Core run keys:

- `name`: stable run name. If omitted, the runner derives one from dataset, architecture, sample, and epsilon.
- `enabled`: set `false` to keep a run documented but skipped by default.
- `dataset`: benchmark family or alias, such as `iris`, `seeds`, `mnist`, or `mnist_onnx`.
- `arch`: architecture alias, such as `1blk_10`, `2blk_25_25`, or `2blk_50_50`.
- `sample_id`: one selected sample.
- `sample_ids`: list of samples; expands one JSON entry into multiple concrete runs.
- `eps`: input perturbation radius.
- `input_epsilon` or `perturbation_radius`: aliases for `eps` in generated commands and reports.

Synthesis and verification keys:

- `bit_lb`, `bit_ub`: search interval for total layer bit-width.
- `preimg_mode`: `milp`, `abstr`, or `comp`.
- `verify_mode`: normally `esbmc`; `milp` is available for lower-level comparison paths.
- `solver`: `cbc` or `gurobi`. CBC is the default open-source backend.
- `target_label`: optional target-label property override.
- `valid_labels`: optional valid-label property override.

Deployment-quality keys:

- `quality_refinement`: set `false` to force `max_quality_refinement_steps = 0`.
- `compare_split`: `test` or `train`.
- `compare_limit`: number of samples for deployment diagnostics; `0` means all samples in the selected split.
- `compile_c_backend`: set `false` to skip generated-C compilation/execution.
- `compiler`: C compiler executable, default `gcc`.
- `enable_diagnostics`: enables fixed-point diagnostics in `qnn_vs_keras_metrics.json`.
- `accuracy_drop_threshold`: reject/refine when fixed-point accuracy drops too much; use `null` to disable in JSON.
- `saturation_threshold`: reject/refine when empirical saturation is too high; use `null` to disable.
- `mismatch_threshold`: reject/refine when fixed-point predictions mismatch quantized Keras too often; use `null` to disable.
- `max_quality_refinement_steps`: maximum refinement attempts.

Formal no-saturation keys:

- `formal_no_saturation` or `formal_saturation_check`: enables ESBMC no-saturation harnesses.
- `require_formal_no_saturation`: if `true`, no-saturation must verify for acceptance; if `false`, it is recorded as optional evidence.
- `empirical_saturation_check`: enables empirical saturation as a deployment-quality gate.
- `no_saturation_continue_on_unknown`: diagnostic mode; if `true`, no-saturation UNKNOWN/TIMEOUT/MEMOUT does not immediately stop remaining no-saturation blocks. This should not be used to claim `deployed-transfer`.

Block-wise ESBMC keys:

- `esbmc_layer_block_size`: `0` for full-layer hidden verification, `N > 0` for hidden-neuron blocks of size `N`.
- `blockwise_fail_fast`: reject a shared `<Q,I,F>` candidate after the first non-verified block.
- `blockwise_run_all_blocks_on_failure`: diagnostic mode that runs all blocks even after a block failure.
- `esbmc_jobs`: number of ESBMC jobs. Use `1` for reproducible, memory-safe default runs.
- `esbmc_profile`: `paper-fast`, `debug`, `fast`, `preimage`, `safety`, or `overflow`.
- `esbmc_timeout_seconds`: per-call ESBMC timeout.
- `esbmc_memlimit`: ESBMC memory limit, such as `6g` or `20g`.

Experiment-control and reporting keys:

- `mode`: `full_pipeline` by default. `full_layer`, `full_layer_verification`, or `monolithic` force `esbmc_layer_block_size = 0`.
- `mrr_enabled`: marks the run for epsilon-sweep/MRR expansion.
- `eps_sweep`: list of epsilon values tested when MRR expansion is enabled.
- `expected_difficulty`: reviewer-facing label such as `low`, `medium`, `high`, or `frontier`.
- `allowed_failure`: documents that a failure is expected scalability evidence rather than a hidden error.
- `scalability_frontier`: marks intentionally hard frontier runs.
- `notes`: free-text explanation for reviewers.
- `export_paper_tables`: writes per-run CSV tables.
- `baseline_results_json`: optional external baseline file for summary comparison.
- `preimage_cache_dir`, `preimage_cache_key`: cached-preimage input/output controls.
- `gurobi_threads`: thread limit for Gurobi reference runs.

Command-line overrides on `preqbmc reproduce` can force several settings across all JSON runs:

- `--solver {cbc,gurobi}`;
- `--no-unsound-contract-tolerance` or `--unsound-contract-tolerance`;
- `--enforce-contract-chaining` or `--no-enforce-contract-chaining`;
- `--propagate-contract-tolerance` or `--no-propagate-contract-tolerance`;
- `--only`, `--skip`, `--max-runs`, and `--include-disabled`.

## 5. Manual Single Run

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

## 6. Solver Backends

Default CBC run:

```bash
preqbmc demo --output output/demo_cbc
```

Gurobi reference run:

```bash
preqbmc demo --solver gurobi --output output/demo_gurobi
```

The Gurobi command requires an importable `gurobipy` package and a valid license. CBC and Gurobi may differ in performance and small numerical tolerances; the formal transfer claim rests on ESBMC verification of generated C harnesses, not on solver identity.


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

## 9. Reporting Guidance

Report `FAILED`, `TIMEOUT`, `MEMOUT`, and `UNKNOWN` runs explicitly. Inconclusive ESBMC statuses characterize the verification scalability frontier and should not be hidden.

When presenting SBSeg results, separate:

- formal contract status;
- deployment-quality acceptance;
- no-saturation evidence;
- Python-vs-C exact match;
- `guarantee_level`.

This separation is important because a successful engineering artifact can still include runs that are harness-verified, partial, timed out, or diagnostic-only.
