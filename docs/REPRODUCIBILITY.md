# Experiment and Reproducibility Guide

## Environment Check
Run the recommended preflight checks before starting experiments to verify Python, ESBMC, CBC, and package availability:

```Bash
preqbmc verify-environment
gcc --version
```
### Quickstart Artifact Demo
The cached demo is intended to run without Gurobi as a small artifact smoke test:

```Bash
preqbmc demo --no-gurobi --output output/demo_run
```

### Batch Runner
The batch runner executes all enabled runs in a provided JSON config, stores per-run outputs, aggregates results, and generates plots.

```Bash
python tool/scripts/run_paper_experiments.py \
  --config experiments/paper_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```
### Important Configuration Warning regarding Saturation:
If the article claims saturation is part of the main method, ensure both formal_saturation_check and empirical_saturation_check are set to true in your `experiments/paper_experiments.json` default configuration. If treating saturation as an ablation, create separate JSON configs to compare results.

### Reproducing Article Results
To reproduce full article experiments, run the article runner and then the aggregation script:

```Bash
# 1. Dry run to test setup
preqbmc reproduce --config experiments/article_experiments.json --only iris --max-runs 1 --dry-run

# 2. Full run and aggregation
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

If article outputs are already present under `output/article_runs`, aggregate them directly:

```Bash
preqbmc aggregate --input-root output/article_runs --output-root output/article_results --plots
```
#### Single-Run Command
To execute a specific robustness pipeline manually:

```Bash
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
  --output-dir output/manual_run
```
### Useful Run Modes
Main Proposed Method: Use `--verify-mode esbmc`, `--formal-saturation-check`, `--require-formal-no-saturation`, and `--empirical-saturation-check`.

Block-wise Verification: Use `--esbmc-layer-block-size` 10 to split hidden affine layers into contiguous blocks to improve ESBMC scalability. Use 0 for full-layer verification.

Ablations: Toggle `--no-formal-saturation-check` or `--no-empirical-saturation-check` to isolate the effects of the formal preimage contract.