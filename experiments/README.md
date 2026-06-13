# Paper Experiment Workflow

This directory contains the default experiment plan for paper/dissertation runs.

The default plan is `experiments/paper_experiments.json`. It is intended to run each configured benchmark once and write per-run outputs under `output/paper_runs/`.

## Run All Paper Experiments Once

```bash
python tool/scripts/run_paper_experiments.py \
  --config experiments/paper_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

This runs the robustness pipeline for each enabled entry, then aggregates CSV/JSON/LaTeX outputs and generates figures.

## Dry Run

```bash
python tool/scripts/run_paper_experiments.py \
  --config experiments/paper_experiments.json \
  --dry-run
```

This prints the commands that would be executed without running ESBMC or writing result artifacts.

## Run Only MNIST

```bash
python tool/scripts/run_paper_experiments.py \
  --config experiments/paper_experiments.json \
  --only mnist \
  --continue-on-error
```

## Aggregate Manually

```bash
python tool/scripts/aggregate_paper_results.py \
  --runs-root output/paper_runs \
  --output-dir output/paper_results
```

## Generate Plots Manually

```bash
python tool/scripts/plot_paper_results.py \
  --input output/paper_results/all_experiments.csv \
  --output-dir output/paper_results/figures
```

## Expected Outputs

```text
output/paper_runs/
output/paper_runs/<run_name>/run_stdout.log
output/paper_runs/<run_name>/run_stderr.log
output/paper_runs/<run_name>/command.txt
output/paper_runs/<run_name>/run_config.json
output/paper_runs/<run_name>/run_status.json
output/paper_runs/<run_name>/reports/experiment_summary.json
output/paper_results/all_experiments.csv
output/paper_results/all_experiments.json
output/paper_results/all_formal_vs_refined.csv
output/paper_results/all_deployment_metrics.csv
output/paper_results/all_resource_metrics.csv
output/paper_results/all_refinement_history.csv
output/paper_results/failed_runs.csv
output/paper_results/paper_summary.md
output/paper_results/figures/
output/paper_results/tables/
```

## Benchmark Discovery

To inspect benchmark weights discovered under `tool/benchmark`:

```bash
python tool/scripts/run_paper_experiments.py \
  --config experiments/paper_experiments.json \
  --discover-benchmarks
```

The discovery command also applies the default MNIST cap described below.

## MNIST Cap

MNIST is intentionally limited to `25_25` in the default experiment plan. The default plan includes:

- `mnist` with `1blk_10`
- `mnist` with `1blk_25`
- `mnist` with `2blk_25_25`

It does not include `mnist_50`, `mnist_100`, `mnist_512`, `mnist_1024`, or larger MNIST architectures. Larger MNIST models can be tested manually, but they are not part of the default paper run because of ESBMC scalability.

## Notes

- The scripts use standard library modules plus `matplotlib` for plotting.
- The runner uses `subprocess` with argument lists and does not require Bash.
- Missing fields in partial or failed runs are left empty in aggregate CSVs.
- Failed, timed out, and skipped runs are still represented in `failed_runs.csv` and `all_experiments.csv`.
- Block-wise ESBMC verification does not change the mathematical property. The hidden-layer contract is a conjunction over output neurons, so verifying all blocks with the same `<Q,I,F>` is equivalent to verifying the whole hidden layer.
