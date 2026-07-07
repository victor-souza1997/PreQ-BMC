# Artifact Evaluation Guide

## Minimal Requirements

- Python 3.10 or newer.
- `pip install -e '.[cbc]'` from the repository root for the default CBC MILP backend.
- ESBMC on `PATH` for verification runs.
- Optional: Gurobi and `gurobipy` for `--solver gurobi` reference runs.
- Optional: `matplotlib` and `pandas` for plots and table analysis.

The cached demo is intended to run without Gurobi.

## Environment Check

```bash
preqbmc verify-environment
```

This command reports Python, ESBMC, CBC/python-mip, optional Gurobi, and Python package availability. Missing Gurobi is not fatal unless `--solver gurobi` is selected.

## Quickstart Demo

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

Expected inputs:

- `examples/preimage_cache/iris_15x2__2blk_15_15__sample27__eps0.05__milp__e904e4f9a08a7a5f/metadata.json`
- `examples/preimage_cache/iris_15x2__2blk_15_15__sample27__eps0.05__milp__e904e4f9a08a7a5f/preimage.npz`

Expected outputs:

- `output/demo_run/reports/pipeline_summary.json`
- `output/demo_run/reports/experiment_summary.json`
- `output/demo_run/layers/`
- `output/demo_run/c_export/qnn_model.c`

Expected runtime depends on ESBMC and host CPU, but the cached Iris example is intended to be a small artifact smoke test rather than a full paper reproduction.

## Commands Requiring A MILP Solver

- Full MILP preimage synthesis without `--no-gurobi` uses CBC by default or Gurobi with `--solver gurobi`.

## Commands Requiring Gurobi

- Reference MILP runs with `--solver gurobi`.
- Preimage cache generation with `tool/scripts/export_gurobi_preimage_cache.py`.

## Commands Requiring ESBMC

- `preqbmc demo --no-gurobi`
- `preqbmc demo`
- Article runs with `--verify-mode esbmc`

If ESBMC is absent, `preqbmc demo` stops before running the pipeline and explains how to proceed.

## Reproduce a Small Article Run

Dry run:

```bash
preqbmc reproduce --config experiments/article_experiments.json --only iris --max-runs 1 --dry-run
```

Actual run, if ESBMC and dependencies are installed:

```bash
preqbmc reproduce --config experiments/article_experiments.json --only iris --max-runs 1
```

## Inspect Precomputed Results

If article outputs are already present under `output/article_runs`, aggregate them:

```bash
preqbmc aggregate --input-root output/article_runs --output-root output/article_results --plots
```

Inspect:

- `output/article_results/all_experiments.csv`
- `output/article_results/table_quality_metrics.csv`
- `output/article_results/table_scalability.csv`
- `output/article_results/article_summary.md`
