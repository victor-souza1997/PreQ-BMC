# Solver Backends

## Default CBC Backend

The active robustness/article pipeline uses CBC through `python-mip` as the default MILP backend:

```bash
pip install -e '.[cbc]'

python tool/scripts/run_robustness_pipeline.py \
  --dataset iris \
  --arch 1blk_10 \
  --sample-id 25 \
  --eps 0.05 \
  --preimage-mode milp \
  --verify-mode esbmc \
  --solver cbc \
  --output-dir output/iris_cbc
```

CBC is license-free and is used for MILP preimage synthesis and MILP forward verification when those modes are selected.

## Gurobi Reference Backend

Gurobi remains available as a reference backend:

```bash
pip install -e '.[gurobi]'

python tool/scripts/run_robustness_pipeline.py \
  --dataset iris \
  --arch 1blk_10 \
  --sample-id 25 \
  --eps 0.05 \
  --preimage-mode milp \
  --verify-mode esbmc \
  --solver gurobi \
  --output-dir output/iris_gurobi
```

This requires an importable `gurobipy` package and a valid Gurobi license.

## Cached Preimage Path

`--no-gurobi` is kept as a backward-compatible cache-loading alias. It means “load cached preimage bounds instead of solving the preimage MILP.” Solver selection remains independent:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

For cached ESBMC-only runs, no MILP backend is constructed unless `--verify-mode milp` is also selected.

## Parity Expectations

CBC and Gurobi use different MILP implementations, so tiny floating-point differences are expected. Backend parity checks compare objective values and selected continuous values within about `1e-6`; end-to-end parity should preserve synthesized layer bit-widths, preimage bounds within tolerance, ESBMC statuses, and final summary statuses for the small Iris/Seeds validation cases.

Run a local parity report with:

```bash
python tool/scripts/compare_solver_backends.py --case iris --case seeds --output-root output/solver_parity
```

Use `--quick` for a shorter smoke comparison that skips deployment-quality/table work.

## Scope

This CBC milestone covers the active robustness/article pipeline centered on `synthesis.preqbmc`. Legacy backdoor scripts remain Gurobi-only in this first milestone.
