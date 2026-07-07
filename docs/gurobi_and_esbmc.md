# Solvers and ESBMC

## ESBMC

ESBMC is the bounded model checker used to verify generated C harnesses. PreQ-BMC expects an `esbmc` executable on `PATH`.

Check:

```bash
esbmc --version
preqbmc verify-environment
```

Article-oriented runs use the existing ESBMC profiles and resource controls in the pipeline. The public CLI exposes MILP backend selection through `--solver` without changing verification semantics.

## CBC

CBC is the default license-free MILP backend for the active robustness/article pipeline. Install it with:

```bash
pip install -e '.[cbc]'
```

Use it explicitly with `--solver cbc`, or omit `--solver` because CBC is the default.

## Gurobi

Gurobi is an optional reference backend for full MILP-based preimage synthesis and MILP forward checks. A valid license and importable `gurobipy` package are required for `--solver gurobi`.

Do not commit:

- `gurobi.lic`
- `*.lic`
- WLS access IDs
- WLS secrets
- license IDs
- logs containing private solver credentials

## Cached Preimage Path

Cached preimage contracts allow reviewers to run harness generation, ESBMC verification, and diagnostics without solving the preimage MILP:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

The demo cache lives under `examples/preimage_cache/`.

To generate new legacy caches with the existing cache export script, use a licensed environment:

```bash
python tool/scripts/export_gurobi_preimage_cache.py \
  --datasets iris_15x2 \
  --archs 2blk_15_15 \
  --sample-ids 27 \
  --eps 0.05 \
  --preimage-mode milp \
  --cache-dir output/preimage_cache
```

This command performs the original Gurobi MILP preimage synthesis and saves the contracts for later `--no-gurobi` runs. For active robustness pipeline runs, use `--solver cbc` by default or `--solver gurobi` for reference comparisons.
