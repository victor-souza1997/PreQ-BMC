# Gurobi and ESBMC

## ESBMC

ESBMC is the bounded model checker used to verify generated C harnesses. PreQ-BMC expects an `esbmc` executable on `PATH`.

Check:

```bash
esbmc --version
preqbmc verify-environment
```

Article-oriented runs use the existing ESBMC profiles and resource controls in the pipeline. The public CLI does not change solver flags or verification semantics.

## Gurobi

Gurobi is used for full MILP-based preimage synthesis. A valid license and importable `gurobipy` package are required for that mode.

Do not commit:

- `gurobi.lic`
- `*.lic`
- WLS access IDs
- WLS secrets
- license IDs
- logs containing private solver credentials

## No-Gurobi Cached Path

Cached preimage contracts allow reviewers to run harness generation, ESBMC verification, and diagnostics without Gurobi:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

The demo cache lives under `examples/preimage_cache/`.

To generate new caches, use the existing script in a licensed environment:

```bash
python tool/scripts/export_gurobi_preimage_cache.py \
  --datasets iris_15x2 \
  --archs 2blk_15_15 \
  --sample-ids 27 \
  --eps 0.05 \
  --preimage-mode milp \
  --cache-dir output/preimage_cache
```

This command performs the original MILP preimage synthesis and saves the contracts for later `--no-gurobi` runs.
