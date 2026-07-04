# Examples

The examples are intentionally small and do not change the research pipeline.

## Iris Cached Demo

Run:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

This uses:

- dataset: `iris_15x2`
- architecture: `2blk_15_15`
- sample id: `27`
- epsilon: `0.05`
- preimage cache: `examples/preimage_cache/iris_15x2__2blk_15_15__sample27__eps0.05__milp__e904e4f9a08a7a5f/`

Expected outputs:

- `output/demo_run/reports/pipeline_summary.json`
- `output/demo_run/reports/experiment_summary.json`
- `output/demo_run/layers/`
- `output/demo_run/c_export/qnn_model.c`

Gurobi is not required for this cached demo. ESBMC is required.

## Cache Contents

The cache stores layer preimage bounds in the existing `quadapter-preimage-cache-v1` format:

- `metadata.json`
- `preimage.npz`

Do not add solver licenses or private credentials to this directory.
