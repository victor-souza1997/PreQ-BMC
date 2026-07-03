# Architecture

PreQ-BMC keeps the article pipeline as a sequence of existing components. This document describes the flow without changing the implementation.

## Pipeline

1. **Model, sample, and epsilon**
   - The runner loads a benchmark model and one input sample.
   - The perturbation radius defines an input box around the sample.
   - The classification property is derived from the model prediction unless a target or valid-label set is provided.

2. **Preimage synthesis**
   - MILP mode uses Gurobi to synthesize layer-wise preimage contracts.
   - Cached mode loads precomputed contracts with `--no-gurobi`.
   - Abstract modes are kept as existing research-prototype options.

3. **Bit-width search**
   - The synthesis stage searches for one shared `<Q,I,F>` per layer.
   - Block-wise verification does not introduce mixed precision; blocks only decompose ESBMC queries.

4. **Fixed-point C harness generation**
   - ESBMC harnesses are generated under `<output>/layers/`.
   - Hidden affine contracts, output properties, and no-saturation properties use the existing C templates.

5. **ESBMC verification**
   - ESBMC checks the generated C harnesses.
   - Dense hidden layers may be decomposed into neuron blocks, while the layer is accepted only if all blocks verify with the same layer `<Q,I,F>`.

6. **Deployment diagnostics**
   - The Python fixed-point backend evaluates the selected format.
   - When enabled, the C backend writes `<output>/c_export/qnn_model.c` and compiles `<output>/c_export/qnn_model.so`.
   - The runner compares Keras quantized outputs, Python fixed-point outputs, and generated-C outputs.

7. **Result aggregation**
   - Per-run JSON/CSV summaries are written under each run directory.
   - Article aggregation scripts collect those outputs under `output/article_results/`.

## Main Directories

- `tool/backends/`: fixed-point Python and generated-C backend.
- `tool/synthesis/`: preimage, bit-width search, and pipeline orchestration.
- `tool/verification/`: ESBMC C templates, runner, and property helpers.
- `tool/scripts/`: existing experiment, aggregation, plotting, and evaluation scripts.
- `experiments/`: article and paper experiment configurations.
- `examples/`: small artifact examples and cached preimage contracts.
