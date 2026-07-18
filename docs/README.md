# PreQ-BMC Methodology Documentation

## 1. Purpose of the work

PreQ-BMC is a methodology and tool pipeline for synthesizing a fixed-point Quantized Neural Network (QNN) from a trained floating-point neural network while preserving a local robustness property and maintaining deployment accuracy. The central research problem is:

> Given a trained neural network, an input perturbation region, and a target classification property, find a per-layer fixed-point format `<Q, I, F>` that is small enough for embedded deployment but still preserves the robustness contract and produces acceptable fixed-point/C inference behavior.

The current implementation extends the original Quadapter/PreQ-BMC idea with ESBMC-based bit-precise verification, fixed-point backend diagnostics, optional formal no-saturation verification, empirical saturation diagnostics, block-wise verification, and paper-oriented result aggregation.

## 2. High-level pipeline

The current paper pipeline is executed through:

```bash
python tool/scripts/run_paper_experiments.py \
  --config experiments/paper_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

At a high level, the workflow is:

1. Load the benchmark dataset, architecture, and trained weights.
2. Select one correctly classified test sample.
3. Define the local input perturbation box `[x - eps, x + eps]`, clipped to the dataset domain.
4. Run the floating-point model and build the classification property.
5. Propagate DeepPoly intervals through the network.
6. Compute backward preimage intervals for hidden affine layers.
7. Search for a layer-wise fixed-point format `<Q, I, F>`.
8. Verify each layer contract with ESBMC-generated C harnesses.
9. Optionally verify formal no-saturation properties with ESBMC.
10. Evaluate Python fixed-point and generated C fixed-point backends.
11. Optionally refine the bit-widths until accuracy, mismatch, and saturation thresholds are satisfied.
12. Export structured reports, CSV tables, aggregate files, LaTeX tables, and plots.

A compact way to describe the method in the dissertation is:

> The method decomposes QNN synthesis into layer-wise contracts. Backward preimage analysis defines admissible output intervals for each affine layer. The search procedure chooses fixed-point formats and invokes ESBMC on generated C harnesses to check whether the quantized affine transformation preserves the corresponding contract. Since preserving the preimage contract alone does not guarantee that the deployed fixed-point implementation behaves well, the pipeline additionally evaluates semantic gaps and saturation through formal and empirical checks.

## 3. Main concepts

### 3.1 Floating-point reference network

The floating-point model is the original trained model. It is used as the semantic reference for:

- the selected sample prediction;
- the target robustness label;
- baseline/full-precision accuracy;
- quantized Keras accuracy after weights are quantized but executed in floating-point TensorFlow/Keras.

The pipeline refuses a sample when the original model misclassifies it, because the local robustness claim would not be meaningful for that sample.

### 3.2 Local robustness region

For a selected input `x`, the perturbation region is represented as a box:

```text
x_low  = clip(x - eps, dataset_min, dataset_max)
x_high = clip(x + eps, dataset_min, dataset_max)
```

The command-line flag `--eps` is treated as the input perturbation radius. The script also exposes aliases such as `--input-epsilon` and `--perturbation-radius`.

### 3.3 Classification property

The output property is one of the following:

- target-label property: the original predicted class must remain the winner;
- valid-label-set property: the prediction must remain inside a permitted set of labels.

The output-layer ESBMC harness checks the fixed-point affine output against this property.

### 3.4 Fixed-point format

Each affine layer receives a fixed-point specification:

```text
Q = total number of bits
I = integer magnitude bits, excluding the sign bit
F = fractional bits
Q = I + F + 1 sign bit
scale_factor = 2^F
```

The fixed-point backend uses signed integer arithmetic, round-half-away-from-zero division, and saturation/clamp semantics.

## 4. Preimage computation

### 4.1 Why preimages are used

The purpose of preimage computation is to turn an end-to-end robustness property into layer-wise sufficient conditions. Instead of asking ESBMC to verify the whole neural network at once, the method computes an admissible interval domain for each affine layer. Then each layer is checked independently.

Informally:

```text
If layer k receives inputs inside its verified input box,
and if the quantized affine computation maps those inputs into the preimage interval required by layer k+1,
then the downstream robustness property remains preserved under the layer-wise contract chain.
```

This is the core decomposition that makes the method tractable.

### 4.2 DeepPoly interval propagation

The symbolic propagation stage calls the DeepPoly network abstraction and stores, for each hidden layer:

- lower and upper bounds before clipping;
- lower and upper bounds after ReLU-style clipping;
- activation mode information when abstraction modes require it.

These bounds are used to define candidate input domains and to formulate backward preimage constraints.

### 4.3 Backward preimage synthesis

The backward preimage pass starts from the output layer and walks backward through the hidden dense layers. For each layer, it computes relaxed lower and upper intervals that are sufficient to keep the downstream property.

The implementation supports these modes:

- `milp`: MILP-based under-approximation using Gurobi;
- `abstr`: abstraction-based fallback/alternative;
- `comp`: combined mode, using MILP when possible and abstraction when needed.

The MILP formulation searches for a relaxation scale around the currently propagated bounds. For output layers, it encodes a violation of the target-class margin as the property to avoid. For hidden layers, it encodes leaving the downstream relaxed interval. The resulting relaxed interval is stored as the layer preimage domain.

### 4.4 Cache support

Because Gurobi preimage computation can be expensive and may depend on licensing, the pipeline can save and load preimage caches. This allows later ESBMC-only runs:

```bash
python tool/scripts/export_gurobi_preimage_cache.py \
  --datasets mnist \
  --archs 1blk_100 \
  --sample-ids 0 \
  --eps 1.0 \
  --preimage-mode milp \
  --cache-dir output/preimage_cache
```

Then:

```bash
python tool/scripts/run_robustness_pipeline.py \
  --dataset mnist \
  --arch 1blk_100 \
  --sample-id 0 \
  --eps 1.0 \
  --preimage-mode milp \
  --verify-mode esbmc \
  --no-gurobi \
  --preimage-cache-dir output/preimage_cache
```

## 5. Bit-width synthesis

The synthesis step searches a per-layer bit-width configuration. The result is represented as:

```json
{
  "success": true,
  "total_bits": [...],
  "integer_bits": [...],
  "fractional_bits": [...],
  "stats": {...}
}
```

The important point for the paper is that the method is not merely evaluating a manually chosen Q format. It is searching for a candidate format and checking whether that candidate satisfies the layer-wise contract.

For each layer, the generated fixed-point weights and biases are obtained by integer quantization. The candidate is accepted only if the configured verifier mode confirms the layer property or if the selected mode declares the candidate valid by synthesis.

## 6. ESBMC layer verification

### 6.1 Generated C harnesses

The method generates C verification harnesses for ESBMC. These harnesses are not the same as the final C deployment implementation. They are small programs whose purpose is to encode one formal obligation.

For hidden affine layers, the harness checks:

```text
For every input in the layer input interval,
the fixed-point affine interval image is inside the preimage interval.
```

For the output layer, the harness checks:

```text
For every input in the output-layer input interval,
the target class remains greater than all other classes,
or the predicted class remains inside the valid class set.
```

The harnesses use nondeterministic inputs and `__ESBMC_assume` to restrict them to the current interval domain. They use `__ESBMC_assert` to encode the property.

### 6.2 ESBMC command profile

The default paper-oriented ESBMC profile uses:

```text
esbmc <file.c> --function main --unwind <inferred> --bitwuzla --bv --timeout <seconds> --memlimit <limit> --interval-analysis --interval-analysis-simplify --result-only
```

The unwind bound is inferred from constants such as `INPUT_SIZE`, `LAYER_SIZE`, `OUTPUT_SIZE`, and generated full-network layer sizes.

The normalized ESBMC statuses are:

- `VERIFIED`: ESBMC reported verification successful;
- `FAILED`: ESBMC reported verification failed;
- `TIMEOUT`: timeout or time-limit marker;
- `MEMOUT`: memory-limit or out-of-memory marker;
- `UNKNOWN`: no conclusive status marker was recognized;
- `SKIPPED`: block intentionally skipped by policy, usually fail-fast.

### 6.3 Block-wise hidden-layer verification

Hidden affine layers can be split into contiguous blocks of output neurons:

```bash
--esbmc-layer-block-size 10
```

The mathematical contract is a conjunction over output neurons. Therefore, verifying all neuron blocks with the same shared `<Q, I, F>` is equivalent to verifying the whole hidden-layer contract, but each ESBMC query is smaller.

Important condition: block-wise verification must not choose different bit-widths per block unless the paper explicitly changes the semantics. The current implementation uses a shared layer-level Q/I/F candidate.

Fail-fast is enabled by default. This means that a bad candidate can be rejected as soon as one block fails, reducing solver time. A diagnostic option can force all blocks to run even after a failure.

## 7. Saturation verification

The project adds saturation checking because the original layer-contract check alone may preserve the mathematical preimage property but still produce bad deployment behavior when the fixed-point backend saturates. This is a strong and valuable addition.

There are two saturation mechanisms.

### 7.1 Empirical saturation diagnostics

The empirical method executes the Python fixed-point backend over a comparison dataset split and records, per layer:

- total evaluated values;
- saturation count;
- saturation rate;
- minimum pre-clamp value seen;
- maximum pre-clamp value seen;
- layer integer range `[q_min, q_max]`.

The empirical check can reject/refine a candidate when any layer saturation rate exceeds the configured threshold:

```bash
--empirical-saturation-check
--saturation-threshold 0.01
```

This is useful as a deployment-quality gate. However, it is not a proof over the entire perturbation region; it is evidence over the evaluated samples.

### 7.2 Formal no-saturation verification

The formal method generates an ESBMC harness for each affine layer. The harness computes the interval image of the affine layer before clamp and asserts:

```text
lower_pre_clamp >= q_min
upper_pre_clamp <= q_max
```

where:

```text
q_min = -2^(Q - 1)
q_max =  2^(Q - 1) - 1
```

If ESBMC verifies this harness, then no value in the interval abstraction can force the layer pre-clamp output outside the signed fixed-point range.

This is stronger than empirical saturation diagnostics, but it can be conservative: if the interval abstraction is loose, ESBMC may fail or time out even when no concrete input saturates.

### 7.3 How saturation interacts with refinement

The quality-refinement loop can use both deployment metrics and formal no-saturation verification. When saturation is detected:

- empirical saturation failure increases integer bits in the layer with the highest saturation rate;
- formal no-saturation failure increases integer bits in the layer reported by the failed ESBMC saturation record;
- accuracy/mismatch failures default to increasing fractional bits in the output layer when no better layer-local error attribution is available.

This creates a practical distinction:

```text
Formal-only synthesis finds a bit-width satisfying the layer contract.
Quality-refined synthesis adjusts that bit-width to make the deployed fixed-point implementation accurate and saturation-safe.
```

This distinction should be emphasized in the dissertation.

## 8. Python fixed-point and C backend validation

After synthesis, the pipeline builds:

1. a quantized Keras model, where weights and biases are quantized but execution is still floating-point;
2. a Python integer fixed-point network;
3. a generated C fixed-point implementation compiled as a shared library.

The Python fixed-point and C backend use the same intended arithmetic:

```text
acc = sum(input_int * weight_int)
value = round_half_away_from_zero(acc / 2^F_in) + bias_int
value = clamp_to_signed_range(value, Q)
if hidden layer: value = max(value, 0)
```

The pipeline compares:

- Keras quantized accuracy;
- Python fixed-point accuracy;
- generated C fixed-point accuracy;
- Python-vs-Keras mismatch rate;
- C-vs-Keras mismatch rate;
- Python-vs-C exact integer output match;
- max/mean absolute logit error;
- per-layer saturation rates.

This is essential because a formal bit-width result is only useful if the actual fixed-point semantics used by the deployment backend are aligned with the verified semantics.

## 9. Quality gate

The quality gate is enabled when at least one quality threshold is active or formal no-saturation is required, and `max_quality_refinement_steps > 0`.

The main thresholds are:

```bash
--accuracy-drop-threshold 0.05
--saturation-threshold 0.01
--mismatch-threshold 0.05
--max-quality-refinement-steps 10
```

A candidate can be rejected if:

- fixed-point accuracy drops too much relative to quantized Keras;
- Python fixed-point predictions mismatch quantized Keras too often;
- empirical saturation exceeds the threshold;
- formal no-saturation verification fails when required.

The refinement history is saved in:

```text
reports/refinement_history.json
```

## 10. Main output artifacts

For a single pipeline run, the main outputs are:

```text
reports/pipeline_summary.json
reports/experiment_summary.json
reports/qnn_vs_keras_metrics.json
reports/qnn_vs_keras_mismatches.csv
reports/refinement_history.json
reports/quantization_config.json
reports/table_formal_vs_refined.csv
reports/table_deployment_metrics.csv
reports/table_resource_metrics.csv
```

For a batch paper run, aggregation produces:

```text
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

## 11. Recommended dissertation claim wording

Use cautious wording that separates formal proof, conservative abstraction, and empirical validation:

> PreQ-BMC synthesizes layer-wise fixed-point formats for affine neural-network layers using backward preimage contracts. For each candidate quantization, ESBMC verifies bit-precise C harnesses that encode the preservation of the local preimage contract. To bridge the gap between contract preservation and deployable fixed-point inference, the method adds fixed-point semantic validation, formal no-saturation checks, and empirical saturation/accuracy diagnostics. The final result is therefore reported at two levels: formal contract preservation and deployment-quality preservation.

Avoid saying:

> Formal verification is not sufficient.

Better:

> The original formal contract is necessary but does not by itself characterize all deployment-level fixed-point effects, especially saturation and backend semantic mismatch. The extended pipeline therefore adds saturation and semantic-gap checks as complementary properties.

## 12. What is formally verified and what is not

### Formally verified

- Layer-wise affine preimage contracts through generated ESBMC harnesses.
- Output-layer classification property under the output-layer input interval.
- Optional formal no-saturation property for affine layer interval images.
- Generated harness safety checks that ESBMC keeps enabled by default.

### Empirically evaluated

- Accuracy preservation over selected train/test comparison split.
- Python fixed-point vs quantized Keras mismatch rate.
- Generated C backend accuracy.
- Python fixed-point vs C integer exact match.
- Saturation rates over evaluated samples.

### Not fully proven by the current implementation

- Complete end-to-end equivalence of the full QNN against the floating-point model for all possible inputs.
- Robustness of arbitrary samples outside the selected local perturbation region.
- Absence of saturation over concrete semantics if the formal no-saturation check is disabled.
- Generalization of empirical accuracy and saturation evidence beyond the evaluated split.
- Full activation semantics as an independently verified property if the paper scope is limited to affine layers.

## 13. Suggested methodology figure

```text
Floating-point NN + sample x + eps
        |
        v
Reference prediction and local robustness property
        |
        v
DeepPoly interval propagation
        |
        v
Backward preimage computation per affine layer
        |
        v
Layer-wise bit-width search <Q,I,F>
        |
        v
Generated ESBMC C harnesses
        |-------------------------------|
        v                               v
Preimage/argmax contract checks     Formal no-saturation checks
        |                               |
        |---------------+---------------|
                        v
              Candidate QNN configuration
                        |
                        v
Python fixed-point + generated C backend
                        |
                        v
Accuracy, mismatch, saturation diagnostics
                        |
                        v
Quality refinement or accepted QNN
                        |
                        v
Paper CSV/JSON/plots/tables
```
