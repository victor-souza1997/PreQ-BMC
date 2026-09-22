# Accuracy and output refinement follow-up

This implements a measurement step missing from briefing 3. It does not retrain
the source model, replace old artifacts, or reinterpret inconclusive outcomes.
For the Android Auto application, host inference is a prerequisite measurement,
not evidence of an installed application, on-device parity, latency or energy.
GTSRB supplies traffic-sign classes, not license-plate strings.

## Full-test accuracy retention

Run one evaluation per immutable C artifact, not per successfully verified image:

```sh
preqbmc gtsrb evaluate-host \
  --study output/sign_affine_residual_pilot_20260921/study.json \
  --artifact output/sign_milp_source_h12_pilot_20260920/artifact/artifact.json \
  --output output/sign_host_quality_new
```

Alternatively pass `--region <region_summary.json>`. Both paths check model and
generated C identities; an uncertified artifact may be measured without gaining
a certificate. Existing output directories are rejected. All frozen `test`
records are evaluated without filtering by class correctness, source eligibility,
clean margin, or verification outcome. The manifest hash includes IDs, image
hashes and labels. Image loading checks each file's hash.

The evaluator reconstructs the restricted Conv/ReLU/Flatten/Dense float32 Keras
architecture from the identity-checked model weights. It reports this provenance
explicitly, rather than claiming execution of an independently validated archived
Keras file. The same byte resize and normalization by 256 are used throughout.

It measures:

- Float32 accuracy, recomputed on the frozen test population.
- Float32 execution with rounded weights and biases. Activations are **not**
  quantized in this diagnostic; it is not the deployed arithmetic.
- Independent Python integer accuracy and generated C accuracy.
- C/Python equality of encoder, hidden values and logits at host `-O0` and `-O2`.
- Prediction mismatch, logit error in real logit units, source-correct images
  made incorrect, and source-incorrect images recovered by quantization.
- The existing uniform Q16/F8 control on exactly the same population.

For test labels y_k and predictions f_k, q_k, reported accuracy loss is

\[
  \Delta_{pp} = \frac{100}{N}\sum_{k=1}^{N}
  (\mathbf{1}[f_k=y_k]-\mathbf{1}[q_k=y_k]).
\]

It is in **percentage points**, not a relative percentage. Prediction mismatch
is a separate statistic: equal accuracies do not imply matching predictions.
`--max-accuracy-drop-pp` defaults to zero and controls only the empirical quality
acceptance field and command exit code. Choose any nonzero allowance before
evaluating the test set. It does not weaken a formal obligation or update a
region's status. Exit 1 can mean a completed measurement with unacceptable
accuracy loss or parity; inspect the JSON rather than assuming a crash.

Outputs: `host_quality.json`, `table_host_quality.csv`, per-method
`predictions.csv`, compiled host libraries and `device_vectors.jsonl`.
Device vectors carry byte-image identity, encoded input, hidden values and
integer logits. They are expected outputs for subsequent device replay, not
Android observations. `android_parity` and performance remain `NOT_MEASURED`;
`android_transfer_verified` remains false.

### First full-test measurement (2026-09-21)

The completed final-code report is
`output/sign_host_quality_final_20260921/host_quality.json`; the initial report
under `output/sign_host_quality_20260921` is preserved and has identical metrics.
It covers all 12,630 frozen test records, with no eligibility filter:

| Execution | Accuracy | Prediction mismatch vs float32 |
|---|---:|---:|
| Reconstructed source float32 | 41.7498% | -- |
| Uniform Q16/F8 C control | 41.7577% | 0.8155% |
| Selected h12/o14 C | 41.6627% | 1.0610% |

Both integer variants matched the independent Python encoder/hidden/logit
reference on every test image at both host optimization levels. The selected
artifact loses 0.0871 percentage points: 38 source-correct images become wrong
and 27 source-wrong images become correct. Its worst absolute logit difference
is 32.2471 real logit units despite the small aggregate accuracy change. Thus
accuracy retention and logit fidelity must not be treated as interchangeable.
The strict zero-loss quality criterion is **not accepted**. No threshold was
relaxed after seeing the result. The source model's low absolute accuracy remains
an application limitation; successful local certificates do not repair it.

Validation: 121 focused tests passed, covering the new quality metrics and CLI,
artifact checks, output refinement, source gate, error budgets, generated C,
fixed-point forward execution and no-saturation regression tests. The prepared
`output/sign_affine_residual_evaluation_20260921/study.json` contains 54 matched
runs (27 regions, 9 images, 2 proof variants), all using the same frozen QIF.
That full campaign has not been launched by this change.

## Existing output refinement

The opt-in `output_refinement=affine_residual` implementation already exists.
The matched pilot under `output/sign_affine_residual_pilot_runs_20260921`
completed: box-only `image6_eps4` is `MARGIN_INCONCLUSIVE`; the refined variant is
`VERIFIED` at `byte-crop-integer-C`. This is not a newly trained model or an
accuracy improvement. The original failed box queries remain in the ledger.

For one hidden affine/ReLU layer and one affine output layer, let integer
parameters be W,b,V,c and rescaling denominators S0,S1. Define

\[
 a=Wx,\quad \rho=S_0\operatorname{round}(a/S_0)-a,
 \quad h=\operatorname{round}(a/S_0)+b+t,
\]
\[
 \eta_j=S_1\operatorname{round}((V_jh)/S_1)-V_jh,
 \quad o_j=\operatorname{round}((V_jh)/S_1)+c_j+u_j.
\]

Here t includes the **actual** hidden clamp and ReLU correction, and u includes
the actual output clamp correction. Neither is assumed zero. With competitor j,
target y and d=V_j-V_y, the exact integer identity is

\[
 S_0S_1(o_j-o_y)=(dW)x+S_0db+S_0S_1(c_j-c_y)
 +d\rho+S_0dt+S_0(\eta_j-\eta_y)+S_0S_1(u_j-u_y).
\]

Composing dW **before** interval maximization retains cancellation through the
shared input. Scalar ESBMC lemmas check the common C rounding/clamp kernel over
each bounded accumulator domain. A composition harness checks the numeric
certificate and the implication from the bounded terms to `o_j-o_y < 0`.
Every prerequisite lemma and the composition check must return `VERIFIED`.
The sufficient bound may fail even when the true network is robust.

This is a compositional proof rule using the algebraic identity above and
sound affine interval maximization. The final checker does not symbolically
execute the whole network from pixels. Trust includes correct linkage of the
certificate to the network, the algebraic rule and range analysis, ESBMC and
its solver. It is not a compiler proof or a bit-exact IEEE proof of the source
gate. The source MILP and float lowering assumptions remain unchanged.

The hidden preimage contracts, input bridge and required chaining still run.
QIF remains shared across blocks. Unsupported shapes, insufficient bounds,
timeouts, and failed prerequisites cannot acquire a certificate through this
refinement. Only one hidden affine/ReLU layer is currently supported.

## Matched evaluation

```sh
preqbmc gtsrb prepare-refinement \
  --config experiments/sign_affine_residual_evaluation.json \
  --output output/sign_affine_residual_evaluation_new
preqbmc gtsrb run \
  --study output/sign_affine_residual_evaluation_new/study.json \
  --output output/sign_affine_residual_evaluation_runs_new
```

These are matched box/refinement variants, not independent samples. The
previously inspected evaluation cohort is not a new blind holdout. Preparation
does not execute verification. Report all configured regions, source-refuted
regions and unresolved outcomes, alongside conditional rates for eligible
regions. Retraining requires a new frozen artifact and a new evaluation; it
cannot preserve the old certificates by changing their labels.

The remaining Android experiment needs the actual device and native build,
byte/hidden/logit replay, independent latency trials, process resource
measurements and synchronized external power traces. Host parity alone is not
Android deployment validation or a guarantee of driving safety.
