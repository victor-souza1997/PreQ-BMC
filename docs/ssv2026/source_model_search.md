# GTSRB source-model search

The source model is an experimental prerequisite, not a formal result. A local
robustness proof around an accurately classified image is useful only if the
source classifier is credible for the traffic-sign task. Model selection must
also remain separate from the official GTSRB test set and from ESBMC outcomes.

## Validation-only search

The search command is:

```bash
preqbmc gtsrb search-model \
  --config experiments/sign_source_model_search.json \
  --output output/sign_source_model_search_<date>
```

It reuses the frozen physical-sign-track split from
`output/plate_experiments/study.json`, decodes only train and validation images,
and applies the same top-left nearest-neighbor resize and `/256` normalization
used by the exact byte encoder. It trains each candidate with deterministic
augmentation, inverse-frequency class weights, early stopping, and learning-rate
reduction. Candidate artifacts are immutable and include their geometry,
parameter counts, lowering cost, validation metrics, and hashes.

The predeclared gate is 90% validation accuracy. Among passing candidates, the
smallest shared parameter count wins; validation accuracy and the candidate ID
break ties. The command does not lower this threshold after seeing results. It
also verifies empirical numerical parity between the Keras graph and the affine
representation consumed by DeepPoly/MILP. This check is not an IEEE-754 proof.

The official test split is not decoded or evaluated during search. Quantization,
ESBMC, and Android measurements are reported as `NOT_RUN`, `NOT_EVALUATED`, or
`NOT_MEASURED` until a candidate passes and is frozen.

## September 2026 pilot

The final-code pilot at `output/sign_source_model_search_20260922_v3` compared two
single-convolution candidates at 16x16 RGB:

| candidate | hidden ReLUs | parameters | validation accuracy |
|---|---:|---:|---:|
| Conv(4, 5x5, stride 4, SAME) | 64 | 3,099 | 49.21% |
| Conv(8, 5x5, stride 4, SAME) | 128 | 6,155 | 56.02% |

Neither candidate met the 90% gate, so neither was selected and no test-set,
formal-verification, or Android result was produced. This is a controlled
negative result: increasing the old model from two to eight filters and from 8x8
to 16x16 was insufficient. More epochs alone are unlikely to repair the gap,
because both curves plateaued.

Both candidates preserved every validation prediction after affine lowering;
the maximum absolute logit discrepancy was $2.29\times10^{-5}$. Thus the low
accuracy is not caused by the lowering adapter.

The next model experiment should add representational depth, preferably a second
restricted convolution/ReLU stage, while retaining an explicit affine-lowering
budget. That requires extending the restricted-CNN adapter and its C/ESBMC
equivalence tests before training the new architecture. A high-accuracy external
CNN or ensemble cannot simply replace this model unless every deployed operation
is given matching fixed-point and formal semantics.

## Multi-stage search

`preqbmc gtsrb search-deep-model` extends the same protocol to several
convolution stages followed by several dense stages, via
`RestrictedSequentialCNN`. Batch normalization is a training-only device: it is
folded into the preceding affine kernel and bias before the restricted object is
constructed, so the deployed semantics stay exactly affine/ReLU. The search also
gained a center-aligned nearest-neighbor resize, with matching Python and C index
formulas in `ByteImageEncoder`.

The ladder, all measured on validation only:

| stage | architecture | validation accuracy |
|---|---|---:|
| shallow 16x16 | Conv(4) / Conv(8), 1 stage | 49.21% / 56.02% |
| two-stage 32x32 | three candidates | 64.1% / 67.2% / 66.8% |
| three-stage 32x32 | `conv16_32_64_dense256` | 87.11% |
| three-stage, larger | `conv24_48_96_dense384` | 89.78% |
| three-stage, regularized | AdamW + dropout | 89.74% |
| **four-stage** | **`conv24_48_96_128_dense384`** | **91.11%** |

The 90% gate was held rather than relaxed at 89.78%, and that near-miss was
reproduced under a second seed before the architecture changed, so the
three-stage ceiling is repeatable and not a seed artifact.

## Held-out test evaluation

The test split is consumed exactly once, after selection, by

```bash
preqbmc gtsrb evaluate-deep-model \
  --search-output output/sign_deep_source_model_search_4stage_20260922 \
  --output output/sign_deep_source_model_test_<date>
```

The command refuses to run unless the search reports `SELECTED` and
`test_status: NOT_EVALUATED`, re-checks the SHA-256 of both the frozen study and
the folded parameter file, and writes with `open(..., "x")` so a report cannot be
silently re-rolled. Recorded at
[test_evaluation.json](../../output/sign_deep_source_model_test_20260922/test_evaluation.json):

| measurement | value |
|---|---:|
| validation accuracy used for selection | 91.11% |
| test accuracy, float32 source | 91.54% |
| test accuracy, folded affine model | 91.54% |
| balanced accuracy, test | 88.76% |
| generalization gap | −0.43 pp |
| test images | 12,630 |

Test accuracy exceeds validation accuracy, so the gate was not fit to the
validation split.

Folding changed **no prediction on any of the 12,630 test images**, with a
maximum absolute logit discrepancy of $4.39\times10^{-5}$. The batched
convolution path used for evaluation was additionally checked against
`RestrictedSequentialCNN.float_reference`, the independent sliding-window
reference, on 64 sampled test images, agreeing to $9.6\times10^{-6}$. Both are
tested numerical equivalences, not IEEE-754 proofs. Together they discharge the
affine lowering as a source of accuracy loss: the object PreQ-BMC consumes is the
object that scores 91.54%.

### The tail matters more than the headline

Balanced accuracy is 2.8 points below raw accuracy, and the per-class
distribution is uneven:

| class | meaning | test accuracy |
|---:|---|---:|
| 27 | pedestrians | 50.0% |
| 30 | beware ice/snow | 54.7% |
| 0 | speed limit 20 km/h | 65.0% |
| 18 | general caution | 70.5% |
| 21 | double curve | 73.3% |

Nineteen of the 43 classes fall below 90%. A local robustness certificate around
an image of a class the model classifies correctly half the time is a sound
statement about an unreliable classifier. Regions selected for the formal
campaign should therefore be stratified by class accuracy and the weak classes
reported, not avoided.

## Verification cost of the selected model

Accuracy and verifiability moved in opposite directions. Against the artifact
currently carried through ESBMC:

| | verified artifact | selected model | ratio |
|---|---:|---:|---:|
| ReLUs | 18 | 11,648 | 647x |
| hidden layers | 1 | 5 | 5x |
| multiply-accumulates | 1,260 | 2,443,392 | 1,939x |
| largest dense-lowered layer | 3,456 | 18,874,368 | 5,461x |

Two qualifications on the last row. It is a *representation* cost, not intrinsic
work: stage 0 is a 5x5x3 convolution with 75 nonzeros per neuron, so the dense
Toeplitz lowering is 97.6% zeros. The honest complexity ratio is the
multiply-accumulate row. Separately,
[sign_deep_source_model_search_4stage.json](../../experiments/sign_deep_source_model_search_4stage.json)
raises `max_affine_entries` to 30,000,000, against a 2,000,000 default on
`RestrictedSequentialCNN` and 1,000,000 on `lower_conv`. That guard was moved to
admit this model rather than met by it.

The harder obstacle is depth, not width. The open precision defect is
$R_\varepsilon \subseteq B_\varepsilon$: the interval box over-approximates the
reachable hidden set. On the single hidden layer of the current artifact that gap
was measured exactly at 152 ULPs (true reachable margin $-15$ against a box-level
margin of $+137$). Chaining compounds it, and the relational refinement that
would close it already exhausts a 1800 s budget at 18 ReLUs. Launching an ESBMC
campaign on 11,648 ReLUs should not be expected to return conclusive verdicts;
measuring where the wall falls, as a function of width and depth along the ladder
above, is the informative experiment.
