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
