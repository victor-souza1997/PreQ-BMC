# Shared quantization search

Use the existing trained model and frozen calibration cohort to search for an
integer program instead of manually trying a handful of fixed-F formats:

```sh
preqbmc gtsrb search \
  --config experiments/sign_qif_search.json \
  --output output/sign_qif_search
```

The search first computes the original DeepPoly source prerequisite for all
27 primary calibration regions. Source-inconclusive regions remain in
`source_regions.csv` and `search_manifest.json`; they are not counterexamples,
and changing quantization cannot change their eligibility. Every eligible
region remains required. A missing MILP property preimage stops selection;
it never removes a difficult region from the required set.

For each layer, integer-bit floors are derived from the extrema of weights,
biases, source bounds and MILP preimages, taking the maximum across eligible
regions. The example searches F=8..12 and integer padding 0 or 1 independently
in each layer, giving 100 network configurations. Each format obeys
Q=I+F+1. All neurons, blocks and regions share the same format for a given
layer; different layers may have different F. The input encoder follows the
first layer's format and is checked for box containment on every region.

Candidates are ordered by nominal affine parameter cost
`sum(parameter_count[layer] * Q[layer])`. This includes the expanded affine
convolution representation and is a search objective, not the actual C binary
or array size: the current exporter stores its parameters in int64 containers.
The first candidate that receives complete certificates over every eligible
calibration region is selected. The claim is the first certified candidate in
this bounded order, not a globally optimal quantization or an accuracy optimum.
Smaller candidates with TIMEOUT/UNKNOWN remain unresolved; no monotonicity of
precision, solver behavior or classification is assumed.

Every candidate is checked by the existing live CBC MILP-preimage and derived
ESBMC layer-contract pipeline. Chaining, nonvacuity, input containment and
generated C identity are acceptance conditions. Neuron blocking remains in
use, with sequential ESBMC calls and fail-fast rejection. A block or region
failure stops the candidate and records skipped regions; skipped work is never
counted as a certificate or an ESBMC failure. The underlying interval error
budgets, rounding, clamps, assertions and E2E policy are unchanged.

The source model and evaluation cohort must be fixed before search and use
the same preprocessing. Their image IDs must be disjoint. After selection,
`evaluation/study.json` contains the independent 36-run campaign and the same
immutable integer C program. Run it without further format selection:

```sh
preqbmc gtsrb run \
  --study output/sign_qif_search/evaluation/study.json \
  --output output/sign_qif_search_evaluation
```

Then run `PYTHONPATH=src python -m scripts.report_ssv_regions` and the full-test
host-quality command documented in the study protocol. Calibration coverage does not establish evaluation coverage,
accuracy, adversarial attack success rate or Android compiler/runtime parity.

## Preparation and resume

The following computes live source checks and preimages and writes the ordered
candidate domain without invoking ESBMC:

```sh
preqbmc gtsrb search \
  --config experiments/sign_qif_search.json \
  --output output/sign_qif_search \
  --prepare-only
```

Continue that prepared or interrupted search with:

```sh
preqbmc gtsrb search \
  --config experiments/sign_qif_search.json \
  --output output/sign_qif_search \
  --resume
```

Resume checks configuration, cohort, implementation and quantized artifact
identities, and reuses completed per-region reports. A different configuration
requires a new directory. Partial harness directories without final reports
require inspection; they are not overwritten or silently marked failed.
`search_summary.json`, `candidate_search.csv`, individual candidate summaries,
source records and ESBMC logs preserve the evidence and rejection reasons.
The command returns code 2 for inconclusive or exhausted searches, with reports
written normally; it does not claim the network is non-robust.
