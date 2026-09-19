# Traffic-sign study: revised scope and experiment protocol

## What the prototype establishes

This is a **fixed-geometry convolution lowered to affine form**, not a scalable
general CNN verifier or an ADAS-ready classifier. The existing model is
8x8x3 -> Conv(2, 3x3, stride 2, VALID) -> ReLU -> Dense(43), giving a
192 -> 18 -> 43 dense integer implementation. Its current full-test float32
accuracy is 41.75%; retain that number and frame this as a proof of concept.
No retraining is required for the campaign below. A larger model might improve
accuracy, but neither its accuracy nor its proof runtime has been measured.

Dense lowering has an explicit size cap. The stored integer parameter arrays
use int64, not bit-packed Q-bit storage. For this model, 4,291 expanded
parameters occupy 34,328 bytes, versus 3,492 bytes for 873 shared float32
parameters. This approximately 9.83x ratio concerns parameter arrays, not APK
size, process memory, or energy. Do not claim compression from nominal Q alone.

## Freeze the program before comparing regions

F=8 alone did not freeze the integer program: I affects clamps and previously
varied with a region. The new `fixed_qif_check` mode accepts an explicit format
per layer and runs the existing source gate, MILP preimages, derived contracts,
chaining, vacuity and ESBMC checks **without format search or integer-bit jumps**.
Relational proof refinement, when enabled, cannot change these formats.

For an already trained study, freeze a new campaign without modifying the
original study, selected images, or historical results:

```sh
PYTHONPATH=src python -m scripts.freeze_ssv_campaign \
  --study output/plate_experiments/study.json \
  --qif 11,2,8 13,4,8 --output output/sign_fixed_campaign
preqbmc gtsrb run --study output/sign_fixed_campaign/study.json \
  --output output/sign_fixed_runs --dry-run
```

The example formats are the existing image8 pilot's formats, not an assurance
that every region will verify. Their post-pilot origin must be disclosed. If
formats are changed later, create a new artifact and recheck **every** region;
do not union certificates belonging to different programs.

The manifest freezes both Q/I/F entries, model parameter hash, encoder and C
source hash. Checking regenerates and compares the source before proving;
successful region exports must have exactly the same source hash. A source
inconclusive run still records the requested artifact identity. A frozen
artifact has `proof_status=NOT_RUN`: freezing alone is not a certificate.

The campaign retains all nine previously selected images, with no replacement:

| Arm | Images | Radii in raw byte levels | Blocks | Cuts | Runs |
| --- | --- | --- | --- | --- | --- |
| Main | All nine | 1, 2, 4 | 1 | off | 27 |
| Block controls | First frozen image in each tertile | 1 | 0, 2 | off | 6 |
| Refinement control | Same three images | 1 | 1 | on | 3 |

Remove `--dry-run` to execute; `--only image8_eps1_beta1_cuts0` selects one
validation run. New output directories are required. The old `plate` names
are preserved in historical files and checksummed provenance; new preparation
defaults use `experiments/sign_experiments.json` and `output/sign_experiments`.
Do not run preparation again just to rename an existing study or retrain it.

To obtain a second, disjoint descriptive cohort without retraining or changing
the source weights, derive it at a predeclared per-tertile rank offset and then
freeze the same fixed program again. `rank_offset_per_stratum=3` is a selection
rule, not a proof-outcome-dependent replacement rule:

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python -m scripts.freeze_ssv_cohort \
  --study output/plate_experiments/study.json \
  --rank-offset-per-stratum 3 --output output/sign_second_cohort
PYTHONPATH=src python -m scripts.freeze_ssv_campaign \
  --study output/sign_second_cohort/study.json \
  --qif 11,2,8 13,4,8 --output output/sign_second_fixed_campaign
preqbmc gtsrb run --study output/sign_second_fixed_campaign/study.json \
  --output output/sign_second_fixed_runs --dry-run
```

The cohort command refuses overlap with its parent selection and records both
parent/source-model hashes. The corresponding declarative specification is
`experiments/sign_second_cohort_experiments.json`; it is a provenance record
for the derivation command, not a request to retrain a second model.

## The proposed exact MILP source gate is not yet implemented

The current source gate remains unchanged and conservative in its decision
policy: DeepPoly inconclusive does not become a source counterexample or
permission to bypass the prerequisite. Its numerical source-model reasoning
is **not an IEEE-equivalence proof** of Keras convolution execution.

A ReLU MILP encodes a real-arithmetic network. CBC uses numerical arithmetic,
feasibility tolerances and branch-and-bound; an `OPTIMAL` or `INFEASIBLE` result
alone is not an independently checked exact certificate, and it does not model
Keras floating-point accumulation, reassociation, or device kernels. Eighteen
ReLUs are small enough to try, not a guarantee of a trivial solve. A continuous
MILP witness may also be outside the byte lattice or inconsistent with repeated
resize coordinates. Only two pilot outcomes do not establish that an entire
margin tertile is ineligible.

Before allowing such a fallback to set source eligibility, specify its theorem:

1. For a real-arithmetic source model, validate exact rational certificates or
   use an exact decision procedure, with exact coefficient and domain encoding.
2. For an IEEE source implementation, model its operations/order or prove a
   sound roundoff envelope and subtract that envelope from the certified margin.
3. Validate any alleged counterexample in the actual source implementation and
   declared input domain. Numerical solver output alone is a candidate witness.

No CBC-only fallback was added to the acceptance path. This remains an open
item, not an implemented "soundness-neutral" improvement. The final integer-C
contracts and input bridge have not been weakened to increase the pass count.

## Threat model and adversarial evaluation

The property is local prediction preservation for cropped RGB byte arrays:

    R(x,e) = {x' in {0,...,255}^(HWC) : ||x'-x||_infinity <= e}.

The encoder is top-left nearest-neighbor selection followed by /256 scaling
and the declared fixed-point rounding/clamp. It excludes decoding, sign
detection, stickers, occlusion, viewpoint and illumination transformations.
Use "local robustness evidence", not a "secure classifier" or vehicle safety
claim. The existing JNI interface calls the C encoder; do not insert Android
bitmap resizing or another normalization before it.

Square Attack against the compiled C logits is an appropriate first empirical
test; use a pinned existing implementation and record its query budget and
seed. All attacks must run against the same frozen artifact, use valid bytes
inside the declared radius, and preserve lowest-index argmax. Independent
8x8 pixel perturbations can be lifted to crop perturbations when sampled source
coordinates are distinct. For upsampling/repeated coordinates, enforce equality
constraints and replay the lifted raw crop: equivalence is not unconditional.

For the same fixed program, property and finite cohort, certified fraction is a
lower bound and attack-surviving fraction an upper bound on its true robust
fraction. An inside-region attack that defeats a claimed certificate demands
investigation, not deletion from the cohort. Attack survival is not a proof.

Nine selected clean-correct images support a **descriptive selected-cohort
certification fraction**, not an estimate of population robustness. Report
full-test clean accuracy separately, disclose clean-correct and margin-based
selection, retain source-inconclusive regions in the primary denominator, and
report the source-eligible conditional fraction separately. Different radii and
ablation runs are not independent images.

## Named baselines and compiler boundary

Use **TFLite post-training int8** and **TFLite float32** of the same shared source
coefficients as named device baselines. The exporter calibrates only on a
deterministic training subset, exports both `.tflite` artifacts, records versions
and hashes, and optionally evaluates the official test split:

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python -m scripts.export_ssv_tflite \
  --study output/sign_fixed_campaign/study.json \
  --output output/sign_tflite_baselines --evaluate-test
```

This uses the upstream [TFLite converter API](https://ai.google.dev/edge/api/tflite/python/tf/lite/TFLiteConverter)
with representative data and int8-only operators. TFLite uses native convolution
and different quantization arithmetic; it is an **uncertified implementation
baseline**, not the certified C artifact or an ablation of QIF alone. The
existing uniform Q16/F8 C baseline remains an internal arithmetic control, not
the named paper baseline. Android execution of TFLite is still a device task.

`scripts.evaluate_ssv_host` now checks the encoder, hidden layer and final logits
at both `-O0` and `-O2`. Device tests must also compare actual raw-crop hashes,
encoded inputs and integer outputs using the identical C encoder. Record NDK,
compiler, ABI, flags and binary hashes. Compiler correctness remains a trust
assumption; finite parity tests do not establish universal equivalence.

The attachment's overflow flag statement is incorrect for this configuration:
`paper-z3` does **not** add `--overflow-check`; only `safety`/`overflow` profiles
do. The current pipeline has separate exact-integer range checks for candidate
arithmetic. Describe these checks and actual recorded commands, not an absent
flag or a blanket proof that all generated C/JNI code is free of undefined
behavior. Host certificates do not automatically prove Android transfer.

## Energy protocol

Neither sub-microsecond inference nor a 2-3 hour campaign is established by
the model size. Measure before making either claim. Do not infer joules from
CPU load. The Android settings in the new JSON are **protocol targets**, not
an implemented device workload runner.

Use at least ten paired trials per artifact, alternating/randomizing method
order. Warm up until stable, start with 100,000 inferences per batch, and adjust
batch size so the trial spans at least ten seconds and 100 meter samples.
These are initial measurement choices, not universal resolution guarantees.
Consume outputs so the compiler cannot remove repeated inference. Include
preprocessing equally across methods, or report it separately for all methods.

Measure matched idle windows immediately around each trial, record thermal,
CPU governor, frequency, screen and charging state, and synchronize meter and
workload timestamps. Report gross board energy and paired incremental energy:

    E_incremental/inference = (integral_active P(t) dt - P_idle * T) / K.

Use independent paired trials, not meter samples or K loop iterations, as the
statistical unit. `summarize_energy_trials` provides a seeded exploratory 95%
bootstrap interval; also report meter calibration, resolution and uncertainty
separately. It retains negative deltas rather than clipping them. If the interval
or instrument uncertainty does not resolve a positive increment, report it as
unresolved. The tool cannot establish that resolution without real traces.

No runtime certificate-membership guard is required or added. No attack,
Android parity, latency or power result should appear as measured until it is
actually collected. Novelty claims such as "no prior tool closes the loop"
require a separate literature-supported comparison and are not established by
this implementation.

## Large fixed-artifact evaluation

`experiments/sign_fixed_qif_sweep.json` prepares a two-stage, at-most
144-configuration study. It records all 27 primary calibration regions for
each of four ordered immutable QIF candidates (nine frozen images times three
byte radii, beta=1 and cuts off): at most 108 runs. The first candidate anchors
the QIF-independent source gate. A candidate is selected only if it verifies
every calibration region for which that source gate is VERIFIED; source-
inconclusive regions remain in the primary report but cannot select a QIF
because the quantized/ESBMC stage is intentionally not entered. The tool then
materializes, but does not execute, 36 independent evaluation runs. The second
cohort is excluded from format selection.

The fixed F=8 candidates increase range together:

| Candidate | Hidden Q/I/F | Output Q/I/F |
| --- | --- | --- |
| `h11_o13_f8` | 11/2/8 | 13/4/8 |
| `h12_o14_f8` | 12/3/8 | 14/5/8 |
| `h13_o15_f8` | 13/4/8 | 15/6/8 |
| `h14_o16_f8` | 14/5/8 | 16/7/8 |

Run this from the repository root. The first command only writes manifests;
it does not invoke ESBMC:

```sh
preqbmc gtsrb prepare-sweep \
  --config experiments/sign_fixed_qif_sweep.json \
  --output output/sign_fixed_qif_sweep
```

Run each calibration candidate in a fresh directory. Set `CUDA_VISIBLE_DEVICES=-1`
when a reproducible CPU-only TensorFlow setup is intended:

```sh
CUDA_VISIBLE_DEVICES=-1 preqbmc gtsrb run \
  --study output/sign_fixed_qif_sweep/calibration/h11_o13_f8/study.json \
  --output output/sign_fixed_qif_sweep/runs/h11_o13_f8
CUDA_VISIBLE_DEVICES=-1 preqbmc gtsrb run \
  --study output/sign_fixed_qif_sweep/calibration/h12_o14_f8/study.json \
  --output output/sign_fixed_qif_sweep/runs/h12_o14_f8
CUDA_VISIBLE_DEVICES=-1 preqbmc gtsrb run \
  --study output/sign_fixed_qif_sweep/calibration/h13_o15_f8/study.json \
  --output output/sign_fixed_qif_sweep/runs/h13_o15_f8
CUDA_VISIBLE_DEVICES=-1 preqbmc gtsrb run \
  --study output/sign_fixed_qif_sweep/calibration/h14_o16_f8/study.json \
  --output output/sign_fixed_qif_sweep/runs/h14_o16_f8
```

After the first candidate has a complete 27-region report, run the selector.
Its `source_eligibility_anchor.source_eligible_run_ids` field identifies the
only calibration run IDs that later candidates must complete; source-inconclusive
rows remain reported but do not enter the QIF-selection predicate. Run those
IDs with repeated `preqbmc gtsrb run --only <run-id>`, then rerun the selector. It selects
the first candidate that verifies every source-eligible calibration region and
produces the independent 36-run evaluation manifest:

```sh
preqbmc gtsrb select-sweep \
  --sweep output/sign_fixed_qif_sweep/sweep.json \
  --runs-root output/sign_fixed_qif_sweep/runs \
  --output output/sign_fixed_qif_selection
```

If no candidate passes all source-eligible calibration regions, this command produces a
selection record with `NO_CANDIDATE_PASSED_CALIBRATION` and does not create an
evaluation artifact. This is an informative result, not permission to choose
a different format per evaluation region.
