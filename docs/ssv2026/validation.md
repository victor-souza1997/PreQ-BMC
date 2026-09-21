# Local validation

## Exact source-gate audit, 2026-09-20

`output/sign_source_gate_audit_20260920/source_audit.json` contains a source-only
audit of the same 54 distinct image/radius pairs across the frozen calibration
and evaluation cohorts. The configured DeepPoly-first MILP gate reported 22
`VERIFIED` source regions and 32 `REFUTED` regions with float-model replayed
counterexamples; no solve was `UNKNOWN`. All ten distinct regions with prior
final `VERIFIED` reports retained a `VERIFIED` source prerequisite. This audit
invoked **no ESBMC query** and does not establish new integer-C certificates.
`output/sign_qif_source_gate_review_20260920/search_manifest.json` recorded
11 source-eligible calibration regions with available MILP property preimages;
the quantization/ESBMC search was prepared but not executed at this checkpoint.

One newly eligible median-margin region, `image3_eps1_beta1_cuts0`, was then
checked with the frozen `h12_o14_f8` artifact. Its DeepPoly margin lower bound
was -0.23485, while the source MILP bound was +0.16526. The fixed-format run
reported `final_status=VERIFIED`, `guarantee_level=byte-crop-integer-C`,
successful chaining and input bridge, and 60 `VERIFIED` ESBMC calls plus two
expected sentinel failures (`output/sign_milp_source_h12_image3_pilot_20260920`).
This is one local image/radius certificate, not a cohort certification rate.
An independently run low-margin smoke case, `image0_eps1_beta1_cuts0`, exited
with `SOURCE_PROPERTY_REFUTED` and did not enter ESBMC
(`output/sign_milp_source_h12_refuted_smoke_20260920`).


## Fixed-artifact follow-up, 2026-09-18

No model was retrained and no historical results were overwritten.

- Frozen campaign: `output/sign_fixed_campaign_review/study.json`, 36 planned
  configurations of the same nine previously selected images. Only the two
  validation checks below were run, not the full campaign.
- Artifact: Q/I/F = (11,2,8), (13,4,8), generated C SHA-256
  `14b00fcd2dc81da0a265bc97b9a2ff87bac1c04e1e93019dcc42d4cdb31d9394`.
- `output/sign_fixed_check_review/image8_eps1_beta1_cuts0/region_summary.json`:
  VERIFIED, byte-crop integer-C guarantee, input bridge and chaining passed,
  227.06 seconds in one run. 62 ESBMC calls: 60 VERIFIED and two expected
  sentinel failures. Source hash exactly matches the frozen artifact and the
  original image8 successful pilot. This is a format-check run, not synthesis.
- `output/sign_fixed_source_gate_review/image0_eps1_beta1_cuts0/region_summary.json`:
  SOURCE_PROPERTY_INCONCLUSIVE, zero ESBMC calls, same requested artifact hash
  and Q/I/F retained. No source failure or source counterexample is claimed.
- `output/sign_fixed_check_tables_review` contains additional CSVs and a ledger
  for the image8 check only; its completed-only fraction is not a campaign result.
- 77 focused tests passed, including format identity/drift, no-search checking,
  source gate, generated C, fixed-point execution, block/no-saturation checks,
  verified-bound tightening, -O0/-O2 host parity on synthetic data, and real
  TFLite conversion/inference on a synthetic fixture.
- `output/sign_tflite_baselines_review` contains actual float32 and PTQ int8
  TFLite exports of the trained shared model, calibrated on 500 training images.
  Full-test baseline accuracy and device execution were not measured in this
  follow-up. All baseline artifacts are explicitly uncertified.
- Android parity, latency and energy remain NOT_MEASURED. The exact source
  MILP fallback remains unimplemented; see [review_response.md](review_response.md).
- `output/sign_second_cohort_review/study.json` freezes a second nine-image
  cohort with `rank_offset_per_stratum=3`; it has no image overlap with the
  original selected cohort and reuses the same source-model hash. Its matching
  36-run fixed-artifact campaign is
  `output/sign_second_fixed_campaign_review/study.json`. Only `--dry-run` was
  executed for this second cohort; it has no verification outcomes yet.
- `output/sign_fixed_qif_sweep_review/sweep.json` materializes the four
  candidate calibration studies for the 108+36 fixed-QIF protocol. It contains
  four 27-region calibration manifests and no ESBMC result. The selector has
  not run, and independent evaluation is intentionally not materialized yet.

## Initial tiny gate, 2026-09-14

The following records the environment and evidence **at that checkpoint**;
the dataset availability statement is historical, not the current state.

The authoritative tiny spatial-CNN run is
`output/ssv2026_tiny_cnn/gate_summary.json`. Earlier uniquely named gate
directories were retained as development evidence, not additional regions.

- Python 3.10.19 in the existing `quad` environment; ESBMC 7.11.0, Z3 profile,
  30-second query timeout, 6g memory limit, sequential queries.
- Spatial 2x2 convolution with overlapping receptive fields and SAME padding,
  2x2x1 input, four hidden outputs, two logits.
- All 81 inputs in the byte box [0,2]^4 matched independent integer convolution
  and compiled C at both layers. Host UBSan passed.
- True classification assertion VERIFIED; deliberately false target FAILED,
  with a counterexample replayed in the direct integer reference.
- Existing CBC MILP-preimage / derived-layer path VERIFIED for beta=0,1,2.
  Input coverage and chaining passed. Every variant selected the same
  Q=10, I=1, F=8 in both layers; the exported and sanity-checked model uses
  these same formats.
- 40 focused tests passed: the two SSV modules, C generation, fixed-point
  forward execution, verified-bound tightening, and solver backend tests.
- Float TensorFlow convolution versus direct lowering was tested for SAME
  and VALID padding and strides 1 and 2. This is not a formal IEEE proof.
- GTSRB configuration validation passed: 8x8x3 input, 3x3 convolution with
  two filters and stride 2, 18 hidden activations, 43 output classes.
- Full-test host evaluation plumbing was tested with a synthetic RGB fixture;
  no GTSRB accuracy result was produced.

No official GTSRB files are present at `data/gtsrb`, so training and the real
GTSRB region campaign were not run. No Android NDK is configured. An attempted
host JNI build also found no JNI headers or `javac` in the installed Java
runtime; only the generated non-JNI C was built successfully. Thus JNI/arm64
parity, device accuracy/performance and power are **NOT_MEASURED**.

The shared exporter intentionally retains unused Q/I/F audit constants;
the Android CMake target suppresses only that unused-constant warning for the
generated source, not arithmetic or other safety diagnostics. No formal
verification flags or legacy benchmark semantics were changed.

The result is a successful restricted prototype gate and study preparation,
not completion of the requested physical-device research evaluation.
