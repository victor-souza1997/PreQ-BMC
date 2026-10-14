# Int16 and int8 QNN candidates for SSV

These are isolated validation experiments, not replacements for the frozen
deployment or its pre-registered test campaign. No `src/` file was changed.
The entry point is `tools/ssv2026/quantization_candidates.py`.

## Result and recommendation

MEASURED on all 7,830 frozen validation images, not on held-out test images:

| candidate | global format | validation accuracy | compact parameter storage | internal buffers | quality decision |
| --- | --- | --- | --- | --- | --- |
| int16 | Q7.8 | 7,066/7,830 = 90.24% | 614,422 bytes | 24,576 bytes | GO for verification experiments |
| int8 | Q3.4 | 6,323/7,830 = 80.75% | 307,211 bytes | 12,288 bytes | NO-GO for the 90% SSV accuracy requirement |

Sources: `output/ssv_quantization_candidates_20260927/int16_validation/validation_summary.json:7`
and `output/ssv_quantization_candidates_20260927/int8_validation/validation_summary.json:7`.
Storage counts are recorded in each candidate's `manifest.json:205`; the baseline
has 2,457,688 parameter bytes and 98,304 internal-buffer bytes.

MEASURED: both candidates have zero compact-versus-wide C mismatches across
all validation images, zero Python/C encoder mismatches, exact NumPy reference
parity on 12 inputs (nine synthetic edge/crop cases plus three validation
images), and exact independent Python-bigint parity on one validation image.
Undefined-behavior/signed-overflow diagnostics also passed on 12 full-range
synthetic inputs per candidate. These checks are measurements, not proofs.
See the two validation reports, lines 14–17 and their `sanitizer` fields.
Seven small standalone regressions additionally check both widths, every input
byte, signed extremes, changed-generator rejection and parameter range checks:
`tools/ssv2026/test_quantization_candidates.py:20`.

GUESS: compact storage could help deployment size and some monolithic formulas.
There is **no measured reduction in complete-verification memory here**, and
no whole-network certificate. The existing CROWN affine harnesses still use
128-bit arithmetic; smaller deployment arrays do not automatically shrink them.

## Exact candidate semantics

The int16 numerical model is the existing Q7.8 model, with 16-bit parameter
arrays and internal buffers instead of 64-bit arrays. Its wide-storage reference
C must reproduce the pinned deployment SHA-256 exactly; otherwise build fails
(`tools/ssv2026/quantization_candidates.py:168`).

The int8 candidate is a **new numerical model**, quantized from the frozen
folded floating-point weights using one global signed Q3.4 format. It is not an
int8 cast of existing Q7.8 weights, and no existing robustness certificate is
reused. Parameters are quantized with half-away rounding and saturation;
no weights/biases were clipped in these two builds. Hidden activations can
still saturate, and input/weight precision changes can affect accuracy.

For both, an RGB crop is resized with nearest-center sampling. Input bytes are
normalized by 256 and rounded into the candidate's declared fixed-point format:

```text
input = clamp_Q(round_half_away(byte * 2^F / 256))
acc   = sum((__int128)weight * (__int128)input)
value = clamp_Q(round_half_away(acc / 2^F) + bias)
hidden = max(value, 0)       # no ReLU at the final output
```

Multiplication is widened **before** it happens; accumulators stay `__int128`.
Never change them to int8/int16. The shared kernel is unchanged
(`src/verification/arith_kernel.py:23`); generated layers clamp before storing
(`src/backends/conv_fixed_point.py:285`).

ASSUMED input/API contract: the public encoder/forward ABI remains `int64_t*`
for compatibility, while parameters and internal activation buffers are narrow.
Call `qnn_encode_bytes` before `qnn_forward_fixed`, or supply integers within
the declared signed input range. Out-of-range direct forward inputs are not
covered by the storage-equivalence claim
(`tools/ssv2026/quantization_candidates.py:121`).

## Reproduce build and validation

Run from the repository root, with Python 3.10+, NumPy, Pillow and GCC available.
TensorFlow, a GPU and training are not needed. The frozen model, search summary,
study manifest and local GTSRB files referenced by the existing validation
configuration must exist. The script verifies their hashes before decoding.
It never decodes test images or reads the test-evaluation report.

Every output directory must be **new**. These example names differ from the
already completed run; choose another suffix when rerunning.

```bash
python tools/ssv2026/test_quantization_candidates.py

python tools/ssv2026/quantization_candidates.py build \
  --output output/ssv_quantization_candidates_repro

python tools/ssv2026/quantization_candidates.py validate \
  --artifact output/ssv_quantization_candidates_repro/int16_f8 \
  --output output/ssv_quantization_candidates_repro/int16_validation

python tools/ssv2026/quantization_candidates.py validate \
  --artifact output/ssv_quantization_candidates_repro/int8_f4 \
  --output output/ssv_quantization_candidates_repro/int8_validation
```

Validation is single-threaded, processes one image at a time, and refuses to
overwrite reports. `--limit 64 --bigint-images 0` gives a quick smoke check;
partial validation cannot pass the promotion gate. To strengthen independent
parity, use `--reference-images 7830` and/or increase `--bigint-images`.
Do not interpret validation command exit 0 as the accuracy gate passing:
check `candidate_quality_eligible` in `validation_summary.json`. The int8 run
completed correctly but that field is false.

An optional `build --int8-fractional-bits 3` creates Q4.3 instead of Q3.4, using
a fresh output directory. This is validation-only tuning, not an accuracy
guarantee. Do not tune or repeatedly evaluate on the frozen test split.

Each artifact contains `qnn_compact.c/.so`, `qnn_reference.c/.so`, signed
8/16-bit `quantized_parameters.npz`, and `manifest.json` with format, provenance,
file/source hashes, compiler commands and conservative accumulator envelopes.
Reports do not mutate that manifest. Outputs are git-ignored; archive them
explicitly when packaging the SSV artifact.

If `src/` changes after the freeze, old source pins intentionally prevent a
candidate from running with a different pipeline. Rebuild into a new directory,
revalidate, and substitute those new artifact/report paths in the commands.
Do not bypass a hash mismatch or modify the old manifests.

## Verification smoke checks

ESBMC is resolved from `PREQBMC_ESBMC`, the repository-local installation, or
PATH. Existing repository setup also honors its configured local executable.
The wrapper runs one job at a time, caps each at 2 GiB, requires at least 8 GiB
available to launch, and stops a running process if availability falls below
6 GiB. It retains interval-analysis simplification, bit-vector semantics,
unwinding assertions/default safety checks, and adds `--overflow-check`.

```bash
free -g
python tools/ssv2026/quantization_candidates.py verify \
  --artifact output/ssv_quantization_candidates_20260927/int16_f8 \
  --output output/ssv_quantization_candidates_20260927/int16_smoke_next \
  --run-esbmc --timeout 60

python tools/ssv2026/quantization_candidates.py verify \
  --artifact output/ssv_quantization_candidates_20260927/int8_f4 \
  --output output/ssv_quantization_candidates_20260927/int8_smoke_next \
  --run-esbmc --timeout 60
```

Without `--run-esbmc`, this only renders the harnesses. The default sample is
validation image 3652, identified by its frozen ID and SHA-256. The four checks
are storage/cast and widened-operand identity, a deliberately wrong
storage assertion (must FAIL), four first-layer interval outputs using the
existing pipeline renderer, and symbolic byte-encoder monotonicity with four
endpoint replays. `smoke_passed` is true only if every result matches its
expected outcome. It still reports `PARTIAL_NOT_CERTIFIED`.

PROVED, limited scope: final smoke checks passed for **both** formats. Each
storage/cast lemma, four-output interval harness and byte-encoder harness
returned VERIFIED; each deliberately bad storage control returned FAILED.
See `output/ssv_quantization_candidates_20260927/int16_proof_smoke_final/proof_smoke_summary.json:13`
and `output/ssv_quantization_candidates_20260927/int8_proof_smoke_final/proof_smoke_summary.json:13`.
These are local obligations, not network robustness certificates.

Earlier int16 experiments combined storage with symbolic rounding or symbolic
128-bit multiplication and timed out at 30 seconds. An intervening launch was
skipped at about 7.8 GiB available. The final lemma proves legal narrow casts,
their widened operand identity, and a post-clamp cast for an arbitrary 128-bit
pre-clamp value. MAC and rounding code are unchanged, not replaced or disabled.
Earlier inconclusive logs remain under `int16_proof_smoke/`,
`int16_proof_smoke_v2/` and `int16_proof_smoke_v3/`; none is counted as success.

## Bounded verification of the actual compact C

Add `--direct` to generate and optionally run a monolithic harness against
`qnn_compact.c`, with its own hash and a strict margin over all 42 competitors:

```bash
python tools/ssv2026/quantization_candidates.py verify \
  --artifact output/ssv_quantization_candidates_20260927/int16_f8 \
  --output output/ssv_quantization_candidates_20260927/int16_direct_next \
  --run-esbmc --direct --timeout 60
```

The result is in `direct_compact_summary.json`, including sampled process-tree
RSS. The domain is a raw uint8 epsilon-1 box on the **resized** 32x32 RGB crop;
the exact C encoder and network execute in the harness. Binding an arbitrary
original-size crop/decoder to this resized domain is outside this harness's
scope. A direct VERIFIED result would certify only that stated validation
domain. TIMEOUT, MEMOUT, UNKNOWN and skipped launches never certify anything.
This path may still suffer memory explosion; it is a bounded feasibility test.

MEASURED: both direct checks timed out under `--memlimit 2g --timeout 30`:

| actual compact C | elapsed seconds | sampled peak process-tree RSS | result |
| --- | --- | --- | --- |
| int16 Q7.8 | 30.16 | 2,196,803,584 bytes (about 2.05 GiB) | TIMEOUT, no certificate |
| int8 Q3.4 | 30.07 | 2,196,766,720 bytes (about 2.05 GiB) | TIMEOUT, no certificate |

Sources: `output/ssv_quantization_candidates_20260927/int16_proof_smoke_v3/direct_compact_summary.json:30`
and `output/ssv_quantization_candidates_20260927/int8_proof_smoke_final/direct_compact_summary.json:30`.
Process-tree RSS includes overhead beyond the ESBMC memory budget. Both runs
are censored at the same limits, so this does not establish equal uncapped
memory requirements. Narrow storage did **not** make full-network verification
succeed within this budget. It does not prove verification impossible under
larger budgets or a different decomposition.

## Optional complete generic pipeline

The generic interval coordinator accepts both numerical formats
(`src/verification/conv_proof.py:604`), but the current CROWN v2 path does not:
its deployment parser accepts only `int64_t`/`int` arrays
(`src/verification/crown_chain_v2.py:101`) and its arithmetic/certificate format
is tied to Q7.8. Do not pass compact C or an int8 candidate to the existing
CROWN protocol, patch its deployment hash, or reuse the original ledger.

After the competing campaign finishes, this command runs the full generic
**wide-storage reference** pipeline on validation image 3652, one job at a
time, with blocks of four, overflow checks, fresh private cache and no omitted
layer obligations:

```bash
python tools/ssv2026/quantization_candidates.py verify \
  --artifact output/ssv_quantization_candidates_20260927/int16_f8 \
  --output output/ssv_quantization_candidates_20260927/int16_complete_next \
  --run-esbmc --complete --timeout 60 \
  --validation-report output/ssv_quantization_candidates_20260927/int16_validation/validation_summary.json
```

This potentially expensive path was **not run** in the current shared-machine
session. It first requires successful smoke checks, a matching eligible full
validation report and a correctly classified center. Int8 Q3.4 is rejected by
the quality gate. Inconclusive output margins remain inconclusive.

ASSUMED trust boundary: model loading, Python quantization, topology/export
mapping, hashing and composition remain trusted components. Local cast lemmas
and empirical parity do not prove the complete Python transformation. Thus,
even a complete reference proof leaves compact deployment status
`NOT_CERTIFIED_EXPORT_BINDING_REQUIRED`; only a successful direct check or a
proper export-binding proof can close that gap. The frozen original source and
its already recorded results remain untouched.

## Reporting in the SSV paper

Keep the current pre-registered primary test results unchanged. Report this
experiment as a validation-only quantization/storage ablation, including the
failed int8 accuracy gate and all resource-limited outcomes. Treat sanitizer,
parity, accuracy and storage as MEASURED; only successful ESBMC obligations are
PROVED, within their exact scopes. Export/composition dependencies are ASSUMED;
any predicted memory advantage is GUESS until measured under equal budgets.

Do not claim 90.24% as held-out test accuracy, claim a whole-network proof from
smoke checks, or transfer an old certificate to a new C hash. Any later paper
test run for a newly selected numerical model needs an explicitly revised,
frozen protocol rather than silently rerunning the original cohort.
