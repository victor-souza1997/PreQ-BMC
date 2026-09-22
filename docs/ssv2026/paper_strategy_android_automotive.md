# Paper strategy: bit-precise PreQ-BMC on Android Automotive

## Corrected positioning

The paper should study the feasibility and measured cost of deploying a
PreQ-BMC-verified fixed-point traffic-sign classifier on an Android Automotive
OS (AAOS) platform. The two indispensable elements are:

1. a local robustness certificate over the exact integer C arithmetic; and
2. execution of the identity-bound C artifact on an AAOS device, with replay
   parity and measured latency, memory and energy.

This is **not currently an int8 paper**. The frozen artifact uses shared h12
`<12,3,8>` and o14 `<14,5,8>` layer formats. Parameters and activations are
stored through the reference C backend in `int64_t`, and multiply-accumulate is
performed in `__int128`. Calling this artifact int8, claiming int32 accumulation,
or deriving a compression ratio from nominal Q would contradict the source that
was verified. TFLite PTQ int8 remains a useful **uncertified deployment
baseline**.

The stronger and defensible question is:

> What formal coverage and deployment cost result when the exact PreQ-BMC
> fixed-point program is transferred to an Android Automotive-class platform,
> relative to conventional float32 and int8 implementations that do not carry
> its bit-precise certificate?

This turns the non-int8 width into an experimental result rather than a defect:
the selected format is the precision/range required by the current methodology
and calibration cohort. A strict eight-bit constraint may be reported only as a
predeclared ablation after it is actually run; it must not silently replace the
selected program.

## Terminology

Android Auto and Android Automotive OS are different systems. Android Auto runs
on a phone and projects its user experience. AAOS runs on in-vehicle hardware.
The Khadas experiment may be called AAOS only when the installed image declares
`android.hardware.type.automotive`; otherwise report it as Android on an
automotive-class board. See the official AOSP distinction:
<https://source.android.com/docs/automotive/start/what_automotive>.

The classifier is an experimental ADAS workload hosted on the Android platform,
not a vehicle-level safety function and not an Android Auto application-category
claim. Camera capture, sign detection, image decoding and vehicle control are
outside the verified boundary.

## Replacement summary

This paper presents a methodology for deploying and evaluating bit-precise
verified fixed-point neural networks on an Android Automotive OS platform. The
case study is local traffic-sign classification on GTSRB under explicit compute,
memory and energy constraints. Rather than assuming a universal int8 format,
PreQ-BMC selects one shared `<Q,I,F>` format per layer and freezes one generated
C program before evaluation. In the current artifact, the hidden and output
formats are `<12,3,8>` and `<14,5,8>`.

For a selected byte-valued image x and raw-byte radius epsilon, the verified
input set is

    R(x,epsilon) = {x' in {0,...,255}^d : ||x'-x||_infinity <= epsilon}.

The byte encoder, fixed-point rounding, saturation, ReLU order and lowest-index
tie rule are part of the generated implementation. The source-model gate and
MILP preimages establish candidate contracts; ESBMC checks the integer C
contracts in neuron blocks using the same layer QIF. At the output, an optional
affine-residual refinement composes shared-input coefficients before interval
maximization and uses ESBMC-checked rounding and clamp lemmas. A region is
accepted only after source eligibility, preimage availability, every required
block, contract chaining, output comparisons and the byte-input bridge succeed.

The verified source is cross-compiled with the Android NDK and replayed on the
target using vectors already matched exactly by the independent Python integer
reference and host C builds. Device replay is empirical evidence that the
cross-compiled artifact agrees on the finite corpus; it is not a compiler proof.
The study compares its latency, throughput, memory and externally measured
energy with TFLite float32 and TFLite PTQ int8. An NNAPI/NPU arm is included only
when the delegate reports actual accelerator execution rather than CPU fallback;
its arithmetic remains uncertified. Current Android documentation marks NNAPI as
deprecated in Android 15, so the exact device/runtime version must be reported.

The result is a per-region local guarantee for the generated integer C program,
not global adversarial robustness, physical-sign robustness, or NPU equivalence.
Brightness, contrast, blur, patch and geometric transformations are not part of
the current formal input set and must not be claimed until separately encoded.

## What the current evidence establishes

The following results are usable now:

- The source float32 model obtains 41.7498% on all 12,630 frozen GTSRB test
  images. This is too low for an ADAS-quality claim; the study is a prototype.
- The selected h12/o14 C implementation obtains 41.6627%, a loss of 0.0871
  percentage points. Its prediction mismatch rate against float32 is 1.0610%.
- The generated C and independent Python integer implementation agree exactly
  on encoder, hidden values and logits for every test image at host `-O0` and
  `-O2`.
- The exact source-gate audit reports 22 source-eligible and 32 genuinely
  refuted image/radius regions out of 54. Refuted source regions contain no
  robustness property for quantization to transfer.
- In the matched `image6_eps4` pilot, the interval-box output check is
  `MARGIN_INCONCLUSIVE`; affine-residual refinement reaches `VERIFIED` with the
  same image, radius, QIF and generated program. The refinement costs additional
  ESBMC calls and time.
- No Android/AAOS, JNI, NPU, latency or power result has yet been measured.

The pilot is evidence that the refinement can recover one abstraction-limited
case. It is not enough to claim a general increase in certification rate.

## Minimum remaining experiments

### Formal campaign

Run the already prepared 54 matched configurations in
`output/sign_affine_residual_evaluation_20260921/study.json`: 27 fixed regions
under box-only verification and the same 27 under affine-residual refinement.
Report all configured regions, including source-refuted, timeout and incomplete
outcomes. The primary denominator is all fixed regions; also report the
conditional rate among source-eligible regions. Compare recovered regions,
ESBMC calls, wall time and peak query RSS. The two variants are paired outcomes,
not 54 independent samples.

At least three independent timing repetitions are needed for representative
easy, recovered and unresolved regions. Do not rerun formal proof outcomes merely
to inflate the region denominator.

### Model usefulness

Either retrain and freeze a materially more accurate source model, then repeat
format selection and every formal/device experiment, or explicitly frame the
current 41.75% model as a proof-of-concept. A robustness certificate on a
misclassified clean image is not useful; region selection therefore remains
clean-correct and frozen before solver outcomes. Improving the source model is
the most important task before claiming practical ADAS relevance.

### Android Automotive deployment

On the exact target image, record build fingerprint, Android/AAOS feature,
kernel, ABI, NDK, Clang, CPU topology and governor. Replay all 12,630 vectors and
require zero integer-logit mismatches. Then run at least ten independent process
trials per method and core assignment after warmup:

- PreQ-BMC generated C on the Cortex-A53 cluster;
- the same C on the Cortex-A73 cluster;
- TFLite float32 CPU;
- TFLite PTQ int8 CPU;
- TFLite PTQ int8 through NNAPI/NPU, only with verified delegate placement.

Use a randomized or alternating method order, fixed thermal conditions and fixed
screen/network state. Report median and IQR across trials, p95 latency within
trials, throughput, CPU-time/wall-time ratio, process RSS and binary/model bytes.
Core numbering must be measured on the installed image rather than inferred from
the SoC marketing description.

For power, use synchronized external-meter traces and paired idle windows. CPU
utilization is not an energy proxy. Report gross and idle-subtracted joules per
inference together with instrument sampling rate, calibration and uncertainty.

### Sanity adversarial evaluation

An empirical attack on the exact C artifact is useful as a bug-finding sanity
check, but it is not a substitute for ESBMC. Every attack point must remain in
the same raw-byte L-infinity set and be replayed through the generated C. Attack
survival is an upper bound on robust fraction; formal certification is the lower
bound. Do not add brightness or contrast to the theorem merely because they are
used in an empirical test.

## Commands now available

Prepare an identity-bound corpus from the completed host evaluation:

```sh
preqbmc gtsrb prepare-android \
  --host-quality output/sign_host_quality_final_20260921 \
  --certificate output/sign_milp_tightest_20260920/image7_eps2_beta1_cuts0/region_summary.json \
  --certificate output/sign_affine_residual_pilot_runs_20260921/image6_eps4_beta1_cuts0_affine_residual/region_summary.json \
  --output output/sign_android_bundle_new
```

Build the JNI library and native benchmark with the Android NDK:

```sh
cmake -S examples/ssv_android -B output/android_build \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DQNN_SOURCE="$PWD/output/sign_android_bundle_new/qnn.c"
cmake --build output/android_build --config Release
```

Replay and benchmark through `adb` (choose a core mask only after inspecting the
device topology):

```sh
preqbmc gtsrb run-android \
  --bundle output/sign_android_bundle_new \
  --binary output/android_build/preqbmc_ssv_benchmark \
  --output output/sign_android_device_a73 \
  --core-mask <measured-mask> --trials 10
```

The runner refuses parity mismatches, hashes the C, corpus and binary, records
whether the Android image declares the automotive feature, and leaves JNI, APK,
NNAPI/NPU and power fields `NOT_MEASURED`. The native benchmark measures
`qnn_forward_fixed` only; it does not include image decode, resize, JNI or UI.
Those costs need a separate application-level measurement.

## Decision

The current results are enough for a methods/prototype narrative and to justify
the next campaign. They are **not enough** for the supplied end-to-end AAOS
claims. The full matched formal campaign, actual AAOS replay, device baselines
and power measurements are mandatory. A higher-accuracy model is mandatory for
an ADAS-readiness claim, but can be deferred if the paper explicitly presents a
formal/deployment feasibility study instead.
