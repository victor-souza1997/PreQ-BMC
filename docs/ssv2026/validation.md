# Local validation, 2026-09-14

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
