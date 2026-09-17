# Restricted CNN study: what is available

Read [feasibility.md](feasibility.md) first. This is an opt-in research prototype,
not general CNN/ONNX support. Existing article configurations are unchanged.

## Validated tiny gate

Use an environment with `pip install -e '.[gtsrb]'`, a host C compiler and
ESBMC on PATH. The local validation used the existing `quad` environment.
Run from the repository root; every output directory must be new:

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python -m scripts.run_ssv_cnn_gate \
  --output output/ssv_cnn_gate_new --timeout 60
PYTHONPATH=src MPLCONFIGDIR=/tmp/preqbmc-mpl python -m scripts.report_ssv_gate \
  --report output/ssv_cnn_gate_new/gate_summary.json \
  --output output/ssv_cnn_tables_new
```

The gate enumerates 81 inputs of a 2x2x1 synthetic Conv/ReLU/Flatten/Dense
classifier. It checks direct integer convolution against host C at both layers,
runs UBSan, proves a true property, replays a deliberately false property's
counterexample, and executes **the existing CBC MILP-preimage / derived-layer
ESBMC pipeline** at beta=0,1,2. Candidate QIF must be shared per layer and match
across these controls. The auxiliary exact-network true/false test is a separate
sanity check, not an E2E replacement for the preimage method.

Results and tables from this gate are explicitly **synthetic**, one region and
one timing observation per variant. They cannot support GTSRB accuracy,
scalability, or device-performance conclusions. The original CNN parameter
sharing is represented by repeated equal weights in a bounded dense lowering;
this is intentionally not an efficient convolution backend.

## GTSRB preparation and execution

The [official dataset page](https://benchmark.ini.rub.de/gtsrb_dataset.html)
describes cropped traffic signs. Download the official training images, final
test images and final test labels, review their terms, and extract them into:

```text
data/gtsrb/Final_Training/Images/00000/GT-00000.csv
data/gtsrb/Final_Training/Images/00000/<track>_<image>.ppm
... classes 00000 through 00042 ...
data/gtsrb/Final_Test/Images/<image>.ppm
data/gtsrb/GT-final_test.csv
```

No archive, license grant, trained weights or sample IDs are fabricated.
The preparation command checks every image hash, creates a deterministic
track-disjoint 80/20 training/validation split, trains one tiny 8x8 RGB CNN,
checks float convolution/lowering predictions on the full official test split,
then freezes nine correctly classified images in three clean-margin tertiles.
It requires multiple classes. Selection never consults ESBMC outcomes.

```sh
PYTHONPATH=src python -m scripts.prepare_ssv_gtsrb --check-config
PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python -m scripts.prepare_ssv_gtsrb \
  --config experiments/ssv2026_gtsrb_pilot.json \
  --dataset-terms-record "<record the actual reviewed terms and their source>"
PYTHONPATH=src python -m scripts.run_ssv_gtsrb \
  --study output/ssv2026_gtsrb_pilot/study.json \
  --output output/ssv2026_gtsrb_runs --dry-run
```

Remove `--dry-run` only after inspecting the frozen matrix. `--only` accepts a
run ID printed by dry-run. The study has 9 images, 27 image/radius regions,
and 162 configurations: radii 1,2,4 raw byte levels, beta=0,1,2, cuts off/on.
All use CBC, derived preimages, F=8, strict chaining, no heuristic tolerance,
no E2E fallback, one ESBMC process, 300 s/query and 6g. **Do not use `preqbmc
reproduce` for this new schema**; it is a separate, restricted-model adapter.
Sub-byte radii would contain only the center in the declared byte domain.

```sh
PYTHONPATH=src python -m scripts.report_ssv_regions \
  --study output/ssv2026_gtsrb_pilot/study.json \
  --input output/ssv2026_gtsrb_runs --output output/ssv2026_gtsrb_tables
```

The ledger retains unrun/crashed variants, while CSVs distinguish configured
and completed denominators. Neither different radii nor repeated timings are
independent images. Optional diagnostic timeouts are reported separately from
the final region status. No missing device metric is replaced by zero.

## Android gate and remaining experiments

After a region verifies, run the full-test host comparison. This evaluates
ordinary uniform Q16/F8 quantization and the selected formats against the
same test split, including exact Python/C intermediate values, encoder parity,
actual library/array sizes, and a streamed device-replay corpus:

```sh
PYTHONPATH=src python -m scripts.evaluate_ssv_host \
  --study output/ssv2026_gtsrb_pilot/study.json \
  --region output/ssv2026_gtsrb_runs/<run_id>/region_summary.json \
  --output output/ssv2026_gtsrb_host_quality
```

The uniform baseline is explicitly uncertified. A local certificate never
implies that all images used for accuracy evaluation are certified.

`examples/ssv_android` provides JNI and CMake integration, **not a tested APK**.
Use a successful run's `qnn.c`, including its encoder, with the NDK toolchain:

```sh
cmake -S examples/ssv_android -B output/android_build \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DQNN_SOURCE="$PWD/output/ssv_cnn_gate_new/qnn.c"
cmake --build output/android_build
```

Channels and dimensions are read from the generated encoder. The NDK setup follows the
[official CMake toolchain documentation](https://developer.android.com/ndk/guides/cmake).
Include `Qnn.java` and the library in a device test app. Feed exact decoded HWC
bytes, without Android bitmap resizing or normalization. Match every vector in
`parity_vectors.json` and intermediate prefix outputs before claiming transfer;
the prefix C files are available from the tiny gate for separate diagnostic
builds. Record source/library hashes, NDK, compiler, ABI, flags and device build.
Host C parity does not certify the compiler or JNI wrapper.

Still required before the proposed paper's deployment results exist:

- Real GTSRB training/region runs and full-test Python/C accuracy and logit parity.
- Physical arm64 device validation including encoder, intermediate values and
  final logits; no NDK/device validation is claimed by the current tests.
- Matched ordinary quantization and float32 baselines. Their execution is not
  automatically certified by the selected PreQ-BMC artifact.
- Repeated device measurements and, when feasible, independent training seeds.
- A named meter and synchronized external power traces.

Use a fixed full-test sequence, warmup, five trials, and fixed screen,
brightness and network state. Record thermal/battery conditions, actual model
and binary bytes, median/p95 latency, throughput, CPU-time/wall-time ratio
(may exceed 100% for multiple cores), and sampled process peak RSS.
Measure idle power separately. Avoid USB charging contamination; record meter
model, sample rate, calibration and uncertainty. `reports.ssv_measurements`
integrates a covering, synchronized trace by the trapezoidal rule, reporting
gross and separately idle-subtracted joules/inference. CPU percentage is not
an energy estimate. This helper does not provide synchronization by itself.
