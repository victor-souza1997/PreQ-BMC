# SSV 2026 — Results so far (2026-09-28)

Paper-ready record of what the GTSRB QNN verification study has produced up to
2026-09-28. Every number below is tagged with its evidence level and its source file.

| tag | meaning |
|---|---|
| **PROVED** | every ESBMC obligation of the certificate returned VERIFIED on the bit-precise C |
| **MEASURED** | an execution or count over a stated sample; no proof |
| **ASSUMED** | a contract or trust-base item the proofs rely on but do not discharge |

A non-VERIFIED row is never evidence of non-robustness. No FAILED/REFUTED ESBMC
verdict occurred in any campaign below.

---

## 1. System under verification

| item | value | source |
|---|---|---|
| task | GTSRB traffic signs, 43 classes, 32×32×3 crops | `docs/ssv2026/source_model_search.md` |
| network | conv 5×5 s2 (6,144) → conv (4,096) → conv (3,072) → depthwise 2×2 pool, stored dense (768) → dense (43); 14,123 neurons | `output/sign_conv_depthwise_pool_quality_20260924/` |
| arithmetic | one global signed Q7.8 (int16 range, F = 8); round-half-away; clamp to Q before ReLU; accumulators `__int128` | `src/backends/conv_fixed_point.py` |
| certified C | `qnn_conv_native.c`, sha256 `dd535f54…45097`, `int64_t` parameter storage | `output/crown_chain_v2_test_20260925/*/box/proof/` |
| perturbation | L∞ ball of ε = 1 raw byte around each input pixel (the encoded box after Q7.8 rounding) | `experiments/ssv2026_test_protocol.json` |
| property | local robustness: the deployed argmax stays the target class for every byte image in the ball | same |

### Table 1 — Accuracy of the certified deployment (MEASURED)

| model / build | split | images | accuracy | balanced acc. | worst class | source |
|---|---|---|---|---|---|---|
| float32 source | test | 12,630 | 91.32% | 87.30% | — | `output/c_test_accuracy_20260925/summary.json` |
| certified C (Q7.8, int64 storage) | test | 12,630 | **91.35%** | 87.33% | class 30: 44.7% | same |
| Python integer model vs C | test | 12,630 | bit-exact logits, 0 differing images | | | same |
| int16-storage compact C (Q7.8) | validation | 7,830 | 90.24% | | | `docs/ssv2026/int8_int16_candidates.md` |
| int8 compact C (Q3.4) | validation | 7,830 | 80.75% | | | same |

The test split was evaluated once, after model selection. Quantization changed
accuracy by +0.02 pp.

---

## 2. Method (what is trusted)

The pipeline has untrusted finders and a single trusted judge, ESBMC.

1. **Box stage.** Interval propagation proposes per-neuron boxes. ESBMC proves each box
   block on the verbatim deployed C (proof-carrying interval, block size 24,
   `paper-z3`, 120 s).
2. **Finder (untrusted).** Exact-integer CROWN back-substitution chains are exported,
   one per needed one-sided neuron bound and one per target-vs-competitor margin. Every
   ReLU relaxation constant, rounding term and fact reference is written into a
   certificate (`crown_chain_certificate_v2`).
3. **Checker.** The certificate is split into straight-line C harnesses:
   - affine chunks;
   - ReLU chunks;
   - the rounding lemma on the deployed kernel;
   - per-instance breakpoint checks;
   - margin clamp-closure.

   ESBMC must return VERIFIED on every harness. Constants are recomputed in C, not
   carried as Python literals.
4. **Aggregate.** An image is VERIFIED only if every obligation of its certificate is
   VERIFIED, and the dependency closure (each fact has exactly one source: an ESBMC box
   block or an ESBMC-checked chain) is complete.

MILP/LP/CROWN are interval finders only; ESBMC decides.

**ASSUMED (trust base).**
- ESBMC 7.11 with Z3 (`paper-z3` profile) is sound for the harnesses.
- The Python binding step maps certificate layers to the deployed C arrays; ESBMC
  corroborates it only on sampled rows (`esbmc-binding` slices).
- The input encoder contract: bytes are encoded by `qnn_encode_bytes`.
- The protocol's pinned interval-analysis flags. An ablation showed no sampled verdict
  needs them, but they stay pinned for the primary result.

---

## 3. Pre-registered test campaign

- Protocol: `experiments/ssv2026_test_protocol.json`.
- Draw: 80 test images, seed 20260925, ε = 1 byte.
- The protocol pins all `src/**/*.py` by hash.
- Results: `output/crown_chain_v2_test_20260925/results_preregistered_primary.json` (primary)
  and `results.json` (with the disclosed rerun).

### Table 2 — Outcomes on the 80 drawn test images

| outcome | primary (pre-registered) | with disclosed rerun | meaning |
|---|---|---|---|
| MISCLASSIFIED | 8 | 8 | deployed C wrong on the centre image; nothing to certify |
| FINDER_INCONCLUSIVE | 54 | 54 | CROWN margin < 1 for ≥ 1 competitor (screen median margin −1,975) |
| CHECK_INCONCLUSIVE | 17 | 16 | finder positive, but some ESBMC obligations ended UNKNOWN/MEMOUT |
| **VERIFIED** | **1** | **2** | PROVED robust at ε = 1 |
| certified rate (all 80) | 1.25% [0.2, 6.7] | 2.5% [0.7, 8.7] | Wilson 95% |
| certified rate (72 correct) | 1.39% [0.2, 7.5] | 2.78% [0.8, 9.6] | Wilson 95% |
| upper bound if every finder-positive image verified | 18/80 = 22.5% [14.7, 32.8] | | MEASURED ceiling of this finder |

**Every CHECK_INCONCLUSIVE is a resource abort, not a solver verdict.**
- The checker aborts the shard when host `MemAvailable` falls below its 6 GiB guard
  (`aborted_low_memory: true` in each `check_summary.json`); in-flight obligations are
  then recorded as UNKNOWN (SIGTERM) or MEMOUT.
- No obligation returned FAILED.
- Causes:
  - job-count amendments: 4 → 12 → 6 → 4, committed as `7aa4a929`, `b390496e`, `23e4fcf3`;
  - host memory held by concurrent editor/IDE processes;
  - three unplanned host reboots.

The amendments are resource-only, each recorded in the protocol's `amendments` list and
in `AMENDMENT_*.json`.

### Table 3 — The 18 finder-positive test images

| # | image | target | screen margin min | chains | primary | now |
|---|---|---|---|---|---|---|
| 005 | 06267 | 40 | 584 | 3,348 | **VERIFIED** (48,871 obligations) | VERIFIED |
| 015 | 04083 | 15 | 2,552 | 3,396 | resource abort | **VERIFIED** (disclosed rerun, 50,851 obligations) |
| 011 | 08804 | 12 | 2,834 | 4,654 | resource abort | resource abort (L4) |
| 016 | 00799 | 15 | 1,126 | 6,086 | resource abort | resource abort (L2) |
| 017 | 04317 | 11 | 1,496 | 4,688 | resource abort | resource abort (L4) |
| 026 | 06060 | 2 | 819 | 5,125 | resource abort | resource abort (L3) |
| 027 | 02111 | 17 | 1,052 | 6,682 | resource abort | resource abort (L2) |
| 032 | 11604 | 16 | 2,430 | 5,024 | resource abort | rerun pending |
| 037 | 06521 | 36 | 413 | 6,240 | resource abort | rerun pending |
| 042 | 03512 | 18 | 1,712 | 4,704 | resource abort | rerun pending |
| 044 | 10547 | 26 | 377 | 5,268 | resource abort | rerun pending |
| 058 | 10596 | 25 | 364 | 6,077 | resource abort | rerun pending |
| 059 | 03336 | 2 | 496 | 6,328 | resource abort | rerun pending |
| 062 | 06642 | 11 | 884 | 4,498 | resource abort | rerun pending |
| 064 | 08301 | 25 | 1,118 | 4,362 | resource abort | rerun pending |
| 069 | 08057 | 38 | 1,220 | 6,054 | resource abort | rerun pending |
| 070 | 05382 | 2 | 537 | 6,257 | resource abort | rerun pending |
| 072 | 06851 | 35 | 2,385 | 3,240 | resource abort | rerun pending |

Image ids are `Final_Test/Images/NNNNN.ppm`. The screen margin is the finder's raw
margin lower bound (untrusted) before checking.

### Development pilot (validation, not a test result)

- Validation image 3652, class 12, ε = 1.
- **PROVED: 58,690 / 58,690 obligations VERIFIED.**
- 4,054 chains, 17,819 facts, finder margin min +2,622.
- This image was used during development, so it is not a sample.

---

## 4. Cost of the decomposed method

### Table 4 — ESBMC checking cost by layer (MEASURED)

| image | layer | chains | harnesses | CPU-h | cumulative CPU-h | peak RSS (MiB) | jobs |
|---|---|---|---|---|---|---|---|
| 3652 (val) | L0 | 314 | 1,257 | 0.06 | 0.06 | 306 | 4 |
| | L1 | 767 | 6,143 | 0.54 | 0.60 | 631 | 4 |
| | L2 | 2,089 | 32,445 | 7.97 | 8.57 | 2,818 | 4 |
| | L3 | 840 | 16,115 | 3.81 | 12.38 | 3,997 | 4 |
| | L4 | 44 | 2,735 | 1.54 | **13.92** | 4,064 | 4 |
| 005 (test) | L0–L4 | 3,348 | 48,871 | | **11.82** | 4,196 | 4 |
| 015 (test) | L0–L4 | 3,396 | 50,851 | | **20.48** | 8,635 | 12/4 |

Source: `output/ssv_scalability_20260928/TABLES.md`.

- **Cost structure.** Above ~4k terms the affine cost is flat at 177–188 µs per multiply
  term, so cost is linear in exported terms. Layer 2 carries 57–69% of it. The only
  lever is fewer terms, i.e. finder work.
- **Wall-clock time.**
  - Screening: ~35–45 s per image.
  - Box stage: ~650–800 s per image.
  - Check: 3–5 h wall per finder-positive image at 4 jobs.
- **015 run conditions.** Its cost was inflated by 12-job contention, and L4 ran at 4 jobs.

---

## 5. The monolithic wall (why decomposition is needed)

### Table 5 — Direct ESBMC on the verbatim compact C (MEASURED, validation 3652)

| scope | bits | free input bytes | solver | status | s | peak GB |
|---|---|---|---|---|---|---|
| whole network, concrete input | 16 | 0 | Z3, ± interval flags | TIMEOUT (in symex, ~230 of 6,144 L0 outputs unrolled) | 600 | 3.0 |
| L0 row 0 (384 outputs) | 16 | 0 | Z3 | TIMEOUT | 600 | 3.0 |
| L0, 24 outputs | 16 | 0 | Z3 | MEMOUT | 114 | 5.2 |
| L0, 24 outputs | 16 | 0 | Bitwuzla | VERIFIED | 80 | 4.45 |
| L0, 48 outputs | 16 | 0 | Bitwuzla | OUT_OF_MEMORY | 122 | 5.4 |
| L0, 24 outputs, ε box | 16 | 1 / 3 / 6 / 12 | Bitwuzla | VERIFIED | 85 / 85 / 91 / 186 | 4.5–4.7 |
| L0, 24 outputs, ε box | 16 | 18 / 24 | Bitwuzla | OUT_OF_MEMORY | 270 / 209 | 4.9 |
| L0, 48 outputs, ε box | 8 | 84 | Bitwuzla | VERIFIED | 117 | 5.3 |
| L0, 96 outputs | 8 | 0 / 132 | Bitwuzla | OUT_OF_MEMORY | 212 / 231 | 5.3 |

Memory limit 5 GB. Source: `output/ssv_scalability_20260928/TABLES.md`.

**Reading.** Monolithic BMC walls at 24–48 of 14,123 neurons (0.2–0.3%) and at about
12 free bytes. The decomposed method checks all 14,123 neurons of the same C over
the full ε = 1 ball in 12–20 CPU-h per image, with every harness under ~1.1 GB.

**Caveat.** The int8 rows look cheaper only because Q3.4 input encoding (step 1/16)
collapses ε ≤ 7 bytes to a zero-width box (column "box width sum" = 0 in the source).
They are not evidence that int8 is cheaper to verify.

---

## 6. What does not work (negative results, MEASURED)

### Table 6 — Abstractions that fail on this network (ε = 1, validation)

| approach | result | source memory / file |
|---|---|---|
| plain interval (IBP) box end to end | output margin fully saturated: −65,535 (int16), −255 (int8); cannot decide anything | `output/ssv_scalability_20260928/saturation_*.json` |
| IBP certified fine-tuning | accuracy collapses before bounds close: κ 0.5 → 52.9% acc / 8.7% certified; κ 0.005 → 82.6% / 0% | `experiments/sign_conv_dwpool_ibp_*.json` |
| CROWN relaxed against ESBMC boxes only (no own intermediate chains) | 0/19 certified, margins −32k … −398k | `scratchpad/arch/crown_eps1_n20_boxonly.jsonl` |
| float64 CROWN with own intermediate bounds | 2/19 positive margins | `scratchpad/arch/crown_eps1_n20.jsonl` |
| final-margin check over separate logit boxes | difference-row bound −246,681; boxes ~10× too wide | `docs/ssv2026/work_plan_20260923.md` §2b |
| deeper, more accurate model (91.54% test) | box vacuous by block 4; block 5 alone OOMs at 20 GB | `docs/ssv2026/source_model_search.md` |

- **Where the wall is.** The binding constraint is the depth of the interval abstraction
  (looseness compounds across layers), not solver throughput.
- **Relational chains.** Exact-integer CROWN chains that ESBMC checks one by one are the
  first method that certified anything on a ≥90% model.

---

## 7. Bit-width study (int16 vs int8)

### Table 7 — Global-format candidates (MEASURED, validation 7,830)

| format | accuracy | parameter storage | buffers | notes |
|---|---|---|---|---|
| Q7.8 int16 | 90.24% | 614,422 B | 24,576 B | 0 compact-vs-wide C mismatches |
| Q3.4 int8 | **80.75%** | 307,211 B | 12,288 B | 0 mismatches; below the 90% target |
| Q4.3 int8 | 65.57% | | | |
| Q2.5 int8 | 71.56% | | | 32 weights clipped |

Source: `docs/ssv2026/int8_int16_candidates.md`.

### Table 8 — CROWN chain v2 finder ported to Q3.4 (MEASURED, ε = 1)

The finder takes `--bits/--frac` and adds two ReLU classes:
- `saturated_top`: the proved lower side ≥ Q_HIGH, so h = Q_HIGH;
- `clipped`: the upper side > Q_HIGH, so h = clamp(r, 0, Q_HIGH).

The Q16 behaviour is unchanged. This work is in worktree branch `feat/crown-v2-qformat`,
uncommitted.

| run | result |
|---|---|
| Q16 regression, val 3652 | identical to the pinned export: margin min 2,622, 42/42 closed, 4,054 chains |
| int8, val 3652, ε = 0 | FINDER_POSITIVE, margin min 73 |
| int8, val 3652, ε = 1 | FINDER_INCONCLUSIVE, margin min −751 (0/42) |
| int8, 200 random box corners of val 3652 | worst exact margin +97; no counterexample |
| int8 bound width L2 / L3 (median) | chain ~29 / ~28 vs sampled ~2 / ~1 (IBP box 32 / 50) |
| int8 sweep, val 0–3500 step 500 | 2 misclassified, 2 margin < 1, 4 bounds exceed Q8; 0/6 correct close |

Source: worktree `output/crown_qformat_20260928/`, `tools/crown_qformat/`.

**Reading.**
- With F = 4, the ±½ LSB rounding slack per neuron is large relative to activations.
  Most L2/L3 neurons become unstable and the linear relaxation loses its margin.
- The int8 network is not shown non-robust. It is **not certifiable by this method at
  ε = 1**, a relaxation gap (INCONCLUSIVE with measured cause).
- No ESBMC check was run for int8 (the finder proposes nothing).

---

## 8. Deployment resources (MEASURED, host and VIM3 board)

### Table 9 — Build sizes

| build | what it is | parameter storage | static binary text+data (host) |
|---|---|---|---|
| int64 certified C | the exact source ESBMC checked | 307,211 × int64 = 2.46 MB | ~2.47 MB |
| int16 compact C | same Q7.8 arithmetic, int16 storage; 0 logit mismatches vs certified on 7,830 val images | 0.61 MB | ~0.62 MB |
| float32 baseline | unquantized, not verified | ~160.5k params = 0.64 MB | |

Source: `output/vim3_bundle_20260928/bundle_manifest.json`, `docs/ssv2026/int8_int16_candidates.md`.

**Claim limits.**
- "The QNN uses fewer resources than the float network" does **not** hold for the
  certified file: int64 storage is 3.8× the float32 size.
- It holds (≈ equal to float32, 0.61 vs 0.64 MB) only for the int16 compact build.
- The compact build is linked to the certified C by MEASURED equivalence (0 mismatches),
  not by a proof. A storage lemma (int16 storage of values proved in range) would close
  that gap; it is not done.

### Table 10 — Khadas VIM3 running AAOS (MEASURED, 2026-09-28)

**Board and software.**
- Board: Khadas VIM3 (Amlogic A311D: 2× Cortex-A53 + 4× Cortex-A73, 3.7 GB RAM).
- OS: AAOS `aosp_yukawa_car` Baklava BP1A.250505.005.B1 userdebug, kernel 7.1.0-mainline.
- Binaries: static aarch64, `-O2`, run via `adb shell` as root. The sha256 of every binary
  and data file on the board matched the bundle manifest.

**Run conditions.**
- Single-threaded, pinned to the A73 cores (`taskset 3c`).
- cpufreq is not exposed on this kernel, so the governor could not be read or set.
- The board clock reads 2026-09-24 (not NTP-synced); host timestamps are authoritative.

**Data.** Validation split only: 7,830 images, and a 1-image file for resident memory.

| build | logits vs host certified C | accuracy (7,830 val) | latency median (ms) | p95 (ms) | peak RSS, 1 image (KiB) | static binary text+data (aarch64) |
|---|---|---|---|---|---|---|
| int64 certified C (the ESBMC-checked source) | **0 mismatches** (exact 43 int64 logits) | 90.24% | 31.68 | 31.85 | 3,132 | 3.01 MB |
| int16 compact C (same Q7.8 arithmetic) | **0 mismatches** | 90.24% | 27.94 | 28.08 | 1,260 | 1.17 MB |
| float32 baseline (not verified) | 0 class mismatches vs host float | 90.36% | 21.99 | 22.13 | 1,856 | 1.79 MB |

- **Row sources.** Latency and accuracy come from the one-pass consistency run.
- **3-pass latency runs** (23,490 inferences each, also 0 mismatches) confirm the medians:
  27.75 / 31.67 / 21.99 ms, with p95 within 0.2 ms of the median.
- **Host (x86-64) medians** for the same builds: 1.99 / 2.55 / 2.38 ms. The board is
  ~12–14× slower.
- **Binary size.** The aarch64 static binaries include ~0.5 MB of libc, which is why they
  are larger than the host builds.
- **Source files.** `output/vim3_runs_20260928/results_vim3.jsonl`, `board_vim3.txt`,
  `run_ordered.sh`.

**Deviation from the bundle's `run_on_vim3.sh` (resource and time only).**
- The same runs were reordered: consistency and RSS first, latency last.
- They were pinned to the A73 cores.

**Reading.**
- **PROVED + MEASURED:** the exact C file whose robustness ESBMC certified runs unmodified
  on the AAOS board. Its outputs are bit-identical to the host on all 7,830 validation
  images (MEASURED equivalence). A VERIFIED certificate therefore applies to what the
  board computes, under the assumption that the aarch64 compiler preserves the C
  semantics (ASSUMED).
- **Speed:** the integer QNN is **slower** than float32 on the A73: 27.9–31.7 ms vs
  22.0 ms, i.e. 1.27–1.44×. That is expected with `__int128` accumulators and no SIMD.
- **Memory:** the int16 compact build uses the least resident memory (1.26 MB vs 1.86 MB
  float) and the smallest binary. The certified int64 build uses the most (3.1 MB).
- **Claim limits:** "fewer resources" holds for memory only, and only for the int16
  compact build. Its link to the certified C is MEASURED (0 mismatches on host and
  board), not proved.

---

## 9. Claims the paper can make

| claim | status |
|---|---|
| ESBMC certifies local robustness of the bit-precise deployed C of a 91.35%-accurate 5-layer GTSRB CNN, per image, at ε = 1 byte | **PROVED** for 2 test images (+1 validation pilot) |
| certified rate on a pre-registered 80-image test draw | MEASURED: 1/80 primary, 2/80 with disclosed rerun; 16 images have no solver verdict (resource aborts) |
| no ESBMC obligation was refuted in any run | MEASURED |
| monolithic BMC walls at 24–48 layer-0 neurons | MEASURED |
| decomposed ESBMC checking costs 12–20 CPU-h per certified image, linear in terms | MEASURED |
| the relaxation (finder), not ESBMC, limits coverage: 54/72 correct images fail at the finder | MEASURED |
| int8 Q3.4 cannot be certified by this method at ε = 1 | MEASURED (0/6 validation images) |
| QNN smaller than float | only for the int16 compact build, MEASURED equivalence |
| the certified C runs bit-identically on a VIM3 under AAOS | MEASURED: 0 logit mismatches on 7,830 validation images |
| QNN faster than float on the board | **false**: 1.27–1.44× slower on the Cortex-A73 |

**Not claimable:**
- a robustness rate of the model;
- the non-robustness of any image;
- that int8 is cheaper to verify;
- a speed advantage of the QNN on the VIM3.

## 10. Open items before the deadline (2026-10-04)

1. **Finish the disclosed rerun of the 16 resource-aborted images at jobs = 4**, with the
   IDE closed. About 2–5 h each, i.e. ~40–80 h total. Report primary and rerun columns
   side by side.
2. **Optional:** α-optimised slopes in the int8 finder; a storage lemma for the int16
   compact C.
