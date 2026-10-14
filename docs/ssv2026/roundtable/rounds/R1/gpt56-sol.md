# GPT-5.6 Sol — Engineer proposal

## 1. Q1: factorial pilot on affine harness shape, size and solver

1. **What.** After the freeze, add a validation-only microbenchmark over stratified `affine_chunk` harnesses: flat versus balanced/scalar expressions, term caps 1k/5k/25k, and Z3 versus Bitwuzla. The present renderer emits one flat literal expression and three assertions per column (`src/verification/crown_chain_v2.py:744-805`); the checker already exposes term cap and profile (`src/scripts/check_crown_chain_v2.py:26-64`).
2. **Soundness.** **PROVED:** unchanged integer identities still go to ESBMC; retain bounds/division/pointer checks, which the runner explicitly warns not to disable (`src/verification/esbmc.py:239-246`). No cached verdicts or omitted obligations.
3. **Expected gain.** **MEASURED:** affine chunks consume 10.1/13.9 CPU-hours (72%) (`docs/ssv2026/roundtable/BRIEFING.md:100-122`). **GUESS:** halving affine time yields 1.56x end-to-end by Amdahl, roughly 3.9 h to 2.5 h; solver choice may yield nothing.
4. **Cost.** 6–8 engineer-hours.
5. **Validation.** Use 200 fixed, term-count-stratified harnesses from validation image 3652, then its full certificate. Go only if every obligation is VERIFIED, peak is below 1.1 GiB/job, corruption controls still fail, and full wall time improves at least 1.4x. Do not increase jobs: 12 jobs produced only 1.6x and caused resource aborts (`experiments/ssv2026_test_protocol.json:314-333`).

## 2. Q2: optimize checker-visible integer ReLU slopes

1. **What.** Add alpha-CROWN-style coordinate/beam optimization over integer `a_out`, separately per root, maximizing the exact integer margin while penalizing chain count. This is not merely “competitor-specific chains”: current `choose_relu` already receives batched per-root coefficients but makes one fixed local choice (`src/scripts/crown_finder/crown_chain_export.py:50-58`).
2. **Soundness.** **PROVED:** the finder remains untrusted; export the chosen integers and let existing ESBMC ReLU harnesses validate each relaxation. Preserve the exporter’s int64 headroom rejection (`src/scripts/crown_finder/crown_chain_export.py:44-47`).
3. **Expected gain.** **GUESS:** highest coverage-per-hour option, but no numeric recovery claim before validation. The current fixed rule leaves 54/80 test cases finder-inconclusive (**MEASURED**, `docs/ssv2026/roundtable/BRIEFING.md:124-127`).
4. **Cost.** 16–24 hours.
5. **Validation.** Freeze a validation cohort containing positives and inconclusives. Go if at least 20% of validation inconclusives become positive, no former positive regresses, and median obligations/time among positives grow by at most 2x. Tune nowhere else.

## 3. Q2 fallback: bounded, exhaustive input splitting

1. **What.** For slope failures only, split one influential byte interval into disjoint integer leaves, cap at four leaves, certify every leaf, and add an ESBMC-checked/aggregator-checked partition manifest.
2. **Soundness.** **PROVED if implemented as specified:** VERIFIED requires every leaf VERIFIED and their union exactly equals the original encoded box; TIMEOUT/UNKNOWN invalidates the image. Arithmetic and encoder remain unchanged. Wang et al. (2021, §§2.3, 3.4) establish BaB completeness when all branches are covered (NotebookLM pack, `docs/ssv2026/roundtable/THREAD.md:32-35`). Main risk is a coverage/endpoint bug.
3. **Expected gain.** **GUESS:** smaller than slope optimization per engineering hour and potentially up to 4x checking cost.
4. **Cost.** 20–30 hours.
5. **Validation.** Ten fixed validation failures; go if at least two close within four leaves, all mutation tests catch gaps/overlaps, and median total check time is at most 2x the unsplit estimate.

## 4. Q3: minimum paper evaluation package

1. **What.** Report the pre-registered 1/80 primary unchanged, with an exact binomial interval and counts for misclassified/finder/check inconclusive; show the resource-only rerun adjacent and explicitly non-replacing. Include float versus deployed accuracy/model bytes, latency, peak RSS, and energy on identical Khadas inputs; verifier wall/CPU/RSS and obligation-kind breakdown; IBP, monolithic ESBMC, and current CROWN baselines.
2. **Soundness.** **PROVED:** preserves the one-use test rule; all new tuning is validation-only. Lin et al. (2021, §4) use binomial confidence intervals (NotebookLM pack, `docs/ssv2026/roundtable/THREAD.md:47-50`); Hashemi et al. (2016, §§IV–V) motivate energy/image and memory metrics (`docs/ssv2026/roundtable/THREAD.md:55-60`).
3. **Expected gain.** **GUESS:** no certified-rate gain; materially stronger reviewer defensibility.
4. **Cost.** 10–14 hours.
5. **Validation.** Rehearse the complete table from validation plus image 3652; require scripts to preserve denominators and labels and repeated board measurements to report dispersion.

## 5. Q4: do not adopt the compiled generic checker for this deadline

1. **What.** Keep fresh ESBMC obligations; extend corruption tests for new slope and partition fields instead.
2. **Soundness.** **ASSUMED:** compiling and executing a once-proved generic checker adds compiler/correspondence trust, while the literature pack reports no proof-transfer treatment (`docs/ssv2026/roundtable/THREAD.md:39-43`). The current aggregator requires every shard VERIFIED, matching checker sources, hashes and deployment (`src/scripts/aggregate_crown_chain_v2.py:103-137`).
3. **Expected gain.** **GUESS:** zero speed; avoids the largest new trust-base/reviewer risk.
4. **Cost.** 6–8 hours for added negative tests.
5. **Validation.** On validation certificates, every clean case must VERIFY and every single-field corruption must FAIL on the named assertion, matching the existing criterion (`src/scripts/regress_crown_chain_v2_data_corruption.py:258-283`).
