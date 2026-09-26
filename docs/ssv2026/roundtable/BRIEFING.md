# Round-table briefing: verified traffic-sign QNN for SBSeg/SSV 2026

**Your role.** You are one of several AI reviewers in a technical discussion about this project. The
others are Claude, GPT, DeepSeek and Codex, and the human owner of the project decides. You
cannot run code; the owner or Claude implements what the group agrees on. Be concrete and
quantitative, and disagree with other participants when they are wrong.

## 1. Goal

- **Paper:** SBSeg 2026, special session SSV (vehicle and traffic security). The deadline is
  2026-10-04.
- **Artefact:** a quantized CNN that classifies GTSRB traffic signs (43 classes, not licence
  plates), with at least 90% accuracy.
- **Claim:** the exact bit-precise C code that is deployed is **formally certified robust** by the
  ESBMC bounded model checker, per input image, for a small perturbation radius.

## 2. Hard constraints (non-negotiable; proposals that violate one are rejected)

1. **ESBMC is the final judge.** An image is VERIFIED only if every ESBMC obligation it needs
   returns VERIFICATION SUCCESSFUL. MILP, LP, CROWN, DeepPoly and Python code are *untrusted
   finders*: they may propose bounds, never decide.
2. **Nothing that trades soundness for a nicer result.** That excludes:
   - weakened contracts;
   - heuristic tolerances;
   - disabled chaining;
   - per-block or per-partition number formats;
   - changes to the deployed arithmetic;
   - counting TIMEOUT or INCONCLUSIVE as proof.
3. **Negative and inconclusive results are legitimate and are reported.** Never propose turning
   one into an unsupported VERIFIED.
4. **The deployed arithmetic is fixed and verified bit-precisely**, including rounding,
   saturation, 128-bit accumulators and the input encoder.
5. **Keep proof, measurement and assumption separate.** Label each claim PROVED, MEASURED,
   ASSUMED or GUESS.
6. **The test set is used once**, via a pre-registered protocol. All tuning happens on
   validation.
7. **Machine:** one workstation (22 cores, 23 GB RAM, WSL2), with no GPU in the loop and no
   cluster. It has crashed from running out of memory before.
8. **Code freeze:** no changes under `src/` until Monday 2026-09-28 about 09:00. A pre-registered
   test run is in progress and pins the hash of every source file. Ideas can be prototyped
   outside `src/` or implemented after the freeze.

## 3. The model and the deployed semantics

- **Architecture:** `conv24_64_192_dwpool43`, 5 layers (convolutions, a depthwise pool, a dense
  output). The input is a 32×32×3 image (3,072 bytes), from a nearest-centre resize of the
  GTSRB crop; the resize and encoder are inside the verified C.
- **Arithmetic per layer:**
  - `r = round_half_away(Σ W·h / 2^8) + b`
  - `z = clamp_to_int16(r)`
  - `h = relu(z)`
  - All values use Q7.8 fixed point (16 bits, 8 of them fractional) and `__int128`
    accumulators. BatchNorm is folded into the weights.
- **Accuracy on the 12,630 test images (MEASURED):**
  - Float model: 91.32%.
  - Deployed C: **91.35%**, bit-exact with an independent Python integer model on every image.
    Worst class: 44.7%.
  - Inference: about 3 ms per image on x86.
- **Property certified per image x:** for every x' with max_j |x'_j − x_j| ≤ ε = 1 (byte units),
  the C returns the same class, with a strict margin against all 42 competitors.

## 4. Current certification pipeline ("proof-carrying" style)

Wall-clock times are for one image.

| Stage | What it does | Time |
|---|---|---|
| Screen | Untrusted CROWN-style finder checks whether a certificate looks possible | ~45 s |
| Box proof | Interval bounds per neuron, checked by ESBMC in blocks of 24 neurons (589 blocks) | ~10 min |
| Chain export | Linear-relaxation chains (CROWN-like) back to the input; each chain step becomes a small C harness | ~90 s |
| Check | ESBMC checks every harness, 5 layer shards, 4 parallel jobs, 60 s timeout per harness | ~4 h |
| Aggregate | Python checks that every obligation, citation and margin closes | seconds |

**Harness kinds:**
- *affine chunk:* one linear step computed correctly, with rounding error bounded;
- *relu chunk:* the ReLU relaxation lines are valid on their interval;
- *closes:* chains are composed;
- *concretize:* the chain is evaluated on the input box;
- *lemmas:* rounding lemma on the deployed kernel, tight (±128 VERIFIES, ±127 FAILS).

**Soundness tests:** a data-corruption regression mutates one number in the certificate at a time
(14 kinds), and ESBMC rejects every mutation.

## 5. Measured results so far

**Validation image 3652 (MEASURED):**
- The certificate has 4,054 chains; ESBMC checked **58,695 obligations, all VERIFIED**.
- ESBMC CPU time was 13.9 h in total, and wall time about 3.9 h with 4 jobs. Peak memory was
  2.8 GB in total across the 4 jobs.
- The minimum margin was +2,622 integer units, against class 14.

Time per obligation:

| Statistic | Seconds |
|---|---|
| Median | 0.31 |
| 90th percentile | 2.9 |
| 99th percentile | 5.9 |
| Maximum | 9.9 |

Time by kind:

| Kind | Count | Total ESBMC time | Mean |
|---|---|---|---|
| affine_chunk | 15,792 | 10.1 h (72%) | 2.3 s |
| relu_chunk | 15,430 | 2.2 h | 0.5 s |
| affine_close | 11,695 | 0.9 h | 0.28 s |
| everything else | about 15.8k | about 0.7 h | — |

**Pre-registered test run (in progress, started 2026-09-26):**
- 80 random test images, ε = 1.
- Screen: 8 MISCLASSIFIED, 54 FINDER_INCONCLUSIVE (the finder margin is below 1 against some
  competitor), 18 FINDER_POSITIVE.
- At about 4 h per positive, only about 10 of the 18 will be checked before the launch cutoff
  (Mon 04:00). The rest are reported as NOT_RUN_BUDGET, and they count against the certified
  rate.

**Earlier approaches that failed (MEASURED):**
- **Monolithic ESBMC query on the network:** no verdict after about 22 min at 20 GB, even with an
  artificially tight box.
- **Interval (IBP) bounds alone:** every coordinate saturates to the full int16 range by block 4,
  so the result is vacuous. The true reachable set is tiny (median width 0, max 20).
- **DeepPoly plus MILP preimage synthesis:** the previous version of this tool, used on Iris,
  Seeds, MNIST and a 41.75% GTSRB model. It does not scale to this CNN.
- **IBP-style certified training:** could not keep 90% accuracy.
- **ESBMC encoding lessons:**
  - Writes to local arrays make symbolic execution quadratic; scalar locals gave speed-ups of
    about 25×.
  - A general nonlinear ReLU lemma TIMEOUTs under z3, bitwuzla and boolector. It was replaced by
    per-instance linear checks.

## 6. Agenda (open questions)

- **Q1: speed.** How can the certificate check be cut from about 4 h per image?
  - Known: batching obligations saves only about 15–20% (startup is small). Parallelism
    (4 → 12 jobs) looks like about 3×.
  - Open: what is behind affine_chunk's 72%? Chunk size, the expression shape, the solver
    (z3 vs bitwuzla vs boolector), ESBMC flags, reusing shared sub-chains across competitors,
    or proving some affine steps once as a lemma.
- **Q2: coverage.** 54 of 80 test images fail in the untrusted finder, before ESBMC is asked
  anything. How can the finder produce tighter certificates that ESBMC can still check piece by
  piece? Options: optimized slopes (alpha-CROWN), branch-and-bound or input splitting (each
  branch certified; the union of branches must cover the ε box), competitor-specific chains.
  Which gives most coverage per hour of implementation?
- **Q3: the paper.** Framing, the minimum convincing evaluation, baselines, threats to validity,
  and what reviewers will attack. What efficiency evidence should we show (see the timing
  data)?
- **Q4: anything we are missing:** a soundness hole, a cheaper design, related work that already
  does this.

## 7. How to answer

- Address the agenda item or the task you are given. Respond to earlier turns by name when you
  agree or disagree.
- For each proposal give:
  1. what it is;
  2. why it is sound (or which constraint it risks);
  3. the expected gain, labelled MEASURED or GUESS;
  4. the implementation cost in hours;
  5. how to measure it on validation images.
- Mark citations you are not sure about with "(verify)". Do not invent numbers.
- Keep it under about 900 words, with no pleasantries.
