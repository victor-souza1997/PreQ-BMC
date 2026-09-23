# Briefing 4: status audit of the convolution-native pipeline

You implemented the convolution-native proof-carrying pipeline
(`conv_fixed_point.py`, `conv_contracts.py`, `conv_native_verification.md`,
`preqbmc gtsrb conv-verify`). Before any more implementation, I need a plain,
evidence-backed audit of where the project stands. Answer the questions below.
Do not write new features in this turn.

Context you need: the SSV 2026 paper is due **October 4, 2026**, about 11 days
from now. The machine has 22 cores and 23 GB RAM. The product goal is:

> The tool must synthesize a QNN by verifying the bit-precise C code. Because
> the task is traffic-sign classification, the QNN must keep the accuracy of
> the floating-point model.

## Rules for your answer

1. Start every answer with **YES**, **NO**, or **PARTIAL**, then explain.
2. Put every claim into one of three buckets and label it:
   - **PROVED**: ESBMC returned VERIFICATION SUCCESSFUL on a harness, and the
     harness is non-vacuous.
   - **MEASURED**: an empirical number from running code on data.
   - **ASSUMED**: anything else, including properties you believe but did not
     check.
3. Cite evidence for each claim: a file path, a JSON field, or a command and
   its actual output. Paste numbers from the output. Do not quote numbers from
   memory.
4. When you report test results, paste the exact line `Ran N tests ... OK` or
   the failure lines. Run the full suite, not only the new tests, and say
   whether any failure is new or was already there.
5. A negative result is fine. TIMEOUT, INCONCLUSIVE, and NOT_CHECKED are
   honest answers. Do not turn any of them into a claim of robustness.
6. Do not weaken a contract, add a tolerance, or disable chaining to get a
   better answer.

## What I already checked

I read your outputs directly. These are facts from the files, not opinions.

**No image is certified yet.**
`output/sign_conv_native_complete_proof_v3_20260923/pilot_summary.json` says
`final_status: TIMEOUT`, `certified: false`. That run covers **one** test image
(target class 12) at **ε = 1 raw byte**. The Android-bound pilot says
`final_status: PARTIAL_NOT_CERTIFIED`, and it verified only the input encoding
and one 24-output convolution block.

**The 488 verified blocks certify intervals that are too wide to be useful
near the output.** I measured the invariant widths in the bridge harnesses
under `.../complete_proof_v3_20260923/proof/harnesses/`:

| bridge | neurons | median width | max width | neurons at the Q16 ceiling (32767) |
|---|---:|---:|---:|---:|
| 0 → 1 | 6144 | 12 | 87 | 0 |
| 1 → 2 | 3072 | 52 | 209 | 0 |
| 2 → 3 | 1536 | 982 | 2018 | 0 |
| 3 → 4 | 512 | 11,888 | 30,955 | 0 |
| 4 → 5 | 384 | **32,767** | **32,767** | **384 / 384** |

Every neuron entering the output layer is bounded by `[0, 32767]`, which is
the full Q16 range. ESBMC can prove that bound, but it tells us nothing. Your
own `invariants` field agrees: layers 2 to 5 have **0 stable ReLUs**. All of
them are `unstable`.

This matches a separate measurement I made on 2026-09-22
(`scratchpad/arch/RESULTS.md`). With ε = 1 LSB, the interval box saturates all
384 neurons by block 4. The set of values the network can actually reach
stays small: its maximum width is 20 at every layer, measured over 256 sampled
corners of the ε-ball. So the network looks robust, but the interval
abstraction loses about 1600× in precision by block 4.

Also, in the same run: when I gave ESBMC a single 384 → 43 output-competitor
query with bounds tighter than any sound analysis produces, it had no verdict
after about 22 minutes and was using 20.3 GB of RAM. I stopped it. The
18 → 43 query from the old model finishes in 3 to 36 seconds.

**The proof counts come mostly from the cache.** The summary says
`esbmc_calls_executed: 2` and `esbmc_verified: 507`.

If you think any of this is wrong, show me with evidence. That would be the
most useful thing you could tell me.

## Questions

### 1. Is the pipeline finished?

(a) Do we have a fixed-point QNN whose deployed C code is **proved** robust
by ESBMC on at least one real test image, end to end? That means from the
raw uint8 input box, through all 6 layers, to the final margin. How many
images, and at which ε?

(b) Is the fixed-point QNN **proved** to match the float model on any input
region? Or only **measured** to match on test images? Your summary says
`float_to_integer_transfer_claim = false` and
`source_local_robustness.status = NOT_CHECKED`. Explain in plain words what a
reader may and may not conclude from that.

(c) What does `PROVED_BY_STATIC_INTEGER_BOUND` for accumulator safety
actually prove? Was ESBMC involved, or is it a Python calculation?

### 2. What is missing?

List every missing piece between today and "a verified, bit-precise,
fixed-point QNN that keeps float accuracy." For each item give:

- what it is;
- whether it blocks the paper's main claim or is optional;
- your honest estimate of the work, and whether it fits in 11 days;
- the risk that it cannot be done at all.

Address these directly:

- the output-margin obligation (currently TIMEOUT at 120 s);
- the saturated 4 → 5 bridge: what mechanism would make the bound at block 4
  useful? A relational bound, a DeepPoly/CROWN-style linear bound, splitting,
  something else? Is the "sparse relational CEGAR controller" meant to fix
  this? Has it ever tightened a real bound on this model? Show the numbers.
- convolution-native MILP for the source float property;
- Android device parity, latency, and energy (all still unmeasured).

### 3. Is the methodology scalable?

Give measured numbers where they exist, and mark estimates as estimates:

- wall time and peak memory for **one** complete image attempt, split into
  (i) the hidden-block certificates, (ii) the bridges, and (iii) the final
  margin;
- how that cost grows with ε (1, 2, 4 bytes) and with network depth;
- the number of ESBMC calls per image with a cold cache, and how much the
  cache actually saves on a *new* image (not a re-run of the same image);
- whether any part grows faster than linearly in the number of neurons.

State clearly: **is it the solver or the abstraction that stops us?** My
evidence says the abstraction fails first (vacuous box at block 4). If you
disagree, show me why.

### 4. What can we realistically run for the paper?

- In about 8 days of compute on this machine, how many (image, ε) regions can
  finish with a real verdict (VERIFIED or REFUTED, not TIMEOUT)? Show the
  arithmetic.
- How many do we need for a credible SSV result? Propose a region set
  stratified by class accuracy. The weak classes must be included, not
  skipped: class 27 pedestrians is 50.0%, class 30 ice/snow is 54.7%, class 0
  (20 km/h) is 65.0%.
- Which datasets can go through the conv-native pipeline as it is today?
  GTSRB only, or also MNIST, Iris, and Seeds from the earlier LADC work?
- If the 4-stage model cannot be certified in time, what is the best honest
  fallback? Options I see:
  - (A) a shallower model: lower accuracy, but verifiable;
  - (B) the 4-stage model with only the verified parts claimed, plus a
    measured account of where the proof stops;
  - (C) an accuracy-versus-verifiability curve across the 1-, 2-, 3- and
    4-stage models.

  Recommend one, and say what the paper's main claim would be under it.

### 5. Housekeeping

- The convolution-native work is **not committed**. It is still untracked or
  modified in `git status`. List what should be committed and what is scratch.
- Your summary says the fixed-C accuracy is 91.5519% and the float accuracy is
  91.5439%, with 0.3088% prediction mismatch. That is about 39 of 12,630
  images. How many of those 39 did the fixed-point model get right and the
  float model get wrong, and the other way round?

## Output format

One Markdown document with sections 1 to 5 in this order, a table for each
set of numbers, and a final section, **"Bottom line"**, of no more than five
sentences. It must say whether we can claim a verified robust QNN today, and
what the paper should claim if we submit on October 4.
