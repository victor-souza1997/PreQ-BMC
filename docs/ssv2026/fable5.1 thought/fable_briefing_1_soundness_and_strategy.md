# Fable briefing 1: soundness review and the strategy decision

You are reviewing a research project 10 days before its paper deadline. Two
other agents (Claude and Codex) have been building it together. I need an
independent reviewer who did not write the code. Your job has two parts:

1. **Soundness review.** Is the proof scheme below sound? Would a false
   `VERIFIED` be possible?
2. **Strategy decision.** Given the numbers, what should the paper claim, and
   what should the remaining 10 days be spent on?

Do not write code or change files in this session, except your answer file
(see "Deliverable").

---

## 1. The project

PreQ-BMC takes a trained neural network and produces a **fixed-point
quantized network (QNN) as C code**. ESBMC, a bounded model checker for C,
must prove the deployed C code **locally robust**. That means: for one image
x₀ and every input in the L∞ box of radius ε raw bytes around it, the C code
returns the same class. The application is traffic-sign classification
(GTSRB, 43 classes, 32×32×3) for Android Automotive, so the QNN must also
keep the float model's accuracy.

- **Venue:** SSV 2026, 6 pages IEEE, due **2026-10-04**. Today is 2026-09-24.
- **Machine:** 22 cores, 23 GB RAM, WSL. It has already crashed from OOM
  twice. An AOSP build sometimes shares the machine.

### Non-negotiable rules from the user

1. **ESBMC makes the final decision.** The user's professor created ESBMC,
   and this is the core of the paper. MILP, LP and CROWN may only *propose*
   bounds and relations. Their output is untrusted: a wrong proposal may make
   ESBMC fail, but must never produce a false VERIFIED. No verdict may come
   from anything else, **including an exact Python or rational checker**.
2. **MILP stays** as a finder, as in the previous paper.
3. **Soundness first.** No weakened contracts, tolerances, disabled chaining,
   per-block formats, or changes to the deployed arithmetic. INCONCLUSIVE,
   TIMEOUT and REFUTED are never counted as proofs.
4. **Tune on validation only.** The test split is used once, after selection.

### Evidence tags

Every claim in this project, and in your answer, must carry one tag:

- **PROVED**: ESBMC returned `VERIFICATION SUCCESSFUL` on a non-vacuous
  harness.
- **MEASURED**: an empirical number from running code on data.
- **ASSUMED**: everything else, including paper-level mathematical arguments.
  Mark these "ASSUMED (argued)".

The distinction between what ESBMC established and what is argued on paper is
what reviewers will judge.

### About the evidence in this briefing

All numbers below come from the project documents listed in section 7. The
result files (`output/…`, `scratchpad/arch/…`) are on another machine and
**not in this checkout**. Treat every number as reported, not verified by you.
The source code *is* in this checkout; you may read it to check a claim.

---

## 2. The model and its deployed semantics

The current model is `conv24_64_192_dwpool43`, with batch norm folded:

```
input 32x32x3 raw uint8, encoded as x/256 in Q16 (8 fractional bits)
L0 conv 5x5, 3->24,   s2 SAME   -> 16x16x24 = 6144 ReLU
L1 conv 3x3, 24->64,  s2 SAME   ->  8x8x64  = 4096 ReLU
L2 conv 3x3, 64->192, s2 SAME   ->  4x4x192 = 3072 ReLU
L3 depthwise 2x2, 192, s2 VALID ->  2x2x192 =  768 ReLU
L4 dense 768 -> 43 logits
```

160,523 parameters and 14,080 ReLUs. All values below are MEASURED on the
full 12,630-image test set:

| | value |
|---|---:|
| float accuracy | 91.32% (balanced 87.30%) |
| fixed-point C accuracy | 91.35% |
| float vs fixed disagreement | 0.26% |
| worst class | class 30 (ice/snow), 45.3% |

**Accuracy is therefore not the open problem. Verification is.**

The deployed integer semantics per neuron are identical in Python and C:

```
acc   = sum_i w_int[i] * x_int[i]                  (__int128)
value = round_half_away_from_zero(acc / 2^8) + b_int
value = clamp(value, -32768, 32767)
hidden layers: value = max(value, 0)
```

Every layer uses Q16 with 8 fractional bits.

---

## 3. What has been tried

### 3.1 Box-to-box ESBMC (the existing pipeline): sound but useless near the output

- ESBMC proves per-block interval contracts, "box k ⇒ box k+1", and the
  bridges between them.
- On one test image (06330, ε = 1): 605 of 607 obligations were PROVED, but
  the final margin was not.
- The boxes are about 10× too wide at L3. MEASURED: the box midpoint margin
  is +27,952, while the sound difference-row lower bound is −246,681.
- Status: `ABSTRACTION_INCONCLUSIVE`. ESBMC replayed a violating point
  *inside the abstract box*. Nobody has shown that a real input reaches that
  point.

### 3.2 Certified (IBP) training: tight bounds, lost accuracy

Every IBP variant at ε = 1 dropped validation accuracy from 90.4% to between
53% and 83% (MEASURED). The plan says not to propose plain IBP again.
CROWN-IBP is untested.

### 3.3 CROWN finder: much tighter, but certifies few images

These are float64 CROWN results on the deployed integer semantics, with ±½
rounding slack. The sample is 19 correctly classified **validation** images
at ε = 1 (MEASURED):

| bounds used for the ReLU relaxations | certified | median margin lower bound |
|---|---:|---:|
| ESBMC boxes only | 0/19 | −118,608 |
| CROWN on L3 only | 0/19 | −109,735 |
| CROWN on L2 + L3 | 0/19 | −31,233 |
| CROWN on all hidden layers | **2/19** | −5,655 |
| forward affine envelopes, exact integer | 0/19 | −25,100 |
| smallest margin found by corner sampling (not a bound) | — | about +1,900 |

- For the 17 images that fail, the gap to the sampled margin is about 5× at
  the median.
- α-CROWN, β-CROWN (branch and bound) and MILP on the last layers are **not
  yet run**.
- Test image 06330, where the box proofs were run, has an exact-integer
  CROWN margin of −18,405. CROWN cannot certify it.

---

## 4. The current proof scheme: ESBMC-checked CROWN chains (review this)

The pilot image is validation index 3652, class 12, ε = 1. It was chosen
**because CROWN certifies it**. Its exact-integer CROWN margin is +2,602
against competitor 14, and all 42 competitors are positive (MEASURED).

### 4.1 Certificate

A CROWN chain is a sequence of layer-local steps proving one scalar
inequality, λ·z ≥ bound, where every z is the **raw pre-clamp** value. The
chain goes from a neuron or margin down to the encoded input box. The
finder's output is exact integer.

**Affine step (z_j → h_{j−1}).**
- g = Wᵀa, from the lowered CSR matrix of layer j.
- a_out = ⌊g / 2⁸⌋, and residual = g − 2⁸·a_out ∈ [0, 2⁸).
- output_constant = input_constant + a·b + ⌊(residual·hlow − 2⁷‖a‖₁) / 2⁸⌋.
- hlow = ReLU(lower bound) of the predecessor neuron, or the input box low at
  j = 0.
- Argued justification: |round_half_away(t) − t| ≤ ½, and residual·h ≥
  residual·hlow.

**ReLU step (h_j → z_j).**
- For **any** integer slope a_out,i, the step uses a_i·ReLU(z) − a_out,i·z ≥
  d_i on [l_i, u_i].
- d_i is the minimum over {l_i, u_i, and 0 if l_i < 0 < u_i}.
- The checker therefore does not care how the slope was chosen (CROWN,
  α-CROWN, anything else).

**Concretize step.**
- scaled = constant + Σ aᵢ·(aᵢ ≥ 0 ? xl : xu).
- For a lower claim, the bound is ⌈scaled / 2¹⁶⌉. For an upper claim
  (λ = −e), it is ⌊−scaled / 2¹⁶⌋.

**Bound records.** Each neuron's [l, u] comes from one of two sources:
- `box_only`: the ESBMC box proof. The exporter asserts that the box is
  strictly inside Q16, so raw = clamped.
- `chain_intersect_box`: chain ∩ box. The chain alone must first put the raw
  value inside Q16.

**Margin closure.** Given raw_t − raw_c ≥ 1, the deployed output satisfies
out_t > out_c iff both of these hold:
- top: raw_c ≤ 32766 or raw_t ≤ 32767;
- bottom: raw_t ≥ −32767 or raw_c ≥ −32768.

For the pilot, this needed an extra target-logit lower chain (raw ≥ 3,139),
because logit 12's box saturates on both sides.

### 4.2 How ESBMC checks it (Option B)

- Generic lemmas are proved once. For example, the round-half-away error
  lemma −128 ≤ 256·round(n/256) − n ≤ 128 was PROVED in 0.15 s.
- Each chain step is checked by ESBMC as **straight-line concrete
  recomputation** in `__int128`.
  - Affine steps are split into chunks of at most 25k multiply terms, plus a
    closing harness that recomputes the constants.
  - ReLU coordinates are checked in 512-coordinate chunks, each with a
    nondeterministic z ∈ [l, u] per coordinate.
- A fully symbolic alternative (Option A) times out even at 32 terms.
- ESBMC is thus partly a calculator here. The composition of steps into a
  network result is a meta-level argument, run by the Python coordinator.
  That is the same kind of argument the box pipeline already relies on.

### 4.3 Pilot result

- **PROVED:** all 178 obligations for the four exported chains are VERIFIED.
  The chains are L3 neuron 162 lower and upper, margin 12−14, and logit 12
  lower. Wall time 155 s at 4 jobs, peak RSS 4.65 GiB.
- **PROVED:** the corruption regressions work. Mutating a slope, a d_i, a
  constant, a bound record, or the property vector turns VERIFIED into
  FAILED.
- **MEASURED:** status is `CONDITIONAL_VERIFIED`, not an image certificate.
  Four things are missing:
  1. the dependency chains;
  2. the box proofs for *this* image (the existing ones are for test image
     06330);
  3. chains for the other 41 competitors;
  4. the deployment hash binding and per-step overflow envelopes.

### 4.4 Cost of one complete image (pilot image)

| scope | chains | multiply terms |
|---|---:|---:|
| naive: both sides of every referenced unstable neuron | 7,884 | 330.8M |
| pruned: one-sided, only the sides a step needs | 3,616 | 172.2M |

Layer 2 dominates, at 100.3M of the 172M pruned terms (MEASURED).

The **pruning rules** are ASSUMED (argued). Review them:
- A ReLU coordinate with a > 0 and a_out ∈ {0, a} needs no bound, because
  a·ReLU(z) ≥ 0 and a·ReLU(z) ≥ a·z hold for all z.
- A ReLU coordinate with a < 0 needs only the lower side when a_out = a, and
  only the upper side when a_out = 0.
- In affine residual folding, hlow = ReLU(l) needs the lower side only when
  l > 0.

**Projection (ASSUMED):** about 5 h summed ESBMC time plus the ReLU chunks,
or about 2–3 h wall at 4 jobs, **per certified image**. ReLU chunks are 61%
of the measured pilot time.

### 4.5 Open solver facts

The generic ReLU-envelope lemma and generic linear monotonicity either time
out or run out of memory in Z3, Boolector and Bitwuzla at 2⁴⁰ coefficient
ranges. A 16-bit range has not been tried yet. If the generic lemma
verifies, each ReLU coordinate reduces to a concrete precondition plus three
concrete evaluations.

### 4.6 Other trust points already known

- The Python renderer turns arrays into scalar locals. That transformation
  is ASSUMED correct, not proved.
- The accumulator overflow bound
  (`n_terms · 2^(b_in−1) · max|w| < 2^127`) is a Python calculation, so it is
  ASSUMED.
- DAG completeness is checked by the Python coordinator: every required
  edge must be backed by a VERIFIED result with a matching identity.

---

## 5. Part 1: soundness questions

Answer each question. Say plainly whether something is **unsound**, **sound
but argued (ASSUMED)**, or **sound and discharged by ESBMC**. If you find a
hole, give a concrete counterexample, or the exact condition under which a
false VERIFIED could occur.

1. **Affine step.** Is the output_constant formula sound for the deployed
   semantics in section 2? Check:
   - the direction of each floor;
   - negative a and negative g;
   - the halfway case of round-half-away;
   - the use of hlow when the predecessor bound record came from a *chain*
     rather than a box.
2. **ReLU step.** Is the 3-point minimum rule correct for integer z and any
   integer slope? Is a per-coordinate check with nondeterministic z ∈ [l, u]
   enough, or does the step's *sum* need its own obligation?
3. **Concretize and margin closure.** Check the rounding directions, and the
   two-sided clamp rule including the equality edge cases.
4. **Pruning rules** (section 4.4). Are they sound? Is any "bound-free" case
   actually dependent on a bound?
5. **Composition.** The final "certified" verdict comes from a Python
   coordinator checking that a DAG of about 13k VERIFIED harness results is
   complete and consistent. Does this respect rule 1 ("no verdict from
   anything other than ESBMC")? The box pipeline makes the same kind of
   argument. If not, what is the smallest change that would make ESBMC the
   decider? For example: a final ESBMC harness that checks the manifest, or
   one harness per edge that asserts the linked constants.
6. **Trust base.** List everything a reader must trust beyond ESBMC and its
   solver for the verdict to be correct, and rank the items by risk. Which of
   them could the team remove with an ESBMC obligation within 10 days?
7. Anything else that could let a wrong finder proposal become VERIFIED.

---

## 6. Part 2: the strategy decision

### What the numbers say (MEASURED unless marked)

- The float and fixed-point models are equally accurate: 91.32% vs 91.35%.
- End-to-end certified images: **0**. One image is `CONDITIONAL_VERIFIED`.
- CROWN certifies 2/19 validation images at ε = 1. Larger ε was not
  measured; it is ASSUMED to be worse.
- Cost is about 2–3 h wall per image (ASSUMED, projected from the pilot).
- The earlier evaluation plan was 18 test images × ε ∈ {1, 2, 4} = 54
  regions, stratified by class accuracy, including the weak classes 27, 30
  and 0.
- About 10 days remain, and the machine is shared.

### Options on the table

- **(B) CROWN-chain certificates on the 91% model.**
  - Finish the dependency-complete checker.
  - Certify whichever pre-selected test images CROWN can close.
  - Report every region, including all `ABSTRACTION_INCONCLUSIVE` ones.
- **(B+) B plus a stronger finder** to raise the 2/19 rate: α-CROWN, MILP on
  L3/L4, or β-CROWN splits. The checker does not change, because the ReLU
  rule accepts any slope. The finder's certified rate is unmeasured.
- **(C) An accuracy-versus-verifiability curve** over 1-, 2-, 3- and 4-stage
  models, using the existing box pipeline. Only fully proved models are
  called certified. The shallow models reach roughly 41.8%, 66%, 87–89% and
  91% accuracy.
- **(D) CROWN-IBP fine-tuning**, to raise the certifiable fraction at a
  smaller accuracy cost than IBP. Untested.
- Any combination, or something the team has not considered.

### Questions

1. **Recommend one plan for the 10 days**, with a day-by-day schedule. Include
   a go/no-go date and the measurement that decides it.
2. **Paper claim.** Write the one-sentence main claim for each plausible
   outcome:
   - at least one test image certified end to end;
   - only `CONDITIONAL_VERIFIED`;
   - zero.
   Say what the paper must *not* claim in each case.
3. **Selection and denominators.**
   - The pilot image was chosen *because* CROWN certifies it.
   - For the paper's test cohort, is it legitimate to run the untrusted CROWN
     finder on pre-registered test images and spend ESBMC time only on those
     it closes?
   - How must the certified rate be reported so that this is not
     cherry-picking?
   - Must the weak classes (27, 30, 0) be in the cohort even if CROWN fails on
     all of them?
4. **Is the cost projection plausible?** Is layer 2 (100M terms) the right
   target? Would early-stopping chains at z1 bound records, or checking
   through a generic lemma instead of concrete recomputation, change the
   order of magnitude?
5. **Is 2/19 the real ceiling of this approach on this model at ε = 1**, or a
   weak finder? What does the literature (α-CROWN, β-CROWN, CROWN-IBP, SABR)
   suggest is achievable for a naturally trained CNN of this size at
   ε = 1/256? Mark estimates as ASSUMED.

---

## 7. Sources in this checkout

- `docs/ssv2026/work_plan_20260923.md`: full status, including what was
  tried and why it failed (sections 2–4).
- `docs/ssv2026/claude_codex_chat.md`: the design discussion for the
  certificate scheme (Turns 1–9).
- `docs/ssv2026/codex_briefing_4_status_audit.md`: the audit questions for
  the older model.
- `docs/ssv2026/conv_native_verification.md`: the box pipeline methodology.
- Code:
  - `src/verification/crown_chain_certificate.py`: chain checker and harness
    renderer.
  - `src/verification/conv_proof.py`: box proof coordinator and status logic.
  - `src/verification/conv_contracts.py`: interval contracts.
  - `src/verification/interval_lemmas.py`: concrete recomputation harnesses.
  - `src/verification/arith_kernel.py`: the deployed rounding and clamp
    kernel.
- **Stale; ignore:** `docs/ssv2026/accuracy_and_refinement.md` describes an
  older 41.7%-accuracy model.

---

## Deliverable

Write one Markdown file, `docs/ssv2026/fable_review_1.md`, with these
sections:

1. **Soundness findings.** Answer questions 5.1–5.7 in order. Put every hole
   first, with its severity.
2. **Trust base**, ranked by risk.
3. **Recommendation and 10-day plan**, including the go/no-go.
4. **Paper claims per outcome**, with the "must not claim" lists.
5. **Selection and reporting protocol.**
6. **Bottom line**, in at most five sentences.

Tag every claim PROVED, MEASURED or ASSUMED. If you think a premise in this
briefing is wrong, say so first and show why. That is the most useful thing
you can tell us.
