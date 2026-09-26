# Fable breakthrough ideas 1: where the leverage is, and what to run tomorrow

Independent consultation, 2026-09-24, ten days before the deadline. I read the
briefing, `work_plan_20260923.md` §2–4, `claude_codex_chat.md` Turns 1–9, the
earlier `fable_review_1.md`, and the code in `src/verification/` and
`src/backends/`. Result files are on another machine; every number from them
is **reported**, not re-measured by me. I ran two small numpy calculations in
the scratchpad (§1 and §3); those are MEASURED (toy) and marked as such.

Tags: **PROVED** (ESBMC `VERIFICATION SUCCESSFUL`, non-vacuous), **MEASURED**
(code run on data; "reported" when I did not run it), **ASSUMED** (argued or
estimated).

---

## 0. Ranked list

| # | idea | P(works in 10 days) | impact | what it changes |
|---|---|---:|---|---|
| 1 | **Attack-to-truth triage + ESBMC-replayed REFUTED rows** | 0.9 | medium-high; gates every other decision | finder (new), checker (one concrete harness type), paper claim |
| 2 | **New artifact by PGD adversarial fine-tuning at ε_train = 2 bytes (not IBP)** | 0.45 | very high: the only lever that moves the certifiable *fraction* by 5–10× | model + deployed C (re-frozen), claim |
| 3 | **New artifact with two ReLU layers and a direct linear head (ARCH-2R)** | 0.4 | very high; also halves checker cost | model + deployed C (re-frozen), claim |
| 4 | **A named enumerable property (byte brightness band) decided exactly by ESBMC** | 0.95 | low-medium; guarantees non-empty PROVED rows | property (named as different), checker (concrete) |
| 5 | **Margin-only case splits with shared neuron chains (β-style)** | 0.3 | medium: a few more images on any artifact | finder + coordinator; checker gains one record kind |

Ideas 2 and 3 share one decisive experiment (a training grid on Sep 25). All
five use the existing ESBMC chain checker unchanged or nearly so, which is why
they fit in ten days. The honest headline: **no finder or checker change can
turn the current model's 2/19 into a respectable certified rate; only a new
artifact can, and that is roughly a coin flip on the 90% gate.** Ideas 1 and 4
exist so the results table is non-empty and fully ESBMC-decided even if the
coin lands wrong.

---

## 1. First principles: where the gap actually comes from

The property is: for x₀ and all δ ∈ {−1, 0, 1}^3072 (byte-clipped), the
deployed integer C returns the same class. The input encoder maps byte v to
integer v (`(v·256 + 128)//256 = v`, `image_encoder.py:38`), so ε = 1 byte is
exactly one input LSB.

**Where a sound relational bound loses.** A CROWN chain is exact through every
affine layer up to the ±½ LSB rounding slack (PROVED per step in the pilot).
It loses only at unstable ReLUs, and the loss at coordinate i is at most
|aᵢ|·(uᵢ − lᵢ)/4 (triangle relaxation; ASSUMED, standard). So the gap is
governed by three model quantities: the number of unstable neurons, their
widths, and the back-substituted coefficient magnitudes |aᵢ| (a local
Lipschitz quantity). The reported numbers say all three are bad on this model:

- box-unstable neurons for the pilot image (MEASURED, reported, Turn 9): L0
  314/6144, L1 560/4096, **L2 2,355/3,072 (77%)**, **L3 692/768 (90%)**;
- box widths (MEASURED, reported): 14, 52, 1,324, 4,385 medians at the four
  bridges; CROWN widths at L3 about 100–570;
- even with CROWN bounds on every layer the median margin bound is −5,655
  against a clean-ish sampled margin of about +1,900.

**This is a property of the model, not of the checker.** Per-term cost
(0.10 ms, MEASURED reported) and chain count (3,616 pruned) are consequences
of the 2,355 box-unstable L2 neurons. A faster checker certifies the same 2/19.
Therefore the leverage is in (a) knowing the *true* robust fraction, so we
stop chasing a ceiling that may be near 2/19 already (Idea 1), and (b)
changing the model so that the three quantities shrink at ε = 1 without an
interval-based loss (Ideas 2 and 3). Everything else is second-order.

---

## 2. Premises in the briefing I believe are wrong or unsupported

**W1. "The network itself is robust; the abstraction loses 1600×." Not
established.** The evidence is 64–256 *random* corners of the ε-box (work plan
§3.4, Turn 1). A random corner of a 3072-dimensional cube is a very weak
attack on a linear form. Toy check (MEASURED, toy, `scratchpad/sampling_gap.py`):
for a linear form a·δ with δ ∈ {−1, +1}ⁿ, the ratio of the true half-range
‖a‖₁ to the half-range seen by 256 random corners is

| n (effective receptive field) | true / sampled |
|---:|---:|
| 75 (an L0 neuron) | 2.3–2.5 |
| 648 | 6.6–7.3 |
| 1,323 (an L2 neuron's input cone) | 9–10 |
| 3,072 (L3, logits) | 14–16 |

So "sampled width ≤ 20 at every layer" is consistent with true L3 widths of
roughly 200–300 LSB, which is the same order as CROWN's 100–570. CROWN may be
close to tight at L3, and the "5× gap to the sampled margin" is partly a gap
on the *sampled* side. The real minimum margin over the box could be far below
+1,900 and negative for a fair share of images. A sign-gradient step recovers
100% of ‖a‖₁ for a linear form; nobody has run one. Idea 1 fixes this in a
morning.

**W2. "Layer 2 is the target" and "cost is the bottleneck."** Cost bounds how
many *already-certifiable* images fit before the deadline. It does not change
2/19. The target is the count of unstable neurons, which is a training and
architecture quantity.

**W3. "Two-stage models reach only about 66%."** MEASURED (reported) for
`conv8_16_dense128`, `conv12_24_dense192`, `conv16_32_dense256`
(`experiments/sign_deep_source_model_search.json`): stride 3 in the first
convolution, 8–16 filters, and a hidden dense ReLU layer. That says nothing
about a stride-2, 24/64–96-filter two-stage network with a 4096–6144-wide
linear head straight to the logits (Idea 3). Do not let that table close the
door.

**W4. "Training-based approaches cost too much accuracy."** MEASURED for IBP
only. IBP's loss is computed through the same 10–1600× inflated intervals that
make the box pipeline useless, so the optimizer fights a proxy that is orders
of magnitude too pessimistic. That explains the 53–83% results and does not
transfer to adversarial training, whose loss is evaluated at actual points in
the box (Idea 2).

**W5. Test-set hygiene.** The test set has already been used once for the
current artifact (91.32%/91.35%). If a new artifact is frozen, its single test
evaluation is a *second* test use for the project. Pre-register it, say so in
the paper, and never iterate on it.

**W6. ε ∈ {1, 2, 4}.** At ε = 4 nothing on any artifact reachable in ten days
will certify (widths scale at least linearly; unstable counts faster). Keep
{1, 2} for the certified funnel and use ε = 4 only for REFUTED rows if Idea 1
finds them. ASSUMED.

---

## 3. The ideas

### Idea 1. Attack-to-truth triage, with ESBMC-replayed REFUTED rows

**Core insight.** Before spending another ESBMC-hour, measure the *true*
robust fraction at ε = 1 with an attack that actually optimizes the margin
over the byte box, on the exact deployed integer semantics. A violating input
is not a failure of the project: replayed by ESBMC on the deployed C it is an
ESBMC-decided REFUTED verdict, which turns `ABSTRACTION_INCONCLUSIVE` rows into
decided rows and tells you whether the finder or the model is the ceiling.

**Quantitative case.** By W1, 64 random corners see about 1/15 of the true
range of a 3072-input linear form (MEASURED, toy). The reported sampled
minimum margin (+1,900 median) is therefore an optimistic estimate of the true
minimum. Naturally trained CNNs are fragile at 1/255: on CIFAR-10, PGD at
1/255 typically removes a third to a half of the clean accuracy (ASSUMED,
literature from memory). GTSRB is easier and the margins bigger, so my guess
is that **3–7 of the 19 validation images fall to a proper attack at ε = 1**
(ASSUMED). If that is right, the certifiable ceiling on this model is 12–16/19,
not 19/19, and CROWN's 2/19 is closer to the truth than the briefing implies.
If instead 0–1 fall, the finder is the ceiling and Idea 5 rises in value.

**Soundness.** ESBMC checks: (i) the concrete adversarial bytes lie in the
byte box (|x_adv − x₀| ≤ ε per pixel, in [0, 255]), (ii) the PROVED encoder
maps them to the integer input, (iii) the deployed forward pass on x_adv and
on x₀, (iv) `assert(argmax equal)`. A `VERIFICATION FAILED` with a trace is a
REFUTED. Nothing is argued on paper except "a concrete counterexample refutes
a universal property", and the trust base is smaller than the chain scheme's
(no CSR, no finder output in the verdict). A finder that proposes a wrong
x_adv makes ESBMC return VERIFIED for that single point, which is *not* a
proof of anything and is recorded as "attack failed". No false VERIFIED path
exists.

**What changes.** Finder: new attack script (sign-gradient PGD on the Keras
surrogate in byte units, project to {−1, 0, 1}, then greedy integer coordinate
descent on the compiled `.so` via `replay_on_so`, `replay.py:62`; about 3 ms per
forward, so 3072 pixels × 3 values × a few sweeps ≈ 30–60 s per image).
Checker: one concrete-replay harness family. If ESBMC cannot symbolically
execute the loop-encoded `qnn_forward_fixed` on a concrete input in a 30-minute
time box (Turn 9 measured the loop encoding as superlinear), fall back to the
straight-line per-layer chunk renderer already used for chain steps: 3.15M
concrete multiply terms in 126 chunks of 25k, about 5 minutes summed (MEASURED
rate, ASSUMED total), composed by concrete values only. Model, C and property:
unchanged. **Paper claim:** the funnel gains a row, "REFUTED by ESBMC with a
concrete counterexample on the deployed C", and the headline becomes "ESBMC
*decided* k + r of N pre-registered regions", which is the standard VNN-COMP
form and is much stronger than "certified k".

**Cost and risk.** One day for one person (half a day attack, half a day
replay harness). Compute: minutes per image for the attack, ≤ 10 minutes of
ESBMC per counterexample. Main failure: the attack finds nothing, in which case
you have learned that the model is robust on the sample and the gap is on the
finder side; that is also a result, and it makes the "sampled margin" column
in the tables honest ("strongest attack margin" instead).

**Cheapest decisive experiment (2–3 hours, Sep 25 morning).** Inputs: the 19
correctly classified validation images at ε = 1 (later the pre-registered test
cohort at ε ∈ {1, 2, 4}). Run: 10 restarts × 20-step PGD on the float
surrogate with the margin loss `logit_t − max_{c≠t} logit_c`, step 0.5 byte,
project to the byte box, round, evaluate on the `.so`; then greedy integer
coordinate descent from the best point until no single-pixel move lowers the
exact integer margin. Measure per image: the attack margin (exact integer),
the ratio sampled-min / attack-min, and the class. **Threshold:** if ≥ 3/19
images flip, the model is the bottleneck: go straight to Ideas 2/3 and add
REFUTED rows. If 0–1 flip, keep the current model as a candidate and push
Idea 5. Either way, replace the "sampled corners" column everywhere with the
attack margin.

### Idea 2. A new frozen artifact by adversarial (PGD) fine-tuning, not IBP

**Core insight.** The CROWN gap is Σ_unstable |aᵢ|·wᵢ/4 minus clean margin.
Adversarial training at a tiny ε shrinks all three factors (fewer unstable
neurons at ε = 1, smaller local widths, smaller back-substituted |aᵢ|) and
raises the clean margin, while its loss is evaluated at real points in the box
and therefore carries none of the 10–1600× interval inflation that made IBP
destroy accuracy. **What is different from the failed IBP runs:** no interval
propagation anywhere in the loss; the only "robust" signal is the network's
own output at PGD points inside the ε = 2 byte box.

**Quantitative case.** Reported (ASSUMED, ERAN/VNN-COMP memory, not checked
here): on CIFAR-10 ConvSmall/ConvBig, DeepPoly/CROWN certifies under 5% of
images on the naturally trained network at ε = 2/255 and roughly 40–50% on
the PGD-trained one, at a clean-accuracy cost of a few points; at half the
training ε the certified fraction approaches the empirical robust accuracy.
GTSRB is easier than CIFAR (the same architecture family gives 91% here versus
about 75–80% there), so the accuracy cost at ε_train = 2/256 should be
≤ 1–2 points (ASSUMED). Target for validation: **fixed-point accuracy ≥ 90%
and CROWN-certified ≥ 8/19 (≥ 40%)**, against 2/19 today. A side effect that
matters for the deadline: the chain count scales with box-unstable neurons
(2,355 at L2 today); halving them halves the 172M pruned terms (ASSUMED,
linear).

Optional add-on if AT alone is short: a ReLU-stability hinge (Xiao et al.
2019) applied **only at L0 and L1**, where the IBP widths (14, 52) are nearly
exact, so the hinge is not inflated: `Σ_i max(0, τ − |z_i|)` on neurons whose
IBP interval straddles 0, small weight. Do not apply it at L2/L3 through IBP
widths; that reproduces the IBP failure.

**Soundness.** Nothing changes in what ESBMC checks. The artifact is quantized
by the existing path, the box pipeline is re-run per image, chains are
exported by the same exporter and checked by the same harnesses (the ReLU rule
accepts any slope). New trust assumption: none. The new artifact must be
frozen (sha256 in the ledger) before any finder runs on test (W5).

**What changes.** Model and deployed C: yes, a new frozen artifact, named,
with validation-only selection and one test evaluation, declared as the second
test use. Finder and checker: unchanged. **Paper claim:** "verification-aware
fine-tuning of the deployed QNN raised the fraction of regions ESBMC could
certify from 0 to k of N, at a clean-accuracy cost of Δ points" — the
natural-vs-fine-tuned comparison is itself a result and a better story than
"we tried to certify a natural network".

**Cost and risk.** Training script: about 100 lines of Keras `GradientTape`
alongside `finetune_ssv_ibp_direct_head.py` (which already has the model
builder, masks and validation loop). PGD-5 makes an epoch about 6× slower:
30–70 s; 30 epochs ≈ 15–35 min per run at about 3 GB RSS (MEASURED baseline,
ASSUMED factor). Quantize + parity + CROWN probe on 19 images: 5 minutes.
Then one box run (35 min) and one full chain run (4–6 h wall per the review)
for the new pilot. Main failure: AT at 2/256 does not move CROWN enough
(certified stays < 6/19) while eating 1–2 points; secondary failure: the
quantized model's accuracy drifts more than the float one (today 0.26%
disagreement, MEASURED reported).

**Cheapest decisive experiment (Sep 25, about 4 hours wall, memory-safe at 2
concurrent runs).** Grid: {ε_train = 1, 2, 3 bytes} × {PGD-5 sign steps} ×
{with / without the L0–L1 hinge}, 25 epochs each, from the 91% checkpoint at
lr 1e-4, BN frozen, depthwise-zero masks kept. For every run: quantize with
the existing Q16/F8 path, measure fixed-point validation accuracy, run
`crown_probe.py` (float64 CROWN on integer semantics, 9 s per image) on the
same 19 validation images plus the attack from Idea 1. Report a table
(val acc, CROWN certified/19, attack-robust/19, box-unstable count per layer
on the pilot image). **Go if any cell has val ≥ 90.0% and CROWN ≥ 8/19;
marginal if 5–7/19; no-go otherwise.**

### Idea 3. ARCH-2R: two ReLU layers and a direct linear head

**Core insight.** All of the damage is at L2 and L3 (77% and 90%
box-unstable, widths 1,324 and 4,385); L0 and L1 are nearly exact (widths 14
and 52; the L0 box *is* the exact per-coordinate interval, Turn 9). A network
whose only nonlinearities are L0 and L1, followed by one linear layer to the
43 logits, has no L2/L3 relaxations at all: CROWN through the head is exact up
to one rounding, so the certificate loses only at the two shallow layers
where the widths are tiny.

**Quantitative case (ASSUMED throughout).** Architecture: `conv 5×5, 3→24, s2`
→ ReLU → `conv 3×3, 24→64 (or 96), s2` → ReLU → `dense 4096 (6144) → 43`;
192k (or 280k) parameters. Expressible today with zero code change
(`RestrictedSequentialCNN` with two conv stages and a logits dense; the
quantizer sets `apply_relu` False on the logits automatically,
`conv_fixed_point.py:179`). Slack accounting for a margin chain: L1 unstable
≈ 600 × |dᵢ| ≈ 15 weight-LSB × width 40/4 ≈ 90k acc-LSB ≈ 350 logit LSB; L0
unstable ≈ 700 × back-substituted |aᵢ| ≈ 70 × width 14/4 ≈ 170k acc-LSB ≈
700 logit LSB; total ≈ 1–1.5k logit LSB against clean margins that a
2-layer network with a large linear head should have in the several-thousand
range. So a large fraction of correctly classified images should certify if
the accuracy is there. Checker cost (MEASURED rates, ASSUMED counts,
`scratchpad/cost_estimates.py`): margin chain 1.35M terms + 10k ReLU
coordinates ≈ 4 min per competitor; 42 competitors ≈ 2.8 h summed; L1 neuron
chains (1,120 × 16k terms) ≈ 1.2 h; total ≈ 4 h summed ≈ 1.4 h wall at 4
jobs, with no L2/L3 chains at all. Every forward pass has 1.5M terms instead of
3.15M, which also halves the Idea-1 replay.

**Soundness.** Unchanged: same checker, same harness types; the head is one
dense affine step (CSR is dense, 4096 terms per row, well under the 25k chunk).
The clamp of the logits still needs the two-sided margin closure from the
pilot. New trust assumption: none.

**What changes.** Model and deployed C: a new, shallower frozen artifact,
named. Finder and checker: unchanged. **Paper claim:** "an architecture with
two nonlinear stages and a linear head is both ≥ 90% accurate on GTSRB and
ESBMC-certifiable at ε = 1 for k of N regions; adding nonlinear depth
(the 91.3% four-stage model) drops the certifiable fraction to 0 at the same
ε" — a clean accuracy-versus-verifiability statement with real certificates on
both ends, unlike option (C) in the briefing, which has certificates only at
41.8%.

**Cost and risk.** Training: 120 epochs of the existing search script with a
new candidate JSON (`sign_deep_source_model_search*.json` pattern), 5–12 s per
epoch → 10–25 min per candidate; three candidates (64, 96, 128 filters at L1)
in parallel fit in memory. Main failure: validation accuracy lands at 86–89%.
The team's three-stage `conv24_48_96_dense384` reached 89.78% (MEASURED
reported) with *fewer* head parameters; the 4096–6144-wide linear head is the
bet. I put it at 40–50% to clear 90%; a 3-ReLU variant with a 3072→43 direct
head (`conv24_64_192_direct43`, also zero code change) is the fallback that
likely holds 90–91% but keeps the 77%-unstable L2, so it gains less.

**Cheapest decisive experiment (Sep 25, same afternoon as Idea 2, 3 candidates
× 20 min).** Train `conv24_64_direct43`, `conv24_96_direct43`,
`conv24_128_direct43` with the existing recipe and augmentation; quantize;
fixed-point validation accuracy; CROWN probe and Idea-1 attack on the 19
images. **Go if any candidate has val ≥ 90.0% and CROWN ≥ 10/19**; if the best
is 88–90% with CROWN ≥ 12/19, one more day of tuning (wider L1, AT from
Idea 2, dropout off) is justified; below 88%, drop it.

### Idea 4. A named, enumerable property: byte brightness band, decided exactly

**Core insight.** ESBMC is strongest at concrete evaluation. A perturbation
set small enough to enumerate needs no abstraction: ESBMC evaluates the
deployed C on every element and either all agree (PROVED) or one does not
(REFUTED). The natural one for traffic signs is a global brightness offset
δ ∈ [−k, k] bytes on every pixel (byte-clipped): 2k + 1 concrete inputs. It is
a *different property* — a one-dimensional slice of the L∞ ball of radius k —
and must be named as such. Its value is that it produces rows the paper can
call PROVED under every rule, for every image, on the exact deployed C.

**Quantitative case.** k = 8 gives 17 evaluations, k = 16 gives 33. One
concrete forward is 3.15M terms ≈ 5.3 min summed in 25k-term chunks (MEASURED
rate) or, if the loop-encoded `qnn_forward_fixed` verifies in ESBMC directly on
a concrete input, one call. A 17-point band per image is about 90 minutes of
ESBMC summed, 30 minutes wall at 4 jobs (ASSUMED). Eighteen images at k = 8:
one night.

**Soundness.** Everything is concrete and checked by ESBMC; the only paper-level
statement is "the property quantifies over a finite set that the harness set
enumerates", verifiable by counting. It must not be presented as, or mixed
into, the L∞ result.

**What changes.** Property (new, named). Checker: the same concrete replay
harness as Idea 1. Model, C, finder: unchanged. **Paper claim:** a separate
table, "exact brightness-band robustness of the deployed C, decided by ESBMC
for all N images at k = 8 and 16".

**Cost and risk.** Half a day once Idea 1's replay harness exists. Risk:
reviewers call it trivial; mitigate by keeping it clearly secondary and by
pairing it with REFUTED rows where the band flips the class (a genuine finding
about the deployed classifier).

**Cheapest decisive experiment (1 hour).** On the `.so`, sweep δ ∈ [−32, 32]
for the 19 validation images and record the smallest |δ| that flips the class
(if any). Pick k so that ≥ 80% of images hold; then time one ESBMC concrete
forward. Go if one forward is under 10 minutes.

### Idea 5. Margin-only case splits with shared neuron chains

**Core insight.** Neuron-bound chains dominate the cost (L2 alone 100M of
172M pruned terms) and are shared by every competitor; margin chains are cheap
(42 × 566k = 23.8M terms, MEASURED reported). So a branch-and-bound that
splits one or two high-slack L3/L2 neurons, re-derives *only the margin chain*
per leaf, and reuses every neuron chain, costs about 3–4 minutes of ESBMC per
leaf, not hours. The split assumption `z_i ≤ 0` (or `≥ 0`) enters the leaf's
certificate as a bound record `[l_i, 0]` (or `[0, u_i]`) tagged
`case_assumption`, and every use of a record in the checker is monotone in the
assumption, so the leaf harnesses are unchanged.

**Quantitative case.** The largest-coefficient unstable L3 neuron on the pilot
(n162) had box [−1455, 292] and chain [−236, 28] (MEASURED reported); splitting
it removes its relaxation entirely. Whether the top-k neurons carry most of a
5,655-LSB deficit is unknown; my prior is that the slack is spread over the
~600 CROWN-unstable L2/L3 neurons, so k ≤ 6 (≤ 64 leaves ≈ 4 h) helps only the
images whose deficit is ≤ 2× after α-CROWN. ASSUMED: +1–3 images on any
artifact.

**Soundness.** ESBMC checks each leaf chain exactly as today. The case
analysis "(z_i ≤ 0 → margin ≥ 1) ∧ (z_i ≥ 0 → margin ≥ 1) ⇒ margin ≥ 1" is a
meta-level argument of the same kind as DAG completeness; to keep every number
inside ESBMC, add a tiny closure harness per split node asserting
`min(bound_left, bound_right) ≥ 1` from the two leaf literals. New trust
assumption: records tagged `case_assumption` must never be used outside their
leaf — a coordinator invariant, checkable by the corruption regression suite.

**What changes.** Finder (float β-style splitting, α-CROWN slopes first),
coordinator (leaf bookkeeping), one record provenance kind. Model, C,
property: unchanged.

**Cost and risk.** Two days. Main failure: slack is spread, leaves multiply,
and nothing closes; you find out in the float finder before any ESBMC time.

**Cheapest decisive experiment (2 hours, float only).** For the 17 failing
validation images: run α-CROWN (finder-only), then rank the margin chain's
per-coordinate slack |aᵢ|·dᵢ, split greedily on the top neuron, recompute the
margin bound per leaf without intermediate refinement, repeat to ≤ 64 leaves.
**Go if ≥ 3 images close within 64 leaves; otherwise drop it.**

---

## 4. Rejected ideas

- **CROWN-IBP / SABR fine-tuning.** Still a certified-training loss whose target
  is IBP tightness; more accuracy cost than AT at this ε and a week of tuning.
- **Coarser input quantization (x ≫ 4) to shrink the reachable set.** Only 2/16
  of pixels sit at a bucket edge, but each of those moves by 16 raw units: the
  expected Σ|aᵢ|·widthᵢ doubles rather than shrinks (ASSUMED, arithmetic).
- **Average-pool front end / 16×16 input.** ‖a‖₁ over raw pixels is unchanged
  by averaging; no bound gain, likely accuracy loss.
- **Lipschitz/orthogonal layers with an ESBMC-checked Gershgorin bound on WᵀW.**
  Elegant, but rounding noise of √n/2 LSB per layer is the same order as the
  L2 radius of the box, and orthogonal convs are a new training regime; not in
  ten days.
- **Weight sparsification to cut terms.** Real 2–3× checker saving, but it does
  not move the certifiable fraction; fold into Idea 2 only if cost becomes the
  binding constraint.
- **Zonotope/DeepZ or differential (ReluDiff-style) domains.** Same tightness
  class as CROWN for one network; no new leverage.
- **Patch (L0) threat model.** Same L2/L3 collapse; the free inputs are fewer
  but 255× wider.
- **MILP as a bound source.** No chain behind a MILP bound, and Option A times
  out at 32 terms; MILP stays a refuter or a slope/split proposer (review P5).
- **Full symbolic BMC of any layer ≥ 2.** Measured dead end (20 GB, no verdict).
- **Certifying a larger ε on the current model.** Strictly harder; W6.

---

## 5. Recommendation

**Start tomorrow morning (Sep 25) with Idea 1**, because it costs three hours,
needs no ESBMC, and decides the rest: it tells you whether the current model's
ceiling is near 2/19 (then only a new artifact helps) or near 19/19 (then the
finder is the problem and Idea 5 rises). While it runs, **launch the Ideas 2/3
training grid** (Idea 2 fine-tunes, Idea 3 candidates; 2–3 concurrent runs,
≈ 3 GB each, no AOSP build). By Sep 25 evening you have one table:
artifact × {fixed-point val acc, CROWN-certified/19, attack-robust/19,
box-unstable per layer}. **Go/no-go G0, Sep 25 22:00:** pick the artifact with
val ≥ 90.0% and the highest CROWN-certified fraction; if no new artifact beats
6/19, stay on the current model and follow the review's plan (B) with Idea 5.

Sep 26: freeze the chosen artifact (sha256, validation-only selection,
pre-registered single test use), run its box pipeline on the validation pilot,
export chains, and start the full ESBMC run; Codex lands the review's fixes
(H1–H4, P1) in parallel, since they are needed on any artifact. Pre-register
the test cohort and ε grid **before** any finder or attack runs on test.
Sep 27 evening: G1, one validation image end-to-end VERIFIED with a complete
DAG. Sep 28–Oct 1: test cohort, one region at a time, VERIFIED / REFUTED /
INCONCLUSIVE all reported. Oct 2: freeze numbers and write.

**Insurance to run in parallel from Sep 26:** Idea 4 (brightness band) and
the REFUTED rows from Idea 1, both on the *current* artifact, both concrete,
both ESBMC-decided. Together they guarantee the paper has a non-empty
ESBMC-decided results table even if no L∞ region certifies. Keep them in their
own named tables.

**If I am wrong about the coin flip** and neither new artifact clears 90%
with a real gain, say so in the paper: the natural 91% model is attackable at
ε = 1 on r/N regions (REFUTED), certifiable on k/N (probably 0–2), and the
checker verifies real certificates at 0.1 ms/term; the open problem is a
training recipe that keeps 90% and makes the deployed integer network
certifiable, and the AT and ARCH-2R numbers from Sep 25 are the first data on
it. That is a smaller paper, but an honest one, and every number in it is
tagged.

---

## 6. Numbers used

| quantity | value | tag |
|---|---:|---|
| float / fixed-point test accuracy | 91.32% / 91.35% | MEASURED (reported) |
| box widths at bridges 0–3 (median) | 14 / 52 / 1,324 / 4,385 | MEASURED (reported) |
| box-unstable neurons, pilot image, L0–L3 | 314 / 560 / 2,355 / 692 | MEASURED (reported) |
| CROWN certified, 19 val images, ε = 1 | 2/19, median bound −5,655 | MEASURED (reported) |
| smallest sampled margin (64 corners) | ≈ +1,900 | MEASURED (reported), weak attack |
| random-corner under-estimation of a linear range, n = 3072 | 14–16× | MEASURED (toy, this session) |
| ESBMC straight-line cost | 0.10 ms and 36 KB per term | MEASURED (reported) |
| ReLU coordinate cost | ≈ 10 ms | MEASURED (reported via review) |
| pruned chain terms per image / L2 share | 172.2M / 100.3M | MEASURED (reported) |
| margin chains | 42 × 566k = 23.8M terms | MEASURED (reported) |
| forward-pass terms, current / ARCH-2R(64) | 3.15M / 1.52M | computed |
| ARCH-2R per-image checker cost | ≈ 4 h summed, ≈ 1.4 h wall | ASSUMED |
| AT accuracy cost at ε_train = 2/256 on GTSRB | ≤ 1–2 points | ASSUMED |
| CROWN-certified fraction after AT | 40–70% of correct images | ASSUMED (literature, from memory) |
| P(ARCH-2R ≥ 90% val) | 0.4–0.5 | ASSUMED |
| failed two-stage candidates | conv8/12/16, stride 3, hidden dense | MEASURED (experiments JSON) |
