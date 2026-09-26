# Fable review 1: soundness of the ESBMC-checked CROWN chains, and the 10-day plan

Independent review, 2026-09-24. I did not write any of the code. Sources: the
briefing, `work_plan_20260923.md`, `claude_codex_chat.md` (Turns 1–9),
`src/verification/crown_chain_certificate.py`, `interval_lemmas.py`,
`conv_contracts.py`, `arith_kernel.py`, `esbmc.py`. Result files are not in
this checkout; every number from them is **reported**, not re-measured by me.

Tags: **PROVED** (ESBMC `VERIFICATION SUCCESSFUL`, non-vacuous), **MEASURED**
(code run on data, or my code inspection of this checkout), **ASSUMED (argued)**
(mathematics or estimates).

---

## 0. Premises in the briefing that I think are wrong or misleading

Say these first, as requested.

**P1. "ESBMC returns VERIFIED only through ESBMC" is not yet true in five
places in the chain checker (MEASURED, code inspection).** These harnesses
assert equality of two Python-computed literals, so ESBMC decides nothing:

| what | where | Python computes |
|---|---|---|
| concretize dot product with the box corners | `crown_chain_certificate.py:463-473` | `scaled` |
| rounded final bound ⌈scaled/2¹⁶⌉ or ⌊−scaled/2¹⁶⌋ | `:574-584` | `expected_bound` |
| ReLU-step constant Σdᵢ | `:433-437` | `expected_constant` |
| ‖a‖₁ in the affine closing harness | `:290, :304` | `abs_sum` |
| hlow = ReLU(l) and the two ReLU corner values f(l), f(u) | `:150, :384-385` | `source_lower`, `f_lower`, `f_upper` |

The mathematics is trivial in every case, but each one is exactly the kind of
"exact Python checker" that rule 1 forbids. All five are one-line fixes (emit
the C expression instead of the number). The decisive checks that *do* carry
the proof (affine `g`, residual, `bias_dot`, `residual_dot`, the nondet-`z`
ReLU inequality, the nondet clamp closure, the rounding lemma) are genuinely
ESBMC-checked.

**P2. Option (C) is not a fallback that produces certificates.** The work plan
§4 already reports (MEASURED) that the 3-stage model saturates 384/384 and the
2-stage model's logit box spans the full range under the box pipeline. So the
box-only "accuracy-versus-verifiability curve" is 1 certified model (41.8%)
and three zeros. It is a good *motivation table* (one paragraph, existing
data) but not insurance for the results section. Insurance has to come from
the chain pipeline itself.

**P3. The per-image cost is more likely 4–6 h wall than 2–3 h (ASSUMED,
from Turn 9's own numbers).** Turn 9 §4 measured ReLU chunks at 61% of the
pilot time (359 s / 68 chunks / 512 ≈ 10 ms per symbolic coordinate) and
estimates 5.0M non-bound-free coordinates in the naive closure; the pruned
closure will be roughly half. That alone is ≈ 7 h summed, on top of ≈ 5 h of
affine terms. At the measured 2.9× speedup for 4 jobs that is ≈ 4 h wall,
before any ε = 2 or 4 attempt. The order of magnitude is unchanged, and it
still fits a handful of images, but do not plan the cohort on 2–3 h.

**P4. "Layer 2 (100M terms) is the right target" is half right.** Layer-2
*chains* are the right target, but the cost they carry is as much ReLU
coordinates as affine terms. Any lever must cut chain count or chain depth,
not only affine terms. Early-stopping L2 chains at the z1 records removes
both (§3.4).

**P5. MILP cannot feed this checker as a bound source.** A MILP bound at L3 has
no chain behind it (its certificate is a branch tree, not a Farkas
combination), and Option A (symbolic recheck) times out at 32 terms. So "MILP
on L3/L4" in option B+ cannot become a bound record. MILP stays legitimately
as (i) an LP-relaxation dual exported as a chain, (ii) a diagnostic that picks
which neurons to tighten, or (iii) a *refuter* whose concrete counterexample
ESBMC replays on the deployed C, giving an honest REFUTED. Say this in the
paper rather than letting a reviewer discover it.

---

## 1. Soundness findings

### Holes, most severe first

**H1 — Bound-free identity slope ignores the output clamp. Unsound as
implemented. Severity: high in principle, low in practice; fix is cheap.**

Code: `render_relu_step`, `crown_chain_certificate.py:365-374`.
`bound_free = a > 0 and a_out in {0, a}` emits three literal assertions and
`continue`s: no bound record is looked at, no nondet `z`. The argued
justification is "a·ReLU(z) ≥ a·z holds for every z". That is true for
h = ReLU(z), but the deployed value is h = ReLU(clamp(z)). For z > 32767,
h = 32767 < z, so a·h − a·z = −a·(z − 32767) < 0, while the step assumes ≥ 0.

*Exact false-VERIFIED condition:* a coordinate with a > 0, a_out = a, whose raw
pre-clamp value can exceed 32767 on the input box, and no upper bound on it is
checked anywhere in the DAG. The finder is untrusted and picks a_out; the
identity slope is exactly what CROWN picks for every stable-active neuron, so
this case is common (Turn 9: 50% of ReLU coordinates are bound-free).

*Why it has not bitten:* on 06330 no hidden neuron reaches the Q16 range
(MEASURED, bridge table). But that is a measurement on another image, not a
proof, and the checker never looks at it.

*Fix (≤ ½ day):* for a_out = a require the neuron's upper fact to be present
and either `box_high < 32767` (box side, strict) or a VERIFIED upper chain with
`chain_high ≤ 32767`. No new chains are needed because every neuron will have
a box proof once the box run for the image exists (Turn 9 §3). Better still,
apply the general fix under H2 and this case disappears.

The a_out = 0 half of the rule (a·h ≥ 0) is genuinely bound-free: h ≥ 0 holds
with or without the clamp. **ASSUMED (argued), correct.**

**H2 — The ReLU harness models h = ReLU(z), never the clamp, and the planned
one-sided export gives it a `z ∈ [l, u]` assumption where one side may not be
a proved fact about the *raw* value. Severity: medium, a v2 design gap.**

Code: `:401-406` (`z ∈ [l,u]`, `value = a·max(z,0) − a_out·z`).
For the pilot both sides are exported and `_validate_bound_record` enforces
strict interior for `box_only` and `chain ∈ Q16` for `chain_intersect_box`
(`:165-175`), so the pilot is sound here. In the pruned v2 export (Turn 9 §3)
a record can have one chain-backed side and one box-backed side. The
box-backed side is a bound on clamp(z), which bounds raw z only if it is
*strictly* inside Q16. If the exporter copies a box endpoint into
`chain_low/chain_high`, the current assertion `chain_low ≥ q_low` accepts
−32768 and the harness then assumes z ≥ −32768 while the real raw z may be
−40000; for a_out < 0 the inequality fails there.

*Fix, which also subsumes H1 and turns all three pruning rules from argued into
ESBMC-checked:* give each record per-side provenance
(`lower_source`, `upper_source` ∈ {`box`, `chain:<id>`, `none`}); in the
harness use `h = ReLU(clamp(z))`, assume only the sides that are backed
(strict for box sides), and let `z` range over a coarse per-layer raw envelope
`[−Z_j, Z_j]` for an unbacked side, where Z_j = Σ|w|·32767/2⁸ + |b| is one
concrete harness per layer per model. Then ESBMC itself returns FAILED for
H1-type coordinates instead of the paper arguing about them. Cost per
coordinate is unchanged (one linear query in `z`).

**H3 — No check that a chain's steps walk the layers in order. Severity:
medium; a malicious or buggy exporter can splice layers. Fix: 10 lines.**

Code: `render_chain_closure`, `:554-570`, and `load_crown_chain_certificate`.
The closure checks coefficient-by-coefficient links and sizes, but never that
`right.layer_index` follows from `left.kind` and `left.layer_index`
(affine at j → relu at j−1 → affine at j−1 → … → relu at 0 → affine at 0 →
concretize), nor that the first step's layer equals the `lambda_var` layer.
Because layer widths decrease (6144, 4096, 3072, 768, 43), a step that
declares a *smaller-indexed* layer than it should passes every range check,
and each step is verified in isolation against its declared CSR. The result is
a chain that is internally consistent for a *different* network (with a layer
skipped or repeated) yet is accepted as a bound on the named neuron.
Also missing: `layers[j].cols == layers[j−1].rows` in the loader.
*Fix:* assert the expected sequence in `chain_close` as literals and in the
loader; both trivial.

**H4 — Deployment binding is absent, and it is inherited from the box
pipeline. Severity: high for the paper's claim, unchanged from the previous
paper.** The chain harnesses prove facts about the JSON's `lowered_csr`. The
box harnesses prove facts about `conv_output_terms()` (Python,
`conv_contracts.py:103-125`, used at `interval_lemmas.py:38`). Neither ever
executes or reads the deployed `qnn_conv_native.c`; the box ledger only
records its sha256 (`conv_proof.py:626`). So today nothing checked by ESBMC
says "these weights, this padding, this stride are what the C program
computes". Turn 8 §6 item 3 asks for a hash binding, which shows the JSON
equals an npz, not that the lowering is right. *Fix (1 day, once per model):*
a concrete harness per layer that `#include`s the deployed C and asserts, for
every output i and every nonzero, that the weight/bias the deployed code
indexes equals the CSR entry, using the deployed code's own indexing (no
Python re-lowering). ≈ 3.2M concrete terms in total ≈ minutes of ESBMC. If the
generated C does not expose per-neuron accumulators, the behavioural variant
(feed 2⁸·e_k and a zero input to a pre-ReLU entry point) needs a small
generator change. This is the single item that most raises the credibility of
"ESBMC decides", and it retroactively strengthens the box pipeline too.

### 5.1 Affine step — sound, discharged by ESBMC except for literals in P1

Let z = round(acc/2⁸) + b with acc = W h. For any integer a:
a·z = Σᵢ aᵢ·round(accᵢ/2⁸) + a·b ≥ (1/2⁸)·g·h − ‖a‖₁/2 + a·b, with g = Wᵀa,
using |round(t) − t| ≤ ½ for either sign of aᵢ and either sign of accᵢ
(**PROVED** as the generic lemma `−128 ≤ 256·round(n/256) − n ≤ 128`,
`:596-616`, over all int64 n, which covers negative and tie cases). Write
gₖ = 2⁸·a_out,k + rₖ with rₖ ∈ [0, 2⁸) (floor division; **PROVED per
coordinate**, `:268-269`, including negative g). Since rₖ ≥ 0 and
hₖ ≥ hlowₖ ≥ 0: g·h/2⁸ ≥ a_out·h + r·hlow/2⁸. Hence
a·z ≥ a_out·h + c_in + a·b + (r·hlow − 2⁷‖a‖₁)/2⁸, and because the left side
and a_out·h are integers the constant may be floored. Floor is the safe
direction for a lower bound. **ASSUMED (argued)** for the derivation;
`g`, `r`, `bias_dot`, `residual_dot`, the floor and the final equality are
**PROVED** in the closing harness (`:291-315`), with ‖a‖₁ the exception (P1).

- *Negative a, negative g:* covered by |aᵢ| and floor division; ESBMC
  asserts `0 ≤ r < 2⁸` on the recomputed r, so a wrong Python floor would
  FAIL. **PROVED.**
- *Halfway case:* the lemma is proved over the deployed `round_half_away`
  code path, ties included. **PROVED.** (Minor: the lemma re-declares the
  rounding function textually, `:602-605`, instead of including
  `render_arith_kernel()`; bind it to the shared kernel.)
- *hlow from a chain record:* h = ReLU(clamp(z)) ≥ ReLU(clamp(l)) = ReLU(l)
  for any l ≤ 32767, by monotonicity of both maps. So the lower side is sound
  from either a chain or a box, and does **not** need the upper-side clamp
  inactivity. **ASSUMED (argued), sound.** It is conditional on that chain
  being VERIFIED, which is the DAG requirement.

### 5.2 ReLU step — sound and discharged by ESBMC; the 3-point rule is not even load-bearing

The harness (`:401-406`) takes a nondeterministic z ∈ [l, u] per coordinate
and asserts a·ReLU(z) − a_out·z ≥ d with the exported d. That is a direct
universally-quantified proof for that coordinate, for *any* integer a_out.
How d was found (3-point rule or otherwise) affects only tightness. The
3-point rule itself is correct: f is piecewise linear with one breakpoint at
0, so its minimum on [l, u] is at l, u, or 0. **ASSUMED (argued)** for the
rule, **PROVED** per coordinate for what is used.

*Sum obligation:* not needed. The coordinates are independent variables, each
inequality holds for all zᵢ in its range, so the sum holds for all joint
values; the constant Σdᵢ is linked to the next step in `chain_close`. The only
gap is that Σdᵢ is a Python literal (P1). The int64 width guard
(`:408-414`) is asserted, so the int32/int64 encoding is safe. **PROVED.**

Caveat: everything here assumes the record's [l, u] bounds the *raw* z on both
sides (H1/H2).

### 5.3 Concretize and margin closure — mathematically sound; concretize is Python-only

*Concretize:* a·x ≥ Σ aᵢ·(aᵢ ≥ 0 ? xlᵢ : xuᵢ) is the exact box minimum;
2¹⁶·z ≥ scaled ⇒ z ≥ ⌈scaled/2¹⁶⌉; for λ = −2¹⁶e, −2¹⁶z ≥ scaled ⇒
z ≤ ⌊−scaled/2¹⁶⌋. Directions correct. **ASSUMED (argued)**; both the dot
product and the rounding are Python literals in the harness (P1), so today
this step is **not** ESBMC-checked.

*Two-sided clamp rule:* with raw_t − raw_c ≥ 1, clamp is monotone so
clamp(t) ≥ clamp(c), with equality only if both saturate on the same side:
(raw_c ≥ 32767 ∧ raw_t ≥ 32768) or (raw_t ≤ −32768 ∧ raw_c ≤ −32769). The
stated conditions are exactly the negations, so the "iff" is right, including
the edges (raw_c = 32766, raw_t = 32767 stays strict). **ASSUMED (argued),
correct**, and the harness (`:785-808`) quantifies both raws in `__int128`,
assumes only the three exported facts and asserts the clamped strict
inequality, so it is **PROVED** independently of the argument. Two notes:
the `top`/`bottom` literals (`:802-803`) are Python booleans and redundant
(harmless); and the competitor's `raw ≤ competitor_box_high` assumption is
sound only because 31,494 < 32767 — assert the strict interior of that box in
the harness, otherwise a saturated box would put an unsound assumption into
the harness (it happens to still FAIL in that case, but by luck of the
structure, not by design).

### 5.4 Pruning rules

- a > 0, a_out = 0: bound-free, sound with the clamp. **ASSUMED (argued).**
- a > 0, a_out = a: **not** bound-free; needs an upper fact (H1).
- a < 0, a_out = a: needs the lower side only; with the clamp, h − z ≤ −l
  still holds for all z ≥ l (saturation only lowers h). Sound.
- a < 0, a_out = 0: needs the upper side only; h ≤ ReLU(u) holds with the
  clamp. Sound.
- residual folding, hlow = ReLU(l): lower side only when l > 0. Sound (5.1).

All **ASSUMED (argued)** today; the H2 fix makes them **PROVED** per coordinate
at no extra cost, which is how I would answer a reviewer.

### 5.5 Composition and rule 1

What ESBMC establishes: ≈13k leaf implications, each between explicitly
listed integer vectors/constants, plus per-chain link equalities
(`chain_close`) and the clamp closure. What Python establishes: (a) every
required leaf exists and returned VERIFIED with a matching identity, (b) the
five literal computations in P1, (c) the layer sequence (H3, currently
nobody). The box pipeline is the same shape (`conv_proof.py`); the professor
accepted it in the previous paper.

My view: (a) is unavoidable in any BMC-based compositional proof and does not
violate rule 1 *provided no number in the verdict is computed outside ESBMC*.
Today (b) and (c) break that. **Smallest change that makes ESBMC the
decider on every number (≈1 day):**

1. Fix P1 and H3 so every arithmetic fact, including the last dot product and
   the ceiling, is C-evaluated inside a harness.
2. One **per-image manifest harness**, scalar-only: for each of the 42
   competitors assert margin-chain bound ≥ 1, target-lower and competitor-upper
   facts as used by the closure; for each bound record assert
   (lower, upper) == (named chain bound or box endpoint) with box sides
   strictly interior; for each `box_only` fact assert the box digest's
   endpoints. ≈30k literal assertions; fits in one or a few files (the 11 MB
   16-neuron file MEMOUTed, this is ≈2 MB).
3. Then Python's only remaining job is bookkeeping: "each harness listed in
   the manifest, with this content hash, has a VERIFIED result with this
   identity". That is auditable by a reader with `sha256sum` and `grep`, and
   the paper should say so in one paragraph with the obligation count.

A single ESBMC harness that *symbolically* checks the whole DAG is not
feasible (Option A timed out at 32 terms); do not promise it.

### 5.6 Trust base — see §2.

### 5.7 Other ways a wrong proposal could become VERIFIED

- Layer splicing (H3). Real.
- Identity-slope on a saturating neuron (H1). Real.
- Box-side endpoint used non-strictly in v2 (H2). Real for v2.
- CSR that is self-consistent but not the program (H4). Real; inherited.
- A `box_only` record whose box was proved for another image (today all
  9,632 of them, Turn 9 §2). Handled by status `CONDITIONAL`, must be bound
  by digest + image identity + deployment sha in v2.
- Result reuse for a different harness: the store keys by digest/identity
  (`verified_cache/<digest>.json`); make the harness content hash part of the
  identity so a VERIFIED file cannot satisfy an edited harness.
- Overflow inside a 25k-term straight-line sum: impossible in `__int128`
  (terms < 2⁶³, < 2¹⁵ terms) — **ASSUMED (argued)**. Turn 8 §6 item 4 wants
  per-step envelopes; cheaper and stronger: check whether the `paper-z3`
  profile enables `--overflow-check` (I could not see the flag list in
  `esbmc.py`, only the profile names). On concrete code the check is nearly
  free and makes the envelope **PROVED**.
- Property identity: fixed in the Turn 8 addendum (lambda ↔ certificate id).
  **PROVED** by the regression.
- ReLU `bound_ids` pointing at another neuron: rejected (`:376-381`). Fine.

Nothing else found. The affine/ReLU/closure core is well built.

---

## 2. Trust base, ranked by risk

| # | Item a reader must trust beyond ESBMC + solver | Tag | Removable in 10 days? |
|---|---|---|---|
| 1 | The lowered CSR / `conv_output_terms` equals the deployed C (H4). Shared with the box pipeline. | ASSUMED | **Yes** — per-layer concrete equality harness against the included deployed C (1 day) |
| 2 | Chain steps walk the layers in order (H3) | nothing checks it | **Yes** — 10 lines |
| 3 | Bound-free identity slope needs no upper fact (H1); per-side strictness in v2 (H2) | ASSUMED, wrong for H1 | **Yes** — clamp in the ReLU harness + per-side provenance (½–1 day) |
| 4 | Python literals: concretize, ceil/floor, Σdᵢ, ‖a‖₁, ReLU(l), f(l), f(u) (P1) | ASSUMED | **Yes** — emit expressions (½ day) |
| 5 | DAG completeness and identity matching (Python coordinator) | ASSUMED | Mostly — manifest harness (§5.5) leaves only file bookkeeping |
| 6 | Box facts for *this* image exist and are bound by digest | MISSING today | **Yes** — box run per image (~35 min) + digest binding |
| 7 | Scalar-local rewrite of the harness renderer | ASSUMED | Partly — the corruption regressions are the evidence; keep them in CI |
| 8 | `__int128` intermediate overflow | ASSUMED | **Yes** — `--overflow-check` on concrete harnesses if not already on |
| 9 | Input encoder and ε-box (byte clipping) | PROVED in the box pipeline | Reuse; bind by sha |
| 10 | Derivations in 5.1–5.4 (paper-level) | ASSUMED (argued) | No, and that is fine: each *instance* is PROVED |

Items 1–4 and 6 are the ones a hostile reviewer will find; all fit before
the go/no-go.

---

## 3. Recommendation and 10-day plan

**Plan: (B), hardened, with α-CROWN as a strictly time-boxed finder upgrade.
Drop (C) as a results-section item (P2); keep it as a one-paragraph motivation
table from existing data. Do not start (D).**

Reasoning: the checker works (178/178 PROVED on real chains, corruption
regressions FAILED as they should); what is missing is completeness (DAG,
42 competitors, box run, binding) and the soundness fixes above, all of which
are days, not weeks. The number of certified images is set by the finder's
closing rate, not by ESBMC time, and α-CROWN improves that rate without
touching the checker (5.2). CROWN-IBP needs retraining, re-quantisation,
re-export and new box runs on a machine that OOMs; not in 10 days.

Machine rules throughout: `esbmc_jobs=4`, 4 GB per query, 6 GiB
`MemAvailable` guard, no AOSP build on campaign nights, cache-backed reruns.

| Day | Codex (checker) | Claude (finder) | Gate |
|---|---|---|---|
| **Sep 25 Thu** | Fix H1–H3 and P1; add `--overflow-check` if absent; add `split` to the box pilot and **run the box proof on val 3652** (~35 min); start the deployment-binding harness (H4) | v2 exporter per Turn 9 §5 with per-side provenance; **pre-register the test cohort** (18 images × ε∈{1,2,4}, selection rule in the repo, committed); 30-min Python experiment: margin cost of early-stopping L2 chains at z1 and of dependency pruning within the 2,601 slack | — |
| **Sep 26 Fri** | Finish H4 harness; manifest harness (§5.5); coordinator status logic: VERIFIED only if manifest complete | Export v2 for 3652 (all 42 margins, pruned closure); start the full run in the afternoon (expect 4–6 h); α-CROWN on the 19 validation images, **frozen by 22:00 whatever the result** | — |
| **Sep 27 Sat** | Rerun anything that FAILED/TIMEOUT; diagnose | Run the frozen finder on all 54 test regions (~10 min); box proofs for every region it closes (35 min each) | **G1 (evening): val 3652 end-to-end VERIFIED with complete DAG, 42/42 competitors, box facts bound, H4 harness VERIFIED?** |
| **Sep 28 Sun – Sep 30 Tue** | ESBMC on closed test regions, ε = 1 first, then 2, 4; one at a time, ~4–6 h each | Optional: for regions the finder cannot close, PGD/MILP search for a concrete counterexample; ESBMC replays the deployed C on it (seconds) → honest REFUTED rows | **G2 (Sep 29 evening): ≥ 1 *test* region VERIFIED?** |
| **Oct 1 Wed – Oct 3 Fri** | Freeze numbers Oct 2 evening; tables from ledgers only; CI runs the corruption regressions | Write; related work; the trust-base paragraph | — |
| **Oct 4 Sat** | Submit | | |

**If G1 fails because of cost:** apply early-stop/dependency pruning only if
the Sep 25 measurement shows the margin survives; rerun overnight. **If G1
fails because an obligation FAILED:** that is the finder or exporter being
wrong, which is the system working; fix and rerun, but do not touch harness
semantics to make it pass. **If G1 has not passed by Sep 28 evening:** the
paper is outcome 2 below; stop new experiments and write.

**If G2 fails** (finder closes nothing on test): outcome 2 with the validation
pilot as the end-to-end demonstration and the 54-region table as the honest
negative; do not move validation images into the cohort.

### 3.4 Cost levers, in the order I would try them

1. **Early-stop L2 chains at z1 records** (finder, measure first). Removes
   the two deepest affine steps *and* the layer-1/0 ReLU steps of every
   layer-2 chain; L2 is 100M of 172M affine terms and most ReLU coordinates.
   The Turn-1 ablation does not predict its effect (it replaced L1 bounds by
   boxes, not by chain-tightened records), so measure on 3652; you have
   2,601 units of slack.
2. **Dependency pruning within slack** (finder). Same measurement.
3. **ReLU chunk size sweep** 512 → 1024 → 2048 with the int32/int64 encoding
   (checker). Per-coordinate cost is mostly per-harness overhead; the 6,143
   file MEMOUTed, so stop at the first MEMOUT.
4. **16-bit generic ReLU lemma** under Z3, one job, 30-minute time box. If it
   verifies, every coordinate becomes three concrete evaluations (with the
   H2 clamp form, still sound). If it does not, drop it; do not spend a day.

---

## 4. Paper claims per outcome

**Outcome 1 — ≥ 1 pre-registered test region VERIFIED end to end.**
Claim: *"For k of 54 pre-registered (image, ε) regions on GTSRB, ESBMC proved
the deployed fixed-point C classifier locally robust, by discharging N
integer obligations that compose an exact-integer CROWN certificate; every
obligation, including the final clamped margin and the binding of the
certificate to the deployed weights, is an ESBMC VERIFICATION SUCCESSFUL."*
Must not claim: a certified *rate* on GTSRB; that the float model is robust
(transfer is MEASURED only); that the method scales to the test set (4–6 h
per image); that MILP tightened anything (P5); that the composition is an
ESBMC theorem (it is a manifest of ESBMC theorems, §5.5).

**Outcome 2 — only `CONDITIONAL_VERIFIED` (or validation-only end to end).**
Claim: *"We give an ESBMC-checkable certificate format for relational
(CROWN-style) bounds on deployed integer networks; on a validation image all
N local obligations verify; end-to-end closure on pre-registered test regions
was not achieved within budget, and we report all 54 as
ABSTRACTION_INCONCLUSIVE / finder-not-closed."* Must not claim: "certified",
"verified image", or any robustness fact about the deployed model; must not
present the validation pilot as a test result.

**Outcome 3 — zero, pilot did not close.**
Claim: the method and the measured gap: boxes collapse (table), CROWN chains
are 10–40× tighter (MEASURED), the checker verifies real chains at
0.1 ms/term (MEASURED), and the remaining obstacle is X (whatever G1 showed).
Must not claim: soundness of the *pipeline* as a whole (only of the checked
pieces); any certified region; that the 91% model is or is not robust.

In all three: the two accuracy numbers (91.32 / 91.35) are MEASURED and can
be stated; `ABSTRACTION_INCONCLUSIVE`, TIMEOUT, REFUTED stay as they are.

---

## 5. Selection and reporting protocol

- **Pilot on a validation image chosen because CROWN certifies it: fine for
  engineering, not a result.** Say so in one sentence.
- **Finder-then-ESBMC on pre-registered test images is legitimate**, on three
  conditions: (i) the cohort and ε grid are committed before any finder run
  on test (Sep 25); (ii) every finder setting, including α-CROWN's, is frozen
  on validation before that run (Sep 26 22:00); (iii) every region appears in
  the table. Nothing is tuned on test, so rule 4 holds.
- **Report a funnel per ε, never a single rate:** cohort N → correctly
  classified → finder closes k → ESBMC VERIFIED m (m ≤ k, with reasons for
  k − m: TIMEOUT / budget / FAILED) → REFUTED r (only if ESBMC replayed a
  concrete counterexample). Also report the box pipeline's verdict on the
  same regions (all INCONCLUSIVE, presumably), which is the honest baseline.
- **Weak classes 27, 30, 0 stay in the cohort.** They were chosen before, so
  removing them after the finder fails on them is cherry-picking. Zero rows
  are informative: robustness verification is hardest where accuracy is
  lowest.
- **Never substitute** a patch-only threat model or a validation image for a
  test region; if used, it is a separate table with its own name.

---

## 6. Bottom line

The certificate core is sound and genuinely ESBMC-discharged: affine steps,
per-coordinate ReLU inequalities with nondeterministic z, the rounding lemma
and the two-sided clamp closure are all PROVED and non-vacuous. Three concrete
holes exist today — the identity-slope bound-free case ignores the output
clamp (H1), nothing checks that a chain's steps walk the layers in order (H3),
and five closing computations are Python literals (P1) — plus the inherited
absence of any ESBMC link between the certificate and the deployed C (H4);
all are fixable in one to two days and should be fixed before the first
end-to-end run. Spend the 10 days on hardening and completing (B), with
α-CROWN frozen by Sep 26 and a pre-registered 54-region test funnel; drop
option (C) as a result because the box pipeline already gives zeros beyond
one stage, and do not start CROWN-IBP. Budget 4–6 h wall per certified image,
not 2–3, and decide on Sep 27 (pilot) and Sep 29 (first test region) which of
the three paper claims you are writing.
