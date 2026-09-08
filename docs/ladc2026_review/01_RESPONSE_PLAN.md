# Revision strategy — how to answer LADC 2026

*Written from the position of a formal-methods researcher preparing a revision.
This document gives the **path** for each objection: what argument to make, what
evidence would make it stick, and what would falsify it. It deliberately does not
pre-write the answers, because several of them depend on measurements we have not
taken yet.*

---

## 0. Read the reviews correctly before acting on them

The three reviews disagree about the recommendation but agree almost completely
about the facts. Sorting them by what they actually contest:

| Contested | Reviewers | Nature |
| --- | --- | --- |
| Soundness of the method | *nobody* | — |
| Novelty of the target (deployed C semantics) | nobody; R1 and R3 praise it | — |
| Evidence for the scalability claim | R1, R2 (implicit), R3 | **empirical, fixable** |
| Honesty/legibility of the failure profile | R1, R3 | **reporting, fixable** |
| Positioning vs. nearest neighbours | R1 (QNNVerifier, QA-IBP) | **literature, fixable** |
| Craft: redundancy, notation, figure | R1, R2, R3 | **writing, fixable** |

No reviewer challenges the correctness of the contracts, the composition argument,
or the equivalence of harness and backend kernels. That is the strongest possible
starting position for a revision: the disputed claims are all *ours to measure*, and
none of the measurements can turn out in a way that invalidates the method — at
worst they show block decomposition helps less than we implied, which is a
publishable finding stated honestly.

The strategic risk is the opposite one: over-claiming under pressure. The expert
reviewer rejected partly *because* the abstract asserts a benefit the results
section retracts. The fix is never to strengthen the abstract; it is to measure the
benefit and let the abstract report what was measured.

**Standing constraint for this revision: no obligation is weakened, no tolerance is
widened, and no `VERIFIED` is obtained by relaxing a check, in order to improve a
number for the rebuttal.** If an arm produces worse numbers than the current paper,
the worse numbers go in.

---

## 1. The scalability claim (RQ3) — the load-bearing repair

### The problem, stated precisely
The paper makes an *equivalence* argument (Eq. 14: block-wise verification is
logically equivalent to full-layer verification) and then silently borrows a
*performance* claim from it ("to reduce query size"). Equation 14 justifies
soundness of the decomposition. It says nothing about cost. The cost claim needs
its own evidence, and currently has none.

### The path
1. **Separate the two claims in the text first.** §4.5 proves equivalence; that
   proof stands and needs no experiment. The abstract and contribution 3 should
   claim only what §5 measures. Doing this edit *before* the experiments run means
   the paper is already consistent even if the campaign is truncated.
2. **Design the sweep as a monotone series, not an A/B.** Two points (β = 0 vs
   β = 2) can be explained away as noise or as one lucky configuration. A sweep over
   β ∈ {0, 1, 2, 5, 10} against per-query peak RSS and per-query wall time exposes
   the shape of the relationship, and the shape is the scientific claim: query cost
   grows superlinearly in block width, so decomposition trades a linear increase in
   query *count* for a superlinear decrease in query *size*.
3. **Report the right dependent variables.** Total wall time is the wrong headline:
   decomposition increases the number of solver invocations, so total time may well
   *rise* while the thing that matters — the largest query the solver must survive —
   falls. Report `max_esbmc_query_peak_memory_mib` and
   `max_esbmc_query_time_seconds` as primary, total time as secondary, and say
   explicitly that the mechanism is feasibility, not speed. This reframing is honest
   and it converts a possible negative result into the correct claim.
4. **Expect and welcome β = 0 failures.** If monolithic MNIST memouts at 20 GiB
   while β = 2 completes, that is the strongest possible form of the result: not
   "x % faster" but "the monolithic query does not terminate within the budget and
   the decomposed one does". Record memouts as data, not as failed runs to be retried.

### What would falsify our claim
If β = 0 completes on MNIST within budget and at comparable memory, block-wise
decomposition buys nothing at these network sizes. Then the correct paper says so,
retains Eq. 14 as an enabling result for *larger* networks, and moves the
contribution's weight onto the deployment-semantics argument — which is what R3 and
R1 already say is the paper's real strength. Plan for this outcome; it is not a
disaster, it is a smaller paper.

---

## 2. The 10 % certification rate — reframe, then improve

### The problem, stated precisely
"1 of 10" invites the reading that the tool works one time in ten. That reading is
wrong in a way we failed to make legible: the ten regions are a *nested sequence of
increasing radii on a single sample*, so the correct statement is that the method
certifies up to a radius and then stops — a monotone frontier, not a 10 % hit rate.
We reported a success *fraction* over a set that was never a random sample of
anything.

### The path
1. **Replace the fraction with a frontier.** For each architecture, report the
   largest ε certified and the status of the next radius up. A frontier is the
   natural summary of a nested family and it is not a rate, so it cannot be misread
   as one. This single change answers most of R1's and R3's objection.
2. **Publish the failure taxonomy.** Every inconclusive outcome already carries a
   machine-readable reason. Three causes with completely different implications are
   currently averaged into one number:
   - `SOURCE_PROPERTY_INCONCLUSIVE` — DeepPoly could not establish the property on
     the *float* network. Nothing about our method is implicated; the region is out
     of scope by construction.
   - `PREIMAGE_DEFLATION_EMPTY` / `MARGIN_INCONCLUSIVE` — precision of the backward
     preimage under the error budget. This is *our* limitation and should be owned.
   - `TIMEOUT` / `MEMOUT` — solver cost. Bounded by the budget, addressable by
     T11's sensitivity curve.
   Presenting these separately turns a weak aggregate into an informative diagnosis,
   which is what a dependability audience actually wants from an inconclusive result.
3. **Add the timeout-sensitivity curve.** Cumulative certifications as a function of
   the wall-clock budget, over the unresolved regions. If the curve is flat between
   300 s and 1800 s, we have shown that the limit is structural rather than a
   badly-chosen constant — a far better answer to R1 than raising the timeout and
   hoping.
4. **Widen the denominator honestly.** The reported campaign in this repository
   spans more regions than the ten in the paper. Report the full campaign with per
   dataset and per architecture denominators. Do not cherry-pick the subset with the
   best ratio; state all of it, including Seeds, where the method currently
   certifies nothing. An expert reviewer trusts a paper that reports its own zeros.

### What must not happen
Improving the ratio by loosening the error budget, disabling chaining enforcement,
or falling back to the heuristic tolerance. Those raise the count and destroy the
certificate. If the ratio does not improve, the ratio does not improve.

---

## 3. Positioning against QNNVerifier — a substantive argument, not a citation

R1's sharpest technical point. QNNVerifier verifies fixed-point neural network
implementations with ESBMC; a reader who knows it will ask what is left for us.
The answer is genuine, but it must be made in terms of *what is quantified over*,
not in terms of tooling:

- **Direction of the problem.** QNNVerifier is an *analysis*: given a network and a
  precision, check the property. PreQ-BMC is a *synthesis*: search the space of
  per-layer formats for one that discharges the property. Precision is an output,
  not an input.
- **Structure of the query.** QNNVerifier poses the property over the whole network.
  PreQ-BMC decomposes it into per-layer contracts with an explicit composition
  obligation (Eq. 9) and an explicit output obligation (Eq. 18). This is the
  assume-guarantee move, and it is why we can carry an error budget across layers
  at all.
- **Preimage guidance.** The contracts are not guessed; they are MILP-derived
  backward preimages, deflated by a soundness budget. Nothing in QNNVerifier plays
  this role.
- **Deployment artifact.** The verified harness and the exported backend instantiate
  the same kernel, so the certificate is about code someone can ship.

The right rhetorical form is a row in Table 1 plus two sentences in §3 — not a
defensive paragraph. The comparison is favourable; state it flatly.

QA-IBP belongs in a different place: it is *certified training*, an orthogonal and
complementary line (make the network robust by construction, rather than certify a
trained one). One sentence positioning it as complementary is sufficient and shows
command of the area.

---

## 4. The missing theorem — the item no reviewer asked for and every FM reader wants

The paper presents a chain of obligations and asserts, in prose, that discharging
them establishes end-to-end robustness. At a formal-methods venue, that assertion
should be a numbered theorem with a proof, and the error budget (Eq. 10) should be a
lemma. Adding them is the difference between "we built a tool" and "we established a
result", and it is the surest way to move an expert reviewer from Weak Reject.

The path:
1. State the trust base explicitly and early (§4.9 half-does this). Four components
   with strictly separated roles: DeepPoly *gates*, MILP *proposes*, ESBMC
   *certifies*, empirical evaluation *never certifies*. Nothing outside ESBMC's
   output contributes to a certificate. Say what is assumed: ESBMC's C semantics,
   the solver, and the source-level (not compiled) correspondence.
2. State Theorem 1 (deployed transfer) and prove it by induction over layers, with
   the composition obligation as the inductive step and the output obligation as the
   base of the classification claim.
3. State Lemma 1 (budget soundness) and **re-derive Eq. (10) before writing it
   down**. As printed, δ is indexed by output neuron `i` while the first
   right-hand-side term is independent of `i`, and the scale in that term is the
   *input* scale `S_{l-1}` where a weight-quantization scale is expected. One of the
   indexing, the scale, or both is a transcription error. The implementation is the
   ground truth here: derive the lemma from what
   [`src/synthesis/preqbmc.py`](../../src/synthesis/preqbmc.py) actually computes,
   then make the paper match the code. If they disagree, that is a finding about the
   code, and it takes priority over the paper.
4. If space is tight, the proofs can go to an appendix or the artifact — but the
   *statements* must be in the body.

---

## 5. The fidelity caveat — close it instead of confessing it

§4.4 currently ends the central equivalence argument with "provided that all
`__int128` intermediate values are representable". That is an undischarged
hypothesis sitting inside the paper's main technical claim, and R3 specifically
praised the equivalence argument — so this is the sentence most likely to be
re-read critically next round.

The repository can discharge it: the no-saturation harness family proves the
pre-clamp accumulator stays in range, and the run configuration can *require* it
rather than record it as optional evidence. The path is to enable it for the
reported campaign, report it as one of the discharged obligations, and rewrite the
sentence from a caveat into a verified side condition. Cost is near zero; the gain
is that the fidelity argument becomes closed under its own assumptions.

---

## 6. Craft: what "lack of proof-reading" actually costs

R1's parenthetical is the tell. An expert who believes the authors did not re-read
the paper discounts everything else in it, including the parts they liked. The
craft items (introduction redundancy, notation, figure, undefined table columns,
inconsistent obligation counts) are individually trivial and collectively decisive.
Treat them as P0 rather than polish, and have someone who has not read the paper
recently do a full pass before submission.

One craft defect is not merely cosmetic and should be fixed carefully:
B_ε(x₀) is defined as a **clipped** box in §2.3 (intersected with the valid input
domain) and as an **unclipped** ℓ∞ ball in Eq. (5). These denote different sets, the
implementation uses the clipped one, and the property verified is therefore the one
in §2.3. A reader checking the formalism will find this, and it looks like
imprecision about the very thing the paper claims to be precise about.

---

## 7. Scope discipline

Two reviewer suggestions should be *declined*, in writing, with reasons:

- **Conv2D support (R3).** Not a revision item. DeepPoly, the MILP encoding, the
  harness renderers, and the C generator all assume dense layers. Promising it
  vaguely invites the next reviewer to check. Instead state the extension path
  concretely — im2col reduces convolution to the existing affine machinery, and
  per-channel scales generalise the format from per-layer to per-block — so the
  reader sees that we know what it would take.
- **CEGAR at contract level (R3).** A research contribution, not a fix. Cite it as
  the leading future direction; it is also the honest answer to the preimage-precision
  failures identified in §2 above, which makes §6 read as a diagnosis rather than a
  wish list.

Declining these with a stated reason is stronger than half-attempting them. The
conclusion should read as though the failure profile *generated* the future work,
because it did.

---

## 8. Decision points that need the authors, not this document

1. **Venue.** Is this a revision for LADC (if the process allows one) or a
   resubmission elsewhere? A resubmission permits restructuring §5 around the full
   campaign rather than the ten-region slice, which is the better paper. A
   camera-ready revision constrains us to edits plus the T1/T2 arms.
2. **Denominator.** Which campaign is *the* reported campaign? The paper's ten
   regions and the repository's broader reported set answer RQ2 differently. Pick
   one, re-run it end to end under one configuration, and report only that. Mixed
   provenance across tables is how the "19 vs 21 obligations" discrepancy arose.
3. **Compute budget.** T1 + T2 + T11 is roughly 3–5 machine-days on the evaluation
   host. If that is not available, cut T11 first (it is the least-cited), then the
   β ∈ {1, 5, 10} sweep points, never the β = 0 arm.
