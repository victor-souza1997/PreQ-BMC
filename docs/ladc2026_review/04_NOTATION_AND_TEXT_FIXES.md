# Editorial and notation fixes

Line-level list for T4, T8, T9, T13 and T16. Each entry gives the location, the
defect, and the corrected form. Items marked **[substantive]** change meaning and are
not cosmetic.

---

## Abstract

**Defect (R2.1).** Eight compressed technical claims in one paragraph; `ESBMC`,
`MILP`, `BMC` and `PreQ-BMC` all appear before being expanded; the reader cannot tell
what is new relative to Quadapter without reading §3. It also asserts that block
decomposition "reduce[s] query size", a claim §5 retracts (T1).

**Path.** Answer four questions in order, one or two sentences each:

1. *What is the gap?* Robustness certificates are established for real-valued models,
   while the executed artifact is an integer kernel with rounding, saturation, finite
   accumulation and a fixed operation order.
2. *What do we generate?* One fixed-point format per affine layer, plus C code
   implementing it.
3. *What do we verify, and how?* Local robustness of that generated C, bit-precisely,
   with a bounded model checker, via per-layer contracts derived from backward
   preimages and composed by explicit obligations.
4. *What was measured?* The MNIST result, the accuracy and memory numbers, the
   comparison verdicts, and — after T1 — what block decomposition actually bought.

Constraints for the rewrite: expand every abbreviation at first use; state the scope
(dense feed-forward affine/ReLU classifiers) in the abstract itself (T5); claim about
block decomposition only what §5 measures.

---

## Introduction (T4, R1.3)

The redundancy is structural, not stylistic. Paragraphs 1–4 make two propositions and
each is made twice:

| Proposition | First statement | Duplicate |
| --- | --- | --- |
| Embedded deployment is resource-constrained, which motivates compression, of which quantization is the most suitable | ¶1 (last sentence) + ¶2 (first half) | ¶3 (entirely) |
| Quantization is not semantics-preserving: rounding, rescaling, clipping, saturation, overflow, operation order | ¶2 (second half) | ¶4 (entirely) |

**Path.** Collapse to one pass and let each paragraph do distinct work:

1. Domain and stakes — safety-relevant deployment of ANNs.
2. Quantization as the enabling technique, and why it is not semantics-preserving
   (one statement, with the mechanism list given once).
3. Why this is a *dependability* problem specifically: the certificate is established
   for an artifact that is not the one executed, so the gap is a latent fault in the
   safety argument (this is also where T7's LADC framing goes).
4. What verification currently does about it, and the gap: synthesis frameworks reason
   at model level; bit-precise verifiers run after the fact and cannot guide synthesis.
5. Our approach and contributions.

This recovers roughly half a column, which T12's theorem statement needs.

---

## §2.1 — Eqs. (1)–(2)

**A.9.** Eq. (2) writes `a[l] = ρ_l(z[l])`; the following sentence says "σ^[l](·) is
the activation function". Two symbols for one object, in the paper's first equation.
Use `ρ_l` throughout (it is what §4.2 uses in `G_act = ρ_l(G_pre)`).

Also: the sentence defines `W[l]` and `b[l]` and then refers to `σ^[l]` without
having introduced `z[l]` and `a[l]` as pre-activation and activation. Name them.

---

## §2.2 — Fixed-point format

**R2.3.** "where N = I + F, i includes the sign and integer bits" — lowercase `i`
should be `I`. Also, "I includes the sign and integer bits" contradicts `N = I + F`
only if the sign bit is counted separately; state it unambiguously, e.g. *`I` counts
the sign bit together with the integer bits, and `F` the fractional bits, so
`N = I + F`.*

**Rendering.** Eq. (3) must use `⌊·⌉` (round-to-nearest, ties away from zero), not
`⌊·⌋`. The surrounding prose says ties away from zero; make sure the macro used in
the source is `\lfloor \cdot \rceil` and that it survives the build. The current PDF
renders inconsistently.

**Clarify** that `clip` here is *saturation*, the same operation `sat_{Q_l}` denotes in
Eq. (6). Two names for one operation is avoidable.

---

## §2.3 — Robustness **[substantive]**

**A.3.** The clipped ball is defined here as
`B_ε(x) = Π_j [max(ℓ_j, x_j − ε), min(u_j, x_j + ε)]`, while Eq. (5) in §4.1 defines
`B_ε(x₀) = {x : ‖x − x₀‖_∞ ≤ ε}`. These are different sets whenever the ball leaves
the valid input domain, and the implementation clips. Since Eq. (5) is *the property
the paper claims to verify*, this must be the clipped definition. Fix Eq. (5), or
define the clipped set once with a distinct symbol and use it in both places.

**R1.5.** "an input `x ∈ I`" — `I` is undefined here; the domain was just called `X`.
Also `f_c` and `f_k` appear with no prose definition: state that `f` denotes the
network's logit vector and `f_c` its component for class `c`. And `N̂` is used before
it is introduced (it is defined in §4, two sections later).

---

## §4 opening paragraph **[substantive]**

**A.10.** As printed, this paragraph is not readable. It contains:

- `P^{2[l]} ⊆ N^{1:2[l]}_{2[l]}(O)` and `f^{2[l]}` — superscripts that look like
  broken LaTeX rather than intended notation. If the intent is "layer `l` of a
  network with `2l` layers counting activations", say so and pick one indexing
  convention for the whole paper.
- A sentence fragment: "Specifically, for each affine layer `f^{2[l]}`."
- Two names for the preimage: `P_i` in one sentence, `P^{2[l]}` in the next.
- Eq. (4) uses `⟨I, O⟩` without defining the property pair, and `I` collides with the
  integer-bit count `I` from §2.2.

**Path.** Rewrite the paragraph with one indexing convention, define the preimage
symbol once, rename the property pair to avoid the `I` collision (e.g. `⟨X, Φ⟩`), and
verify the built PDF rather than the source.

---

## §4.3 — Eq. (10) **[substantive]**

**A.2.** `δ^{[l]}_i` carries an output-neuron index `i` on the left, but the first
right-hand-side term `⌈Σ_j M_j / (2 S_{l−1})⌉` has no `i` dependence — it is the same
for every output neuron of the layer. For a term described as bounding *weight*
quantization error, one expects the row weights of neuron `i` and the *weight* scale,
not the input scale `S_{l−1}`.

This is the paper's only derived bound and the basis of every deflated target. Before
writing Lemma 1 (T12), re-derive it from what
[src/synthesis/preqbmc.py](../../src/synthesis/preqbmc.py) computes and make the paper
match the implementation. If the implementation is what the formula says, the formula
needs a justification; if it is not, the paper has a typo in its central bound.

Also: `δ` is a vector in Eq. (12) and scalars `δ_t`, `δ_j` in Eq. (17). State the
convention.

---

## §4.4 — the fidelity caveat (T14)

Current: "…provided that all `__int128` intermediate values are representable."
After enabling the required no-saturation obligation, rewrite as a discharged side
condition: the absence of accumulator overflow is itself verified by ESBMC for every
reported configuration, and the equivalence is therefore unconditional at the C-source
level. Keep the compiler-verification disclaimer — that boundary is correctly drawn
and should stay.

---

## §5.1 / Table 2

**R2.4 / A.4.** Define "Calls" in the caption: *number of ESBMC queries executed*. Then
reconcile with §6's "19 obligations discharged" — 19 ≠ 21. Decide what each number
counts (queries executed vs. distinct proof obligations, including or excluding
vacuity sentinels and cut-validation prefixes) and either align them or name them
differently and explain the difference.

**A.6.** Peak memory is `max ESBMC-query RSS` for PreQ-BMC and `process-tree RSS` for
the baselines. Harmonize (see [03_EXPERIMENT_MATRIX.md](03_EXPERIMENT_MATRIX.md#t15--table-stage-attributed-cost-and-harmonized-memory)).

**A.7.** Attribute the 415.77 s by stage. Comparing our end-to-end synthesis +
verification + accuracy evaluation against Quadapter's end-to-end time is defensible,
but only if the reader can see the composition.

**A.5.** Contribution 4 promises five networks across three datasets. §5 gives
PreQ-BMC results on MNIST only. Fix by reporting the Iris/Seeds arms (preferred — the
runs exist) or by narrowing the contribution.

---

## §4.8 / Figure 1 / Algorithm 1 — status vocabulary (T13, A.11)

Three overlapping vocabularies are in use:

| Figure 1 | Algorithm 1 | §5 prose | Implementation |
| --- | --- | --- | --- |
| `VERIFIED`, `VERIFIED (e2e)` | `DeployedTransfer` | "deployed-transfer certificate" | `guarantee_level: deployed-transfer` |
| `MARGIN_REFUTED` | `Failed` | — | `MARGIN_REFUTED` → `failed` |
| `MARGIN_INCONCLUSIVE` | `Unknown` | `Margin-Inconclusive` | `MARGIN_INCONCLUSIVE` → `unknown` |
| — | — | `Layer-Inconclusive` | `LAYER_INCONCLUSIVE` → `unknown` |
| `DEFLATION_EMPTY` | (candidate rejected) | — | `PREIMAGE_DEFLATION_EMPTY` |
| `NOT ELIGIBLE` | `SourcePropertyInconclusive` | — | `SOURCE_PROPERTY_INCONCLUSIVE` |

**Path.** Print the implementation's taxonomy as a small table in §4.8, including the
mapping from run status to guarantee level, then use exactly those names in Figure 1,
Algorithm 1 and §5. The distinction between `harness-verified` and `deployed-transfer`
deserves its own sentence: it is the paper's main honesty mechanism and it is
currently only implicit in the phrase "otherwise, the result is reported only as a
layer-level harness-verification verdict" (§1).

---

## Figure 1 (T10, R3.6)

- Re-render as vector (TikZ preferred, or SVG → PDF).
- Reduce to the three trust tiers — gate (DeepPoly), propose (MILP), certify (ESBMC) —
  with the empirical stage visibly outside the certifying path. That layout also
  illustrates §4.9's trust separation, which currently exists only as prose.
- Unify the status names with the table above.
- The current caption is "Proposed Methodology Workflow"; make it say what the reader
  should take from the figure.

---

## Global pass

- Every symbol defined at first use; consider a notation table if space permits.
- One indexing convention for layers throughout (`l` vs `2[l]`).
- Consistent tool-name capitalization: `PreQ-BMC`, `ESBMC`, `Quadapter`, `CEG4N`,
  `DeepPoly`, `QNNRepair`.
- Verify the *built PDF*, not the source: several of the defects above are rendering
  artifacts that a source-only proof-read would miss.
- Have a reader who has not seen the paper recently do the final pass. R1's
  "demonstrating lack of proof-reading" is the comment that most cheaply erases the
  goodwill the method earns.
