# Full-text read of the submitted PDF

A pass over the submitted PDF as a *text*: equations, symbols, internal consistency,
numbers, references, prose. Complements
[04_NOTATION_AND_TEXT_FIXES.md](04_NOTATION_AND_TEXT_FIXES.md), which is organised by
reviewer thread; this one is organised by defect severity and is keyed to the PDF's
own equation numbers. Items marked **[substantive]** change a claim's meaning or
truth value; the rest are consistency and craft.

Three findings were checked against the implementation and the result artefacts
rather than argued from the text alone: F1 (`src/backends/fixed_point.py`), F6
(`output/cross_tool_comparison/results.tex`), F19 (`src/synthesis/preqbmc.py`).

---

## 1. Findings that change a claim

### F1 — `N = I + F` is arithmetically false in this paper **[substantive]**

§2.2 states `⟨N, I, F⟩` with `N = I + F` and "`I` includes the sign and integer bits".
Table 2 and §5.3/§6 report the synthesised format as `⟨13, 4, 8⟩`, and `4 + 8 = 12 ≠ 13`.

The implementation settles it. `src/backends/fixed_point.py:24-25`:

> `integer_bits` excludes sign bit. signed Q invariant is:
> `total_bits == integer_bits + fractional_bits + 1`.

and `__post_init__` raises unless that holds. So the convention is `N = 1 + I + F`
with `I` counting integer bits only. §2.2 is wrong on both halves of the sentence,
and the paper's headline format violates the paper's own definition — a reviewer who
does the addition finds this in ten seconds.

**Fix.** "`I` is the number of integer bits and `F` the number of fractional bits;
with one sign bit, `N = I + F + 1`." Then `⟨13, 4, 8⟩` checks out, and the
59.4 % memory figure (`1 − 13/32 = 59.4 %`) is consistent with it.

### F2 — Eq. (4) states something the paper elsewhere denies **[substantive]**

As printed, Eq. (4) reads

```
N̂^{1:2[l]}_{2[l]}(I) ⊆ P^{2[l]}  ⟹  N̂ ⊨ ⟨I, O⟩
```

for a single `l`, and the preceding prose says "By confirming this property at every
layer, we establish that the entire `N̂` satisfies the global robustness property."

§4.7 opens with: "The last hidden-layer contract alone does not establish
classification." §4.2 adds the composition obligation (9). So the paper's own
methodology says containment at every layer is *not* sufficient — one also needs
(9) for every `l`, the base case, and the output proof (17)/(18). Eq. (4) is the
first formal statement a reviewer meets and it overstates the method.

Two further defects in the same paragraph:

- `P^{2[l]} ⊆ N^{1:2[l]}_{2[l]}(O)` applies the network *forward* to an output
  region. A preimage is an inverse image: it needs `(N^{...})^{-1}(O)`. As printed
  the containment is backwards and type-incorrect.
- The quantifier `∀l` is missing, and `⟨I, O⟩` uses `I` for the input region while
  §2.2 uses `I` for the integer-bit count (see F11).

**Fix.** State the chain as a numbered implication with all four premises — base case,
per-layer contract (8), composition (9), output obligation (18) — and let Lemma 1
(T12) discharge it. This is also the natural place for the base case, F12.

### F3 — Two different perturbation sets, one called `B_ε` **[substantive]**

§2.3: `B_ε(x) = Π_j [max(ℓ_j, x_j − ε), min(u_j, x_j + ε)]` — clipped to the valid
domain. §4.1 Eq. (5): `B_ε(x₀) = {x : ‖x − x₀‖_∞ ≤ ε}` — unclipped. These differ
exactly when the ball leaves `[0,1]^d`, which for `ε = 4` in raw pixel units on
MNIST is most sampled pixels. Eq. (5) is the property statement of the paper, and
the implementation clips. (Already A.3 in doc 04; repeated here because it is the
single most consequential symbol collision in the paper.)

### F4 — "optimal bit-width" is claimed but never defined or computed **[substantive]**

§4 opening: "which aims to find the optimal bit-width for each layer of a QNN".
Algorithm 1 lines 7–19 iterate candidate formats and `accept the format and proceed
to the next layer` at the first one that discharges its obligations — first-accept
greedy, per layer, in whatever order `Q` is enumerated. No objective function is
defined anywhere in the paper, so "optimal" has no referent, and even with one the
procedure would not attain it.

**Fix.** "smallest admissible width from a user-supplied candidate list, accepted
greedily per layer" — and if the candidate list is ascending, say so, because that
is what makes the result locally minimal.

### F5 — The paper says it uses an SMT theory it does not use **[substantive]**

§4 opening: "while ensuring that the defined bit-widths satisfy robustness properties
using the SMT Theory of Fixed-Point Arithmetic" — a pointer to Baranowski et al. [2].
§5.1: "We use ESBMC 7.11.0 with Z3 in bit-vector mode". §4.4 confirms the harnesses
are integer C with `__int128` products and explicit RHAZ division. The fixed-point
theory of [2] is discussed in §3 as related work and is not the encoding used.

**Fix.** Delete the clause, or replace with "over the bit-vector encoding of the
generated integer C". §3's positioning against [2] is correct and should stay.

### F6 — The 94.00 % accuracy is a 100-probe figure reported as full-test **[substantive]**

Table 2's caption says "Accuracy is measured on the full test set" and lists 94.00
for PreQ-BMC and 80.23 for Quadapter. §5.3 then writes "obtained 94.00 % full-test
accuracy on the generated C backend, whereas Quadapter's accuracy fell from 90.26 %
to 80.23 %", inviting a 13.8-point comparison.

The project's own audit, `output/cross_tool_comparison/results.tex:400-411`, states
the opposite:

> The float32 reference of 90.26 % and the quantized Keras accuracy of 90.30 % are
> full-test-set figures; the 94.00 % shared by quantized Keras, the Python
> interpreter, and the generated C program is measured on the 100-sample probe set
> fixed by `compare_limit`. […] The 94.00 % figure must not be differenced against
> the 90.26 % baseline […] The full-test accuracy of the deployed C program was
> never measured.

The submitted PDF does precisely what the audit forbids, in the abstract
("94.00 % test accuracy on the generated C backend"), in Table 2, in §5.2 RQ4, in
§5.3 and in §6. Four occurrences.

**Fix (audit's own wording).** Full test set: float32 90.26 % → quantized Keras
90.30 %. Probe set (100 samples): quantized Keras, Python fixed-point and generated
C all 94.00 %, i.e. the three backends are indistinguishable. Quadapter's 80.23 % is
full-test and comparable to 90.30 %, not to 94.00 %. Split Table 2's accuracy column
or footnote the evaluation set per row. This is the finding most likely to be caught
by a reviewer who opens the artefact, and it is the one that costs the most.

### F7 — §6 asserts mismatches that RQ1 reports as zero **[substantive]**

§6: "The results also confirm that accuracy alone does not characterize deployment
fidelity; prediction mismatches and logit deviations expose numerical effects
invisible at the model level."

RQ1 (§5.2 and §5.3): "zero classification mismatch", "zero observed saturation",
maximum absolute logit deviation `9.01 × 10⁻³` *versus quantized Keras*.

No prediction mismatch was observed. The sentence also silently conflates two
different comparisons: C vs. the Python fixed-point interpreter (exact, bit-for-bit)
and C vs. quantized Keras (float reference, `9.01 × 10⁻³`). Only the second can
deviate at all.

**Fix.** "Bit-exact agreement between the generated C and the fixed-point reference,
with a residual `9.01 × 10⁻³` logit gap to the float-trained Keras model, shows that
the deployment gap lives between the float and fixed-point semantics, not in code
generation." That is both true and a better point.

### F8 — §2.4 gives BMC a two-valued semantics the rest of the paper rejects **[substantive]**

§2.4: "If the generated constraints are satisfiable, the solver returns a
counterexample […]; otherwise, the property holds within the specified bound."

§4.9 and §5.1: "resource-bounded solver outcomes (Timeout, Memout, Unknown) are
reported as inconclusive, never as refutations." The background section teaches the
reader a dichotomy that the contribution spends a subsection dismantling — and a
skimming reviewer will read §5's nine inconclusive regions through §2.4's lens.

**Fix.** "if the constraints are unsatisfiable the property holds within the bound;
if the solver exhausts its resources the outcome is inconclusive." One clause, and
it sets up §4.8 for free.

### F9 — §4.4 describes two incompatible harnesses in adjacent paragraphs **[substantive]**

¶1: "constrains nondeterministic inputs to the layer assumption and asserts the
corresponding guarantee. ESBMC therefore checks the satisfiability of (13)."
¶2: "the harness computes sign-aware lower and upper accumulator bounds over the
complete input box using interval arithmetic […] verifying that the resulting
interval lies within the contract interval establishes the layer obligation."

These are different programs. The first quantifies over a symbolic input vector; the
second evaluates two deterministic endpoint accumulators and has no quantified input
at all. Both exist in `src/verification/c_templates.py` — `nondet_longlong()` +
`__ESBMC_assume` at lines 164, 1020, 1235, and the endpoint form documented at
line 218 as avoiding "nondeterministic hidden vectors". The paper never says which
obligation uses which.

This matters for soundness, not just clarity: the endpoint argument is valid only
because the affine map attains its extrema at box vertices coordinatewise and RHAZ,
bias, clamp, ReLU are monotone. That is a proof obligation the text asserts in one
sentence and it deserves to be stated as such.

**Fix.** A short table: obligation → harness form → why it is exact. Layer contract
(8) → endpoint propagation, exact over a box by monotonicity. Cut validation (16),
output obligation (18), end-to-end tier → nondeterministic input with `__ESBMC_assume`.

### F10 — The abstract claims a scalability result §5 retracts **[substantive]**

Abstract: "decomposes wide layers into equivalent neuron blocks to reduce query size
without weakening the property." RQ3: "the exact time and memory savings of
block-wise verification remain unmeasured in this evaluation." §6 lists the missing
β = 0 arm as the immediate priority. The equivalence half of the claim (Eq. 14) is
proved; the reduction half is not measured. Say only the first until T1 lands.

---

## 2. Symbol collisions and undefined notation

### F11 — Overloaded symbols

| Symbol | Meaning A | Meaning B | Meaning C |
| --- | --- | --- | --- |
| `N` | the ANN (§4) / the QNN `N̂` | total bit-width `N` in `⟨N,I,F⟩` (§2.2), `sat_{N_l}` (6) | — |
| `I` | integer+sign bits (§2.2) | input domain, "an input `x ∈ I`" (§2.3) | property pair `⟨I,O⟩` (4) |
| `Q` | quantization map `Q(A,⟨N,I,F⟩)` (3) | set of candidate widths (Alg. 1, Require) | — |
| `B` | perturbation ball `B_ε(x)` (§2.3, 5) | neuron blocks `B_1,…,B_k` (§4.5) | — |
| `C` | cut constants `C_k^-`, `C_k^+` (15) | the conjunction of validated cuts `C` (18) | — |
| `f` | layer map `f^{[l]}` (§4) | logit function `f_c`, `f_k` (§2.3) | — |
| `⟨·⟩` | fixed-point format (3) | layer contract (7) | property pair (4) |
| `δ` | per-neuron scalar `δ_i^{[l]}` (10) | vector `δ` (12) | scalars `δ_t`, `δ_j` (17) |
| `S` | scale `S_l = 2^{F_l}` (§4.3) | — | — |

`fp` deserves its own line. Eq. (5) writes `N̂_fp` for the *fixed-point* network,
while §2.2 and §1 use "floating-point" so often that every reader will expand `fp`
that way. Use `N̂_fxp`, `N̂_Q`, or just `N̂`.

### F12 — Symbols used before or without definition

- `σ^{[l]}` in the prose under Eq. (2) vs. `ρ_l` in Eq. (2) itself. One activation,
  two symbols, first equation of the paper.
- `f_c`, `f_k` (§2.3): never introduced as logit components.
- `N̂` (§2.3): defined two sections later.
- `x ∈ I` (§2.3): the domain was just named `X`.
- `A^{[l]}_j` (§4): the `j` subscript appears once and never again.
- `P_i` then `P^{2[l]}` (§4): two names for the preimage, one sentence apart.
- `M_j` (11): the underline/overline interval-endpoint notation is used only here and
  is never defined; nor are its *units* — see F19.
- `m_l` (14): number of output neurons, never stated.
- `D_k`, `ĥ` (15): direction vector and hidden state, never introduced.
- `X̂` (16): the quantized input region, never introduced; collides with `X` from §2.3.
- `ŷ_t`, `ŷ_j` (18): a third notation for logits, after `f_c` (§2.3) and
  `N̂_fp(q(x))_t` (5).
- `G_t^-`, `G_j^+` (17): the `±` endpoint convention was `p^-`, `p^+` in (12); and
  the superscript `[L]` is dropped.
- `β` (§5.2 RQ3, §5.3, §6): the block size. §4.5 introduces blocks as `B_1,…,B_k`
  and never names their size. `β = 2` and `β = 0` appear five times in §5–§6 with no
  definition.
- Target class: `c` in §2.3, `t` from §4.1 onward.
- Figure 1 uses `bit-jump`, `no-sat`, `witness replays Python & .so?` — none of the
  three appears anywhere in the body text.

### F13 — Sets used as predicates

Eqs. (8), (9), (12) treat `A^{[l]}`, `G^{[l]}_pre`, `P`, `T` as sets (`∈`, `⊆`, `⊕`).
Eqs. (13) and (14) treat them as predicates: `A^{[l]}(â) ∧ ¬G^{[l]}_pre(ẑ^{[l]}(â))`,
`⋀ G_i(x)`. Pick one. If the predicate form is wanted for the SMT encoding, write
`â ∈ A^{[l]} ∧ ẑ^{[l]}(â) ∉ G^{[l]}_pre` and say that this is the formula handed to
the solver.

Related: Eq. (8) writes `∀â^{[l−1]} ∈ A^{[l]}, ẑ^{[l]} ∈ G^{[l]}_pre` with `ẑ^{[l]}`
free, while Eq. (13) correctly writes `ẑ^{[l]}(â)`. Make (8) match (13).

---

## 3. Equation-by-equation

**(1)–(2).** `ρ_l` vs `σ^{[l]}`; `z^{[l]}` and `a^{[l]}` are used before being named
as pre-activation and activation.

**(3).** The PDF renders `⌊2^F A⌋` (floor) while the following sentence defines
`⌊·⌉` (round-to-nearest, ties away from zero). Check the macro survives the build.
Also: `I` appears in the argument `⟨N,I,F⟩` but nowhere on the right-hand side —
the map depends only on `N` and `F`. Note that `I` is determined by `N` and `F`
(F1), or the reader will hunt for the missing dependence. And `clip` here is the
same operation as `sat_{N_l}` in (6); two names, one operation.

**(4).** See F2.

**(5).** See F3 and the `fp` subscript in F11. The quantifier ranges over a real ball
while the harness quantifies over an integer box; the step
`q(B_ε(x₀)) = A^{[1]}` is exact because `q` is monotone coordinatewise, but the paper
never states it. This is the missing base case of the assume–guarantee chain
(F2/F12): §4.2 gives (8) and (9) for `l` and `l+1` and never anchors `l = 1`.

**(6).** The formats of `Ŵ^{[l]}`, `b̂^{[l]}` and `â^{[l−1]}` are unstated, and the
equation only type-checks under a specific choice: `Ŵ` and `b̂` at scale `2^{F_l}`,
`â^{[l−1]}` at `2^{F_{l−1}}`, so the product carries `2^{F_l + F_{l−1}}` and the
`RHAZDiv(·, 2^{F_{l−1}})` restores `2^{F_l}` before the bias is added. State this;
it is one sentence and it makes the rescaling term of (10) legible.

`sat_{N_l}` collides with the network symbol `N` (F11).

**(8).** See F13.

**(10)–(11).** The first term `⌈Σ_j M_j / (2 S_{l−1})⌉` has no `i` index although the
left-hand side does, and it uses the *input* scale `S_{l−1}` for a quantity described
as weight-quantization error. Both look like errors and neither is — see F19. The
missing ingredient is that `M_j` is measured in **input integer ULPs**, which the
paper never says. `δ` is per-neuron here, a vector in (12), and scalar in (17), with
no stated convention.

**(12).** Correct, and the `T ⊕ [−δ,δ] ⊆ P` remark is the right thing to say.

**(13).** See F13; and "A Verified result establishes that this formula is
unsatisfiable" is worth one more clause: *and therefore that (8) holds for every
integer input in `A^{[l]}`*.

**(14).** Correct as a logical identity. `m_l` undefined; `G_i` is a predicate here
and a set elsewhere; `B_r` collides with `B_ε`.

**(15)–(16).** `D_k`, `ĥ`, `X̂` undefined. §4.6 says PreQ-BMC "may introduce"
directional cuts without any criterion for when, how many, or how the directions are
chosen — yet `src/verification/c_templates.py:216` records the actual rule ("Every
cut direction is exactly the corresponding quantized weight row"), which is both
specific and easy to justify. Put it in the paper.

**(17).** `G_t^-`, `G_j^+` are undefined and the `±` convention differs from `p^±`.
More importantly: it is not said which network's outputs `G^±` bounds. If `G^{[L]}`
is a contract on the *deployed integer* logits — which is what ESBMC discharges —
then the implementation error is already inside it and the `±δ` correction is
unmotivated. If it is a float-level bound, say so. As written the reader cannot tell
whether the test is conservative or double-counting.

**(18).** `ŷ` undefined (third logit notation); `A^{[L]}` is the input box of the
output layer, which is fine but worth saying since every other `A^{[l]}` was
introduced as an assumption on the *previous* layer's activations.

---

## 4. Numbers, tables, experimental reporting

**F14 — Calls 21 vs. 19 obligations.** Table 2 reports 21, §6 reports 19. The audit
resolves it: `results.tex:500` — "21 ESBMC calls — 19 obligations and 2 vacuity
sentinels". Put that in the caption and both numbers become informative.

**F15 — CEG4N execution count.** §5.3 says "all six completed MNIST executions ended
in timeout, with median wall times of 1209.06 s for `1blk_10` and 1209.53 s for
`1blk_25`". The audit says seven — "four `1blk_10` and three `1blk_25` … median wall
times of 1209.12 s and 1209.53 s". Reconcile: 4 + 3 = 7, and the PDF's own split is
missing. Also "all six completed … ended in timeout" reads as a contradiction;
"terminated at the timeout" is what is meant.

**F16 — Unmatched cohorts and unequal budgets.** PreQ-BMC runs under a 300 s
per-query timeout, CEG4N under a 1200 s whole-run limit; the paper reports both
without noting that the budgets are neither equal nor comparable in kind (per query
vs. per run). The audit further records that "one `1blk_10` sample present in the
Quadapter campaign is absent from the CEG4N results, so the MNIST cohorts are not
sample-identical across baselines", and that CEG4N's MNIST accuracy field is on a
different scale (float32 baseline reads 28.84 against 90.26 measured by the other two
tools on the same weights). §5.1 promises "differences in outcome are attributable to
the tools rather than to the models they were given" — that promise needs these three
caveats next to it or it is not kept.

**F17 — "one valid Iris execution".** §5.3 leaves "valid" unexplained. The audit:
nine Iris rows failed with a missing `ONNX2C_PATH` and are excluded as infrastructure
failures rather than `Unknown` verdicts. Say that; excluding a competitor's runs
without stating the reason is the kind of thing a reviewer will read uncharitably.

**F18 — Setup gaps.** §5.1 says the 20 GiB cap was "chosen to bind well below machine
RAM" without ever stating the machine RAM. It says "we report Python, compiler, and
solver versions for all tools" — only ESBMC 7.11.0 and "CBC" appear anywhere in the
paper. It says "provided in the artifact" with no DOI, URL or footnote. Also missing:
core count, OS, whether `Wall (s)` is single-run or median (Table 2 says one region,
not repeated), and the stage attribution of the 415.77 s (the audit has it:
410.55 s inside formal queries, longest query 53.00 s).

**F19 — Eq. (10) is right; its description is what is wrong.**
`src/synthesis/preqbmc.py:2484-2515` derives it: a real weight rounded to `W_int/S_out`
carries at most `0.5/S_out` real error; multiplying by `|A_j|/S_in` and converting back
to output ULPs (`× S_out`) gives `0.5·|A_j|/S_in` output ULPs per input, so the term
is `⌈Σ_j |A_j| / (2 S_in)⌉` — no `S_out`, no row index, because `S_out` cancels and the
bound uses only the assumption box. `A_j` there is `max(|assumed_lo_int|,
|assumed_hi_int|)`, i.e. the *integer* activation bound, which is the paper's `M_j`.

So the fix is not to re-derive but to say two things the paper omits: that `M_j` is in
input integer ULPs, and that `S_out` cancels — after which the absent `i` index and
the `S_{l−1}` denominator both read as consequences rather than typos. Worth adding
in the same breath: the "+1" is exactly the RHAZ half-ULP plus the bias-quantization
half-ULP, and the third term uses the *stored float* weights, not `|W_int|`, because
`|W_int|` alone is not conservative when a weight rounds toward zero (the code says
so at line 2505 and uses the `2 S_out |W_real| ≤ 2|W_int| + 1` fallback otherwise).
That last detail is a genuine soundness subtlety and it currently appears nowhere in
the paper.

---

## 5. Prose, grammar, references

**Grammar and wording**

- §4: "We call the original ANN as `N`" → "We denote the original ANN by `N`".
- §4: "we use the conservative abstract interpretation (DeepPoly) [23] abstract
  interpretation framework" — "abstract interpretation" twice in one noun phrase.
- §4: "Specifically, for each affine layer `f^{2[l]}`." — sentence fragment.
- §2.3: "small perturbations measured with respect to a chosen norm, and often
  visually imperceptible for image inputs may cause misclassification" — missing the
  closing comma of the parenthetical clause; the sentence has no readable subject.
- §2.3: the definition ends with a comma before "Thus".
- §2.3: the sentence says "`N̂` is locally robust" but the condition it states is on
  `f`. The subject and the formula are about different networks.
- §2.4: "For QNN, SMT-based BMC models…" → "For QNNs".
- §5.3: "This is smaller and substantially faster than PreQ-BMC" — "this" is a
  memory-reduction percentage; a percentage is not faster than a tool. Name the
  subject: "Quadapter's format is more compact and its run substantially faster".
- §6: "Our work presented a deployment-aware methodology" → "We present".
- §6: "19 obligations discharged without a single resource outcome" — "resource
  outcome" is internal vocabulary; "without a timeout, memout or unknown verdict".
- `counter-example` (§4 opening) vs `counterexample` (§4.8 and everywhere else);
  Figure 1 has `COUNTER-EXAMPLE FOUND`.
- `ε` (§2.3, §4.1) vs `ϵ` (Table 2, §5.2, §5.3). Also "radii 0.5, 1, 2, and 4" with
  no units, where the units are raw pixels and matter.
- "300 s timeout" vs "1200-second verifier limit" — one unit style.
- §5.1: "The three datasets (benchmarks) were chosen…" — drop the parenthetical.
- §5.1: "MNIST is therefore the only dataset on which all three tools have published
  results, or executable configurations" — the comma inverts the meaning.

**Structure**

- §3's closing paragraph restates the §1 contribution list nearly verbatim. Cut one.
- §1 ¶1–¶4 state two propositions twice each (documented in doc 04); that redundancy
  is where the space for Lemma 1 and the base case comes from.
- Table 1 lacks the column that is the paper's actual differentiator — "verifies the
  executed integer code?" — and the analysis-level entry "Empirical network" for [13]
  is not a level.
- Giacobbe et al. [10] is cited twice in §1–§2 for exactly this problem (how many bits
  does it take to quantize your network) and never appears in §3 or Table 1.
- Figure 1's caption, "Proposed Methodology Workflow", tells the reader nothing. Say
  what to take from it.

**References** (a reviewer already wrote "lack of proof-reading"; this is where it shows)

- [5] "Cordeiro, L., et al.: Advanced automated verification and software analysis.
  In: International Symposium on Programming (ESOP) (2025)" — `et al.` is not
  permitted in an LNCS reference list, ESOP is the *European* Symposium on
  Programming, and this entry is used for ESBMC in §4 while [8] (Gadelha et al.,
  ESBMC v6.0) is used for ESBMC in §2.4. One tool, two citations, one of them
  malformed.
- §2: "specialized verification frameworks such as Reluplex and Marabou [15]" — [15]
  is the Marabou paper. Reluplex (Katz et al., CAV 2017) is uncited.
- Sentence-cased acronyms throughout: "An smt theory" [2], "Challenging smt solvers"
  [21], "ansi-c" [4], "Qnnrepair" [24], "res-se-cnn" [16], "parkinson's" [17],
  "huffman coding" [12]. Fix the BibTeX with braces.
- Broken URLs: [24] ends at "https://doi.org"; [26] ends at "https://doi.org_18".
- Month artefacts from the reference manager: "(01 2004)" [4], "(07 2010)" [20],
  "(07 2019)" [15], "(01 2025)" [17], "(02 2025)" [16].
- [4] is TACAS 2004; it is formatted as a journal article in LNCS vol. 2988.
- [1]: "Ald, R.H." looks like a truncated surname; "Huzaifa Shah, M." is formatted
  unlike its neighbours.
- [11] "MIT press" → "MIT Press".
- [25] Szegedy et al. has an ICLR 2014 version; cite it rather than the arXiv entry.

---

## 6. What to fix first

1. **F6** (accuracy evaluation set) — the artefact contradicts the paper; highest cost.
2. **F1** (`N = I + F`) — a two-second arithmetic check that fails.
3. **F2** (Eq. 4) — the paper's central formal claim, overstated and type-incorrect.
4. **F4, F5** (optimality, SMT theory) — two sentences claiming things the system
   does not do.
5. **F7, F8, F10** — internal contradictions between abstract/background and results.
6. **F19** — the one derived bound; explaining its units converts an apparent error
   into a strength.
7. Everything in §2 and §5 of this document — mechanical, but this is the category the
   reviewer named.
