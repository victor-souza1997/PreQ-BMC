# Adversarial methodology audit — PreQ-BMC (LADC 2026 submission)

**To the reviewing model:** you are acting as a hostile but honest program-committee
reviewer with a formal-methods background (think CAV/FMCAD/TACAS). Your job is to
decide whether this artifact proves what it claims, and to say so plainly. I am the
author. I would rather find out now that my methodology is weak than find out from a
reviewer, or publish something unsound.

Read the code before answering. Do not answer from the summary below alone — the
summary is my own reading and may itself be wrong.

---

## 0. Ground rules

1. **Never propose anything that trades soundness for a better-looking result.** If a
   fix requires weakening a contract, loosening a tolerance, or waiving an obligation,
   say so explicitly and reject it.
2. **Distinguish throughout between what the code PROVES, what it MEASURES, and what
   it ASSUMES.** Most of my confusion is probably a category error between these three.
3. **A conclusive negative is a legitimate result.** "Your scalability claim is not
   supported" / "this is sound but the ESBMC usage is decorative" / "this benchmark
   cannot discriminate your ablations" are all acceptable and useful answers.
4. **If a concern of mine is wrong, say it is wrong and show why.** Do not validate me
   to be agreeable. I have listed concerns below that I am not sure about.
5. Cite `file:line` for every claim about the code.

---

## 1. What the artifact claims to be

PreQ-BMC verifies local robustness of **fixed-point quantized** neural networks, where
the object of verification is meant to be the *deployed integer kernel*, not a real-valued
idealisation. The pipeline is:

1. **DeepPoly** abstract interpretation on the float network → establishes the source
   (float) robustness property over an input box.
2. **MILP backward preimage** → for each layer, the region of layer outputs that is
   *permitted* for the property to hold.
3. **ESBMC bounded model checking** → per-layer assume–guarantee contracts
   `{A_{l-1}} C_l {G_l ∧ R_l}`, composed by induction with `A_l = G_l ∩ R_l`.
4. A **guarantee ladder** (`deployed-transfer` / `harness-verified` / `unknown` /
   `failed`) gated on seven transfer preconditions.

Headline claims I believe the paper makes, and which you should attack:

- **(C1) Soundness** — a `VERIFIED` verdict implies the property holds for the deployed
  integer kernel on every point of the input region.
- **(C2) Fixed-point fidelity** — the verified object is bit-exact with deployment
  (`__int128` accumulation, `div_round_half_away_from_zero(acc, 2^F) + bias` → clamp →
  ReLU → clamp).
- **(C3) Scalability** — layer-contract decomposition, with block size β
  (`esbmc_layer_block_size`), makes networks tractable that monolithic verification
  cannot handle.
- **(C4) The components earn their place** — the ablation arms (margin cuts, e2e
  invariants, β) each contribute.

Key entry points: [src/synthesis/preqbmc.py](../../src/synthesis/preqbmc.py) (~6k lines;
`_record_hidden_chaining_check`, `_layer_input_bounds_int`, `_candidate_reachable_bounds`),
[src/verification/c_templates.py](../../src/verification/c_templates.py) (harness
renderers), [src/verification/arith_kernel.py](../../src/verification/arith_kernel.py)
(shared C arithmetic), [src/reports/experiment_summary.py](../../src/reports/experiment_summary.py)
(`_final_status`, `_guarantee_level`, transfer preconditions).
Configs: [experiments/ladc2026_experiment_matrix.json](../../experiments/ladc2026_experiment_matrix.json),
[experiments/ladc2026_mnist_timeout_escalation.json](../../experiments/ladc2026_mnist_timeout_escalation.json).
Results: `output/ladc2026_matrix_runs/` (156 runs), `output/ladc2026_mnist_timeout_escalation_runs/` (4 runs).

---

## 2. Facts I have already measured (verify these, then use them)

These are from the committed outputs. Re-derive them; tell me if any is wrong.

**F1 — No refutation has ever occurred.** Across 7,094 `experiment_summary.json`
records in `output/`: 4,981 VERIFIED, 631 FAILED, 591 PARTIAL_VERIFIED, 243
SOURCE_PROPERTY_INCONCLUSIVE, 180 TIMEOUT, 135 PREIMAGE_DEFLATION_EMPTY, 131 UNKNOWN,
99 MARGIN_INCONCLUSIVE, 77 LAYER_INCONCLUSIVE, 18 PARTITION_INCONCLUSIVE, 5 MEMOUT,
3 PREIMAGE_UNAVAILABLE. **`MARGIN_REFUTED` appears in zero output files**, though the
status exists in code (`experiment_summary.py:307`, `preqbmc.py:1842`). The sampled
`FAILED` runs are tooling/quality failures, not refutations (`counterexamples: false`).

**F2 — The 27 "confirmed counterexamples" are not property refutations.** They are
contract-candidate counterexamples (keys: `layer_index`, `Q`, `F`, `inputs_int`,
`counterexample_neuron`, `counterexample_preclamp`) that reject a candidate (Q,F)
format; those runs end `MARGIN_INCONCLUSIVE`. So the CEX machinery fires only during
format search, never against a property.

**F3 — Every LADC run is `PARTIAL_VERIFIED`, and every one is badged
`deployed-transfer` with `soundness: null`.** All 156 matrix runs + 4 escalation runs.
Zero timeout blocks. The mechanism: the config sets
`require_formal_no_saturation: true`, but the run records
`no_saturation_status: "SKIPPED"`, `no_saturation_verified: false` → `_final_status`
(`experiment_summary.py:144-152`) returns `PARTIAL_VERIFIED`. Separately,
`contract_harness_semantics.no_saturation_required_for_deployed_transfer: false`
makes the precondition `no_saturation_required: false`, so the ladder still awards
the top guarantee level.

**F4 — No layer-contract harness contains any nondeterminism.** Neither the block
harnesses nor the full-layer (β=0) harnesses contain `nondet_longlong()`. They compute
output bounds by deterministic endpoint propagation (`s_lb = mac(s_lb, w, (w>=0)?lo:hi)`)
and then assert containment in the contract box. ESBMC reports
`Generated 114069 VCC(s), 25 remaining after simplification` for a 5-neuron block.
Only the **vacuity sentinels** and **output competitors** use `nondet_longlong()`; the
competitor harness draws `input[k] = nondet_longlong()` constrained by `__ESBMC_assume`
to the *hidden activation* box, and asserts classification.

**F5 — β does not behave like a scalability knob, and does not change the verdict.**
Iris sample 0: β=1 → 34.9s/81 calls; β=2 → 28.6s/46; β=5 → 25.2s/25; β=10 → 24.2s/18;
full layer → 24.5s/18. MNIST 1blk25 sample 3: β=1 → 1099s/43; β=2 → 1135s/31;
β=5 → 1464s/23. Verdict is `PARTIAL_VERIFIED` at every β, in every arm.

**F6 — All three LADC arms produce identical results.** Arm A (β sweep), arm B
(`cuts_off`), arm C (`no_invariants`) are all uniformly `PARTIAL_VERIFIED` /
`deployed-transfer`.

**F7 — End-to-end verification is disabled in LADC.**
`end_to_end_verification: {enabled: false, invariants_injected: true, status: "NOT_RUN"}`.
The only harness family that quantifies over *real network inputs* never runs.

**F8 — `fidelity_by_construction` is a config echo, not a check.**
`experiment_summary.py:268` returns
`bool(semantics.get("uses_shared_deployed_arithmetic_kernel", False))` — the pipeline
asserting a property of itself. (Mitigating evidence: `render_arith_kernel()` is in fact
textually shared across 8 renderer sites in `c_templates.py`, and `python_c_exact_match:
true` is recorded.)

**F9 — The input-space claim rests on float DeepPoly, and fixed-point semantics are
declared.** `verification_claims` records
`source_property_verification: "deeppoly_sufficient_condition"`,
`fixed_point_semantics: "declared_backend_semantics"`,
`accumulator_range: "static_interval_analysis"`,
`blockwise_verification: "equivalent_hidden_contract_decomposition_when_enabled"`.

---

## 3. The questions

### Q1 — Is it sound? Give me the actual proof, or the actual hole.

State the soundness theorem formally and then check it against the code:

- Write the induction precisely. For `A_l = G_l ∩ R_l`, soundness needs
  `A_l ⊇ reachable_l` under assumption `A_{l-1} ⊇ reachable_{l-1}`. Verify that the
  code's `A_l` really over-approximates the reachable set *of the deployed kernel*, not
  of some intermediate abstraction.
- **Enumerate every assumption the theorem rests on that is NOT discharged by ESBMC.**
  My candidates: float DeepPoly soundness (F9), MILP solver correctness (CBC) for the
  preimage, the `derived` error budget's per-neuron bound, the quantized-input model
  (is the input box the set of *representable* deployed inputs, or a superset/subset?),
  DeepPoly's float bounds being soundly quantized by `floor`/`ceil` at
  `preqbmc.py:2865-2866`, and `__int128` non-overflow.
- Is endpoint propagation exact *and* sound here? It is exact for affine-over-box, but
  check the composition order: the C asserts a post-clamp **pre-activation** range
  (`ACTIVATION_KIND 0`), and Python then applies `np.maximum(·, 0)`. Confirm the
  deployed order (clamp → ReLU → clamp) makes that monotone step sound, and that the
  outer clamp is genuinely a no-op.
- `PREIMAGE_DEFLATION_EMPTY` (135 runs): is that a sound "cannot conclude", or can an
  empty deflated target ever be silently treated as a vacuous success?

**Deliverable:** either a proof sketch I can put in the paper with every assumption
named, or a concrete counterexample-to-soundness scenario.

### Q2 — Why has this system never refuted anything, and is that a fatal credibility problem?

This is my biggest worry. 7,094 runs, zero refutations (F1). A verifier that has never
said "false" has never been tested against ground truth.

- Is the absence of counterexamples **expected by construction**? My hypothesis: with a
  box abstraction, a counterexample found in hidden-activation space (F4) need not be
  realizable from any real input, so it can never be *confirmed*, so `MARGIN_REFUTED`
  is effectively unreachable on the layer-scope path. Is that right? If so, say plainly
  that the methodology is **sound but structurally unable to falsify**, and that the
  paper must state this.
- Design the **negative controls** this artifact is missing. Concretely: what is the
  minimal set of experiments that would demonstrate the pipeline *can* report a
  refutation with a replayable concrete input through the deployed `.so`? E.g. a
  deliberately false property (wrong target label), an ε large enough to cross a true
  decision boundary, a mutated weight, a deliberately too-narrow (Q,F).
- Is there a **mutation / metamorphic testing** protocol you would require before
  believing any of the 4,981 VERIFIED verdicts?
- What fraction of the claimed benchmark is *trivially true* (huge clean margin, tiny
  ε)? `clean_margin` and `certified_margin_lower_bound` are recorded — check whether
  the properties are so easy that no method could fail them.

### Q3 — Is β (block size) a legitimate scalability axis, or am I gaming my own benchmark?

My original fear: with β=1 I can decompose any network into trivial pieces and
"verify" anything, so C3 is circular — I can always claim scalability by shrinking β.

- First, tell me whether my fear is even *empirically* true, given F5: in my own data
  β=1 is **slower** than the full layer (Iris 34.9s vs 24.5s; MNIST β=1 1099s vs β=5
  1464s is non-monotone), and the verdict never changes. If my fear is wrong, say so.
- Then address the version I think is actually damaging: **β partitions output neurons
  only, and affine rows are independent given the input box, so the decomposition is
  trivially sound and each block query is a closed-form interval computation.** Does
  that make the β sweep a measurement of *per-query constant-folding cost* rather than
  of verification difficulty? If so, C3 is not a verification-scalability result.
- Is `verification_claims.blockwise_verification = "equivalent_hidden_contract
  _decomposition_when_enabled"` a proved equivalence or an assertion? Where is the
  proof that per-block contracts compose to the layer contract?
- Is a β sweep that never changes a verdict (F6) publishable as an ablation at all?

### Q4 — What is ESBMC actually contributing? (the determinism finding)

F4 is the finding that worries me most after Q2. On the main path, ESBMC is handed a
**deterministic straight-line integer program** and asked to check an assertion about
it. There is no existential quantification, no search, no unwinding of a property
space — 114,069 VCCs collapse to 25.

- Is it fair to say ESBMC here is a **bit-exact arithmetic oracle** (checking
  overflow, rounding, clamping under `__int128`) rather than a model checker? That is a
  real and defensible contribution, but it is not "bounded model checking of neural
  networks".
- Could `exact_layer_interval` in [src/verification/invariants.py](../../src/verification/invariants.py)
  reproduce every layer-contract result in pure Python? Note that
  `_candidate_reachable_bounds` already calls exactly that function to *propose* the
  bounds that ESBMC then confirms. If Python proposes and Python could also check, what
  does the SMT call add beyond independent implementation cross-validation?
- Given that, is the honest framing "**we use BMC to certify the fixed-point arithmetic
  of a closed-form abstract transformer**"? Would that framing survive review, and does
  it still justify the paper's title/abstract?
- Where *is* genuine search happening (output competitors, vacuity sentinels), what
  fraction of runtime is it, and should the paper's scalability story be about *that*
  instead?

### Q5 — Is `PARTIAL_VERIFIED → deployed-transfer` a soundness-relevant bug or a reporting choice?

F3. Every LADC number I would put in a paper comes from a run where (a) the final
status is explicitly *partial*, (b) a formal obligation the config declared **required**
was `SKIPPED`, and (c) a second flag
(`no_saturation_required_for_deployed_transfer: false`) silently waives it, so the run
is still awarded the top guarantee level with `soundness: null`.

- Is the no-saturation obligation load-bearing for (C1)/(C2)? If saturation is never
  formally excluded, can the deployed kernel clamp where the abstraction assumed it
  did not — and would that break the contract chain or merely lose precision?
- Is `soundness: null` (vs `derived_budget` in other runs) a reporting gap or does it
  indicate the soundness label was never computed for this path?
- Should `PARTIAL_VERIFIED` ever reach `deployed-transfer`? Propose the correct ladder
  semantics. I would rather report a lower guarantee level honestly than publish
  `deployed-transfer` on a waiver.
- Is there any other place where a precondition is satisfied by a config echo rather
  than a check? (Start from F8, then audit all seven.)

### Q6 — Is the fixed-point fidelity claim (C2) actually established?

- F8: `fidelity_by_construction` proves nothing by itself. What *would* establish it?
  I think: a differential test of the harness arithmetic against the compiled deployed
  `.so` over a large random/boundary input set, reported as an artifact. Is
  `python_c_exact_match: true` that test, and what exactly does it compare?
- Is the *same C text* used by both the harness and the deployed kernel, or two
  renderings that happen to agree? `render_arith_kernel()` is shared, but
  `c_export/qnn_model.c` is a separate renderer — confirm they cannot drift.
- `fixed_point_semantics: "declared_backend_semantics"` (F9): if the semantics are
  *declared*, in what sense is deployment verified rather than specified? What breaks
  if the real deployment target (TFLite? CMSIS-NN? a custom kernel?) rounds differently
  (round-half-to-even, truncation, different clamp order, int32 accumulators)?
- Is the input region the set of genuinely representable deployed inputs? If real inputs
  are uint8 pixels but the box is taken over quantized reals with fractional bits, the
  verified region may not match the deployable one in either direction.

### Q7 — Where does the guarantee actually live, end to end?

Compose F4, F7, F9 into one honest statement of the theorem a reader gets. My current
understanding, which you should correct:

> *If* float DeepPoly is sound, *and* CBC's preimage is correct, *and* the declared
> fixed-point semantics match deployment, *and* the derived error budget bounds the
> per-neuron gap, *then* ESBMC has certified that the exact integer interval transfer
> over the given box stays inside the permitted preimage, hence the property holds.

- Is that the real claim? If so, ESBMC certifies the *middle* link only, and the two
  ends (input-space property; deployment semantics) are assumed or float-certified.
- Does the chain actually close? Specifically: DeepPoly certifies the **float** network;
  the contracts are about the **quantized** network. What bridges that gap, and is the
  bridge sound or heuristic? `SOURCE_PROPERTY_INCONCLUSIVE` (243 runs) suggests the
  float step is a gate — but passing it certifies the float network, not the integer one.
- Since end-to-end is disabled (F7), is there *any* artifact connecting a verified
  contract chain back to a concrete deployed input? If not, that is the experiment I
  most need to add.

### Q8 — Can this benchmark discriminate anything? (ablation validity)

F6: arms A, B, C are indistinguishable. Either margin cuts and e2e invariants do
nothing, or the benchmark is too easy, or the arms were not actually applied.

- Verify the ablation flags (`margin_cuts: "on"` vs `cuts_off`, `e2e_invariants`) are
  genuinely reaching the pipeline in arms B and C, rather than being ignored.
- If they are applied and nothing changes, what is the correct conclusion to publish?
- Design a benchmark that *can* discriminate: which (network, ε) regime puts the method
  near its precision boundary, where removing cuts or invariants flips a verdict?

### Q9 — The precision ceiling: a box is a box.

The abstraction discards all inter-neuron correlation. Intervals are exact
*per-neuron* for affine-over-box, so after the recent bound-tightening the per-layer
abstraction is as tight as an interval can be — meaning **all remaining imprecision is
the box abstraction itself**, and no amount of further engineering inside this domain
can recover it.

- Is that correct? If so, is the method's ceiling now structural, and should the paper
  say so?
- Is the honest next step a relational domain (zonotope/polyhedra/DeepPoly-style
  linear relations) over *exact fixed-point* semantics — and is *that* the genuinely
  novel contribution, rather than the current pipeline?
- Where does the method sit against the obvious baselines on *quantized* robustness
  (Marabou/α,β-CROWN with quantization, SMT-based QNN verifiers, Giacobbe et al. on
  quantized BNN/QNN BMC)? Is there prior work that already does bit-exact fixed-point
  NN verification with BMC, and if so what is left that is new?

### Q10 — What should I actually claim?

Given everything above, rewrite my contribution statement honestly. Tell me which of
C1–C4 survive as stated, which need to be downgraded, and which must be dropped. If the
defensible contribution is narrower than the current framing — e.g. "assume–guarantee
contracts over the bit-exact deployed integer kernel with a derived, ESBMC-checked error
budget" — say that, and tell me what the minimum additional experiments are to support
*that* claim properly.

---

## 4. Deliverables

1. A verdict on each of C1–C4: **supported / needs downgrading / unsupported**, with
   `file:line` evidence.
2. A formal statement of the soundness theorem with **every** undischarged assumption
   listed.
3. A ranked list of defects, separating **soundness bugs** from **overclaiming** from
   **experiment-design weaknesses**. Tell me which single one would sink the paper.
4. A concrete negative-control / falsification protocol (Q2) I can run.
5. A concrete differential-testing protocol for fixed-point fidelity (Q6).
6. A rewritten, honest contribution statement and the minimum experiments to support it.

If you find that a concern of mine is unfounded, say so explicitly and show the code
that refutes it. I want the real state of this methodology, not reassurance.
