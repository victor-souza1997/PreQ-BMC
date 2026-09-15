# PreQ-BMC methodology audit response

Audit date: 2026-09-10. Source revision: `a848dc16ad4303834d28607a4c717582bbed1168`, with the current working-tree experiment configurations. This answers [the audit prompt](methodology_audit_prompt.md). It audits the current derived, tightened-bound, layer-scope path and distinguishes it from historical output directories and optional E2E paths.

No production code, experiment configuration, or existing result was changed for this audit. The accompanying scripts collect existing evidence and exercise an isolated input-domain diagnostic. They do not rerun the LADC campaign.

## 1. Verdict

The compositional integer proof has a defensible mathematical basis. Deterministic hidden-layer harnesses do not invalidate it. However, I cannot endorse the unrestricted statement that every reported `VERIFIED` establishes robustness on the originally requested real-valued perturbation region. There is a reproducible gap in the input-region conversion, and the reporting ladder is not an independently checked certificate.

| Claim | Verdict | What the evidence supports |
|---|---|---|
| C1: `VERIFIED` implies deployed robustness throughout the input region | **Needs downgrading** | A conditional proof for the encoded integer input box, provided the composition, arithmetic-range, cut-validation, artifact-identity, and toolchain assumptions below hold. The bridge from the requested real box to that integer box needs repair or an explicitly narrower, justified input specification. |
| C2: bit-exact deployment fidelity | **Needs downgrading** | Strong construction and test evidence for the generated reference C backend. Three primitive helpers are literally shared; complete layer/network renderers are separate. This is not a theorem about TFLite, CMSIS-NN, or arbitrary firmware. |
| C3: blocking makes otherwise intractable networks tractable | **Needs downgrading** | Blocking changes per-query cost, memory, and process overhead. Current completed runs demonstrate cost differences, not a general increase in certified regions over a fully matched monolithic campaign. Beta zero is monolithic **per layer**, not whole-network verification. |
| C4: cuts, E2E invariants, and blocking each improve effectiveness | **Unsupported as stated** | All 26 matched matrix regions select the same formats with cuts on/off; hidden CEGAR executes zero rounds; E2E is disabled in both invariant arms. Cuts can be expensive without improving these outcomes. |

C1 evidence: input construction at [pipeline.py:1120](../../src/synthesis/pipeline.py#L1120), integer assumptions at [preqbmc.py:2207](../../src/synthesis/preqbmc.py#L2207), chaining at [preqbmc.py:2859](../../src/synthesis/preqbmc.py#L2859), exact output checks at [preqbmc.py:4550](../../src/synthesis/preqbmc.py#L4550). C2 evidence: [arith_kernel.py:4](../../src/verification/arith_kernel.py#L4), [c_qnn_generator.py:55](../../src/backends/c_qnn_generator.py#L55), [c_templates.py:573](../../src/verification/c_templates.py#L573). C3 evidence: [preqbmc.py:2075](../../src/synthesis/preqbmc.py#L2075), [preqbmc.py:4681](../../src/synthesis/preqbmc.py#L4681). C4 evidence: [preqbmc.py:3449](../../src/synthesis/preqbmc.py#L3449), [preqbmc.py:4734](../../src/synthesis/preqbmc.py#L4734), [c_templates.py:1297](../../src/verification/c_templates.py#L1297), and the campaign measurements below.

## 2. Rechecking F1-F9

The complete numerical snapshot is [methodology_audit_evidence.json](methodology_audit_evidence.json). Recompute it without changing outputs:

```bash
python docs/ladc2026_review/collect_methodology_audit_evidence.py
```

The collector counts files, nested status occurrences, and model/sample/epsilon tuples separately; see [collect_methodology_audit_evidence.py:134](collect_methodology_audit_evidence.py#L134). Counts include archived files where explicitly stated. They are not independent trial counts.

### F1: zero network refutations, but an incorrect denominator

There are **986 `experiment_summary.json` files**, not 7,094 independent experiments. Their top-level statuses include 345 `VERIFIED`, 193 `PARTIAL_VERIFIED`, 218 `FAILED`, 81 `SOURCE_PROPERTY_INCONCLUSIVE`, 45 `TIMEOUT`, and other inconclusive outcomes. Recursively counting every `final_status` field produces **7,197 occurrences**, including 5,081 `VERIFIED`. The prompt's numbers have the pattern of an earlier recursive count: sections and layer records repeat statuses inside the same report.

No inspected `pipeline_summary.json` contains `MARGIN_REFUTED`. This supports the absence of recorded network refutations; it does not establish that all historical failures have been individually diagnosed. A raw ESBMC failure is also used deliberately for nonvacuity sentinels; those failures are not robustness refutations ([preqbmc.py:4158](../../src/synthesis/preqbmc.py#L4158)).

### F2: distinguish replay scope and duplicate payloads

The count **27** is reproducible when counting confirmed entries only in `reports/pipeline_summary.json`. Including duplicate pipeline summaries elsewhere gives 53 occurrences; there are 14 distinct serialized payloads. Payload identity is not a reliable independent-region identity. These are not all confined to `MARGIN_INCONCLUSIVE` in the complete archive; some appear in `FAILED` and `PARTITION_INCONCLUSIVE` runs.

The central interpretation is correct: `replay_confirmed` in the ordinary candidate loop means a layer computation violates its candidate contract at the recorded **layer input**, not that the original network has a reachable adversarial input. See [preqbmc.py:4008](../../src/synthesis/preqbmc.py#L4008) and [preqbmc.py:5962](../../src/synthesis/preqbmc.py#L5962).

### F3: status contradiction confirmed; `soundness` was computed

The current campaign has **156 matrix reports and five escalation reports**, including a newly available MNIST `1blk_100` beta-one result. Every one has top-level `PARTIAL_VERIFIED`, `deployed-transfer`, and skipped no-saturation in its experiment summary. However, every corresponding **pipeline summary explicitly contains `soundness: derived_budget`**. The experiment report omits the top-level field; reading it with `.get()` returns null. This is a reporting omission, not evidence that the pipeline never computed the label ([pipeline.py:1214](../../src/synthesis/pipeline.py#L1214), [experiment_summary.py:810](../../src/reports/experiment_summary.py#L810)).

There are zero timed-out **preimage blocks**, but **208 timed-out relational-cut validation queries in the matrix**, plus nine in the escalation directory. Saying that these runs had no ESBMC timeouts would be false.

### F4: hidden harness determinism confirmed; the stated exception list is incomplete

All 3,017 matrix preimage query records reference harnesses without `nondet_longlong(`. All 767 output-competitor queries, 777 sentinels, and **308 exact-prefix cut-validation queries** reference harnesses containing it. This was checked against the generated files, not inferred only from a renderer name.

Exact-prefix cut validation also starts from the original encoded network input. Thus the claim that only the disabled E2E family ever quantifies over original inputs is false ([c_templates.py:1428](../../src/verification/c_templates.py#L1428), especially [c_templates.py:1520](../../src/verification/c_templates.py#L1520)).

The hidden computations have bounded loops, rather than literally being straight-line source. Unrolling makes their constant-data computation deterministic. Source nondeterminism does not by itself measure residual solver search after simplification ([c_templates.py:573](../../src/verification/c_templates.py#L573), [esbmc.py:140](../../src/verification/esbmc.py#L140)).

### F5: the quoted times are not the current total-runtime field

Using `timing_metrics.total_runtime_seconds`, Iris sample 0 has:

| Beta | Total seconds | Recorded calls | Selected Q per layer |
|---|---:|---:|---|
| 1 | 35.78 | 81 | 10, 15, 7 |
| 2 | 29.45 | 46 | 10, 15, 7 |
| 5 | 26.00 | 25 | 10, 15, 7 |
| 10 | 24.85 | 18 | 10, 15, 7 |
| 0 | 25.50 | 18 | 10, 15, 7 |

MNIST `1blk_25`, sample 3, epsilon 0.25 has 1,135.05 / 1,174.33 / 1,496.29 seconds at beta 1 / 2 / 5. That particular sequence is **monotone increasing**, not non-monotone. The original concern confuses different datasets and different timing fields. Small blocks help this MNIST case but add overhead for Iris.

### F6: identical final outcomes, not identical execution

The matrix covers **26 distinct model/sample/epsilon regions**, replicated across configurations. At matched beta two, all 26 cuts-on/cuts-off pairs select the same Q/I/F. All 156 matrix reports record `cegar.rounds_run = 0`.

Nevertheless, cuts change cost. Across the 13 matched MNIST region pairs, the median **per-pair** total-runtime ratio, cuts-on divided by cuts-off, is **7.32**. This is a descriptive paired result, not a repeated-timing confidence estimate. Arm B has no cut-validation queries; Arm C still has 76. The arms are not identical workloads.

### F7-F9: qualifications

- E2E verification is disabled throughout these campaigns, but exact-prefix cut proofs are active. The constructor records the requested invariant setting even for `NOT_RUN` E2E ([preqbmc.py:410](../../src/synthesis/preqbmc.py#L410)).
- `fidelity_by_construction` reads a renderer-semantics declaration, not an equivalence proof. The declaration is hard-coded by the pipeline, rather than an arbitrary user toggle ([pipeline.py:1261](../../src/synthesis/pipeline.py#L1261), [experiment_summary.py:268](../../src/reports/experiment_summary.py#L268)). Shared code and actual differential tests are supporting evidence, discussed under Q6.
- The `verification_claims` strings are provenance labels, not discharged proof obligations ([pipeline.py:1267](../../src/synthesis/pipeline.py#L1267)). Their presence alone neither establishes nor refutes soundness.

## 3. Q1 and Q7: the theorem and its actual dependencies

### 3.1 Concrete semantics

Fix the exported weights, biases, layer order, and one shared format per layer. Let layer l have total width Q_l and output/weight fractional width F_l. Let F_0 be the input fractional width. In this backend F_0 normally equals the first layer's fractional width, but the mathematical definitions must distinguish input and output scales.

Define signed saturation and rounding by

\[
\kappa_Q(v)=\min(2^{Q-1}-1,\max(-2^{Q-1},v)),\qquad
\rho_s(a)=\operatorname{roundHalfAwayFromZero}(a/2^s).
\]

The clamped affine result and hidden transition are

\[
z_{l,j}(a)=\kappa_{Q_l}\!\left(
 \rho_{F_{l-1}}\!\left(\sum_k W^q_{l,jk}a_k\right)+b^q_{l,j}
\right),\qquad
T_l(a)_j=\kappa_{Q_l}(\max(0,z_{l,j}(a))).
\]

The output layer omits ReLU. This matches [c_qnn_generator.py:67](../../src/backends/c_qnn_generator.py#L67) and [fixed_point.py:190](../../src/backends/fixed_point.py#L190). Dividing every layer by its own output scale would be wrong when adjacent fractional widths differ; the current renderer uses the predecessor/input scale.

Let B be the intended, clipped and normalized real input region, and let eta be the actual deployed input quantizer. Define the encoded integer box

\[
A_0=\prod_k [L_{0,k},U_{0,k}]\cap\mathbb Z^d.
\]

The essential input obligation is **eta(B) subseteq A_0**. Equality is useful for precision but unnecessary for soundness. The current endpoint-quantization helper is appropriate for the endpoints it receives; its exact-image description does not prove that those endpoints enclose the original requested B ([fixed_point.py:62](../../src/utils/fixed_point.py#L62)).

### 3.2 Endpoint lemma

For an integer box A = [L,U], define

\[
m_j=\sum_k W^q_{jk}\begin{cases}L_k&W^q_{jk}\ge0\\U_k&W^q_{jk}<0\end{cases},\qquad
M_j=\sum_k W^q_{jk}\begin{cases}U_k&W^q_{jk}\ge0\\L_k&W^q_{jk}<0.\end{cases}
\]

For every a in A, m_j <= sum_k W^q_jk a_k <= M_j. Both extrema are attained on the Cartesian box. Rounding by a positive scale, adding a fixed bias, saturation, and ReLU are monotone. Applying them to these endpoints therefore gives exact **coordinatewise extrema over this box**. The joint reachable set need not fill the resulting box, and the scalar image can contain gaps.

This mathematical lemma supplies the universal quantification absent from the deterministic C harness. ESBMC checks each instantiated endpoint calculation and its containment assertions; it does **not** prove this general lemma for all possible networks. The implementation uses sign-dependent endpoints and the shared rounding/clamp helpers ([c_templates.py:585](../../src/verification/c_templates.py#L585), [c_templates.py:599](../../src/verification/c_templates.py#L599)); Python computes the same affine-box extrema ([invariants.py:90](../../src/verification/invariants.py#L90)).

With no hidden-row cuts, the relation checked is

\[
\{a\in A_{l-1}\}\ C_l^{\rm affine}\ \{z_l\in G_l\cap R_l\},
\]

where G_l is the emitted tolerated preimage target and R_l is the additionally proposed clamped-affine box. R_l here denotes a **box enclosure**, not the exact reachable set and not a relational cut. The proposed-R assertion is emitted by [c_templates.py:179](../../src/verification/c_templates.py#L179).

Python then propagates

\[
A_l=\operatorname{ReLU}(G_l)\cap\operatorname{ReLU}(R_l),
\]

with the signed-range clamp included as necessary. Actual hidden outputs belong to both terms, hence to their intersection. This is what [preqbmc.py:2885](../../src/synthesis/preqbmc.py#L2885) through the tightening logic implements. The incoming guarantee is pre-ReLU; the next assumption is post-ReLU. The second clamp is an identity because ReLU maps a Q-bit signed value into [0, 2^(Q-1)-1]. No no-saturation theorem is needed for that observation.

With validated relational cuts, the premise changes. A cut J_l is established over the exact selected prefix:

\[
\{u\in A_0\}\ C_1;\ldots;C_{l-1}\ \{J_l(a)\}.
\]

The local obligation is then over A_(l-1) intersect J_l, rather than the entire Cartesian box. Hidden cuts must match the exact quantized row, and only ESBMC-validated cuts are accepted ([preqbmc.py:3302](../../src/synthesis/preqbmc.py#L3302), [preqbmc.py:5443](../../src/synthesis/preqbmc.py#L5443)). An endpoint tightened using such a cut is sound over reachable prefix states; it is not generally an exact bound over the original unrestricted box.

### 3.3 Conditional deployment theorem

For the current target-class, tightened-bound, layer-scope path, assume:

1. eta(B) is contained in the nonempty encoded A_0.
2. The generated affine operations implement the concrete transition T_l above, with no undefined arithmetic or memory behavior.
3. Every selected hidden-layer block proves its emitted G_l and R_l bounds for the same weights, biases, predecessor assumptions, and Q/I/F.
4. Every cut used as an assumption is valid for the selected exact prefix on the same A_0.
5. The propagated A_l contains the post-ReLU image of the verified guarantees.
6. Every output competitor query proves strict target dominance on the final assumption set, including any validated cuts.
7. The exported reference C program implements these same transitions and input encoding; the checker, solver, compiler, and execution platform behave according to their specified semantics.

Then

\[
\forall x\in B,\ \forall k\ne t:\quad
\widehat N(\eta(x))_t>\widehat N(\eta(x))_k.
\]

**Proof.** The base case follows from assumption 1. Assume the selected prefix's reachable states are contained in A_(l-1). Validated prefix cuts also hold there. The endpoint lemma and the successful per-block obligations imply that the next clamped affine result lies in G_l and R_l. Monotonic ReLU and the propagated intersection place every hidden output in A_l. Induct over hidden layers. The output obligations imply every strict target-versus-competitor conjunct, hence the result. Blocks partition a conjunction; they do not partition the deployed network or its formats.

The selected-candidate checks, chaining, nonvacuity, and recursive downstream search appear at [preqbmc.py:1643](../../src/synthesis/preqbmc.py#L1643). Integer predecessor bounds are retained without a float round-trip when tightening is enabled ([preqbmc.py:2207](../../src/synthesis/preqbmc.py#L2207), [preqbmc.py:2842](../../src/synthesis/preqbmc.py#L2842)). Every competitor is sent to an actual C query in this mode, even if the analytic pre-pass passes ([preqbmc.py:4575](../../src/synthesis/preqbmc.py#L4575)).

This is a conditional theorem for the encoded implementation, not a proof that the complete Python implementation satisfies all premises on every supported configuration.

### 3.4 What remains outside ESBMC's discharged obligations

| Dependency | Status in this path |
|---|---|
| Requested-region normalization and quantizer-image coverage | Trusted Python preprocessing. A concrete boundary failure is reproduced below. |
| General monotonicity and affine endpoint theorem | Mathematical argument; not verified as a generic transformer theorem by the instance harnesses. |
| Tensor orientation, activation choice, scales, formats, and identity of selected/exported artifacts | Generator/control-flow assumptions plus tests. Sharing primitive helpers does not prove whole-program equivalence. |
| Arithmetic definedness | Exact-Python static checks cover layer MAC products, every accumulation prefix, rounding temporaries, bias addition, widths, and input endpoints. These checks are not ESBMC overflow proofs. Extra directional-cut arithmetic and Python endpoint/budget conversions also need range validity. |
| Checker and solver implementation | Trusted ESBMC frontend, interval simplifier, bit-vector backend, status parser, and sufficient unwinding. No externally checked proof certificate is consumed by reporting. |
| Compiler and runtime | Trusted compilation of generated C, ABI and integer representation assumptions, and the absence of faults outside the modeled program. |
| Cut validity and reuse | Exact-prefix ESBMC checks provide proof evidence, but correct association with the current input box, prefix tensors, and formats remains an orchestration obligation. |
| Root source certificate | DeepPoly numerical implementation and real-versus-IEEE semantics are assumed for the *source-model* claim. They are not proved by the integer harness. |
| MILP preimage correctness | Needed to call the numerical object a true property preimage. It is not needed as an independent axiom for fixed-point robustness once all actual integer reachability and exact output obligations above pass. |
| Derived error-budget correctness | Needed for a claimed quantitative float/integer error theorem and for analytic-only output discharge. With tightening and every exact output competitor checked, it guides candidate contracts and cuts; a bad proposal cannot replace the final integer property proof. |
| Report-to-proof correspondence | A status/boolean is derived from records and declarations, not independently checked against hashed proof artifacts. |

Evidence for arithmetic checks: [invariants.py:68](../../src/verification/invariants.py#L68), [invariants.py:106](../../src/verification/invariants.py#L106), [invariants.py:153](../../src/verification/invariants.py#L153), and the pre-ESBMC guard at [preqbmc.py:4294](../../src/synthesis/preqbmc.py#L4294). The paper profiles do not enable `--overflow-check`; that flag is added only for safety/overflow profiles ([esbmc.py:182](../../src/verification/esbmc.py#L182)). Do not describe the paper profile as an independent universal signed-overflow proof.

The important correction to the prompt's proposed theorem is that **the current integer output proof closes the chain**. It does not infer quantized robustness merely from float robustness plus an asserted budget. DeepPoly and MILP still gate and guide the implemented synthesis procedure ([preqbmc.py:935](../../src/synthesis/preqbmc.py#L935)), but the exact output query is a separate certificate of the selected integer implementation. Their precision affects whether synthesis succeeds; their numerical optimality is not the logical foundation of that final exact query.

### 3.5 Concrete input-domain gap

The pipeline constructs endpoints from float32 dataset samples, then explicitly passes float32 normalized endpoints to GPEncoding ([loaders.py:38](../../src/datasets/loaders.py#L38), [pipeline.py:1120](../../src/synthesis/pipeline.py#L1120), [pipeline.py:1190](../../src/synthesis/pipeline.py#L1190)). Deployment comparison normalizes in float64 and quantizes float64 values ([pipeline.py:128](../../src/synthesis/pipeline.py#L128), [fixed_point.py:183](../../src/backends/fixed_point.py#L183)). Conversion to float32 is round-to-nearest, not a certified outward enclosure.

The diagnostic [check_input_domain_boundary.py](check_input_domain_boundary.py) reproduced the following using the existing C exporter, input quantizer, and ESBMC E2E renderer:

```bash
PYTHONPATH=src python docs/ladc2026_review/check_input_domain_boundary.py
```

- Center x0 = 0.5, epsilon = 0.437500000001, scale 2^3, signed Q = 6.
- Requested lower endpoint is approximately 0.06249999999900002.
- The float32 lower endpoint becomes 0.0625.
- Quantizing the requested lower endpoint gives integer **0**; quantizing the rounded lower endpoint gives **1**.
- The generated integer box is **[1,8]**, omitting integer 0 even though a float64 input within the requested real region quantizes to it.
- Toy source: h = ReLU(x), logits (2h, b), where b is stored float32 0.1. Its minimum real-arithmetic source margin over this region is approximately **0.0249999985 > 0**.
- At Q=6, F=3, the integer model has hidden W=8, output W=(16,0), output biases (0,1). It strictly predicts class 0 for every integer input in [1,8]. ESBMC returned **VERIFIED** on that box.
- At the omitted input 0, the compiled generated C library returns **[0,1]**, changing the prediction. ESBMC returned **FAILED** with witness **[0]** when the harness box was widened to [0,8]. No E2E invariants were injected in either diagnostic query.

This is a concrete counterexample to the **domain-transfer implication**. It is not a reproduced false `deployed-transfer` result from the complete GPEncoding synthesis pipeline, and it does not establish that any collected LADC region has this mismatch. It does establish that the unrestricted real-region theorem needs an additional premise that preprocessing currently does not guarantee.

If deployment is explicitly restricted to float32 sensor values, the admissible input set must be specified accordingly and this float64 witness may be excluded. That restriction cannot be introduced silently after claiming all real points in B. For uint8 images, specify the raw discrete domain and the exact normalization; a continuous normalized box can conservatively cover it, but inclusion must be checked.

### 3.6 Other qualifications to a source/preimage theorem

DeepPoly here uses ordinary NumPy multiply/add bound propagation, without an explicit directed-rounding enclosure in that routine ([DeepPoly_preqbmc.py:77](../../src/symbolic_pp/DeepPoly_preqbmc.py#L77)). It reasons about an affine/ReLU model with stored coefficients; it is not an ESBMC proof of TensorFlow's IEEE floating-point execution. The source gate also accepts `target_lower >= maximum_competitor_upper`, while the integer property is strict `>` ([preqbmc.py:950](../../src/synthesis/preqbmc.py#L950), [c_templates.py:74](../../src/verification/c_templates.py#L74)). Equality alone cannot establish strict source dominance or a chosen tie-breaking class. This is a source-claim issue, not a bypass of the exact integer output query.

The numerical preimage construction minimizes an existential violation boundary using solver tolerances and symbolic input-dependent bounds ([preqbmc.py:1295](../../src/synthesis/preqbmc.py#L1295), [preqbmc.py:1360](../../src/synthesis/preqbmc.py#L1360), [preqbmc.py:1410](../../src/synthesis/preqbmc.py#L1410)). Storing its optimal boundary as a closed safe Cartesian box needs justification; a minimum violating boundary can contain a violating endpoint. The output violation uses a positive tolerance and hidden disjunctions retain hard-coded 1000 constants. Do not claim that merely trusting CBC proves these formulas encode exactly the desired strict property or that all stored boxes are certified safe preimages.

Similarly, [preqbmc.py:2322](../../src/synthesis/preqbmc.py#L2322) converts a stored real preimage with floor(lower*S) and ceil(upper*S). This is outward rounding: it encloses a set but does not preserve membership in a safe target set. For example, P=[0.1,0.9], S=1 produces integer [0,1], whose decoded endpoints are outside P. It is suitable as an enclosure, not a standalone proof that integer containment implies real preimage preservation. In the current tightened path, exact downstream contracts and the exact output query protect the integer robustness conclusion; the stronger statement that every real preimage is preserved exactly is not established by this conversion.

The derived budget uses exact rational coefficient amplification and an upward-rounded weight/input/rounding bound ([preqbmc.py:2490](../../src/synthesis/preqbmc.py#L2490)). Its half-ULP parameter error argument presumes weights and biases are not clipped outside their representable range. A clamp on the quantized activation is nonexpansive relative to a reference already inside its range; nonexpansiveness alone does not bound deviation from an unclamped real value outside that range. These are additional premises for a quantitative source-to-integer error theorem. They must not be inferred from the presence of a `derived_budget` label.

`PREIMAGE_DEFLATION_EMPTY` is an inconclusive synthesis outcome, not a counterexample. Last-hidden deflation is tested before harness generation, rejected when inverted, and cannot be chained ([preqbmc.py:2397](../../src/synthesis/preqbmc.py#L2397), [preqbmc.py:4329](../../src/synthesis/preqbmc.py#L4329), [preqbmc.py:2889](../../src/synthesis/preqbmc.py#L2889)). Within representable integer arithmetic, deflate-then-expand cancels the budget and returns the emitted integer preimage. That algebra does not cure the outward-rounding or real-preimage provenance issues above. Python int64 endpoint/budget operations also require range validity.

## 4. Q2: refutations and negative controls

The layer-only mode with `e2e_fallback=off` is a **sufficient certifier and candidate-rejection procedure**, not a complete network falsifier. A hidden-state violation can be spurious because the state is unreachable, but it can also be reachable. The current path does not solve a reachability/replay query to distinguish those alternatives. Therefore saying that such counterexamples *can never* be confirmed mathematically is incorrect; saying this mode *does not confirm them as original-input refutations* is correct ([preqbmc.py:3842](../../src/synthesis/preqbmc.py#L3842)).

`MARGIN_REFUTED` is reachable in code through the E2E fallback, with a failed exact-network query and both Python and compiled-C replay confirmation ([preqbmc.py:3891](../../src/synthesis/preqbmc.py#L3891)). Direct network mode records its own `FAILED`/replay result rather than necessarily that same status name ([preqbmc.py:844](../../src/synthesis/preqbmc.py#L844)). Searching only for `MARGIN_REFUTED` is therefore not a complete general falsification audit.

Also, a source-region gate deliberately stops many large-epsilon or false-target tests before the integer verifier ([preqbmc.py:971](../../src/synthesis/preqbmc.py#L971)). Such a stop is a legitimate `SOURCE_PROPERTY_INCONCLUSIVE`, not evidence that the bit-vector verifier can refute. The gate is based on the original prediction, while the property can carry an explicit target; test and label these cases separately ([preqbmc.py:445](../../src/synthesis/preqbmc.py#L445), [pipeline.py:1126](../../src/synthesis/pipeline.py#L1126)).

### Minimum negative-control protocol

Freeze network tensors, input quantizer, Q/I/F, input bounds, target, and compiler before each verification. Disable format repair for a falsification test: changing the candidate after finding a violation answers a different question.

1. **False target control.** On a tiny fixed integer network and nonempty box, require the wrong output class. Run the exact-network harness without invariants; expect a property failure and a concrete integer input. Replay it through Python and `qnn_forward_fixed` in the compiled library. An explicitly separate source-gate test may stop before ESBMC; do not confuse the two tests.
2. **Source-robust, quantization-broken control.** Use a one-input toy source with hidden ReLU and logits `(0.14, 0.13)`, constant over the region. At F=2 both output biases round to integer 1, violating strict dominance despite the source's positive margin. A stronger misclassification control uses h=x, logits `(2h,0.1)`, B=[0.055,0.060], and F=3: source class 0 is robust, but input quantization gives 0 and C outputs `[0,1]`. Verify the fixed candidate, not a repaired one.
3. **True decision-boundary control.** Choose a tiny exact-integer network with known boundary and a box containing points on both sides. The exact harness must return a replayable violating original input. Repeat with a subset known to be safe and expect verification.
4. **Spurious-box control.** Use two hidden neurons `h1=ReLU(x)` and `h2=ReLU(x)`, x in [0,1], with logits `h1-h2+1` and 0, at integer scale one. The reachable network is safe, but the hidden Cartesian box admits `(0,1)`, producing a tie. Expect a local failure/inconclusive result, an exact-prefix proof of h1=h2, and success when that proved relation or exact E2E checking is used. This exercises the reason for CEGAR without weakening the property.
5. **Vacuity and bounds control.** Inverted/empty boxes must not produce a deployed certificate. An insufficient unwind bound must not be mistaken for a property refutation or successful verification. A nonempty sentinel should fail its intentional assertion, with that purpose recorded.

For every negative, archive integer input, its membership check in A_0, a real/raw preimage when the claim concerns B, full integer logits, strict-property and argmax outcomes separately, candidate Q/I/F, tensor/source hashes, command, solver version, and both replays. The repository provides integer layer replay and direct shared-object replay ([replay.py:25](../../src/verification/replay.py#L25), [replay.py:63](../../src/verification/replay.py#L63)). Membership matters: replaying a trace is not by itself a proof that it is in the claimed region.

These controls are required for confidence in failure handling; their absence does not logically falsify every positive certificate. Existing CEX tests mainly check trace parsing, local replay, and a mocked candidate filter ([test_cex_loop.py:15](../../src/tests/test_cex_loop.py#L15)). The existing low-precision E2E test deliberately expects success after removing an unreachable lattice point, not a real negative ([test_network_end_to_end_verification.py:151](../../src/tests/test_network_end_to_end_verification.py#L151)). The domain diagnostic added for this audit does demonstrate a real ESBMC failure and compiled-C misclassification, but not the complete synthesis/refutation reporting path.

### Mutation and metamorphic checks

Use exhaustively enumerable tiny integer networks as ground truth. Mutate one behavior at a time: positive and negative half-tie rounding, predecessor versus output rescaling, bias placement, clamp omission, clamp width, a weight sign, an input bound narrowed by one ULP, a mismatched cut direction, or one omitted output competitor. Keep only mutations whose expected effect is established by exhaustive evaluation; harmless/equivalent mutations should not be required to fail. The fidelity oracle or proof-association check must detect every deliberately behavior-changing mutation.

For fixed tensors and fixed Q/I/F, successful verification on B must remain mathematically valid on any subset; a replayed witness remains a witness in any superset containing it. Changing beta must preserve the logical property. Increasing precision is **not** a valid metamorphic monotonicity rule because rounding, weights, saturation and even the input quantizer change. Compare proof outcomes with exhaustive ground truth rather than assuming that more bits always improve robustness.

### Are the regions trivial?

All matrix source-margin lower bounds are positive and comfortably away from zero in raw logit units: Iris minimum 1.4033, MNIST 3.1746, Seeds 4.7128. These values are not comparable across models because logit scaling is arbitrary. They establish that the chosen source regions are not near the **recorded source bound's** zero margin. They cannot establish that every quantized candidate or every competing verifier finds them easy.

Some first-layer candidate input boxes have zero symbolic coordinates after quantization, while others retain many coordinates. Report final-candidate integer cardinality and varying-coordinate counts, not epsilon alone. The matrix's 156 configurations contain only 26 distinct model/sample/epsilon regions, including multiple radii of the same sample. None should be presented as 156 statistically independent inputs.

## 5. Q3: what beta demonstrates

For fixed assumptions, parameters, cuts, and Q/I/F, write H_j(a) for neuron j's obligation. If blocks B_b cover all output neurons,

\[
\forall a\in A:\bigwedge_j H_j(a)
\quad\Longleftrightarrow\quad
\bigwedge_b\left(\forall a\in A:\bigwedge_{j\in B_b}H_j(a)\right).
\]

This elementary equivalence is the proof of decomposition. It does not require the neurons to be statistically independent. Separate blocks may reach their extrema at different inputs; conjunction of universally quantified obligations remains equivalent. The implementation slices rows, retains all input bounds, shares candidate formats, and checks sequentially ([preqbmc.py:2081](../../src/synthesis/preqbmc.py#L2081), [preqbmc.py:5348](../../src/synthesis/preqbmc.py#L5348), [preqbmc.py:4681](../../src/synthesis/preqbmc.py#L4681)). Failure is retained when remaining blocks are skipped ([preqbmc.py:4915](../../src/synthesis/preqbmc.py#L4915)).

The fear that beta one lets the tool verify anything is **wrong**. An out-of-contract affine endpoint still violates its one-neuron obligation; an unproved output competitor still blocks success. Smaller blocks neither shrink the input set nor permit block-specific precision.

The cost concern is legitimate. For the no-cut hidden path, the mathematical extrema are computable by integer arithmetic without SMT search. Beta mostly partitions generated-program size, constant propagation, unrolling, arithmetic checking, and process startup. It is an implementation-cost mechanism, not a theorem about solving a PSPACE-hard verification problem by checking independent scalar neurons.

There is useful measured evidence: in MNIST `1blk_10`, sample 3, epsilon 0.25, hidden-query totals at beta 1/2/5 are approximately **25.87/43.56/135.33 seconds**, and maximum sampled hidden-query RSS is approximately **190.11/331.64/609.65 MiB**. The total runtimes are **100.76/117.41/209.19 seconds**. These were calculated from `esbmc_call_records` in the corresponding [matrix run directories](../../output/ladc2026_matrix_runs/). Memory is sampled process-tree RSS, not a guaranteed exact peak; the monitor is at [esbmc.py:455](../../src/verification/esbmc.py#L455).

The escalation directory now also has successful contract runs for beta 0 and beta 10 at larger per-query timeouts. They complete in about 413-529 total seconds; their final experiment statuses remain partial for the no-saturation reporting reason. They demonstrate that monolithic-per-layer queries can finish with a larger budget on this case. Do not describe them as permanent monolithic infeasibility or compare their timeout-limited certification rate to small blocks without exposing the unequal budgets.

An ablation can be publishable when certification does not change: it can establish cost/overhead and semantic consistency. It cannot establish improved certification power. Use paired fixed-Q/I/F queries, per-property time/memory, equal resource budgets, and repeated timings. Changing beta during full synthesis can also change candidate schedules or cut selection under resource limits; the equivalence above applies to the same logical obligations, not necessarily equal runtime schedules.

## 6. Q4: ESBMC's contribution

The hidden no-cut path uses ESBMC to check **instance-specific C interval certificates** under the shared fixed-point primitives. That is a legitimate BMC application to a deterministic bounded program. It does not symbolically enumerate original input executions in those particular harnesses, nor prove the interval algorithm correct for every possible instance ([c_templates.py:555](../../src/verification/c_templates.py#L555)).

For this path, exact Python interval calculation plus containment can reproduce the hidden mathematical decision, subject to the same integer semantics and target intervals ([invariants.py:90](../../src/verification/invariants.py#L90), [preqbmc.py:2769](../../src/synthesis/preqbmc.py#L2769)). ESBMC supplies a separately executed C-level check with compiler/frontend semantics and assertion checking. Calling it a general arithmetic oracle that proves absence of overflow would overstate the paper profile, which relies on the Python static safety gate for signed overflow.

The full pipeline has additional BMC obligations with symbolic inputs: exact selected-prefix cut validation and output-class competitors. Output competitors retain one shared hidden vector, so they can prove a margin even when independently minimizing the target and maximizing the competitor is inconclusive ([c_templates.py:159](../../src/verification/c_templates.py#L159), [test_output_margin_harness.py:59](../../src/tests/test_output_margin_harness.py#L59)). Prefix proofs check relations against actual encoded original inputs ([c_templates.py:1520](../../src/verification/c_templates.py#L1520)). Therefore saying that ESBMC is merely decorative throughout the method is unsupported.

Across the 4,869 matrix query records:

| Obligation | Calls | Recorded seconds | Share of recorded ESBMC time |
|---|---:|---:|---:|
| Hidden preimage/endpoint | 3,017 | 15,316.80 | 17.11% |
| Nonvacuity sentinel | 777 | 163.58 | 0.18% |
| Exact-prefix relational-cut validation | 308 | 73,648.43 | 82.27% |
| Output competitor | 767 | 391.48 | 0.44% |

There are 100 verified cut-validation calls and 208 timeouts. A timed-out cut is excluded, not used as an assumption ([preqbmc.py:3404](../../src/synthesis/preqbmc.py#L3404), [preqbmc.py:3563](../../src/synthesis/preqbmc.py#L3563)). Successful final certification after such a timeout is legitimate because the cut is an optional proposal, whereas the final property still has to pass. This differs fundamentally from skipping a required final property.

The percentages are obligation-family costs, not solver-search fractions. To attribute time to constant folding versus solving, collect preprocessing/solver phase timings; VCC counts alone do not provide that split. The most immediate empirical result is that attempted cuts dominate this workload without improving the matched certification or precision outcomes.

An honest framing is: **preimage-guided quantization with compositional certification of generated fixed-point C, combining checked interval instances, optional exact-prefix relations, and symbolic output obligations**. The title need not abandon BMC, but the paper must explain these different uses.

## 7. Q5: saturation and the guarantee ladder

The clamp is modeled in the hidden, output, E2E, and prefix transitions inspected here ([c_templates.py:610](../../src/verification/c_templates.py#L610), [c_templates.py:144](../../src/verification/c_templates.py#L144), [c_templates.py:1319](../../src/verification/c_templates.py#L1319), [c_templates.py:1485](../../src/verification/c_templates.py#L1485)). Thus absence of saturation is **not** a premise of the compositional integer theorem. Saturation behavior is included in the proof; saturation-freedom is a different assertion.

The current status mismatch is nevertheless a defect in experimental accounting. Required no-saturation is coupled to a quality-refinement path that is disabled when the maximum number of refinement steps is zero ([pipeline.py:201](../../src/synthesis/pipeline.py#L201), [pipeline.py:885](../../src/synthesis/pipeline.py#L885), [pipeline.py:979](../../src/synthesis/pipeline.py#L979)). `_final_status` treats its absence as partial, while the transfer ladder consults the clamp declaration ([experiment_summary.py:124](../../src/reports/experiment_summary.py#L124), [experiment_summary.py:445](../../src/reports/experiment_summary.py#L445)). Aggregation can then overwrite the partial status with `VERIFIED` based on synthesis success ([aggregate_article_results.py:438](../../src/scripts/aggregate_article_results.py#L438)).

Separate three concepts explicitly:

1. **Robustness certificate:** do the encoded integer property and all transfer premises hold?
2. **No-saturation certificate:** was the stronger absence-of-clamping property proved?
3. **Experiment completion:** were all obligations requested in the saved configuration discharged?

It is logically possible to have the first certificate while the third is partial. If retaining `deployed-transfer` in that situation, make its precise scope explicit and do not label the configured experiment fully successful. Alternatively require full completion for the top public badge. Either design can be coherent; the current silent status rewriting is not.

For a paper whose obligation includes clamp behavior but does not require saturation-freedom, declare no-saturation optional prospectively and retain the historical configuration in provenance. Do not edit old run configurations after the fact to suggest an obligation was executed or never requested. No new robustness proof is logically required solely to waive an experimentally optional, non-load-bearing stronger property, but reports must state what actually happened.

### The transfer preconditions are records, not a certificate checker

The current layer ladder checks more than the prompt's seven items: contracts, fidelity declaration, chaining, non-degraded soundness, no-saturation-if-needed, derived margin, vacuity, and tightened-bound safety, plus the continue-on-unknown exclusion ([experiment_summary.py:452](../../src/reports/experiment_summary.py#L452)).

- Fidelity is a hard-coded renderer declaration ([experiment_summary.py:268](../../src/reports/experiment_summary.py#L268)).
- Chaining consumes `all_ok` and the enforcement setting; the underlying integer computations are useful evidence, but the report does not replay them ([experiment_summary.py:238](../../src/reports/experiment_summary.py#L238)).
- Soundness rejects a small list of labels; a missing label is not itself treated as degraded ([experiment_summary.py:231](../../src/reports/experiment_summary.py#L231)).
- Tightening checks recorded status, arithmetic safety, and shared-format flags; if the feature is absent/disabled it returns true ([experiment_summary.py:275](../../src/reports/experiment_summary.py#L275)).
- The derived-margin guard is conditional on the recorded mode ([experiment_summary.py:287](../../src/reports/experiment_summary.py#L287)).
- Disabled vacuity checking returns true rather than independently proving nonemptiness ([experiment_summary.py:322](../../src/reports/experiment_summary.py#L322)).
- Source `INCONCLUSIVE` is blocked, but the report is not a standalone verifier of a positive source certificate, input-domain coverage, tensor hashes, or compiler fidelity ([experiment_summary.py:340](../../src/reports/experiment_summary.py#L340)).

These are not evidence of fabricated records in the inspected runs. They mean the trust boundary includes the orchestrator and report construction; a badge cannot replace the proof premises.

## 8. Q6: what fidelity testing establishes

The exact text returned by `render_arith_kernel()` is inserted into both deployment and harness sources. It implements MAC, signed clamp, and integer half-away rounding ([arith_kernel.py:4](../../src/verification/arith_kernel.py#L4), [c_qnn_generator.py:100](../../src/backends/c_qnn_generator.py#L100)). The surrounding programs remain separate renderers. Input scale, bias position, activation order, array indexing, and parameter selection can drift even while those helpers remain shared. Consequently, "they cannot drift" is too strong.

`python_c_exact_match` compares **all final integer logit coordinates for each evaluated sample**, not just predicted labels or approximate real logits. Both paths apply the same Python input quantizer, so this comparison does not independently test the preprocessing/region boundary ([c_qnn_generator.py:240](../../src/backends/c_qnn_generator.py#L240), [c_qnn_generator.py:249](../../src/backends/c_qnn_generator.py#L249)). The sample set is selected by the comparison split and limit ([pipeline.py:885](../../src/synthesis/pipeline.py#L885), [pipeline.py:700](../../src/synthesis/pipeline.py#L700)). All 161 current campaign experiment summaries record exact agreement, which is empirical evidence, not universal equivalence.

Existing tests already compile the generated C, compare a toy network, exercise 200 random primitive cases, and check rounding/clamp boundaries ([test_c_qnn_generation.py:30](../../src/tests/test_c_qnn_generation.py#L30), [test_c_qnn_generation.py:56](../../src/tests/test_c_qnn_generation.py#L56), [test_c_qnn_generation.py:126](../../src/tests/test_c_qnn_generation.py#L126)). A tiny exhaustive interval test checks attained extrema against replay ([test_e2e_invariants.py:16](../../src/tests/test_e2e_invariants.py#L16)). They are a useful foundation; they do not yet establish universal equivalence of all renderers and deployed networks.

### Differential-testing protocol

1. Enumerate every input in bounded low-dimensional integer boxes for several tiny one-, two-, and three-layer networks. Compare every layer and final integer output across an arbitrary-precision reference, generated deployed C, and instrumented generated harness transitions. Use varying adjacent fractional widths, including F_in != F_out.
2. Cover positive/negative half ties and their neighboring numerators, exact multiples, Q-bit clamp limits and one unit outside, zero/ReLU crossings, bias-induced saturation, positive/negative weights, zero weights, cancellation, and accumulation prefixes requiring more than 64 bits. Test intended rejection near the signed-128 safety limits separately.
3. For endpoint harnesses, compare their asserted lower/upper outputs with exhaustive minima/maxima. For cut-based harnesses, restrict ground-truth inputs to the exact reachable prefix or a proved relation; never demand equality with an unrestricted box after using a relational cut.
4. For each real campaign model, replay selected-domain endpoints, deterministic random lattice inputs, recorded candidate witnesses, and input-quantizer boundaries through the compiled library. Do not call all Cartesian corners enumerable for MNIST; report the sampled coverage honestly.
5. Exercise the mutation checks from Q2, verify property results against exhaustive ground truth, and save compiler/version/flags, model and C hashes, seeds, tested counts, maximum integer difference, and any witness. A runtime sanitizer build is useful for test execution, but passing it is not a universal proof.

For a universal generator-fidelity claim, additionally prove the template correspondence or reduce all transition emitters to one shared layer/network implementation and prove any optimized encodings equivalent. Differential testing can expose drift; it cannot prove its absence on all inputs.

The target here is the generated `qnn_forward_fixed` C ABI. Replacing it with a framework whose accumulators, rounding, bias scaling, zero points, or saturation differ requires a new equivalence argument or new verification. The word "declared" in the summary is appropriate only when accompanied by the exact declaration and the restriction to this backend.

## 9. Q8 and Q9: ablation validity and the precision limit

The flags are forwarded: margin cuts through `--margin-cuts` and E2E invariants through their positive/negative CLI flags ([run_article_experiments.py:675](../../src/scripts/run_article_experiments.py#L675), [run_article_experiments.py:712](../../src/scripts/run_article_experiments.py#L712)). Cuts-off is applied by `_margin_cut_bounds` and `_hidden_contract_cut_bounds` ([preqbmc.py:3449](../../src/synthesis/preqbmc.py#L3449), [preqbmc.py:3596](../../src/synthesis/preqbmc.py#L3596)). The saved reports corroborate that cuts-off has no cut queries.

E2E invariants are only emitted by the E2E renderer. The separate prefix-cut renderer does not consume `e2e_invariants` ([c_templates.py:1246](../../src/verification/c_templates.py#L1246), [c_templates.py:1428](../../src/verification/c_templates.py#L1428)). With E2E and fallback disabled, Arm C is a **non-discriminating ablation**. It is effectively another execution of the same proof strategy, not evidence that an active invariant optimization has no effect. It also does not disable `tighten_verified_bounds`.

The correct experimental conclusion is: "For the selected regions, disabling optional cuts preserved certificates and selected formats; for MNIST it often reduced runtime. E2E invariant effectiveness was not evaluated by the layer-only matrix." Do not claim a CEGAR effectiveness improvement when every hidden CEGAR round count is zero.

For a discriminating experiment, use the known correlated-state toy from Q2 first, then identify real-model regions near an **integer-domain** abstraction boundary using a preregistered source-eligible margin/epsilon sweep. Freeze the selected Q/I/F to compare representations. Record integer-box cardinality, rounding-cell transitions, empirical/replayed witnesses, and all unsuccessful regions. There is no principled basis to promise a particular MNIST epsilon will flip the cut ablation before measuring it.

Run E2E invariants on/off only in actual network-scope experiments with the same model, region, formats, and budgets. Those invariants are Python-computed exact-box enclosures inserted as `__ESBMC_assume`; their soundness is part of the invariant generator's trusted argument, not automatically proved by assuming them ([preqbmc.py:761](../../src/synthesis/preqbmc.py#L761), [c_templates.py:1297](../../src/verification/c_templates.py#L1297)).

The pure box abstraction does have a structural precision ceiling. The h1=h2 example proves that exact coordinatewise bounds cannot recover lost correlation. More careful endpoint arithmetic cannot remove that loss. However, "all remaining failures are necessarily box imprecision" is too strong: source eligibility, numerical preimage construction, deflation, finite precision search, resource limits, input encoding, and optional cut costs remain separate factors ([preqbmc.py:935](../../src/synthesis/preqbmc.py#L935), [preqbmc.py:1838](../../src/synthesis/preqbmc.py#L1838)).

The implementation already goes beyond pure boxes through exact-prefix validated directional relations and shared-input output queries. A richer relational abstract domain could improve precision, but it would have to model rounding and saturation soundly and control its own cost. It is a research direction, not automatically a novel contribution or a guaranteed scalability fix.

### Prior work and defensible novelty

- **QNNVerifier** already converts neural networks into C and uses ESBMC with abstract-interpretation support. Therefore "using ESBMC on fixed-point neural networks" is not a novel claim by itself. Position the specific preimage-guided synthesis and compositional certificate design against its encoding and property scope. [Authors' QNNVerifier artifact](https://github.com/HymnOfLight/QNNVerifier).
- **Giacobbe, Henzinger, and Lechner, TACAS 2020**, already study bit-precise quantized-network verification with bit-vector SMT. Avoid claiming the first exact treatment of quantization. [How Many Bits Does it Take to Quantize Your Neural Network?](https://pmc.ncbi.nlm.nih.gov/articles/PMC7480702/).
- **Henzinger, Lechner, and Zikelic, AAAI 2021**, study scalable bit-exact QNN verification and its complexity. Their results prevent equating the full problem with the cheap affine-box subproblem. [Scalable Verification of Quantized Neural Networks](https://arxiv.org/abs/2012.08185).
- **Quadapter**, in *Certified Quantization Strategy Synthesis for Neural Networks*, already uses property preimages to guide precision synthesis. Distinguish the lineage of the backward formulation from the new C-level semantics and proof obligations. This is not the unrelated GPT-2 adapter with the same name. [Authors' paper repository](https://research-repository.griffith.edu.au/bitstreams/1af885d7-e9b5-4a20-956c-0affcdebe96b/download).
- **CEG4N** combines quantization search and equivalence checking. The property differs from certifying one target class over a local region; a comparison must match property, arithmetic, and domain or clearly state the difference. [Counterexample Guided Neural Network Quantization Refinement](https://ieeexplore.ieee.org/document/10324349/).
- **Marabou and alpha-beta-CROWN** are relevant neural-network verification baselines, but loading the float ONNX model does not make their query equivalent to this integer C implementation. Any comparison must encode or soundly enclose the actual quantizer, rounding, saturation, and input preprocessing. Do not infer lack of support or relative performance without a version-specific encoding study. [Marabou's maintained artifact](https://github.com/NeuralNetworkVerification/Marabou), [alpha-beta-CROWN's maintained artifact](https://github.com/Verified-Intelligence/alpha-beta-CROWN).

These sources establish that the broad ingredients have precedent. They do not establish that the exact integration in this repository is novel or non-novel; that requires a narrower algorithmic comparison and matched experiments. No competing tool was run during this audit.

## 10. Ranked concerns

| Priority | Category | Finding and consequence |
|---|---|---|
| 1 | **Soundness-relevant domain defect** | Round-to-nearest float32 region construction can omit inputs admitted by the requested continuous/float64 region. The diagnostic proves this implication can fail. Repair coverage or establish the explicitly intended admissible-input model before an unrestricted C1 claim. |
| 2 | **Unestablished stronger theorem** | The source IEEE execution and all stored real preimages are not formally certified by the integer proof. Numerical preimages, outward target rounding, source tie handling, and budget assumptions prevent treating their labels as proofs. The exact tightened integer path has a narrower independent argument. |
| 3 | **Reporting defect** | Required-but-skipped no-saturation, missing top-level soundness, and aggregate status overriding produce contradictory paper-facing outcomes. They do not show that modeled saturation makes the integer proof unsound, but they undermine the evidence trail. |
| 4 | **Experiment-design defect** | The E2E invariant ablation changes an inactive option; hidden CEGAR never executes. These experiments cannot support the claimed contributions of those mechanisms. |
| 5 | **Overclaiming scalability/effectiveness** | The completed campaign does not show that all components improve certification. Cuts dominate cost and give no matched certificate/precision gains here; beta has a workload-dependent cost effect. |
| 6 | **Validation gap** | The full pipeline lacks demonstrated negative controls with concrete original-input membership, exact C replay, and correctly propagated final refutation statuses. Existing arithmetic/unit tests are useful but insufficient for this purpose. |
| 7 | **Trusted implementation/portability gap** | Separate renderers, preprocessing, static arithmetic guards, artifact association, and toolchain behavior remain trusted. A C reference-backend result is not universal firmware equivalence. |
| 8 | **Statistical weakness** | Nested statuses are not runs; configurations and repeated radii are not independent samples; single timings do not establish timing distributions. Historical outputs can also reflect older code. |

**The single most serious issue for an unrestricted soundness claim is item 1.** It concerns what input set was proved, not the optional no-saturation check. For a paper explicitly restricted to the encoded integer domain with that boundary premise satisfied, the strongest immediate empirical weakness is the non-discriminating invariant/CEGAR evaluation and overstated component benefit.

I did not reproduce a false certificate from an existing LADC run. I did reproduce the unsafe domain implication and the contradictory reporting, and I do not treat tests or a `derived_budget` string as a machine-checked proof of the entire orchestrator.

## 11. Q10: contribution statement and minimum remaining evidence

Suggested contribution statement, conditional on resolving the domain boundary and reporting issues:

> PreQ-BMC synthesizes one fixed-point format per layer using property-preimage proposals and certifies the selected generated C implementation over an explicitly encoded local input region. The certificate combines checked integer interval bounds for hidden affine layers, optional relational constraints validated against the exact quantized prefix, and bit-vector checks of output-class dominance. The generated verification and reference-deployment programs share arithmetic primitives for accumulation, rescaling, and saturation. Neuron blocking preserves the layer obligation and provides a configurable tradeoff between query size and solver-process overhead. Our evaluation characterizes certification outcomes and the time and memory cost of these obligations on small fully connected ReLU networks.

Do not add "all components improve effectiveness," "first bit-exact QNN verification," "universal deployment fidelity," or "monolithic verification cannot handle these networks" to that statement without additional evidence. Do not say the error budget itself is an ESBMC-proved general float/integer error theorem: ESBMC checks the instantiated integer obligations, while the quantitative budget derivation is Python/mathematical reasoning.

Minimum remaining work for this narrower claim:

1. Establish and test the input-domain bridge, including the reproduced half-ULP boundary case and raw-image normalization. Preserve the requested perturbation domain; weakening it for better results is unacceptable.
2. Make experiment status, certificate scope, and no-saturation completion consistent. Preserve the original records and explain the current partial statuses; do not silently relabel them as discharged no-saturation proofs.
3. Execute the tiny positive/negative/spurious controls and mutation/differential checks described above, including a complete fixed-candidate refutation reporting test with compiled-C replay.
4. Present the existing matched beta/cuts data by obligation family, with memory, time, calls, candidate counts, and all resource-limit outcomes. Include the new completed monolithic escalations with their actual budgets. For comparative timing claims, repeat a small matched set at least three times; report median/IQR and the small repetition count.
5. Add one genuine active-invariant experiment if retaining that contribution, and one demonstrated hidden relational-repair case if retaining a CEGAR effectiveness claim. Otherwise remove those effectiveness claims while documenting the available mechanisms.
6. Run a pure exact-Python interval baseline on fixed candidates and a direct generated-C E2E baseline on a small matched set. This exposes where BMC certification adds assurance and where symbolic search adds precision. It does not replace the ESBMC proof in the main method.
7. Report independent model/sample/epsilon denominators, integer-domain cardinality, source eligibility, and fixed versus synthesized formats. Restrict cross-tool conclusions to measured, semantically matched comparisons.

## 12. Checks performed in this audit

The following existing tests completed successfully:

```bash
PYTHONPATH=src python -m unittest \
  tests.test_c_qnn_generation \
  tests.test_fixed_point_forward \
  tests.test_e2e_invariants \
  tests.test_verified_bound_tightening \
  tests.test_output_margin_harness \
  tests.test_cex_loop \
  tests.test_experiment_summary_guarantee_level \
  tests.test_network_end_to_end_verification
```

Result: **55 tests discovered, 52 passed, 3 skipped**. The three complete network-pipeline tests require TensorFlow, which is unavailable in the current interpreter. The existing tests include mocked orchestration checks; passing them is not equivalent to rerunning the LADC experiments.

The separate input-boundary diagnostic actually compiled the generated C library and executed two real ESBMC queries: `VERIFIED` for the rounded box and `FAILED`, witness `[0]`, for the covering box. The generated C replay changed the target prediction at the omitted input. The evidence collector re-read the saved reports and generated harnesses. No expensive benchmark campaign, external-tool comparison, or production fix was performed.
