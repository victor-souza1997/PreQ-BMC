# Round-table thread

Append-only. Each turn starts with `## Turn N — <participant> (R<round>, <date>)`. Write new turns
with `roundtable.py`; don't edit old ones.

## Turn 0 — Owner (2026-09-26)

Opening the round table. Read the briefing, then address Q1 (speed) and Q2 (coverage) first.
Q3 (paper) comes next. Build on each other's turns: agree, disagree, refine. The goal is a short,
ranked list of changes to implement after the Monday code freeze, each with an expected gain and
a way to measure it.

## Turn 1 — NotebookLM (R0, 2026-09-27)

### 1. Certificate Checking for Neural-Network Verification

* **Works & Mechanisms:** In *Neural Network Verification is a Programming Language Challenge* (Cordeiro et al., 2025, Section 3.5), Marabou is highlighted: *"Proof production mechanism, supporting several piecewise-linear activation functions, was implemented on top of Marabou. The proofs produced by Marabou are checked by a proof checker implemented within Marabou."* Imandra is also deployed as an external checker. In *Marabou 2.0: A Versatile Formal Analyzer of Neural Networks* (Wu et al., 2024, Section 2.3), Marabou generates certificates based on Farkas lemma proof vectors.
* **Checking Cost:** In *Marabou 2.0* (Wu et al., 2024, Section 3), the authors state that *"Marabou was able to produce certificates for 113 out of the 180 tested benchmarks, with only mild overhead incurred by proof generation and certification."* However, Cordeiro et al. (2025, Section 3.5) note that external checkers using precise real arithmetic come *"at a price of limited performance."*

---

### 2. Bit-Precise Verification of Quantized Neural Networks

* **Deployed C Code Verification:** Yes. In *Verifying Quantized Neural Networks using SMT-Based Model Checking* (Sena et al., 2021, Sections 1 & 3) and *QNNVerifier* (Song et al., 2021, Section 1), tools verify deployed C code via ESBMC: *"QNNVerifier checks the source code of the ANN rather than its abstract mathematical model. By doing so, we can verify the actual ANN implementation..."*
* **Network Sizes & Verification Times:**
  * **Sena et al. (2021, Section 4):** Small ANNs (Iris: \\(4 \times 10 \times 3\\)) complete in seconds; Vocalic (\\(25 \times 10 \times 4 \times 5\\)) takes minutes to hours; ACAS Xu (\\(5 \times 300^6 \times 5\\)) takes minutes to hours per property.
  * **Matos et al. (2024, Section 4.4, *Counterexample Guided Neural Network Quantization Refinement*):** Monolithic ESBMC times out (1200s limit) on larger models (MNIST, CIFAR-10, ACAS Xu).
  * **QVIP** (Zhang et al., 2022, Section 5, *QVIP: An ILP-based Formal Verification Approach for Quantized Neural Networks*): Verifies integer linear models (858–894 neurons) on MNIST in 0.02s to 568s average per sample.

---

### 3. Tightening Linear Relaxations (\\(\alpha,\beta\\)-CROWN and Branch & Bound)

* **Branch Coverage:** In *Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification* (Wang et al., 2021, Sections 2.3 & 3.4), completeness is proven because: *"Branch and bound (BaB) method is widely adopted in complete verifiers... we divide the domain of the verification problem \\(C\\) into two subdomains \\(C_1 = \{x \in C, z_j^{(i)} \ge 0\}\\) and \\(C_2 = \{x \in C, z_j^{(i)} < 0\}\\)... The final bound returned by BaB is \\(\min_i \bar{f}_{C_i}\\) which represents the worst case over all subdomains... Completeness is guaranteed when all unstable ReLU neurons are split."*
* **Certified Accuracy Gains:** Wang et al. (2021, Section 4.2, Table 2) report that \\(\beta\\)-CROWN FSB achieves higher certified accuracy than single/multi-neuron relaxations (e.g., on MNIST ConvBig under \\(L_\infty, \varepsilon=0.3\\): 70.5% vs. 43.4% for CROWN/DeepPoly, 44.5% for kPoly, and 51.0% for PRIMA).

---

### 4. Reusing Proofs and Incremental Verification

* **Reported Performance & Speed-ups:** In *Verifying Quantized Neural Networks using SMT-Based Model Checking* (Sena et al., 2021, Section 4.3.2), incremental verification via SMT solver push/pop operations was evaluated on software implementations of quantized ANNs.
* **Key Sentence & Quote:** *"Interestingly, incremental verification (cf. Fig. 11c) does not improve verification time as expected. We believe this happens because the cost of deriving and storing new facts during the verification process outweighs the reduction in search space they induce since it performs various calls to the solver."* (Sena et al., 2021, Section 4.3.2). Thus, no speed-up was achieved; overhead increased instead.
* **Proof Transfer & Generic Checker Verification:** The sources do not cover proof transfer across inputs/output classes or verifying a generic checker once to run on concrete data (**not in sources**).

---

### 5. Reporting Metrics and GTSRB Benchmarks

* **Reporting Inconclusive Instances & Sample Sizes:** In *How Many Bits Does it Take to Quantize Your Neural Network* (Giacobbe et al., 2020, Section 5.3), robust accuracy is reported on test subsets: *"Accuracy of the MNIST classifiers on the 244 test samples for which all quantization levels could be check within the given time budget."* In *Integer-arithmetic-only Certified Robustness for Quantized Neural Networks* (Lin et al., 2021, Section 4), Certified Percentage and Certified Accuracy are evaluated with binomial \\(p\\)-value confidence intervals (\\(99.9\%\\) confidence, \\(\alpha=0.001\\)).
* **Pre-registration:** **not in sources**.
* **GTSRB Benchmarks:** In *The Fourth International Verification of Neural Networks Competition (VNN-COMP 2023)* (Brix et al., 2023, Section 4.6), the "Traffic Signs Recognition" benchmark is based on GTSRB: *"We trained and tested various BNN architectures using the German Traffic Sign Recognition Benchmark (GTSRB) dataset."* *Safety Verification of Deep Neural Networks* (Huang et al., 2017, Section 5.1) also evaluates DLV on GTSRB.

---

### 6. Resource Comparisons of Quantized vs. Floating-Point Networks on Embedded Boards

* **Metrics & Measurement Methods:**
  * **Hashemi et al. (2016)** (*Understanding the Impact of Precision Quantization on the Accuracy and Energy of Neural Networks*, Sections IV & V): Measure energy per image inference (\\(\mu\text{J}\\)), design area (\\(\text{mm}^2\\)), power consumption (\\(\text{mW}\\)), and memory footprint (\\(\text{KB}\\)) using Caffe/Ristretto for accuracy and Synopsys Design Compiler / CACTI hardware models for circuit area and power. Quote: *"Fixed-Point (8,8) [achieves] 79.94% power saving and 84.06% area saving... fixed-point 16-bit saves 58.36% energy per image."*
  * **Gholami et al. (2021)** (*A Survey of Quantization Methods for Efficient Neural Network Inference*, Section II): Measure operation energy (\\(\text{pJ}\\)) and area (\\(\mu\text{m}^2\\)). Quote: *"For example, performing INT8 addition is \\(30\times\\) more energy efficient and \\(116\times\\) more area efficient as compared to FP32 addition."* They evaluate ARM Cortex-M microcontrollers using the CMSIS-NN library with fixed-point power-of-two bit-shifting.
  * **Al-Shaikh et al. (2023)** (*A Comparative Study on the Performance of 64-bit ARM Processors*): Measure Energy-Delay Product (\\(\text{EDP} = \text{Energy} \times \text{Performance}\\)) and load power (\\(\text{mW}\\)) on Raspberry Pi 3B+ / ARM Cortex-A55.

## Turn 2 — GPT-6 Sol (R1, 2026-09-27)

# GPT-6 Sol — Round 1

## 1. Re-shape, do not weaken, affine ESBMC obligations (Q1)

**What.** After the freeze, run a validation-only renderer tournament over affine chunk size, flat versus balanced partial sums, scalar temporaries versus fixed loops, and z3/bitwuzla/boolector. Keep one ESBMC verdict per generated obligation and choose by summed affine CPU subject to memory. The present renderer emits a flat multiplication expression per column and chunks by a 25,000-term default (`src/verification/crown_chain_v2.py:728`, `src/verification/crown_chain_v2.py:766`, `src/verification/crown_chain_v2.py:1094`).

**Soundness.** **ASSUMED:** the proposed renderers preserve the contract and arithmetic. **PROVED:** the existing aggregator requires every record VERIFIED and rejects unsafe flags (`src/scripts/aggregate_crown_chain_v2.py:104`, `src/scripts/aggregate_crown_chain_v2.py:130`). Each alternative must pass the 14 corruption classes.

**Expected gain.** **GUESS:** 1.5–2.0x on affine CPU would imply 1.3–1.6x end-to-end by the measured 72% affine share; batching alone is not the experiment.

**Cost.** **GUESS:** 10–16 h.

**Validation pilot.** Three fixed validation certificates, stratified by affine term count; accept only a configuration with at least 1.5x affine speed, zero non-VERIFIED obligations, and peak/job at most 1.1 GB.

## 2. Optimize integer ReLU slopes, then split only unresolved inputs (Q2)

**What.** First replace the single closed-form slope choice with competitor-specific coordinate/gradient search over integer slopes. The current finder fixes `slope = round(a*u/(u-l))` (`src/scripts/crown_finder/crown_chain_export.py:50`). If that still leaves an open competitor, branch the integer input box on one selected byte and export a partition manifest whose leaves cover the original box.

**Soundness.** **PROVED:** slopes remain proposals: the checker independently recomputes affine coefficients and validates every ReLU class/constant (`src/verification/crown_chain_v2.py:404`, `src/verification/crown_chain_v2.py:468`). For splitting, VERIFIED requires every leaf VERIFIED plus an ESBMC-VERIFIED disjoint-cover obligation; no timeout counts. **PROVED (literature):** Beta-CROWN establishes completeness of exhaustive branching in its setting (Wang et al., 2021, §§2.3, 3.4, NotebookLM Turn R0), but our integer/clamped composition must be proved independently.

**Expected gain.** **GUESS:** optimized slopes offer the best coverage/hour; splitting trades approximately linearly more checking for residual cases. The literature's 70.5% versus 43.4% example is context, not a forecast (Wang et al., 2021, Table 2, NotebookLM Turn R0).

**Cost.** **GUESS:** 18–28 h for slope search; 16–24 h for sound partition certificates.

**Validation pilot.** Lock 30 validation failures; report additional finder-positive images and certificate/check cost. Proceed to splitting only if slopes recover at least 20% relative positives; cap leaves at four and require complete union verification.

## 3. Content-addressed reuse of identical ESBMC obligations (Q1)

**What.** Canonicalize assertion labels, key each harness by source bytes plus ESBMC binary/version, profile, flags, arithmetic-kernel hash and deployment hash, and reuse only a stored VERIFICATION SUCCESSFUL record. The aggregator already re-hashes every harness (`src/scripts/aggregate_crown_chain_v2.py:131`). This targets exact repeats across shards/images, not “similar” chains.

**Soundness.** **ASSUMED:** replay of an identical closed ESBMC query preserves its verdict; SHA-256 collision resistance and provenance storage enter the trust base. Any key mismatch reruns ESBMC. Incremental solving is not the same mechanism and did not help Sena et al. (2021, §4.3.2, NotebookLM Turn R0).

**Expected gain.** **GUESS:** unknown until weighted duplicate analysis; reject if reusable hits represent under 15% of affine CPU.

**Cost.** **GUESS:** 12–18 h.

**Validation pilot.** Canonicalize five validation certificates without solving, compute exact hit rate weighted by recorded obligation time, then replay cache versus fresh results and inject wrong-key/wrong-log mutations.

## 4. Minimum publishable evaluation matrix (Q3)

**What.** Report all 80 primary outcomes with the **MEASURED** 1/80 certified rate and a binomial interval; show the disclosed rerun in a separate column, never replacing primary data. Include float/deployed accuracy, monolithic ESBMC and IBP failures, screen/box/check ablations, obligation-time/RSS distributions, and VIM3 model bytes, latency, energy/inference and EDP under matched workloads. Quantized-resource metrics follow Hashemi et al. (2016) and Al-Shaikh et al. (2023) (NotebookLM Turn R0).

**Soundness.** **PROVED:** the registered rule counts every non-VERIFIED outcome as not certified (`experiments/ssv2026_test_protocol.json:98`, `experiments/ssv2026_test_protocol.json:108`). **ASSUMED:** board measurements do not affect proof claims.

**Expected gain.** **GUESS:** reviewer credibility, not certification speed.

**Cost.** **GUESS:** 12–20 h excluding already-running checks.

**Validation pilot.** Freeze table schemas now; reproduce every table from manifests and disclose sampling, class imbalance, weak worst-class accuracy, ε=1, and workstation instability.

## 5. Publish the trust contract and reject a compiled generic checker for the headline claim (Q4)

**What.** Keep a generic compiled checker only as an exploratory secondary result. For the headline, ESBMC must still judge certificate-specific obligations. A once-proved checker followed by native execution adds the C compiler/runtime and proof-to-binary correspondence as trusted components and weakens the plain meaning of “ESBMC is the final judge.”

**Soundness.** **ASSUMED:** this is the conservative interpretation of constraint 1. The paper must enumerate Python parsing/composition already in the TCB; the current summary acknowledges parsing, hashing and composition (`src/scripts/aggregate_crown_chain_v2.py:185`). Marabou proof checking supports the general architecture, not this trust transfer (Wu et al., 2024, §2.3; Cordeiro et al., 2025, §3.5, NotebookLM Turn R0).

**Expected gain.** **GUESS:** avoids the strongest soundness objection; zero runtime gain.

**Cost.** **GUESS:** 6–10 h for a trust diagram, threat table, and added omission/reordering/cache/partition corruption cases.

**Validation pilot.** Every mutation must fail closed; independently trace each final margin through all roots, facts, hashes and ESBMC successes.

## Turn 3 — GPT-5.6 Sol (R1, 2026-09-27)

# GPT-5.6 Sol — Engineer proposal

## 1. Q1: factorial pilot on affine harness shape, size and solver

1. **What.** After the freeze, add a validation-only microbenchmark over stratified `affine_chunk` harnesses: flat versus balanced/scalar expressions, term caps 1k/5k/25k, and Z3 versus Bitwuzla. The present renderer emits one flat literal expression and three assertions per column (`src/verification/crown_chain_v2.py:744-805`); the checker already exposes term cap and profile (`src/scripts/check_crown_chain_v2.py:26-64`).
2. **Soundness.** **PROVED:** unchanged integer identities still go to ESBMC; retain bounds/division/pointer checks, which the runner explicitly warns not to disable (`src/verification/esbmc.py:239-246`). No cached verdicts or omitted obligations.
3. **Expected gain.** **MEASURED:** affine chunks consume 10.1/13.9 CPU-hours (72%) (`docs/ssv2026/roundtable/BRIEFING.md:100-122`). **GUESS:** halving affine time yields 1.56x end-to-end by Amdahl, roughly 3.9 h to 2.5 h; solver choice may yield nothing.
4. **Cost.** 6–8 engineer-hours.
5. **Validation.** Use 200 fixed, term-count-stratified harnesses from validation image 3652, then its full certificate. Go only if every obligation is VERIFIED, peak is below 1.1 GiB/job, corruption controls still fail, and full wall time improves at least 1.4x. Do not increase jobs: 12 jobs produced only 1.6x and caused resource aborts (`experiments/ssv2026_test_protocol.json:314-333`).

## 2. Q2: optimize checker-visible integer ReLU slopes

1. **What.** Add alpha-CROWN-style coordinate/beam optimization over integer `a_out`, separately per root, maximizing the exact integer margin while penalizing chain count. This is not merely “competitor-specific chains”: current `choose_relu` already receives batched per-root coefficients but makes one fixed local choice (`src/scripts/crown_finder/crown_chain_export.py:50-58`).
2. **Soundness.** **PROVED:** the finder remains untrusted; export the chosen integers and let existing ESBMC ReLU harnesses validate each relaxation. Preserve the exporter’s int64 headroom rejection (`src/scripts/crown_finder/crown_chain_export.py:44-47`).
3. **Expected gain.** **GUESS:** highest coverage-per-hour option, but no numeric recovery claim before validation. The current fixed rule leaves 54/80 test cases finder-inconclusive (**MEASURED**, `docs/ssv2026/roundtable/BRIEFING.md:124-127`).
4. **Cost.** 16–24 hours.
5. **Validation.** Freeze a validation cohort containing positives and inconclusives. Go if at least 20% of validation inconclusives become positive, no former positive regresses, and median obligations/time among positives grow by at most 2x. Tune nowhere else.

## 3. Q2 fallback: bounded, exhaustive input splitting

1. **What.** For slope failures only, split one influential byte interval into disjoint integer leaves, cap at four leaves, certify every leaf, and add an ESBMC-checked/aggregator-checked partition manifest.
2. **Soundness.** **PROVED if implemented as specified:** VERIFIED requires every leaf VERIFIED and their union exactly equals the original encoded box; TIMEOUT/UNKNOWN invalidates the image. Arithmetic and encoder remain unchanged. Wang et al. (2021, §§2.3, 3.4) establish BaB completeness when all branches are covered (NotebookLM pack, `docs/ssv2026/roundtable/THREAD.md:32-35`). Main risk is a coverage/endpoint bug.
3. **Expected gain.** **GUESS:** smaller than slope optimization per engineering hour and potentially up to 4x checking cost.
4. **Cost.** 20–30 hours.
5. **Validation.** Ten fixed validation failures; go if at least two close within four leaves, all mutation tests catch gaps/overlaps, and median total check time is at most 2x the unsplit estimate.

## 4. Q3: minimum paper evaluation package

1. **What.** Report the pre-registered 1/80 primary unchanged, with an exact binomial interval and counts for misclassified/finder/check inconclusive; show the resource-only rerun adjacent and explicitly non-replacing. Include float versus deployed accuracy/model bytes, latency, peak RSS, and energy on identical Khadas inputs; verifier wall/CPU/RSS and obligation-kind breakdown; IBP, monolithic ESBMC, and current CROWN baselines.
2. **Soundness.** **PROVED:** preserves the one-use test rule; all new tuning is validation-only. Lin et al. (2021, §4) use binomial confidence intervals (NotebookLM pack, `docs/ssv2026/roundtable/THREAD.md:47-50`); Hashemi et al. (2016, §§IV–V) motivate energy/image and memory metrics (`docs/ssv2026/roundtable/THREAD.md:55-60`).
3. **Expected gain.** **GUESS:** no certified-rate gain; materially stronger reviewer defensibility.
4. **Cost.** 10–14 hours.
5. **Validation.** Rehearse the complete table from validation plus image 3652; require scripts to preserve denominators and labels and repeated board measurements to report dispersion.

## 5. Q4: do not adopt the compiled generic checker for this deadline

1. **What.** Keep fresh ESBMC obligations; extend corruption tests for new slope and partition fields instead.
2. **Soundness.** **ASSUMED:** compiling and executing a once-proved generic checker adds compiler/correspondence trust, while the literature pack reports no proof-transfer treatment (`docs/ssv2026/roundtable/THREAD.md:39-43`). The current aggregator requires every shard VERIFIED, matching checker sources, hashes and deployment (`src/scripts/aggregate_crown_chain_v2.py:103-137`).
3. **Expected gain.** **GUESS:** zero speed; avoids the largest new trust-base/reviewer risk.
4. **Cost.** 6–8 hours for added negative tests.
5. **Validation.** On validation certificates, every clean case must VERIFY and every single-field corruption must FAIL on the named assertion, matching the existing criterion (`src/scripts/regress_crown_chain_v2_data_corruption.py:258-283`).

## Turn 4 — Claude Opus 5.5 (R1, 2026-09-27)

### What a VERIFIED rests on today

| Component | Decided by | Label |
|---|---|---|
| Box bounds on the deployed loop (`qnn_conv_native.c:87`), symbolic over the ε-box | ESBMC | PROVED |
| ReLU instances, class lemmas, rounding lemma | ESBMC + solver | PROVED |
| Affine, close, concretize steps | ESBMC constant folding, no solver | PROVED (ground evaluation) |
| Chain weights = deployed C (`crown_chain_v2.py:144-251`: regex plus a Python conv-index model) | Python | ASSUMED |
| Coverage, citations, margin closure (`:520-636`); column support (`:92-99`) | Python | ASSUMED |
| No `__int128` wrap (harnesses run without `--overflow-check`) | unstated | ASSUMED |

The 14-case regression mutates certificate *data*. It cannot catch a renderer or aggregator that
*omits* an obligation.

### Measured cost structure

MEASURED from the ESBMC logs of all 58,695 harnesses of image 3652:

- **44,251 harnesses never reach a solver.** They include every affine_chunk, affine_close,
  relu_close, concretize and chain_close, and 1,015 of the 15,430 relu_chunks; each prints
  `0 remaining after simplification`.
- **Of the 13.9 h:** 8.5 h (61%) is the front end (parse and GOTO creation), 2.4 h is symex, and
  0.76 h (5.5%) is SMT solving, all of it in relu_chunk.
- **affine_chunk:** 6.1 h front end, 2.3 h symex, 0 s solver. Peak memory: median 313 MiB,
  maximum 1,090 MiB. The 1.1 GB per-job sizing is set by the largest chunks at
  `max_affine_terms = 25000`.
- **Consequence:** a solver sweep cannot touch 72% of the time.
- **Test run, 2026-09-27:** the checker process itself peaks at about 2.7 GB while it renders a
  shard. That, with 4 ESBMC jobs and the editor, tripped the 6 GiB guard again.

### Proposals, ranked

**1. Cheaper ground harnesses; no change to the trust base.**
- *What:*
  - (a) Lower `max_affine_terms` (e.g. 8,000), so that the peak per job drops and more jobs fit.
  - (b) Render each chunk as `static const` arrays with a read-only loop and a scalar
    accumulator, instead of one 25k-term expression per column. Reads only: we measured that
    array writes make symex quadratic.
- *Soundness:* the same assertions.
- *Gate:* identical verdicts on all 15,792 affine harnesses of image 3652, and the regression
  passes.
- *Gain:* GUESS; (a) up to 2× from more jobs, (b) 2–4× on the 6.1 h front end.
- *Hours:* (a) 2, (b) 6.
- *Measure:* re-render layer 2 of 3652; wall time, peak RSS, verdicts.

**2. The trust-base question: a "verified generic checker, run natively" is REJECTED as a
replacement.**
- *Why:*
  - For ground obligations ESBMC is already a C interpreter.
  - "Prove the checker once" is vacuous for `g == Σ w·c`: the specification *is* the code, so
    ESBMC could prove only the absence of undefined behaviour and overflow.
  - Native execution swaps ESBMC for gcc and the CPU. That is a change of trust, not a proof,
    and it violates constraint 1.
- *Allowed:* as an untrusted pre-filter.
- *If the owner accepts it anyway:* a named trust tier, an ESBMC proof of no undefined
  behaviour or overflow over the envelope, and 15,792/15,792 differential agreement.
- *Gain:* 10.1 h down to minutes (GUESS). *Hours:* 8.

**3. Bind the chain weights to the deployed loop in ESBMC.**
- *What:* a harness per model (not per image), asserting
  `acc_deployed(row, x) == Σ csr_row·x` for nondet Q16 x over each row's receptive field.
- *Gain:* removes an ASSUMED row; no speed.
- *Hours:* 6.
- *Measure:* whether Z3 decides rows of up to 768 terms (layer 4) within 60 s (GUESS: uncertain).
- *Fallback:* disclose Python binding as trusted.

**4. Check the overflow argument.**
- *Argument:* |a|, |a_out| ≤ 2³¹ is asserted (`crown_chain_v2.py:878`). MEASURED from the
  deployed C: |w| ≤ 1,349 and fan-out ≤ 1,728, so |g| < 2⁵³. It is argued only on paper.
- *What:* one ground assert per affine_close, `max|w|·max|c|·fan_in < 2¹²⁶`, or
  `--overflow-check` on the ground harnesses.
- *Hours:* 1. *Gain:* GUESS: about 0 s, since it is constant-folded.

**5. An omission regression.**
- *What:* drop a column, a harness or a citation; the aggregator must refuse each.
- *Hours:* 2.

### Q2: gate for coverage proposals

Input splitting, or α/β-CROWN splits (Wang et al. 2021, R0 §3), need three things:
- an ESBMC coverage lemma: any x in the ε-box lies in some branch;
- branch-tagged facts, with no citations across branches;
- neuron splits on raw `r`, assumed only inside their own branch, with both branches present.

Every branch pays close to the full check cost, so Q1 first.

### Q3: the paper

- **Primary result: 1/80, Wilson 95% CI [0.2%, 6.8%].** Lin et al. 2021 (R0 §5) use binomial
  CIs; pre-registration is "not in sources", so it is a rigour point we can claim.
- **The disclosed rerun:** a separate table, with `aborted_low_memory` as the recorded cause.
- **Expected reviewer attacks:**
  - *"ESBMC is a calculator":* the decision-mode split above.
  - *"Aborts hide failures":* SIGTERM is recorded as UNKNOWN, and the primary result stays as
    recorded.
  - *"Python is trusted":* proposals 3 and 5.
- **Resources:** QNN versus float, memory, latency and energy on the VIM3 (Hashemi et al. 2016;
  Al-Shaikh et al. 2023, R0 §6).

### Q4: what is missing

- "58,695 ESBMC obligations" overstates the symbolic work: report 44,251 ground and 14,444
  solver-backed.
- The Python rows of the trust table are absent from the paper framing.

## Turn 5 — NotebookLM (R2, 2026-09-27)

**1. SUPPORTED**
* **Citation:** Wang et al. (2021), Sec 2.3, 3.4, App A.3 (*Beta-CROWN*).
* **Deciding Quote:** *"The process terminates when all subdomains are verified. The completeness is guaranteed when all unstable ReLU neurons are split."* (Sec 2.3) / *"Branch and bound (BaB) with β-CROWN is sound because for each subdomain \\(C_i\\) ... [Theorem 3.1 yields] a sound lower bound."* (App A.3)
* **(a)** Yes, it assumes exact real-valued arithmetic/LP relaxations over continuous domains \\(\mathbb{R}^d\\).
* **(b)** It splits unstable ReLUs (Sec 2.3/3.4); input domain splitting is noted as being limited by high input dimensionality (Sec 5).
* **(c)** Yes, Appendix A.3 provides a formal soundness and completeness proof showing that \\(\beta\\)-CROWN computes valid lower bounds \\(f_{C_i}\\) for all subdomains \\(C_i\\).

---

**2. SUPPORTED**
* **Citation:** Wang et al. (2021), Sec 4.2, Table 2 & App C.1.
* **Deciding Quote:** *"MNIST CNN-A-Adv | CROWN/DeepPoly: 43.4% | β-CROWN FSB: 70.5%"* (Table 2).
* **Details:** Confirmed. Model: `CNN-A-Adv` (ConvBig) on MNIST; perturbation radius \\(\epsilon = 0.3\\). Evaluated on 1,000 test images with a 1-hour total timeout threshold per benchmark (average solver time for \\(\beta\\)-CROWN FSB is 21.1s vs PRIMA 135.9s and SDP-FO >20h).

---

**3. SUPPORTED**
* **Citation:** Wang et al. (2021), Sec 2.2, 4.2 & App C.2 (Table A4).
* **Deciding Quote:** *"Xu et al. showed that it is possible to optimize the slope of the lower bound, \\(\alpha\\), using gradient ascent, to further tighten the bound (sometimes referred to as \\(\alpha\\)-CROWN)."* (Sec 2.2)
* **Gains & Networks:** Compares fixed lower-bound slope CROWN vs optimized slope \\(\alpha\\)-CROWN across MNIST (`CNN-A-Adv`, `ConvSmall`, MLPs) and CIFAR-10 (`CNN-A-Adv`, `CNN-B-Adv`, `Base`, `Wide`, `Deep`). On MNIST `CNN-A-Adv` (\\(\epsilon=0.3\\)), certified accuracy increases from 43.4% (CROWN) to 45.0% (\\(\alpha\\)-CROWN) and 70.5% (\\(\alpha,\beta\\)-CROWN).

---

**4. PARTLY**
* **Citation:** Sena et al. (2021), Sec 4.3.2.
* **Deciding Quote:** *"Interestingly, incremental verification (cf. Fig. 11c) does not improve verification time as expected. We believe this happens because the cost of deriving and storing new facts ... outweighs the reduction in search space..."*
* **Sub-Qs:** Sena et al. confirm incremental BMC via SMT push/pop failed to save time due to solver overhead. Caching/sharing verification results or proof transfer (e.g., Fischer'22) is **not in sources**.

---

**5. SUPPORTED**
* **Citation:** Lin et al. (2021), Sec 3.1 & Sec 4 (Alg 1/2).
* **Deciding Quote:** *"Let \\(p_A\\) be the one-sided \\((1-\alpha)\\) lower confidence interval for \\(\text{Binomial}(N_2, n_2)\\)... if \\(p_A > 1/2\\) then Return \\(\hat{c}_A\\) ... else ABSTAIN."* (Alg 1)
* **Details:** One-sided Binomial lower confidence bound (Clopper-Pearson); confidence level \\(1-\alpha = 0.999\\) (\\(\alpha = 0.001\\)); sample size \\(N_1\\) inferences for selection, \\(N_2\\) for certification. Undecided instances return `ABSTAIN` and are explicitly excluded from Certified Percentage/Accuracy.

---

**6. SUPPORTED**
* **Citation:** Al-Shaikh et al. (2023, p. 107), Hashemi et al. (2016, Sec IV–V), Gholami et al. (2021, Sec II).
* **Deciding Quote:** *"The Energy-Delay Product (EDP) was computed as the product of energy by performance."* (Al-Shaikh et al., 2023, p. 107) / *"Fixed-Point (8,8) [achieves] 79.94% power saving ... 16-bit saves 58.36% energy per image."* (Hashemi et al., 2016).
* **(a)** Al-Shaikh et al. explicitly state "product of energy by performance".
* **(b)** Hashemi et al. use hardware CAD models (Synopsys Design Compiler 32nm & CACTI), whereas Al-Shaikh et al. and Gholami et al. measure on physical ARM boards (Raspberry Pi 3B+/4).
* **(c)** Gholami et al. (2021) compares ARM latency/memory/energy of INT8 vs FP32 on Raspberry Pi 4 (1.40\\(\times\\) speedup) and Cortex-M microcontrollers.

---

**7. SUPPORTED**
* **Citation:** Wu et al. (2024), Sec 2.3 & 3; Cordeiro et al. (2025), Sec 3.5.
* **Deciding Quote:** *"Marabou produces proof certificates based on a constructive variant of the Farkas lemma... 99.99% were certified by the native proof checker, while the remaining were certified by a trusted SMT solver."* (Wu et al., 2024, Sec 2.3)
* **(a)** Marabou 2.0 uses a native C++ proof checker + trusted SMT solver ("formally verified proof-checker under development"); Imandra (Cordeiro et al., 2025) is an external proof checker in a functional language.
* **(b)** Trusted base: external checker/SMT solver and Farkas lemma mathematical proof rules.
* **(c)** Checking cost: Wu et al. report certifying 113/180 ACAS Xu benchmarks with "only mild overhead" over 1.46M proof-tree leaves; Cordeiro et al. note checking precise real arithmetic comes "at a price of limited performance".

---

**8. PARTLY**
* **Citation:** Sena et al. (2021), Sec 4.2 & Fig 10; Haihang (2020, Sec 4.1.2).
* **Deciding Quote:** *"Bitwuzla and Boolector have nearly identical performance... Yices exhibits a considerable advantage... being, in some specific cases, even two orders of magnitude faster... Z3 struggled to complete the majority of verification runs..."* (Sena et al., 2021, Sec 4.2).
* **Details:** Solvers compared: Z3, Bitwuzla, Boolector, Yices on fixed-point ANSI-C implementations of ANNs (Iris, Vocalic, Chinese Digits). **Yices won** (fastest by up to 2 orders of magnitude); Z3 failed/timed out on most queries.
* **Model checker time breakdown:** Explicitly splitting model checker time (parsing vs symbolic execution vs SMT solving) is **not in sources**.

---

**9. PARTLY**
* **Citation:** Brix et al. (2023), Sec 4.6; Huang et al. (2017), Sec 5.1.
* **Deciding Quote:** *"We trained and tested various BNN architectures using the German Traffic Sign Recognition Benchmark (GTSRB) dataset."* (Brix et al., 2023, Sec 4.6).
* **Details:** Brix et al. evaluate BNN/CNN architectures on GTSRB in VNN-COMP 2023; Huang et al. evaluate DLV.
* **Bit-precise certification:** Bit-precisely certifying a quantized GTSRB network or its deployed C-code is **not in sources** (sources only verify smaller quantized C-code models like Iris, Vocalic, MNIST, ACAS Xu).

## Turn 6 — GPT-5.6 Sol (R2, 2026-09-27)

# GPT-5.6 Sol — Round 2 cross-attack

## 1. Solver choice

> GPT-5.6 Sol: “Z3 versus Bitwuzla.”

I retract that part of my proposal. **MEASURED:** I scanned the recorded stdout/stderr paths in all five image-3652 summaries: 15,792/15,792 `affine_chunk` records say `0 remaining after simplification`. Thus solver selection cannot reduce affine time. Opus’s decomposition—6.1 h front end, 2.3 h symex, 0 solver—is consistent with the renderer: every operand is a literal and the asserted result is literal (`src/verification/crown_chain_v2.py:744-805`). Drop solver choice; retain expression-shape and chunk-size tests. Sena et al.’s solver comparison is irrelevant here because it concerns queries that reach SMT (NotebookLM check, `docs/ssv2026/roundtable/THREAD.md:338-342`).

## 2. Chunk cap and jobs

> GPT-5.6 Sol: “Do not increase jobs.”

That is too categorical; Opus is right that cap and concurrency must be tested jointly. **MEASURED:** among the 15,792 affine records, the 8,111 with at most 8,000 terms peaked at 403.9 MiB/job (99th percentile 331.6 MiB), whereas the complete set peaked at 1,090.1 MiB. The current default is 25,000 terms (`src/scripts/check_crown_chain_v2.py:37`). At 25k, memory limits job count; after shrinking, repeated front-end work and disk I/O may limit wall time.

Pilot an **8,000-term cap** first, with 4/6/8 jobs on one fixed validation shard. **GUESS:** 8k makes 6–8 jobs memory-feasible, but only measured wall time decides. Go only with all VERIFIED, no low-memory abort, MemAvailable never below 6 GiB, and at least 1.25x wall improvement. Do not deploy a higher job count independently of the cap.

## 3. Result cache

> GPT-6 Sol: “Content-addressed reuse of identical ESBMC obligations.”

**DROP. MEASURED:** of 58,695 harnesses, 58,676 hashes are unique: 19 reusable occurrences, a 0.0324% hit rate. All six duplicate groups are shared lemmas; the 15,792 affine harnesses have **zero** duplicates. Avoiding the repeats saves only 3.56 of 50,123 recorded ESBMC seconds (0.0071%). The aggregator already re-hashes each harness (`src/scripts/aggregate_crown_chain_v2.py:131-137`), but persistent verdict reuse would add key/log integrity assumptions for no material gain.

## 4. Twenty-five-hour schedule

> GPT-5.6 Sol: optimized slopes are the “highest coverage-per-hour option.”

Not before this deadline. Opus is correct that speed comes first: each new branch otherwise nearly replicates a four-hour check. My own slope estimate is 16–24 h (`docs/ssv2026/roundtable/THREAD.md:138-144`), leaving no safe validation window by 2026-10-02.

Order and budget: (1) 8 h for the 8k cap plus read-only-loop/scalar renderer and validation pilot; (2) 3 h for overflow and omission checks; (3) a hard-capped 6 h deployment-binding lemma pilot; (4) 8 h for reproducible paper tables, trust contract and board measurements. Total 25 h. **ASSUMED:** those estimates hold. Alpha slopes and all splitting move to future work; NotebookLM confirms alpha-CROWN gains on other MNIST/CIFAR models, not this integer GTSRB system (`docs/ssv2026/roundtable/THREAD.md:299-302`).

## 5. Trust gaps

> Opus: “Chain weights = deployed C … Python … ASSUMED”; “No `__int128` wrap … ASSUMED”; and “An omission regression.”

- **Weight binding — close before submission.** **PROVED (code inspection):** Python hashes the C, parses loop geometry, reconstructs every convolution row, and checks ordered indices, weights and biases (`src/verification/crown_chain_v2.py:144-251`). That is strong but remains a trusted Python interpretation. Timebox Opus’s ESBMC row-equivalence lemma at 6 h; if it fails, state the Python binding explicitly as an **ASSUMED trusted component**, not as ESBMC-proved.
- **Overflow — close before submission.** **PROVED (code inspection):** `paper-z3` does not add `--overflow-check`; only safety/overflow profiles do (`src/verification/esbmc.py:196-246`). ReLU instances enforce coefficient envelopes (`src/verification/crown_chain_v2.py:877-883`), but affine ground sums lack an explicit 128-bit headroom assertion. Add constant envelope assertions or run the overflow profile; require VERIFIED.
- **Omitted obligations — close before submission.** The aggregator already checks authoritative chain counts, unique layers, all-VERIFIED counts and harness hashes (`src/scripts/aggregate_crown_chain_v2.py:80-87`, `src/scripts/aggregate_crown_chain_v2.py:95-137`). **PROVED (code inspection):** the existing regression mutates accepted data (`src/scripts/regress_crown_chain_v2_data_corruption.py:1-7`), not omissions. Add drop-chain, drop-harness, drop-fact/citation and drop-competitor cases; each must fail closed.

## Revised top 3

1. 8k affine chunks + loop/scalar renderer, then 4/6/8-job validation sweep — 8 h.
2. Overflow and omission checks, plus timeboxed ESBMC deployment-binding lemma — 9 h.
3. Reproducible evaluation/trust tables and VIM3 measurements — 8 h; cache, slopes and splitting are out.

## Turn 7 — GPT-6 Sol (R2, 2026-09-27)

# GPT-6 Sol — Round 2

## 1. Solver choice

I retract my Turn 2 proposal to compare “z3/bitwuzla/boolector” for affine chunks, and disagree with GPT-5.6 Sol's Turn 3 “factorial pilot on affine harness shape, size and solver.” **MEASURED:** I checked the stderr logs named by the five validation `check_summary.json` files: all **15,792/15,792** affine-chunk logs contain `0 remaining after simplification`; a representative is `output/crown_chain_v2_20260924/image3652/claude_hardened_l0/esbmc/001_chain_L0_n32_lower_s0_affine_0.stderr.log:9-12`. The summaries record 15,792 successes and 10.1 affine CPU-hours. **PROVED (code):** these are literal multiplication expressions and ground equality assertions (`src/verification/crown_chain_v2.py:766-780`). Solver choice cannot reduce their SMT time of zero. Keep the measured 8.5 h front-end and 2.4 h symex breakdown from Opus Turn 4 as the optimization target; solver experiments belong only to remaining symbolic ReLU work.

## 2. Chunk size and jobs

I disagree with Opus Turn 4's “lower `max_affine_terms` ... so ... more jobs fit” as a wall-time forecast. I agree with GPT-5.6 Sol's “Do not increase jobs” for this deadline. **MEASURED:** 12 jobs gave only 1.6× throughput and caused memory aborts; the protocol records retreat from 12 to 6 to 4 (`experiments/ssv2026_test_protocol.json:311-333`). Opus measured 1,090 MiB maximum per affine job and about 2.7 GB for the rendering parent (Turn 4). **PROVED (code):** the checker renders an entire shard before launching ESBMC (`src/scripts/check_crown_chain_v2.py:42-64`), so smaller chunks create more harnesses and may increase parent load and fixed parse cost. **GUESS:** a cap of 8,000 could reduce peak/job but worsen total wall time. Pilot **25,000 versus 8,000 terms at four jobs**, with a three-job safety comparison if the 6 GiB guard fires; do not pilot more than four jobs before October 2. Gate on full validation-shard wall time, parent-plus-children RSS, minimum MemAvailable above 6 GiB, and all verdicts VERIFIED. Expression shape is the likelier speed lever.

## 3. Result cache

I retract my Turn 2 “Content-addressed reuse” proposal. **MEASURED:** streaming `harness_sha256` from all five image-3652 summaries yields **58,695 records, 58,676 distinct hashes, only 19 repeat occurrences (0.032%)**. All six repeated hashes are `__shared_lemmas__` (five ReLU kinds and rounding); **zero of 15,792 affine chunks repeat**. One repeated rounding record is identifiable in `output/crown_chain_v2_20260924/image3652/claude_hardened_l0/check_summary.json:40053-40060`. **PROVED (code):** assertion labels embed step IDs (`src/verification/crown_chain_v2.py:775-780`), so normalized cross-image caching would require new equivalence and provenance reasoning. **Verdict: DROP.** The measured within-image benefit is negligible, and a stored verdict would expand the trust base. Cross-image hits remain unmeasured, not a submission assumption.

## 4. Twenty-five-hour order

I disagree with my Turn 2 “Optimize integer ReLU slopes, then split” ordering. **MEASURED:** current check costs roughly four hours per finder-positive validation image; 54/80 test screens were inconclusive (`docs/ssv2026/roundtable/BRIEFING.md:80-85`, `:124-127`). **PROVED (literature):** the NotebookLM check reports only 43.4% to 45.0% from fixed to optimized slopes on one MNIST network, with 70.5% requiring branching (Wang et al. 2021, Table 2; `docs/ssv2026/roundtable/THREAD.md:290-302`). Those figures do not predict this QNN. **GUESS:** my 18–28 h slope estimate cannot leave time for a complete validation certificate and paper. Allocate **2 h** chunk pilot, **6 h** affine renderer, **6 h** deployment binding pilot, **2 h** overflow closure, **2 h** omission regression, **7 h** evaluation and writing = **25 h**. Pilot changes on validation by October 2. Slopes and partition certificates move to future work.

## 5. Trust gaps

I accept Opus Turn 4's objection to my Turn 2 trust table: naming Python in the TCB was insufficient. **Deployment binding: close before submission if the six-hour ESBMC equivalence pilot works; otherwise state an explicit trusted Python mapping assumption and narrow the claim.** **PROVED (code):** deployment-to-CSR equality is currently implemented by Python regex, convolution indexing and row comparison, then reported as `EXACT_ORDERED_MATCH` (`src/verification/crown_chain_v2.py:144-250`); ESBMC does not decide that correspondence. **Overflow: close before submission.** **PROVED (code):** `--overflow-check` is enabled only for `safety`/`overflow`, not the paper profile (`src/verification/esbmc.py:239-245`). Add ESBMC-checked bounds for every relevant intermediate, including 128-bit sums; a hand bound alone remains **ASSUMED**. **Omission regression: close before submission.** **PROVED (code):** the existing negative regression mutates values and requires named assertion failures (`src/scripts/regress_crown_chain_v2_data_corruption.py:245-283`), while aggregation checks records against the generated manifest (`src/scripts/aggregate_crown_chain_v2.py:103-138`). Add independent expected-obligation enumeration and delete-one-column/harness/citation cases; failure must be non-VERIFIED.

Revised top 3: **(1)** affine expression and 8k/25k pilot at four jobs; **(2)** ESBMC deployment binding plus overflow bounds; **(3)** omission regression and transparent evaluation. **GUESS:** these fit 25 h; no result is claimed before validation.

## Turn 8 — Claude Opus 5.5 (R2, 2026-09-27)

## 1. Solver choice: settled, DROP everywhere before the deadline

Both GPT turns retracted it; I agree. Any solver work is capped at relu_chunk's 0.76 of 13.9 h
(5.5%, MEASURED), so even Sena's fastest solver (Yices, Turn 5) gains at most 5%.

## 2. Chunk size: I retract my Turn 4 proposal 1(a) as a speed lever

GPT-6 is right that "lower the cap so more jobs fit" is not a wall-time forecast. New evidence,
MEASURED on all 15,792 affine records of image 3652:

| Terms per chunk | Chunks | µs per term |
|---|---|---|
| 4k–8k | 2,262 | 177 |
| 8k–16k | 1,452 | 178 |
| 16k–25k | 6,229 | 187 |

- **Cost is linear above about 4,000 terms.** Splitting a chunk saves no CPU.
- **An 8k cap adds 18,672 harnesses**, at about 0.15 s fixed cost each (the median for chunks of
  at most 100 terms). That is +0.8 CPU-h, 8% of affine time.
- **Memory:** peak per job falls from 1,090 to 404 MiB (GPT-5.6's number, confirmed).
- **Verdict:** keep the 8k cap as a memory-safety lever (fewer guard aborts, no host freezes),
  not in the speed column.

In GPT-6's citation (`check_crown_chain_v2.py:42-65`), `certificate` stays referenced while
`benchmark` runs, so the parent may hold its 2.7 GB peak (MEASURED) through every job (GUESS).
Releasing it is a 1 h change, neutral for soundness.

**Affine cost scales with Σ multiply terms** (layer 2 alone: 5.84 of 10.09 h). The real lever is
fewer terms, which is finder work for later. The shape pilot keeps the same literals, hence the
same parse volume (GUESS: at most 1.5×).

## 4. Schedule: the binding constraint is machine time, not engineer hours

Both turns budget 25 engineer-hours. Neither accounts for two facts:

- **Edits to `src/` wait for the rerun.** PROVED (the code): the driver aborts if any file under
  `src/` differs from its pin, including a new file
  (`run_crown_chain_v2_protocol.py:76-82`, `:185-198`). The disclosed rerun therefore blocks every
  `src/` edit in this checkout until it finishes. Post-freeze work goes in a separate git worktree.
- **The rerun takes days.** MEASURED:
  - Test image 005 took 11.8 CPU-h (3.4 h wall at 4 jobs); 015 took 21.6 CPU-h.
  - 12 of the 16 pending images still need layers 2–4, which hold 9.8 of 10.1 affine hours on
    image 3652.
  - GUESS: 40–80 h of wall time, ending 2026-09-29 to 10-01 if it starts tonight. No validation
    pilot can run alongside it, for lack of memory.

A renderer pilot on a full validation image cannot safely finish by Oct 2, and the pinned code
fixes the primary result anyway. **Order (about 20 h):**
1. Omission regression: 2 h, seconds of machine time; it tests the pinned aggregator as it is.
2. Overflow closure (§5): 2 h.
3. Binding lemma pilot: 6 h cap, minutes of machine time.
4. Evaluation tables and VIM3: 8–10 h, on the board.
5. Speed: the measured cost model is the scalability evidence; renderer and slopes are future work.

## 5. Trust gaps

**A label fix first.** GPT-5.6 labels the Python binding "PROVED (code inspection)", and GPT-6
labels facts about the code "PROVED (code)". Under constraint 5, PROVED means an ESBMC verdict;
reading code gives ASSUMED or MEASURED. The paper's trust table must not inherit this.

**Overflow: narrower than my Turn 4 said.**
- `+` and `×` on `__int128` are exact modulo 2¹²⁸. A ground harness asserting `Σ w·c == L`, with
  L an exact in-range literal, can falsely VERIFY only if |Σ w·c| ≥ 2¹²⁷.
- The literals rule that out: at most 25,000 terms, |w| ≤ 1,349 (MEASURED from the deployed C),
  and |c| ≤ 2³¹, so |Σ w·c| < 2⁵⁷. Floor and residual act on g after it is exact.
- **Open point:** ESBMC asserts |c| ≤ 2³¹ only on ReLU outputs (`crown_chain_v2.py:877-883`).
  Check that every affine input coefficient is such an output or a ±1 margin.
- **Closure, decided by ESBMC:** after the rerun, rerun only the VERIFIED images' ground
  harnesses with `--overflow-check`.

**Binding.** GPT-6 found `EXACT_ORDERED_MATCH` (`crown_chain_v2.py:144-250`): the CSR row is in
deployed-loop order. Render both sums of the ESBMC binding harness in that order, so that symex
yields the same term and the simplifier may discharge it without SAT (GUESS); a reordered sum is a
hard adder-equivalence query. Pilot layer 0 (75-term rows), then layer 4 (768).

**Omission.** Agreed; GPT-6's independent enumeration must not reuse the renderer's loops.

## Revised top 3

1. Omission regression, and overflow closure on the existing VERIFIED images (4 h).
2. The binding-lemma pilot in deployed-loop order, capped at 6 h; otherwise a stated trusted
   assumption.
3. Evaluation, with the measured cost model as the scalability result. Run the rerun first and
   develop in a worktree; speed work and slopes are future work.

## Turn 9 — NotebookLM (R3, 2026-09-27)

### 1. Speeding Up Bounded Model Checking and SMT Verification

* **QVIP** (*QVIP: An ILP-based Formal Verification Approach for Quantized Neural Networks*, Zhang et al., 2022, Sec 5.1–5.2): Replaces explicit `If-Then-Else` SMT constructs with Boolean variables and interval analysis. Achieves a **\\(100\times\\) speed-up ("two orders of magnitude faster")** over bit-vector SMT verifiers on MNIST QNNs (up to 894 neurons).
  * *Quote:* *"QVIP is two orders of magnitude faster than the state-of-the-art in both the single- and multiple-thread modes."* Keeps verification **sound and complete**.
* **EQV** (*Towards Efficient Verification of Quantized Neural Networks*, Huang et al., 2024, Sec 1 & 5.1): Combines integer linear programming (ILP) with Marabou bound propagation. Achieves **up to \\(78\times\\) speed-up** over QVIP on MNIST FC1-100 (\\(104\text{s}\\) vs \\(11,751\text{s}\\)).
  * *Quote:* *"avoiding the use of piecewise constraints can significantly improve the verification efficiency."* Keeps verification **sound and complete**.
* **Sena et al. (2021) / ESBMC** (*Verifying Quantized Neural Networks using SMT-Based Model Checking*, Sena et al., 2021, Sec 4.3): Frama-C interval analysis and slicing reduce ESBMC search spaces on fixed-point Iris/Vocalic networks.
  * *Quote:* *"adding the intervals computed by FRAMA-C reduces verification times of a large number of safety properties."* Keeps verification **sound**.

---

### 2. Compositional or Layer-by-Layer Verification

* **Quadapter** (*Certified Quantization Strategy Synthesis for Neural Networks*, Zhang et al., 2025, Sec 3.1 & 5.2): Computes layer-wise under-approximate preimages (\\(P_{2i} \subseteq \mathcal{N}_{[2i+1:2i+2]}^{-1}(P_{2i+2})\\)) using MILP. In the forward pass, it checks whether each quantized layer's reachable set is contained within its preimage (\\(\gamma(\hat{A}_{2i}) \subseteq P_{2i}\\)).
* **Soundness Argument:** Proposition 1 proves inductively that if every quantized layer's output stays inside its layer preimage, the global property \\(\langle I, O \rangle\\) holds soundly for the full QNN.
* **Reported Gains:** Solves synthesis across 2-to-6-layer MNIST/Fashion-MNIST networks (up to 512 neurons/layer) in minutes (\\(10\text{s}\\)–\\(300\text{s}\\) per layer), whereas monolithic verification times out.
* *Quote:* *"Proposition 1 states that regardless of the quantization configurations of the first \\(2i\\) layers, the property \\(\langle I,O \rangle\\) is always preserved in the resulting QNN, as long as the reachable region of the quantized layer \\(\hat{f}_{2i}\\)... remains within its respective preimage \\(P_{2i}\\)."* Preserves **soundness**.

---

### 3. Making Proof or Certificate Checking Cheaper

* **Marabou 2.0** (*Marabou 2.0: A Versatile Formal Analyzer of Neural Networks*, Wu et al., 2024, Sec 2.3 & 3): Emits Farkas-lemma proof certificates during DPLL(T) search, checked by a lightweight C++ proof checker or trusted SMT solver.
  * **Reported Cost:** Certified 113 of 180 ACAS Xu benchmarks with minimal overhead over 1.46M proof-tree leaves.
  * *Quote:* *"Marabou was able to produce certificates for 113 out of the 180 tested benchmarks, with only mild overhead incurred by proof generation and certification."* Soundness is preserved by independent validation.
* **Cordeiro et al. (2025)** (*Neural Network Verification is a Programming Language Challenge*, Cordeiro et al., 2025, Sec 3.5): Evaluates external proof checkers (e.g., Imandra) using exact real arithmetic.
  * *Quote:* *"The proofs produced by Marabou are checked by a proof checker... [at a] price of limited performance."*
* Specific numerical breakdowns comparing finding time vs. checking time for proof certificates: **not in sources**.

---

### 4. Problem and Network Size Reduction Before Verification

* **CEG4N** (*Counterexample Guided Neural Network Quantization Refinement*, Matos et al., 2024, Sec I & IV-B / Cordeiro 2024, Abstract): Employs Counterexample-Guided Abstraction Refinement (CEGAR) in a genetic algorithm loop to search layer bit-widths using formal equivalence verifiers (NNEQUIV/ESBMC).
  * **Reported Gain:** Achieves up to **72%** (Matos et al., 2022) and **163%** (Matos et al., 2024) accuracy gains over GPFQ quantization on small/sparse networks (Iris, Seeds, ACAS Xu). However, ESBMC suffers 67.9% timeouts on larger models (MNIST, CIFAR-10).
  * *Quote:* *"CEG4N employs an iterative optimization loop that leverages formal verification tools to systematically detect and correct discrepancies between the original and quantized models..."* Preserves **soundness**.
* **Pruning, Weight Sparsity, and Neuron Merging Verification Speed-ups:** Mentioned generally as model compression techniques (*Gholami et al., 2021*), but specific verification speed-up numbers for pruning/merging are **not in sources**.

---

### 5. Tighter Bounds Specific to Quantized/Integer Networks

* **QEBVerif** (*QEBVerif: Quantization Error Bound Verification of Neural Networks*, Zhang et al., 2023, Sec 1 & 5.2): Introduces Differential Reachability Analysis (DRA) with integer rounding inside an abstract domain tailored to QNNs. Injecting DRA bounds into MILP (`DRA + MILP + Diff`) speeds up verification from \\(1045\text{s}\\) down to **\\(2.92\text{s}\\)–\\(26.4\text{s}\\)** on ACAS Xu / MNIST.
  * *Quote:* *"We propose a novel DRA to compute sound and tight quantization error intervals accompanied by an abstract domain tailored to QNNs, which can significantly and soundly tighten the quantization error intervals."* **Sound and complete**.
* **EQV** (*Towards Efficient Verification of Quantized Neural Networks*, Huang et al., 2024, Sec 3 & 5.1): Encodes PyTorch fixed-point `Round` operations into direct ILP integer constraints. Reduces verification time on MNIST FC1-100 (\\(r=4\\)) from \\(11,751\text{s}\\) (QVIP) to **\\(104\text{s}\\)** (EQV), solving 100/100 instances.
  * *Quote:* *"avoiding the use of piecewise constraints can significantly improve the verification efficiency."* **Sound and complete**.

---

### 6. Novelty Check for Deployed GTSRB CNN Verification Claim

* **Closest Work for Deployed C Verification:** *QNNVerifier* (*Song et al., 2021*, Sec 1) and *Sena et al. (2021)* verify deployed C code bit-precisely using ESBMC, but only scale to small MLPs (Iris, Vocalic: 25-input, 4-layer). *Matos et al. (2024)* attempted C-code verification on MNIST/CIFAR via ESBMC but suffered **100% timeouts (\\(1200\text{s}\\))**.
* **Closest Work for GTSRB Dataset:** *VNN-COMP 2023* (*Brix et al., 2023*, Sec 4.6) includes GTSRB BNN/CNN benchmarks, but all participating verifiers (\\(\alpha,\beta\\)-CROWN, Marabou) evaluate **abstract mathematical models**, **not deployed C code or ESBMC**.
* **Largest Bit-Precisely Verified Network:** *EQV* (*Huang et al., 2024*, Sec 5) verified **FC2-256 (\\(1,306\\) neurons)** and CNNs on MNIST via ILP (\\(205\text{s}\\)–\\(4,019\text{s}\\)/property). *QVIP* (*Zhang et al., 2022*) verified **894-neuron** MNIST QNNs (\\(0.10\text{s}\\)–\\(204\text{s}\\)/property).
* **Novelty Assessment:** No source in the literature certifies deployed C code of a quantized CNN with \\(>90\%\\) accuracy on GTSRB using ESBMC.
  * *Quote:* *"QNNVerifier checks the source code of the ANN rather than its abstract mathematical model..."* (*Song et al., 2021*, Sec 1).

---

🔍 *Would you like to formalize any of these comparative speed-up metrics or compositional preimage theorems for your thesis's Related Work chapter?*

## Turn 10 — GPT-6 Sol (R3, 2026-09-27)

# GPT-6 Sol — Round 3 final plan

## Assigned trial: omission regression

**NOT RUN (environment refusal), not a pass.** `free -g` reported 19 GiB total and 9 GiB available, so the 8 GiB gate passed. Creation of the kickoff-authorized pilot directory was rejected by the managed approval layer after 5.2 s; therefore no copy was made, no mutation or aggregator command ran, and there were zero ESBMC/non-VERIFIED results.

Exact attempted command:

```sh
mkdir -p scratchpad/roundtable_pilots/gpt6-sol
```

I did complete the required independent read-only enumeration without calling the renderer. Layer 2 contains **32,445** `.c` harnesses; its summary contains 32,445 records and declares 32,445 harnesses; its shard contains **2,089/2,089 unique chains**, matching `chains_checked`, and **561,989** residual-fact citations. The corrected exact enumeration command took 4.2 s:

```sh
python -c 'import gzip,json; p="output/crown_chain_v2_20260924/image3652/chains_L2.jsonl.gz"; cs=[json.loads(x) for x in gzip.open(p,"rt")]; print(len(cs),len({c["certificate_id"] for c in cs}),sum(len(q.get("fact_ids",[])) for c in cs for s in c["steps"] for q in [s.get("residual_lower_sources")] if isinstance(q,dict)))' && python -c 'import json; d=json.load(open("output/crown_chain_v2_20260924/image3652/claude_hardened_l2/check_summary.json")); print(len(d["records"]),d["harness_count"],d["chains_checked"])'
```

Static inspection remains **ASSUMED**, not a trial result: aggregation checks declared counts and each present record, but does not compare `len(records)` with `harness_count` (`src/scripts/aggregate_crown_chain_v2.py:103-138`). The deletion trial remains mandatory.

## Surviving changes

| change | trust-base change | expected gain | hours | validation pilot | main risk |
|---|---|---|---:|---|---|
| Independent omission regression: delete one harness record, required chain, citation, and competitor | none | **GUESS:** prevents incomplete evidence from being accepted; current 14 cases mutate values, not omissions (`src/scripts/regress_crown_chain_v2_data_corruption.py:1-7`) | 2 | Baseline aggregates; every deletion must be refused. Enumeration must scan artifacts independently of renderer/checker | An “independent” enumerator may duplicate the same bug |
| Close arithmetic overflow with `--overflow-check` on ground obligations | none | **GUESS:** no speed gain and small overhead because affine chunks already simplify to zero solver goals; removes an unstated assumption | 2 | 300 largest validation affine chunks: 300 VERIFIED, zero timeout/failure; report time ratio | Hidden coefficient path exceeds the argued envelope |
| Bind deployed-loop rows to CSR rows with ESBMC, in identical term order | a new ESBMC lemma | **GUESS:** removes Python mapping from the headline trust claim; current equality is Python parsing/indexing (`src/verification/crown_chain_v2.py:144-251`) | 6 | Representative layer-0 and layer-4 rows VERIFIED within 60 s each; otherwise disclose Python as **ASSUMED** | Reordered sums become hard bit-vector equivalence queries |
| Publish reproducible evaluation tables and explicit trust contract | none | **GUESS:** reviewer credibility; **MEASURED:** 1/80 primary, 17 resource-inconclusive, 54 finder-inconclusive, 8 misclassified (`docs/ssv2026/roundtable/BRIEFING.md:124-136`) | 8–10 | Rebuild tables from manifests; denominators/statuses must match, and VIM3 repetitions report dispersion | Board energy measurement or rerun misses deadline |
| Use 8k affine cap only as memory safety; release the rendered certificate before workers | none | **MEASURED:** observed peak/job 1,090→404 MiB, but estimated +0.8 CPU-h from extra harnesses (Opus Turn 8) | 2 | After rerun: 25k versus 8k at four jobs; go only if all VERIFIED, no guard abort, and wall penalty ≤10% | Parent memory persists or extra parsing outweighs safety |

## Rejected proposals

- Solver sweep: DROP; 15,792/15,792 affine chunks have zero remaining solver goals.
- Result cache: DROP; only 19/58,695 reusable records (0.032%), none affine.
- Alpha slopes and input/ReLU splitting: FUTURE WORK; no complete validation pilot fits before October 2.
- Compiled generic checker: DROP for the headline; it substitutes compiler/runtime trust for ESBMC.
- Affine expression-shape rewrite and >4 jobs: FUTURE WORK; no safe full validation run while the disclosed rerun occupies the machine.
- QVIP/EQV as final judges: DROP under constraint 1; discuss as related work or future untrusted finders.

## Q3: minimum convincing evaluation

Report the immutable pre-registered **1/80** primary with a named 95% binomial interval and the full outcome taxonomy. Put the disclosed resource-only rerun in an adjacent, separately labelled column—never as replacement. Include float/deployed accuracy (91.32%/91.35%), worst-class 44.7%, exact ε=1 property, model bytes, matched VIM3 latency/energy/EDP, and verifier wall/CPU/RSS distributions. Separate **44,251 ground** from **14,444 solver-backed** obligations. Baselines are monolithic ESBMC, IBP, current CROWN, and published deployed-C/GTSRB work; do not transplant QVIP/EQV speedups to this model.

Three strongest attacks and answers:

1. **“1/80 at ε=1 is too weak.”** Answer: scope the contribution to bit-precise deployed-C certification, report the interval and every negative outcome, and make no broad robustness claim.
2. **“ESBMC is mostly a calculator; Python is trusted.”** Answer: disclose the ground/symbolic split and trust table; add overflow and binding lemmas plus omission regressions, or explicitly retain and name the Python assumption.
3. **“Resource aborts/reruns cherry-pick.”** Answer: the primary remains unchanged; SIGTERM is UNKNOWN, reruns are separate, all method tuning uses validation only, and every VERIFIED requires all ESBMC obligations successful.

## Turn 11 — GPT-5.6 Sol (R3, 2026-09-27)

# GPT-5.6 Sol — Round 3 final plan

## Assigned trial: overflow

The runner selected the 300 `affine_chunk` records with greatest `multiply_terms` across image 3652's five summaries (24,998–25,000 terms). It invoked one ESBMC process at a time, checked `free -g` before every invocation, required at least 8 GiB available, changed the recorded `--memlimit` to `2g`, and appended `--overflow-check`.

Exact outer command:

```text
python scratchpad/roundtable_pilots/gpt56-sol/run_overflow_trial.py
```

Exact first ESBMC command (all 300 are recorded in `scratchpad/roundtable_pilots/gpt56-sol/selection.json`):

```text
/home/joao/esbmc-linux/bin/esbmc /home/joao/code/preqbmc/output/crown_chain_v2_20260924/image3652/claude_hardened_l2/harnesses/chain_00122/chain_L2_n535_lower_s4_affine_0.c --function main --unwind 2 --z3 --bv --timeout 60 --memlimit 2g --interval-analysis --interval-analysis-simplify --result-only --overflow-check
```

**PROVED:** 300/300 returned VERIFICATION SUCCESSFUL; zero FAILED, TIMEOUT, MEMOUT or UNKNOWN (`scratchpad/roundtable_pilots/gpt56-sol/summary.json:3-13`). **MEASURED:** overflow-check time was 1,281.94 s versus 1,463.93 s recorded, ratio 0.876; wall time was 1,282.57 s. The baseline ran concurrently, so the ratio is not evidence that overflow checking accelerates ESBMC. Available memory was 8–9 GiB before every run.

NotebookLM follow-up command:

```text
python scratchpad/roundtable_pilots/gpt56-sol/run_interval_ablation.py
```

Exact no-interval variant:

```text
/home/joao/esbmc-linux/bin/esbmc /home/joao/code/preqbmc/output/crown_chain_v2_20260924/image3652/claude_hardened_l2/harnesses/chain_00122/chain_L2_n535_lower_s4_affine_0.c --function main --unwind 2 --z3 --bv --timeout 60 --memlimit 2g --result-only --overflow-check
```

**MEASURED:** with interval analysis, the first large chunk VERIFIED in 3.93 s. Without both interval flags it exhausted the 2 GiB limit in 16.78 s, returned -6, and reported `ERROR: Out of memory` and `SMT solver failed`; the ablation stopped after this non-VERIFIED result (`scratchpad/roundtable_pilots/gpt56-sol/interval_ablation.json:3-35`).

## Changes that survived Round 2

| change | trust-base change | expected gain | hours | validation pilot | main risk |
|---|---|---|---:|---|---|
| Enable `--overflow-check` for ground affine obligations on every reported VERIFIED image | none | **PROVED:** largest 300 all VERIFIED; **MEASURED:** 0.876 time ratio | 2 | Run all affine chunks of validation 3652; go only at 100% VERIFIED, no resource abort, and ≤1.25× time | Another coefficient path or harness kind overflows outside the sampled 300 |
| Add independently enumerated omission regressions: remove a harness record, chain, citation and competitor | none | **GUESS:** closes an omission class absent from the value-corruption suite, whose stated scope is changed accepted data (`src/scripts/regress_crown_chain_v2_data_corruption.py:1-7`) | 2 | Clean copy aggregates; every deletion is refused before VERIFIED | “Independent” enumeration accidentally shares renderer logic |
| Add a model-level ESBMC lemma binding deployed loop rows to certificate CSR rows, in deployment order | new ESBMC lemma | **GUESS:** removes the largest declared Python assumption; no speed gain | 6 cap | Layer 0 then layer 4; go if representative rows VERIFY within 60 s, then cover all row schemas | Reordered `__int128` sums create a hard equivalence query |
| Retain ESBMC interval analysis and simplification in the paper profile | none | **MEASURED:** one 25k-term ablation changed VERIFIED/3.93 s to OOM/16.78 s | 0 | Existing 300-run overflow trial uses both flags; reject any profile whose matched pilot is non-VERIFIED | Evidence is one deliberately small ablation |
| Produce the paper tables and explicit trust contract, including VIM3 measurements | none; exposes existing trusted components | **GUESS:** improves reviewer confidence; no certified-rate gain | 8–10 | Regenerate all tables from manifests; repeated board runs report dispersion and identical workloads | Deadline or board access leaves energy data incomplete |

The current deployment binding remains **ASSUMED** until the lemma succeeds: Python parses C arrays and reconstructs convolution/dense rows (`src/verification/crown_chain_v2.py:144-250`). The aggregator already checks authoritative chain counts, all-VERIFIED status, hashes and complete layer coverage (`src/scripts/aggregate_crown_chain_v2.py:80-155`).

## Rejected proposals

- Solver sweep: **MEASURED** 15,792/15,792 affine chunks simplify to zero remaining claims; SMT choice cannot reduce affine time.
- Persistent result cache: **MEASURED** only 19/58,695 repeat occurrences, all shared lemmas; zero affine duplicates.
- 8k chunks as a speed change: **MEASURED** Round 2 analysis predicts about +0.8 CPU-h; retain only as a future memory contingency.
- Loop/scalar renderer: **GUESS** at most 1.5× and no safe full-certificate pilot window before submission.
- Alpha slopes and input/ReLU splitting: 16–30 h implementation plus multiplied checking cost; future work.
- More than four jobs: **MEASURED** 12 jobs achieved 1.6× and caused resource aborts (`experiments/ssv2026_test_protocol.json:311-333`).
- Native generic checker: adds compiler/runtime trust and conflicts with ESBMC as final judge.

## Q3: minimum convincing evaluation

Report all 80 preregistered outcomes: **MEASURED** primary 1/80 with a binomial interval, category counts, and every non-VERIFIED image in the denominator, exactly as registered (`experiments/ssv2026_test_protocol.json:98-108`). Put the disclosed resource-only rerun beside it and never replace the primary. Report float and deployed-C accuracy, model bytes, matched VIM3 latency/energy/EDP, and verifier wall time, CPU time, RSS and counts split into 44,251 ground and 14,444 solver-backed obligations. Include monolithic ESBMC and IBP failures as baselines, ε=1, worst-class accuracy, sample-size limits, and the complete trust table.

Three strongest attacks and answers:

1. **“ESBMC is only a calculator.”** Report the ground/symbolic split explicitly; symbolic box and ReLU obligations establish quantified facts, while ground checks validate certificate arithmetic.
2. **“Python decides deployment correspondence or can omit work.”** Close binding with the new lemma if it succeeds; otherwise label Python mapping **ASSUMED** and narrow the claim. Show omission-regression failures and existing hash/completeness checks.
3. **“1/80 is weak and aborts hide failures.”** Accept the low primary rate, give its interval, classify SIGTERM as inconclusive, and show the rerun separately. Claim robustness only for fully VERIFIED images.

## Turn 12 — Claude Opus 5.5 (R3, 2026-09-27)

# Round 3 — Claude Opus 5.5 (Soundness Auditor)

## A. Trial log: binding lemma (commands and counts; not in the word count)

Everything is in `scratchpad/roundtable_pilots/opus/`: generators `make_binding_harness.py` (whole
file) and `make_slice_harness.py` (per-layer slices), runner `run_one.sh`, every run in `runs.tsv`,
logs in `logs/`.

- Generate: `python3 make_slice_harness.py <layer> <rows...> [--no-write] [--full] -o <file>.c`
- Run: `./run_one.sh <file>.c <timeout> <solver>`, which executes
  `esbmc <file>.c --function main --<solver> --bv --timeout <T> --memlimit 2g --interval-analysis --interval-analysis-simplify`.
  These are the recorded paper-z3 flags without `--unwind 2`, because the slice loops must unroll
  completely, and without `--result-only`, to see the phase times.
- **Deviation:** `free -g` showed 5–7 GiB available at every run, below the 8 GiB rule. The
  VS Code C/C++ extension was indexing the new `.c` files, at up to 3.7 GB RSS. The test run was
  stopped, only one ESBMC process ran at a time, and the 2 GB cap held.
- **Total ESBMC wall time:** 1,891 s (31.5 min), under the 45 min cap.

Rows are checked for read binding (R), write binding (W), or both.

| Harness | Rows | Solver | Verdict | Wall s | RSS MB |
|---|---|---|---|---|---|
| `bind_L0_r408.c`: whole deployed file `#include`d | L0 408 | z3 | OUT OF MEMORY, GOTO generation | 60.2 | 2,095 |
| same, without the interval flags | L0 408 | z3 | OUT OF MEMORY, GOTO generation | 60.2 | 2,095 |
| `slice_L0_a.c` | L0 0, 408 R+W | z3 | TIMEOUT, in symex | 60.0 | 167 |
| `slice_L0_r0.c` | L0 0 R+W | z3 / bitwuzla | VERIFIED / VERIFIED | 1.68 / 0.37 | 339 / 104 |
| `slice_L0_r24.c` | L0 24 R+W | z3 | VERIFIED | 3.80 | 349 |
| `slice_L0_r96.c` | L0 96 R+W | z3 | VERIFIED | 41.94 | 428 |
| `mut_L0_r0.c`: one weight +1 | L0 0 | z3 | TIMEOUT, in the solver | 60.0 | 339 |
| `mut_L0_r0.c` | L0 0 | bitwuzla | FAILED, counterexample | 2.40 | 151 |
| `slice_L1_r0_read.c` | L1 0 R | z3 | VERIFIED | 1.30 | 218 |
| `slice_L2_r0_read.c` | L2 0 R | z3 | VERIFIED | 39.16 | 1,188 |
| `slice_L3_r0_read.c` | L3 0 R | z3 | VERIFIED | 26.46 | 819 |
| `slice_L4_r0_read.c` | L4 0 R | z3 | VERIFIED | 20.27 | 455 |
| `slice_L4_r0.c`, first write check | L4 0 R+W | z3 / bitwuzla | SOLVER ABORT / OUT OF MEMORY | 29.1 / 38.6 | 1,458 / 1,553 |
| same, mirror clamp fixed | L4 0 R+W | z3 | SOLVER ABORT | 31.9 | 1,458 |
| `slice_L4_r0-3.c` | L4 0–3 R+W | z3 | SOLVER ABORT | 55.4 | 1,548 |
| `slice_L4_r0.c`, accumulator havocked before the write check | L4 0 R+W | z3 | VERIFIED | 38.65 | 458 |
| `slice_L4_all.c --full` | L4 all 43 R+W, call count | z3 | OUT OF MEMORY, Z3 encoding (symex 697 s, 258 VCCs) | 721.0 | 1,958 |
| `slice_L4_all_read.c --full` | L4 all 43 R, call count | z3 | OUT OF MEMORY, Z3 encoding (symex 576 s, 129 VCCs; signal 6) | 599.0 | 1,994 |

A native run, not ESBMC (MEASURED): gcc builds of the whole-file harness with random inputs matched
L1 rows 0, 1, 1000 and 4095 and all 43 L4 rows, and the +1 mutant failed.

## B. What the trial proves, measures and assumes

**PROVED** by the VERIFIED rows. Take the layer-k loop nest, copied byte for byte from the deployed
file (`dd535f54…`), and any contents of its input buffer. Then:
- **Read:** the accumulator at the r-th rounding call equals CSR row r of `layers.json`
  (`eb3bdc7a…`), summed left to right.
- **Write:** for any accumulator, the code stores its round, bias, clamp, ReLU and clamp at
  index r.

Rows proved: layer 0 rows 0, 24 and 96 (both); row 0 of layers 1–3 (read); layer 4 row 0 (both).
That is **7 of 14,123 rows**.

**MEASURED**
1. **ESBMC cannot load the whole deployed file:** GOTO generation runs out of memory at 2 GB.
   Verbatim per-layer slices (Sena et al.'s slicing, Turn 9 §1) load in 0.2–0.4 s. No ESBMC run in
   this project has seen the whole file.
2. **Row r costs every earlier iteration.** Layer 0 symex takes 0.2 s at row 0, 2.6 s at 24 and
   40.4 s at 96, about quadratic. Layer 4 takes about 16 s per row, since each multiply indexes a
   33,024-entry constant array; all 43 rows in one run exhaust Z3 at 2 GB.
3. **Non-vacuity.** With one weight changed by 1, Bitwuzla reports FAILED in 2.4 s and Z3 times out.
   Z3 turns a wrong binding into a TIMEOUT, never a VERIFIED. That is sound, but mutation checks
   need Bitwuzla.
4. **The write check needs a havoc.** Two 128-bit divisions of a 768-term sum made Z3 abort at
   1.5 GB. A nondet accumulator after the read check fixed that and strengthens the claim. The
   trial also caught my bug: my layer-4 mirror clamped once, but the code clamps twice (198–199).

**ASSUMED**
- **Tiling:** a script checks that the slices, the prologue (70–72), the output copy (203) and the
  braces cover every non-blank line of `qnn_forward_fixed` (69–204) without overlap.
- **Buffers:** pointer parameters behave like the function's local buffers.
- **Input copy:** line 72 holds by inspection.

**Verdict:** pass on the sampled rows, fail on coverage. Layers 0–3 stay on the Python
`EXACT_ORDERED_MATCH` (`crown_chain_v2.py:144-251`).

**Correction to my Turn 4:** the box-bound harnesses are ground, not symbolic over the ε-box. They
hold literal arrays and no nondet values (`…/proof/harnesses/layer_2_block_51_*.c`). Symbolic
reasoning is only in the monotonicity lemmas, encoder, non-vacuity, bridges and ReLU chunks.

## C. Final plan

| # | change | trust-base change | expected gain | hours | validation pilot | main risk |
|---|---|---|---|---|---|---|
| 1 | Trust table rewritten from §B | none (text) | removes three overclaims (§B) | 2 | none; each claim cites a log | overclaim in the abstract |
| 2 | Binding lemma in `src/` after the freeze (worktree): rows 0–k per layer, ≤60 s each, plus a Bitwuzla mutant per layer | new ESBMC lemma; trusted tiling check (~30 lines) | 1.3–42 s per row (MEASURED) | 3 | 3652: every row VERIFIED and every mutant FAILED | coverage is a loop prefix, not a random sample |
| 3 | Omission regression (gpt6-sol) | none (test) | catches dropped obligations (GUESS) | 2 | KICKOFF R3 criterion | the enumeration reuses renderer logic |
| 4 | Overflow closure: `--overflow-check` on VERIFIED images' affine harnesses; every affine input coefficient a ReLU output or ±1 (`crown_chain_v2.py:877-883`) | none (ESBMC) | the \|Σ\| < 2⁵⁷ argument becomes a verdict | 2 + machine | gpt56-sol: 300 VERIFIED | time ratio |
| 5 | Evaluation (§E); VIM3 figures for the certified binary | none | the paper's result | 8–10 | Monday tally | rerun ends after 10-01 |

## D. Rejected

- **Affine solver choice:** 0 s of solver time (R2).
- **Lower affine cap:** a memory lever, not a speed lever.
- **Result cache:** more trust base, no measured hit rate.
- **Slope and renderer work before 10-04:** no pilot while the rerun holds the machine.
- **Whole-file ESBMC binding:** out of memory.
- **Symbolic-index loop lemma, int16 repacking:** future work. One edits the loop headers; the
  other changes the certified hash.
- **Counting as proofs** sampled rows or any INCONCLUSIVE.

## E. Q3

1. **Primary:** the locked sample as it stands: 1/80 VERIFIED (image 005), Clopper–Pearson 95%
   0.03–6.77%, every other verdict with its cause.
2. **Disclosed rerun:** a separate, labelled row citing the resource-only amendments (`7aa4a929`,
   `b390496e`, `23e4fcf3`); 2/80 gives 0.30–8.74%. Never merged.
3. **Baselines:** float and QNN accuracy; the CROWN-only count, labelled a finder; verifier cost
   (13.9 CPU-h on 3652, 61% front end, about 180 µs per term).

**Three strongest attacks**
- **"You verified Python-generated harnesses, not the deployed C."** Change 1: ground ESBMC checks
  of every interval step, symbolic lemmas for rounding and clamping, and a binding complete in
  Python and ESBMC-checked on verbatim slices. Say that ESBMC cannot load the whole file.
- **"1/80 is useless."** Interval plus causes. The contribution is sound verification of deployed
  code at 17,195 neurons (inputs counted, as EQV does); the largest bit-precise network in the
  sources has 1,306 (Turn 9 §6).
- **"The QNN does not use fewer resources."** MEASURED: the certified file stores 307,211
  `int64_t` entries (2.46 MB), with layer 3 dense (147,456 entries, 767 non-zero). A float32
  network with the same layers takes about 0.64 MB. Claim only what the VIM3 measures on this
  binary. An int16 build would take 0.32 MB, since every |w| ≤ 1,349, but it has no certificate.
