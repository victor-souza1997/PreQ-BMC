# PreQ-BMC Dissertation Integration Guide

**Prepared:** 15 November 2026  
**Source documents:**

1. *Master_Dissertation___Network_Equivalence (8).pdf* — 44-page qualification-era dissertation draft (January 2026).
2. *LADC_2026_João_Victor (9).pdf* — 17-page submitted paper, *Preimage-Guided Bit-Precise Verification of Fixed-Point Quantized Neural Networks*.
3. *SBESC___SSV_2026___AAOS_with_PreQ_BMC (10).pdf* — 6-page submitted paper, *Verified Edge AI for Android Automotive: Deployment-Aware Verification of Fixed-Point Neural Networks*.

**Purpose:** Turn the existing qualification draft and two submitted papers into one accurate master's dissertation, **without pretending that the LADC and SBESC workflows are the same algorithm or offer identical guarantees**.

---

## 1. One-sentence thesis and intended scientific storyline

**Recommended thesis statement:**

> This dissertation investigates how neural-network abstraction and compositional software model checking can provide bit-precise robustness guarantees for generated fixed-point implementations, from preimage-guided quantization synthesis for small dense networks to deployment-aware certification of a larger convolutional classifier, while characterizing the practical limits of proof scalability and embedded execution.

**Recommended title (optional):**

*Deployment-Aware Bit-Precise Verification of Quantized Neural Networks: From Preimage-Guided Synthesis to Embedded CNN Certification*

**Narrative progression:**

- **Core problem:** Proofs about idealized network arithmetic do not automatically transfer to a deployed fixed-point C program.
- **Study A (LADC):** Use DeepPoly forward analysis and MILP backward preimages to guide per-layer quantization synthesis; check generated C contracts using ESBMC, with block decomposition. Demonstrated on small fully connected ReLU networks.
- **Study B (SBESC):** Given a selected global Q7.8 CNN, use interval facts and integer CROWN candidate chains, checked with ESBMC and a trusted structural/binding layer; evaluate robustness and real AAOS execution.
- **Cross-study conclusion:** Decomposition permits otherwise intractable checking, but large networks generate many proof obligations. Relaxation precision, verified dependency management, and host/embedded resource costs become limiting factors.

**Crucial boundary:** These are two implementations/studies connected by a shared **problem, concrete arithmetic semantics, and proof philosophy**, not one continuous DeepPoly → MILP → CROWN algorithm. Do not write a diagram or algorithm that falsely suggests SBESC uses MILP for its CNN certification.

## 2. Proposed final dissertation architecture

### Chapter 1 — Introduction

1.1. Embedded inference and fixed-point quantization  
1.2. Implementation gap: abstract guarantees versus deployed C semantics  
1.3. Research problem and scope  
1.4. Research questions  
1.5. Contributions and relationship to the two submitted papers  
1.6. Dissertation organization

**Draft research questions:**

- **RQ1 — Semantic fidelity:** What obligations and trust assumptions are needed to establish local classification robustness of generated fixed-point C rather than an abstract QNN?
- **RQ2 — Quantization synthesis:** To what extent can preimage-guided per-layer quantization produce C implementations whose local contracts and deployed-transfer properties can be checked bit-precisely?
- **RQ3 — Verification scalability:** How do block decomposition and integer linear-certificate decomposition affect verification feasibility and computational cost as architecture size increases from dense networks to a CNN?
- **RQ4 — Practical limits and deployment:** What kinds of inconclusive results arise, and how do the certified and compact implementations behave on an embedded AAOS platform?

**Contribution statements (claim only what is supported):**

1. An implementation-oriented formulation of input encoding, finite-width semantics, contracts/dependencies, and strict output-margin verification, with its trusted boundary explicitly stated.
2. Preimage-guided, per-layer fixed-point synthesis integrated with ESBMC checking of generated C contracts, evaluated on dense networks.
3. A larger-CNN verification study using integer CROWN-derived chains and compositional ESBMC obligations with an independently described Python certificate/dependency checker.
4. Empirical scalability and deployment evidence, including unfavorable outcomes and bounded diagnostics of incompleteness.

Avoid promising globally optimal widths, universal ANN/QNN equivalence, backdoor-freeness certification, compiler correctness, or AAOS-level safety.

### Chapter 2 — Foundations

2.1. Feed-forward and convolutional neural networks, affine/ReLU layers  
2.2. Quantization formats and signed ranges, Q=1+I+F, scaling S=2^F  
2.3. Precise fixed-point C operations: accumulator → RHAZ rescaling → bias → saturation → activation  
2.4. Local robustness, input bytes, preprocessing and strict target dominance  
2.5. Abstract interpretation / DeepPoly / MILP preimages / integer CROWN (different roles)  
2.6. SMT/BMC, ESBMC, bit vectors, nondeterministic inputs and safety checks  
2.7. Assume-guarantee composition and proof dependencies

Keep concise; do not repeat full methodology or implementation-specific results here.

### Chapter 3 — Related work and precise positioning

Organize by **verification target and guarantee** instead of listing citations chronologically:

- Abstract or real-valued robustness: DeepPoly, CROWN, Marabou/Reluplex.
- Quantization synthesis: Quadapter, CEG4N.
- Fixed-point/QNN verification: QNNVerifier, QEBVerif, relevant integer/MILP/BMC works.
- Program-semantics challenge and implementation gap: Cordeiro et al., *Neural Network Verification is a Programming Language Challenge*.
- Embedded deployment/quantized inference: relevant work used in SBESC.

Use a comparison matrix with columns for quantization synthesis, numerical semantics, C-level checking, complete output-property closure, and on-target measurement. Do not equate different tools' checked properties or claim head-to-head superiority from nonidentical proof obligations.

### Chapter 4 — Shared deployment-aware verification framework

**This chapter must be written as dissertation synthesis, not copied from either article.** It explains what is common, then states the different proof frontends.

4.1. Verified object and deployment identity  
4.2. Mathematical property and input coverage  
4.3. Fixed-point execution semantics, assumptions, rounding, clamp, overflow scope  
4.4. Candidate generation versus formal discharge  
4.5. Proof obligations, composition, closure, and trusted components  
4.6. Two specializations: preimage-guided synthesis vs CNN certificate generation  
4.7. Provenance, outcome taxonomy and reproducibility

Define, using consistent notation,

\[
\forall x\in \mathcal B_\varepsilon(x_0),\quad
\widehat{\mathcal N}(q(x))_t>\widehat{\mathcal N}(q(x))_j,
\quad \forall j\ne t.
\]

For an **assume-guarantee contract mode**, the sufficiency argument is based on input coverage, valid verified layer transitions, guarantee-to-next-assumption inclusion, and output strict dominance. For **CROWN chain mode**, establish valid arithmetic and relaxation facts, sound dependency/fact identities, verified required harnesses, correct layer/link composition, and output clamp closure. Do **not** assert these are identical certificate formats.

**Explicit trust table:**

| Element | LADC implementation | SBESC implementation | Scope |
|---|---|---|---|
| Candidate construction | DeepPoly/MILP + error budget / format search | Interval facts + integer CROWN | Proposals are not proofs |
| Local verification | ESBMC on generated fixed-point C contracts/blocks | ESBMC on affine, ReLU, rounding and related C obligations | Bit-precise checked obligations |
| Composition | Contract chaining, output guarantee, guarantee status | Python certificate DAG/fact/binding checks and margin/clamp closure | Trusted composition software must be identified |
| Input coverage | Encoded input-region overapproximation | Bounded uint8 region and C encoder | Claims limited to declared domains |
| Deployment | Generated C; empirical Python/C comparisons | Reference C and separate compact build on AAOS | Native parity is measured, not compiler/binary equivalence |

Do not imply every structural identity and array binding is checked directly by ESBMC. Cite exact versions and code snapshots for each study.

### Chapter 5 — Study A: preimage-guided quantization synthesis (LADC)

5.1. Scope: small fully connected ReLU models  
5.2. Forward DeepPoly source prerequisite and backward MILP preimages  
5.3. Per-layer candidate formats and conservative implementation-error budgets  
5.4. Integer C contracts, post-activation chaining and output proof  
5.5. Block verification and optional relational cuts (describe actual mode and trust dependency)  
5.6. Experiments: Iris, Seeds, MNIST; per-query budgets, comparators, accuracy and nominal size  
5.7. Results and limits: contract coverage ≠ deployed transfer; conditional results; comparison caveats

**Source anchor:** LADC paper §§4–5, Tables 2–4, Fig. 2.

**Required precision:** LADC reports 26 encoded-domain contract cases in its primary suite, but its Discussion states only 1/10 MNIST regions achieved the stricter deployed-transfer guarantee; do not describe all 26 as complete-network deployed robustness. LADC's table also includes an additional MNIST 100 exploratory row not part of the stated 26-region denominator. Preserve the exact result population.

**Source-specific caution:** LADC §4.6 presents relational cuts as separately verified by a prefix ESBMC harness; the qualification draft's §2.8.4.3 describes a conditional analytical MILP-dual and widening argument. Inspect the exact experimental mode/artifact and distinguish these approaches rather than choosing one wording without verification.

### Chapter 6 — Study B: integer-CROWN CNN verification and AAOS deployment (SBESC)

6.1. Why the dense-network approach does not directly scale to CNN-scale verification  
6.2. Frozen GTSRB network, global Q7.8, byte input domain and source-to-C identity  
6.3. Interval box facts and integer CROWN candidate chain construction  
6.4. ESBMC harnesses, shared lemmas, dependency closure and output-clamp closure  
6.5. Formal experiment protocol: n=80 test image-radius regions; monolithic control  
6.6. Certification results: 1/80 primary, 2/80 disclosed rerun; 8 nominally wrong, 54 finder inconclusive, 16 checker inconclusive after rerun  
6.7. Checking costs: ~49–51k calls per certified test region and ~12–20 h summed ESBMC time  
6.8. Deployment: Khadas VIM3/AAOS; float32, certified int64 reference, compact int16 variant  
6.9. Scope: host/C/board fidelity, limits of native-output parity, no AAOS API formal verification

**Source anchor:** SBESC §§III–VI, Fig. 1, Tables II–IV, Fig. 2.

**Avoid incorrect inference:** The CNN's Q7.8 format was selected/fixed before the CROWN robustness campaign; MILP-based layerwise synthesis is not the CNN verifier frontend. The int16 compact deployment is not formally equivalent to the certified int64 implementation merely because its outputs match on all 7,830 validation images.

### Chapter 7 — Integrative discussion, scalability, and limitations

7.1. Comparison of the two verification workflows and proofs  
7.2. Different meanings of `VERIFIED`: layer contract, deployed transfer, complete CNN region  
7.3. Scalability transition: reducing single-query size vs accumulating thousands of verified obligations  
7.4. Incompleteness: false property, incomplete bounds, time/memory limit and trusted dependency closure  
7.5. Optional post-submission diagnostics (P3, I-B, P3b/P3c, I-P, I-R and exact-encoding pilot)  
7.6. Implementation gap, source/target correspondence and formal proof boundaries  
7.7. Threats to internal, external and construct validity

**Post-submission diagnostics must remain supplemental.** Clearly tag them as exploratory development measurements, not additions to the frozen SBESC 80-case test denominator or LADC 26-case contract denominator. Specific observations available from the subsequent project work:

- Local margin slope search recovered no previously inconclusive images in the tested development cohort.
- Bound-chain slope refinement improved 016/c5 from -876 to -647, but did not close the region.
- A single-coordinate split of 016/c5 gave a worst-child gain of +1 while doubling the prospective proof graph.
- A fixed pair of upstream ReLU-remainder obligations and its decoupled control both timed out at 60 s; later exact-width reduction cut formula assignments and memory substantially but still did not settle them. No relational proof was obtained.
- These observations do **not** show that lost correlations caused the failure or that relational reasoning is impossible. They show limitations of the particular bounded interventions.

Include source hashes, experiment IDs, baseline parent identities, outcome statuses, and any audit supplements if they are made part of the submitted dissertation evidence.

### Chapter 8 — Conclusions and future work

Answer RQ1–RQ4 individually; state what was **proved**, what was **measured**, what remains **conditional**, and what is **unknown**.

Future work: better certificate-aware refinement and selective relational information; stronger input/deployment binding; improved solver scalability; larger/different models and hardware; hardware-specific integer kernels/verification of native binaries. Do not present speculative refinements as already implemented.

---

## 3. Migration map from existing qualification dissertation

| Existing draft location | Action | New destination | Reason |
|---|---|---|---|
| Title/cover, Abstract and preliminary pages | **Replace/update** | Front matter | Current PDF identifies itself as a January 2026 qualifying exam; abstract incomplete |
| Ch. 1 opening (general AI/model growth) | **Condense strongly** | Ch. 1 | Lead with implementation gap, not broad neural-network history |
| §1.1 Problem Statement | **Rewrite** | Ch. 1 | Retain implementation-gap core, remove overly broad exclusivity/optimality claims |
| §1.2 Objectives, §1.3 Contributions | **Replace** | Ch. 1 | Present achieved research, not proposed backdoor/equivalence/theory deliverables |
| §1.4 Dissertation Structure (`Chapter ??`) | **Replace** | Ch. 1 | Match final eight-chapter structure |
| Ch. 2 §§2.1–2.3 (region, symbolic bounds, preimage) | **Keep selected formal definitions, revise** | Ch. 2 and Ch. 5 | No longer narrate as mandatory stages of CNN workflow |
| Ch. 2 §§2.4–2.7 (old search, C harness, flags, tiers, Fig. 2.1, Algorithm 1) | **Remove/rewrite** | Ch. 4/5, historical appendix if needed | Obsolete assumptions, misleading flow and old tool flags |
| Ch. 2 §2.8.1 positioning | **Merge/correct** | Ch. 3 | Resolve `CITATION NEEDED`; verify citation/claim scope |
| Ch. 2 §2.8.2 semantics, §2.8.3 contracts | **Keep with audit** | Ch. 4/5 | Strong basis for common semantics and dense-network study |
| Ch. 2 §2.8.4 blocks, cuts, output, e2e | **Split by implemented mode** | Ch. 5 and Ch. 6 | Different actual guarantees and checker mechanisms |
| Ch. 2 §2.8.5–2.8.6 verdicts/reproducibility | **Keep/reconcile** | Ch. 4 and study-specific experimental setups | Each artifact has different settings and trust assumptions |
| Ch. 3 old Iris/Seeds comparison tables | **Demote to preliminary or archive** | Ch. 5 intro or Appendix A | Superseded by more complete LADC evaluation; retain only if independently documented |
| Ch. 3 old tables/configuration | **Replace** | Chs. 5 and 6 | Do not mix early hardware/solver measurements with frozen 2026 campaigns |
| Ch. 4 old conclusion and activity schedule | **Replace/remove** | Ch. 8 | Final findings, not proposed activity timeline |
| Bibliography | **Deduplicate/validate** | Bibliography | Prefer published journal/conference versions and correct DOIs |

**Do not preserve obsolete figures merely because they are in the LaTeX tree.** Replace the old too-wide, tier-based Fig. 2.1 with a legible overview explaining two candidate-generation paths that converge at implementation-level verification.

---

## 4. Exact claims that must be reconciled before submission

1. **Contract vs deployed robustness:** LADC Discussion: 26/26 encoded-domain contract cases vs 1/10 MNIST stricter deployed-transfer. Its Conclusion says all 26 regions were certified. Qualify/reconcile this before writing the dissertation results and conclusions.
2. **Two different CNN/dense proof modes:** LADC uses preimage-guided per-layer format synthesis; SBESC uses a fixed Q7.8 CNN and integer CROWN candidate bounds; neither is simply a benchmark of the other.
3. **Relational cut trust:** LADC's separately ESBMC-checked prefix cuts vs dissertation's conditional analytical MILP-dual cuts are not automatically the same mechanism. Determine which results used which.
4. **ESBMC scope:** ESBMC discharges reported C arithmetic obligations. Trusted Python parsing, fact/dependency linking, completeness and array binding also matter, and source-to-native-hardware correspondence is not fully proved.
5. **Input domain:** LADC's encoded real/tabular/dense-network input regions vs SBESC's resized raw byte `L_inf` box. Specify normalization, encoder and whether a theorem is about the encoded box or an original crop domain.
6. **Finite-width semantics:** Keep sign-inclusive `Q=1+I+F`, rescaling/round-half-away, bias after rounding, clamp before ReLU. Do not call bit-vector BMC implementation a direct implementation of Baranowski's dedicated SMT fixed-point theory unless it truly is.
7. **Bit-width optimality:** Format search selects certifiable configurations according to its policy; no proof of global bit-width optimality is shown. Q7.8 is an evaluated global choice, not the unique optimal bit-width.
8. **QuAdapter comparison:** Different proof properties and unverified tensor identity prohibit unconditional performance/superiority comparisons.
9. **SBESC outcomes:** Keep primary 1/80 separate from disclosed rerun 2/80; retain misclassified and resource-limited cases in original denominator. Do not convert finder positives into certificates.
10. **Post-submission changes:** P1/P2 checker auditing and P3/I-B/I-P/I-R diagnostics have their own commit/provenance stories; distinguish new supplementary evidence from the frozen paper experiment.
11. **Hardware fidelity:** float32 and fixed QNN performance claims are specific to the Cortex-A73 and generated scalar kernels. Empirical parity for compact int16 is not a proof of binary equivalence; AAOS is an environment, not a formally verified OS.
12. **Metadata/cleanliness:** Replace qualification wording; remove `Chapter ??`, `CITATION NEEDED`, duplicate tables, mixed Portuguese headings, and unused activity schedules.

---

## 5. Recommended unified results tables and figures

### Table A — Scope of the two studies

Columns: objective; architectures; dataset; quantization strategy; bound proposal; C semantics; ESBMC obligation type; proof composition; final property; validation population; hardware metrics.

### Table B — Formal result taxonomy

Rows: dense encoded contracts; dense deployed-transfer; CNN test primary; CNN test disclosed rerun; post-submission validation-only diagnostics.

**Do not put these rows under one denominator or aggregate them into one pooled certification rate.**

### Table C — Scalability

LADC: block size vs maximum query seconds and RSS (same exact model/sample).  
SBESC: monolithic vs decomposed completion, calls, summed verification seconds, per-call/aggregate memory (state which).  
Use two panels if necessary; never compare absolute machine wall times between studies as controlled benchmark speedup.

### Table D — Deployment

Float32, certified-reference int64 Q7.8, compact int16 Q7.8: accuracy, median/p95 inference latency, memory, parity scope, formal proof status. Distinguish the 12,630-image test accuracy corpus from 7,830-image validation parity corpus.

### Figure 1 — One-page dissertation map

Input network/property → **branch A** (DeepPoly/MILP/per-layer format synthesis) / **branch B** (fixed Q7.8 + integer CROWN proposals) → generated fixed-point C/obligations → ESBMC local verification + correctly specified checker/composition → result (proved/inconclusive) → separate on-target measurement.

### Figure 2 — Semantic proof boundary

Distinguish `source ANN`, `generated C semantics`, `ESBMC harness`, `trusted Python composition`, `native compiled binary`, and `AAOS`. Draw solid verified links only where actually established; dashed measured/assumed links elsewhere.

### Figure 3 — Scaling trade-off

Separate plots: dense block-size query cost; CNN proof-call count/work distribution; optional zoom of post-submission inconclusive-margin diagnostics.

---

## 6. Reusable bridge paragraphs (drafts, revise to fit final evidence)

**Bridge from Chapter 4 to Chapter 5:**

> The implementation-level verification objective can be approached through different forms of sound candidate information. The first study uses forward abstract bounds and backward preimages to guide the synthesis of per-layer fixed-point formats. Generated integer C contracts are checked with ESBMC, and block decomposition limits the cost of individual dense-layer obligations. This study establishes the quantization-synthesis aspect of the framework and motivates the need to account explicitly for the difference between checked local contracts and deployed end-to-end guarantees.

**Bridge from Chapter 5 to Chapter 6:**

> The dense-network experiments establish feasibility for small architectures but do not demonstrate that the same preimage-based search scales directly to convolutional networks. The second study therefore fixes a deployment-relevant Q7.8 CNN and changes the candidate-generation strategy to integer CROWN-style linear bounds. The certifying target remains the fixed-point C implementation, but the proof representation and composition mechanism differ. This separation permits a focused evaluation of CNN-scale verification and native execution without attributing CNN certification to the earlier MILP quantization-synthesis procedure.

**Bridge from Chapter 6 to Chapter 7:**

> Taken together, the two studies show complementary dimensions of scalability. Neuron-block decomposition reduces the size of individual obligations in small fully connected networks, whereas large CNN certification requires extensive proof decomposition and verification of many affine and activation lemmas. The resulting incompleteness is not equivalent to a demonstrated robustness violation: it may arise from conservative abstractions or from verification resource limits. The following discussion compares these failure modes, the trusted composition boundaries, and the fidelity of the deployed implementation.

---

## 7. Practical editing workflow (recommended order)

### Gate 0 — Preserve and classify evidence

- Freeze/source-control the current thesis tree and both submitted paper versions.
- Build `evidence_matrix.csv` with `claim`, `source document`, `page/section`, `experiment artifact`, `proof level`, `data cohort`, `hardware`, `trust assumption`, `allowed wording`.
- Mark `PROVED`, `MEASURED`, `CONDITIONAL`, `INCONCLUSIVE`, `ASSUMED` separately. An old manuscript sentence is **not** proof that a newer codepath performed the stated check.

### Gate 1 — Fix technical truth before prose

- Settle LADC 26-vs-1/10 guarantee levels and its statement about relational cuts.
- Audit which result counts belong to the five primary models/26 regions and which exploratory model/samples are supplemental.
- Settle how SBESC's checker/aggregation and deployed-array binding are described; identify all trusted components and any P1/P2 follow-up audit.
- Define exactly which experiment populations were validation, frozen test, post-submission diagnostic.

### Gate 2 — Rebuild the LaTeX skeleton

- Replace qualification front matter and old activity schedule.
- Write final chapters and subsection placeholders before importing paragraphs.
- Give every table/figure/algorithm a unique label and a single home. Do not duplicate an algorithm under two chapters.
- Maintain one notation glossary and one fixed-point semantics definition.

### Gate 3 — Write the dissertation's genuinely new synthesis

- **Write Chapter 4 first**: common problem and guarantees, with two explicitly separate branches.
- **Write Chapter 7 second**: compare proof strength, scaling and limitations without averaging unrelated results.
- Then migrate and rewrite LADC into Chapter 5 and SBESC into Chapter 6, expanding details for a dissertation.

### Gate 4 — Replace front/back matter

- Rewrite Abstract, Introduction/RQs and Conclusion using results now grounded in Chapters 4–7.
- Explicitly disclose source papers, collaboration and your own contribution; confirm university and publisher reuse rules for figures/text.
- Complete citations, including journal/conference versions and DOIs where available.

### Gate 5 — Technical and typesetting audit

- Verify every `VERIFIED` is supported by exact outcome records and the claimed guarantee level.
- Check arithmetic formulas against the C operation ordering and types; verify proof-graph trust assumptions.
- Cross-check all numeric tables against source CSV/JSON; never manually add denominators from distinct studies.
- Render the PDF; inspect every equation, algorithm, figure and wide table for legibility; re-run LaTeX with all references resolved.

---

## 8. Definition of ready for supervisor review / submission

**Supervisor-review ready** when all of the following are satisfied:

- One thesis statement and explicit RQs connect the two works.
- Two methods are distinguished without a fictitious unified algorithm.
- Chapter 4 states the exact verification boundary/trust base.
- LADC's contract vs deployed-transfer guarantee discrepancy is reconciled.
- SBESC primary/rerun results and hardware-parity claims remain faithful to their experimental protocols.
- All figures, results, citations, and theorem hypotheses are auditable.
- The current 44-page qualification text has been revised rather than merely supplemented.

**Submission ready** additionally requires final university formatting, faculty/advisor review, institutional front matter, citation/permission checks, complete build, and sign-off on the final evidence and contribution claims.

**Suggested next artifact:** a line-by-line thesis edit plan referencing the actual `.tex` files. PDFs establish content; to apply this plan safely, obtain the current Overleaf/LaTeX project and map each section to source filenames and labels before editing.
