# Scientific conflicts and blockers register

Phase 0 audit, 2026-11-16. Read-only: no dissertation file, experiment record or result was changed.

Each entry states what the sources say, what the recorded evidence shows, and how the dissertation should resolve it. Claim IDs refer to [evidence_ledger.csv](evidence_ledger.csv).

**Status values**

- **Resolved by evidence**: the records settle the question; the dissertation only has to use the right wording.
- **Author decision**: the evidence is clear, but the author or advisor must choose how to present it.
- **Blocker**: information is missing. Do not write the affected text until it is supplied.

**Source abbreviations**

| Tag | File |
|---|---|
| D-TEX | `latex_source/Master_Dissertation___Network_Equivalence/` (builds to 44 pages) |
| D-PDF | `docs/papersANDdissertation/Master_Dissertation___Network_Equivalence (6).pdf` (66 pages) |
| L-PDF | `docs/papersANDdissertation/LADC_2026_…_Victor (1).pdf` (17 pages) |
| L-OLD | `docs/LADC_2026/samplepaper.tex` (first-submission source, superseded) |
| L-REC | `docs/LADC_2026/results_revision/generated/` (frozen run records) |
| L-AUD | `docs/ladc2026_review/methodology_audit_response.md` (2026-09-10) |
| S-PDF | `docs/papersANDdissertation/SBESC___SSV_2026___AAOS_with_PreQ_BMC (2).pdf` (6 pages) |
| S-EVID | `docs/ssv2026/camera_ready_review_evidence.md` |
| S-PROTO | `experiments/ssv2026_test_protocol.json` |
| WT | worktree `../preqbmc-qformat/docs/ssv2026/` (branch `feat/crown-v2-qformat`) |

---

## Summary

| ID | Conflict | Severity | Status |
|---|---|---|---|
| C-01 | LADC "26 certified" against "1 of 10 deployed-transfer" | Critical | Author decision, with one blocker |
| C-02 | Three different relational mechanisms described as one | Critical | Resolved by evidence |
| C-03 | "ESBMC discharges every obligation" against what ESBMC actually checks | Critical | Resolved by evidence |
| C-04 | Synthesis (Study A) against a fixed global format (Study B) | High | Resolved by evidence |
| C-05 | Formal robustness against measured deployment fidelity | High | Resolved by evidence |
| C-06 | Input domains: encoded real box against discrete byte images | High | Author decision, with one blocker |
| C-07 | Fixed-point notation and width convention | High | Resolved by evidence |
| C-08 | Claimed use of the SMT fixed-point theory | High | Blocker |
| C-09 | Two incompatible ESBMC configurations inside the dissertation | High | Resolved by evidence |
| C-10 | Comparative and "optimal" claims without support | High | Resolved by evidence |
| C-11 | SSV counts, rerun and protocol history | Medium | Resolved by evidence |
| C-12 | Certification rates that cannot be compared across studies | Medium | Resolved by evidence |
| C-13 | Dissertation objectives never carried out | Medium | Author decision |
| C-14 | Error budget and chaining rule differ between paper and dissertation | Medium | Blocker (code check) |
| C-15 | "Monolithic" and "block" mean different things in the two studies | Medium | Resolved by evidence |
| C-16 | Source versions do not match the instruction file | Medium | Author decision |
| C-17 | Qualification-era results have no located artifacts | Medium | Blocker |
| C-18 | Authorship, publication status and reuse | Low | Author decision |
| C-19 | LADC solver: paper says Z3, half of the cohort used Bitwuzla | Medium | Resolved by evidence |

---

## C-01. LADC: "26 certified" against "1 of 10 deployed-transfer"

**What the sources say**

- L-PDF abstract and §5.3: all contract checks passed for 26 encoded-domain regions, 13 of them MNIST.
- L-PDF §5.3, same paragraph: "Under the stricter deployed-transfer criterion, however, only 1 of 10 MNIST regions was certified, while the remaining 9 were inconclusive."
- L-PDF conclusion: "PreQ-BMC certified all 26 evaluated local regions."

**What the evidence shows**

- The "1 of 10" figure belongs to the first-submission campaign. L-OLD lines 1596 and 1658 report ten nested MNIST regions for sample 3: `1blk_10` certified at ε = 0.25, Margin-Inconclusive at 0.5, 1, 2 and 4, and all five `2blk_10_10` runs Layer-Inconclusive.
- The revision package calls that statement obsolete (`results_minimal_integration/INTEGRATION_GUIDE.md`, step 2) and says the new cohort is "not a paired causal improvement over the old ten-region campaign" (`results_revision/reviewer_coverage.md`).
- In the frozen records (L-REC `per_run.csv`, `validation.json`) every one of the 26 main runs, including the same ten MNIST identifiers, has:
  - `contract_status = VERIFIED`;
  - `raw_guarantee_level = deployed-transfer`;
  - `raw_final_status = PARTIAL_VERIFIED`, because the no-saturation check was `SKIPPED`;
  - `real_input_coverage_independently_audited = False`.
- L-AUD §1 downgrades the pipeline's own `deployed-transfer` label: the proof holds for the encoded integer box, and the bridge from the requested real-valued box is not established (see C-06).

So the two numbers are not two criteria applied to one cohort. They come from two campaigns run on different code revisions. The sentence in L-PDF §5.3 has no support in the 26-region records, and the conclusion overstates in the other direction.

**Resolution**

Report three separate statements and never one pooled rate:

1. **26 of 26 configured regions** have complete encoded-integer-box contract evidence (LADC-001, CONDITIONAL on the trust base).
2. **0 of 26** have an independently audited transfer to the real-valued region (LADC-004).
3. **Historical:** in the earlier ten-region MNIST campaign, 1 of 10 reached the pipeline's deployed-transfer level (LADC-002, MEASURED, superseded implementation).

Also keep the raw `PARTIAL_VERIFIED` status visible (LADC-003).

**Open points**

- **Author decision:** confirm that the "1 of 10" sentence in the submitted PDF is a carry-over, as the artifacts indicate. This is my inference from the files, not a recorded statement.
- **Blocker:** the run directory and commit of the ten-region campaign were not located. Without them the historical row can only cite the old manuscript.

## C-02. Three relational mechanisms are described as one

**What the sources say**

- D-TEX `3-metodologia.tex` L631–660, Lemma "Cut validity, conditional": cuts are MILP global dual bounds, widened outward by exact rational terms. The argument "is analytical … rather than a per-cut ESBMC obligation", works only for single-hidden-layer networks, and the artifact is "pinned at commit `f3c0fc3`".
- L-PDF §4.6, Eq. (15): each cut is verified by a separate prefix harness before it is assumed.
- S-PDF: no cuts. Table II states "no MILP is used in this campaign".

**What the evidence shows**

| Mechanism | Where it was used | Status |
|---|---|---|
| (i) Analytic MILP dual bound plus widening | Artifact `f3c0fc3a`, 2026-08-02 | CONDITIONAL on MILP bound reporting. Produced none of the reported results. |
| (ii) MILP proposal, then exact-prefix ESBMC validation | LADC matrix campaign, commits `59042988`, `a848dc16` and neighbours, 2026-09-08 to 09-10 | 308 validation queries: 100 VERIFIED, 208 TIMEOUT and discarded. 82.3% of recorded ESBMC time (LADC-008). |
| (iii) Integer CROWN affine chains, each step an ESBMC lemma | SSV campaign | Relational information is carried by the chains, not by cuts. |

Two further facts limit what can be claimed:

- With cuts disabled, all 26 regions keep the same outcome and the same formats, and no refinement round ran (LADC-007). Cuts were not load-bearing in the reported results.
- The only attempt at a relational lemma in the CNN study (two neurons sharing a layer-1 ancestor) timed out in both encodings (DIAG-002, DIAG-003). No relational lemma was proved.

**Resolution**

- Chapter 5 describes mechanism (ii) as the one behind the 26-region results and reports that it did not add certificates.
- Mechanism (i) appears only as a historical remark, tagged CONDITIONAL, and Lemma "Cut validity, conditional" is not cited for any reported result.
- Chapter 6 states that Study B has no cut mechanism.
- Chapter 7 compares the three and reports the inconclusive relational diagnostic as supplementary.
- Theorem "Conditional soundness" premise (iv) and its last sentence ("to the MILP backend's bound validity") must be rewritten for mechanism (ii): the remaining trust is the association of each validated cut with its input box, prefix and formats.

## C-03. "ESBMC discharges every obligation" against what ESBMC checks

**What the sources say**

- D-TEX `3-metodologia.tex` L396: ESBMC "discharges every certifying proof obligation".
- L-PDF conclusion: "every certifying obligation is discharged bit-precisely by ESBMC".
- D-TEX L566–570: hidden-layer harnesses take "nondeterministic integers assumed inside" the assumption box.

**What the evidence shows**

*Study A (L-AUD §2 F4, §3.2–3.4, §6, §7)*

| Element | Who establishes it |
|---|---|
| Hidden-layer contract | ESBMC checks a **deterministic** interval-endpoint program: none of the 3,017 hidden harnesses contains a nondeterministic input (LADC-017). Coverage of the whole box rests on a pen-and-paper endpoint and monotonicity lemma. |
| Output competitors, cut validation | ESBMC with symbolic inputs (767 and 308 queries). |
| Contract chaining | Python integer containment test. |
| Accumulator range | Python exact-integer static check. Paper profiles do not enable `--overflow-check` (LADC-018). |
| No saturation | Not proved: `SKIPPED` in all 166 records. |
| Guarantee level | Derived from records and declarations; not an independent certificate checker. |

*Study B (S-PDF §III-B, §V-B, §VI; WT `P1_checker_closure.md`; S-EVID)*

| Element | Who establishes it |
|---|---|
| Box facts | ESBMC on ground harnesses with literal arrays. |
| Affine, ReLU and rounding lemmas of each chain | ESBMC. |
| Step linkage, dependency closure, margin and clamp closure | Python. |
| Binding of certificate layers to the deployed C arrays | Python; ESBMC corroborates sampled rows only. |
| Accumulator range | Python static bound. |
| Whole deployed file | Never verified by ESBMC as one program: the monolithic run times out at 600 s (S-PDF Table IV). |
| Zero-product affine columns | Missing at submission; checked after submission in 1,510 supplementary harnesses, all VERIFIED (SSV-005). |

**Resolution**

- Replace "every obligation" by an explicit table in Chapter 4 with three columns: checked by ESBMC, checked by trusted Python, proved on paper.
- Tag region-level results CONDITIONAL and the individual solver verdicts PROVED (see the ledger convention in [dissertation_audit.md](dissertation_audit.md) §6).
- Correct the description of the hidden-layer harness in Chapter 5.

## C-04. Synthesis against a fixed global format

**What the sources say**

- D-TEX title, objectives and methodology describe per-layer bit-width synthesis throughout.
- S-PDF contribution 1: "connects per-layer fixed-point quantization, explicit C arithmetic, and local-robustness contracts".
- S-PDF §II-D: "our current methodology does not use formal verification to synthesize the quantization configuration". Table I marks "Synth. quant." as absent for the paper.

**What the evidence shows**

Study B fixes one global Q7.8 format before verification. It was chosen with a 90% validation-accuracy gate; Q3.4 reached 80.75% (SSV-011). No per-layer search ran.

**Resolution**

- Study A is the only synthesis study. Study B is certification of a given format.
- Do not reuse S-PDF contribution 1 verbatim.
- State in Chapter 6 why the Study A search was not applied to the deployed CNN, but only with evidence. Neither paper measures DeepPoly/MILP preimages on that CNN, so the supported wording is that the search "was not applied", not that it "does not scale".
- **Author decision:** project notes record an unpublished pilot (2026-09-17, `experiments/plate_experiments.json`) in which the Study A pipeline was run on a much smaller GTSRB CNN (8×8×3 input, 41.7% test accuracy). It could support the transition between the studies, but it is in neither paper. Use it only if its run records are located and pinned, and label it as an unpublished pilot. I did not audit its artifacts.

## C-05. Formal robustness against measured deployment fidelity

**What the sources say and the evidence shows**

| Claim | Level | Note |
|---|---|---|
| SSV certificates concern the C file with SHA-256 `dd535f54…`, stored as `int64_t` | CONDITIONAL | SSV-001, SSV-002 |
| Compact `int16_t` build equals the certified build | MEASURED | Equal logits on 7,830 validation images only (SSV-015). No certificate. |
| Host and board outputs agree | MEASURED | 7,830 images, all 43 logits (SSV-014). Assumes the compiler preserves C semantics. |
| AAOS | Environment | Static binary run from `adb shell`; no AAOS service or API used (SSV-016). |
| LADC harness and backend "identical by construction" | CONDITIONAL | Three helper functions are shared; the surrounding renderers are separate (LADC-015). |
| LADC Python and C agree | MEASURED | Exact on evaluated test splits (LADC-012). |

Two corpora must stay apart: the 12,630-image test set (accuracy 91.35%) and the 7,830-image validation set (board parity; accuracies 90.36% and 90.24% in S-PDF Fig. 1).

Wording slips in S-PDF to avoid copying:

- "12–20 CPU-hours" are summed per-call elapsed times, 11.8 h and 20.5 h (SSV-008).
- The int16 slowdown is given as 1.27×, but 27.75 / 21.99 = 1.26; Fig. 1(b) shows 27.9 ms (SSV-012).

**Resolution**

Chapter 6 has two clearly separated parts, "formal results" and "measured deployment results", and Chapter 4 draws the proof-boundary figure with solid links only where a proof exists.

## C-06. Input domains

**What the sources say**

- L-PDF §2.3 defines the ball clipped to the valid domain; §4.1 defines it unclipped.
- D-TEX Lemma "Input coverage" (L487–500) claims the integer harness box contains the image of the real region.
- S-PDF §III-A: the region is the set of valid byte images within 1 raw byte.

**What the evidence shows**

- Study A builds endpoints in float32 and rounds to nearest, not outward. L-AUD §3.5 reproduces a toy case in which a real input inside the requested region quantizes to an integer outside the verified box and changes the prediction. No collected LADC region was shown to have this defect, and none was audited (LADC-004).
- L-AUD §4 reports that some first-layer encoded boxes have no varying coordinate after quantization (LADC-024).
- MNIST radii in Study A are 0.25 to 4 raw grey levels. Below one level, the region contains no second valid byte image; it is meaningful only as a real-valued box.
- Study B quantifies over a discrete set (3,072 bytes, each within ±1) through an exact byte encoder.

**Resolution**

- Chapter 4 defines both domains and states that results are not comparable across them.
- Chapter 5 states the encoded-domain scope with every count and gives the lemma only for the endpoints actually used.
- **Blocker:** the number of varying coordinates per region must be extracted from the run records before the MNIST radii are interpreted.
- **Author decision:** whether to restate Study A's MNIST property over byte images (which would need new analysis, not new runs) or keep the real-valued box with its stated limitation.

## C-07. Fixed-point notation

| Source | Format symbol | Width relation |
|---|---|---|
| D-TEX `3-metodologia.tex` L433, L566 | φ = ⟨Q, I, F⟩ | "Q = F + I" (omits the sign bit) |
| D-TEX old sections and Algorithm 1 | ⟨Q, F⟩ and ⟨n, i, f⟩ | not stated |
| D-TEX `2-fundamentos.tex` | affine quantizer with min/max scale and zero point | a different quantization model |
| L-PDF §2.2 | ζ = ⟨I, F⟩ | Q = I + F + 1 |
| S-PDF §III-B | ⟨Q, I, F⟩ | Q = 1 + I + F |

The records agree with Q = 1 + I + F (for example Q = 13, I = 4, F = 8).

Other slips: L-PDF Eq. (3) defines saturation in terms of RHAZ; S-PDF gives the Q7.8 range as [−128, 128], whose upper end is 128 − 2⁻⁸; L-PDF §4.5 uses β both for block size and for the number of blocks.

**Resolution**

One definition, used everywhere: ζ = ⟨I, F⟩, Q = 1 + I + F, scale S = 2^F, saturation to [−2^(Q−1), 2^(Q−1) − 1], rescaling by the predecessor scale. The min/max affine quantizer in the background chapter is presented as general background and explicitly not the scheme used.

## C-08. Claimed use of the SMT theory of fixed-point arithmetic

**What the sources say**

- D-TEX introduction (objectives, contributions), methodology (L3, L9, L156) and conclusion (L3) claim integration of the theory of Baranowski et al.
- D-TEX `2-fundamentos.tex` L545: "This theory was integrated into off-the-shelf ESBMC".
- L-PDF §4 opening: robustness is checked "using the SMT Theory of Fixed-Point Arithmetic".

**What the evidence shows**

The harnesses are C programs over `int64_t` and `__int128`, checked by ESBMC with a bit-vector backend (`--bv`, Z3 or Bitwuzla). No evidence of the dedicated theory was found (SH-005).

**Resolution**

**Blocker.** Unless exact code evidence is produced, write "bit-precise bounded model checking of C integer arithmetic with a bit-vector SMT backend" and cite Baranowski et al. only as related work. The sentence about ESBMC integrating the theory needs a source or must be removed.

## C-09. Two ESBMC configurations inside the dissertation

| Location | Configuration |
|---|---|
| D-TEX `3-metodologia.tex` L185–198 | `--boolector`, `--ir`, `--loop-invariant`, `--incremental-bmc`, `--interval-analysis`, `--no-unwinding-assertions`, `--state-hashing`, plus `--no-bounds-check`, `--no-div-by-zero-check`, `--no-pointer-check` |
| D-TEX `3-metodologia.tex` L787–810 | `--unwind N`, `--z3` or `--bitwuzla`, `--bv`, `--timeout`, `--memlimit`, `--interval-analysis`, `--interval-analysis-simplify`, `--result-only`; "no option disabling assertions or checks is used" |
| D-TEX `4-experimentos.tex` Table 2 | ESBMC 7.5+, Gurobi 9.5+ |
| L-PDF §5.1, S-PDF Table II | ESBMC 7.11.0, Z3 in bit-vector mode; CBC (Study A); Bitwuzla only for the SSV monolithic baseline |

`--ir` selects integer/real arithmetic encoding, which is not bit-precise. It contradicts the thesis of the document if left as the described method.

**Resolution**

Remove the first configuration from the method. If the qualification results are kept as an appendix (C-17), list that configuration there as the setting of the prototype. Each study chapter states its own configuration; interval-analysis flags are listed in the trust base.

## C-10. Comparative and "optimal" claims

| Claim | Location | Problem |
|---|---|---|
| "optimal bit-width" | D-TEX intro L129, L159; methodology L3, L12, L138 | The search accepts the first format that passes; no optimality proof (LADC-021). |
| "clear speed-up compared to CEG4N"; "comparable or better" | D-TEX `2-fundamentos.tex` L743, L816 | Different properties; old implementation; one sample (QUAL-002). |
| "MILP … speed-up of approximately 490×"; "bit-precise reasoning can lead to more robust quantization" | D-TEX `4-experimentos.tex` L138–143 | One sample and one radius; accuracy differences on small test splits (QUAL-001). |
| "The composition … ensures the preservation of soundness" | D-TEX `3-metodologia.tex` L205 | Unqualified; superseded by the conditional theorem. |
| Quadapter "does not account for the bit precision" | D-TEX `3-metodologia.tex` L211 | Should be stated as a difference in verified object. |

L-PDF itself limits the tool comparison to "positioning and capability claims" and notes that tensor identity between tools was not established (LADC-013).

**Resolution**

No superiority or speed-up statement anywhere. Comparison tables carry the property each tool proves in the column header.

## C-11. SSV counts, rerun and protocol history

- **Primary and rerun.** Primary: 1 of 80, with 17 checker-inconclusive. After the disclosed rerun: 2 of 80, with 16. S-PDF leads its abstract with 2 of 80 and its text mentions only 16 aborts. The dissertation leads with the primary result and gives both rows.
- **Protocol history.** S-PROTO was first committed with n = 100 (`92f17ea2`) and redrafted to n = 80 (`049764a4`), the first 80 of the same seeded draw, before any test output. Three amendments changed only the number of parallel jobs. The reduction is not in the paper; the dissertation should disclose it (SSV-017).
- **Post-submission supplement.** The zero-product column check (SSV-005) must be stated next to both results.
- **Validation campaign.** 5 of 20 on validation (SSV-018) is supplementary, pre-specified but not pre-registered, and never pooled with the test result.
- **Finder ceiling.** 18 of 80 is a ceiling, not a result (SSV-004).

## C-12. Certification rates cannot be compared across studies

| Aspect | Study A | Study B |
|---|---|---|
| Region selection | Fixed, hand-configured suite; nested radii around 16 centres | Seeded random draw of 80 test images, fixed in a committed protocol |
| Eligibility filter | DeepPoly source gate; failing regions are excluded | None; 8 misclassified images stay in the denominator |
| Missing runs | 13 matrix runners failed without a report | Every drawn pair has an outcome |
| Result | 26 of 26 encoded contracts | 1 of 80 primary |

L-PDF RQ1 asks about "independently selected input regions"; the revision notes say the regions are "fixed and sometimes nested, not independently sampled".

**Resolution**

Chapter 7 presents the two results in one taxonomy table with separate rows and denominators and states that no rate is comparable. The 100% and the 1.25% describe different populations, input domains and proof formats.

## C-13. Objectives in the dissertation that were not carried out

D-TEX `1-introdução.tex` L154–203 lists backdoor-freeness benchmarks, "bit-precise equivalence verification", adapting the MILP formulation to SMT, and formalizing the SMT theory. None appears in either paper.

**Resolution**

**Author decision:** replace the objectives with achieved research questions (the guide proposes RQ1–RQ4). The project title "Network Equivalence" and the cover title should be reconsidered at the same time.

## C-14. Error budget and chaining rule

- L-PDF Eq. (8): the post-activation guarantee is G_act = ρ(G_pre), with G_act ⊆ A_in of the next layer, optionally intersected with a separately verified affine enclosure.
- D-TEX Eq. (chaining): ReLU(G ⊕ Δ) ⊆ A of the next layer, that is, the guarantee is inflated by the budget before chaining.
- L-AUD §3.2: the implemented rule is A_l = ReLU(G_l) ∩ ReLU(R_l).
- L-PDF Eq. (9): the first term does not depend on the neuron index (review item A.2 in `02_TRACEABILITY.md`).

**Resolution**

**Blocker (code check, no new experiment):** confirm against the campaign revision which chaining rule and budget formula ran, then state that one. L-AUD §3.4 notes that with every competitor checked exactly, the budget is not the logical basis of the integer conclusion; say so.

## C-15. "Monolithic" and "block"

| Term | Study A | Study B |
|---|---|---|
| Monolithic | β = 0: one query per hidden layer | One ESBMC run on the whole deployed C |
| Block | β output neurons of one dense layer | 24-neuron blocks in the box stage; chains in the robustness stage |

**Resolution**

Define both in Chapter 4 and never present the two baselines in one column.

## C-16. Source versions

The instruction file names PDFs "(8)", "(9)" and "(10)" with 44, 17 and 6 pages. The folder holds "(6)", "(1)" and "(2)" with 66, 17 and 6 pages.

- The LaTeX source builds to 44 pages because `000-main.tex` comments out `\input{capitulos/2-fundamentos}`. With that line active it matches the 66-page PDF.
- Every count quoted in the integration guide matches PDFs "(1)" and "(2)".

**Resolution**

**Author decision:** confirm that "(1)" and "(2)" are the submitted texts, and whether the background chapter was removed on purpose.

## C-17. Qualification-era results

D-TEX `4-experimentos.tex` and `2-fundamentos.tex` §"The proposal work against state-of-the-art works" report Iris and Seeds tables from the January 2026 prototype (QUAL-001, QUAL-002).

- No run directory or commit was located.
- `anotacoes.md` describes a different kernel (bias added before rescaling) and the chapter lists `--ir`.
- Two tables report the same experiment with different timings.
- Cells marked "---" are described as time-limit problems, but their totals are 5 to 14 s.

**Resolution**

**Blocker** for keeping them in the main text. Recommended: remove from the main text; keep at most one table in an appendix labelled as prototype results under different semantics.

## C-18. Authorship, publication status and reuse

- L-PDF lists five authors including the advisor. S-PDF lists five authors from one institute; the advisor is not among them.
- The acceptance status of either paper is not recorded in the repository. S-EVID refers to a camera-ready deadline, which suggests acceptance of the SSV paper but is not a statement of it.
- D-TEX reuses a figure from Zhang et al. (`preimage_backpropagation.png`) and a figure from Goodfellow et al.

**Resolution**

**Author decision:** state each paper's status exactly as known, describe the author's own contribution to each, and check reuse permissions for text and figures.

## C-19. LADC solver backend

**What the sources say**

L-PDF §5.1: "ESBMC 7.11.0 was used with the Z3 SMT solver in bit-vector mode".

**What the evidence shows**

L-REC `per_run.csv`, column `effective_solver_profile`, for the 26 main runs: the 13 MNIST regions used `paper-z3`; the 10 Iris and 3 Seeds regions used `paper-fast`. In `src/verification/esbmc.py` (line 196, current checkout) `paper-fast` selects `--bitwuzla`. `docs/ladc2026_review/03_EXPERIMENT_MATRIX.md` states the same split as the campaign design.

**Resolution**

Chapter 5 reports the solver per dataset: Z3 for MNIST, Bitwuzla for Iris and Seeds. The solver is part of the trust base, so the trust table lists both. Confirm the profile mapping at the campaign revision (`59042988`) when the chapter is written; I read it from the current checkout only.

