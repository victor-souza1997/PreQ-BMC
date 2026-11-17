# Proposed dissertation structure and migration map

Phase 0 audit, 2026-11-16. This is a proposal; nothing in the LaTeX tree was changed.

Related files: [dissertation_audit.md](dissertation_audit.md), [scientific_conflicts.md](scientific_conflicts.md) (C-xx), [evidence_ledger.csv](evidence_ledger.csv) (claim IDs), [dissertation_edit_plan.md](dissertation_edit_plan.md).

All paths below are relative to `latex_source/Master_Dissertation___Network_Equivalence/`.

---

## 1. Governing thesis

Proposed thesis statement (from the integration guide, unchanged):

> This dissertation investigates how neural-network abstraction and compositional software model checking can provide bit-precise robustness guarantees for generated fixed-point implementations, from preimage-guided quantization synthesis for small dense networks to deployment-aware certification of a larger convolutional classifier, while characterizing the practical limits of proof scalability and embedded execution.

One caution on wording: "bit-precise robustness guarantees" must be read with the trust boundary of Chapter 4. A safer final clause for the abstract is "guarantees for explicitly stated obligations of generated fixed-point C".

The title on the cover ("… with SMT Solvers") and the project name ("Network Equivalence") no longer describe the work. A title is an author and advisor decision (C-13).

**Rules that hold in every chapter**

- Two studies, two candidate-generation methods, two certificate formats. No unified algorithm and no pooled rate.
- Every result carries its proof level and its denominator.
- Study A results are stated for the encoded integer box. Study B results are stated for byte images at ε = 1.
- Post-submission work is supplementary and outside both frozen denominators.

---

## 2. Chapter-to-file map

New files use ASCII names. Existing files are kept untouched as an archive until the new build is accepted.

| Ch. | Title | New file | Nature | Main sources |
|---|---|---|---|---|
| — | Front matter | `capitulos/0-capa_n_contra.tex` (edited), `0-glossary.tex` (edited) | Replace | — |
| 1 | Introduction | `capitulos/1-introduction.tex` | Rewrite, written last | `1-introdução.tex`; L-PDF §1; S-PDF §I |
| 2 | Background | `capitulos/2-background.tex` | Condense and extend | `2-fundamentos.tex`; L-PDF §2; `3-metodologia.tex` L106–126 |
| 3 | Related Work | `capitulos/3-related-work.tex` | New, assembled | L-PDF §3; S-PDF §II; `3-metodologia.tex` L403–426; commented tables in `1-introdução.tex` and `5-conclusão.tex` |
| 4 | Deployment-Aware Verification Framework | `capitulos/4-framework.tex` | New synthesis | `3-metodologia.tex` L428–500, L582–629, L683–736; L-PDF §4.1; S-PDF §III-A to III-C |
| 5 | Study A: Preimage-Guided Quantization Synthesis for Dense Networks | `capitulos/5-study-a-synthesis.tex` | Rewrite from paper | L-PDF §4–5; `3-metodologia.tex` L502–681, L738–810; L-AUD; L-REC |
| 6 | Study B: Integer-CROWN Certification of a CNN and Execution on Android Automotive | `capitulos/6-study-b-cnn.tex` | Rewrite from paper | S-PDF §III–VI; S-EVID; S-PROTO; `docs/ssv2026/methodology.md` |
| 7 | Discussion, Scalability, Incompleteness and Threats to Validity | `capitulos/7-discussion.tex` | New synthesis | Ledger; worktree notes P1–P3c, I-P, I-R; `4-experimentos.tex` L146–176 (structure only) |
| 8 | Conclusions and Future Work | `capitulos/8-conclusion.tex` | Rewrite, written last | Chapters 4–7 |
| A | Preliminary Prototype Results (optional) | `capitulos/A-preliminary-results.tex` | Demoted, pending B-4 | `4-experimentos.tex`; `2-fundamentos.tex` L699–879 |
| B | Publications | `capitulos/B-publications.tex` | Rewrite | `6-apendice.tex` |

---

## 3. Chapter outlines

Action tags: **RETAIN** (keep with light edits), **REWRITE**, **RELOCATE** (move from another place), **NEW**, **REMOVE**.

### Chapter 1 — Introduction

| § | Section | Action | Source | Constraints |
|---|---|---|---|---|
| 1.1 | Embedded inference and fixed-point quantization | REWRITE | `1-introdução.tex` L25–60, condensed to about one page | Drop the GPT-scale opening |
| 1.2 | The implementation gap | REWRITE | `1-introdução.tex` L127–134; L-PDF §1; S-PDF §I | No "optimal bit-width"; fix author names (audit §3.1) |
| 1.3 | Research problem and scope | NEW | Guide §2 | Dense ReLU networks and one CNN; local L∞ robustness only |
| 1.4 | Research questions | NEW | Guide RQ1–RQ4 | Replace the objectives list (C-13) |
| 1.5 | Contributions and relation to the two papers | NEW | Guide §2; C-18 | State what was proved, measured and assumed; name co-authors |
| 1.6 | Organization | REWRITE | — | Eight chapters |

Remove: `1-introdução.tex` L154–203 (objectives, contributions) and all commented blocks.

### Chapter 2 — Background

| § | Section | Action | Source | Constraints |
|---|---|---|---|---|
| 2.1 | Feed-forward and convolutional networks | RETAIN and extend | `2-fundamentos.tex` L5–56, L229–242; L-PDF §2.1 | Add convolution and pooling as affine layers (needed for Ch. 6). TODO source for the CNN description. |
| 2.2 | Quantization and signed fixed-point formats | REWRITE | L-PDF §2.2; S-PDF §III-B; `2-fundamentos.tex` L244–369 as general background only | ζ = ⟨I, F⟩, Q = 1 + I + F (C-07). State that the min/max affine scheme is not the one used. |
| 2.3 | Fixed-point arithmetic in C | NEW | L-PDF Eq. (5); S-PDF §III-B | Operation order only; the formal definition lives in Ch. 4 |
| 2.4 | Adversarial examples and local robustness | RETAIN | `2-fundamentos.tex` L434–498; L-PDF §2.3 | Remove the blue note and the informal equation; check the Goodfellow attribution |
| 2.5 | SMT, bounded model checking and ESBMC | REWRITE | `2-fundamentos.tex` L539–577; L-PDF §2.4; `3-metodologia.tex` L150–174 | No claim about the fixed-point theory (C-08) |
| 2.6 | Abstract interpretation and DeepPoly | NEW, short | `3-metodologia.tex` L106–112; L-PDF §4 | Role: forward bounds |
| 2.7 | MILP encodings and backward preimages | RELOCATE | `3-metodologia.tex` L114–126; L-PDF §4 | Role: proposals; figure reuse needs permission (C-18) |
| 2.8 | CROWN-style linear bound propagation | NEW, short | S-PDF §II-C | Describe real-valued CROWN only; the integer variant belongs to Ch. 6 |
| 2.9 | Assume-guarantee reasoning | NEW, short | L-PDF §4.2; S-PDF §III-C | — |

Remove: `2-fundamentos.tex` L57–227 (commented training material), L371–431, L579–693 (commented blocks).

### Chapter 3 — Related Work

| § | Section | Action | Source |
|---|---|---|---|
| 3.1 | Verification of real-valued networks | NEW | Reluplex, Marabou, DeepPoly, CROWN |
| 3.2 | Verification of quantized networks | RELOCATE | `2-fundamentos.tex` L545–549; L-PDF §3; S-PDF §II-B |
| 3.3 | Quantization synthesis and repair | RELOCATE | `3-metodologia.tex` L403–426; L-PDF §3 (Quadapter, CEG4N, QNNRepair, QA-IBP) |
| 3.4 | Program-level verification and the implementation gap | RELOCATE | L-PDF §3 (Cordeiro et al., QNNVerifier) |
| 3.5 | Fixed-point inference on embedded platforms | RELOCATE | S-PDF §II-A |
| 3.6 | Positioning | NEW | Merge L-PDF Table 1 and S-PDF Table I into one matrix |

Constraints: columns of the matrix state the verified object and property of each tool; no superiority claim (C-10). Several entries are missing from `referencias.bib` (audit §3.1).

### Chapter 4 — Deployment-Aware Verification Framework (new synthesis)

| § | Section | Action | Source | Constraints |
|---|---|---|---|---|
| 4.1 | Verified object and deployment identity | NEW | S-PDF §III-A, §III-B; L-PDF §4.4 | Generated C source; identity by shared helpers (A) and by file hash (B) |
| 4.2 | Property and input domains | RELOCATE and rewrite | `3-metodologia.tex` L470–500; L-PDF Eq. (4); S-PDF Eq. (1)–(2) | Strict inequality; two domains (C-06); coverage lemma scoped to the endpoints used |
| 4.3 | Fixed-point execution semantics | RELOCATE | `3-metodologia.tex` L428–468; L-AUD §3.1 | One definition; fix Q = F + I; overflow handled by a trusted static check |
| 4.4 | Candidate generation and formal discharge | NEW | SH-003 | Proposals are not proofs |
| 4.5 | Obligations, composition and closure | RELOCATE and generalize | Lemmas "Block conjunction" and "Output decomposition" (`3-metodologia.tex` L592–629); S-PDF Eq. (3)–(6) | Shared elementary results only |
| 4.6 | Trust boundary | NEW | C-03; L-AUD §3.4; worktree `P1_checker_closure.md` | Table: checked by ESBMC, trusted Python, paper proof, measured, assumed; one column per study |
| 4.7 | Two specializations | NEW | Audit §4.2 | Two branches that never merge |
| 4.8 | Outcome taxonomy and provenance | RELOCATE | `3-metodologia.tex` L695–724 (status table); S-PROTO `outcomes` | One table of outcomes per study; definitions of "monolithic" and "block" (C-15) |

Figures: one overview with two branches converging on "ESBMC obligations plus stated composition"; one proof-boundary figure (solid links proved, dashed links measured or assumed).

### Chapter 5 — Study A (LADC)

| § | Section | Action | Source | Constraints |
|---|---|---|---|---|
| 5.1 | Scope | REWRITE | L-PDF §5.1 | Small dense ReLU networks |
| 5.2 | Source prerequisite and backward preimages | RELOCATE | `3-metodologia.tex` L505–513; L-PDF §4.2 | Gate is an eligibility filter (LADC-020) |
| 5.3 | Candidate formats and the derived error budget | REWRITE | L-PDF §4.3; `3-metodologia.tex` L535–549 | Pending C-14; no optimality (LADC-021) |
| 5.4 | Layer contracts, chaining and the output proof | REWRITE | L-PDF §4.2, §4.4, §4.7; L-AUD §3.2 | Hidden harness is a deterministic endpoint check (LADC-017); endpoint lemma stated as a paper result |
| 5.5 | Block-wise verification | RETAIN | `3-metodologia.tex` L582–603 | Refers to the shared lemma in Ch. 4 |
| 5.6 | Relational cuts | REWRITE | L-PDF §4.6; L-AUD §3.2, §6 | Mechanism (ii) only; mechanism (i) as a historical remark (C-02) |
| 5.7 | Conditional soundness | REWRITE | `3-metodologia.tex` L738–759; L-AUD §3.3 | Seven premises of the audit; CONDITIONAL (LADC-019) |
| 5.8 | Experimental setup | REWRITE | L-PDF §5.1; LADC-022 | Solver per dataset (C-19) |
| 5.9 | Results | REWRITE | L-PDF Tables 2–4, Fig. 2; L-REC | Three separate statements of C-01; raw status shown; MNIST 100 separate (LADC-014) |
| 5.10 | Limits of the study | NEW | L-AUD §1; LADC-004, LADC-024 | Encoded domain; pre-filtered cohort; single timing observations |

Remove from the chapter: end-to-end modes as an evaluated feature (disabled in the campaign), the old Algorithm 1, the old ESBMC flag list (C-09).

### Chapter 6 — Study B (SSV)

| § | Section | Action | Source | Constraints |
|---|---|---|---|---|
| 6.1 | From dense networks to a CNN | NEW | C-04 | "Was not applied", unless the pilot is pinned |
| 6.2 | Network, global Q7.8 format and byte input domain | REWRITE | S-PDF §III-B, §IV-A | Format selected by validation accuracy (SSV-011) |
| 6.3 | Interval box facts and integer CROWN chains | REWRITE and expand | S-PDF §III-C; `docs/ssv2026/methodology.md` | TODO source: the paper has no formal statement of chain-step soundness. Do not invent one. |
| 6.4 | Harnesses, dependency closure and margin closure | REWRITE and expand | S-PDF §III-C, §V-A; worktree `P1_checker_closure.md` | Trust list of SSV-006 |
| 6.5 | Protocol | REWRITE | S-PDF §IV; S-PROTO | n = 100 redrafted to 80; resource-only amendments (SSV-017) |
| 6.6 | Certification results | REWRITE | S-PDF Table III | Primary first, rerun second, P2 supplement stated (C-11) |
| 6.7 | Verification cost | REWRITE | S-PDF Table IV | Summed elapsed time, not CPU-hours (SSV-008) |
| 6.8 | Deployment measurements | REWRITE | S-PDF §V-C, §V-D, Fig. 1 | Measured only; two corpora kept apart (C-05) |
| 6.9 | Limits of the study | REWRITE | S-PDF §VI; S-EVID §7 | AAOS is the environment (SSV-016) |

### Chapter 7 — Discussion (new synthesis)

| § | Section | Source | Constraints |
|---|---|---|---|
| 7.1 | The two workflows compared | Audit §4 | Table A of the guide |
| 7.2 | What "verified" means at each level | C-01, C-11 | Taxonomy table, separate denominators |
| 7.3 | Scalability: smaller queries against many obligations | LADC-005/006; SSV-007/008/009 | Qualitative; no cross-study ratio |
| 7.4 | Incompleteness | Audit §5 item 5 | Not evidence of non-robustness |
| 7.5 | Post-submission diagnostics | DIAG-001 to DIAG-007; SSV-005; SSV-018 | Supplementary; pending B-5 |
| 7.6 | Proof boundary and the implementation gap | C-03, C-05 | Refers to the Ch. 4 trust table |
| 7.7 | Threats to validity | `4-experimentos.tex` L146–176 for structure; S-PDF §VI; L-AUD | Internal, external, construct |

### Chapter 8 — Conclusions and Future Work

Answers RQ1–RQ4 one by one, each with a short "proved / measured / conditional / unknown" statement. Future work only as directions (S-EVID §4). The activity schedule and the future-tense text of `5-conclusão.tex` are removed.

---

## 4. Migration table for existing sections

| Existing location | Lines | Action | Destination |
|---|---|---|---|
| `000-main.tex` class option, inputs | 1, 107–111 | Edit | `msc`; new chapter inputs |
| `0-capa_n_contra.tex` | all | Rewrite | Front matter |
| `0-glossary.tex` | all | Extend | Add BMC, CNN, AAOS, RHAZ, GTSRB, and others in use |
| `1-introdução.tex` opening | 19–78 | Condense | §1.1 |
| `1-introdução.tex` problem statement | 124–146 | Rewrite | §1.2 |
| `1-introdução.tex` objectives, contributions, structure | 154–207 | Remove and replace | §1.4–1.6 |
| `1-introdução.tex` commented table of related work | 215–296 | Mine for content | Ch. 3 |
| `2-fundamentos.tex` neural networks | 5–56, 229–242 | Retain, condense | §2.1 |
| `2-fundamentos.tex` quantization | 244–369 | Rewrite | §2.2 |
| `2-fundamentos.tex` safety and robustness | 434–498 | Retain with fixes | §2.4 |
| `2-fundamentos.tex` formal verification | 500–577 | Split | §2.5 and Ch. 3 |
| `2-fundamentos.tex` comparison with CEG4N | 699–879 | Remove from main text | Appendix A or delete (B-4) |
| `3-metodologia.tex` opening, old Algorithm 1 | 1–71 | Remove | — |
| `3-metodologia.tex` input region | 73–104 | Check against LADC, then relocate | §5.8 |
| `3-metodologia.tex` symbolic bounds, preimage | 106–126 | Relocate | §2.6, §2.7 |
| `3-metodologia.tex` old search, code generation | 128–183 | Remove; keep the nondeterministic-input example | §2.5 |
| `3-metodologia.tex` old ESBMC execution | 185–205 | Remove | Appendix A at most (C-09) |
| `3-metodologia.tex` "Central Problem" | 206–211 | Remove | — |
| `3-metodologia.tex` "Algortihm" macros and comments | 214–380 | Remove | — |
| `3-metodologia.tex` overview | 382–401 | Rewrite | §4.4 |
| `3-metodologia.tex` positioning | 403–426 | Relocate | Ch. 3 |
| `3-metodologia.tex` notation, semantics, coverage lemma | 428–500 | Relocate with fixes | §4.2, §4.3 |
| `3-metodologia.tex` contracts, chaining | 502–558 | Rewrite | §5.2–5.4 |
| `3-metodologia.tex` search, blocks, output | 560–629 | Split | §5.3, §5.5; shared lemmas to §4.5 |
| `3-metodologia.tex` relational invariants | 631–660 | Rewrite | §5.6 |
| `3-metodologia.tex` end-to-end modes | 662–681 | Reduce to a remark | §5.10 |
| `3-metodologia.tex` status table | 695–724 | Relocate | §4.8 |
| `3-metodologia.tex` fidelity lemma, theorem | 726–759 | Rewrite | §4.1, §5.7 |
| `3-metodologia.tex` algorithm | 761–785 | Reconcile with LADC Algorithm 1 | §5.3 |
| `3-metodologia.tex` environment | 787–810 | Rewrite | §5.8 |
| `4-experimentos.tex` setup, results, analysis | 6–144, 178–222 | Remove from main text | Appendix A or delete (B-4) |
| `4-experimentos.tex` threats to validity | 146–176 | Reuse structure only | §7.7 |
| `5-conclusão.tex` | 1–38 | Remove and rewrite | Ch. 8 |
| `5-conclusão.tex` commented classification table | 40–136 | Mine for content | Ch. 3 |
| `6-apendice.tex` | all | Rewrite | Appendix B |
| `7-related_works.tex` | all | Mine for content, then archive | Ch. 3 |
| `safety_against_adversarial_attck.tex`, `smt_verification.tex`, `00-plano-de-trabalho.tex`, `Research/00_repairer.tex` | all | Archive | — |
| `figuras/3-metodologia/main_diagram_preqbmc_pdf.drawio.pdf`, `diagram_v1.png` | — | Replace | New Ch. 4 overview |

---

## 5. Formal statements: inventory and status

| Statement | Current location | Destination | Status | Note |
|---|---|---|---|---|
| Lemma, input coverage | `3-metodologia.tex` L487 | §4.2 | CONDITIONAL | Holds for the endpoints given to it (C-06) |
| Lemma, block conjunction | L592 | §4.5 | PROVED (paper) | Shared by both studies |
| Lemma, output decomposition | L621 | §4.5 | PROVED (paper) | Shared |
| Lemma, endpoint extrema over a box | L-AUD §3.2 only | §5.4 | PROVED (paper) | Exists only in the audit note; must be written out and checked by the advisor |
| Lemma, chaining composition | L551 | §5.4 | CONDITIONAL | Reconcile the rule first (C-14) |
| Lemma, cut validity (analytic) | L651 | §5.6, as a remark | CONDITIONAL | Historical; not used for results |
| Statement, validated prefix cut | L-PDF Eq. (15); L-AUD §3.2 | §5.6 | PROVED per verified query | New wording, sourced |
| Lemma, arithmetic correspondence | L726 | §4.1 | CONDITIONAL | Downgrade the wording (C-05) |
| Theorem, conditional soundness (Study A) | L738; L-AUD §3.3 | §5.7 | CONDITIONAL | Use the audit's seven premises |
| Composition of S-PDF Eq. (3)–(6) | S-PDF §III-C | §6.4 | PROVED (paper, elementary) | State as a proposition with the trust list as premises |
| Soundness of an integer CROWN chain step | not stated in any source | §6.3 | — | TODO: source required. Do not write a theorem without it. |

No other theorem should be added.

---

## 6. Tables and figures

| Item | Chapter | Content | Source |
|---|---|---|---|
| Table A, scope of the two studies | 7.1 (forward reference from 4.7) | Objective, architecture, dataset, format, proposal method, obligation types, composition, property, population | Audit §4 |
| Table B, result taxonomy | 7.2 | Rows: dense encoded contracts 26/26; historical ten-region campaign 1/10; CNN primary 1/80; CNN with rerun 2/80; validation-only 5/20; diagnostics. No totals. | Ledger |
| Table C, scalability | 7.3 | Two panels: Study A block size; Study B monolithic and decomposed | L-PDF Table 4; S-PDF Table IV |
| Table D, deployment | 6.8 | float32, certified int64, compact int16: accuracy, latency, memory, parity scope, formal status | S-PDF Fig. 1; S-EVID §3 |
| Trust-boundary table | 4.6 | Per study: ESBMC, Python, paper proof, measured, assumed | C-03 |
| Figure, overview | 4 | Two branches | New |
| Figure, proof boundary | 4 | Source network, generated C, harnesses, Python composition, binary, AAOS | New |
| Figures, block cost | 5.9 | L-PDF Fig. 2 | `docs/LADC_2026/results_revision/generated/block_cost.pdf` |
| Figure, deployment | 6.8 | S-PDF Fig. 1 | `docs/ssv2026/vim3_deployment_results_v2.pdf` |

Each table, figure and algorithm gets one label and one home.

---

## 7. Notation

| Symbol | Meaning |
|---|---|
| 𝒩, 𝒩̂ | Source network; deployed fixed-point network |
| ζ^[l] = ⟨I^[l], F^[l]⟩, Q^[l] = 1 + I^[l] + F^[l] | Layer format and total width |
| S^[l] = 2^(F^[l]) | Scale |
| RHAZ, sat_Q | Round half away from zero; signed saturation |
| B_ε(x₀) | Declared region; defined separately for each study |
| A^[l], G^[l] | Assumption and guarantee of layer l |
| δ^[l] | Derived error budget (Study A only) |
| β | Block size in Study A only; the number of blocks gets a different letter |
| t, j | Target class; competitor |

The existing macros `\fmt`, `\Abox`, `\Gbox`, `\dbud`, `\Nfp` in `3-metodologia.tex` L216–220 should move to the preamble and be aligned with this table.
