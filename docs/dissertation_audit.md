# Dissertation audit: current state and comparison of the two studies

Phase 0 audit, 2026-11-16. Read-only. No LaTeX file was edited, no experiment was run, nothing was committed.

Companion files:

- [scientific_conflicts.md](scientific_conflicts.md): conflict and blocker register (C-01 to C-19).
- [evidence_ledger.csv](evidence_ledger.csv): 61 claims with source, proof level, denominator and allowed wording.
- [dissertation_structure.md](dissertation_structure.md): eight-chapter structure and migration map.
- [dissertation_edit_plan.md](dissertation_edit_plan.md): file-by-file editing sequence.

Source abbreviations are defined at the top of [scientific_conflicts.md](scientific_conflicts.md).

---

## 1. Main findings

1. **The dissertation is still the January 2026 qualification text.** Its objectives, contributions, experiments and conclusion describe planned work. Only one section, "The PreQ-BMC Methodology" in `3-metodologia.tex`, reflects the later research, and it describes an August 2026 artifact, not the campaign reported in the LADC paper.
2. **The two papers share a problem and an arithmetic semantics, not an algorithm.** Study A synthesizes per-layer formats for small dense networks from DeepPoly bounds and MILP preimages. Study B certifies one fixed Q7.8 CNN from interval facts and integer CROWN chains. Their certificates, checkers, input domains and sampling protocols differ.
3. **Neither study supports "ESBMC proves everything".** In both, ESBMC discharges local C obligations, and trusted Python code or a paper argument composes them into the region-level claim (C-03).
4. **The LADC "1 of 10" figure comes from a superseded campaign** and should not be read as a stricter criterion applied to the 26 regions (C-01).
5. **Five blockers** need information from the author before the affected text is written (§7).

---

## 2. Materials inspected

| Material | Location | State |
|---|---|---|
| Instruction file | `latex_source/CLAUDE_Dissertation_Instructions.md` | Read in full. There is no `CLAUDE.md` in the repository; this file was treated as the project instructions. |
| Integration guide | `latex_source/Dissertation_Integration_Guide.md` | Read in full. Treated as a plan, not as evidence. |
| Dissertation source | `latex_source/Master_Dissertation___Network_Equivalence/` | All chapter files read. Built in a scratch copy: 44 pages, no LaTeX errors. |
| Dissertation PDF | D-PDF "(6)", 66 pages | Same source with the background chapter enabled. |
| LADC paper | L-PDF "(1)", 17 pages | Read in full. |
| SSV paper | S-PDF "(2)", 6 pages | Read in full. |
| LADC records | L-REC, L-AUD, `docs/ladc2026_review/` | Per-run CSV tabulated; audit response read (§1–§8). |
| SSV records | S-EVID, S-PROTO, `docs/HANDOFF_20261031.md`, worktree notes P1, P2, I-P, I-R | Read. Protocol history checked with `git log`. |

**Not inspected:** the run directories under `output/` (external disk). Numbers that rest only on notes are marked in the ledger's `open_issue` column.

**Version mismatch (C-16):** the instruction file names PDFs "(8)", "(9)", "(10)"; the folder holds "(6)", "(1)", "(2)".

---

## 3. Audit of the existing dissertation

### 3.1 Build and front matter

| Item | Finding | Class |
|---|---|---|
| Document class option | `mscexam` (qualification exam). Must become `msc`. | Outdated |
| Cover | "Exame de Qualificação", date 01/2026, two empty examiner entries | Outdated |
| Abstract | Empty in both languages; only keywords | Incomplete |
| Dedication, acknowledgements | Placeholder text and an empty chapter | Incomplete |
| Language | `babel` loads `english,brazil`, so the active language is Portuguese; glossary title and table headers are in Portuguese | Inconsistent |
| Background chapter | `\input{capitulos/2-fundamentos}` is commented out | Causes "Chapter ??" in the introduction and "Section ??" in the conclusion |
| Citations | Nine literal "[CITATION NEEDED]" strings in `3-metodologia.tex` | Incomplete |
| Bibliography | 81 entries, 57 cited, no duplicate keys, two BibTeX warnings. Style is `apalike`, not the UFAM style shipped in `cls_bst/` | Check against university rules |
| Misleading keys | `cai2025certified` is Zhang et al. (Quadapter, FM 2024); `abdi2021counterexample` is Matos Jr. et al. (CEG4N, 2022); `han2020understanding` is Hashemi et al. (2017). The text then writes "Cai et al." and "Abdi et al.", which is wrong. | Technically inaccurate |
| Missing entries | CROWN (Zhang et al. 2018), GTSRB (Stallkamp et al. 2011), QA-IBP (Lechner et al. 2023), CBC, CBMC (Clarke et al. 2004), SMT handbook chapter (Barrett and Tinelli), Bishop 2006, ESBMC 7.4 tool paper, and the SSV-only references (Goyal, Romano, Novac, Mistry, Humayed, Kukkala, Lazzaroni, Khadas) | Incomplete |
| Preamble | Several packages loaded two or three times; `\geometry{margin=1in}` overrides the class margins | Check against template |
| Link colour | `hyperref` sets internal links and citations to blue | Conflicts with using blue for suggestions; see the edit plan |

### 3.2 Chapter-by-chapter assessment

**`1-introdução.tex` (377 lines; about 240 are comments)**

| Lines | Content | Assessment |
|---|---|---|
| 19–78 | General AI growth, compression, QNN verification history | Outdated emphasis. Leads with model scale (GPT-3/4), not with the implementation gap. Condense. |
| 124–146 | Problem statement | Partly reusable. Contains "optimal quantization bit-width" and a confused sentence on PSPACE-hardness. States the goal as integrating the SMT fixed-point theory (C-08). |
| 154–190 | Objectives | Outdated. Backdoor-freeness, equivalence verification and preimage-inclusion checking were not carried out (C-13). |
| 195–203 | Contributions | Outdated. "A formalization of the SMT Theory…" and "an adaptation of the MILP formulation … to an SMT formulation" are not supported. |
| 205–207 | Structure | Outdated; refers to five chapters, one of which is not compiled. |
| 210–377 | Commented notes and a Portuguese table of related work | Remove from the source; the table is raw material for Chapter 3. |

**`2-fundamentos.tex` (879 lines; not compiled)**

| Lines | Content | Assessment |
|---|---|---|
| 1–243 | Neural networks | Reusable after condensing. No convolutional layer is described, which Chapter 6 needs. |
| 244–369 | Quantization | Technically inaccurate for this work. It defines a min/max affine quantizer with zero point, and the text after Eq. (quantization) describes a different formula from the one displayed. The scheme actually used is a power-of-two fixed-point format (C-07). One sentence calls floating-point "fake quantization" the "most crucial aspect … in our scope", which contradicts the thesis. |
| 434–498 | Safety and robustness | Reusable. Contains a leftover blue note (L485) and an informal equation. One sentence has a double negative ("robustness is not nonmonotonic"). The claim attributed to Goodfellow et al. about 4-bit and 5-bit quantization needs its source checked. |
| 500–577 | Formal verification and SMT encoding | Partly reusable. The sentence that the fixed-point theory "was integrated into off-the-shelf ESBMC" is unsupported (C-08). The safety formula uses "∃ … →", which is not the intended statement. Nothing on DeepPoly, MILP preimages, CROWN, bounded model checking internals or assume-guarantee reasoning. |
| 699–879 | "The proposal work against state-of-the-art works" | Redundant with `4-experimentos.tex` and unsupported as a comparison (C-10, C-17). One table caption is in Portuguese. |

**`3-metodologia.tex` (810 lines)**

| Lines | Content | Assessment |
|---|---|---|
| 1–71 | Chapter opening, old Algorithm 1, methodology figure | Outdated. "Optimal bit-width", "SMT Theory of Fixed-Point Arithmetic", "Cai et al.". Superscripts such as `f^{2[l]}` are malformed. |
| 73–104 | Input region | Partly reusable for Study A (per-feature radius for tabular data). Check against the LADC setup, which states radii directly. |
| 106–126 | Symbolic bounds, backward preimage | Reusable as background for Study A; too brief. The figure is taken from Zhang et al. |
| 128–183 | Quantization search, code generation, safety properties | Outdated. Describes rounding to nearest without saturation and a preimage assertion with an `eps` slack. |
| 185–205 | ESBMC execution | Technically inaccurate for the reported work (C-09): `--ir`, disabled checks, solver name misspelled. |
| 206–211 | "Central Problem" | Redundant with the introduction. |
| 214–380 | "Algortihm" (misspelled): macro definitions and three commented algorithms | Remove. |
| 382–401 | PreQ-BMC overview | Good basis, but "ESBMC discharges every certifying proof obligation" overstates (C-03). |
| 403–426 | Positioning | Move to Chapter 3; all citations are placeholders. |
| 428–500 | Notation, deployed semantics, Lemma "Input coverage" | Strong basis for Chapter 4. Fix Q = F + I (C-07), the argmax form of the property, and the scope of the lemma (C-06). |
| 502–558 | Contracts, budget, Lemma "Chaining" | Basis for Chapter 5. Chaining rule disagrees with the paper and the audited code (C-14). |
| 560–629 | Format search, blocks, output obligation, Lemmas "Blocks" and "Output" | Basis for Chapter 5. The hidden-harness description is inaccurate for the campaign (C-03). |
| 631–660 | Relational invariants, Lemma "Cut validity, conditional" | Describes a mechanism that produced no reported result (C-02). |
| 662–681 | End-to-end modes | End-to-end verification was disabled throughout the LADC campaign (L-AUD F7). Keep only as an optional mode, clearly marked as not evaluated. |
| 683–759 | Deployment stages, status table, Lemma "Arithmetic correspondence", Theorem "Conditional soundness" | Keep with corrections (C-02, C-05). Premise (vii) is not a logical premise. |
| 761–785 | Algorithm (layer-contract mode) | Keep one algorithm for Study A; reconcile with LADC Algorithm 1. |
| 787–810 | Environment | Describes the August artifact (`f3c0fc3`). The last sentence promises a "Sect. 4" evaluation of five networks that the current Chapter 4 does not contain. |

**`4-experimentos.tex` (222 lines)**

Entirely qualification-era (C-17). The table is duplicated with different timings; the dataset description is garbled ("two hidden layers containing 10, 15, 3, and 4 neurons, respectively"); the software table is outdated; column headers are in Portuguese; the analysis draws conclusions the data do not support (C-10). The threats-to-validity structure (internal, external, construct) can be reused in Chapter 7.

**`5-conclusão.tex` (136 lines)**

Written in the future tense ("is expected to make a considerable contribution"). Contains the activity schedule with deadlines. The commented classification table (L40–136) is raw material for Chapter 3.

**Other files**

| File | Assessment |
|---|---|
| `7-related_works.tex` | Reading notes, partly in Portuguese, with a broken `\chapter` command. Not compiled. Raw material only. |
| `6-apendice.tex` | One Portuguese sentence. Rewrite as a publications appendix. |
| `safety_against_adversarial_attck.tex`, `smt_verification.tex`, `00-plano-de-trabalho.tex`, `Research/00_repairer.tex` | Unused fragments or empty. |
| `anotacoes.md` | Notes on the January prototype. Useful only to document that its kernel differed (C-17). |
| `figuras/3-metodologia/main_diagram_preqbmc_pdf.drawio.pdf` | Old tier-based flow. Replace with the two-branch overview. |

---

## 4. Comparison of the two manuscripts

### 4.1 Shared formal foundations

| Element | Evidence |
|---|---|
| Verified object is the generated fixed-point C program | L-PDF §1, §4.1; S-PDF §I, §III-A |
| Property: strict target dominance for every input of a declared region | L-PDF Eq. (4); S-PDF Eq. (2) (SH-001) |
| Signed fixed-point format with Q = 1 + I + F | L-PDF §2.2; S-PDF §III-B (SH-002) |
| Operation order: accumulate in `__int128`, round half away from zero, add bias, saturate, then ReLU | L-PDF Eq. (5), §4.4; S-PDF §III-B |
| Assume-guarantee contracts with an explicit inclusion between consecutive layers | L-PDF Eq. (6)–(8); S-PDF Eq. (3)–(6) |
| Untrusted proposers and a separate formal discharge | L-PDF §4.9; S-PDF §III-C (SH-003) |
| ESBMC 7.11.0 in bit-vector mode with fully unwound loops (Z3; Bitwuzla for the Iris and Seeds runs of Study A, see C-19) | L-PDF §5.1; S-PDF Table II; L-REC |
| Inconclusive outcomes never counted as refutations | L-PDF §4.8; S-PDF §III-C (SH-004) |
| Decomposition of a layer into output-neuron blocks | L-PDF §4.5; S-PDF §III-C |

### 4.2 Different algorithms

| Aspect | Study A (LADC) | Study B (SSV) |
|---|---|---|
| Goal | Synthesize a format per layer and certify it | Certify one given global format |
| Networks | Dense ReLU, 3 to 4 affine layers, up to 100 hidden neurons | CNN, 5 affine layers, 14,123 neurons |
| Format | Per layer, selected by search (Q from 6 to 15 in Table 2) | Global Q7.8, fixed before verification |
| Eligibility | DeepPoly must prove the real-valued source property | None |
| Proposal of facts | DeepPoly forward bounds; MILP backward preimages (CBC); derived error budget | Interval box facts; exact-integer CROWN back-substitution chains |
| Relation to the float model | Contracts derived from real-valued preimages | Finder works on the integer model |
| Certificate | Chain of per-layer boxes plus output-competitor proofs | Dependency graph of box facts and chain lemmas plus margin and clamp closure |
| Refinement | Optional MILP-proposed, ESBMC-validated directional cuts | None in the reported campaign |

### 4.3 Different guarantees

| Aspect | Study A | Study B |
|---|---|---|
| Input region | Clipped real-valued box, encoded to an integer box | All valid byte images within 1 raw byte |
| Hidden-layer obligation | ESBMC on a deterministic endpoint program; coverage by a paper lemma | ESBMC on ground box harnesses and on chain lemmas |
| Output obligation | ESBMC with a symbolic hidden vector, one query per competitor | Margin closure from verified chains, checked in Python with clamp rules |
| Composition | Python containment test between consecutive contracts | Python dependency closure, step linkage and array binding |
| Overflow | Python static range check | Python static bound |
| Binding to the shipped file | Shared helper text; separate renderers | SHA-256 of the C file; Python binding with sampled ESBMC corroboration |
| Level reached | Encoded-domain contract (CONDITIONAL) | Complete-network certificate for a byte region (CONDITIONAL) |

### 4.4 Experimental contributions

| Study A | Study B |
|---|---|
| 26 of 26 configured regions with encoded-contract evidence (LADC-001) | 1 of 80 primary; 2 of 80 with disclosed rerun (SSV-001, SSV-002) |
| Block-size effect on query time and memory (LADC-005, LADC-006) | Monolithic wall at 24–48 layer-0 neurons; about 49,000–51,000 calls per certified image (SSV-007, SSV-008) |
| Cuts ablation: no additional certificate (LADC-007) | Finder gap for 54 pairs; resource aborts for 16–17 (SSV-003) |
| Accuracy and nominal size of synthesized formats (LADC-010, LADC-011) | Accuracy of Q7.8 and Q3.4 (SSV-010, SSV-011) |
| Positioning against Quadapter and CEG4N on four cases (LADC-013) | Board latency, memory and output parity (SSV-012 to SSV-016) |

### 4.5 Incompatible assumptions and claims

Each is developed in [scientific_conflicts.md](scientific_conflicts.md).

- Sampling: configured suite against pre-registered random draw (C-12).
- Input domain: real box against discrete bytes (C-06).
- Source prerequisite: required in A, absent in B (C-12).
- Meaning of "monolithic" (C-15).
- Meaning of a relational fact (C-02).
- Role of the error budget: central in the LADC text, absent in SSV (C-14).
- "Per-layer quantization" in the SSV contribution list (C-04).

---

## 5. New dissertation-level arguments

These do not appear in either paper and must be written for the dissertation. Each is limited to what the ledger supports.

1. **A common statement of the verified object and property.** One semantics definition and one property, with the two input domains defined side by side (Chapter 4).
2. **A trust-boundary analysis that holds for both studies.** A table that separates, per study, what ESBMC checked, what trusted Python computed, what a paper proof supplies, and what was only measured. This is the main original synthesis (C-03).
3. **A taxonomy of what "verified" means.** Local harness obligation; encoded-domain contract chain; complete-network byte-region certificate; measured parity. Each reported number is placed in exactly one row (C-01, C-11).
4. **Two forms of decomposition and their costs.** In Study A, smaller blocks reduce the size of each query at modest growth in query count. In Study B, decomposition makes the problem feasible at the price of tens of thousands of obligations whose total cost grows linearly with certificate size. The argument is qualitative; no cross-study timing ratio is given.
5. **Where incompleteness arises.** Proposal precision (LADC margin and layer inconclusives in the earlier campaign; 54 finder-inconclusive pairs in SSV), solver resources (cut-validation timeouts; 16–17 resource aborts), and composition coverage (the zero-product columns found after submission). None of these is evidence of non-robustness.
6. **Why the second study changed the proposal method.** Supported only as "was not applied" unless the unpublished small-CNN pilot is pinned (C-04).
7. **Formal results and deployment measurements answer different questions.** The certified file, the compact file and the board execution are linked by measurement, not proof (C-05).
8. **Bounded negative results as evidence.** The post-submission diagnostics show limits of specific interventions and are reported outside both frozen denominators (DIAG-001 to DIAG-007).

---

## 6. Evidence ledger: conventions and priorities

**Proof levels used**

| Level | Meaning in the ledger |
|---|---|
| PROVED | A solver verdict of VERIFIED on a stated obligation, or an elementary paper proof (marked as such). |
| CONDITIONAL | A region-level or theorem-level claim that follows from PROVED obligations only together with listed trusted components or premises. |
| MEASURED | An observed value: a count of outcomes, a time, a memory figure, an accuracy, an output comparison. |
| INCONCLUSIVE | A timeout, memory-out, unknown or unresolved result. |
| ASSUMED | A premise that is trusted or unverified, or a claim with no located support. |

**Change from the supplied template.** The template marked the 26-region, 1/10, 1/80 and 2/80 rows as PROVED. The ledger splits each into a PROVED row for the solver verdicts (LADC-001a, SSV-001a) and a CONDITIONAL row for the region-level claim (LADC-001, SSV-001, SSV-002), because the composition is done by trusted code. The "1 of 10" row is MEASURED and historical (LADC-002).

**Rows to settle first**

1. LADC-002 and LADC-003 (C-01).
2. LADC-008 and LADC-009 (C-02).
3. LADC-017, LADC-018 and SSV-006 (C-03).
4. SSV-005 and SSV-017 (disclosures missing from the paper).
5. SH-005 (C-08).

---

## 7. Blockers

| ID | Missing information | Needed for |
|---|---|---|
| B-1 | Run directory and commit of the ten-region LADC campaign; author confirmation that "1 of 10" in the submitted PDF refers to it (C-01, LADC-002) | Chapter 5 results, Chapter 7 taxonomy |
| B-2 | Code evidence for or against use of the dedicated SMT fixed-point theory (C-08, SH-005) | Chapters 1, 2, 4 |
| B-3 | Varying-coordinate count of each LADC encoded input box (C-06, LADC-024) | Chapter 5 results and threats |
| B-4 | Artifacts and commit for the qualification-era Iris/Seeds tables, or a decision to drop them (C-17, QUAL-001) | Appendix A |
| B-5 | The inherited-bound loss range in the singleton diagnostic: two notes disagree (DIAG-006) | Chapter 7 diagnostics |

Also required but not blocking the first editing task: the chaining rule and budget formula of the campaign revision (C-14); the solver-profile mapping at the campaign revision (C-19); confirmation of PDF versions (C-16); publication status and authorship statement (C-18).
