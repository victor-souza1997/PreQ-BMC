# Dissertation edit plan: file-by-file sequence

Phase 0 audit, 2026-11-16. Nothing here has been carried out. Each step needs authorization before it starts.

Related files: [dissertation_structure.md](dissertation_structure.md) (target structure), [scientific_conflicts.md](scientific_conflicts.md) (C-xx), [evidence_ledger.csv](evidence_ledger.csv) (claim IDs), [dissertation_audit.md](dissertation_audit.md) (blockers B-1 to B-5).

All LaTeX paths are relative to `latex_source/Master_Dissertation___Network_Equivalence/`.

---

## 1. Working rules

1. **One chapter per step.** Only the files named in a step are edited.
2. **Old chapter files are not edited or deleted.** New files are created beside them; `000-main.tex` decides which are compiled.
3. **Suggestions appear in blue.** A macro in the preamble keeps this manageable:

   ```latex
   \newcommand{\sug}[1]{\textcolor{blue}{#1}}
   ```

   Two side effects need a decision: `hyperref` already colours links and citations blue, so suggestions and links look alike; and `2-fundamentos.tex` L485 already contains an old blue note. Recommended: change `linkcolor` and `citecolor` to another colour while drafting.
4. **Unsupported statements are not written.** Where a source is missing, the LaTeX gets `% TODO: SOURCE REQUIRED — …` and the ledger gets an open issue.
5. **Each number is copied from the ledger row,** with its proof level and denominator.
6. **Build after every step** in the project directory with `latexmk -pdf 000-main.tex`, and report errors, unresolved references and placeholder strings.
7. **No commits or pushes** unless asked. Frozen records, protocol files and the papers are never modified.

Baseline build (scratch copy, 2026-11-16): 44 pages, no errors, 2 unresolved references, 9 "[CITATION NEEDED]" strings, 2 BibTeX warnings.

---

## 2. Decisions needed before editing

| # | Decision | Affects | Reference |
|---|---|---|---|
| D-1 | Is the "1 of 10" sentence in the submitted LADC paper a carry-over from the first campaign? | Ch. 5, 7 | C-01, B-1 |
| D-2 | Keep the qualification-era Iris/Seeds tables as Appendix A, or drop them? | Appendix A | C-17, B-4 |
| D-3 | Was the background chapter removed from the build on purpose? Are PDFs "(1)" and "(2)" the submitted versions? | Ch. 2 | C-16 |
| D-4 | Final title; research questions RQ1–RQ4 as proposed in the guide? | Front matter, Ch. 1 | C-13 |
| D-5 | Use the unpublished small-CNN pilot as a bridge between the studies? | Ch. 6.1 | C-04 |
| D-6 | Publication status of each paper and the authorship statement | Ch. 1.5, Appendix B | C-18 |
| D-7 | Bibliography style: `apalike` (current) or the UFAM style in `cls_bst/` | `000-main.tex` | Audit §3.1 |

Steps 1 and 2 below do not depend on any of these.

---

## 3. Sequence

### Step 1 — Skeleton (first recommended task)

Purpose: a compiling eight-chapter document with headings only, so every later step has a fixed place to write into.

| File | Action |
|---|---|
| `000-main.tex` | Change `mscexam` to `msc`. Replace the chapter inputs (L107–111) with the eight new files and the two appendices; keep the old inputs as comments. Add `\sug`. Move the macros from `3-metodologia.tex` L216–220 to the preamble. Remove duplicate package loads. Make English the active `babel` language. |
| `capitulos/1-introduction.tex` … `capitulos/8-conclusion.tex` | Create. `\chapter`, `\label`, and the `\section` headings of [dissertation_structure.md](dissertation_structure.md) §3. No body text. |
| `capitulos/A-preliminary-results.tex`, `capitulos/B-publications.tex` | Create with a heading only. |
| `capitulos/0-capa_n_contra.tex` | Not edited in this step. |

Labels to define now, because existing text refers to them: `cap:intro`, `chap:background`, `chap:related`, `chap:framework`, `chap:studyA`, `chap:studyB`, `chap:discussion`, `chap:conclusion`.

Check: build succeeds; table of contents shows eight chapters; no unresolved references; old files unchanged (`git status`).

### Step 2 — Bibliography

| File | Action |
|---|---|
| `referencias.bib` | Add the missing entries listed in the audit §3.1, taking each from the reference list of L-PDF or S-PDF (both give full data). Fix the two BibTeX warnings. Do not rename existing keys yet; record the three misleading keys for a later global rename. |

Check: every entry added is cited by one of the two papers; no entry is written from memory.

### Step 3 — Chapter 4, framework (new synthesis)

Write this first among the content chapters: it fixes the semantics, the property, the trust table and the vocabulary that Chapters 5–7 use.

| File | Action | Inputs |
|---|---|---|
| `capitulos/4-framework.tex` | Write §4.1–4.8 | `3-metodologia.tex` L428–500, L592–629, L695–736; L-PDF §4.1; S-PDF §III-A to III-C; L-AUD §3.1, §3.4; worktree `P1_checker_closure.md` |
| `figuras/4-framework/` | Two new figures (overview; proof boundary) | Structure §6 |

Constraints: C-03 (trust table), C-06 (two domains), C-07 (notation), C-15 (terms). Ledger rows SH-001 to SH-006, LADC-015 to LADC-018, SSV-006.

Blocked parts: any sentence about the SMT fixed-point theory (B-2).

### Step 4 — Chapter 5, Study A

| File | Action | Inputs |
|---|---|---|
| `capitulos/5-study-a-synthesis.tex` | Write §5.1–5.10 | L-PDF §4–5; `3-metodologia.tex` L502–681, L738–810; L-AUD; L-REC |
| `figuras/5-study-a/` | Copy block-cost figures | `docs/LADC_2026/results_revision/generated/` |

Before writing, three read-only checks on the campaign revision (`59042988`): the chaining rule and budget formula (C-14); the solver-profile mapping (C-19); the varying-coordinate count per region (B-3).

Constraints: C-01 (three separate statements; raw status shown), C-02 (mechanism (ii)), C-10 (no superiority). Ledger rows LADC-001 to LADC-024.

Blocked parts: the historical "1 of 10" row (B-1, D-1); interpretation of MNIST radii (B-3).

### Step 5 — Chapter 6, Study B

| File | Action | Inputs |
|---|---|---|
| `capitulos/6-study-b-cnn.tex` | Write §6.1–6.9 | S-PDF §III–VI; S-EVID; S-PROTO; `docs/ssv2026/methodology.md`; worktree `P1_checker_closure.md`, `P2_reaudit.md` |
| `figuras/6-study-b/` | Copy deployment figure | `docs/ssv2026/vim3_deployment_results_v2.pdf` |

Constraints: C-04 (no synthesis), C-05 (formal against measured), C-11 (primary first; protocol history; supplement). Ledger rows SSV-001 to SSV-020.

Blocked parts: §6.1 if the pilot is used (D-5); any formal statement of chain-step soundness (structure §5, TODO source).

### Step 6 — Chapter 7, discussion (new synthesis)

| File | Action | Inputs |
|---|---|---|
| `capitulos/7-discussion.tex` | Write §7.1–7.7 | Chapters 4–6 as written; ledger; worktree notes P3, P3b, P3c, I-P, I-R, encoding pilot |

Constraints: C-12 (no cross-study rate), C-02 (three relational mechanisms), diagnostics outside both denominators. Ledger rows DIAG-001 to DIAG-007, SSV-005, SSV-018.

Blocked parts: the singleton-control figures (B-5).

### Step 7 — Chapters 2 and 3

| File | Action | Inputs |
|---|---|---|
| `capitulos/2-background.tex` | Write §2.1–2.9 | `2-fundamentos.tex` L5–56, L229–369, L434–577; `3-metodologia.tex` L106–126; L-PDF §2; S-PDF §II-C |
| `capitulos/3-related-work.tex` | Write §3.1–3.6 | L-PDF §3 and Table 1; S-PDF §II and Table I; `3-metodologia.tex` L403–426; commented tables in `1-introdução.tex` L215–296 and `5-conclusão.tex` L40–136 |
| `capitulos/0-glossary.tex` | Add acronyms in use | — |

Written after the technical chapters so that the background covers exactly what they need. Constraints: C-07, C-08, C-10. Global rename of the three misleading citation keys happens here.

### Step 8 — Chapters 1 and 8, front and back matter

| File | Action | Inputs |
|---|---|---|
| `capitulos/1-introduction.tex` | Write §1.1–1.6 | `1-introdução.tex` L25–60, L127–134; Chapters 4–7 |
| `capitulos/8-conclusion.tex` | Answer RQ1–RQ4 | Chapters 5–7 |
| `capitulos/0-capa_n_contra.tex` | Title, date, examiners, abstracts in both languages, keywords, acknowledgements | D-4 |
| `capitulos/B-publications.tex` | List the two papers with exact status and the author's contribution | D-6 |
| `capitulos/A-preliminary-results.tex` | One table and its settings, or remove the file | D-2 |

### Step 9 — Consistency audit

| Check | Method |
|---|---|
| Every occurrence of "verified", "certified", "sound", "complete", "optimal", "equivalence", "proof", "AAOS" | Search; compare each with its ledger row |
| Every number in every table | Compare with L-REC, S-PDF tables and the ledger |
| Notation | One pass against structure §7 |
| Theorems and lemmas | Hypotheses match structure §5; no statement added without a source |
| Build | No errors, unresolved references, placeholders or overfull tables; front matter complete |
| Blue text | Decide which suggestions are accepted; remove the macro colour for the final PDF |
| Old files | Move the unused chapter files to an `archive/` folder, with authorization |

### Step 10 — Submission

Advisor review, UFAM formatting check, permissions for reused figures and text. Outside the scope of automated editing.

---

## 4. Dependencies at a glance

| Step | Depends on | Can start now |
|---|---|---|
| 1 Skeleton | — | Yes |
| 2 Bibliography | — | Yes |
| 3 Chapter 4 | Step 1 | Yes, except the B-2 sentences |
| 4 Chapter 5 | Step 3; D-1; B-1, B-3 for two subsections | Partly |
| 5 Chapter 6 | Step 3; D-5 for §6.1 | Mostly |
| 6 Chapter 7 | Steps 4, 5; B-5 for one paragraph | After 4 and 5 |
| 7 Chapters 2, 3 | Steps 3–5; D-3 | After 5 |
| 8 Chapters 1, 8, front matter | Steps 3–7; D-4, D-6, D-2 | Last |

---

## 5. Report after each step

Files changed; sections added, moved or removed; claims written with their ledger IDs; TODO and blocker entries added; build status; a short diff summary; the next proposed step.
