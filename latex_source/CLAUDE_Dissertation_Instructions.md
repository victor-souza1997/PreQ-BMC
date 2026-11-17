# PreQ-BMC Master's Dissertation — Claude/Fable Project Instructions

## Mission
Transform the existing January 2026 UFAM qualification draft and two submitted papers into a coherent, technically defensible master's dissertation in **academic English**. The output is a single LaTeX dissertation, not two pasted conference papers. Preserve the UFAM template and existing working bibliography unless an explicit change is authorized.

## Required primary sources
Read the *actual supplied source files* before using their details:
1. `Master_Dissertation___Network_Equivalence (8).pdf` (qualification draft; 44 pages).
2. `LADC_2026_João_Victor (9).pdf` (submitted LADC paper; 17 pages).
3. `SBESC___SSV_2026___AAOS_with_PreQ_BMC (10).pdf` (submitted SSV paper; 6 pages).
4. `PreQ_BMC_Dissertation_Integration_Guide.md` (integration plan).
5. The editable dissertation LaTeX project (`.tex`, `.bib`, figures, class/styles). If not provided, perform content audit only and explicitly request the source project before editing. Do not fabricate files/paths.

Where direct source conflicts with the integration guide, **surface the discrepancy**; the guide is a plan, not primary evidence. Prefer frozen experiment logs, hashes and protocol files when validating reported counts and formal statuses. Do not assume access to private code or logs unless actually available.

## Governing thesis
The research addresses the implementation gap between abstract QNN verification and actual finite-width fixed-point C semantics. Two **different experimental workflows** investigate the goal:
- **Study A (LADC):** DeepPoly forward analysis; MILP backward preimages; per-layer quantization search; generated C layer contracts; ESBMC; neuron-block decomposition; Iris, Seeds and MNIST small fully connected models.
- **Study B (SBESC SSV):** frozen global Q7.8 CNN; interval facts and integer-CROWN candidate chains; local bit-precise ESBMC checks; Python dependency/composition/binding checks; GTSRB, Khadas VIM3, AAOS.
Never describe Study B as running Study A's MILP synthesizer or claim the studies have identical certificate formats/checker trust boundaries.

## Mandatory scientific distinctions
1. Separate proof statuses: `PROVED`, `MEASURED`, `CONDITIONAL`, `INCONCLUSIVE`, `ASSUMED`; where needed distinguish fact from inference.
2. LADC: the paper reports **26 encoded-domain contract cases** but only **1/10 MNIST regions** under its stricter deployed-transfer criterion. These are not one certification statistic. Reconcile paper internal wording before paraphrasing.
3. SBESC: primary **1/80** test regions; disclosed rerun **2/80**; eight nominally misclassified; remaining results finder/checker inconclusive. Do not reclassify inconclusive as robustness failures.
4. SBESC native int16 output parity with the certified int64 C reference is empirically observed on a finite corpus, **not** a proof of binary equivalence, compiler correctness, hardware correctness, or AAOS correctness.
5. ESBMC verifies specific C harness obligations, not necessarily every source-to-certificate identity or proof-graph link. Python parsing, dependency closure, certificate-to-program binding and correctness of harness translation are trust assumptions unless independently proved.
6. Do not assert global optimal bit-widths, general ANN/QNN equivalence, backdoor-freeness certification, universal CNN verification, or bit-precise verification of native binaries.
7. DeepPoly/MILP and integer CROWN *propose* facts or synthesize candidates. They are not interchangeable with ESBMC discharge of the C obligations.
8. Do not represent direct implementation of Baranowski's dedicated SMT fixed-point theory unless exact code evidence establishes that; describe bit-precise C/BMC/bit-vector reasoning.
9. Fixed-point operation order is essential: integer accumulation, RHAZ rescaling, quantized bias addition, saturation/clamping, then ReLU (where applicable). Q=1+I+F.
10. Distinguish input domains: LADC encoded input intervals versus SBESC valid bounded uint8 images after defined resizing/encoding.
11. Keep post-submission P3/I-B/P3b/P3c/I-P/I-R/encoding diagnostics **outside** the frozen LADC and SBESC certification denominators. I-P gave +1 worst-child gain with roughly 2x prospective proof graph. I-R and reduced-width encoding both timed out at the pre-specified 60 s queries. Neither proved a relational lemma or recovered an image.
12. Where the paper and dissertation disagree on relational-cut verification (prefix ESBMC versus conditional MILP-derived bounds), do not silently reconcile. Identify which mode produced which recorded result and state the exact trust assumptions.
13. Do not invent citations, DOI, data, proofs, timings, code paths, model sizes, datasets or artifacts. If unsupported, place `% TODO: SOURCE REQUIRED — ...` in LaTeX or a BLOCKER entry in the audit ledger.

## Recommended chapters
1. Introduction — problem, objectives, research questions, contributions.
2. Background — networks, quantization, finite-width C, robustness, SMT/BMC, DeepPoly, MILP, integer CROWN.
3. Related Work — grouped by verification target, property, and guarantees.
4. Deployment-Aware Verification Framework — common semantics and **two distinct** candidate-generation/verification paths, with trust-boundary table.
5. Preimage-Guided Quantization Synthesis and Dense-Network Evaluation — LADC.
6. Integer-CROWN CNN Verification and Embedded AAOS Evaluation — SBESC.
7. Comparative Discussion, Scalability, Incompleteness and Threats to Validity — supplementary diagnostics.
8. Conclusions and Future Work — answers aligned with evidence.
Write Chapters 4 and 7 as genuinely new synthesis; rewrite/expand 5 and 6 from the papers. Rewrite Introduction and Conclusion after the technical body is stable.

## Work protocol
**Phase 0 — Audit only:** inspect input PDFs, LaTeX structure and bibliography. Generate (a) chapter migration table; (b) claim/evidence matrix with source and exact location; (c) conflict/blocker register; (d) proposed file-level edit sequence. Do **not** edit the dissertation during this audit.

**Phase 1 — Skeleton:** after review, create/modify the LaTeX skeleton, preserving the original project/template. Put placeholder headings only; no speculative technical text.

**Phase 2 — One chapter at a time:** edit only the authorized chapter/files. Cite only available bibliography keys. Cross-reference existing equations/tables rather than duplicating. Keep notation globally consistent: original ANN `\mathcal N`, deployed QNN `\widehat{\mathcal N}`, fixed-point format `\zeta^{[l]}=\langle I^{[l]},F^{[l]}\rangle`, `Q^{[l]}=1+I^{[l]}+F^{[l]}`, and deployed strict margin `\widehat y_t>\widehat y_j` for all competitors. Avoid inventing a new unified algorithm.

**Phase 3 — Chapter gate:** for each chapter, produce a concise report: modified files, added/moved/deleted sections, claims and source anchors, conflicts/TODOs, compile status, and a diff summary. Run LaTeX build if dependencies are present. Do not edit other chapters while fixing it.

**Phase 4 — Full consistency audit:** inspect all `VERIFIED`, `certified`, `equivalence`, `optimal`, `sound`, `complete`, `proof`, `implementation gap` and `AAOS` occurrences; validate each claim against source. Compare all result tables with source artifacts; check proofs' hypotheses and defined trust boundaries. Verify compilation, references, figure sizes, page overflows and front matter.

**Phase 5 — Submission:** final user/advisor technical review and UFAM formatting check. Do not claim any university approval or paper acceptance beyond the evidence.

## Token efficiency and execution discipline
- Read each primary source once per work phase and maintain a brief persistent source-index and claim ledger; do not re-summarize all sources every turn.
- At the start of a task, state the intended files and invariants briefly. Then perform the authorized action to completion, rather than endlessly planning.
- Use `high` effort normally; `xhigh` for cross-study soundness/claims audit or severe contradictory evidence; `medium` for well-specified prose revisions, formatting and tables; avoid `max` except a measured exceptional need.
- Do not spawn multiple redundant subagents or duplicate the same PDF analysis. Assign narrow independent work only when useful (evidence extraction, LaTeX audit, citation audit).
- Return changed paths, concise diff summary, blockers and next suggested chapter. Avoid long narration, repeated project background, and unsupported performance predictions.
- Never alter frozen historical result records, source hashes, protocol files or experiment outcomes as part of dissertation writing.
- Never commit, push, replace original uploaded papers or delete files without explicit instruction; maintain backup or separate Git branch for dissertation editing.

## Preferred source-of-truth evidence table
Maintain `evidence_ledger.csv` with columns:
`claim_id,claim_text,study,source_file,source_location,artifact_path,experiment_scope,proof_level,denominator,trust_boundary,allowed_wording,open_issue`.

## First task
Perform Phase 0 (read-only audit). Return the proposed chapter-to-file map, scientific discrepancy list, evidence ledger priorities and a precise first editing task. Do not start writing the entire dissertation in one pass.
