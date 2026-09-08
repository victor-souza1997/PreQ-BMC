# LADC 2026 — PreQ-BMC: consolidated to-do list

Paper: *Preimage-Guided Bit-Precise Verification of Fixed-Point Quantized Neural Networks*.
Scores: R1 Weak Reject (expert), R2 Weak Accept (familiar), R3 Weak Accept (some knowledge).

The recommendation split is diagnostic. The two positive reviews are from the two
less-expert reviewers and praise the *idea*; the expert rejects on *evidence and
craft*. Nothing in the reviews attacks the soundness of the method. Every blocking
objection is either (a) an experiment we can run with flags that already exist, or
(b) writing. That is the good case, and it is what this list is ordered around.

Legend: **P0** = the paper is rejected again without it. **P1** = a reviewer asked
for it explicitly and its absence will be re-raised. **P2** = strengthens the paper
or pre-empts the next round. `[repo]` = actionable in this repository.
"Effort" is person-days of work; compute time is listed separately where it dominates.

---

## P0 — Blocking

### T1. Run the β = 0 monolithic arm and report RQ3 quantitatively `[repo]`
All three reviewers raised this; it is the single most-cited defect. The paper
currently *claims* block-wise decomposition "reduces query size" (abstract,
contribution 3) and then admits in RQ3 that the gain is unmeasured. A claim asserted
in the abstract and retracted in the results section is what turns a weak accept into
a weak reject.

- The arm is one flag: `--esbmc-layer-block-size 0` (see
  [run_robustness_pipeline.py:60](../../src/scripts/run_robustness_pipeline.py#L60)).
- A plan file with `esbmc_layer_block_size: 0` already exists
  ([experiments/article_experiments_no_block.json](../../experiments/article_experiments_no_block.json)),
  and `output/article_runs_no_block/` already holds Iris runs.
- The aggregator already emits `table_ablation.csv` and `table_scalability.csv` with
  `blockwise_enabled`, `block_size`, `max_esbmc_query_peak_memory_mib`,
  `max_esbmc_query_time_seconds`, `largest_neurons_per_query`.
- Do not stop at β ∈ {0, 2}. Run the sweep β ∈ {0, 1, 2, 5, 10} so the trend, not a
  two-point difference, carries the claim. Design in
  [03_EXPERIMENT_MATRIX.md](03_EXPERIMENT_MATRIX.md).

Effort: 0.5 d setup + 1 d analysis. Compute: ~2–4 machine-days for the full sweep
(dominated by MNIST β = 0, which is expected to time out — that *is* the result).

### T2. Run the cuts-disabled and invariants-disabled arms `[repo]`
R1 and R3 both name the missing "cuts-disabled" arm. Without it, §4.6 (relational
refinement) is an unsupported claim.

- `--margin-cuts off` ([run_robustness_pipeline.py:245](../../src/scripts/run_robustness_pipeline.py#L245))
- `--no-e2e-invariants` ([run_robustness_pipeline.py:235](../../src/scripts/run_robustness_pipeline.py#L235))
  is a second, unrequested axis that costs nothing extra to sweep and directly
  supports §4.4.

Effort: 0.25 d. Compute: shares the T1 campaign.

### T3. Replace the 1-of-10 headline with the full campaign, and tabulate *why* the rest failed `[repo]`
R1 calls "1 certification in 10" a critical weakness; R3 repeats it. The framing is
self-inflicted: the repository's broader reported set is materially better than the
number in the paper, and the paper never gives the reader the failure taxonomy that
would make the inconclusive results informative rather than embarrassing.

- The pipeline already records a precise status taxonomy —
  `MARGIN_INCONCLUSIVE`, `LAYER_INCONCLUSIVE`, `PREIMAGE_DEFLATION_EMPTY`,
  `MARGIN_REFUTED`, `SOURCE_PROPERTY_INCONCLUSIVE`, plus per-run
  `esbmc_timeout_count` / `esbmc_memout_count`
  ([experiment_summary.py:305-420](../../src/reports/experiment_summary.py#L305)).
  None of this reaches the paper. Add a failure-mode table (skeleton in
  [03_EXPERIMENT_MATRIX.md](03_EXPERIMENT_MATRIX.md#t3-failure-mode-table)).
- Separate the three causes the reviewers conflate: solver resource exhaustion
  (timeout/memout), preimage precision collapse (`DEFLATION_EMPTY`, margin
  inconclusive), and source-property ineligibility (DeepPoly cannot even establish
  the float property — not our failure at all).
- Report the denominators honestly per dataset and per architecture rather than as a
  single 10 %.

Effort: 1.5 d (aggregation + table + text).

### T4. Rewrite the Introduction to remove the paragraph-level duplication
R1: "text redundancy across the first four paragraphs (demonstrating lack of
proof-reading)". This is correct and it is the cheapest score to recover.
Paragraphs 1–2 and 3–4 state the same two propositions twice (embedded deployment
motivates compression; quantization is not semantics-preserving). Cut to a single
pass. Details and a proposed structure in
[04_NOTATION_AND_TEXT_FIXES.md](04_NOTATION_AND_TEXT_FIXES.md#introduction).

Effort: 0.5 d.

### T5. State the scope restriction up front, in the abstract and conclusion
R1 and R3 both flag this. The method is restricted to dense feed-forward
affine+ReLU networks; there is no convolution and no per-channel quantization.
This is verifiable in the code: the DeepPoly front end admits only
`INPUT_LAYER`, `AFFINE_LAYER`, `RELU_LAYER`
([DeepPoly_preqbmc.py:38-40](../../src/symbolic_pp/DeepPoly_preqbmc.py#L38)).
Say so in the abstract, in §4.1 as a standing assumption, and in §6 as the concrete
extension path. R1 also asks that the x86 evaluation host be described as a
numerical model of the deployed kernel rather than as an edge platform — agree, and
say it in §5.1.

Effort: 0.25 d.

---

## P1 — Explicitly requested

### T6. Differentiate PreQ-BMC from QNNVerifier, and add QA-IBP `[related work]`
R1 names both. QNNVerifier is the closest possible relative — same model checker
(ESBMC), same fixed-point target — and its absence from Table 1 is the kind of gap
an expert reviewer treats as a literature failure rather than an oversight. The
differentiators are real and defensible (synthesis vs. fixed precision; preimage
contracts and compositional obligations vs. monolithic queries; block decomposition;
generated deployable backend). Draft text and the full gap analysis in
[05_RELATED_WORK_GAPS.md](05_RELATED_WORK_GAPS.md).

Effort: 1 d.

### T7. Add LADC-community framing and prior-edition citations
Both R1 and the LADC review form ask for it. This needs a real literature pass over
recent LADC proceedings — see the search protocol in
[05_RELATED_WORK_GAPS.md](05_RELATED_WORK_GAPS.md#ladc-prior-editions). **Do not
invent citations for this section**; the venue's own reviewers will recognise them.
Frame quantization-induced numerical deviation as a fault-injection / dependability
concern: the deployed integer kernel is the executed artifact, and a certificate that
does not cover it is a latent fault in the safety argument.

Effort: 1 d (plus reading).

### T8. Fix the abstract `[R2]`
R2: dense, unexplained abbreviations, does not state the contribution in plain terms.
Rewrite to answer four questions in order: what is generated, what is verified, what
is new relative to Quadapter, what was measured. Proposed draft in
[04_NOTATION_AND_TEXT_FIXES.md](04_NOTATION_AND_TEXT_FIXES.md#abstract).

Effort: 0.25 d.

### T9. Fix the notation defects `[R1, R2]`
R2 caught the `⟨N, I, F⟩` vs lowercase `i` slip in §2.2 and the undefined "Calls"
column in Table 2. R1 reports the broader problem: notation used before definition.
Our own pass found several more, including a **substantive** one — B_ε(x₀) is
defined as a *clipped* box in §2.3 and as an unclipped ℓ∞ ball in Eq. (5), which are
not the same set and the implementation uses the clipped one. Full list, each with
the corrected form, in
[04_NOTATION_AND_TEXT_FIXES.md](04_NOTATION_AND_TEXT_FIXES.md).

Effort: 0.5 d.

### T10. Re-render Figure 1 as vector art `[R3]`
The current figure is raster, over-dense, and its label vocabulary does not match
Algorithm 1 or §4.8 (see T13). Re-draw in TikZ or export SVG→PDF; simplify to the
three trust tiers; unify the status names.

Effort: 0.5 d.

### T11. Justify or raise the 300 s timeout `[R1]`
R1 asks for evidence about cutoff times beyond 300 s. Two responses are needed, and
the first matters more: show a *timeout-sensitivity curve* (cumulative certifications
vs. wall-clock budget) so the reader can see whether the inconclusive regions are
near-misses or asymptotically out of reach. Our prior measurements suggest the latter
for MNIST — say so with data rather than asserting it. `--esbmc-timeout-seconds`
takes any budget ([run_robustness_pipeline.py:103](../../src/scripts/run_robustness_pipeline.py#L103)).

Effort: 0.5 d. Compute: ~1 machine-day at a 1800 s budget on the unresolved regions.

---

## P2 — Should do; strengthens the submission

### T12. State and prove the soundness theorem
The paper lists obligations (8), (9), (13), (16), (18) but never states the theorem
they compose into, and never proves the error budget (10). For a formal-methods
venue this is the most conspicuous scientific gap, and no reviewer caught it — which
means it will be caught next time. Add:
- **Theorem 1 (deployed transfer).** If every layer obligation (8), every chaining
  obligation (9), every validated cut (16), and every competitor obligation (18) is
  discharged, and the assumption boxes are non-vacuous, then Eq. (5) holds for the
  generated implementation. Proof by induction over layers.
- **Lemma 1 (budget soundness).** Eq. (10) over-approximates the per-neuron deviation
  between the ideal and deployed pre-activations. *This needs care*: δ is indexed by
  the output neuron `i` on the left, but the first term of the right-hand side has no
  `i` dependence, and the scale `S_{l-1}` appears where a weight-quantization scale is
  expected. Either the formula or the indexing is wrong as printed. Re-derive before
  submitting.
- **Trust base.** An explicit paragraph: what is trusted (ESBMC + Z3, the C semantics
  ESBMC assigns, the shared arithmetic kernel), what is not (MILP output — it only
  proposes; DeepPoly — it only gates eligibility; the compiler — explicitly out of
  scope), and where the empirical stage sits (never certifying).

Effort: 2 d. This is the highest-value P2 item.

### T13. Unify the status vocabulary across Figure 1, Algorithm 1, §4.8, and §5
The paper currently uses at least three overlapping vocabularies: Figure 1's
`VERIFIED / MARGIN_REFUTED / MARGIN_INCONCLUSIVE / DEFLATION_EMPTY`, Algorithm 1's
`DeployedTransfer / Failed / Unknown / SourcePropertyInconclusive`, and §5's
`Margin-Inconclusive / Layer-Inconclusive`. The implementation has exactly one
taxonomy ([experiment_summary.py](../../src/reports/experiment_summary.py)); print it
as a table and use it everywhere. Cheap, and it makes the paper look engineered
rather than assembled.

Effort: 0.5 d.

### T14. Promote the no-overflow obligation from caveat to discharged side condition `[repo]`
§4.4 states Python/C equivalence "provided that all `__int128` intermediate values
are representable" — an unverified hypothesis inside the central fidelity claim. The
repository already has the harness that discharges it
(`render_no_saturation_program`, and `require_formal_no_saturation` in the run
configs). Turn it on for the reported campaign, report it as a verified obligation,
and delete the caveat. This converts the weakest sentence in §4 into a strength and
directly reinforces R3's stated reason for liking the paper.

Effort: 0.5 d. Compute: negligible.

### T15. Make Table 2's cross-tool metrics comparable, or mark them incomparable
Table 2 reports peak memory as *max ESBMC-query RSS* for PreQ-BMC and *process-tree
RSS* for the baselines. Those are different quantities and the table invites a
comparison between them. The runner already measures process-tree RSS
(`esbmc_memory_measurement = linux_procfs_process_tree_rss`), so report that for all
three tools. Likewise, break the 415.77 s down by stage (preimage / bit-width search
/ ESBMC / deployment evaluation — all four are already in
`table_scalability.csv`) so the 60× gap is attributed rather than merely stated.
No reviewer caught this; an expert on a second pass will.

Effort: 0.5 d.

### T16. Reconcile the internal numeric inconsistencies
- §6 says "19 obligations discharged"; Table 2 reports `Calls` = 21. One is wrong,
  or they count different things and neither is defined.
- Contribution 4 promises "five networks across three datasets"; §5 reports
  PreQ-BMC results on MNIST only (Iris and Seeds appear solely as CEG4N rows).
  Either report the Iris/Seeds PreQ-BMC arms — the repository has them — or narrow
  the contribution.
- The abstract's "decomposes wide layers … to reduce query size" must be reconciled
  with T1; after T1 it becomes a measured claim, which is the point.

Effort: 0.5 d, but do it *last*, after the numbers settle.

### T17. Report per-region results, not medians of small samples
R3 objects that the three-way comparison rests on one region. The honest fix is not
more medians but a per-region appendix table: every region, every tool, its verdict,
its budget consumption. With n ≈ 10 the raw table is more informative and more
credible than any summary statistic, and it removes the "statistical generalization"
objection by declining to generalize.

Effort: 0.5 d.

### T18. Scope Conv2D as future work with a concrete plan, not an apology `[R3]`
R3 asks for Conv2D support. That is a real engineering project, not a revision item:
DeepPoly, the MILP preimage encoding, the harness renderers, and the C generator all
assume dense layers. Do not promise it vaguely. State the specific extension (im2col
reduction to the existing affine machinery, per-channel scales as a per-block format
rather than a per-layer one) so the reviewer sees a plan.

Effort: 0.25 d of writing (the implementation is out of scope for this revision).

---

## Explicitly out of scope for this revision

State these as limitations rather than attempting them:

- **Convolutional and non-ReLU networks** (T18). Architectural.
- **Compiler verification.** §4.4 already scopes the claim to the generated C source.
  Keep that boundary and defend it; do not widen it.
- **Physical edge-hardware measurement.** R1's framing note (T5) is the right
  response: the contribution is the numerical semantics, not the board.
- **CEGAR at contract level** (R3's suggestion). Genuinely interesting and the
  repository has an exploratory `cegar` branch of results, but it is a research
  contribution in its own right, not a revision. Cite it as the primary future
  direction in §6, which also answers R3's "tighter preimage" comment.

---

## Suggested ordering

```
Week 1   T1 + T2 launch (compute runs in background)   ─┐
         T4, T5, T8, T9  (writing, no dependencies)     │ parallel
Week 2   T3, T11 aggregation once T1/T2 land            │
         T6, T7 related work                            │
Week 3   T12 (theorem), T13, T14                        │
Week 4   T15, T16, T17, T10, T18; full read-through    ─┘
```

T16 is deliberately last: it reconciles numbers that T1–T3 will change.
