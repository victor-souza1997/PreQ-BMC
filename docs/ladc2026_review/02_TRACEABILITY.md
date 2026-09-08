# Reviewer comment → action traceability matrix

Every distinct comment from the three reviews, mapped to a task in
[00_TODO.md](00_TODO.md) and to the artifact that will evidence it. Use the
"Response" column as the seed for the response-to-reviewers letter; use the
"Status" column as the working checklist.

Status key: `todo` / `doing` / `done` / `declined` (with reason in the response).

## Reviewer 1 (Expert, Weak Reject)

| # | Comment | Task | Evidence artifact | Response line | Status |
| --- | --- | --- | --- | --- | --- |
| R1.1 | Only 10 % certification on MNIST; remaining 90 % inconclusive | T3, T11 | failure-mode table; timeout-sensitivity curve | Reframed as a radius frontier; added a failure taxonomy separating source-ineligibility, preimage precision, and solver cost; added a budget-sensitivity curve | todo |
| R1.2 | Scalability claimed from block partitioning without the β = 0 baseline | T1 | `table_ablation.csv`, `table_scalability.csv` for β ∈ {0,1,2,5,10} | Added the monolithic arm and the full sweep; primary metric is now max per-query memory/time, not total wall time | todo |
| R1.3 | Introduction redundant across the first four paragraphs | T4 | §1 rewrite | Restructured to a single pass; duplication removed | todo |
| R1.4 | Scope restricted to small dense ReLU nets, not stated up front | T5 | abstract, §4.1, §6 | Scope now stated in the abstract and as a standing assumption in §4.1 | todo |
| R1.5 | Notation used before definition; `f`, `f_c` undefined | T9 | §2, §4 | All symbols defined at first use; notation table added | todo |
| R1.6 | Relate quantization error to vulnerability injection / control deviation in critical systems (LADC framing) | T7 | §1, §3 | Added dependability framing and LADC prior-edition citations | todo |
| R1.7 | Document architectural dependencies: dense MLP + ReLU only, no convolution, no per-channel quantization | T5, T18 | abstract, §4.1, §6 | Stated explicitly; convolution scoped as future work with a concrete path | todo |
| R1.8 | State that x86 evaluation is a numerical model, not an edge platform | T5 | §5.1 | Added to the experimental-setup section | todo |
| R1.9 | Tabulate reasons for inconclusive runs (timeout vs MILP amplitude error) | T3 | failure-mode table | Added; the pipeline already records the taxonomy per run | todo |
| R1.10 | Evaluate cutoffs beyond 300 s | T11 | sensitivity curve | Added; shows whether the limit is structural or budgetary | todo |
| R1.11 | Differentiate from QNNVerifier (also ESBMC-based) | T6 | Table 1 row + §3 paragraph | Added; differentiated on synthesis vs analysis, compositional vs monolithic queries, preimage guidance, exported backend | todo |
| R1.12 | Discuss QA-IBP and robust training with discrete guarantees | T6 | §3 | Added as a complementary (certified-training) line | todo |

## Reviewer 2 (Familiar, Weak Accept)

| # | Comment | Task | Evidence artifact | Response line | Status |
| --- | --- | --- | --- | --- | --- |
| R2.1 | Abstract is dense; abbreviations unexplained; contribution vs Quadapter unclear | T8 | abstract rewrite | Rewritten to state what is generated, what is verified, what is new vs Quadapter, what was measured | todo |
| R2.2 | Evaluation base is narrow | T3, T17 | full-campaign tables; per-region appendix | Broadened the reported campaign and added per-region results; scope limits stated rather than generalized over | todo |
| R2.3 | §2.2 lowercase `i` vs format ⟨N, I, F⟩ | T9 | §2.2 | Fixed | todo |
| R2.4 | Table 2 "Calls" column unexplained | T9, T16 | Table 2 caption | Defined as the number of ESBMC queries executed; reconciled with the count in §6 | todo |

## Reviewer 3 (Some knowledge, Weak Accept)

| # | Comment | Task | Evidence artifact | Response line | Status |
| --- | --- | --- | --- | --- | --- |
| R3.1 | Missing β = 0 arm and cuts-disabled arm | T1, T2 | ablation tables | Both arms added | todo |
| R3.2 | 1 of 10 MNIST regions conclusive; scalability limits | T3, T11 | frontier + taxonomy tables | Reframed and diagnosed | todo |
| R3.3 | No CNNs or non-ReLU activations | T18 | §6 | Declined for this revision with a stated extension path (im2col + per-channel formats) | todo |
| R3.4 | Three-way comparison rests on a single region | T17 | per-region appendix table | Reported per region without summary statistics; the single matched region is stated as a capability result, not a rate | todo |
| R3.5 | Investigate tighter preimages or contract-level CEGAR | T12 note, §6 | §6 | Declined as out of scope for the revision; cited as the leading future direction, tied to the measured failure profile | todo |
| R3.6 | Figure 1 low resolution | T10 | vector figure | Re-rendered as vector; status vocabulary unified with Algorithm 1 | todo |

## Author-identified items (not raised by reviewers)

These were found while auditing the paper against the implementation. They are not
required by any review, but each is the kind of defect an expert finds on a second
pass.

| # | Finding | Task | Why it matters |
| --- | --- | --- | --- |
| A.1 | No soundness theorem; obligations composed only in prose | T12 | The central scientific gap for an FM venue |
| A.2 | Eq. (10): δ indexed by `i` but the first RHS term is `i`-independent; `S_{l-1}` appears where a weight scale is expected | T12 | Either a transcription error or a modelling error in the paper's only derived bound |
| A.3 | B_ε(x₀) defined clipped in §2.3, unclipped in Eq. (5) | T9 | Two different sets; the implementation uses the clipped one |
| A.4 | §6 reports 19 discharged obligations; Table 2 reports 21 calls | T16 | Internal inconsistency in the headline result |
| A.5 | Contribution 4 claims five networks across three datasets; §5 reports PreQ-BMC on MNIST only | T16 | Unsupported contribution claim |
| A.6 | Table 2 peak memory: max-query RSS for PreQ-BMC vs process-tree RSS for baselines | T15 | Two different quantities presented as comparable |
| A.7 | 415.77 s reported without stage attribution | T15 | The 60× gap is partly preimage/synthesis/evaluation, not solving |
| A.8 | `__int128` representability is an undischarged hypothesis inside the fidelity claim | T14 | The harness to close it already exists |
| A.9 | Eq. (1)–(2) introduce ρ_l; the following text calls it σ^[l] | T9 | Symbol drift in the first equation of the paper |
| A.10 | §4 opening paragraph has mangled superscripts (`f^{2[l]}`, `P^{2[l]}`) and a sentence fragment | T9 | Renders the core methodology paragraph unreadable |
| A.11 | Status vocabulary differs across Fig. 1, Alg. 1, §4.8, §5 | T13 | The implementation has one taxonomy; the paper should print it |
