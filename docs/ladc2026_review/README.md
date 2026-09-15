# LADC 2026 review response — document set

Working documents for revising *Preimage-Guided Bit-Precise Verification of
Fixed-Point Quantized Neural Networks* after LADC 2026 review (R1 Weak Reject
expert, R2 Weak Accept, R3 Weak Accept).

| Document | Purpose |
| --- | --- |
| [00_TODO.md](00_TODO.md) | Prioritized task list (P0/P1/P2), with effort and compute estimates and a suggested schedule |
| [01_RESPONSE_PLAN.md](01_RESPONSE_PLAN.md) | The revision strategy: for each objection, the argument to make, the evidence that would make it stick, and what would falsify it |
| [02_TRACEABILITY.md](02_TRACEABILITY.md) | Every reviewer comment → task → evidence artifact → response line; plus 11 author-identified defects no reviewer caught |
| [03_EXPERIMENT_MATRIX.md](03_EXPERIMENT_MATRIX.md) | Campaign design for the missing arms: flags, commands, region cohorts, result-table skeletons |
| [04_NOTATION_AND_TEXT_FIXES.md](04_NOTATION_AND_TEXT_FIXES.md) | Line-level editorial and notation corrections, including three substantive ones |
| [05_RELATED_WORK_GAPS.md](05_RELATED_WORK_GAPS.md) | QNNVerifier differentiation, QA-IBP, and a search protocol for LADC prior-edition citations |
| [06_TEXT_REVIEW.md](06_TEXT_REVIEW.md) | Full read of the submitted PDF: 19 findings across equations, symbol collisions, reported numbers, prose and references |
| [07_NOTATION_SCHEME.md](07_NOTATION_SCHEME.md) | The notation fix: convention, rename decisions, symbol table, rewritten equations, plus `notation.tex` macros and a `check_notation.sh` build guard |
| [08_CORRECTION_SCRIPT.md](08_CORRECTION_SCRIPT.md) | The ordered procedure against the actual `.tex`: ten passes, each edit given as before/after LaTeX with the reason it is wrong and the file that settles it |
| [fix_notation.pl](fix_notation.pl) | Applies the mechanical half of pass 08 (`--apply`, keeps a `.bak`); reports every judgement item with its rationale instead of rewriting it |

## The short version

No reviewer challenged the soundness of the method. Every blocking objection is
either an experiment that existing flags already support, or writing. The two
critical paths:

1. **Run the β = 0 monolithic arm** (`--esbmc-layer-block-size 0`) and the
   cuts-disabled arm (`--margin-cuts off`). All three reviewers named the first; it is
   the difference between a claim and an assertion.
2. **Reframe the "1 of 10" result.** The ten regions are a nested radius sequence on
   one sample, so the honest summary is a certification *frontier* plus a failure
   taxonomy — both of which the pipeline already records and the paper never printed.

The highest-value item nobody asked for is a stated and proved soundness theorem
(T12). Its absence is the largest scientific gap for a formal-methods venue, and
while re-deriving it, check Eq. (10): as printed, δ is indexed by output neuron `i`
but the first right-hand term is `i`-independent and uses the input scale where a
weight-quantization scale is expected.

**Standing constraint:** no obligation is weakened and no tolerance widened to improve
a number for the rebuttal. If an arm produces worse numbers, the worse numbers go in.
