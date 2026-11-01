# I-P: one bounded two-child input partition (016 c5), 2026-10-31

Uncommitted. A diagnostic finder screen (SCREEN_BOUND_UNCHECKED). No ESBMC certification, no export, and no VERIFIED count changes. This doc implements the Astra decision in `docs/claude_gpt_chat.md` (reply to Turn 8). Code:
- `src/scripts/ip_partition_016.py`;
- `src/verification/crown_partition_diag.py`;
- tests: `src/tests/test_crown_partition_diag.py`.

**Out of scope:** recursive branching, adaptive proof search, relational constraints, new checker rules, and any further finder optimization.

## Specification (written before the run)

- **Parent: the frozen P3c state of 016.**
  - Layer-2 and layer-3 cascade bounds are rebuilt from `output/p3c_cascade_pilot_20261031T192435Z/sides.jsonl.gz`.
  - On top of them goes the P3 margin search: c5 −647, 38/42 closed.
  - The rebuild must reproduce the P3c summary for both the ADAPTIVE and the margin-search configurations, and every policy chain must reproduce its logged value. Any mismatch is an AUDIT_STOP.
- **Control: 017, unchanged.**
  - The 017 parent is rebuilt the same way.
  - It is then recomputed through the child pipeline on the *unchanged* domain. Every bound, margin and closed set must be identical; otherwise AUDIT_STOP.
- **Architecture check (diagnostic only, never used by the split).**
  - Inputs: the layer-3 neurons in the D-T top-16 rankings of each image's focus competitors.
  - Recorded: their immediate layer-2 CSR predecessor sets and all pairwise intersections; also the shared two-hop (layer-1) support.
- **Split.**
  - **Coordinate.** From the parent c5 margin chain's final affine form, take the varying byte coordinate i that maximizes `|A_i|·(high_i − low_i)`, ties to the lowest index. It is frozen before any child is evaluated.
  - **Children.** With `m = floor((low_i + high_i)/2)`: child 0 has `b_i ≤ m`, child 1 has `b_i ≥ m+1`. Every other byte interval stays the parent's.
  - **Exact checks.** Disjoint, union equal to the parent, boundaries covered, no value dropped or duplicated, all other dimensions identical. Failure: INCONCLUSIVE.
  - **Partition accounting control.** The parent's c5 final affine form, minimized over each child box, must have its worst-child minimum equal to the parent's scaled bound exactly. Otherwise AUDIT_STOP (PARTITION_ACCOUNTING_ERROR).
- **Children, frozen proof choices.**
  - **What is fixed.** Every ReLU output coefficient `a'` in every chain (supporting chains of layers 0–3, margin roots, logit roots) is the one the parent made: ADAPTIVE, the P3c arm-A/B policies, and the P3 margin policies. The child computes it by calling the parent's choice function on the parent bounds.
  - **What changes.** Only the ReLU constants (exact breakpoint minimum on the child interval, with the parent's saturated/clipped classes), the residual lower endpoints and the input corners.
  - **Bound sources.** Box versus chain stays the parent's per side. Box sides use the child box propagation.
  - **Unsupported replay → INCONCLUSIVE.** This covers a parent box side unavailable in the child, a changed coefficient path, or an override that does not apply exactly once.
  - **Soundness checks (stop on failure).** Each child bound must be at least as tight as its parent bound (monotonicity), and must hold at every witness inside that child. Exact slack accounting of the child c5 chain must pass at those witnesses.
- **Evaluated.** Child c5, all 42 margins, every parent-closed competitor and its clamp conditions.
- **Recorded.**
  - Scaled and integer bounds, clamp rules, closed sets, and the digests and width sums of the intermediate selected bounds.
  - The dependency graph (chains per layer, policy chains, box facts needed) and multiply terms.
  - Wall time, peak memory, and every source and policy identity.
  - Planned assertion and harness counts and serialized size are **not measured**: there is no exporter path for replayed child chains. Multiply terms and chains are the measured proxies.
- **Budget.** 1,200 s for the 016 child stage and 8 GiB of memory; the parent rebuild and the 017 control are timed separately. Exceeding a limit → INCONCLUSIVE.

## Outcome classification (pre-registered)

| Outcome | Conditions |
|---|---|
| DIAGNOSTIC_PASS | All coverage, replay and arithmetic checks pass; worst-child c5 ≥ −614 (+33); all 38 parent-closed competitors stay closed in both children; 017 reproduces unchanged. |
| STRONG_PASS | DIAGNOSTIC_PASS, plus a worst-child gain ≥ 100, or a parent-open competitor closed in both children. |
| FAIL | A valid completed experiment with a worst-child gain < 33 (or the threshold met but a parent-closed competitor regressed). |
| INCONCLUSIVE | Unsupported frozen-policy replay, a missing dependency, a timeout, memory exhaustion, or incomplete child coverage. |

**Production-cost gate.**
- Combined child multiply terms ≤ 2× the parent's.
- Partition/bridge obligations counted separately: the coverage lemma, the child encoding contracts, the child non-vacuity witnesses, and the child box ledgers (block count ASSUMED from the 24-neuron block size).
- Memory within 8 GiB.

**What each outcome supports.**
- A DIAGNOSTIC_PASS supports only the claim that domain conditioning improves this proof under frozen choices. It does not justify recursive branch-and-bound, does not establish robustness, and changes no VERIFIED count.
- A FAIL rejects only this split under these frozen choices and this budget.

## Result (2026-10-31, MEASURED screen): FAIL

Run: `output/ip_partition_016_20261031T212249Z/` (441 s in total; child stage 283 s; peak RSS 1.42 GB; no audit stop).

1. **CSR check.** Immediate layer-2 predecessor supports are pairwise **disjoint** for all ranked layer-3 neurons:
   - 016: 33 neurons, 528 pairs, 4 predecessors each;
   - 013: 37 neurons, 666 pairs;
   - 017: 13 neurons, 78 pairs.

   The two-hop (layer-1) supports overlap in almost every pair: all pairs on 016 and 013, 74 of 78 on 017, with up to 1,592 shared layer-1 neurons. The hypothesis of disjoint immediate supports is **confirmed**. Dependence exists only upstream.
2. **Split.** Encoded/byte index 371, HWC (3, 27, 2), parent coefficient A = −166,146, byte interval [14, 16], m = 15. Child 0 = {14, 15}, child 1 = {16}. All exact partition checks passed. The partition accounting control passed: the parent form's child minima are −42,236,403 and −42,402,549, and the worst equals the parent's scaled bound.
3. **c5.** Parent −647; child 0 −644; child 1 −646.
4. **Worst-child gain.** +1, against a threshold of +33.
5. **Closed-competitor regressions.** None: both children keep all 38 parent-closed competitors. None newly closed in both.
6. **Dependency/cost.**
   - Each child has the same graph as the parent: 7,747 chains, 342,736,986 multiply terms, 1,553 policy chains, 11,625 box facts. Combined children are 2.00× the parent, exactly at the gate limit.
   - Separate bridge/partition obligations: 1 coverage lemma, 2 encoding contracts, 2 non-vacuity witnesses, and 2 full child box ledgers (1,178 blocks, ASSUMED from the 24-neuron block size).
   - Intermediate width sums shrink by ≤ 0.03 % per layer.
   - Assertions, harnesses and serialized size: not measured (no exporter path).
7. **Outcome: FAIL.** A valid, completed experiment with a worst-child gain below 33. The 017 control reproduced unchanged (388, 42/42).
8. **Production two-child composition rule: not justified** by this result.

**Reading (ASSUMED).** Halving one byte out of 3,072 removes almost no uncertainty. The best single coordinate contributes |A|·width = 332,292 scaled units (about 5 raw units) to the parent's input slack. The intermediate bounds barely move, so the root bound barely moves. This rejects only this split, under frozen choices. Per the spec it does not show that partitioning fails in general, and it does not validate I-R.
