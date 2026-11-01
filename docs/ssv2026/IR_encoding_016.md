# I-R encoding pilot: an exact reduced encoding of the same relational obligation (016 c5), 2026-11-01

Uncommitted. This is a single pilot following the Astra recommendation pasted after the I-R timeout. The original I-R result (`IR_upstream_016.md`, run `output/ir_upstream_016_20261031T235011Z/`) **stays INCONCLUSIVE**. Code:
- `src/scripts/ir_encoding_016.py`;
- `harness_reduced()` and `width_proof()` in `src/verification/crown_relational_diag.py`;
- tests in `src/tests/test_crown_relational_diag.py`.

**Unchanged from I-R.**
- The relation `e_13 + e_436 >= 2,162,688` (Δ = 33·2^16).
- The parent, the pair L3:13 / L3:436, every assumption and its provenance, the shared and decoupled contexts, and the Z3 `paper-z3` profile without `--result-only`.
- The budget: 2 queries, each 60 s and 4 GiB; a 600 s stage limit; an 8 GiB scope.
- The 250k exact-prefix cap.

No new production rule is implemented.

## Measurements behind the pilot (MEASURED, from the I-R run)

- **Symbolic execution was not the bottleneck.** Symex took 11.0 / 14.2 s; the 60 s timeout hit inside Z3: 21 VCCs, with 17,517 / 18,465 assignments after slicing.
- **Most inputs are fixed.** Of the 2,541 layer-1 variables, 1,811 have the interval [0, 0], leaving 730 free (810 in the decoupled harness).
- **Few shared inputs vary.** Only 80 of the 316 shared coordinates vary.
- **Most products are zero.** 3,002 of the 4,174 layer-2 products multiply a variable fixed at 0.

## The transformation (written before the run)

1. **Constant substitution.**
   - A layer-1 input whose assumed interval is a single value v becomes the constant v.
   - Constant products are folded into the starting value of their layer-2 accumulator (here all fixed values are 0).
   - Every layer-2 row, every r2 and r3 assumption and the target assertion are kept. A row is never removed because an activation is zero.
2. **Width reduction to int32, only with a complete mechanical justification.**
   - `width_proof()` bounds **every** intermediate of the reduced program in its exact evaluation order: products, partial sums, the folded constant, `num + 128`, `-num` and `(-num) + 128` inside rounding, the rounding result, bias addition, clamp and ReLU, the layer-3 accumulator, `a·h3`, `a'·r3`, their difference, `- d`, `e_A + e_B` and every literal.
   - Values before an assumption use the unassumed range; values after it use the intersected range.
   - Each bound must lie in [INT32_MIN + 1, INT32_MAX], so every negation stays representable. Any failure refuses the transformation, and the outcome is then INCONCLUSIVE.
   - The rounding, clamp and activation helpers are the deployed ones written in int32 with the same operations (`(num + 128) / 256` for den = 256).
3. **Equivalence argument.**
   - Under the assumptions, substitution is exact (a variable with a single admissible value).
   - With no intermediate leaving int32 in either program (the original uses `__int128`), both compute the same integers in the same C semantics: integer addition is associative when no overflow occurs, and truncating division is identical.
   - **Trusted:** the Python interval computation. Its arithmetic is exact integers.
   - **Supporting evidence, not the proof:**
     - gcc with `-fsanitize=signed-integer-overflow`, reduced vs original vs exact Python replay, on the 13 concrete states;
     - the unit tests on random contexts.
4. **Formula comparison before any query.**
   - ESBMC `--smt-formula-only` (no solver call) on the original and reduced harnesses, for both arms. Recorded: assignments after slicing, SMT bytes, `bvmul` count and BitVec width histogram.
   - **Run gate:** the two queries run only if the reduced formula differs materially from the original: fewer assignments after slicing, or no 128-bit vectors in the arithmetic. If the formula is equivalent in size and width, the queries are not run, and the result is `NO_REPRESENTATION_CHANGE`.
   - These formula-only invocations are measurement, not solver queries. Each has a 120 s cap, and their cost is recorded.
5. **Queries.** Shared first, then decoupled, with the reduced harnesses. Any target violation is replayed by the **original** exact replay (`rd.replay`) on the complete assignment: the trace values of the free inputs, plus the single admissible value of each fixed input.

## Outcome classification (pre-registered)

| Outcome | Conditions | Consequence |
|---|---|---|
| FAIL | Shared FAILED on the target, with an assignment confirmed by the original replay (complete, all assumptions hold, e-sum < Δ) | Refutes this lemma in this context. It is not a counterexample of the deployed network. |
| PASS | Shared VERIFIED, and decoupled FAILED on the target with a confirmed replayed assignment | Supports a dependence contribution to this +33. Still conditional on UNCHECKED dependencies (CONDITIONAL_ESBMC_LEMMA). |
| DEPENDENCE_NOT_DEMONSTRATED | Both VERIFIED | Sharing was not needed for this gain. |
| INCONCLUSIVE | Timeout or memout, refused width proof, parity failure, a non-target failure, or an incomplete or mismatched replay | **Close the pilots of this line for the current work.** |
| NO_REPRESENTATION_CHANGE | The formula comparison shows no material difference; no queries run | The original INCONCLUSIVE stands; close the line. |

If the result is again INCONCLUSIVE, the defensible statement is: *the proposed relation remained undecided under the contexts, encodings and budgets evaluated*. It is not a claim that relations are infeasible or that they cause the incompleteness.

## Result (2026-11-01): INCONCLUSIVE; the line is closed for the current work

Run: `output/ir_encoding_016_20261101T001547Z/` (222 s in total).

- **Width proof: passed.**
  - 2,432 intermediates per arm.
  - The largest is |e_A + e_B| ≤ 20,442,160, far inside int32.
  - The largest layer-2 partial accumulator is 187,084, confirming the Astra figure.
- **Reduction.** 1,811 fixed inputs, all with the value 0. Free inputs: 730 in the shared harness, 810 in the decoupled one; 80 shared coordinates vary. 3,002 layer-2 products removed and 1,172 kept.
- **Parity: passed.** The original harness, the reduced harness and the exact Python replay agree on all 13 concrete states, for both arms (26 checks, signed-overflow sanitizer on).
- **Formula comparison** (`--smt-formula-only`, no solver call):

  | Arm | Assignments (original → reduced) | SMT bytes (original → reduced) | bvmul | BitVec widths, original → reduced |
  |---|---|---|---|---|
  | shared | 17,517 → 3,654 | 4.10 MB → 0.97 MB | 1,172 → 1,172 | 128-bit: 8,594 and 64-bit: 8,822 → 32-bit: 3,542 and 64-bit: 11 |
  | decoupled | 18,465 → 3,894 | 4.46 MB → 1.05 MB | 1,172 → 1,172 | 128-bit: 8,594 and 64-bit: 9,770 → 32-bit: 3,782 and 64-bit: 11 |

  **MEASURED: ESBMC had already removed the zero products.** The original formula carried the same 1,172 multiplications. The material change is width (no 128-bit vectors) and about 4.8× fewer assignments, so the run gate passed.
- **Shared query: TIMEOUT** at 60 s (peak 0.53 GB, against 3.1 GB before).
- **Decoupled query: TIMEOUT** at 60 s (peak 0.55 GB).
- **Outcome: INCONCLUSIVE.** By the pre-registered rule, the pilots of this line are closed for the current work.

**Defensible statement.** The proposed relation `e_13 + e_436 ≥ 33·2^16` remained undecided under the contexts, encodings and budgets evaluated:
- two contexts (shared and decoupled);
- two exact encodings (`__int128` deployed form, and the int32 reduced form);
- Z3, 60 s per query.

**What this does not mean.**
- It does not show that relations are infeasible.
- It does not show that they cause the incompleteness.
- It does not change the I-R result, which is INCONCLUSIVE.

What stays measured but unproved: the relation holds by a factor of about 2 on all 13 sampled reachable states. That is not evidence over the context.
