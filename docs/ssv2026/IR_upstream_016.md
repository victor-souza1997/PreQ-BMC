# I-R (upstream): shared layer-1 context for layer-3 neurons 13 and 436 (016 c5), 2026-10-31

Uncommitted. A bounded diagnostic, specified by João (pasted request after the I-P FAIL). This doc records the spec as implemented, before the run. Code:
- `src/scripts/ir_upstream_016.py`;
- `src/verification/crown_relational_diag.py`;
- tests: `src/tests/test_crown_relational_diag.py`.

**Out of scope:** finder redesign, new production certificate rules, another neuron pair after failure, a weaker +33 threshold, and a larger context or solver budget after results are seen.

## Specification

- **Parent.** The frozen P3c state of 016 with P3 margin search (c5 −647), rebuilt and checked exactly as in `IP_partition_016.md`.
- **Preflight (before any harness).**
  - Reproduce c5 = −647.
  - Read a, a', d, the raw interval, the class and the cited fact identities for L3:13 and L3:436 from the recorded c5 margin chain; recompute d from the interval.
  - Recompute the CSR cone: 8 layer-2 rows, the union and shared layer-1 coordinates, and the multiply terms. Expected: 2,541, 316 and ≈ 4,182.
  - Any mismatch stops the run before solving.
- **Algebraic precondition.**
  - In the recorded chain, every step after the layer-3 ReLU adds a term that does not depend on the incoming constant. So raising the layer-3 ReLU constant by Δ = 33·2^16 = 2,162,688 gives exactly `new_scaled = parent_scaled + Δ`, and `ceil(new_scaled / 2^16)` must equal −614.
  - e_j at the nominal input must equal the per-coordinate ReLU slack of `account_chain` at L3:j.
  - Otherwise ALGEBRAIC_PRECONDITION_FAILED.
- **Shared-context harness.**
  - Symbolic layer-1 activations for the 2,541 union coordinates, one variable per coordinate.
  - The 8 layer-2 neurons and L3:13 and L3:436 are computed in the deployed arithmetic: the `mac_i128` accumulator, `div_round_half_away_from_zero_i128(acc, 256) + bias`, then clamp, ReLU, clamp. The helpers are copied verbatim from the deployed C, and the weights and biases come from the deployed model lowering.
  - It asserts `e_13 + e_436 >= 2162688`, under a unique label `diag/016/I-R/L3n13+L3n436/<arm>`.
- **Unary assumptions (explicitly identified; every one recorded with its provenance).** Each is a frozen parent fact on a side the current checker can cite:
  - **Layer 1:** `x_k ∈ [max(0, H(lo_k)), H(hi_k)]`.
    - The lower side applies only if citable: a box lower for stable-active neurons, or a chain lower for unstable ones.
    - The upper side applies only if citable: a chain upper, or a proved box upper (`hh < Q_HIGH`, not saturated).
    - Otherwise the semantic range [0, Q_HIGH] of an activation applies.
  - **Layer 2:** the raw r2 interval of each of the 8 rows, by the same citability rule.
  - **Layer 3:** the raw r3 interval of 13 and 436, by the same rule. The parent's d_j were computed on this interval.
- **Decoupled control.** Identical, except that each of the 316 shared coordinates gets one copy per cone, with the same interval. Within a cone, sharing is unchanged. This is checked mechanically: the decoupled harness differs from the shared one only by those copies.
- **Fidelity checks before solving.**
  - Compiled with gcc on the nominal hidden state, both harnesses must give the same e values as the exact Python replay.
  - The nominal state must satisfy every assumption (non-vacuity). Its r3 values must equal the exact forward pass.
  - e_13 + e_436 at the nominal input and at every retained witness is recorded (MEASURED).
- **Budget.** 2 solver queries (shared, then decoupled), each with 60 s and 4 GiB, using `paper-z3` without `--result-only`. The stage limit is 600 s, at most 10,000 local multiply terms, and 8 GiB for the stage. The 250k exact-prefix cap is unchanged. The solver controls are not rerun: they are validated by the unit tests, and they do not count against the 2 queries.

## Outcome classification (pre-registered)

| Outcome | Conditions |
|---|---|
| PASS | Shared VERIFIED; decoupled FAILED on the target assertion, with a complete trace assignment that the exact replay confirms satisfies every assumption with e_13 + e_436 < Δ; the algebraic precondition gives −614; all identity, arithmetic and resource checks pass. |
| FAIL | Shared FAILED on the target assertion, with a complete assignment whose replay satisfies every assumption and violates the relation. This rejects only this pair, this context and the +33 relation. |
| DEPENDENCE_NOT_DEMONSTRATED | Both shared and decoupled VERIFIED. |
| INCONCLUSIVE | Timeout, memout, a missing dependency, an incomplete trace or replay mismatch, a non-target (safety or unwinding) failure, an arithmetic or parity failure, or an unsupported assumption context. |

**Evidence label.** A VERIFIED shared query is at most a CONDITIONAL_ESBMC_LEMMA. The cited predecessor facts are finder screen values and box screen facts, none of them ESBMC-checked in this run. The decoupled assignment is a hidden-state control assignment, not a network counterexample. No VERIFIED count changes.

## Result (2026-10-31): INCONCLUSIVE (both queries TIMEOUT at 60 s)

Run: `output/ir_upstream_016_20261031T235011Z/` (147 s in total).

1. **Structure (MEASURED).**
   - 8 layer-2 rows: 13 → {13, 205, 781, 973}; 436 → {1588, 1780, 2356, 2548}. The two sets are disjoint.
   - Layer-1 coordinates: 1,594 / 1,263 per cone; union 2,541; shared 316.
   - 4,182 multiply terms. All match the expected values.
2. **Frozen relaxation (MEASURED).** Both coordinates are in class `lower_upper`.
   - L3:13: a = −26,368, a' = −20,844, d = −2,855,908, raw [−137, 517].
   - L3:436: a = −19,456, a' = −9,432, d = −3,508,704, raw [−372, 350].
   - d was recomputed exactly from the interval.
   - The cited facts are chain-backed with P3c policies:
     - lower sides: arm A, policy [−1, −1, 0], on both neurons;
     - upper sides: arm B, `a' = 0` on L1:1833 (for 13) and on L1:3220 (for 436).
3. **Algebraic +33 precondition: PASSED.**
   - Rerunning the frozen c5 chain with the layer-3 ReLU constant raised by Δ gives exactly −42,402,549 + 2,162,688 = −40,239,861, so ceil = −614.
   - e_13 = 2,042,992 and e_436 = 2,225,952 at the nominal input. Both equal the `account_chain` ReLU slack exactly.
4. **Shared-context query: TIMEOUT** at 60 s (Z3, 21 VCCs, 17,517 assignments, peak 3.1 GB). No conclusion.
5. **Decoupled control: TIMEOUT** at 60 s (peak 3.3 GB). No conclusion.
6. **Replay.** No solver assignment, so nothing to replay. Fidelity checks before solving all passed:
   - gcc parity of both harnesses on the nominal state;
   - harness-arithmetic r3 equal to the exact forward pass on all 13 concrete states;
   - every concrete state satisfies every assumption (non-vacuity).
7. **Solver resources.** 60.06 s and 60.08 s wall; 62.1 s and 62.2 s CPU; peak 3.11 GB and 3.28 GB.
8. **Assumptions** (`assumptions.json`). Layer 1: 2,541 upper sides and 1,039 lower sides cited. Layer 2: 16 sides. Layer 3: 4 sides. Eight of them are policy-chain facts. Dependency status: UNCHECKED, because box facts are screen facts and chain facts are finder values.
9. **Local lemma size.** Shared harness 435,348 bytes (2,541 variables); decoupled 467,658 bytes (2,857 variables).
10. **Outcome: INCONCLUSIVE** (timeout). Per the spec this is not FAIL and not DEPENDENCE_NOT_DEMONSTRATED.

**Measured, not a solver result.** e_13 + e_436 ranges from 4,268,944 to 6,244,700 on all 13 concrete states (nominal plus 12 retained witnesses). That is about 2× Δ on every one, so no sampled real state violates the relation.

**What this supports.** The +33 relation is algebraically sufficient in the frozen chain. It holds, with room to spare, on every sampled reachable state. Deciding it over the stated context is beyond 60 s for Z3 at this size.

**What it does not establish.**
- It does not establish the relation over the context; no lemma was proved.
- It does not establish that shared ancestry is what makes it hold, because the decoupled control is also undecided.
- It does not establish any robustness or VERIFIED change.

Sampled states are not a proof, and they say nothing about the decoupled worst case. Per the spec, the budget is not enlarged after seeing results.
