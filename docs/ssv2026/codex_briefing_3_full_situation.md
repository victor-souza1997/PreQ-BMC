# Briefing 3: full situation, and what to do about it

Audience: a reasoning model with no prior context on this project, asked to
propose next steps. This document is self-contained. It states the goal, what
exists, what is proven versus measured versus assumed, what is blocked, and
what was tried and failed. Numbers are from artifacts in this repository as of
2026-09-21; paths are given so every claim can be checked.

---

## 1. The goal

Build a tool that **synthesises a quantized neural network (QNN) and proves the
correctness of its bit-precise C implementation**, for traffic-sign
classification on embedded hardware.

Three requirements, all of which must hold simultaneously. They are currently in
tension, and that tension is the whole problem:

1. **Soundness.** Proofs must be about the integer arithmetic that actually
   executes on the device — not a real-valued idealisation, not a float proxy.
   The verification backend is ESBMC (bounded model checking over bit-vectors).
2. **Accuracy retention.** The quantized network must keep the accuracy of the
   float32 source model. Traffic-sign classification is the application, so a
   certificate on a network that misclassifies is worthless — it certifies
   wrong answers.
3. **Scalability.** The method must reach network sizes that are useful. It
   currently does not.

Requirement 2 is the one most at risk and the least addressed so far. See §7.

---

## 2. What the tool does today

Per *region* — one image plus one L∞ radius ε in byte units — the pipeline runs
in this order (`src/synthesis/preqbmc.py`):

1. **Source gate.** Decide whether the **float** network is robust on the input
   box. Originally DeepPoly abstract interpretation; now an exact MILP
   (`source_verification=milp_exact`). If the float network is not robust, the
   region aborts and nothing downstream runs — correctly, since there is no
   true property to transfer.
2. **Bit-width selection.** Choose or verify a fixed-point format ⟨Q,I,F⟩ per
   layer. Q = total bits, I = integer bits, F = fractional bits.
3. **Backward preimage.** A MILP (CBC) computes, per layer, the set of inputs
   still yielding the correct class.
4. **Derived error budget.** A per-neuron integer tolerance δ in output ULPs,
   decomposed into `dw` (weight rounding) + `dr` (rounding/bias, always 1) +
   `dp` (inherited input error). The preimage is *deflated* by δ. If any
   neuron's deflated interval is empty the region reports
   `PREIMAGE_DEFLATION_EMPTY`.
5. **ESBMC layer contracts.** Generate C per neuron block, call ESBMC
   (`--z3 --bv`), require every block to verify.
6. **Contract chaining and input bridge.** Check the per-layer contracts compose
   and that the integer input box contains the true byte box.

Only if all six hold does the region report `VERIFIED` at guarantee level
`byte-crop-integer-C`.

### 2.1 The deployed integer semantics being verified

Per neuron: `acc = Σ w·x` (exact integer); `value = round_half_away_from_zero(acc, 2^F) + bias`;
`clamp(value, Q)`; ReLU (hidden layer only); `clamp` again. Formats in use:
hidden `h12` = 12 total / 3 integer / 8 fractional (clamp ±2047); output
`o14` = 14/5/8 (clamp +8191 / −8192).

### 2.2 The model under study

`Conv2D(2 filters, 3×3, stride 2, VALID) → ReLU → Flatten → Dense(43)` on an
8×8×3 input. The convolution is **lowered to a dense 18×192 Toeplitz matrix**
before verification, so the verifier only ever sees affine layers plus ReLU.
18 hidden ReLUs total, 27 nonzeros per hidden neuron out of 192 inputs.

**Float32 test accuracy: 41.7%.** This is the single most important number in
this document. See §7.

---

## 3. What is proven, what is measured, what is assumed

This separation is a hard requirement of the project and must be preserved in
any proposal.

**Proven (by ESBMC, over bit-precise C):**
- Per-block layer contracts for regions that reach step 5 and pass.
- Contract chaining and the input bridge for those regions.
- For one region, `image7_eps2`, the full chain: `final_status: VERIFIED`,
  `guarantee_level: byte-crop-integer-C`, 62 ESBMC calls, 221 s wall
  (`output/sign_milp_tightest_20260920/image7_eps2_beta1_cuts0/region_summary.json`).

**Measured (real numbers from runs, not proofs):**
- Source-gate outcomes, box widths, error budgets, runtimes.
- The MILP reachability diagnostic in §6.1.

**Assumed (and must be stated as such):**
- That the MILP solver's verdicts are correct. CBC is not a proof-producing
  solver and has already shown one dual-bound inconsistency on this model.
- Anything about on-device behaviour. `android_transfer_verified: False`,
  `android_parity: NOT_MEASURED`, `deployment_quality_status: NOT_MEASURED`.
  No claim about the embedded target has been validated.

---

## 4. Measured state

### 4.1 Source gate, all 54 regions

With the exact MILP source gate
(`output/sign_source_gate_audit_20260920/source_audit.json`):

```
total_regions: 54
by_status: {'VERIFIED': 22, 'REFUTED': 32}
esbmc_attempted: False        # float source gate only, no integer C here
```

This replaced DeepPoly, which left 44 of 54 regions undecided at the gate. Every
region is now decided. **But note what the decision says:** 32 of 54 regions are
`REFUTED` — the *float* network is genuinely not robust there, with a validated
counterexample. That is not a tool limitation. It is the model being weak.

So the honest reading of "22 verified" is: 22 regions where a true property
exists to be transferred, and 32 where there is nothing to prove because the
property is false.

### 4.2 End-to-end, integer C

Two regions have been run end-to-end through the full pipeline with the exact
gate (`output/sign_milp_tightest_20260920/`):

| region | result | guarantee level |
|---|---|---|
| `image7_eps2` | **VERIFIED** | `byte-crop-integer-C` |
| `image6_eps4` | `MARGIN_INCONCLUSIVE` | `unknown` |

`image6_eps4` is the subject of §5 and §6.

### 4.3 What predicts the outcome

Across the earlier DeepPoly-gated cohorts, outcome was almost entirely
determined by `clean_margin` and `epsilon` — two quantities known *before* any
verification runs. No region outside the high-margin stratum verified at any
radius; within the high stratum ε=1 verified 6/6 and ε=4 verified 0/6.

---

## 5. The open technical problem

`image6_eps4` reaches the output layer with the preimage available, all 18 block
contracts VERIFIED, and `contract_status: VERIFIED`. It then fails exactly one of
11 output competitor checks — against class 11 — and stops.

The failing assertion is `out[target] > out[competitor]` evaluated over the
**hidden interval box**, with the hidden vector drawn nondeterministically from
per-neuron bounds. Writing `H_Q` for the deployed quantized hidden map and
`X_ε` for the byte input box:

```
R_ε = { H_Q(x) : x ∈ X_ε }   ⊆   B_ε = the harness interval box
```

A failed assertion over `B_ε` exhibits a violating hidden state **in the box**.
It demonstrates a real vulnerability only if that state is reachable from `X_ε`.
`B_ε` is a Cartesian product of per-neuron intervals, so it contains
combinations of hidden values that no single input produces jointly.

The intended fix is **relational margin cuts**: each cut bounds a scalar
projection `d_k · h` of the hidden vector, restoring some of the inter-neuron
correlation the box discards. A cut must itself be ESBMC-verified before it may
be injected as an assumption. That rule is non-negotiable.

---

## 6. Results from 2026-09-21

### 6.1 The violating witness is unreachable — the region is not a true negative

An exact integer MILP was built over the **real input box** `X_ε` (192 integer
input variables, not a hidden box), maximising `out[11] − out[1]` under the
deployed fixed-point semantics.

```
validation - real executions violating the encoding : 0 of 4000
  [neuron 9 unsaturated] best objective -15, replays to -15 [EXACT] (OPTIMAL)
  [neuron 9 unsaturated] 'exists x with out[11] >= out[1]' -> INFEASIBLE
  [neuron 9 saturated]   best objective -15, replays to -15 [EXACT] (OPTIMAL)
  [neuron 9 saturated]   'exists x with out[11] >= out[1]' -> INFEASIBLE
```

The exact reachable margin maximum over `X_ε` is **−15**; the property fails only
at ≥ 0. The same margin over the interval box `B_ε` is **≈ +137**. The
abstraction admits the violation, the exact model does not, and the ~152 ULP gap
is precisely the discarded inter-neuron correlation.

The MILP is an exact model, not a relaxation, and each nonlinearity is discharged
by argument: ReLU and the low clamp are provably inactive (minimum layer-0
pre-activation bound is 86 > 0); only one neuron can reach the Q12 ceiling and it
is handled by an exact two-case split rather than big-M; the Q14 output clamps
provably cannot bind (logit range [−8125, 326] inside [−8192, 8191]);
round-half-away-from-zero is an integer bracket checked exhaustively for
**uniqueness** as well as admissibility. Two validations back it: no real
execution contradicts the encoding, and the MILP's witness replays through the
deployed integer path to exactly the claimed value.

Non-vacuity was checked by sweeping the threshold: feasible at −17, −16, −15 and
infeasible at −13, −1, 0, so the boundary sits exactly at the reported optimum.

**This is a diagnostic from a bespoke MILP, not an ESBMC proof.** It does not
discharge the obligation. `MARGIN_INCONCLUSIVE` remains the correct status. What
it establishes is *where to spend effort*: `image6_eps4` is a precision failure,
not a real vulnerability, so refinement on it is worth doing.

### 6.2 Integer width is not the blocker — tested and refuted

Every generated harness accumulates in `__int128`, far wider than needed: worst
layer-0 accumulator 480 729, worst cut direction value 388 119, i.e. ~4470×
headroom at int32. Narrowing the harness looked like a cheap win. **It is not.**

| harness | overflow-check | remaining VCCs | result |
|---|---|---|---|
| `__int128` | no | 55 | Timed out (1800 s) |
| int64 | no | 1 | Timed out |
| int32 | no | 1 | Timed out |
| int64 / int32 | yes | 7 346 | Timed out |
| int32 (Bitwuzla) | yes | 7 346 | Out of memory |

Cutting surviving VCCs from 55 to 1 changed nothing. **Bit-width is not the
hardness.** The single remaining VCC *is* the hard query: the exact deployed
prefix over all 192 inputs and all 18 hidden neurons at once. Also
`--overflow-check` makes the problem strictly worse (26 607 VCCs generated), so
a narrowed harness cannot even be cheaply proved overflow-free by that route.

### 6.3 Why cuts do not currently work

Enabling relational margin cuts was tried
(`output/sign_milp_cuts_probe_20260920/`). Exact-prefix validation of the
proposed cuts for classes 0, 2, 11 and 18 **timed out** at 300 s; the cut
targeting the failing class 11 was never established, so refinement never
reached the obligation it was meant to discharge. Two structural causes:

- **Ordering.** Cuts are proposed and validated for *every* analytically
  unresolved competitor, before any output check runs — so effort is spent on
  competitors that turn out not to matter.
- **Per-cut cost.** Each cut harness is the full 192-input → 18-neuron exact
  prefix. This is exactly the monolithic query that per-neuron block
  decomposition exists to avoid; block harnesses are 1–2 neurons each. Cuts
  couple the hidden neurons through their shared inputs and reintroduce the
  entire first-layer problem, even though every block contract already passed.

Making the ordering demand-driven (run output checks first, then cut only for
the failing competitor) is correct and worth doing, but **would not have rescued
this run** — it goes straight to the class-11 cut, which is the one that times
out. The binding constraint is the cost of a *single* exact-prefix cut
validation, not the number of them.

---

## 7. The accuracy problem, which is probably the real priority

The float32 source model is at **41.7% test accuracy** on GTSRB traffic signs.
Everything above is verification machinery operating on a network that is wrong
most of the time.

Consequences that follow directly from the measurements:

- 32 of 54 regions are `REFUTED` at the float gate. The network genuinely is not
  robust there. No verification improvement can change this; only a better model
  can.
- Outcome is predicted by `clean_margin`, and the low-margin stratum sits at
  clean margin ~0.001. Those regions are structurally hopeless.
- A robustness certificate on a misclassifying network certifies a wrong answer.
  For traffic signs this is not a cosmetic issue.

The network is tiny and trains in seconds, so this is cheap to attack: more
epochs and a better LR schedule; more filters; a larger input crop. **Constraint:
any architecture change must keep the affine lowering feasible.** `lower_conv`
(`src/models/restricted_conv.py:56`) raises if `n_in * n_out >
max_affine_entries`, default 1 000 000, because lowering a convolution to a
dense matrix is quadratic in feature-map size. Raising input resolution exhausts
this budget quickly. Check the budget before committing to a shape.
`scripts.prepare_ssv_gtsrb.validate_config` validates the study config that
feeds this.

Note also that requirement 2 of §1 — *maintain the accuracy of the float model* —
has **not been measured at all**. There is no recorded quantized-vs-float
accuracy comparison, no logit parity measurement, and no on-device validation.
The pipeline proves *local robustness certificates per region*; it does not
currently demonstrate that the quantized network matches the float network's
accuracy over the test set. That is a gap in the claim, not just in the results.

---

## 8. Hard constraints on any proposal

These are non-negotiable and a proposal violating them will be rejected outright.

- **Nothing that trades soundness for a nicer result.** If a fix requires
  weakening a contract, say so explicitly and reject it.
- Only ESBMC-verified cuts may be injected as assumptions.
- `MARGIN_INCONCLUSIVE` must not become `VERIFIED` without discharging the
  obligation. **A timeout is not a proof.** Neither is the MILP diagnostic of
  §6.1.
- No weakened contract, no heuristic tolerance, no disabled chaining, no
  per-block or per-partition formats, no change to the deployed arithmetic.
- **A conclusive negative is a legitimate result.** `REFUTED`,
  `PREIMAGE_DEFLATION_EMPTY` with a slack decomposition, and
  `MARGIN_INCONCLUSIVE` with a cause are all acceptable outcomes. Do not propose
  anything that converts one of these into an unsupported `VERIFIED`.
- Prefer changes that make the system **conclude more often** over changes that
  make it run faster.
- Verification must remain sound and must use ESBMC, and it has to scale.
- Distinguish throughout between what the code proves, what it measures, and
  what it assumes.
- If a line of inquiry turns up nothing, say so plainly. Do not manufacture
  findings.

---

## 9. Open questions

In rough priority order, but the ordering itself is a fair question to challenge.

1. **Is verification even the bottleneck?** With 41.7% accuracy and 32 of 54
   regions genuinely non-robust, is further work on the margin refinement
   problem the right investment, or should the model be fixed first? Argue from
   the measurements above.
2. **How to make one exact-prefix cut validation tractable.** The available
   structure: the lowered convolution is sparse (27 of 192 nonzeros per hidden
   neuron) with overlapping 3×3 stride-2 receptive fields on an 8×8×3 input, and
   a cut bounds a single scalar projection `d_k · h`. Does that projection admit
   a decomposition preserving exactness? Bit-width is ruled out (§6.2). Is there
   a formulation that keeps the per-block decomposition instead of collapsing to
   the monolithic query?
3. **Is there a sound way to use the reachability information of §6.1?** The
   exact reachable margin is known to be −15 with 15 ULPs of slack. Can that
   knowledge be turned into an ESBMC-discharged obligation rather than remaining
   an unusable diagnostic — for example by deriving cut directions from the MILP
   solution and then verifying them, which keeps the soundness rule intact?
4. **How should accuracy retention be measured and claimed?** Requirement 2 is
   currently unevidenced. What is the minimal honest experiment?
5. **Does the approach scale beyond 18 hidden ReLUs**, and if not, what is the
   actual limiting factor — the monolithic prefix query, the affine lowering
   budget, or the source gate?

---

## 10. Reproducing anything above

```
conda activate quad
export MPLCONFIGDIR=/tmp/preqbmc-mpl TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=src

# the two end-to-end runs in section 4.2
python src/scripts/run_ssv_gtsrb.py \
  --study output/sign_milp_source_h12_pilot_20260920/study.json \
  --output <new-dir> --only image7_eps2_beta1_cuts0 --only image6_eps4_beta1_cuts0

# the cuts probe in section 6.3
python src/scripts/run_ssv_gtsrb.py \
  --study output/sign_milp_cuts_probe_20260920_study.json \
  --output <new-dir> --only image6_eps4_beta1_cuts1

# the source-gate audit in section 4.1
python src/scripts/audit_ssv_source_gate.py --output <new-dir> \
  --prior-calibration <prior> --prior-evaluation <prior>

# the reachability diagnostic in section 6.1 (needs `mip`; CBC, not Gurobi)
python scratchpad/verify_enc.py          # rounding bracket is exact
python scratchpad/reach_milp.py          # validations + the decision
```

Notes for whoever runs this. ESBMC verdicts are in the **`.stderr.log`** files;
the `.stdout.log` files are empty. `gurobipy` is installed but unlicensed, so
MILP calls must pass `solver_name="CBC"`. `scratchpad/` is gitignored;
`reach_lib.py` parses the generated C files rather than re-transcribing their
numbers, so if those artifacts are regenerated the paths at its top need
updating. On CBC: a zero objective states the feasibility question most cleanly
but leaves CLP without a search direction and it cycles (`SIGABRT` in
`ClpSimplexProgress::looping`) on satisfiable thresholds; `--directed` adds an
objective direction without changing the feasible set.

Related documents: `docs/ssv2026/codex_briefing.md` (work items and claim
discipline), `docs/ssv2026/codex_briefing_2_margin_refinement.md` (the margin
problem in full detail), `docs/ssv2026/methodology.md` (proof scope and
limitations).
