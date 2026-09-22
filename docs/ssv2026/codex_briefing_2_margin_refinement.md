# Briefing 2: the output-margin bottleneck at large epsilon

Status date: 2026-09-21. Supersedes Work Item 1 of `codex_briefing.md`, which is
done and audited. This document states an open problem precisely. It does not
prescribe a solution, and it records which solutions are already eliminated by
evidence, so that effort is not spent re-deriving them.

## 1. What is settled

The opt-in exact MILP source gate landed (`d896bcc`) and was independently
audited: 54 regions, 22 VERIFIED / 32 REFUTED / 0 UNKNOWN, no regression across
the 10 prior certificates. All 32 counterexamples replay as genuine adversarial
examples of the float model; all 22 verified regions survive a falsification
sweep with no empirical margin below its certified lower bound. The float source
gate is no longer a bottleneck.

Three source-eligible regions have since been run end to end at the frozen
format h12/o14 F8 (hidden 12/3/8, output 14/5/8):

| region | eps | certified float margin | result | ESBMC calls | time |
|---|---|---|---|---|---|
| `image3_eps1` | 1 | 0.1653 | VERIFIED | 62 | 367 s |
| `image7_eps2` | 2 | 0.0387 | VERIFIED | 62 | 222 s |
| `image6_eps4` | 4 | 0.0568 | **MARGIN_INCONCLUSIVE** | 30 | 73 s |

The certified float margin does not predict the outcome: the tightest margin in
the whole cohort verified.

## 2. The failing obligation

`image6_eps4` reaches the output layer with preimage AVAILABLE, all 18 block
contracts VERIFIED and `contract_status: VERIFIED`. It then fails exactly one of
11 output competitor checks, against class 11, and stops (hence 30 = 18 + 11 + 1
calls). `final_status: MARGIN_INCONCLUSIVE`, `guarantee_level: unknown`.

The failing assertion is `out[TARGET_CLASS] > out[COMPETITOR_CLASS]` over the
hidden interval box, with the hidden vector drawn nondeterministically from
per-neuron bounds. Writing `H_Q` for the deployed quantized hidden map and
`X_eps` for the byte input box:

```
R_eps = { H_Q(x) : x in X_eps }  subset of  B_eps = the harness interval box
```

A failed assertion over `B_eps` exhibits a violating hidden state **in the box**.
It demonstrates a vulnerability of the QNN only if that state, or some other
violating state, is reachable from `X_eps`. It has not been shown that the
witness is unreachable. `MARGIN_INCONCLUSIVE` is therefore the correct status,
and a genuine fixed-point counterexample remains possible. **This is the first
question to settle, because if the QNN is genuinely non-robust here then all
refinement work on this region is wasted.**

It has now been settled — see §3. `MARGIN_INCONCLUSIVE` remains the correct
status regardless, for the reason given there.

## 3. The witness is an abstraction artefact (diagnostic, not a proof)

**Read this first: the result below is produced by a bespoke MILP, not by ESBMC.
It does not discharge the obligation and must not flip `MARGIN_INCONCLUSIVE` to
VERIFIED.** Its only job is to say where refinement effort is worth spending. On
that question it is decisive: `image6_eps4` is **not** a true negative, so the
refinement work is not wasted.

### 3.1 What was computed

An exact integer MILP over the *real* input box `X_eps` — 192 integer input
variables, not a hidden box — maximising `out[11] - out[1]` under the deployed
fixed-point semantics. Reproduce with `scratchpad/reach_milp.py` (see §10).

```
validation 1 - real executions violating the encoding: 0 (must be 0)
              best margin seen by sampling            : -377
  [neuron 9 unsaturated] best objective -15, replays to -15 [EXACT] (OPTIMAL)
  [neuron 9 unsaturated] 'exists x with out[11] >= out[1]' -> INFEASIBLE
  [neuron 9 saturated]   best objective -15, replays to -15 [EXACT] (OPTIMAL)
  [neuron 9 saturated]   'exists x with out[11] >= out[1]' -> INFEASIBLE
best witness found anywhere (needs >= 0 to violate): -15
```

The exact reachable maximum of the margin over `X_eps` is **−15**; the property
needs ≥ 0 to fail. Against this, the same margin evaluated over the harness
interval box `B_eps` is **≈ +137**. The abstraction admits the violation; the
exact model does not. That gap — ~152 ULPs — is precisely the inter-neuron
correlation the interval box discards: `B_eps` is a Cartesian product, so it
admits combinations of hidden values that no single `x` produces jointly.
**Lost correlation is therefore confirmed as the cause for this region**, which
is what makes relational cuts the right instrument rather than a guess.

### 3.2 Why the MILP is an exact model and not a relaxation

Each nonlinearity is discharged by an argument, not relaxed. Every item was
re-derived from the generated harnesses, which `scratchpad/reach_lib.py` parses
directly rather than re-transcribing, so the modelled arithmetic cannot drift
from the arithmetic that is verified and deployed.

| nonlinearity | treatment | justification |
|---|---|---|
| ReLU, Q12 low clamp | **dropped** | layer-0 pre-activation lower bounds are all positive; min = **86** |
| Q12 high clamp | exact 2-case split, no big-M | only neuron 9 can reach 2047; both branches solved |
| Q14 output clamps | **dropped** | logit range over the h-box is [−8125, 326], inside [−8192, 8191] |
| round-half-away-from-zero | exact integer bracket | `scratchpad/verify_enc.py`, below |

`verify_enc.py` checks the rounding bracket exhaustively over 2201 accumulators
(8.6 full periods of 256), in both sign branches, for two properties: the true
`q` is admitted, and **no other integer is**. Result: 0 true-`q` rejected, 0
spurious-`q` admitted. Uniqueness is what makes the bracket a reformulation
rather than a relaxation — without it the MILP could report a margin the
deployment never produces.

Two independent validations back the encoding empirically:

1. **No real execution contradicts it.** 4000 executions of the exact deployed
   semantics (including both box corners): 0 violate any structural claim the
   MILP relies on.
2. **The optimum replays exactly.** The MILP's witness `x`, run through the
   deployed integer path, reproduces the claimed objective to the ULP — the
   `[EXACT]` tag above. A relaxation would report a value the replay could not
   match.

### 3.3 Non-vacuity

An `INFEASIBLE` answer is only meaningful if the model is not trivially
infeasible. Sentinel sweep, same model with `margin >= t` for descending `t`:

| threshold | unsaturated branch | saturated branch |
|---|---|---|
| −17 | FEASIBLE (replays −15) | FEASIBLE (replays −15) |
| −16 | FEASIBLE (replays −15) | FEASIBLE (replays −15) |
| −15 | FEASIBLE (replays −15) | FEASIBLE (replays −15) |
| −14 | NO_SOLUTION_FOUND † | INFEASIBLE |
| −13 | INFEASIBLE | INFEASIBLE |
| −1 | INFEASIBLE | INFEASIBLE |
| 0 | INFEASIBLE | INFEASIBLE |

The feasible/infeasible boundary sits exactly at the reported optimum of −15:
every threshold at or below it is satisfied, every threshold above it is not.
That is what an exact model should do and what a vacuous one cannot.

† `NO_SOLUTION_FOUND` is the solver hitting its 400 s limit, not a proof of
infeasibility, and it is reported as such rather than folded into the
INFEASIBLE column. That cell is settled independently: the optimisation run on
the same branch in §3.1 terminated `OPTIMAL` at −15, which already excludes
−14. Note also that infeasibility at −13 does **not** imply infeasibility at
−14 — the constraint is weaker, so the feasible set is larger — which is why
the sweep is run threshold by threshold instead of inferred from one endpoint.

### 3.4 What this does and does not license

- **Proved by ESBMC:** nothing here. Unchanged.
- **Measured:** the exact reachable margin maximum over `X_eps` is −15, under an
  encoding validated as above.
- **Assumed:** that CBC's `INFEASIBLE` verdicts are correct. They are consistent
  with the optimisation runs and the sentinel boundary, but they are not
  machine-checked proofs, and CBC has already shown one dual-bound inconsistency
  on this model (which is why the decision is posed as a feasibility query
  rather than read off a bound).

So: `image6_eps4` is very probably robust, the region is worth refining, and its
status stays `MARGIN_INCONCLUSIVE` until ESBMC says otherwise.

## 4. What correlates with the outcome

Hidden-box widths in integer ULPs at F=8, from the generated harnesses:

| run | eps | width median | width max | box max | result |
|---|---|---|---|---|---|
| `image3_eps1` | 1 | 40 | 62 | 1920 | VERIFIED |
| `image6_eps1` | 1 | 48 | 62 | 2043 | VERIFIED |
| `image6_eps2` | 2 | 95 | 124 | 2047 | VERIFIED |
| `image7_eps2` | 2 | 118 | 125 | 996 | VERIFIED |
| `image6_eps4` | 4 | **188** | **248** | 2047 | MARGIN_INCONCLUSIVE |

Width tracks epsilon roughly linearly. It is dominated by the genuine reachable
range of the hidden activations, not by budget slack: the derived error budget is
~12 ULPs median in both the failing run and in `image3_eps1`, which verified.

Treat this as correlation across five runs, not as an established mechanism.

## 5. Eliminated by evidence — do not re-propose without new data

1. **"Hidden saturation at the Q12 ceiling is the cause."** `image6_eps2` also
   exceeds the ceiling (exact reachable max 8.047 > 7.996, box max pinned at
   2047) and verified. Saturation is survivable in at least one region.
2. **"More integer bits will fix it."** For `image6_eps2` the harness box widths
   are byte-identical at h12, h13 and h14 (95/124 in all three); only the clamp
   moves. **Caveat:** this is measured at eps=2 only. It does **not** establish
   equal behaviour at eps=4, and it does not establish that saturation is
   harmless in `image6_eps4` specifically. Running `image6_eps4` at h13 with
   `source_verification=milp_exact` is cheap and would close this properly.
3. **"Just enable relational margin cuts."** Tested and false as stated; see §6.

## 6. Enabling cuts exposes a second bottleneck

Probe run: `output/sign_milp_cuts_probe_20260920/`, a one-run study derived from
the h12 pilot with `margin_cuts=True` as the only change (verified to be the sole
difference from the frozen `cuts1` variant besides `run_id`). The process was
killed before producing a region report.

Exact-prefix validation of the proposed cuts for classes 0, 2, 11 and 18 **timed
out** at `timeout_seconds: 300`; class 19 has a solver-start line with no terminal
result. Critically, **the cut targeting the failing class 11 was never
established**, so the refinement never reached the obligation it was meant to
discharge.

Two structural causes, both confirmed in the code and artifacts:

- **Ordering.** `_margin_cut_bounds` is called at `src/synthesis/preqbmc.py:4565`
  and its result is passed to `_verify_output_margin_competitors_with_esbmc` at
  :4578. The loop at :3686 proposes a cut for every analytically unresolved
  competitor. So every cut is proposed and ESBMC-validated *before* any output
  check runs, whether or not that competitor turns out to be decisive.
- **Per-cut cost.** Each generated cut harness is `INPUT_SIZE 192` to
  `PREFIX_OUTPUT_SIZE 18` — the exact deployed prefix over all 192 inputs and all
  18 hidden neurons at once. This is the monolithic query that per-neuron block
  decomposition exists to avoid; the block harnesses are 1–2 neurons each. Cuts
  couple the hidden neurons through their shared inputs and reintroduce the full
  first-layer problem, even though every block contract already succeeded.

## 7. Demand-driven refinement: correct, but not sufficient here

The proposal — run output checks first, then attempt a cut only for the failing
competitor, then retry that obligation — is sound and clearly better than the
current ordering. It is soundness-neutral provided the existing rule is kept:
only an ESBMC-verified cut may become an assumption.

**It would not have rescued this run.** Demand-driven ordering would have gone
directly to the class-11 cut, which is precisely the one that timed out. The
saving is the wasted validation of classes 0, 2, 18 and 19 — real, roughly
4 x 300 s here — but the binding constraint is the cost of a *single* exact-prefix
cut validation at eps=4, not the number of them.

So there are two separable work items, and they should not be conflated:

- **W1 (cost ordering).** Make refinement demand-driven. Well-understood, low
  risk, strictly reduces wasted work.
- **W2 (the actual blocker).** Make one exact-prefix cut validation tractable at
  eps=4, or establish that it need not be. This is the open problem.

### 7.1 Integer width is NOT the blocker — tested and refuted

Every generated harness accumulates in `__int128`, which is far wider than
needed. Measured worst cases over the class-11 cut harness for `image6_eps4`:

| quantity | worst magnitude | int32 headroom | int64 headroom |
|---|---|---|---|
| layer-0 accumulator | 480 729 | 4.47e3 | 1.92e13 |
| cut direction value | 388 119 | 5.53e3 | 2.38e13 |

It looked like an easy win — narrow the harness (representation only, never the
deployed arithmetic) and give the solver less bit-width to chew on. The VCC
counts were encouraging: 15 541 generated at every width, but **55** remaining
after simplification at `__int128` versus **1** at int32 and int64.

**It does not work.** Every configuration still times out at 1800 s:

| harness | solver | overflow-check | remaining VCCs | result |
|---|---|---|---|---|
| `__int128` | Z3 | no | 55 | Timed out |
| int64 | Z3 | no | 1 | Timed out |
| int32 | Z3 | no | 1 | Timed out |
| int64 | Z3 | yes | 7 346 | Timed out |
| int32 | Z3 | yes | 7 346 | Timed out |
| int32 | Bitwuzla | yes | 7 346 | Out of memory |

Two conclusions, both negative and both worth keeping:

1. **Bit-width is not what makes this query hard.** Reducing the surviving VCCs
   from 55 to 1 changed nothing about tractability. The single remaining VCC
   *is* the monolithic query — the exact deployed prefix over all 192 inputs and
   all 18 hidden neurons at once (§6) — and that is the hardness, not the
   accumulator type.
2. **`--overflow-check` makes it strictly worse**, 26 607 VCCs with 7 346
   surviving. So the narrowing cannot even be cheaply *proved* safe by the
   obvious route; the headroom figures above are an interval computation of
   mine, not a machine-checked result. Any future use of a narrowed harness
   needs that overflow obligation discharged some other way.

So W2 remains open and the problem statement is now sharper: the target is the
monolithic 192→18 exact-prefix query itself. Note the cut is not vacuous — the
direction range is [−388 119, −40 251] against `CUT_LOW/HIGH = -379111 /
-47855`, so it sits strictly inside the interval range and the assertion has
real content. Decomposition of the projection `d_k . h`, exploiting the 27-of-192
sparsity, is the remaining idea; see §8.3.

## 8. What would settle it

In priority order.

1. ~~**Reachability of the violating witness.**~~ **Done — see §3.** The exact
   reachable maximum is −15 against a threshold of 0, so the region is not a
   true negative and refinement is worth doing. Note this came out far stronger
   than the "failure to find one is weak evidence only" that was expected of a
   search: the question was decided, not sampled. It does not discharge the
   obligation.
2. **`image6_eps4` at h13**, `source_verification=milp_exact`. Closes the caveat
   in §5.2 with a measurement instead of an extrapolation.
3. **W2.** Note the structure available: the lowered convolution is sparse, 27
   nonzeros of 192 per hidden neuron, with overlapping receptive fields
   (3x3, stride 2, on 8x8x3). A cut bounds a single scalar projection
   `d_k . h` of the hidden vector. Whether that projection admits a decomposition
   that preserves exactness is the question; no answer is assumed here.

## 9. Soundness constraints — unchanged and non-negotiable

- Only ESBMC-verified cuts may be injected as assumptions.
- `MARGIN_INCONCLUSIVE` must not become VERIFIED without discharging the
  obligation. A timeout is not a proof.
- No weakened contract, no heuristic tolerance, no disabled chaining, no
  per-block or per-partition formats, no change to deployed arithmetic.
- A conclusive negative is a legitimate result. If `image6_eps4` is genuinely
  non-robust, report that.

## 10. Reproducing the measurements

```
conda activate quad
export MPLCONFIGDIR=/tmp/preqbmc-mpl TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=src

# the two end-to-end runs in section 1
python src/scripts/run_ssv_gtsrb.py \
  --study output/sign_milp_source_h12_pilot_20260920/study.json \
  --output <new-dir> --only image7_eps2_beta1_cuts0 --only image6_eps4_beta1_cuts0

# the cuts probe in section 6
python src/scripts/run_ssv_gtsrb.py \
  --study output/sign_milp_cuts_probe_20260920_study.json \
  --output <new-dir> --only image6_eps4_beta1_cuts1

# the source-gate audit in section 1
python src/scripts/audit_ssv_source_gate.py --output <new-dir> \
  --prior-calibration <prior> --prior-evaluation <prior>

# the reachability diagnostic in section 3 (needs `mip`; CBC, not Gurobi)
python scratchpad/verify_enc.py          # 3.2: rounding bracket is exact
python scratchpad/reach_milp.py          # 3.1: validations + the decision
python scratchpad/reach_milp.py --threshold -15 --directed --max-seconds 400
```

`scratchpad/` is gitignored. `reach_lib.py` parses the two generated C files
rather than re-transcribing their numbers; if those artifacts are regenerated,
update the paths at the top of it.

On CBC: a zero objective states the feasibility question most cleanly, but it
leaves CLP without a search direction and it cycles (`SIGABRT` in
`ClpSimplexProgress::looping`) on the *satisfiable* thresholds. `--directed`
adds an objective direction without changing the feasible set, so a
FEASIBLE/INFEASIBLE answer means the same thing either way. The `margin >= 0`
query that carries the actual verdict runs clean without it.

Box widths and saturation are read directly out of the generated harnesses:
`<run>/layers/layers/output_competitors/*.c`, arrays `input_bounds_low` and
`input_bounds_high`. ESBMC verdicts are in the **`.stderr.log`** files; the
`.stdout.log` files are empty.
