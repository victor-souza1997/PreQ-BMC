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

## 3. What correlates with the outcome

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

## 4. Eliminated by evidence — do not re-propose without new data

1. **"Hidden saturation at the Q12 ceiling is the cause."** `image6_eps2` also
   exceeds the ceiling (exact reachable max 8.047 > 7.996, box max pinned at
   2047) and verified. Saturation is survivable in at least one region.
2. **"More integer bits will fix it."** For `image6_eps2` the harness box widths
   are byte-identical at h12, h13 and h14 (95/124 in all three); only the clamp
   moves. **Caveat:** this is measured at eps=2 only. It does **not** establish
   equal behaviour at eps=4, and it does not establish that saturation is
   harmless in `image6_eps4` specifically. Running `image6_eps4` at h13 with
   `source_verification=milp_exact` is cheap and would close this properly.
3. **"Just enable relational margin cuts."** Tested and false as stated; see §5.

## 5. Enabling cuts exposes a second bottleneck

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

## 6. Demand-driven refinement: correct, but not sufficient here

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

## 7. What would settle it

In priority order.

1. **Reachability of the violating witness.** Extract the hidden state from the
   failed class-11 check in `output/sign_milp_tightest_20260920/` and decide
   whether any `x` in `X_eps` yields it under `H_Q`. Equivalently, attack the
   deployed quantized network over `X_eps` directly. A genuine violation makes
   this region a true negative and ends the refinement question for it; failure
   to find one is weak evidence only. This is cheap and should come first.
2. **`image6_eps4` at h13**, `source_verification=milp_exact`. Closes the caveat
   in §4.2 with a measurement instead of an extrapolation.
3. **W2.** Note the structure available: the lowered convolution is sparse, 27
   nonzeros of 192 per hidden neuron, with overlapping receptive fields
   (3x3, stride 2, on 8x8x3). A cut bounds a single scalar projection
   `d_k . h` of the hidden vector. Whether that projection admits a decomposition
   that preserves exactness is the question; no answer is assumed here.

## 8. Soundness constraints — unchanged and non-negotiable

- Only ESBMC-verified cuts may be injected as assumptions.
- `MARGIN_INCONCLUSIVE` must not become VERIFIED without discharging the
  obligation. A timeout is not a proof.
- No weakened contract, no heuristic tolerance, no disabled chaining, no
  per-block or per-partition formats, no change to deployed arithmetic.
- A conclusive negative is a legitimate result. If `image6_eps4` is genuinely
  non-robust, report that.

## 9. Reproducing the measurements

```
conda activate quad
export MPLCONFIGDIR=/tmp/preqbmc-mpl TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=src

# the two end-to-end runs in section 1
python src/scripts/run_ssv_gtsrb.py \
  --study output/sign_milp_source_h12_pilot_20260920/study.json \
  --output <new-dir> --only image7_eps2_beta1_cuts0 --only image6_eps4_beta1_cuts0

# the cuts probe in section 5
python src/scripts/run_ssv_gtsrb.py \
  --study output/sign_milp_cuts_probe_20260920_study.json \
  --output <new-dir> --only image6_eps4_beta1_cuts1

# the source-gate audit in section 1
python src/scripts/audit_ssv_source_gate.py --output <new-dir> \
  --prior-calibration <prior> --prior-evaluation <prior>
```

Box widths and saturation are read directly out of the generated harnesses:
`<run>/layers/layers/output_competitors/*.c`, arrays `input_bounds_low` and
`input_bounds_high`. ESBMC verdicts are in the **`.stderr.log`** files; the
`.stdout.log` files are empty.
