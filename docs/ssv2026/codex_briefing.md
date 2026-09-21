# PreQ-BMC: what must be built next, and why

**Audience.** An architect model that will assess feasibility and then write
hands-on instructions for a smaller implementation model. Every code anchor
below is verified against the working tree as of 2026-09-20 (branch
`feat/cnn-feature`). Line numbers are approximate to ±5; function names and
signatures are exact.

**Goal of the project.** Produce a fixed-point quantized neural network,
exported as an integer C program, that is *provably* robust on specified input
regions, and deploy that exact program on an embedded target. The proof must be
machine-checked (ESBMC) and must apply to the deployed arithmetic, not to a
float idealization of it.

---

## 1. What the pipeline does today

Per region (one image + one L∞ radius), in order:

1. **Source gate** — `GPEncoding.check_source_region()`
   (`src/synthesis/preqbmc.py:1073`). Runs DeepPoly abstract interpretation on
   the **float** network over the input box. Computes
   `certified_margin_lower_bound = lb[target] − max_{j≠target} ub[j]`. If that
   is `< 0`, the whole region aborts with `SOURCE_PROPERTY_INCONCLUSIVE`
   (`preqbmc.py:959`) and **nothing downstream runs**.
2. **Bit-width selection / fixed-format check** — chooses or verifies ⟨Q,I,F⟩
   per layer.
3. **Backward preimage** — MILP (CBC) computes, per layer, the set of inputs
   that still yield the correct class.
4. **Derived error budget** — per-neuron integer tolerance δ in output ULPs,
   decomposed `dw` (weight rounding) + `dr` (RHAZ/bias, always 1) + `dp`
   (inherited input error). The preimage is *deflated* by δ; if any neuron's
   deflated interval is empty → `PREIMAGE_DEFLATION_EMPTY`.
5. **ESBMC layer contracts** — generates C per neuron block, calls ESBMC
   (`--z3 --bv --unwind N`), and requires every block to verify.
6. **Contract chaining + input bridge** — checks layer contracts compose and
   that the integer input box contains the true byte box.

Only if all of 1–6 hold does the region report `VERIFIED` with guarantee level
`byte-crop-integer-C`.

### Model under study

Restricted CNN on GTSRB: `Conv2D(2 filters, 3×3, stride 2, VALID) → ReLU →
Flatten → Dense(43)`, input 8×8×3. The convolution is **lowered to a dense
18×192 Toeplitz matrix** before verification, so the verifier only ever sees
affine layers + ReLU. 18 hidden ReLUs total. Source float32 test accuracy:
**41.7%**.

---

## 2. Measured state (2026-09-20, post-budget-fix)

A sparse-aware fix to the derived error budget landed this session (see §6). All
numbers below are from after that fix.

**Fixed-QIF sweep**, 4 ordered candidates, all F=8, calibration cohort of 27
regions:

| candidate | hidden Q/I/F | output Q/I/F | verified / source-eligible |
|---|---|---|---|
| h11_o13_f8 | 11/2/8 | 13/4/8 | 2/4 |
| **h12_o14_f8** | **12/3/8** | **14/5/8** | **4/4 ← selected** |
| h13_o15_f8 | 13/4/8 | 15/6/8 | 4/4 |
| h14_o16_f8 | 14/5/8 | 16/7/8 | 4/4 |

**Independent evaluation cohort** (36 runs, disjoint images, frozen artifact,
no format re-search): 9 VERIFIED, 27 `SOURCE_PROPERTY_INCONCLUSIVE`, 0 FAILED.

**Aggregated over both cohorts, deduplicated to distinct image×radius pairs
(54 total):**

| clean margin | stratum | pairs | verified | reached ESBMC |
|---|---|---|---|---|
| ~2.05 | high | 18 | 10 | 10 |
| ~0.61 | median | 18 | 0 | 0 |
| ~0.001 | low | 18 | 0 | 0 |

### The diagnosis

**44 of 54 pairs (81%) never invoke ESBMC at all** — they die at the float
source gate with `esbmc_calls = 0`.

No region outside the high-margin stratum verified at any radius. Within the
high stratum, ε=1 verified 6/6 and ε=4 verified 0/6. **The outcome table is
almost entirely predicted by `clean_margin` and `epsilon`, two quantities known
before any verification runs.**

DeepPoly's abstraction loss at ε=1/256 measures **0.6–1.7 margin units** on this
network. The median stratum's entire clean margin is 0.612 — *smaller than the
typical loss*. Those 18 pairs are structurally unreachable by DeepPoly no matter
what quantization is chosen.

Ranked near-misses (`certified_margin_lower_bound`, negative = gate failed):

```
-0.129  image7_eps2           high
-0.196  image5_eps1           median
-0.235  image3_eps1           median
-0.336  cohort2_image4_eps1   median
-0.350  cohort2_image3_eps1   median
-0.457  cohort2_image7_eps4   high
-0.605  image4_eps1           median
-0.682  cohort2_image5_eps1   median
-0.832  image6_eps4           high
-0.880  image1_eps1           low
```

**Consequence:** the experiments currently measure DeepPoly's precision on an
18-ReLU network, not the contribution the tool is claiming. The quantization
search has exactly one meaningful decision boundary across the whole sweep
(h11 fails, h12 passes); h12/h13/h14 are indistinguishable.

---

## 3. WORK ITEM 1 — Exact MILP source gate (highest yield)

### Why this is the right target

The gate is a *float* question about a network with **18 hidden ReLUs**. That is
exactly decidable by MILP in milliseconds-to-seconds. DeepPoly is being used
where an exact method is affordable. Replacing it recovers all 0.6–1.7 units of
abstraction loss and converts "we don't know" into either a proof or a genuine
counterexample.

### The machinery already exists

`GPEncoding._solve_margin_direction_milp(direction, *, hidden_layer_count=None)
-> tuple[float, float, float]` at `src/synthesis/preqbmc.py:3191` already
encodes:

- the real input box as continuous variables;
- each hidden affine layer exactly (`layer_paras[0]` is row-per-neuron,
  `layer_paras[1]` is bias);
- **exact ReLUs** — binary big-M with correct shortcuts for stable neurons
  (`hi <= 0` → fix 0; `lo >= 0` → identity; else binary indicator with the four
  standard constraints);
- DeepPoly pre-activation bounds **only as big-M constants** (sound: they are
  valid outer bounds);
- returns `(lower, upper, elapsed)` using the solver's **global objective
  bound** (`model.objective_bound()`, not the incumbent) with outward
  `math.nextafter` rounding.

It currently stops at the last *hidden* layer. `self.output_layer`
(`preqbmc.py:459`) is a separate `LayerEncoding` with the same
`layer_paras = [W.T, b]` row-per-neuron convention and **no ReLU**.

### What to build

Add a method, e.g. `check_source_region_exact(lb, ub)`, that:

1. Extends the MILP one more affine layer to the 43 output logits (no ReLU on
   the output).
2. For each competitor class `j != target`, minimizes the direction
   `e_target − e_j` over the region. This is a linear objective over the
   existing variables.
3. **Prunes using DeepPoly first.** Only classes where
   `ub[j] >= lb[target]` can possibly violate. Run DeepPoly (cheap, already
   implemented) and solve exact MILPs only for the surviving candidates. On this
   data most of the 42 competitors are far away, so expect a handful of solves.
4. Decides:
   - all minima `> 0` → source **VERIFIED** exactly;
   - any minimum `< 0` → **REFUTED**, with the MILP's argmin as a concrete
     adversarial input;
   - any solve hits timeout / non-OPTIMAL → **UNKNOWN** for that region.

### Soundness requirements — non-negotiable

- **Use `objective_bound()`, not the incumbent value**, and round outward with
  `nextafter`, exactly as the existing code does. The incumbent is a feasible
  point; the bound is what makes the conclusion valid.
- **A MILP solver is floating-point with tolerances.** This gate is exact *up to
  solver tolerance*, not bit-exact. Record it that way; never call it a
  bit-exact proof.
- **Validate every counterexample.** If a minimum is negative, evaluate the
  actual float network on the returned input point and confirm it really
  misclassifies. If it does not, the result is `UNKNOWN` (tolerance artifact),
  **not** `REFUTED`. Report the validated adversarial input.
- **Timeout must map to UNKNOWN, never to VERIFIED.** Add an explicit solver
  time limit.
- **Do not weaken anything downstream.** A region whose source property is now
  exactly VERIFIED still has to pass preimage deflation, ESBMC contracts,
  chaining and the input bridge unchanged. This item only removes a
  *false negative* at the gate.

### Record-keeping

`source_region_record` (`preqbmc.py:1089`) currently hardcodes
`"method": "deeppoly"`. Add `"milp_exact"` as a distinct value and keep the
DeepPoly bound alongside the exact one, so the paper can report *how much*
abstraction loss was recovered per region — that comparison is itself a result.
Keep the DeepPoly path available behind a config flag; do not delete it.

`source_region_summary()` is at `preqbmc.py:5601`; reports and tests that assume
`method == "deeppoly"` must be updated.

### Acceptance criteria

- A unit test on a tiny hand-built network where the exact robust margin is
  known in closed form, asserting the MILP gate returns it within tolerance and
  that DeepPoly returns something strictly looser.
- A test that a deliberately non-robust region yields `REFUTED` **with a
  validated counterexample** that the float network genuinely misclassifies.
- A test that solver timeout yields `UNKNOWN`, not `VERIFIED`.
- Re-run the 54 pairs and report the new breakdown. **Any region that was
  `VERIFIED` before must still be `VERIFIED`.** A flip in the other direction is
  a bug, not an improvement.

### Expected outcome (estimate, not a promise)

Plausibly 8–15 of the 44 currently-inconclusive pairs become VERIFIED. Most of
the low stratum (clean margin ~0.001) will likely become **genuine
counterexamples** — real adversarial images at ε=1/256. That is a legitimate and
publishable result; it is not a failure of the tool.

---

## 4. WORK ITEM 2 — Retrain the source model

41.7% float32 test accuracy, and the low-margin stratum sits at clean margin
~0.001. Robustness certificates on a model that is wrong most of the time
certify wrong answers.

This network is tiny (8×8×3 input, 2 conv filters) and trains in seconds.
Options in increasing order of effort: more epochs / better LR schedule; more
filters; larger input crop. **Any architecture change must keep the affine
lowering feasible** — `prepare_ssv_gtsrb.validate_config` enforces
`prod(input_shape) * prod(output_shape) <= max_affine_entries` because lowering
is quadratic in feature-map size. Raising input resolution blows this up fast;
check the budget before committing to a shape.

Larger clean margins mean more regions clear the gate *and* a more defensible
story. Note this invalidates all frozen cohorts and requires a full re-run from
`prepare`.

---

## 5. WORK ITEM 3 — Certified-radius search instead of a fixed ε ladder

Today: 3 fixed radii ε ∈ {1,2,4}/256, giving binary outcomes. ε=4 fails
universally; ε=1 nearly always succeeds where the gate passes. Most cells carry
little information.

Better: **binary-search the largest ε each image certifies at.** 54 binary
outcomes become 18 continuous measurements, the pipeline is exercised far
harder, and the result is a threshold curve rather than a mostly-inconclusive
table. Needs a bisection driver over the existing per-region runner, a
resolution floor (e.g. 1/256 granularity), and careful recording that the
reported radius is the largest *certified* one, not the true robustness radius
(the gap depends on gate precision — which Work Item 1 shrinks).

---

## 6. Context: the budget fix already landed (do not redo)

`_derived_error_budget_components_int` (`preqbmc.py:~2600`) previously computed
the weight-rounding term `dw` by summing input magnitudes over **all 192
inputs**, never consulting the weight matrix — charging every neuron for inputs
it does not touch. In the lowered convolution each neuron touches only 27 of
192, so the budget was inflated ~5–7× and produced spurious
`PREIMAGE_DEFLATION_EMPTY`.

Now `dw` is computed per neuron over the **real-weight support**. The soundness
argument: a stored real weight of exactly `0.0` quantizes to exactly `0` at
every scale, so it contributes zero rounding deviation. **The trap:**
`W_int == 0` does *not* imply zero error — a small nonzero real weight that
rounds to zero still carries the full `0.5/S_out`. The test is therefore on the
**real** weight, with the dense sum retained as fallback when real weights are
unavailable. Guarded by two tests in `src/tests/test_error_budget.py`.

Effect: `image6_eps1` went `PREIMAGE_DEFLATION_EMPTY` → `DEFLATED`; the sweep
now selects h12 for a real reason (image6 needs 3 integer bits) instead of
h11/h12 both spuriously deflating off an identical flat budget that never varied
with I.

**Implication for any pre-2026-09-19 result:** `PREIMAGE_DEFLATION_EMPTY` on a
convolutional region is not evidence and must be re-run.

---

## 7. Claim discipline — read before writing any paper text

The stated project goal is a QNN "invulnerable to adversarial attack." **The
tool does not and cannot establish that.** Be precise:

**What the code proves.** For *one* image, *one* L∞ radius, the *deployed
integer C program* has no input in that ball that changes the predicted class —
subject to ESBMC's soundness, the C semantics it assumes, and the derived error
budget being a valid over-approximation.

**What the code measures.** Runtimes, ESBMC call counts, peak memory, certified
margin bounds, which ⟨Q,I,F⟩ the search selects.

**What the code assumes.** That the compiler preserves C semantics (no
compiler-correctness proof); that the float lowering of the convolution matches
the Keras model (this is an *empirical float32 check*, explicitly labelled
`tested_numerical_equivalence_not_IEEE_proof` in the study manifest); that the
MILP solver's bounds are correct within tolerance.

**What is NOT established, and must never be claimed:**

- **Global robustness.** Certificates are per-region. 13 verified regions say
  nothing about the other images in GTSRB.
- **"Invulnerable to adversarial attack."** Only L∞ balls of radius ≤ 4/256 are
  covered. Nothing is proven about L2, patch, semantic, physical, or
  transformation attacks.
- **Attack-scale robustness.** ε ∈ {1,2,4}/256 is *sensor-noise scale*. Standard
  adversarial-robustness work uses 8/255 and upward. Do not present these radii
  as an attack threat model without saying so.
- **Anything about accuracy.** 41.7% clean accuracy is a separate, weak fact.
  Soundness and accuracy are independent claims; state both.
- **Deployment parity.** Android/embedded runtime, latency and power are
  currently `NOT_MEASURED` in every manifest. Do not report them as measured
  until real traces exist. Energy at this network scale will be dominated by
  idle power — the existing `summarize_energy_trials` gives a seeded bootstrap
  CI and explicitly excludes instrument calibration error.

**A conclusive negative is a legitimate result.** `REFUTED` with a validated
counterexample, `PREIMAGE_DEFLATION_EMPTY` with a slack decomposition, and
`SOURCE_PROPERTY_INCONCLUSIVE` with a cause are all publishable outcomes. Never
convert one of these into an unsupported `VERIFIED`. Never trade soundness for a
better-looking table; if a proposed change requires weakening a contract,
disabling chaining, introducing a heuristic tolerance, or counting inconclusive
results as proofs, reject it and say why.

**What IS genuinely strong and should be foregrounded:** the
calibration/evaluation split with a SHA-pinned frozen artifact selected *before*
any evaluation outcome was observed, with no per-region format re-search. That
discipline is stronger than most NN-verification papers and is a real
methodological contribution.

---

## 8. Minor known defects

- `src/scripts/run_ssv_gtsrb.py:11` hard-sets `os.environ["CUDA_VISIBLE_DEVICES"]
  = "0"`, overriding an externally set `-1`. Should respect the environment.
- Directory/run-ID naming still says `plate`/`cohort2` in places; the study is
  traffic *signs*. Names leak into result tables.
- `src/benchmark/plate/` holds **2.6 GB of GTSRB archives and is not
  gitignored** — it must not be committed. Add it to `.gitignore`.

## 9. How to run

```sh
conda activate quad          # provides the `preqbmc` console script
export MPLCONFIGDIR=/tmp/preqbmc-mpl TF_CPP_MIN_LOG_LEVEL=3

preqbmc gtsrb prepare-sweep --config experiments/sign_fixed_qif_sweep.json \
  --output output/sign_fixed_qif_sweep
CUDA_VISIBLE_DEVICES=-1 preqbmc gtsrb run \
  --study output/sign_fixed_qif_sweep/calibration/h11_o13_f8/study.json \
  --output output/sign_fixed_qif_sweep/runs/h11_o13_f8     # repeat per candidate
preqbmc gtsrb select-sweep --sweep output/sign_fixed_qif_sweep/sweep.json \
  --runs-root output/sign_fixed_qif_sweep/runs --output output/sign_fixed_qif_selection
CUDA_VISIBLE_DEVICES=-1 preqbmc gtsrb run \
  --study output/sign_fixed_qif_selection/evaluation/study.json \
  --output output/sign_fixed_qif_selection/evaluation_runs
```

Output directories must not already exist (this is deliberate — runs are never
silently overwritten). `output/` is gitignored. Tests:
`PYTHONPATH=src python -m unittest discover -s src/tests`. One pre-existing
failure, `test_iris_block_size_zero_and_one_select_consistent_qif`, is unrelated
to the work above.
