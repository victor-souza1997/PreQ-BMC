# PreQ-BMC — repository primer

Formal synthesis of quantized (fixed-point) neural networks from floating-point
models, using DeepPoly abstract interpretation + MILP backward preimage analysis +
ESBMC bounded model checking. Output is an embeddable C implementation whose
arithmetic has been formally verified against per-layer contracts.

Branch this primer describes: `feat/sbsec-new-methodology`.
~29.5k lines of Python across ~87 files, all under `src/`.
Packaged as `preqbmc` (Apache-2.0); see [CITATION.cff](CITATION.cff) for the
SBSeg 2026 Tool Track artifact reference.

## What the tool actually does

Input: a trained float32 feed-forward ReLU network + a local robustness property
(sample `x`, radius `eps`, target label or accepted label set) + a bit-width search
range.

Output: a per-layer fixed-point format `(Q, I, F)` (total/integer/fractional bits),
a generated C implementation using those formats, a **guarantee level** naming
exactly what was proved, and JSON/CSV reports.

Pipeline stages:

1. **Source region check** — establish the property on the *float* model first via
   DeepPoly. If that fails, the run stops at `SOURCE_PROPERTY_INCONCLUSIVE` rather
   than producing a meaningless downstream result.
2. **DeepPoly forward pass** — concrete per-layer interval bounds for the input box.
3. **Backward preimage** — for each layer, an admissible-output interval (a
   "contract") such that staying inside it preserves the robustness property.
   Solved exactly via MILP (CBC by default, Gurobi optionally), or approximately
   via interval abstraction.
4. **Bit-width search** — per layer, sweep fractional bits `F` in `[bit_lb, bit_ub]`,
   quantize weights/biases, emit a small C harness, and let ESBMC prove the
   candidate satisfies its contract. First verified candidate wins.
5. **Backends** — realize accepted formats as (a) a quantized Keras model,
   (b) a Python fixed-point interpreter, (c) generated C compiled to a `.so`.
6. **Quality refinement** — compare the three backends empirically (accuracy drop,
   prediction mismatch, saturation rate); if thresholds fail, reallocate bits and
   re-run the search, bounded by `max_quality_refinement_steps`.
7. **Reporting** — `pipeline_summary.json`, `experiment_summary.json`, CSV tables.

## Layout

```
src/
  synthesis/      preqbmc.py (6k lines, CORE), pipeline.py, solver_backend.py,
                  preimage_cache.py, forward.py
  scripts/        CLI entry points + experiment orchestration + aggregation
  utils/          encoding helpers, fixed-point utilities, dataset/ONNX plumbing
  verification/   ESBMC harness generation, subprocess runner, invariants, replay
  backends/       Python fixed-point engine + C code generator
  reports/        JSON/CSV summaries, guarantee levels, resource metrics
  symbolic_pp/    DeepPoly_preqbmc.py (interval propagation)
  models/         model loading (Keras .h5, ONNX, sklearn)
  datasets/       loaders.py — MNIST, Fashion-MNIST, MNIST-64, Iris, Seeds
  configs/        hardware_profiles.py — FixedPointProfile dataclass
  tests/          unittest/pytest suite (23 modules)
  shells/         shell wrappers
  cli.py          the packaged `preqbmc` console entry point
docs/             installation.md, REPRODUCIBILITY.md, expected_results.md
examples/         iris_demo/ + bundled preimage cache for the offline demo
experiments/      JSON experiment plans for the article/paper runs
prominent_results/  committed result tables for the reported experiments
```

### Files that matter most

- [src/synthesis/pipeline.py](src/synthesis/pipeline.py) — `RobustnessPipelineConfig`,
  `run_robustness_pipeline()`, `_run_quality_refinement()`, `compare_qnn_to_keras()`.
  This is the orchestrator; read it first.
- [src/synthesis/preqbmc.py](src/synthesis/preqbmc.py) — the `GPEncoding` class
  (line 287) and everything under it: the bit-width search loop, the MILP and
  interval preimage solvers, contract chaining, error budgets, vacuity and
  end-to-end summaries. Formerly `quadapter.py`.
- [src/synthesis/solver_backend.py](src/synthesis/solver_backend.py) — a
  `Protocol`-based abstraction with `GurobiBackend` and `CbcBackend`
  implementations, so the artifact runs license-free.
- [src/verification/c_templates.py](src/verification/c_templates.py) — all ESBMC
  harnesses, generated with plain f-strings (no templating engine). See harness
  families below.
- [src/verification/esbmc.py](src/verification/esbmc.py) — `ESBMCRunner`; wraps the
  `esbmc` binary as a subprocess, parses `VERIFIED / FAILED / TIMEOUT / MEMOUT / UNKNOWN`.
- [src/verification/invariants.py](src/verification/invariants.py) — exact integer
  box transformer over deployed layer semantics, injected into end-to-end harnesses.
- [src/verification/replay.py](src/verification/replay.py) — replays an ESBMC
  counterexample against both the Python engine and the compiled `.so`, so a
  reported `FAILED` can be confirmed rather than trusted.
- [src/verification/arith_kernel.py](src/verification/arith_kernel.py) — the single
  shared C fixed-point kernel (`div_round_half_away_from_zero_i128`,
  `clamp_to_signed_range_i128`, `mac_i128`) used by generated code and harnesses alike.
- [src/backends/c_qnn_generator.py](src/backends/c_qnn_generator.py) — emits the
  deployable C, compiles with `gcc -shared -fPIC -O2`, loads back via `ctypes.CDLL`.
- [src/backends/fixed_point.py](src/backends/fixed_point.py) — Python reference
  implementation of the same arithmetic (quantization, round-half-away-from-zero,
  clamping).
- [src/reports/experiment_summary.py](src/reports/experiment_summary.py) —
  `derive_guarantee_level()`; turns raw pipeline status into a transfer claim.
- [src/cli.py](src/cli.py) — the `preqbmc` argparse CLI (demo / reproduce /
  aggregate / verify-environment / install-esbmc).
- [src/scripts/run_robustness_pipeline.py](src/scripts/run_robustness_pipeline.py) —
  the lower-level argparse CLI (~60 flags).

## Harness families (verification/c_templates.py)

The renderers split into two genuinely different styles, and the distinction
matters more than the naming suggests. **Only 5 of the 14 renderers emit
`nondet_longlong()`**; the rest are straight-line interval computations that ESBMC
checks rather than searches.

**Symbolic** — nondet inputs bounded by `__ESBMC_assume`, property proved by search:

- `outerlayer_fixed_int`, `outerlayer_fixed_int_multiclass` — target class
  outscores all others.
- `render_network_end_to_end_program` — `--harness-scope network`; the whole
  forward pass in one query, optionally strengthened with the exact invariants
  from `invariants.py`.
- `render_assumption_sentinel_program` — proves the assumption box is non-empty,
  so a `VERIFIED` cannot be vacuous.
- `render_clamp_correctness_program`, `render_prefix_direction_cut_validation_program`.

**Static interval** — no nondet symbols; concrete `[low, high]` propagated through
integer weights in `__int128`, with `__ESBMC_assert` on the result:

- `innerlayer_fixed_int`, `innerlayer_fixed_int_bounds_only`,
  `render_hidden_affine_bounds_program`, `..._block_program` — the layer
  contracts, asserting the output lands inside its preimage interval subject to
  the error budget.
- `render_output_target_program`, `render_output_valid_set_program`.
- `render_no_saturation_program`, `..._block_program` — the pre-clamp accumulator
  never exceeds the signed range.

The layer-contract family used by the reported pipeline is in the **static** group.
That is sound (interval arithmetic over-approximates) and it is why per-layer
checks scale so well — there is no search. It is also the direct cause of the
`MARGIN_INCONCLUSIVE` outcomes: a box cannot express correlation between neurons,
so a violating hidden point need not be reachable from any single input, and the
check can neither verify nor refute. See [preqbmc.py:3592](src/synthesis/preqbmc.py#L3592).

Wide layers are split into neuron blocks (`--esbmc-layer-block-size N`) so no single
SMT query blows up. ESBMC is invoked with `--bitwuzla --bv` by default, or `--z3 --bv`
under `--esbmc-profile paper-z3` (needed for the larger MNIST nets). Built-in
bounds/div-by-zero/pointer checks are left enabled, so candidates get memory-safety
guarantees as a side effect.

## Fixed-point semantics (shared by Python and C)

`configs/hardware_profiles.py::FixedPointProfile` is the single source of truth:

```python
rounding_mode    = "half_away_from_zero"
overflow_mode    = "saturate"
accumulator_type = "__int128"
```

Generated C uses `static const int64_t` weight/bias arrays, a `buffer_a`/`buffer_b`
ping-pong forward pass, `__int128` MAC accumulation, and exposes
`qnn_forward_fixed(int64_t* in, int64_t* out)`. A `QNN_VERIFY_WITH_ESBMC`
compile-time macro swaps a no-op assertion for a real `__ESBMC_assert`, so the *same
file* is both the deployment artifact and its own verification artifact.

## Error budgets and contract chaining (important, easy to miss)

Hidden-layer contract checks are not necessarily exact containment. The policy is
selected by `--error-budget`:

- `heuristic` (flag default) — the legacy, empirically-tuned, **unproved** slack:
  `tau = floor(2^F / 1000) + floor(|p_U - p_L| / 100)`, i.e. an absolute term
  (~1e-3 in real units) plus 1% of the contract width. It *widens* the preimage,
  so a `VERIFIED` under it does not entail the property; the pipeline records
  `soundness: degraded` and can claim at most `harness-verified`. Kept for
  compatibility with the older cached results. Aliased by `--unsound-contract-tolerance`.
- `derived` — **the reported methodology.** A sound per-neuron budget computed
  from the layer's integer weights and assumption box, with the upstream budget
  inherited via `_inherited_error_budget_int`, the next layer's assumption set to
  the verified guarantee, and an output-margin check proving the accumulated error
  still preserves the classification margin. This is the mode that reaches
  `deployed-transfer`.
- `zero` — strict exact containment, no slack. Aliased by
  `--no-unsound-contract-tolerance`.

Layer contracts only compose if `G^[l] ⊆ A^[l+1]`. That condition is now
**checked and enforced by default** (`--enforce-contract-chaining`, on unless you
pass `--no-enforce-contract-chaining` for diagnostics), and
`--propagate-contract-tolerance` gives a sound way to carry a non-zero budget
downstream by widening the next layer's assumption to the verified widened contract.

For difficult derived contracts, `--tighten-verified-bounds` asks the same ESBMC
harness to prove an additional affine-box bound and propagates its intersection
only after every block passes. It preserves MILP preimages and one shared Q/I/F
per layer. See
[docs/verified_bound_tightening.md](docs/verified_bound_tightening.md) and the
matched Iris/Seeds pilot in
[experiments/iris_seeds_verified_bounds_pilot.json](experiments/iris_seeds_verified_bounds_pilot.json).

## Guarantee levels

Rather than a binary verified/failed, every run reports a transfer claim, derived
by `_guarantee_level` in
[src/reports/experiment_summary.py:317](src/reports/experiment_summary.py#L317):

| Level | Meaning |
| --- | --- |
| `deployed-transfer` | The property transfers to the deployed implementation. Requires *all* preconditions below. |
| `harness-verified` | Every harness discharged, but at least one transfer precondition is unmet (typically the heuristic error budget or chaining). |
| `unknown` | Inconclusive — timeout, memout, unresolved source property, or an unconfirmed counterexample. |
| `failed` | A refutation, a vacuous assumption box, or a rejected deployment-quality gate. |

Transfer preconditions: `contracts_verified`, `fidelity_by_construction`,
`chaining_ok`, `soundness_not_degraded`, `no_saturation_verified_if_required`,
`derived_margin_ok`, `vacuity_guard_passed`.
When verified-bound tightening is enabled, `verified_bound_tightening_ok` is
also required.

`preqbmc demo` exposes this as `--contract-profile`: `paper-slack` (default,
matches the bundled cache, can claim at most `harness-verified`) versus `strict`
(zero tolerance plus chaining). The bundled legacy cache is *expected* to be
rejected under `strict` — that is a formal result, not a bug. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Known gaps / rough edges

Treat these as real, not as things to paper over:

- **`heuristic` is the flag default but not the reported methodology.** The article
  configs ([experiments/preqbmc_reported_experiments.json](experiments/preqbmc_reported_experiments.json),
  [experiments/all_datasets_source_aware.json](experiments/all_datasets_source_aware.json))
  set `error_budget_mode: derived` and reach `deployed-transfer`. Anyone invoking
  the CLI without `--error-budget derived` silently gets the degraded slack instead,
  which is an easy footgun; consider flipping the default.
- **Unresolved runs dominate the harder benchmarks.** In
  [output/preqbmc_reported_results/](output/preqbmc_reported_results/): 82
  `deployed-transfer`, 74 `unknown`, zero `failed`. MNIST is 22/64 certified; Seeds
  is 0/12. The `unknown` bucket is resource limits and source-inconclusive cases,
  not refutations — but it is the honest denominator.
- **Symbolic harnesses do not scale past a handful of neurons.** Measured on this
  machine (bitwuzla `--bv`, 12 GB): a single symbolic layer verifies at n=8 in ~12 s
  and memouts at n=10. The static layer-contract harnesses are unaffected (n=20 in
  ~1 s) because they do not search. Every `TIMEOUT` in the reported set comes from
  the symbolic side.
- **Do not "fix" the int64 accumulator encoding.** When `invariants.py` proves a
  64-bit accumulator suffices, the emitted code still evaluates each MAC through
  `mac_i128` and truncates. That looks like a missed optimisation; rewriting it to
  native `int64_t` arithmetic was measured at a **median 0.22x (≈4.5x slower)** over
  six seeds and turned three of six ~5 s verifications into 10 GB memouts. Pinned by
  `NarrowAccumulatorEncodingTest` in [src/tests/test_e2e_invariants.py](src/tests/test_e2e_invariants.py).
- **Gurobi is the bottleneck** on larger nets; hence CBC as the default and the
  `.npz` preimage cache (`--save-preimage-cache` / `--no-gurobi`).
- [CITATION.cff](CITATION.cff) still carries `TODO` author names and DOI.
- [experiments/README.md](experiments/README.md) still documents the pre-rename
  `python tool/scripts/...` commands.
- Legacy underscore flag aliases (`--bit_lb`) coexist with dashed ones (`--bit-lb`),
  and `Preqbmc_*_main.py` at the `src/` root are thin wrappers around `scripts/`.

## Running it

Install (see [docs/installation.md](docs/installation.md)):

```bash
pip install -e '.[paper]'     # or '.[cbc]' for a minimal license-free setup
preqbmc verify-environment    # add --install-missing-esbmc to fetch ESBMC
```

The cached Iris demo needs no solver license and is the recommended smoke test:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

Reproduce the article experiments:

```bash
preqbmc reproduce --config experiments/article_experiments.json --only iris --max-runs 1
preqbmc aggregate --input-root output/article_runs --output-root output/article_results --plots
```

The lower-level pipeline entry point, for one-off configurations:

```bash
python src/scripts/run_robustness_pipeline.py \
    --dataset mnist --arch 1blk_100 --sample-id 5346 \
    --eps 2 --bit-lb 1 --bit-ub 16 \
    --verify-mode esbmc --output-dir ./output/
```

Requires `esbmc` on PATH and `gcc`. CBC is the default MILP solver; `--solver gurobi`
needs a license. Flag groups: target/problem spec, bit-width range, preimage/solver,
ESBMC options, error budget/chaining, quality thresholds, output/reporting, general.

Tests: `pytest src/tests/` for the full suite (block-wise verification, C generation,
the Python fixed-point forward pass, model loading, no-saturation harnesses, error
budgets, guarantee levels, solver parity). CI
([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs only the five
solver-free modules via `python -m unittest`.

## Conventions worth preserving

- Python and generated C must stay bit-identical. Any change to rounding, clamping,
  or accumulation must land in [src/backends/fixed_point.py](src/backends/fixed_point.py),
  [src/backends/c_qnn_generator.py](src/backends/c_qnn_generator.py),
  [src/verification/arith_kernel.py](src/verification/arith_kernel.py), and the
  harnesses in [src/verification/c_templates.py](src/verification/c_templates.py)
  together, or the whole soundness story breaks.
- Harnesses are plain f-strings on purpose — keep them dependency-free and readable
  as C.
- Nothing that changes numerics should be added without a corresponding test in
  [src/tests/](src/tests/).
- Anything that weakens a proof obligation must be reflected in the guarantee level,
  not hidden behind a `VERIFIED`.
