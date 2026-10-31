# PreQ-BMC: bounded diagnostics for unresolved CROWN margins

Specification date: 2026-10-31. Status: **implementation-ready proposal; no diagnostics implemented or executed by this specification**. This is development work, not a certification-rate evaluation or a replacement for P3. P3 remains a completed negative experiment.

## 1. Scope, checkout, and observed baseline

The user explicitly selected Claude's worktree: `/home/joao/code/preqbmc-qformat`, branch `feat/crown-v2-qformat`, inspected HEAD `23e4fcf358c0502a001192b70a26ed9393a82d96`, **including its uncommitted P1/P3 changes**. Do not switch branches, reset, commit, or cherry-pick. The initially named `origin/feat/cnn-feature` exists at `fb69215e2127ec6579530b1c479652ee4f852401`; its relevant committed sources match the base of Claude's branch, but it lacks the uncommitted corrections. All source paths below are relative to Claude's worktree.

Run with working directory `/home/joao/code/preqbmc` because the existing loader uses relative `output/` paths. Set `PYTHONPATH=/home/joao/code/preqbmc-qformat/src`. Record the resolved `__file__` and SHA-256 of every project module imported: **loading main-checkout modules accidentally is a failed preflight**. Do not write source files in the main checkout. Output must go to a fresh directory on the mounted output disk; never overwrite October artifacts.

Inspected source identities (capture all imported sources again at implementation freeze):

| File | SHA-256 before diagnostic implementation |
|---|---|
| `src/scripts/crown_finder/crown_chain_export_v2.py` | `1541896e43a01ada3d688d311f8c3db676b83ba371a88920cc5d3f2db94a9146` |
| `src/scripts/crown_finder/crown_chain_export.py` | `84e61a573f180e39a6e537caf3232405d77e2f48fac15eae178ffd6df4342c36` |
| `src/scripts/crown_finder/affine_export.py` | `7657897bac0420885f6825f1e15179ce498a77134e4c13065abe8a4ef4fdbfc0` |
| `src/verification/crown_chain_v2.py` | `c6628c449d48f3968cbb7b0ce2a0495b101ef793903f02edffe9a2ca284f76ac` |
| `src/verification/crown_chain_v2_coverage.py` | `768168d8cbe10eb749e7ae6ccd55ccd3d3fbfcff0ecb2d6eb614feb51a562dbb` |

The deployed source must remain SHA-256 `dd535f5480bec45cdae7774f849afca49f7afb5d645f02037e86a44d16f45097`. Deployment Q7.8, coefficient scale `S=16`, five affine layers, and 43 classes remain fixed, except the explicitly diagnostic coefficient-scale experiment I-C.

### Observations, not hypotheses

The following values were read from `output/p3_finder_pilot_20261031/{013,016,017}.json`. Ordinals are draw positions, **not validation dataset indices**; always select by full record ID.

| Ordinal | Record ID | Target | Focus competitors | Primary competitor | Closed | Minimum raw-margin bound | Planned chains | Multiply terms |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 013 | `Final_Training/Images/00038/00039_00015.ppm` | 38 | 34,35,36,37,40 | 40 | 37/42 | -997 | 6,770 | 331,893,368 |
| 016 | `Final_Training/Images/00002/00023_00020.ppm` | 2 | 0,1,5,21,31 | 5 | 37/42 | -876 | 7,851 | 365,356,538 |
| 017 | `Final_Training/Images/00039/00007_00017.ppm` | 39 | 37 (positive control) | 37 | 42/42 | 352 | 3,240 | 157,246,375 |

All ten unresolved competitors were reported as `margin < 1`, not merely clamp-closure failures. Still log every clamp condition separately. Image 017 was reaudited as `ACCEPTED_WITH_SUPPLEMENT` in `output/p2_reaudit_20261030/validation_017.json`; preserve the historical certificate plus supplementary evidence. P3 modified no positive-control margins because it searched only open margins. Here the control **does receive the specified probes**.

Do not claim any of these observations establish the bottleneck. In particular, fixed per-layer slope failure does not establish that intermediate intervals are the cause.

## 2. Exact semantics and evidence categories

Use these names consistently:

- `b`: uint8 RGB bytes; domain membership is tested before encoding.
- `x`: encoded input, produced by `qnn_encode_bytes` / `ByteImageEncoder.encode`.
- `s_j = W_j h_(j-1)`: accumulator **without bias**.
- `r_j = RHA(s_j/256) + bias_j`: raw pre-clamp affine output.
- `z_j = clamp(r_j,-32768,32767)`.
- `h_j = max(z_j,0)` at hidden layers; final deployed logits are `z_4`, with no ReLU.
- `e_j = 256*RHA(s_j/256)-s_j`: scaled rounding error, an integer in [-128,128].

Bias is added **after** the division/rounding. Do not move it into the accumulator. Use the existing C arithmetic kernel and its negative half-tie behavior. Compute differences in signed 64-bit or wider types, never uint8/int16 subtraction.

For fixed target `t` and competitor `c`, the deployed property is `z_4[t]-z_4[c] >= 1` for every allowed input. A value <=0, including a tie, is a violation regardless of `argmax` tie-breaking. Target stays fixed throughout search. Also log `r_4[t]-r_4[c]`; the current chain lower bound concerns this **raw** margin. Do not describe that bound as a numeric lower bound on the clamped margin. The existing separate clamp conditions establish strictness, not equality of the two margins.

### Input-domain distinction

`affine_export.image_region()` constructs a Cartesian box of resized byte coordinates through `ByteImageEncoder.byte_box()`, then encodes its endpoints. Reconstruct exactly that domain, called `D_box`. With F=8, the encoder's scalar byte mapping is identity, but still execute and check the encoder. Derive its finite image explicitly; do not generalize identity to other formats.

Also reconstruct `D_crop`: perturb the original decoded crop by at most one byte and then apply center-nearest resizing. Build the exact output-coordinate to source-coordinate map from `ByteImageEncoder.resize()`. Repeated source coordinates must share a byte in `D_crop`. `D_crop` maps into `D_box`, but the inclusion can be strict. Log the number and sizes of alias groups. Decoding/RGB conversion remain outside the arithmetic theorem, as in the existing encoder.

Primary attack search operates on **source alias groups**, so every candidate has an original-crop lift. Save the original crop with changed source bytes and replay the pinned C encoder. Separately evaluate the unconstrained `D_box` affine minimizer. If it cannot be lifted, classify it `BOX_ONLY_WITNESS`; it refutes robustness over `D_box` when exact execution violates dominance, but does not establish an original-crop violation. Never mix these counts. At epsilon zero the domains coincide at the nominal encoded input.

### Evidence labels (orthogonal to solver status)

Every result must have `evidence_kind` from:

- `MEASURED_EXACT_EXECUTION`: concrete execution of pinned C, with input/domain identity.
- `SCREEN_BOUND_UNCHECKED`: the current mathematical finder computation, not ESBMC evidence.
- `CONDITIONAL_ESBMC_LEMMA`: an ESBMC-verified local assertion whose input-bound dependencies may remain unverified.
- `ESBMC_CHECKED_OBLIGATION`: the identified assertion was checked; this is not automatically a complete-network certificate.
- `COUNTERFACTUAL_ONLY`: intentionally supplied hypothetical bounds, never eligible for export.
- `UNSUPPORTED_CERTIFICATE_FORM`: mathematically motivated diagnostic result not representable by the production checker.

Record `solver_status` separately (`NOT_RUN`, `VERIFIED`, `FAILED`, `TIMEOUT`, `MEMOUT`, `UNKNOWN`), the failed assertion ID/type, and `dependency_status`. Failure of an unwinding assertion, safety assertion, or certificate arithmetic assertion is not a network counterexample. Never promote evidence by changing its label after a successful numerical result.

## 3. Implementation surface and required artifacts

Add one diagnostic driver, proposed path `src/scripts/diagnose_crown_finder.py`, and a diagnostic-only support module `src/verification/crown_diagnostics.py`. These paths and the CLI in section 12 are **interfaces to implement, not existing commands**. Keep production behavior unchanged when diagnostic options are absent. No adaptive search controller, training, new network, or evaluation cohort is part of this task.

| Existing file/function | Required diagnostic use or minimal hook |
|---|---|
| `crown_finder/affine_export.py`: `load`, `image_region`, `lowered` | Load the pinned artifact; expose byte/crop identity; use identical CSR geometry. No new format selection. |
| `crown_finder/crown_chain_export.py`: `setup`, `choose_relu`, `relu_const`, `XU`, `ENC`, `S` | Reuse baseline choices; explicit per-region state save/reset. Never let a singleton or child upper box leak into the next run. |
| `crown_finder/crown_chain_export_v2.py`: `run_chain` | Observer for immutable step records, including raw constants and coefficients; diagnostic-only single-step overrides described below. |
| Same: `bounds_v2` | Return all candidate bounds and selected sources; optional explicit replacement of one side, with provenance. Do not silently intersect unproved sample extrema. |
| Same: `relu_choice`, `_relu_choice_adaptive`, `classify`, `clipped_const` | Record choices/classes and implement I-S coordinate overrides. |
| Same: `needs`, `main` dependency planning, nested `emit` | Reuse exactly the same immutable overrides for calculation, dependency collection, and serialization; support separate policies for bound chains in I-B. Extract helpers only if needed; no behavior change. |
| `crown_finder/box_ledger.py`: `facts_from_box`, `facts_from_ledger` | Keep screening facts distinct from VERIFIED ledger facts. Never fabricate a ledger. |
| `backends/image_encoder.py`: `ByteImageEncoder.resize/encode/byte_box/render_c` | Encoder parity and alias map; byte-domain membership. |
| `datasets/gtsrb_study.py`: `load_crop` | Original crop/ROI and dataset identity. |
| `backends/conv_fixed_point.py`: `generate_conv_qnn_source`, `_finish_value`, `forward_conv_fixed_point_single` | Unmodified C oracle; optional observer-only trace rendering for accumulator/raw/clamp/ReLU. Default generated C must retain the deployment hash. Existing Python traces contain post-activation values only. |
| `verification/arith_kernel.py`: `render_arith_kernel` | Reuse verbatim; no arithmetic change. |
| `verification/conv_interval_fast.py`: `propagate_box`, `_finish` | Baseline/singleton boxes. Its `pre_low/pre_high` are **after clamp, before ReLU**, not raw `r`. |
| `verification/conv_contracts.py`: `conv_output_terms`, `dense_output_terms`, `IntegerInvariant` | Local CSR witnesses/obligations and explicit predecessor domains. |
| `verification/conv_proof.py`: `render_dense_relational_cut_validation`, `render_byte_encoding_contract` | Reference for two-neuron local harness and child encoding obligations. Existing relational helper is dense-only and is not integrated with CROWN v2. |
| `verification/conv_cegar.py`: `SparseRelationalCut` | Optional record representation only. **Do not instantiate `DemandDrivenMarginCEGAR`.** |
| `verification/crown_chain_v2.py`: `validate_chain`, `_validate_facts`, `validate_margin_closure`, `_render_affine`, `_render_relu`, `render_rounding_lemma`, `render_v2_harnesses` | Inspect representability and prospective checking costs; retain production validation. Do not teach the production checker new proof rules in this diagnostic task. |
| `verification/crown_chain_v2_coverage.py`: `required_assertions`, `verify_shard_coverage` | Independent assertion inventory, including zero-cancelled CSR columns. |
| `verification/esbmc.py`: `ESBMCConfig`, `ESBMCRunner.run_file/build_command/infer_unwind` | Run bounded diagnostic obligations; save full command and assertion identity. |

All finder file names in the table have prefix `src/scripts/`; all remaining source names have prefix `src/`.

Create these outputs under a new `output/crown_diagnostics_<UTC timestamp>/`:

| Artifact | Contents |
|---|---|
| `manifest.json` | Source/data/deployment hashes, dirty diff hash, config, Python/compiler/ESBMC versions, module paths, machine limits, all seeds, start/end times. |
| `images.json` | Record IDs, aliases, byte/crop/encoded hashes, targets, expected baselines, historical provenance. |
| `runs.jsonl` | One common run envelope for every baseline, probe, skipped probe, replay, and ESBMC call. |
| `margins.csv` | Every evaluated output margin, raw bound/actual values and clamp conditions. |
| `steps.jsonl.gz`, `coordinates.jsonl.gz` | Step accounting and per-neuron arithmetic; no decimal-float-only exact values. |
| `bounds.jsonl.gz` | Bound alternatives, chosen sides, provenance, widths, stability, changes. |
| `witnesses/` | Raw uint8 crop/tensor `.npy`, exact encoded arrays, logits, trace, SHA-256 and replay metadata. `allow_pickle=False`. |
| `interventions.jsonl` | Deterministically selected candidates, parameters, support/failure, delta, representation status. |
| `obligations/` | Generated standalone diagnostic C, stdout/stderr, status and exact dependencies. |
| `prospective/` | Explicitly untrusted serialization/counting artifacts for finalists; **not named `certificate.json`**. |
| `costs.csv`, `decisions.csv`, `report.md` | Resources, exact/estimated/missing metric distinctions, outcome-to-next-action mapping. |

### Common JSON envelope

Required fields for every record in `runs.jsonl` (null values require a `missing_reason` entry):

```json
{
  "schema": "preqbmc_crown_diagnostic_v1",
  "run_id": "013/I-S/c40/L2/n123/ap0",
  "parent_run_id": "013/baseline/eps1",
  "diagnostic": "I-S",
  "ordinal": "013",
  "image_id": "Final_Training/Images/00038/00039_00015.ppm",
  "target": 38,
  "competitor": 40,
  "epsilon_raw_bytes": 1,
  "domain": "D_box",
  "region_sha256": "<canonical byte/encoded bounds and alias-constraint hash>",
  "assumption_context_sha256": "<hypotheses plus dependency identities>",
  "deployment_sha256": "<pinned hash>",
  "source_manifest_sha256": "<manifest hash>",
  "scale_bits": 16,
  "status": "COMPLETED",
  "evidence_kind": "SCREEN_BOUND_UNCHECKED",
  "solver_status": "NOT_RUN",
  "dependency_status": "SCREEN_ONLY",
  "parameters": {},
  "metrics": {
    "wall_seconds": 0.0,
    "cpu_seconds": null,
    "peak_process_tree_rss_bytes": null,
    "peak_cgroup_memory_bytes": null,
    "solver_elapsed_sum_seconds": 0.0,
    "serialized_bytes": null,
    "compressed_bytes": null,
    "planned_chains": null,
    "planned_multiply_terms": null,
    "planned_assertions": null,
    "planned_harnesses": null,
    "additional_assertions": null,
    "additional_harnesses": null,
    "executed_solver_calls": 0
  },
  "metric_scope": "this_probe",
  "missing_reason": {},
  "artifacts": {},
  "interpretation": {
    "hypothesis_id": "H-S",
    "assessment": "SUPPORTS|NO_SUPPORT_WITHIN_PROBE|INCONCLUSIVE|IMPLEMENTATION_ERROR",
    "reason_code": "<specified below>"
  }
}
```

Use decimal **strings** for exact coefficients, accumulators, scaled bounds, constants, sums/products and rational numerators/denominators; ordinary JSON integers suffice for indices/counts and bounded logits. A rational is `{"num":"...","den":"..."}` in reduced form, positive denominator. Floating-point approximations are additional display fields only. CSV uses decimal strings too. `status` also admits `SKIPPED_NO_CANDIDATE`, `UNSUPPORTED`, `NOT_RUN_BUDGET`, `DEPENDENCY_FAILED`, `OVERFLOW_GUARD`, `CANCELLED`, `INVALID_DOMAIN`, `PARITY_FAILURE`.

`margins.csv` columns: `run_id,ordinal,domain,epsilon,target,competitor,scale_bits,scaled_raw_lb,raw_lb,nominal_raw_margin,nominal_deployed_margin,best_seen_raw_margin,best_seen_deployed_margin,witness_id,top_rule,top_reference,top_holds,bottom_rule,bottom_reference,bottom_holds,closed,baseline_raw_lb,delta_raw_lb,material_gain,evidence_kind,dependency_status`.

`costs.csv`: `run_id,metric_scope,wall_seconds,cpu_seconds,peak_process_tree_rss_bytes,peak_cgroup_memory_bytes,solver_elapsed_sum_seconds,serialized_bytes,compressed_bytes,planned_chains,planned_multiply_terms,planned_assertions,planned_harnesses,additional_assertions,additional_harnesses,executed_solver_calls,cost_status,missing_reason`.

`interventions.jsonl` adds `candidate_rank`, `selection_score` (exact rational), `selection_trace_hash`, `changed_fields`, `frozen_fields_hash`, `before`, `after`, `raw_scaled_delta`, `raw_integer_delta`, `material_gain`, `closure_changed`, `certificate_compatibility`, and `conditional_dependencies`.

Define material gain before observing new results: an integer raw-bound gain >= `max(1,ceil(0.05*max(1,1-baseline_raw_lb)))`, or a newly closed competitor with all clamp conditions. Log smaller gains; they support numerical sensitivity but not the material-gain criterion. For a control margin, the threshold is one unit. This threshold is an experimental decision, not a theorem or an acceptance-rate objective.

## 4. D-P: preflight and exact execution oracle

**Question/hypothesis H-P:** Are the nominal domain, implementation and baseline computations the same ones that produced P3? All must agree before attributing a difference to precision.

1. Validate artifact identities against `experiments/ssv2026_validation_protocol_20261014.json` without launching that expired protocol. Hash crops and record IDs. Load the selected model only; do not reconstruct a similarly named model.
2. Compile the pinned generated C as `-O2` and `-O0` shared libraries using the existing compile helper `backends/c_qnn_generator.py:compile_c_qnn_shared_library`. Record compiler path/version and full arguments. Never use `-ffast-math`.
3. Expose existing `qnn_encode_bytes` and `qnn_forward_fixed`. Execute nominal crop and box low/high. Confirm encoder parity, all 43 logits, and Python reference parity. For these nine inputs additionally obtain all-layer traces.
4. Add optional observer output in `generate_conv_qnn_source` only for diagnostic C: for each neuron save `s,r,z,h` at their actual locations in the arithmetic. Preserve unobserved generated source byte-for-byte. Observer C must agree with unmodified C logits on every retained witness. The Python arbitrary-integer replay computes the same raw intermediates from CSR; compare all coordinates, not just final logits, on nominal inputs.
5. Rerun baseline `screen 1 <full ID> --cost`, **without `--search`**, and check all expected minima/closed sets/counts above. Compare all 42 margins with saved raw roots if available; record unavailable values, never invent them. In the diagnostic in-process baseline pass also persist immutable numeric bounds, selected fact sources, every focus root's complete step records/final input coefficients, all root scaled bounds, and closure edges. D-V can then use the affine corner before D-T performs witness accounting. These cached baseline arrays must remain available after D-0 restores epsilon-one state.
6. Reject any nominal output tie as a strict-property violation at the nominal point. An `argmax`-correct nominal is not sufficient. A violation or bound inconsistency in 017 halts all experiments pending an identity/arithmetic audit.

**Log/schema:** common envelope; `oracle.json` with source/compiler/library hashes, byte/encoded hashes, shapes, all raw and deployed logits, `O0_O2_mismatch_count`, `C_Python_mismatch_count`, `trace_mismatch_count`, `alias_groups`, expected/actual baseline values. Costs include compile and setup separately.

**Cost:** expected minutes; hard stage cap 900 s, 8 GiB process group. No ESBMC call. A mismatch rejects H-P and prevents downstream interpretation. Existing certificates need no change; these are measurements, not new proofs.

**Soundness risks:** int64 overflow in diagnostic vectorization; confusion of post-clamp and raw arrays; stale global `XU`; mixed source trees; truncated C traces. Use Python integers for diagnostic checks and abort on unsupported native-width arithmetic rather than wrap.

## 5. D-V: concrete violation search

**Question:** Are any unresolved properties false under deployed integer semantics? **H-V:** at least one focus margin admits a concrete byte-valued violation. This is an existence hypothesis: finding one verifies the observed violation; failing to find one does not reject robustness or establish it.

**Hooks:** oracle from D-P; `ByteImageEncoder.resize/byte_box/encode`; `affine_export.image_region`; final coefficients from `run_chain(record=True)`. New search lives only in the diagnostic driver/support module. Do not call the old `check_mode()` as the oracle: it samples encoded corners and its intermediate `pre_*` trace is clamped.

### Fixed search, no optimizer choice left to the implementation

Run focus competitors in numeric order, image order 013,016,017. For each focus competitor:

1. Evaluate nominal, complete low/high byte corners, and the final affine-form minimizing corner (`low` for a positive coefficient, `high` for negative, nominal for zero). All C calls also check every other competitor. Deduplicate by byte/domain hash.
2. Primary search variables are source alias groups from D-P. Give each group the allowed source-byte values in `[max(0,b0-1),min(255,b0+1)]`; all its resized copies move together. Unused crop pixels remain nominal. The linear corner for this domain uses the **sum of final coefficients in the group**. Log the independent box corner separately and test whether it has a crop lift.
3. Four starts: nominal; source-group linear corner; two uniform random assignments over each group's allowed values. Use `numpy.random.Generator(PCG64(20261031 + 1000*ordinal_int + 10*competitor + restart))`, with restart 0..3. Persist NumPy version and seed. No float-model gradient/attack is part of this first diagnostic.
4. Each start has at most 256 proposals including its starting point. The first 127 subsequent proposals are coordinate-descent candidates: rank movable groups by `abs(sum input coefficient)*(high-low)`, descending, then source-flat index; take at most 32 groups. Visit in that order and test alternative allowed byte values in ascending order, cycling if necessary. Build each candidate from the current incumbent.
5. The remaining proposals up to 256 are random block moves. Alternate block sizes 8 and 32, clipped to the number of movable groups. Sample groups uniformly without replacement and each replacement uniformly among its other allowed values. This can select groups outside the first 32. If no movable groups, record `SKIPPED_NO_CANDIDATE`.
6. Accept a candidate only when the pair `(deployed_margin,raw_margin)` decreases lexicographically. Equal pairs retain the incumbent. Search does not use class label or float loss as the acceptance criterion. Cache hits count as proposals but not as new C evaluations.
7. Stop that focus competitor on the first deployed margin <=0; retain the other focus searches. If any checked margin in control 017 violates dominance, stop the entire run as an audit failure. For 013/016, discovered violations of non-focus competitors are logged and replayed too.

Limit 1,024 proposals per focus competitor plus four initial probes; maximum about 11,308 proposals before cache reuse, with a stage wall cap of 2,400 s. Keep a count of incomplete starts and unexecuted proposals; never label a truncated search exhausted.

### Concrete replay and output

Save all improvements' byte hashes/objectives; retain full arrays for nominal, low/high corners, each focus margin's best witness, and every violating witness. Normal retained trace set is at most 8,8,4 witnesses for the three images before deduplication. Replay violations with **unmodified** C at both -O0 and -O2 and the Python integer oracle. Verify each original source byte, epsilon, shape, alias constraint, encoder output and target. Persist `changed_byte_count`, `linf_raw`, all 43 logits, raw/deployed margin, tie flag and crop-lift status.

For each distinct violating witness, run one fixed-input ESBMC replay assertion `output[t] > output[c]` with exact bytes and pinned C. Identify failure of that specific assertion; a timeout does not erase measured C evidence but prevents an ESBMC-confirmed label. Maximum three ESBMC witness replays (first violation per image; 017 would trigger the audit stop). Extra violations remain explicitly measured.

`witness.json` fields: common identity fields plus `search_seed,restart,proposal_index,search_objective,bytes_path,bytes_sha256,crop_path,crop_sha256,encoded_path,encoded_sha256,byte_bounds_satisfied,source_bounds_satisfied,alias_constraints_satisfied,crop_lift_exists,raw_logits,deployed_logits,raw_margin,deployed_margin,is_tie,O0_replay,O2_replay,python_replay,esbmc_replay,violated_assertion_id`.

**Interpretation:** a lifted, replayed violation supports H-V and explains why no sound finder can close that property. A box-only violation diagnoses the enlarged box if the intended application is original-crop robustness. Neither is a checker failure by itself. Search exhaustion is `NO_WITNESS_FOUND`, not `ROBUST`. A hidden-state assignment from any later diagnostic is never accepted here without a complete byte witness and replay.

**Certificate compatibility:** counterexamples are separate evidence; current positive CROWN certificates do not encode them. Fixed-input ESBMC needs no CROWN rule change. **Risks:** accepting raw-logit violations when clamped logits do not violate; ignoring ties; silently changing target; approximate/surrogate inference; source-resize alias inconsistency; treating failed safety checks as the target violation.

## 6. D-0: epsilon-zero precision diagnostic

**Question:** How much conservatism persists without input uncertainty, and which effect changes when only bounds are refreshed? **H-0:** the normal chain retains a material gap at a singleton. This is not, on its own, evidence of a soundness bug.

Run all 42 competitors on each image, using identical nominal C execution as reference, in these three arms:

| Arm | Input bounds | Intermediate bounds | Meaning |
|---|---|---|---|
| Z-inherited | nominal singleton | exact saved baseline epsilon-one bounds | Collapse only input uncertainty, retain old intermediate information. |
| Z-fresh | nominal singleton | freshly run `propagate_box`, `facts_from_box`, `bounds_v2` at epsilon zero | Normal finder on the genuinely new singleton region. |
| Z-exact | nominal singleton | exact raw `r_j` singleton bounds from C trace, for every hidden neuron | Diagnostic upper-information control; actual values supplied only because domain is one point. |

Never obtain raw singleton bounds from `BoxTrace.pre_low`: those values are clamped. In Z-exact recompute saturation/clipped/stable states from raw values; if the production class validator cannot represent a value, record `UNSUPPORTED_CERTIFICATE_FORM`. An explicit singleton arithmetic replay can still be measured. No bypass of the production validator is allowed. Z-exact is not an export-ready bound source.

**Log:** `scaled_raw_lb`, `raw_lb=ceil(scaled/65536)`, exact nominal raw and clamped margins, `raw_gap=nominal_raw_margin-raw_lb`, exact scaled gap, `gap_fraction=raw_gap/max(1,abs(nominal_raw_margin))`, all clamp flags, all-layer bound widths and sources. Define a significant singleton gap as `raw_gap >= max(1,ceil(0.05*abs(nominal_raw_margin)))`. Record both this criterion and the continuous values. Negative raw gap means a bound exceeds concrete execution: stop and audit.

`singleton.jsonl` adds `arm,nominal_witness_id,raw_gap,raw_gap_scaled,gap_fraction,significant_gap,bound_widths_by_layer,source_substitutions,production_representable` to the common envelope. All 126 margin rows per arm are written to `margins.csv` (three images times 42), including control margins.

**Interpretation:**

- Z-inherited to Z-fresh improvement: sensitivity to recomputed intermediate information, with consequent slope changes; not an isolated estimate of interval loss.
- Z-fresh to Z-exact improvement: the normal bound construction/source selection still loses information at a singleton.
- Gap remaining in Z-exact: inspect affine rounding/residual/constant-floor accounting, coefficient choices, or unsupported classes. Actual input uncertainty cannot explain it.
- All gaps zero: rules out significant **singleton** conservatism only; says nothing about precision over epsilon one.
- Positive nominal deployed margin with negative singleton finder bound: legitimate incompleteness unless a parity/inequality check fails.

**Hooks:** `setup`, `XU`, `bounds_v2`, `run_chain`, `classify`, `propagate_box`, diagnostic oracle. Restore epsilon-one state by explicit reload after the stage. **Cost:** three arms per image, expected minutes; 900 s cap, no ESBMC by default. **Compatibility:** Z-fresh fits current arithmetic rules but requires a new singleton ledger/export to be certified. Z-inherited cannot pretend its parent ledger is the singleton's exact ledger; Z-exact requires a new bound source if promoted. No certificate is emitted here.

## 7. D-T: precision accounting along concrete traces

**Question:** Which inequalities are loose on actual inputs, and which bounds/choices influence the open margins? **H-T:** one or more identifiable steps show concentration of slack. A large slack at one witness is a localization signal, not proof that the step controls the true minimum.

Instrument all 11 focus chains at epsilon one; for each image evaluate them at every retained witness from D-V. Also record all 42 baseline bounds and closure edges. Repeat accounting for the primary margin of each D-0 arm. Use `run_chain(record=True)`; add observer records without changing calculations.

### Exact accounting identity

For a fixed chain and exact execution, let `a` be incoming coefficients of an affine step, `a'=floor(W^T a/256)`, `R=W^T a-256a'`, `P=128*sum(abs(a))`, `hlow` the actual lower endpoints used, `B=R·hlow-P`, `q=floor(B/256)`, `rho=B-256q` in [0,255]. With actual predecessor activation `h` (encoded `x` at layer 0):

`affine_slack = a·r - a'·h - a·bias - q`

`256*affine_slack = R·(h-hlow) + (a·e + P) + rho`.

Log all three numerator terms separately as:

1. `residual_endpoint_slack_num = R·(h-hlow)`;
2. `rounding_envelope_slack_num = a·e+P`;
3. `constant_floor_slack_num = rho`.

Each must be nonnegative. Use exact Python integers; their sum must be divisible by 256 and equal the directly computed affine slack. At a ReLU/clamp step:

`relu_slack_i = a_i*h_i - a'_i*r_i - d_i >= 0`.

At input concretization:

`input_slack = A·x - (max(A,0)·xl + min(A,0)·xu) >= 0`.

The sum of affine, ReLU and concretization slacks must equal:

`65536*(actual raw target margin) - final_scaled_bound`.

Separately record the final integer rounding credit `ceil(final_scaled_bound/65536)-final_scaled_bound/65536`. It strengthens a lower bound; it is not a loss. Failure of any identity or nonnegativity check is `IMPLEMENTATION_ERROR`, not a bottleneck finding.

This is an exact **fixed-chain, fixed-witness** accounting. Do not add hypothetical gains from interventions to these terms, subtract observed slack from a bound, or call the terms independent causal contributions. Changing a slope changes coefficients, rounding budgets, facts and earlier steps together. A nominal point can be far from the minimizing point even when the global bound is tight.

### Required records

`steps.jsonl.gz`: envelope identities plus `witness_id,chain_id,step_id,kind,layer,a_hash,a_out_hash,c_in,c_out,coefficient_l1,nonzero_coefficients,csr_touched_columns,zero_cancelled_columns,g_envelope,max_product,bias_dot,residual_dot,rounding_term,B,q,rho,slack,residual_endpoint_slack_num,rounding_envelope_slack_num,constant_floor_slack_num,input_slack,accounting_identity_passed`.

`coordinates.jsonl.gz`: `run_id,witness_id,step_id,layer,neuron,a,a_out,raw_r,clamped_z,activation_h,accumulator_s,rounding_e,lower,upper,width,lower_source,upper_source,lower_evidence,upper_evidence,clipped,saturated,relu_class,d,relu_slack,residual_coefficient,residual_endpoint_slack_num,rounding_envelope_slack_num` (inapplicable fields null).

`bounds.jsonl.gz`: `run_id,layer,neuron,box_postclamp_pre_relu_low,box_postclamp_pre_relu_high,box_activation_low,box_activation_high,chain_raw_low,chain_raw_high,selected_raw_low,selected_raw_high,lower_box,upper_chain,raw_source_validity,selected_width,stable_class,clipped,saturated,sampled_raw_min,sampled_raw_max,sampled_activation_min,sampled_activation_max,sample_count,lower_slack_at_nominal,upper_slack_at_nominal`. Sample extrema are observations, never sound global bounds.

Also log, separately:

- availability of tighter calculated chain bounds that source-selection rules did not select;
- proof restrictions, such as unavailable lower facts on stable-inactive neurons;
- clamp closure's two edges per competitor and their source identities;
- the exact minimizing input corner, its crop-lift validity and actual C margins;
- encoded endpoint gaps/unattainable values and source alias groups (F=8 removes scalar encoder gaps, not alias constraints);
- whether any coefficient truncation/top-k pruning was actually executed. The inspected v2 `run_chain` performs no top-k truncation; do not attribute losses to the different `affine_export.truncate` routine merely because it exists.

For each focus chain rank candidate neurons by **maximum observed per-coordinate ReLU slack over retained valid witnesses**, descending; ties by larger `abs(a)*(upper-lower)`, then layer descending, then neuron ascending. Only finite, representable bounds qualify. Use this one frozen ranking for I-S and I-B; do not rerank after each trial. Record both the top 16 globally and the top 16 within each hidden layer, not just the chosen neuron. Apply the same ranking formula to the unit-bound chains when I-B constructs them.

**Interpretation/cost:** concentration supports a targeted probe; no concentration does not establish that CROWN is optimal. Expected seconds to minutes after traces; 900 s stage cap, streaming records, 8 GiB memory. No new solver calls. **Compatibility:** accounting is external measurement; unchanged chains remain representable. **Risks:** raw/clamped confusion, interpreting sample extrema as invariants, double-counting intermediate widths as extra additive slack, or using float summation for identities.

## 8. Isolated interventions

Every arm starts from the saved epsilon-one baseline, never from another intervention's winner. Freeze model, domain, S=16, all unaffected bounds, logit closure roots and rules. Recompute upstream consequences of the single changed choice using normal deterministic baseline rules. No greedy accumulation of improvements. Store the exact changed-field list and hashes of everything declared frozen. A discovered real violation does not prevent diagnostic measurements, but a new bound contradicting that witness is an audit failure.

Except I-S, use only each image's primary competitor (40,5,37). I-S covers all 11 focus competitors. Reevaluate all 42 raw margins and closure flags only for a finalist; this does not constitute a full certification.

### I-S: coordinate relaxation choices

**Question/H-S:** Does baseline slope selection leave readily recoverable precision with the same intermediate bounds?

For each focus chain, take the first eight eligible coordinates in D-T's frozen ranking. Eligible means `l<0<u`, not saturated or clipped, and `Q_LOW <= l <= u <= Q_HIGH`. Both signs of nonzero incoming `a` are allowed. Test the distinct integers `{0, a, floor(a/2), baseline_a_out}`, excluding the unchanged value. Recompute `d` using `relu_const` and rerun the complete chain. Each trial overrides **one** coordinate's `a_out`; all other decisions use the normal heuristic. Assert the incoming `a` at that selected step equals its saved value, since no later-in-backsubstitution choice was changed. No multi-coordinate combinations, gradient tuning or additional passes.

At most 264 changed-coordinate trials, fewer after duplicates. Replay each produced inequality on retained witnesses. Record negative/positive coefficient sign separately so a negative-coefficient effect is not attributed to P3's positive-only policy family.

**Hooks:** `relu_choice`, `run_chain`, `needs`, nested `emit`; add an immutable keyed override `(root_id,layer,neuron,expected_a,a_out)` shared by all three. Supporting bound chains remain baseline. Do not pass one array to execution and recompute a default policy in dependency discovery.

**Log:** `candidate_rank,l,u,a,baseline_a_out,trial_a_out,baseline_d,trial_d,class_before,class_after,required_fact_sides_before/after,scaled_bound_before/after,integer_bound_before/after,all_witness_checks,root_terms,root_two_sided_count`, common resource fields. Schema: `interventions.jsonl`, hypothesis H-S; finalist dependency costs in section 10.

**Support:** material gain supports H-S for that margin; any gain establishes sensitivity within this finite family. No gain after all selected trials rejects only the claim that **these single-coordinate alternatives** help; it does not establish an optimal relaxation or rule out combined/per-neuron continuous optimization. `NOT_RUN_BUDGET` is inconclusive.

**Cost:** 1,200 s total cap, 8 GiB; no ESBMC during screening. **Certificate:** existing ReLU classes can encode these coefficients when their required bounds satisfy current class restrictions. Finder/export changes only; no new checker rule. Additional bound chains and two-sided facts can raise cost. Unsupported class/fact values are logged, not coerced. A positive margin without all dependencies checked remains a proposal.

### I-B: change one intermediate bound side

**Question/H-B:** Can an attainable improvement in one intermediate endpoint materially improve the root? Separate sensitivity from the ability to prove the endpoint.

For each primary margin take the top two eligible D-T coordinates with nonzero raw interval width, within supported Q16 bounds. For each coordinate test lower and upper sides separately, giving at most four targets per image. Let nominal raw value be `r0` and baseline sides `l,u`. The hypothetical tighter endpoint is `l'=ceil((l+r0)/2)` or `u'=floor((u+r0)/2)`. Skip an unchanged endpoint. **Counterfactual arm:** replace only that endpoint and rerun the margin with automatic baseline slope/class recomputation; label `COUNTERFACTUAL_ONLY`. It is not an epsilon-one invariant just because it contains the nominal point. Check all retained witnesses and log `hypothetical_endpoint_refuted_by_witness` and the witness ID. Such a refutation is an expected possible result of a hypothetical arm, not a global instrumentation failure; never rank it as attained evidence.

In candidate order, attempt a representable bound improvement for each targeted side: create its unit lower/upper root and apply I-S's same one-coordinate candidate family to the first eight eligible coordinates of that bound chain, ranked by the same trace rule. Choose the largest unit-root scaled lower bound, ties by fewer root multiply terms, then candidate order. Preserve its full exact trace. This is at most 24 trials per side, and at most 288 across three images. If the resulting integer endpoint is strictly tighter, apply **that attained endpoint alone** to the primary margin, freezing every other intermediate endpoint. Log how much of the hypothetical gain was attained. Never replace the result with the hypothetical endpoint.

For the one target side per image with greatest positive counterfactual root gain (ties: ranking, lower before upper), obtain one exact probe as a fixed, gated part of this stage: an ESBMC full-prefix assertion of the midpoint endpoint over the original `D_box` byte domain. If the selected endpoint already has a concrete refuting witness, skip the query as `REFUTED_BY_KNOWN_WITNESS`; do not choose another endpoint. Otherwise generate the ancestor cone exactly; if more than 250,000 nonzero multiply terms or 8 MiB generated harness, skip `STRUCTURAL_SIZE_CAP`. If neither gate applies, use a 60 s query, 4 GiB, one worker. Cone includes every predecessor and exact rounding/clamp; no hidden state is independently nondeterministic. Log cone size even if skipped. SAT supplies an input violating this proposed intermediate bound, **not necessarily the output property**; replay it at the whole network before any claim about output robustness. When the cone omits irrelevant input bytes, lift a partial witness by filling omitted bytes with nominal values, then replay the full encoder/network; do not claim a complete witness before this succeeds.

**Hooks:** `bounds_v2` for explicit one-side injection; `run_chain/relu_choice/needs/emit` for the new unit-root policy; dependency closure in `main`; `conv_output_terms/dense_output_terms` and shared kernel for the exact prefix. Keep lower-layer dependencies acyclic. `box_ledger` endpoints must never be edited to pretend the new bound came from an old block.

**Log/schema:** `target_layer,target_neuron,side,baseline_endpoint,nominal_raw_value,hypothetical_endpoint,hypothetical_endpoint_refuted_by_witness,refuting_witness_id,counterfactual_gain,attained_endpoint,attained_gain,bound_chain_id,bound_policy,bound_dependencies,root_classes_changed,prefix_terms,prefix_source_bytes,prefix_esbmc_status,prefix_counterexample_kind`; common intervention fields and separate run IDs for hypothetical, attained, and exact-prefix arms.

**Support:** an attained endpoint plus root gain supports refinement. Hypothetical gain alone shows sensitivity, not feasibility. Tighter endpoints with no gain reject those endpoints as useful in this frozen root computation. Failed midpoint assertion rejects that proposed bound over its exact tested domain; timeout leaves it unresolved. No improved unit chain says nothing about exact tightness.

**Cost:** 1,200 s stage cap including at most three prefix queries. **Certificate:** an improved endpoint justified by an ordinary lower/upper chain fits the current checker, with correct fact identity, exact source bound equality, and changed supporting-chain policies. It may require exporting chain-backed bounds where the current exporter chooses box bounds. A new direct ESBMC-prefix fact is **not** an accepted source kind in `_validate_facts`; it requires a future schema/provenance extension if promoted. The hypothetical arm is never exportable.

### I-R: one two-neuron relational inequality

**Question/H-R:** Does preserving one shared-input relation improve a bound that separate neuron intervals cannot? This probe tests one selected relation, not all relational domains.

Use each primary chain. From the 16 highest-ranked nonzero-coefficient coordinates at each hidden ReLU step, enumerate oriented pairs `(p,q)` with `a_p>0`, `a_q<0`. Require at least one common nonzero predecessor and at most 256 distinct predecessor coordinates in the union of their CSR rows. Exclude pairs with singleton activation difference bounds. For each pair let `gamma=min(a_p,-a_q)` and activation endpoints `hp_lo=clip(l_p,0,Q_HIGH)` etc. The independent difference interval is `[L0,U0]=[hp_lo-hq_hi,hp_hi-hq_lo]`.

Rank pairs by the exact rational score `gamma*(U0-L0)*(shared_support_size/union_support_size)`, descending; ties by layer descending, then p,q ascending. Select exactly one pair per image; no fallback to a second pair after failure. Log `SKIPPED_NO_CANDIDATE` if none. This selection heuristic is a proposal, not measured evidence of correlation.

Let `vmin` be the minimum exact `h_p-h_q` over retained witnesses in the tested domain. Propose **one lower relation** `h_p-h_q >= L1`, where `L1=ceil((L0+vmin)/2)`. If `L1<=L0`, skip `NO_NONTRIVIAL_CUT`. Sample data proposes the cut; it never validates it.

Generate one exact two-row C harness over their shared predecessor activation coordinates, with bounds from the saved baseline. Compute both neurons with the same nondeterministic predecessor variables, correct biases, rounding, saturation and ReLU, then assert `h_p-h_q>=L1`. For layer 0, predecessors are encoded byte variables. For deeper layers, they are an overapproximate hidden box. Use a 60 s ESBMC query, 4 GiB. Include a concrete nominal predecessor witness for nonvacuity and parity. Read `conv_proof.render_dense_relational_cut_validation` as a template, but create a diagnostic CSR-two-row renderer so convolution weight/coordinate sharing is preserved; do not flatten the whole CNN into a dense layer.

Only if this local assertion returns VERIFIED, compute a conditional diagnostic chain with the one relation inserted before that layer's ReLU substitution:

`a_p <- a_p-gamma; a_q <- a_q+gamma; C <- C+gamma*L1`.

Then continue ordinary ReLU/backsubstitution. The added inequality is `gamma*(h_p-h_q-L1)>=0`; check it on all retained witnesses and add its slack to the trace identity. All other bounds and policies stay baseline. A solver failure produces only a **local hidden-state refinement witness** unless layer-0 bytes or a full input lift are actually available and replayed. UNKNOWN/TIMEOUT means no relation may be consumed, even experimentally as a validated relation; an optional numerical illustration must be labelled counterfactual and is excluded from decision support.

**Hooks:** diagnostic pair renderer using `conv_output_terms/dense_output_terms`, `render_arith_kernel`, and `SparseRelationalCut` if useful; observer at `run_chain` before `relu_choice`. Production `validate_chain` has no relational step and `_chain_property` does not accept arbitrary two-neuron relation roots. Do not disguise this as an ordinary neuron-bound chain.

**Log:** `pair,layer,a_p,a_q,gamma,support_indices,support_overlap,predecessor_box_hash,predecessor_bound_dependencies,L0,U0,vmin,L1,local_harness_sha256,solver_status,local_witness,nominal_nonvacuity,cut_consumed,conditional_root_gain,new_relation_slack`, plus common metrics. `dependency_status=SCREEN_ONLY` if predecessor bounds lack ledger/chain proofs, even when the local lemma is VERIFIED.

**Support:** a checked local relation with conditional material root gain supports implementing this proof form. VERIFIED cut with no gain rejects this cut as useful in the fixed chain. A local counterexample refutes the proposed cut over that predecessor box, not necessarily over reachable network states. No selected cut is not evidence that correlations are absent.

**Cost:** three local queries maximum, 600 s stage cap including construction/replay. **Certificate:** requires a new relation fact/step, exact predecessor-context binding, composition rule and independent coverage if promoted. Existing dense-cut infrastructure does not make it automatically CROWN-v2-compatible. **Risks:** independent copies of shared inputs; unverified cut assumptions; allowing nominal/sample correlation as a theorem; importing hidden-state failures as real counterexamples.

### I-P: one two-child input case split

**Question/H-PART:** Can a single sound partition improve bounds enough to justify future case splitting? Test input splitting first; do not implement hidden-neuron branching.

For each primary baseline chain choose one encoded-input coordinate with nonzero byte width maximizing `abs(final_A_i)*(byte_high_i-byte_low_i)`, ties by flat input index. If every score is zero but some byte coordinate varies, select the lowest varying index and record `ZERO_LINEAR_SCORE`. If none vary, skip. Work over `D_box`, not a subset of original-crop inputs: one coordinate is an independent dimension of that existing proof domain. For byte interval `[lo,hi]`, set `m=floor((lo+hi)/2)` and define children `b_i<=m` and `b_i>=m+1`. Keep every other byte dimension unchanged. The two finite domains are disjoint and their union is the full parent box.

Create fresh child encoded endpoints, box propagation, facts and baseline `bounds_v2`; use no optimized policies or other interventions. Evaluate all 42 competitors in each child. Record each child's raw bound and clamp edges separately. For each competitor, the partition raw bound is `min(child0_raw_lb,child1_raw_lb)`; strict dominance is closed only if both children's raw bounds and both clamp edges close. Compare with the parent's bound and report also `max(parent_lb,partition_lb)` as an available combined lower bound, without concealing either child's regression. No recursive split or best-of-many dimension search.

**Hooks:** diagnostic explicit-input-region entry point into `setup`'s downstream operations and `run_chain/bounds_v2`; original `setup(eps,image)` remains unchanged. `ByteImageEncoder` produces endpoints from child **byte** bounds. Reset `XU` per child. Do not simulate this by changing epsilon for the whole image or modifying nominal pixels.

**Log:** `selected_coordinate,linear_score,lo,hi,threshold,parent_region_hash,child_region_hashes,child_bounds,child_clamp_edges,child_closed_counts,union_raw_lb,union_closure,partition_coverage_passed,parent_and_child_costs,shared_dependency_ids`. The domain hash includes which coordinate changed. Test coverage symbolically using integer endpoint arithmetic and enumerate the at-most-three allowed values of that coordinate as a control.

**Support:** improved worst-child bound, particularly material gain or both children closing a formerly open competitor, supports further input splitting. Improvement in only one child without worst-child gain does not. No gain from this one split is not evidence against hidden-neuron branching or other dimensions.

**Cost:** six fresh child screens; expected roughly a small multiple of baseline screening (historical single-image P3 screen/search/cost runs were 56–85 s for these images), 1,200 s stage cap. No ESBMC child campaign in this diagnostic.

**Certificate:** each rectangular child uses existing arithmetic chain rules, but needs matching child encoding/box evidence. The current exporter requires ledger equality with its propagated box; parent evidence is not automatically a child ledger. A proof of the original region from two child certificates needs a new explicit partition-coverage/composition record and checker support. No full-child checker runs are authorized by this specification alone. **Risks:** omitting the second child, dropping boundary byte values, reusing branch-local facts globally, or counting both children as separate recovered images.

### I-Q: more precise rounding, leaving arithmetic unchanged

**Question/H-Q:** Does the universal independent rounding budget discard useful information about actual integer accumulators? Changing the model's rounding mode is prohibited.

For each primary chain select the affine layer with largest maximum-over-witnesses `rounding_envelope_slack_num`; ties by layer descending. Baseline coefficients/residuals/bounds remain fixed. At every nonzero-coefficient output row of that layer, derive an accumulator interval `[slo,shi]` using exact weighted interval arithmetic over the saved predecessor activation box. Compute `g=gcd(256, weights of non-singleton predecessor coordinates)`, taking g=256 if all predecessors are fixed. Set `c=sum(weight*fixed_value)` over singleton predecessor coordinates, modulo g. Every attainable accumulator satisfies `s ≡ c (mod g)`; using it with `[slo,shi]` is an overapproximation, not an exact reachable set.

Enumerate residues `rho=0..255` with `rho ≡ c (mod g)`. For each residue check, by exact integer arithmetic, whether some `s=rho+256*k` lies in `[slo,shi]` and in each sign interval `s>=0` or `s<=-1`. Do not test only interval endpoints. For possible nonnegative s, `e=-rho` for rho<128, otherwise `e=256-rho`. For possible negative s, `e=-rho` for rho<=128, otherwise `e=256-rho`. This distinction preserves opposite signs of exact half ties. Derive `emin,emax` over these possible cases. An empty set is an audit failure, since the interval/congruence must contain the nominal accumulator.

For coefficient a, replace its baseline contribution `-128*abs(a)` with `min(a*emin,a*emax)`. Let K be the sum of all these minima. Run two separate arms from baseline:

- Q-one: tighten only the row giving the largest nonnegative contribution gain; ties by row index.
- Q-layer: tighten all rounding contributions at this **one selected layer**.

At that affine step use `C_out=C_in+a·bias+floor((R·hlow+K)/256)`; other affine steps retain the old budget. Do not alter coefficient division, constant-floor convention, ReLU choices, or intermediate bounds. A zero gain is recorded rather than searching another layer.

Generate one compact ESBMC validation harness per image for Q-one: nondeterministic `__int128 s` satisfying its accumulator interval/congruence and explicit no-overflow range, compute e with the shared kernel, assert `emin<=e<=emax`. Include all residue/sign cases as a separate Python exact check and nominal nonvacuity. This validates the rounding statement under its conditions; justification of the accumulator interval/congruence remains an explicit dependency. Q-layer's other rows are mathematical diagnostic calculations, not automatically ESBMC-checked facts.

If Q-layer gives no material gain, perform exactly one further probe Q-pair, independent of Q-one/Q-layer: among nonzero rows of the selected affine layer, choose a pair with shared predecessors and union support <=256 maximizing `(abs(a_p)+abs(a_q))*shared/union`, ties by row indices. Propose a joint lower bound on `a_p*e_p+a_q*e_q` halfway (ceiling) between its baseline independent lower bound and the minimum observed value over retained witnesses. If unchanged, skip. Check this one predicate with exact two-row accumulator C over the shared predecessor box, 60 s. Only after VERIFIED substitute that joint bound for these two baseline rounding contributions and rerun the root. This distinguishes independent-error approximation from restrictions on individual rounding residues without building a general modular solver.

**Hooks:** `run_chain` affine `rounding_term` calculation and diagnostic override; CSR rows via `lowered`; C kernel; prospective changes would touch `validate_chain`, `_render_affine`, `render_rounding_lemma`, `_validate_facts`/new rounding provenance, and coverage. Do not make these production-checker changes now.

**Log/schema:** `selected_layer,row,slo,shi,gcd,congruence_offset,possible_residue_sign_pairs,nominal_s,nominal_e,emin,emax,a,old_error_contribution,new_error_contribution,K,old_constant,new_constant,root_gain,local_lemma_status,interval_dependencies`; Q-pair adds `rows,shared_support,independent_bound,observed_min,proposed_joint_bound,verified_joint_bound,hidden_state_witness`. Use the common envelope and exact decimal strings throughout.

**Support:** Q-one/Q-layer improvement supports residue/sign-aware bounds. Only Q-pair improvement supports preserving correlation between rounding errors. A large trace rounding slack but no gain in these probes is **not** proof that rounding is irrelevant; it may be irrelevant at the true minimizer, require other rows/layers, or require stronger reachable-state information. Distinguish these possibilities as unresolved.

**Cost:** 900 s stage cap; at most six 60 s local solver calls (one residue lemma and one conditional pair per image), streaming residue enumeration. **Certificate:** no. Current `validate_chain` demands exactly `128*sum(abs(a))` and the renderer rechecks it. Tighter budgets need explicit conditions/evidence and a new rule; never just write a smaller `rounding_term` into v2. **Risks:** sign/tie errors, omitted residue cases, treating a congruence overapproximation as equality of reachable sets, accumulator overflow, or assuming independently worst rounding errors can/cannot co-occur without evidence.

### I-C: coefficient-precision control (additional identified loss)

**Question/H-C:** Is scale-S coefficient division/residual folding materially limiting the margin, independently of the deployed quantization format?

For each primary root rerun once with root coefficient scale `S=24` instead of 16, preserving the **numeric epsilon-one intermediate endpoints**, baseline choice algorithm and deployed Q7.8. The root coefficient scale is certificate arithmetic; changing it does not change the network. Recompute the root, not its supporting bound construction, and compare `scaled_bound/2^S` and integer raw bounds. Use explicit scale arguments in the diagnostic path; do not mutate imported module-level S in a running process. Log whether exact integer coefficient choices/classes changed as a consequence.

Before every vectorized multiplication/sum check conservative Python-integer bounds for all intermediates, including D, bias products, residuals, constants, l1 norms and concretization, not just `W^T a`. If int64 is unsafe, use the arbitrary-integer diagnostic replay for this root. Respect the stage deadline; if it cannot finish, report `OVERFLOW_GUARD` or budget exhaustion. Do not narrow a value merely to finish.

**Hooks:** diagnostic root construction and scale parameter in `run_chain`; `choose_relu/relu_const` and exact observer. **Log:** `S_before,S_after,all_headroom_bounds,arithmetic_backend,coefficient_bit_lengths,residual_stats,constant_floor_stats,normalized_root_gain,integer_root_gain,changed_classes,root_terms`, common schema.

**Support:** improvement supports coefficient-precision sensitivity; no gain at S=24 does not prove arbitrary precision would not help. The final constant floor loses less than one coefficient unit per affine step; use D-T's measured amount to judge its relevance rather than declaring all integer division a large loss.

**Cost:** 600 s stage cap, no ESBMC. **Certificate:** the root formula and `scale_bits` mechanism can express S=24, but v2 uses one scale for the complete export. This diagnostic's frozen S=16-derived supporting endpoints do not automatically have S=24 chains yielding exactly the same values. A full compatible export would need consistent supporting chains/fact identities and new overflow envelopes. Record `ROOT_RULE_COMPATIBLE_FULL_EXPORT_UNRESOLVED`; do not claim a certificate from the mixed diagnostic. No change to deployment bit width is contemplated.

### I-E: bounded exact ESBMC fallback diagnostic

**Question/H-E:** Can the exact solver settle a small remaining obligation within the diagnostic budget, and do failures in local probes come from unreachable hidden states?

Run once per primary output margin (three queries). Use the pinned full deployment C and independent symbolic byte variables over **the same D_box** as the baseline, passing the HWC resized tensor through the pinned encoder at its output dimensions, where nearest-center resizing is identity. Assert the single deployed margin `output[t] > output[c]`. Do not assume intermediate bounds, do not replace the property with a raw-logit assertion, and do not drop unrelated predecessor computations unless exact dependency slicing has been mechanically checked. Use the whole forward function for this first probe.

Use 120 s/query, 4 GiB, one worker. A failed target assertion with complete byte values is replayed through the unmodified oracle and classified by crop-lift status. VERIFIED proves that one margin over the stated box under the recorded ESBMC configuration, not all 42. A timeout says only that this exact encoding exceeded this budget. This arm is a benchmark of fallback feasibility, not a new solver architecture.

**Hooks:** pinned `generate_conv_qnn_source`, encoder, diagnostic harness and `ESBMCRunner`. **Log:** `target_assertion,domain,harness_sha256,unwind_bound,solver_command,formula_statistics_if_available,status,failed_claim_kind,complete_witness,exact_replay,crop_lift_status`, common cost/schema. Store missing formula statistics as null.

**Certificate:** standalone exact margin evidence is meaningful but not a current v2 root-chain source. Integration would require an explicit alternate-root obligation and full competitor/dependency coverage; do not relabel it as a v2 certificate. **Cost:** 600 s stage cap including setup; three 120 s solver calls. **Risks:** insufficient unwinding, infeasible assumptions, trace extraction omitting bytes, and a nominal-correctness assumption accidentally excluding adversarial inputs.

## 9. Solver contract, controls, and stopping rules

All diagnostic ESBMC calls use the installed executable pinned by path/hash/version, Z3 bit-vector mode, default safety checks, and enabled unwinding assertions. Base arguments come from `ESBMCRunner.build_command(..., profile='paper-z3')`. A **diagnostic-only** runner subclass removes `--result-only` so the first run retains the counterexample trace; record the effective profile as `diagnostic-paper-z3-with-trace`. Keep the interval-analysis flags and state that they remain in the trust base. Call `run_file(extract_counterexample=False)` so the existing automatic trace rerun cannot double an unrecorded budget. Parse the saved complete log, using a new trace parser if the existing `input[i]` pattern cannot recover `__int128` variables. Missing bytes mean `INCOMPLETE_WITNESS`, not a concrete counterexample.

For each generated harness derive a conservative maximum iteration count over **every** loop, including CSR offsets, encoders, fixed-width arrays and complete forward loops. Add a diagnostic `#define INPUT_SIZE <max_loop_iterations>` if needed to use `infer_unwind`; ensure its returned bound is at least max iterations +1. Save a loop-bound inventory. Do not rely on the old regex finding literal `for` bounds inside generated CNN C. No `--no-unwinding-assertions`, disabled range/pointer checks, solver slack or altered arithmetic.

Name assertions uniquely (`diag/<run_id>/<predicate>`). A VERIFIED local conditional assertion is attached to its exact assumption context, including bound sources, source/region identity and arithmetic rules. For a bounded symbolic domain, replay a known concrete assignment satisfying assumptions as a nonvacuity control. If the assumptions do not admit the nominal witness where they should, stop. Mathematical interval/congruence facts derived in Python remain explicitly in the dependency/trust record, even if the local C predicate verifies.

Before launching solver probes, run compact diagnostic harness controls: a true rounding lemma and a deliberately false one, including s=+128 and s=-128; a valid and invalid two-neuron inequality on a tiny fixed example; and the target assertion vs tie assertion at a fixed synthetic logit pair. Use these to check assertion identity/status parsing, not to claim production soundness. They are implementation validation, outside scientific outcome denominators. Cap each at 10 s, total 60 s, include their resources in D-P's cap.

Global stopping rules:

- Any pinned artifact/source mismatch, negative slack in a purportedly valid arm, trace accounting identity failure, baseline regression, wrongly classified domain witness or control-017 concrete violation stops all dependent work as an audit issue. Explicit `COUNTERFACTUAL_ONLY` inequalities can fail on other witnesses: record that refutation without treating the hypothetical bound as a theorem or an implementation defect. A correctly labelled box-only witness is not a wrong-domain reporting error.
- A local ESBMC timeout/unknown fails only that probe; it does not stop independent probes and does not count as negative scientific evidence.
- On a deadline or memory kill, persist all completed records and mark remaining scheduled IDs `NOT_RUN_BUDGET`. Do not replace partial failure with `NO_GAIN`.
- Use one solver process at a time; no competing experiments. Limit each solver to 4 GiB, driver/stage cgroup to 8 GiB, output growth to 16 GiB per run. Suspend new work if available memory falls below 6 GiB; if it does not recover within the remaining stage budget, report budget exhaustion. No unlimited retry/resume or enlarged limits.
- Never alter epsilon one, strict dominance, target, deployment or arithmetic to get a positive outcome. Epsilon-zero and child domains are explicitly different diagnostic arms with their own identities.

## 10. Verification cost and prospective certificate burden

**Question/H-COST:** Is an observed bound gain affordable to check, and where does it add proof obligations? This is independent of whether it closes an image.

Every trial logs root work (multiply terms, nonzero coefficients, two-sided coordinates, maximum coefficient/constant magnitude), wall time, actual CPU time if available, process-tree RSS and cgroup memory peak. Measure monotonic wall time and poll process-tree resources at 50 ms. A missed/unsupported resource metric is null with a reason, never zero. Do not call summed per-call elapsed time CPU time.

Computing a complete dependency graph for every candidate would dominate the diagnostic. Therefore do it for the baseline and **one finalist per intervention per image**, selected deterministically as the largest positive integer raw-bound gain across that intervention's eligible trials; ties by exact normalized scaled gain, then fewer root multiply terms, then trial ID. If no positive integer gain, retain the largest positive scaled gain as `SUBINTEGER_FINALIST`; otherwise select no finalist and record why. Never choose a counterfactual bound as a proof-cost finalist. For I-P the finalist includes both children. I-R/I-Q finalists are conditional/new-rule prototypes and must be labelled accordingly.

For each finalist:

1. Traverse the complete planned dependencies using the same recorded policy/bound overrides as execution. Count root and supporting chains, facts by source/side, multiply terms, ReLU classes, two-sided facts and closure roots. Record changed/new dependencies relative to **this image's baseline**, not relative to another image.
2. Serialize a prospective trace stream to a counting writer, recording exact uncompressed byte count and gzip byte count. Store it only within the output cap. Name it `proposal_trace`, never an accepted certificate. Keep absent ledger references visibly unresolved. The existing exporter cannot legitimately use `facts_from_box` as VERIFIED facts.
3. For current-rule-compatible traces, derive assertion and harness counts using the same renderer/chunking configuration and the independent coverage logic. Count all touched affine columns, including products summing to zero. Do not fake a `CertificateV2` that passes the production loader. If a validation-ready object cannot be constructed without missing ledger evidence, a diagnostic counting adapter may consume the already constructed arithmetic records, and its counts must be labelled `PROSPECTIVE_CONSTRUCTION_COUNT`, not checker acceptance.
4. Identify shared physical obligations by source content, predicate and assumption-context hash; keep root-to-dependency edges separately. An unchanged arithmetic expression under a different branch hypothesis is not automatically reusable evidence. Count additional unique obligations, total obligations and unverified dependencies independently.
5. For new-rule prototypes, report bytes actually serialized and local solver obligations actually executed. Report whole-certificate/checker counts as null with `NEW_RULE_NOT_IMPLEMENTED`, rather than pretending the additional relation/split/rounding rule is free. Include a named list of required future rule types.

For a small empirical checker-cost sample, use only existing-rule I-S/I-B finalists. Choose the best material-gain finalist for 013 and 016, plus the corresponding baseline samples from 017. For each selected image sample exactly three prospective arithmetic harnesses: the first changed affine chunk in traversal order (or first affine chunk if none changed), the affine chunk with maximum multiply terms, and the first changed ReLU chunk (or first ReLU chunk). Deduplicate identical harnesses. Run at most nine calls, 60 s each, 4 GiB. Do not run a full box or full-certificate campaign. For 013/016, VERIFIED arithmetic chunks do not discharge unproved bound dependencies. If no existing-rule finalist has material gain, skip its samples.

`proof_cost.jsonl` fields: common envelope plus `candidate_id,selection_reason,root_terms,whole_graph_terms,root_chains,supporting_chains,facts_by_kind,unverified_dependency_count,unique_assertions,unique_harnesses,additional_unique_assertions,additional_unique_harnesses,shared_with_baseline,serialized_bytes,compressed_bytes,max_coefficient_bits,max_abs_constant,required_new_rule_types,cost_status,sampled_harness_ids,sampled_elapsed,sampled_cpu,sampled_memory`.

**Interpretation:** historical `298–330 microseconds/term` may be displayed only as an explicitly labelled old-context estimate of summed solver elapsed time. It is not a wall-time model, does not predict new rule costs, and is not a measured speedup. Report `gain_per_added_term` and `gain_per_added_harness` only when denominators are known and positive; when zero use a categorical `NO_ADDITIONAL_COUNTED_WORK`. Do not select a technique solely by gain-per-term: soundness/representability and actual resource results have precedence.

**Cost:** 1,800 s stage cap for serialization, counting and at most nine sample checks. New certificates remain unverified; the existing 017 certificate remains the historical control. A larger serialization workload is a recorded diagnostic finding, not a reason to omit cost fields.

## 11. Decision table

Priorities are decision order, not promised effectiveness: P0 = resolve before interpreting other findings; P1 = first implementation candidate after diagnostic support; P2 = conditional next experiment; P3 = defer unless simpler probes fail. Apply all matching rows and cite their run IDs. If several P1 rows match, rank current-rule-compatible techniques first, then smaller measured incremental checker burden; if costs are missing, the next action is cost measurement, not an invented ranking. No gain in these limited probes justifies a universal negative conclusion.

| Observed evidence | Likely bottleneck | Next technique to implement | Certificate/checker change required? | Priority |
|---|---|---|---|---|
| Source/parity failure, negative accounting slack, baseline mismatch, or a bound above a concrete execution | Identity, arithmetic, instrumentation or soundness defect | Fix/re-audit the affected implementation before optimization | Depends on defect; audit current checker boundary | P0 |
| Lifted byte witness with deployed margin <=0, replayed exactly | Property false for that region/competitor | Preserve the counterexample; no finder technique can certify that property | Separate counterexample evidence only | P0 |
| Box-only violation that fails original-crop alias constraints | Cartesian resize abstraction includes unreachable inputs | Preserve encoder/alias relations if original-crop robustness is the intended theorem; keep D_box results separate | Yes for an alias-aware proof domain; do not silently change the old theorem | P1 |
| I-S material gain with same bound facts and manageable added dependencies | Relaxation choice is locally suboptimal | Slope/relaxation optimization, initially per-neuron and per-margin | Usually no new arithmetic rule; consistent finder/export policies required | P1 |
| I-B obtains a tighter ordinary bound chain and material root gain | Attainable intermediate-bound imprecision | Intermediate-bound refinement on selected sides | No new rule for current-form bound chains; new source kind if exact-prefix facts are used | P1 |
| Only hypothetical I-B bound changes help; attempts cannot establish them | Sensitivity established, attainable bound unknown | Bounded exact intermediate-bound query or stronger bound-chain search | Depends on chosen source; no promotion of hypothetical facts | P2 |
| I-B proves/refines an endpoint but root bound barely changes | That particular endpoint is not influential enough under the tested chain | Deprioritize that endpoint; test supported slope/rounding/relation result | According to next supported technique | P2 |
| Checked I-R relation yields material conditional gain whereas single-side refinement does not | Lost joint dependence for the tested pair | Relational constraints, starting with that one proof rule | Yes: relation fact, composition, context identity and coverage | P1 if costs acceptable; otherwise P2 |
| I-R local assertion fails only at an approximate hidden assignment | Predecessor box too broad, or proposed relation false there | Refine predecessor bounds or test exact prefix reachability before consuming the cut | Potential new exact-prefix/relational source; no real-CE claim | P2 |
| Q-one/Q-layer improves materially | Independent worst-case rounding interval is unnecessarily broad for selected accumulators | Exact or tighter rounding reasoning using range, sign and congruence | Yes: conditional rounding rule and bound/congruence provenance | P1 |
| Q-pair improves while individual rounding ranges do not | Correlation between rounding errors matters for that pair | Small joint exact rounding constraint | Yes: pair lemma and composition/dependency coverage | P1/P2 by measured cost |
| Z-exact retains large rounding slack, but I-Q gives no gain | Singleton rounding conservatism observed; region-wide relevance still unestablished | Target a bounded exact rounding/prefix obligation; do not assume a global fix | Yes if a stronger rule results | P2 |
| S=24 gains materially with intermediate numeric bounds frozen | Coefficient-scale/residual discretization sensitivity | Higher-precision certificate arithmetic with overflow control | Existing scale mechanism may suffice, but consistent supporting chains must be regenerated/checked | P1/P2 by measured cost |
| Both children improve the worst bound materially or close an open competitor | Single global abstraction is insufficient at the tested partition | Branch-and-bound/case splitting, beginning with explicit two-child composition | Child chain rules existing; parent coverage/composition and child evidence required | P1/P2 by measured cost |
| Only one child improves and the other keeps the parent bottleneck | Chosen split does not improve worst-case proof precision | Defer branching expansion; inspect the hard child before another hypothesis | Parent composition still required if pursued | P2 |
| Exact ESBMC proves a primary margin cheaply while CROWN stays open | Approximation/proof language is the limiting factor for that query | Exact ESBMC fallback for selected obligations | Yes for alternate-root integration; standalone evidence already possible | P1 |
| Exact ESBMC times out or runs out of memory | This exact encoding exceeds the chosen budget | Prefer whichever bounded refinement has positive measured evidence | No conclusion about robustness/completeness | P2 |
| Positive mathematical gains require excessive dependencies, serialization, memory or checking time | Proof/checking cost limits practical usefulness | Reuse sound context-identical evidence or reduce the chosen refinement scope before expansion | Possibly packaging/coverage changes; no weakened checking | P1 cost work |
| All bounded probes finish with no material gain and no concrete violation | Cause remains unresolved; may be distributed losses or insufficient probe coverage | Record the negative; choose one new hypothesis from trace evidence, not a larger blind campaign | Undetermined | P3 |

Do not infer that a gain from one technique excludes other bottlenecks. The decision table records evidence for the next **minimal** implementation, not an adaptive architecture.

## 12. Fixed configuration and command interface

The coding agent should implement a JSON-configured driver with stages `preflight`, `epsilon0`, `violations`, `trace`, `slopes`, `bounds`, `relation`, `split`, `rounding`, `coefficient_scale`, `exact_fallback`, `cost`, `report`, and `all`. The `all` order is exactly the order above. Every stage reads frozen inputs and writes new records; it must not reinterpret historical summaries. A stage rerun goes in a fresh run ID and cannot replace the first result. `report` performs no network inference or solver calls.

Proposed configuration (save as `experiments/crown_diagnostics_013_016_017.json` **when implementing**, not as an already-run protocol):

```json
{
  "schema": "preqbmc_crown_diagnostic_config_v1",
  "worktree": "/home/joao/code/preqbmc-qformat",
  "working_directory": "/home/joao/code/preqbmc",
  "python": "/home/joao/miniconda3/envs/quad/bin/python",
  "output_parent": "/home/joao/code/preqbmc/output",
  "deployment_sha256": "dd535f5480bec45cdae7774f849afca49f7afb5d645f02037e86a44d16f45097",
  "images": [
    {"ordinal":"013","record_id":"Final_Training/Images/00038/00039_00015.ppm","target":38,"focus":[34,35,36,37,40],"primary":40},
    {"ordinal":"016","record_id":"Final_Training/Images/00002/00023_00020.ppm","target":2,"focus":[0,1,5,21,31],"primary":5},
    {"ordinal":"017","record_id":"Final_Training/Images/00039/00007_00017.ppm","target":39,"focus":[37],"primary":37,"control":true}
  ],
  "epsilon_raw_bytes": 1,
  "singleton_epsilon": 0,
  "qif": {"total_bits":16,"fractional_bits":8},
  "coefficient_scale_bits": 16,
  "coefficient_scale_probe_bits": 24,
  "seed_base": 20261031,
  "attack_restarts": 4,
  "attack_proposals_per_restart": 256,
  "slope_coordinates_per_root": 8,
  "bound_coordinates_per_primary": 2,
  "relation_top_coordinates_per_layer": 16,
  "local_pair_support_cap": 256,
  "prefix_multiply_term_cap": 250000,
  "prefix_harness_byte_cap": 8388608,
  "case_split_depth": 1,
  "solver_workers": 1,
  "solver_memory_gib": 4,
  "stage_memory_gib": 8,
  "minimum_available_memory_gib": 6,
  "artifact_cap_gib": 16,
  "local_solver_timeout_seconds": 60,
  "full_margin_timeout_seconds": 120,
  "global_wall_cap_seconds": 14400,
  "stage_wall_caps_seconds": {
    "preflight":900,"epsilon0":900,"violations":2400,"trace":900,
    "slopes":1200,"bounds":1200,"relation":600,"split":1200,
    "rounding":900,"coefficient_scale":600,"exact_fallback":600,"cost":1800
  },
  "full_certificate_campaign": false,
  "modify_production_checker_rules": false,
  "optimize_model_or_training": false
}
```

Stage caps sum to 13,200 s (3 h 40 min); the 14,400 s global cap leaves 20 min for reporting/overhead. These are **proposed hard limits**, not measured expected runtime and not authorization to run the campaign. Most pure trace/probe work may finish sooner; exact solver feasibility is unknown. Keep scientific call-count limits even if time remains. Conversely, time caps can leave scheduled calls unrun. For I-S and I-B process images in order 013,016,017 and round-robin one candidate per image so the control is not systematically starved by a deadline.

Existing baseline reproduction command example (read/write diagnostic output only, do not execute as part of preparing this document):

```sh
cd /home/joao/code/preqbmc
PYTHONPATH=/home/joao/code/preqbmc-qformat/src \
  /home/joao/miniconda3/envs/quad/bin/python \
  /home/joao/code/preqbmc-qformat/src/scripts/crown_finder/crown_chain_export_v2.py \
  screen 1 Final_Training/Images/00038/00039_00015.ppm --cost \
  --report /absolute/new/run/013_baseline.json
```

The new diagnostic CLI, **after implementation**, must accept:

```sh
cd /home/joao/code/preqbmc
PYTHONPATH=/home/joao/code/preqbmc-qformat/src \
  /home/joao/miniconda3/envs/quad/bin/python \
  /home/joao/code/preqbmc-qformat/src/scripts/diagnose_crown_finder.py \
  --config /home/joao/code/preqbmc-qformat/experiments/crown_diagnostics_013_016_017.json \
  --stage all --output /absolute/new/run
```

Also implement `--stage <name>` and `--from-run /absolute/prerequisite/run`, requiring compatible manifest/config/region hashes. `--stage report --from-run ...` writes `report.md` and `decisions.csv` without resuming computation. No `--search-until-certified`, automatic retries, automatic commits or silent budget changes.

### Expected work and interpretation by stage

| Stage | Work count | Expected order of cost, not a measured prediction | Hard wall cap |
|---|---|---|---:|
| D-P | 3 baseline screens + oracle/parity controls | Minutes; baseline counts must reproduce | 900 s |
| D-0 | 3 singleton arms x 3 images, 42 margins each | Minutes; fresh bound construction dominates | 900 s |
| D-V | <=1,028 proposals x 11 focus roots; <=3 fixed-witness solver replays | Native inference dominates; hardware-dependent | 2,400 s |
| D-T | 11 chains on <=20 retained standard witnesses, plus singleton primary chains | Seconds to minutes, streaming storage | 900 s |
| I-S | <=264 one-coordinate trials | Minutes; depends on sparse root support | 1,200 s |
| I-B | <=12 endpoint targets, <=288 bound-root trials, <=3 prefix queries | Minutes; exact-prefix probes may time out | 1,200 s |
| I-R | <=3 two-neuron assertions and conditional root reruns | Small local checks, possibly timeouts | 600 s |
| I-P | 2 child screens x 3 images | Multiple baseline screen costs | 1,200 s |
| I-Q | Residue analysis, <=3 single-error + <=3 joint-error queries | Minutes; pair checks uncertain | 900 s |
| I-C | 3 S=24 roots | Small if int64-safe, potentially slower with Python integers | 600 s |
| I-E | 3 exact full-margin queries | Assume each may consume its full 120 s | 600 s |
| Cost | Baseline/finalist graph counting/serialization + <=9 sample queries | May dominate I/O; no complete-network proof run | 1,800 s |

## 13. Acceptance criteria for the diagnostic implementation

These are checks of instrumentation and experimental discipline, not tests asserting that a proposed technique must improve the model.

1. Default source generation preserves the deployment hash. Observer and unmodified C agree on nominal and retained-witness outputs. Nominal raw, clamped and activated traces agree with arbitrary-integer replay.
2. Preflight reproduces all three baseline minima, closure sets and planned term/chain counts. All project modules resolve into Claude's worktree. Save the original uncommitted diff hash and a separate diagnostic diff hash.
3. A tie fixture is classified as a property violation; a raw-margin violation without deployed-margin violation is not. A resize-alias-inconsistent witness cannot acquire `CROP_WITNESS` status. Byte clamping at 0 and 255 is covered.
4. D-T exact identities pass for baseline and supported intervention traces, including negative coefficients, negative half ties, saturating values, zero-cancelled CSR columns and final integer ceiling. Diagnostic relation steps have an explicit extra slack term. No nonexistent top-k loss is reported.
5. Every intervention record proves by field/hash comparison that only its declared parameters changed, apart from their required deterministic upstream consequences. Independent trials do not inherit each other's state. Epsilon-zero, child and baseline `XU` values cannot cross-contaminate.
6. Every I-S/I-B representable trace replays the same coefficients and bound in dependency discovery and prospective serialization. For genuinely validated source fixtures use production `validate_chain`; do not weaken loader/closure checks to accept an incomplete image. Required facts outside current class limits produce `UNSUPPORTED`.
7. A sample-derived/hypothetical endpoint or unchecked relation cannot be exported or reported as verified. A VERIFIED local lemma with unverified predecessor facts retains conditional dependency status. A hidden-state counterexample cannot be counted as a byte counterexample.
8. Split fixtures cover all parent byte values exactly, including the threshold and endpoints. A parent cannot close with an unfinished/failed child. Child facts carry child-context hashes.
9. Rounding residue/sign enumeration agrees with direct exact arithmetic for all integers in a bounded fixture spanning at least [-1024,1024], and specifically at +/-128 and +/-384; empty ranges are rejected. Integer gcd/congruence assumptions and no-overflow conditions are stored.
10. Solver parsing distinguishes property, unwinding and safety failures. Timeout, output truncation and incomplete trace fixtures remain inconclusive. Deadline termination writes all remaining scheduled probe IDs and all resource data already collected.
11. The report includes all scheduled outcomes, actual runtime/memory/storage, planned versus executed checking, missing metrics, and hypotheses receiving no support. It must not replace the old 5/20 or P3 results with these selected development cases.

Final `report.md` must answer, per image and competitor: (a) whether a concrete violation was observed and over which domain; (b) singleton gaps; (c) fixed-witness slack concentrations; (d) attained versus hypothetical intervention gains; (e) formal representability and unverified dependencies; (f) additional checking/storage/memory burden; and (g) the matching decision-table rows with supporting run IDs. If the evidence does not select a cause, state **bottleneck unresolved within this diagnostic budget**. Do not fill that gap with an architectural recommendation.
