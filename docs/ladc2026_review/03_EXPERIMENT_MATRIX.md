# Experimental design for the revision

Concrete campaign design for T1, T2, T3, T11, T14, T15 and T17. All flags below were
verified against
[src/scripts/run_robustness_pipeline.py](../../src/scripts/run_robustness_pipeline.py)
in this working tree. Nothing here requires new code — the ablation axes are existing
flags and the aggregator already emits the columns the tables need.

## Fixed configuration (held constant across every arm)

| Setting | Value | Flag |
| --- | --- | --- |
| Error budget | `derived` (the sound mode) | `--error-budget derived` |
| Contract chaining | enforced | `--enforce-contract-chaining` |
| ESBMC profile | `paper-z3` for MNIST, `paper-fast` otherwise | `--esbmc-profile` |
| Per-query timeout | 300 s (1800 s in the T11 arm only) | `--esbmc-timeout-seconds` |
| Per-query memory | 20 GiB | `--esbmc-memlimit 20g` |
| MILP solver | CBC | `--solver cbc` |
| Solver jobs | 1, pinned | `--esbmc-jobs 1` |
| No-saturation obligation | **required** (T14) | `--require-formal-no-saturation` |

`--error-budget derived` is not the CLI default; the flag default is the unsound
`heuristic` slack, which can claim at most `harness-verified`. Every arm in this
campaign must set it explicitly. See the "Known gaps" section of the
[README](../../README.md).

## Ablation axes

| Axis | Values | Flag | Answers |
| --- | --- | --- | --- |
| Block width β | 0 (monolithic), 1, 2, 5, 10 | `--esbmc-layer-block-size N` | RQ3 / T1 |
| Relational cuts | on, off | `--margin-cuts on\|off` | T2 / §4.6 |
| E2E invariants | on, off | `--e2e-invariants` / `--no-e2e-invariants` | T2 / §4.4 |
| Timeout budget | 300 s, 1800 s | `--esbmc-timeout-seconds` | T11 |

Do **not** run the full cross-product. The cuts and invariants axes are only
meaningful at the configuration that reaches the output-layer proof, so:

- **Arm A (block sweep):** β ∈ {0, 1, 2, 5, 10} × cuts on × invariants on. 5 runs per region.
- **Arm B (cuts):** β = 2 × cuts ∈ {on, off} × invariants on. 1 additional run per region.
- **Arm C (invariants):** β = 2 × cuts on × invariants ∈ {on, off}. 1 additional run per region.
- **Arm D (budget):** the unresolved regions from Arm A at β = 2, timeout 1800 s.

Arms B and C reuse Arm A's β = 2 run as their shared control, so the marginal cost is
2 runs per region, not 4.

## Region set

Fix one region set for the whole revision and use it in every table. The current
paper mixes provenances, which is how the "19 vs 21 obligations" discrepancy arose.

| Cohort | Purpose | Regions |
| --- | --- | --- |
| MNIST `1blk_10`, sample 3, ε ∈ {0.25, 0.5, 1, 2, 4} | the nested radius frontier | 5 |
| MNIST `2blk_10_10`, sample 3, same radii | depth effect | 5 |
| MNIST `1blk_25`, ≥ 3 samples | width effect, matches the Quadapter cohort | ≥ 3 |
| Iris, Seeds | cross-tool overlap with CEG4N; also the datasets contribution 4 claims | per `experiments/preqbmc_reported_experiments.json` |

If Iris and Seeds are not run under this configuration, contribution 4 must be
narrowed (T16 / A.5). Running them is cheaper than defending the claim without them.

## Commands

Single region, monolithic arm:

```bash
python src/scripts/run_robustness_pipeline.py \
    --dataset mnist --arch 1blk_10 --sample-id 3 --eps 0.25 \
    --bit-lb 3 --bit-ub 40 \
    --error-budget derived --enforce-contract-chaining \
    --esbmc-layer-block-size 0 \
    --margin-cuts on \
    --esbmc-profile paper-z3 --esbmc-timeout-seconds 300 --esbmc-memlimit 20g \
    --solver cbc --verify-mode esbmc \
    --output-dir output/ladc_rq3_beta0/
```

Cuts-disabled arm (identical except two flags):

```bash
python src/scripts/run_robustness_pipeline.py \
    ... --esbmc-layer-block-size 2 --margin-cuts off \
    --output-dir output/ladc_rq3_nocuts/
```

Campaign form — derive a plan file from the existing no-block plan, which already
sets `esbmc_layer_block_size: 0`:

```bash
cp experiments/article_experiments_no_block.json experiments/ladc2026_rq3_block_sweep.json
# edit: output_root, aggregate_output_root, error_budget_mode=derived,
#       enforce_contract_chaining=true, require_formal_no_saturation=true,
#       one entry per beta value
preqbmc reproduce --config experiments/ladc2026_rq3_block_sweep.json
preqbmc aggregate --input-root output/ladc_rq3_runs \
                  --output-root output/ladc_rq3_results --plots
```

## Result tables to produce

The aggregator already emits the underlying columns; these are the paper-facing
projections.

### T1 — Table: block-width sweep (RQ3)

Primary metrics are per-query, not totals. Decomposition trades query count for
query size; total wall time may legitimately increase.

| β | Queries | Max query time (s) | **Max query peak RSS (MiB)** | Total ESBMC time (s) | Largest neurons/query | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (monolithic) | | | | | | |
| 1 | | | | | | |
| 2 | | | | | | |
| 5 | | | | | | |
| 10 | | | | | | |

Source columns: `number_of_esbmc_calls`, `max_esbmc_query_time_seconds`,
`max_esbmc_query_peak_memory_mib`, `total_esbmc_time_seconds`,
`largest_neurons_per_query`, `final_status` — all present in
`table_scalability.csv`.

A memout or timeout at β = 0 is a **result**, not a missing cell. Report it as the
outcome and say so in the caption.

### T2 — Table: refinement ablation

| Configuration | Regions certified | Margin-inconclusive | Layer-inconclusive | Cuts validated | Cuts used |
| --- | --- | --- | --- | --- | --- |
| β = 2, cuts on, invariants on (control) | | | | | |
| β = 2, cuts **off** | | | | | |
| β = 2, invariants **off** | | | | | |

If disabling cuts changes nothing, §4.6 is machinery without measured effect and the
paper should say that plainly rather than describing it as a contribution.

### T3 — Failure-mode table

One row per unresolved region. This is the table R1.9 asked for.

| Region (arch, sample, ε) | Final status | Stage | Cause class | Detail |
| --- | --- | --- | --- | --- |
| | `MARGIN_INCONCLUSIVE` | output proof | preimage precision | deflated target width at layer L |
| | `LAYER_INCONCLUSIVE` | layer contract | Cartesian over-approximation | hidden witness not replayable |
| | `PREIMAGE_DEFLATION_EMPTY` | candidate rejection | budget exceeds preimage | δ vs contract width |
| | `TIMEOUT` / `MEMOUT` | ESBMC query | solver cost | query size, budget consumed |
| | `SOURCE_PROPERTY_INCONCLUSIVE` | eligibility gate | **out of scope** | DeepPoly could not establish the float property |

Cause classes come from `final_status` plus `failure_reason`,
`esbmc_timeout_count`, `esbmc_memout_count` in `all_experiments.csv`; the status
constants are defined in
[experiment_summary.py:305-420](../../src/reports/experiment_summary.py#L305).

Keep the `SOURCE_PROPERTY_INCONCLUSIVE` rows visible but **excluded from the
certification denominator**, with the exclusion stated. Those regions are not ones
the method failed on; they are ones no method in this family can be asked about.

### T11 — Figure: timeout sensitivity

Cumulative regions certified (y) against per-query wall-clock budget (x), at
{60, 150, 300, 600, 1200, 1800} s. A flat curve past 300 s is the answer to R1.10 and
is much stronger than a prose assertion that more time would not help.

### T15 — Table: stage-attributed cost, and harmonized memory

Replace the single 415.77 s with the breakdown already recorded per run:
`preimage_time_seconds`, `bitwidth_search_time_seconds`,
`esbmc_contract_time_seconds`, `esbmc_no_saturation_time_seconds`,
`deployment_eval_time_seconds`, `refinement_time_seconds`.

For Table 2, report **process-tree peak RSS for all three tools**
(`esbmc_memory_measurement` is already `linux_procfs_process_tree_rss`), or keep both
columns and label them as measuring different things. Do not present max-query RSS
and process-tree RSS side by side as if they were the same metric.

### T17 — Appendix: per-region results

Every region × every tool × verdict × wall time × peak memory, no aggregation. With
n ≈ 10 this is both more informative and more defensible than any median.

## Recording discipline

- One configuration per campaign; do not merge runs from different error-budget modes
  into one table.
- Keep `guarantee_level` in every exported table. A `VERIFIED` ESBMC status with
  `guarantee_level: harness-verified` is **not** a certified region and must never be
  counted as one.
- Pin and report: ESBMC version, solver and mode, CBC version, Python version, gcc
  version, thread pinning. §5.1 promises these; make sure the artifact contains them.
- Archive raw logs alongside the normalized records, as §5.1 claims.
