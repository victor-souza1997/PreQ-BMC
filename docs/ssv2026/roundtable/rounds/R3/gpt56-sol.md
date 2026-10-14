# GPT-5.6 Sol — Round 3 final plan

## Assigned trial: overflow

The runner selected the 300 `affine_chunk` records with greatest `multiply_terms` across image 3652's five summaries (24,998–25,000 terms). It invoked one ESBMC process at a time, checked `free -g` before every invocation, required at least 8 GiB available, changed the recorded `--memlimit` to `2g`, and appended `--overflow-check`.

Exact outer command:

```text
python scratchpad/roundtable_pilots/gpt56-sol/run_overflow_trial.py
```

Exact first ESBMC command (all 300 are recorded in `scratchpad/roundtable_pilots/gpt56-sol/selection.json`):

```text
/home/joao/esbmc-linux/bin/esbmc /home/joao/code/preqbmc/output/crown_chain_v2_20260924/image3652/claude_hardened_l2/harnesses/chain_00122/chain_L2_n535_lower_s4_affine_0.c --function main --unwind 2 --z3 --bv --timeout 60 --memlimit 2g --interval-analysis --interval-analysis-simplify --result-only --overflow-check
```

**PROVED:** 300/300 returned VERIFICATION SUCCESSFUL; zero FAILED, TIMEOUT, MEMOUT or UNKNOWN (`scratchpad/roundtable_pilots/gpt56-sol/summary.json:3-13`). **MEASURED:** overflow-check time was 1,281.94 s versus 1,463.93 s recorded, ratio 0.876; wall time was 1,282.57 s. The baseline ran concurrently, so the ratio is not evidence that overflow checking accelerates ESBMC. Available memory was 8–9 GiB before every run.

NotebookLM follow-up command:

```text
python scratchpad/roundtable_pilots/gpt56-sol/run_interval_ablation.py
```

Exact no-interval variant:

```text
/home/joao/esbmc-linux/bin/esbmc /home/joao/code/preqbmc/output/crown_chain_v2_20260924/image3652/claude_hardened_l2/harnesses/chain_00122/chain_L2_n535_lower_s4_affine_0.c --function main --unwind 2 --z3 --bv --timeout 60 --memlimit 2g --result-only --overflow-check
```

**MEASURED:** with interval analysis, the first large chunk VERIFIED in 3.93 s. Without both interval flags it exhausted the 2 GiB limit in 16.78 s, returned -6, and reported `ERROR: Out of memory` and `SMT solver failed`; the ablation stopped after this non-VERIFIED result (`scratchpad/roundtable_pilots/gpt56-sol/interval_ablation.json:3-35`).

## Changes that survived Round 2

| change | trust-base change | expected gain | hours | validation pilot | main risk |
|---|---|---|---:|---|---|
| Enable `--overflow-check` for ground affine obligations on every reported VERIFIED image | none | **PROVED:** largest 300 all VERIFIED; **MEASURED:** 0.876 time ratio | 2 | Run all affine chunks of validation 3652; go only at 100% VERIFIED, no resource abort, and ≤1.25× time | Another coefficient path or harness kind overflows outside the sampled 300 |
| Add independently enumerated omission regressions: remove a harness record, chain, citation and competitor | none | **GUESS:** closes an omission class absent from the value-corruption suite, whose stated scope is changed accepted data (`src/scripts/regress_crown_chain_v2_data_corruption.py:1-7`) | 2 | Clean copy aggregates; every deletion is refused before VERIFIED | “Independent” enumeration accidentally shares renderer logic |
| Add a model-level ESBMC lemma binding deployed loop rows to certificate CSR rows, in deployment order | new ESBMC lemma | **GUESS:** removes the largest declared Python assumption; no speed gain | 6 cap | Layer 0 then layer 4; go if representative rows VERIFY within 60 s, then cover all row schemas | Reordered `__int128` sums create a hard equivalence query |
| Retain ESBMC interval analysis and simplification in the paper profile | none | **MEASURED:** one 25k-term ablation changed VERIFIED/3.93 s to OOM/16.78 s | 0 | Existing 300-run overflow trial uses both flags; reject any profile whose matched pilot is non-VERIFIED | Evidence is one deliberately small ablation |
| Produce the paper tables and explicit trust contract, including VIM3 measurements | none; exposes existing trusted components | **GUESS:** improves reviewer confidence; no certified-rate gain | 8–10 | Regenerate all tables from manifests; repeated board runs report dispersion and identical workloads | Deadline or board access leaves energy data incomplete |

The current deployment binding remains **ASSUMED** until the lemma succeeds: Python parses C arrays and reconstructs convolution/dense rows (`src/verification/crown_chain_v2.py:144-250`). The aggregator already checks authoritative chain counts, all-VERIFIED status, hashes and complete layer coverage (`src/scripts/aggregate_crown_chain_v2.py:80-155`).

## Rejected proposals

- Solver sweep: **MEASURED** 15,792/15,792 affine chunks simplify to zero remaining claims; SMT choice cannot reduce affine time.
- Persistent result cache: **MEASURED** only 19/58,695 repeat occurrences, all shared lemmas; zero affine duplicates.
- 8k chunks as a speed change: **MEASURED** Round 2 analysis predicts about +0.8 CPU-h; retain only as a future memory contingency.
- Loop/scalar renderer: **GUESS** at most 1.5× and no safe full-certificate pilot window before submission.
- Alpha slopes and input/ReLU splitting: 16–30 h implementation plus multiplied checking cost; future work.
- More than four jobs: **MEASURED** 12 jobs achieved 1.6× and caused resource aborts (`experiments/ssv2026_test_protocol.json:311-333`).
- Native generic checker: adds compiler/runtime trust and conflicts with ESBMC as final judge.

## Q3: minimum convincing evaluation

Report all 80 preregistered outcomes: **MEASURED** primary 1/80 with a binomial interval, category counts, and every non-VERIFIED image in the denominator, exactly as registered (`experiments/ssv2026_test_protocol.json:98-108`). Put the disclosed resource-only rerun beside it and never replace the primary. Report float and deployed-C accuracy, model bytes, matched VIM3 latency/energy/EDP, and verifier wall time, CPU time, RSS and counts split into 44,251 ground and 14,444 solver-backed obligations. Include monolithic ESBMC and IBP failures as baselines, ε=1, worst-class accuracy, sample-size limits, and the complete trust table.

Three strongest attacks and answers:

1. **“ESBMC is only a calculator.”** Report the ground/symbolic split explicitly; symbolic box and ReLU obligations establish quantified facts, while ground checks validate certificate arithmetic.
2. **“Python decides deployment correspondence or can omit work.”** Close binding with the new lemma if it succeeds; otherwise label Python mapping **ASSUMED** and narrow the claim. Show omission-regression failures and existing hash/completeness checks.
3. **“1/80 is weak and aborts hide failures.”** Accept the low primary rate, give its interval, classify SIGTERM as inconclusive, and show the rerun separately. Claim robustness only for fully VERIFIED images.
