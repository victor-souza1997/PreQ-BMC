# GPT-6 Sol — Round 3 final plan

## Assigned trial: omission regression

**NOT RUN (environment refusal), not a pass.** `free -g` reported 19 GiB total and 9 GiB available, so the 8 GiB gate passed. Creation of the kickoff-authorized pilot directory was rejected by the managed approval layer after 5.2 s; therefore no copy was made, no mutation or aggregator command ran, and there were zero ESBMC/non-VERIFIED results.

Exact attempted command:

```sh
mkdir -p scratchpad/roundtable_pilots/gpt6-sol
```

I did complete the required independent read-only enumeration without calling the renderer. Layer 2 contains **32,445** `.c` harnesses; its summary contains 32,445 records and declares 32,445 harnesses; its shard contains **2,089/2,089 unique chains**, matching `chains_checked`, and **561,989** residual-fact citations. The corrected exact enumeration command took 4.2 s:

```sh
python -c 'import gzip,json; p="output/crown_chain_v2_20260924/image3652/chains_L2.jsonl.gz"; cs=[json.loads(x) for x in gzip.open(p,"rt")]; print(len(cs),len({c["certificate_id"] for c in cs}),sum(len(q.get("fact_ids",[])) for c in cs for s in c["steps"] for q in [s.get("residual_lower_sources")] if isinstance(q,dict)))' && python -c 'import json; d=json.load(open("output/crown_chain_v2_20260924/image3652/claude_hardened_l2/check_summary.json")); print(len(d["records"]),d["harness_count"],d["chains_checked"])'
```

Static inspection remains **ASSUMED**, not a trial result: aggregation checks declared counts and each present record, but does not compare `len(records)` with `harness_count` (`src/scripts/aggregate_crown_chain_v2.py:103-138`). The deletion trial remains mandatory.

## Surviving changes

| change | trust-base change | expected gain | hours | validation pilot | main risk |
|---|---|---|---:|---|---|
| Independent omission regression: delete one harness record, required chain, citation, and competitor | none | **GUESS:** prevents incomplete evidence from being accepted; current 14 cases mutate values, not omissions (`src/scripts/regress_crown_chain_v2_data_corruption.py:1-7`) | 2 | Baseline aggregates; every deletion must be refused. Enumeration must scan artifacts independently of renderer/checker | An “independent” enumerator may duplicate the same bug |
| Close arithmetic overflow with `--overflow-check` on ground obligations | none | **GUESS:** no speed gain and small overhead because affine chunks already simplify to zero solver goals; removes an unstated assumption | 2 | 300 largest validation affine chunks: 300 VERIFIED, zero timeout/failure; report time ratio | Hidden coefficient path exceeds the argued envelope |
| Bind deployed-loop rows to CSR rows with ESBMC, in identical term order | a new ESBMC lemma | **GUESS:** removes Python mapping from the headline trust claim; current equality is Python parsing/indexing (`src/verification/crown_chain_v2.py:144-251`) | 6 | Representative layer-0 and layer-4 rows VERIFIED within 60 s each; otherwise disclose Python as **ASSUMED** | Reordered sums become hard bit-vector equivalence queries |
| Publish reproducible evaluation tables and explicit trust contract | none | **GUESS:** reviewer credibility; **MEASURED:** 1/80 primary, 17 resource-inconclusive, 54 finder-inconclusive, 8 misclassified (`docs/ssv2026/roundtable/BRIEFING.md:124-136`) | 8–10 | Rebuild tables from manifests; denominators/statuses must match, and VIM3 repetitions report dispersion | Board energy measurement or rerun misses deadline |
| Use 8k affine cap only as memory safety; release the rendered certificate before workers | none | **MEASURED:** observed peak/job 1,090→404 MiB, but estimated +0.8 CPU-h from extra harnesses (Opus Turn 8) | 2 | After rerun: 25k versus 8k at four jobs; go only if all VERIFIED, no guard abort, and wall penalty ≤10% | Parent memory persists or extra parsing outweighs safety |

## Rejected proposals

- Solver sweep: DROP; 15,792/15,792 affine chunks have zero remaining solver goals.
- Result cache: DROP; only 19/58,695 reusable records (0.032%), none affine.
- Alpha slopes and input/ReLU splitting: FUTURE WORK; no complete validation pilot fits before October 2.
- Compiled generic checker: DROP for the headline; it substitutes compiler/runtime trust for ESBMC.
- Affine expression-shape rewrite and >4 jobs: FUTURE WORK; no safe full validation run while the disclosed rerun occupies the machine.
- QVIP/EQV as final judges: DROP under constraint 1; discuss as related work or future untrusted finders.

## Q3: minimum convincing evaluation

Report the immutable pre-registered **1/80** primary with a named 95% binomial interval and the full outcome taxonomy. Put the disclosed resource-only rerun in an adjacent, separately labelled column—never as replacement. Include float/deployed accuracy (91.32%/91.35%), worst-class 44.7%, exact ε=1 property, model bytes, matched VIM3 latency/energy/EDP, and verifier wall/CPU/RSS distributions. Separate **44,251 ground** from **14,444 solver-backed** obligations. Baselines are monolithic ESBMC, IBP, current CROWN, and published deployed-C/GTSRB work; do not transplant QVIP/EQV speedups to this model.

Three strongest attacks and answers:

1. **“1/80 at ε=1 is too weak.”** Answer: scope the contribution to bit-precise deployed-C certification, report the interval and every negative outcome, and make no broad robustness claim.
2. **“ESBMC is mostly a calculator; Python is trusted.”** Answer: disclose the ground/symbolic split and trust table; add overflow and binding lemmas plus omission regressions, or explicitly retain and name the Python assumption.
3. **“Resource aborts/reruns cherry-pick.”** Answer: the primary remains unchanged; SIGTERM is UNKNOWN, reruns are separate, all method tuning uses validation only, and every VERIFIED requires all ESBMC obligations successful.
