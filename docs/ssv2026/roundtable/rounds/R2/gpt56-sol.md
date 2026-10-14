# GPT-5.6 Sol — Round 2 cross-attack

## 1. Solver choice

> GPT-5.6 Sol: “Z3 versus Bitwuzla.”

I retract that part of my proposal. **MEASURED:** I scanned the recorded stdout/stderr paths in all five image-3652 summaries: 15,792/15,792 `affine_chunk` records say `0 remaining after simplification`. Thus solver selection cannot reduce affine time. Opus’s decomposition—6.1 h front end, 2.3 h symex, 0 solver—is consistent with the renderer: every operand is a literal and the asserted result is literal (`src/verification/crown_chain_v2.py:744-805`). Drop solver choice; retain expression-shape and chunk-size tests. Sena et al.’s solver comparison is irrelevant here because it concerns queries that reach SMT (NotebookLM check, `docs/ssv2026/roundtable/THREAD.md:338-342`).

## 2. Chunk cap and jobs

> GPT-5.6 Sol: “Do not increase jobs.”

That is too categorical; Opus is right that cap and concurrency must be tested jointly. **MEASURED:** among the 15,792 affine records, the 8,111 with at most 8,000 terms peaked at 403.9 MiB/job (99th percentile 331.6 MiB), whereas the complete set peaked at 1,090.1 MiB. The current default is 25,000 terms (`src/scripts/check_crown_chain_v2.py:37`). At 25k, memory limits job count; after shrinking, repeated front-end work and disk I/O may limit wall time.

Pilot an **8,000-term cap** first, with 4/6/8 jobs on one fixed validation shard. **GUESS:** 8k makes 6–8 jobs memory-feasible, but only measured wall time decides. Go only with all VERIFIED, no low-memory abort, MemAvailable never below 6 GiB, and at least 1.25x wall improvement. Do not deploy a higher job count independently of the cap.

## 3. Result cache

> GPT-6 Sol: “Content-addressed reuse of identical ESBMC obligations.”

**DROP. MEASURED:** of 58,695 harnesses, 58,676 hashes are unique: 19 reusable occurrences, a 0.0324% hit rate. All six duplicate groups are shared lemmas; the 15,792 affine harnesses have **zero** duplicates. Avoiding the repeats saves only 3.56 of 50,123 recorded ESBMC seconds (0.0071%). The aggregator already re-hashes each harness (`src/scripts/aggregate_crown_chain_v2.py:131-137`), but persistent verdict reuse would add key/log integrity assumptions for no material gain.

## 4. Twenty-five-hour schedule

> GPT-5.6 Sol: optimized slopes are the “highest coverage-per-hour option.”

Not before this deadline. Opus is correct that speed comes first: each new branch otherwise nearly replicates a four-hour check. My own slope estimate is 16–24 h (`docs/ssv2026/roundtable/THREAD.md:138-144`), leaving no safe validation window by 2026-10-02.

Order and budget: (1) 8 h for the 8k cap plus read-only-loop/scalar renderer and validation pilot; (2) 3 h for overflow and omission checks; (3) a hard-capped 6 h deployment-binding lemma pilot; (4) 8 h for reproducible paper tables, trust contract and board measurements. Total 25 h. **ASSUMED:** those estimates hold. Alpha slopes and all splitting move to future work; NotebookLM confirms alpha-CROWN gains on other MNIST/CIFAR models, not this integer GTSRB system (`docs/ssv2026/roundtable/THREAD.md:299-302`).

## 5. Trust gaps

> Opus: “Chain weights = deployed C … Python … ASSUMED”; “No `__int128` wrap … ASSUMED”; and “An omission regression.”

- **Weight binding — close before submission.** **PROVED (code inspection):** Python hashes the C, parses loop geometry, reconstructs every convolution row, and checks ordered indices, weights and biases (`src/verification/crown_chain_v2.py:144-251`). That is strong but remains a trusted Python interpretation. Timebox Opus’s ESBMC row-equivalence lemma at 6 h; if it fails, state the Python binding explicitly as an **ASSUMED trusted component**, not as ESBMC-proved.
- **Overflow — close before submission.** **PROVED (code inspection):** `paper-z3` does not add `--overflow-check`; only safety/overflow profiles do (`src/verification/esbmc.py:196-246`). ReLU instances enforce coefficient envelopes (`src/verification/crown_chain_v2.py:877-883`), but affine ground sums lack an explicit 128-bit headroom assertion. Add constant envelope assertions or run the overflow profile; require VERIFIED.
- **Omitted obligations — close before submission.** The aggregator already checks authoritative chain counts, unique layers, all-VERIFIED counts and harness hashes (`src/scripts/aggregate_crown_chain_v2.py:80-87`, `src/scripts/aggregate_crown_chain_v2.py:95-137`). **PROVED (code inspection):** the existing regression mutates accepted data (`src/scripts/regress_crown_chain_v2_data_corruption.py:1-7`), not omissions. Add drop-chain, drop-harness, drop-fact/citation and drop-competitor cases; each must fail closed.

## Revised top 3

1. 8k affine chunks + loop/scalar renderer, then 4/6/8-job validation sweep — 8 h.
2. Overflow and omission checks, plus timeboxed ESBMC deployment-binding lemma — 9 h.
3. Reproducible evaluation/trust tables and VIM3 measurements — 8 h; cache, slopes and splitting are out.
