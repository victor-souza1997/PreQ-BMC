# P2: re-audit of the seven saved v2 exports under the P1 checker (2026-10-30)

Script: `tools/ssv2026/p2_reaudit.py`. Outputs: `output/p2_reaudit_20261030/` (external disk, not in git). Old shard summaries were read only, never modified.

## Procedure

For each export:

1. Load the full certificate with the corrected structural checks: layer traversal, chain-fact identity and clamp references.
2. Per shard, compare the assertions required by the independent coverage module with the labels in the old, hash-matched, VERIFIED harnesses.
3. Missing assertions that are exactly the zero-product affine columns (skipped by the old renderer) get supplementary harnesses. These are checked with the original ESBMC settings: `paper-z3`, 4 jobs, 60 s, 4 GB.
4. Recompute coverage over the old and supplementary records.

## Result (MEASURED)

| Export | Structural | Zero-product columns | Supplementary harnesses | Summed ESBMC s | Verdict |
|---|---|---:|---:|---:|---|
| test 04083 | PASS | 535 | 264 | 42.8 | ACCEPTED_WITH_SUPPLEMENT |
| test 06267 | PASS | 448 | 241 | 37.9 | ACCEPTED_WITH_SUPPLEMENT |
| validation 006 | PASS | 436 | 213 | 34.9 | ACCEPTED_WITH_SUPPLEMENT |
| validation 008 | PASS | 452 | 197 | 31.5 | ACCEPTED_WITH_SUPPLEMENT |
| validation 011 | PASS | 489 | 216 | 34.6 | ACCEPTED_WITH_SUPPLEMENT |
| validation 012 | PASS | 514 | 224 | 35.5 | ACCEPTED_WITH_SUPPLEMENT |
| validation 017 | PASS | 360 | 155 | 24.0 | ACCEPTED_WITH_SUPPLEMENT |
| **Total** | | | **1,510, all VERIFIED** | **241.2** | |

- **Missing assertions:** in every shard they were exactly the zero-product columns. There were 0 unexplained, 0 unexpected and 0 duplicated labels.
- **Coverage:** after the supplement it is COMPLETE in all 35 shards.

## How to report it

The certificates rely on the historical shard checks plus a later supplementary ESBMC pass for the zero-product columns. The trusted components listed in `P1_checker_closure.md` ("What remains trusted") still apply.
