# Round-table synthesis (after Round 3)

Sources: THREAD.md Turns 0–12 and the R3 pilot folders. Labels as in the briefing: **PROVED** means an
ESBMC verdict, **MEASURED** means a count or time from a log, **ASSUMED** means argued or checked
outside ESBMC. The owner decides; this ranks the options.

## 1. Bottom line

1. **No sound speed-up is ready before 10-04.**
   - Affine checking is 72% of the time, and all 15,792 affine chunks have zero goals left for the
     solver (R2). A solver choice cannot help.
   - The result cache would hit 19 of 58,695 records.
   - The literature's big speed-ups (QVIP, EQV: ILP instead of SMT) replace the judge, which
     constraint 1 forbids.
   - Report the cost as measured: 13.9 CPU-h on image 3652.
2. **No coverage fix is ready before 10-04.** Alpha slopes and input splitting each need 16–30 h
   plus a full validation run that does not fit while the rerun holds the machine. They are future
   work.
3. **The pre-deadline work is claims hygiene, not method changes.** Four checks, none of which
   edits `src/`, make the trust story defensible (§3).
4. **One framing problem needs a decision now:** the certified C does not use less memory than the
   float network (§5, D1).

## 2. What Round 3 measured

| Trial (owner) | Result | Label |
|---|---|---|
| Binding lemma (Opus) | Deployed layer loops match the certificate's CSR rows on **7 of 14,123 rows**. ESBMC cannot load the whole deployed file (out of memory at 2 GB). Cost of row r grows about as r², and one run over all 43 layer-4 rows drives Z3 out of memory at 2 GB. A +1 weight mutant: Bitwuzla FAILED in 2.4 s, Z3 TIMEOUT. | PROVED on sampled rows; full coverage infeasible (MEASURED) |
| Overflow check (GPT-5.6) | 300/300 largest affine chunks VERIFIED with `--overflow-check`. GPT-5.6 also reported that its chunk runs out of memory at 2 GB without interval analysis; that happens only when `--overflow-check` is on (§2.3). | PROVED for those 300 |
| Omission regression (GPT-6) | **Not run**: the chat's approval layer refused `mkdir`. It confirmed layer-2 counts by hand (32,445 harnesses, 2,089 chains). | none |
| Audit follow-ups (Opus, this synthesis) | §2.1 to §2.3 | see below |

### 2.1 Omission: a latent gap, not triggered

The aggregator (`src/scripts/aggregate_crown_chain_v2.py:103-138`) checks the following:
- chain counts against the shard files;
- `status_counts == {"VERIFIED": harness_count}`;
- the hash of each listed record.

It never checks `len(records) == harness_count`. Also, `harness_count` comes from the same checker
that renders the harnesses, so no independent obligation count exists. GPT-6 is right, and so is
GPT-5.6's narrower claim that chain counts and hashes are checked.

**MEASURED on every shard behind a VERIFIED result** (3652 layers 0–4; test images 005 and 015
layers 0–4): `harness_count` equals the number of records, the number of unique harness paths, and
the number of `.c` files on disk. The two aborted 015 layer-4 shards (1,056 and 23 records against
2,720) would be refused by the `status_counts` test.

So no reported VERIFIED is affected. The deletion regression is still needed to show that a
dropped obligation is refused.

### 2.2 Overflow: GPT-5.6's sample misses the int64 code

The affine chunks are ground `__int128` sums of literals, far below 2^127, so the check could hardly
fail there. The `int64_t` arithmetic sits elsewhere:
- `relu_gap64` in the `lower_upper` breakpoint asserts of relu chunks (`crown_chain_v2.py:893-898`);
- the product lemma (`:982`);
- the rounding lemma's int64 accumulator envelope (`:1002`).

I ran `--overflow-check` with GPT-5.6's command rule (`scratchpad/roundtable_pilots/opus/overflow_int64.py`):

| Selection (image 3652) | Result | Time with the flag vs without |
|---|---|---|
| All 29 shared lemmas (rounding, product, five ReLU classes) | **29/29 VERIFIED** | 4.5 s vs 5.5 s recorded |
| The 10 relu chunks with the most breakpoints (64–67 each) | **10/10 VERIFIED** | 391.3 s vs 59.0 s run the same way: **6.6×** |
| Mutant: `relu_gap64` with a = 2^50, outside the envelope | **FAILED**, "arithmetic overflow on mul" in `relu_gap64` | 0.19 s |

The mutant shows that the flag really checks this code under the paper flags. Without the flag,
the same mutant fails only at the outer assert, after the value has wrapped.

Available memory was 5.7–7.4 GiB during this audit, below the 8 GiB rule, because VS Code was
open. Only one ESBMC process ran at a time, at 2 GB.

The script selected 30 relu chunks. I stopped it after 10, on purpose, to stay within the ESBMC
time budget.

For the rounding lemma, the envelope must contain every deployed accumulator. **MEASURED** from the
certificate's CSR rows: max row |w|-sum × 2^15 is at most 2^29.6 (layers 2 and 4). That holds if
every layer input fits int16. The deployed code clamps to 16 bits before each later layer; layer 0
reads the ε-box image.

### 2.3 Interval analysis: in the trust base, but no sampled verdict needs it

Every recorded command passes `--interval-analysis --interval-analysis-simplify`. Per `esbmc --help`,
the first adds assumes to the program and the second simplifies asserts. An interval that is too
narrow would cut real executions, so every VERIFIED obtained with these flags trusts ESBMC's
abstract interpreter.

I reran sampled obligations of image 3652 with their recorded commands, minus the two flags, at
`--memlimit 2g`, one process at a time (`scratchpad/roundtable_pilots/opus/interval_ablation*.py`,
results in `interval_ablation.jsonl`):

| Selection (image 3652) | Without the interval flags | Time vs with the flags |
|---|---|---|
| All 29 shared lemmas | **29/29 VERIFIED** | 4.5 s vs 5.5 s recorded |
| The 10 relu chunks of §2.2 | **10/10 VERIFIED** | 63.4 s vs 59.0 s run the same way (1.08×) |
| 10 layer-2 affine chunks at size deciles (136 KB to 1.1 MB) | **10/10 VERIFIED** | 27.4 s vs 26.7 s recorded |
| GPT-5.6's chunk (`chain_00122/…_affine_0.c`, 904 KB) | **VERIFIED** | 3.8 s, 955 MB |
| The 16 symbolic box obligations: input encoding, non-vacuity, 10 monotonicity lemmas, 4 bridges | **16/16 VERIFIED** | 142.4 s vs 51.4 s recorded (2.8×); encoding 118 s vs 41 s |
| With `--overflow-check`: 29 lemmas and the 3 relu chunks with the most breakpoints | **32/32 VERIFIED** | lemmas 4.8 s vs 4.5 s; relu 115.5 s vs 124.3 s |
| With `--overflow-check`: GPT-5.6's chunk | **OUT OF MEMORY** at 2 GB | 17.7 s |

Native cross-check, outside ESBMC: an affine chunk makes no nondet call, so it has one execution
and gcc can run it. The shim makes `__ESBMC_assert` abort. The build uses
`-fsanitize=signed-integer-overflow -fno-sanitize-recover=all`. Selection: the 10 decile chunks plus
the 5 largest (up to 1.18 MB). Result: **15/15 PASS**. A probe confirms that this sanitizer catches
an `__int128` overflow ("cannot be represented in type '__int128'").

What this shows:
- **PROVED on 66 obligations:** their verdicts hold without ESBMC's interval analysis. This includes
  the int64 overflow verdicts (lemmas and relu chunks).
- **MEASURED:** the flags buy almost no speed on chain obligations (1.0–1.08×). The symbolic box
  lemmas run 2.8× slower without them, but they are small in absolute terms. So the flags are a
  trust cost with little return in this pipeline.
- **Still ASSUMED:** the remaining ~58,600 chain obligations and 673 box obligations ran only with
  the flags. The primary result trusts the interval analysis.
- GPT-5.6's 300 affine overflow verdicts need interval simplification to fit in memory. For those,
  the native run is corroboration, not a proof. The affine sums are `__int128` literals far below
  2^127, so an overflow there is not plausible.

## 3. Ranked plan

None of these edits `src/`. The rerun pins every `src/**/*.py` until it ends (40–80 h from
tonight), so any `src/` fix comes after the paper.

| # | Change | Trust-base change | Gain | Hours | Validation / go rule | Main risk |
|---|---|---|---|---|---|---|
| 1 | **Trust table in the paper:** 44,251 ground vs 14,444 solver-backed obligations; box harnesses are ground; the binding is a Python check (`EXACT_ORDERED_MATCH`) corroborated by ESBMC on sampled rows; ESBMC never loads the whole file; ESBMC's interval analysis is in the trust base, and no sampled verdict depends on it (§2.3) | none (text) | removes the overclaims all three chats flagged | 2 | every line cites a log | an abstract that still says "ESBMC verifies the deployed C" without qualification |
| 2 | **Artifact audit script** (tracked, outside `src/`): for each reported VERIFIED, recheck count = records = unique paths = `.c` files; plus a deletion test on a copy (drop one record, chain, citation or competitor; the aggregator or the script must refuse) | none; the script is extra evidence | closes §2.1 for reported results | 2–3 | clean copy passes; every deletion refused | the deletion shows the aggregator accepts a dropped record; then say so and let the script be the check |
| 3 | **Overflow closure as reported:** lemmas and a relu-chunk sample under `--overflow-check` (§2.2), plus the ESBMC-asserted envelope `\|a\| < 2^31` and int16 `z` | none | the no-overflow argument becomes partly a verdict | done; 0.5 to write | §2.2 | a full relu-chunk sweep may cost about 6.6× (the worst-case sample); do not run it before 10-04 |
| 4 | **Binding corroboration in the artifact:** move `make_slice_harness.py` to a tracked path; row 0 of each layer plus one Bitwuzla mutant per layer | none (corroboration only) | a reviewer can rerun it | 1 | 5 rows VERIFIED, 5 mutants FAILED | presenting it as full binding |
| 5 | **Evaluation and VIM3** (§4) | none | the paper's result | 8–10 | Monday tally; VIM3 runs with dispersion | rerun ends after 10-01 |

## 4. Q3: the paper

- **Primary result:** exactly as pre-registered. 1/80 VERIFIED (image 005), Clopper–Pearson 95%
  0.03–6.77%. Give all 80 outcomes by cause: 8 MISCLASSIFIED, 54 FINDER_INCONCLUSIVE, 17
  CHECK_INCONCLUSIVE (resource aborts, not solver verdicts).
- **Disclosed rerun:** a separate, labelled column that cites the resource-only amendments
  (`7aa4a929`, `b390496e`, `23e4fcf3`). Now 2 VERIFIED (005, 015), 5 aborted again, 11 pending;
  2/80 would give 0.30–8.74%. Never merged into the primary.
- **Baselines:**
  - float and deployed accuracy (91.32% and 91.35%), with the worst class at 44.7%;
  - monolithic ESBMC (no verdict after 22 min at 20 GB);
  - IBP (vacuous by block 4);
  - CROWN alone, labelled a finder;
  - verifier cost as CPU-h, wall time, RSS and per-kind time.
- **Related work:** the closest deployed-C work (QNNVerifier, Sena et al.) reaches small MLPs. The
  largest bit-precise network in the sources is EQV's 1,306 neurons; this one has 17,195, counting
  inputs the same way. VNN-COMP's GTSRB entries check models, not C.

**Strongest attacks, with answers:**
1. **"ESBMC checked Python-generated harnesses, not the deployed C."** Answer with the trust table
   (plan row 1), the sampled ESBMC binding, and a plain statement of what Python checks.
2. **"1/80 is useless."** Answer with the interval, the cause of every negative, and a narrow
   claim: sound certification of deployed C at a scale no source reaches. No broad robustness
   claim.
3. **"The QNN does not save resources."** See D1; this attack currently lands.
4. **"The rerun is cherry-picking."** The primary result is unchanged, the rerun is labelled and
   separate, the amendments are resource-only, and SIGTERM counts as inconclusive.

## 5. Decisions for the owner

- **D1, the resource claim.** The certified file stores 307,211 `int64_t` entries (2.46 MB). The
  depthwise layer is stored dense: 147,456 entries, 767 non-zero. The float32 network with the same
  layers takes about 0.64 MB, so on weight memory the certified QNN is about 3.8× larger.
  - **Recommend:** claim only what the VIM3 measures on the certified binary (latency, energy,
    integer-only execution), state the storage figure, and name an int16 build (0.32 MB, since
    every |w| ≤ 1,349) as future work.
  - An int16 build changes the deployment hash, needs a new certificate, and breaks the
    pre-registration pin.
- **D2, the overflow scope.** Accept §2.2 plus the asserted envelope, or run a full relu-chunk sweep
  after the rerun.
  - The sweep would cost up to about 15 CPU-h per image: 2.2 h × 6.6. That ratio comes from the
    worst chunks; typical chunks are unmeasured.
  - **Recommend** the former, stated as a sample plus an argument, not as a proof over every chunk.
- **D3, the aggregator fix.** Adding `len(records) == harness_count` and an independent obligation
  count touches pinned `src/`. **Recommend** the plan-row-2 script now and the `src/` fix after the
  paper.
- **D4, the interval analysis.** The protocol pins the two flags, so the primary result keeps them.
  - (a) Disclose §2.3 in the trust table.
  - (b) After the rerun, re-verify each reported VERIFIED image without the two flags. This can
    only turn a VERIFIED into a non-VERIFIED, never the reverse. Declare it in the amendment log
    before running it, and report it as a separate column.
  - Cost is estimated from the sample, not measured: about 15 CPU-h per image, about 4 h wall at
    4 jobs.
  - **Recommend** (a) now, and (b) only if the rerun ends by about 10-01. If (b) passes, the
    headline VERIFIED results no longer trust the interval analysis.

## 6. Rejected (consensus of the three chats, with reason)

| Proposal | Reason |
|---|---|
| Solver sweep | zero solver goals in affine chunks |
| Result cache | 0.03% hit rate; adds trust |
| Compiled generic checker | puts the compiler in the trust base |
| QVIP or EQV as judge | ESBMC must stay the judge |
| Lower affine cap as speed | costs about +0.8 CPU-h; memory contingency only |
| More than 4 jobs | 1.6× at 12 jobs, with resource aborts |
| Whole-file ESBMC binding | out of memory |
| Dropping the interval flags from the primary run | the protocol pins them; use D4(b) as a separate column |
| Alpha slopes, splitting, renderer rewrite, int16 repack | no validation window before 10-04 |
| Counting sampled rows or INCONCLUSIVE as proofs | violates constraint 1 |

## 7. Timeline

- **09-27, tonight:** relaunch the disclosed rerun with VS Code closed and memguard on.
- **09-28 to 09-30:** plan rows 1–4, all outside `src/`.
- **Rerun end, about 10-01:** tally, confidence intervals, and VIM3 on the VERIFIED list.
- **10-02 to 10-03:** write.
- **10-04:** submit.
