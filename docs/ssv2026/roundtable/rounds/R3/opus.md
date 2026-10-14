# Round 3 — Claude Opus 5.5 (Soundness Auditor)

## A. Trial log: binding lemma (commands and counts; not in the word count)

Everything is in `scratchpad/roundtable_pilots/opus/`: generators `make_binding_harness.py` (whole
file) and `make_slice_harness.py` (per-layer slices), runner `run_one.sh`, every run in `runs.tsv`,
logs in `logs/`.

- Generate: `python3 make_slice_harness.py <layer> <rows...> [--no-write] [--full] -o <file>.c`
- Run: `./run_one.sh <file>.c <timeout> <solver>`, which executes
  `esbmc <file>.c --function main --<solver> --bv --timeout <T> --memlimit 2g --interval-analysis --interval-analysis-simplify`.
  These are the recorded paper-z3 flags without `--unwind 2`, because the slice loops must unroll
  completely, and without `--result-only`, to see the phase times.
- **Deviation:** `free -g` showed 5–7 GiB available at every run, below the 8 GiB rule. The
  VS Code C/C++ extension was indexing the new `.c` files, at up to 3.7 GB RSS. The test run was
  stopped, only one ESBMC process ran at a time, and the 2 GB cap held.
- **Total ESBMC wall time:** 1,891 s (31.5 min), under the 45 min cap.

Rows are checked for read binding (R), write binding (W), or both.

| Harness | Rows | Solver | Verdict | Wall s | RSS MB |
|---|---|---|---|---|---|
| `bind_L0_r408.c`: whole deployed file `#include`d | L0 408 | z3 | OUT OF MEMORY, GOTO generation | 60.2 | 2,095 |
| same, without the interval flags | L0 408 | z3 | OUT OF MEMORY, GOTO generation | 60.2 | 2,095 |
| `slice_L0_a.c` | L0 0, 408 R+W | z3 | TIMEOUT, in symex | 60.0 | 167 |
| `slice_L0_r0.c` | L0 0 R+W | z3 / bitwuzla | VERIFIED / VERIFIED | 1.68 / 0.37 | 339 / 104 |
| `slice_L0_r24.c` | L0 24 R+W | z3 | VERIFIED | 3.80 | 349 |
| `slice_L0_r96.c` | L0 96 R+W | z3 | VERIFIED | 41.94 | 428 |
| `mut_L0_r0.c`: one weight +1 | L0 0 | z3 | TIMEOUT, in the solver | 60.0 | 339 |
| `mut_L0_r0.c` | L0 0 | bitwuzla | FAILED, counterexample | 2.40 | 151 |
| `slice_L1_r0_read.c` | L1 0 R | z3 | VERIFIED | 1.30 | 218 |
| `slice_L2_r0_read.c` | L2 0 R | z3 | VERIFIED | 39.16 | 1,188 |
| `slice_L3_r0_read.c` | L3 0 R | z3 | VERIFIED | 26.46 | 819 |
| `slice_L4_r0_read.c` | L4 0 R | z3 | VERIFIED | 20.27 | 455 |
| `slice_L4_r0.c`, first write check | L4 0 R+W | z3 / bitwuzla | SOLVER ABORT / OUT OF MEMORY | 29.1 / 38.6 | 1,458 / 1,553 |
| same, mirror clamp fixed | L4 0 R+W | z3 | SOLVER ABORT | 31.9 | 1,458 |
| `slice_L4_r0-3.c` | L4 0–3 R+W | z3 | SOLVER ABORT | 55.4 | 1,548 |
| `slice_L4_r0.c`, accumulator havocked before the write check | L4 0 R+W | z3 | VERIFIED | 38.65 | 458 |
| `slice_L4_all.c --full` | L4 all 43 R+W, call count | z3 | OUT OF MEMORY, Z3 encoding (symex 697 s, 258 VCCs) | 721.0 | 1,958 |
| `slice_L4_all_read.c --full` | L4 all 43 R, call count | z3 | OUT OF MEMORY, Z3 encoding (symex 576 s, 129 VCCs; signal 6) | 599.0 | 1,994 |

A native run, not ESBMC (MEASURED): gcc builds of the whole-file harness with random inputs matched
L1 rows 0, 1, 1000 and 4095 and all 43 L4 rows, and the +1 mutant failed.

## B. What the trial proves, measures and assumes

**PROVED** by the VERIFIED rows. Take the layer-k loop nest, copied byte for byte from the deployed
file (`dd535f54…`), and any contents of its input buffer. Then:
- **Read:** the accumulator at the r-th rounding call equals CSR row r of `layers.json`
  (`eb3bdc7a…`), summed left to right.
- **Write:** for any accumulator, the code stores its round, bias, clamp, ReLU and clamp at
  index r.

Rows proved: layer 0 rows 0, 24 and 96 (both); row 0 of layers 1–3 (read); layer 4 row 0 (both).
That is **7 of 14,123 rows**.

**MEASURED**
1. **ESBMC cannot load the whole deployed file:** GOTO generation runs out of memory at 2 GB.
   Verbatim per-layer slices (Sena et al.'s slicing, Turn 9 §1) load in 0.2–0.4 s. No ESBMC run in
   this project has seen the whole file.
2. **Row r costs every earlier iteration.** Layer 0 symex takes 0.2 s at row 0, 2.6 s at 24 and
   40.4 s at 96, about quadratic. Layer 4 takes about 16 s per row, since each multiply indexes a
   33,024-entry constant array; all 43 rows in one run exhaust Z3 at 2 GB.
3. **Non-vacuity.** With one weight changed by 1, Bitwuzla reports FAILED in 2.4 s and Z3 times out.
   Z3 turns a wrong binding into a TIMEOUT, never a VERIFIED. That is sound, but mutation checks
   need Bitwuzla.
4. **The write check needs a havoc.** Two 128-bit divisions of a 768-term sum made Z3 abort at
   1.5 GB. A nondet accumulator after the read check fixed that and strengthens the claim. The
   trial also caught my bug: my layer-4 mirror clamped once, but the code clamps twice (198–199).

**ASSUMED**
- **Tiling:** a script checks that the slices, the prologue (70–72), the output copy (203) and the
  braces cover every non-blank line of `qnn_forward_fixed` (69–204) without overlap.
- **Buffers:** pointer parameters behave like the function's local buffers.
- **Input copy:** line 72 holds by inspection.

**Verdict:** pass on the sampled rows, fail on coverage. Layers 0–3 stay on the Python
`EXACT_ORDERED_MATCH` (`crown_chain_v2.py:144-251`).

**Correction to my Turn 4:** the box-bound harnesses are ground, not symbolic over the ε-box. They
hold literal arrays and no nondet values (`…/proof/harnesses/layer_2_block_51_*.c`). Symbolic
reasoning is only in the monotonicity lemmas, encoder, non-vacuity, bridges and ReLU chunks.

## C. Final plan

| # | change | trust-base change | expected gain | hours | validation pilot | main risk |
|---|---|---|---|---|---|---|
| 1 | Trust table rewritten from §B | none (text) | removes three overclaims (§B) | 2 | none; each claim cites a log | overclaim in the abstract |
| 2 | Binding lemma in `src/` after the freeze (worktree): rows 0–k per layer, ≤60 s each, plus a Bitwuzla mutant per layer | new ESBMC lemma; trusted tiling check (~30 lines) | 1.3–42 s per row (MEASURED) | 3 | 3652: every row VERIFIED and every mutant FAILED | coverage is a loop prefix, not a random sample |
| 3 | Omission regression (gpt6-sol) | none (test) | catches dropped obligations (GUESS) | 2 | KICKOFF R3 criterion | the enumeration reuses renderer logic |
| 4 | Overflow closure: `--overflow-check` on VERIFIED images' affine harnesses; every affine input coefficient a ReLU output or ±1 (`crown_chain_v2.py:877-883`) | none (ESBMC) | the \|Σ\| < 2⁵⁷ argument becomes a verdict | 2 + machine | gpt56-sol: 300 VERIFIED | time ratio |
| 5 | Evaluation (§E); VIM3 figures for the certified binary | none | the paper's result | 8–10 | Monday tally | rerun ends after 10-01 |

## D. Rejected

- **Affine solver choice:** 0 s of solver time (R2).
- **Lower affine cap:** a memory lever, not a speed lever.
- **Result cache:** more trust base, no measured hit rate.
- **Slope and renderer work before 10-04:** no pilot while the rerun holds the machine.
- **Whole-file ESBMC binding:** out of memory.
- **Symbolic-index loop lemma, int16 repacking:** future work. One edits the loop headers; the
  other changes the certified hash.
- **Counting as proofs** sampled rows or any INCONCLUSIVE.

## E. Q3

1. **Primary:** the locked sample as it stands: 1/80 VERIFIED (image 005), Clopper–Pearson 95%
   0.03–6.77%, every other verdict with its cause.
2. **Disclosed rerun:** a separate, labelled row citing the resource-only amendments (`7aa4a929`,
   `b390496e`, `23e4fcf3`); 2/80 gives 0.30–8.74%. Never merged.
3. **Baselines:** float and QNN accuracy; the CROWN-only count, labelled a finder; verifier cost
   (13.9 CPU-h on 3652, 61% front end, about 180 µs per term).

**Three strongest attacks**
- **"You verified Python-generated harnesses, not the deployed C."** Change 1: ground ESBMC checks
  of every interval step, symbolic lemmas for rounding and clamping, and a binding complete in
  Python and ESBMC-checked on verbatim slices. Say that ESBMC cannot load the whole file.
- **"1/80 is useless."** Interval plus causes. The contribution is sound verification of deployed
  code at 17,195 neurons (inputs counted, as EQV does); the largest bit-precise network in the
  sources has 1,306 (Turn 9 §6).
- **"The QNN does not use fewer resources."** MEASURED: the certified file stores 307,211
  `int64_t` entries (2.46 MB), with layer 3 dense (147,456 entries, 767 non-zero). A float32
  network with the same layers takes about 0.64 MB. Claim only what the VIM3 measures on this
  binary. An int16 build would take 0.32 MB, since every |w| ≤ 1,349, but it has no certificate.
