## 1. Solver choice: settled, DROP everywhere before the deadline

Both GPT turns retracted it; I agree. Any solver work is capped at relu_chunk's 0.76 of 13.9 h
(5.5%, MEASURED), so even Sena's fastest solver (Yices, Turn 5) gains at most 5%.

## 2. Chunk size: I retract my Turn 4 proposal 1(a) as a speed lever

GPT-6 is right that "lower the cap so more jobs fit" is not a wall-time forecast. New evidence,
MEASURED on all 15,792 affine records of image 3652:

| Terms per chunk | Chunks | µs per term |
|---|---|---|
| 4k–8k | 2,262 | 177 |
| 8k–16k | 1,452 | 178 |
| 16k–25k | 6,229 | 187 |

- **Cost is linear above about 4,000 terms.** Splitting a chunk saves no CPU.
- **An 8k cap adds 18,672 harnesses**, at about 0.15 s fixed cost each (the median for chunks of
  at most 100 terms). That is +0.8 CPU-h, 8% of affine time.
- **Memory:** peak per job falls from 1,090 to 404 MiB (GPT-5.6's number, confirmed).
- **Verdict:** keep the 8k cap as a memory-safety lever (fewer guard aborts, no host freezes),
  not in the speed column.

In GPT-6's citation (`check_crown_chain_v2.py:42-65`), `certificate` stays referenced while
`benchmark` runs, so the parent may hold its 2.7 GB peak (MEASURED) through every job (GUESS).
Releasing it is a 1 h change, neutral for soundness.

**Affine cost scales with Σ multiply terms** (layer 2 alone: 5.84 of 10.09 h). The real lever is
fewer terms, which is finder work for later. The shape pilot keeps the same literals, hence the
same parse volume (GUESS: at most 1.5×).

## 4. Schedule: the binding constraint is machine time, not engineer hours

Both turns budget 25 engineer-hours. Neither accounts for two facts:

- **Edits to `src/` wait for the rerun.** PROVED (the code): the driver aborts if any file under
  `src/` differs from its pin, including a new file
  (`run_crown_chain_v2_protocol.py:76-82`, `:185-198`). The disclosed rerun therefore blocks every
  `src/` edit in this checkout until it finishes. Post-freeze work goes in a separate git worktree.
- **The rerun takes days.** MEASURED:
  - Test image 005 took 11.8 CPU-h (3.4 h wall at 4 jobs); 015 took 21.6 CPU-h.
  - 12 of the 16 pending images still need layers 2–4, which hold 9.8 of 10.1 affine hours on
    image 3652.
  - GUESS: 40–80 h of wall time, ending 2026-09-29 to 10-01 if it starts tonight. No validation
    pilot can run alongside it, for lack of memory.

A renderer pilot on a full validation image cannot safely finish by Oct 2, and the pinned code
fixes the primary result anyway. **Order (about 20 h):**
1. Omission regression: 2 h, seconds of machine time; it tests the pinned aggregator as it is.
2. Overflow closure (§5): 2 h.
3. Binding lemma pilot: 6 h cap, minutes of machine time.
4. Evaluation tables and VIM3: 8–10 h, on the board.
5. Speed: the measured cost model is the scalability evidence; renderer and slopes are future work.

## 5. Trust gaps

**A label fix first.** GPT-5.6 labels the Python binding "PROVED (code inspection)", and GPT-6
labels facts about the code "PROVED (code)". Under constraint 5, PROVED means an ESBMC verdict;
reading code gives ASSUMED or MEASURED. The paper's trust table must not inherit this.

**Overflow: narrower than my Turn 4 said.**
- `+` and `×` on `__int128` are exact modulo 2¹²⁸. A ground harness asserting `Σ w·c == L`, with
  L an exact in-range literal, can falsely VERIFY only if |Σ w·c| ≥ 2¹²⁷.
- The literals rule that out: at most 25,000 terms, |w| ≤ 1,349 (MEASURED from the deployed C),
  and |c| ≤ 2³¹, so |Σ w·c| < 2⁵⁷. Floor and residual act on g after it is exact.
- **Open point:** ESBMC asserts |c| ≤ 2³¹ only on ReLU outputs (`crown_chain_v2.py:877-883`).
  Check that every affine input coefficient is such an output or a ±1 margin.
- **Closure, decided by ESBMC:** after the rerun, rerun only the VERIFIED images' ground
  harnesses with `--overflow-check`.

**Binding.** GPT-6 found `EXACT_ORDERED_MATCH` (`crown_chain_v2.py:144-250`): the CSR row is in
deployed-loop order. Render both sums of the ESBMC binding harness in that order, so that symex
yields the same term and the simplifier may discharge it without SAT (GUESS); a reordered sum is a
hard adder-equivalence query. Pilot layer 0 (75-term rows), then layer 4 (768).

**Omission.** Agreed; GPT-6's independent enumeration must not reuse the renderer's loops.

## Revised top 3

1. Omission regression, and overflow closure on the existing VERIFIED images (4 h).
2. The binding-lemma pilot in deployed-loop order, capped at 6 h; otherwise a stated trusted
   assumption.
3. Evaluation, with the measured cost model as the scalability result. Run the rerun first and
   develop in a worktree; speed work and slopes are future work.
