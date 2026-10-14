# Kickoff: how each chat takes part

The owner pastes the block for the current round into each chat, filling in `<you>` with one of
`gpt6-sol`, `gpt56-sol` or `opus`.

## Rules for every round (all participants)

1. Read `docs/ssv2026/roundtable/BRIEFING.md` in full and `docs/ssv2026/roundtable/THREAD.md`.
   THREAD.md holds every finished round; nothing of the current round is in it yet.
2. Your role is the row with your name in the briefing's participant table. Keep it every round.
3. **Blind within a round.** Write your turn to `docs/ssv2026/roundtable/rounds/R<N>/<you>.md`.
   Do not open or list any other file in `rounds/R<N>/`. Files from earlier rounds are already
   merged into THREAD.md.
4. **Do not edit** THREAD.md, BRIEFING.md or any file other than your own round file. Do not edit
   `src/` (the code freeze runs until Mon 2026-09-28 09:00). Do not run ESBMC, tests, training, or
   anything that uses much memory or disk: a pre-registered test run is using this machine.
5. Obey the hard constraints in §2 of the briefing. Label claims PROVED, MEASURED, ASSUMED or GUESS.
   Cite file:line for anything you read in the repository.
6. When your file is written, reply in the chat with one line: `R<N> done: <path>, <word count>`.

## Round 1: independent proposals (paste this)

> Round-table Round 1. You are `<you>`. Follow `docs/ssv2026/roundtable/KICKOFF.md` (rules and
> Round 1 section). Write `docs/ssv2026/roundtable/rounds/R1/<you>.md`.

Task: from your role, answer the agenda (Q1 and Q2 first, then Q3 and Q4). Give at most 5
proposals, ranked, each with the fields in §7 of the briefing. Use the NotebookLM literature pack
(Turn R0 in THREAD.md) for citations; mark any other citation "(verify)". Under 900 words.

## Round 2: cross-attack (paste this)

> Round-table Round 2. You are `<you>`. Follow `docs/ssv2026/roundtable/KICKOFF.md` (rules and
> Round 2 section). Write `docs/ssv2026/roundtable/rounds/R2/<you>.md`.

**Round 2 is sequential, not blind.** The order is `gpt56-sol`, then `gpt6-sol`, then `opus`. Read
every file that already exists in `rounds/R2/` and answer it by name; this overrides rule 3.

Task: Round 1 mostly agreed. This round settles the points where Round 1 turns conflict or where
one turn ignored another. Read Turns 2–4 and the NotebookLM citation check. For each point below,
quote the turn you disagree with by name, give your answer, and back it with file:line, a number
from `output/crown_chain_v2_20260924/image3652/claude_hardened_l*/check_summary.json`, or a
citation. "I agree" without new evidence is not an answer. Under 800 words.

1. **Solver choice for affine chunks.** GPT-6 and GPT-5.6 propose a Z3/Bitwuzla/Boolector
   comparison. Opus measured 0 s of solver time on all 15,792 affine chunks: every one is
   `0 remaining after simplification`. Check it in the logs or summaries. If it holds, drop solver
   choice from the affine pilot, or say why not.
2. **Chunk size and job count.** GPT-5.6: do not raise jobs. Opus: lower `max_affine_terms` so each
   job needs less memory and more jobs fit. Which one limits wall time on this 20 GB machine, and
   what cap should the pilot test?
3. **Result cache (GPT-6, proposal 3).** It adds hashing and stored logs to the trust base. Estimate
   the hit rate from the image 3652 harnesses (are byte-identical harnesses present at all?). KEEP
   or DROP.
4. **Slope optimisation first (GPT-6, GPT-5.6) versus speed first (Opus).** The code freeze ends on
   Mon 2026-09-28 and the paper is due on 2026-10-04. Round 1 estimates add up to more than 60 hours.
   Assume 25 engineering hours in total. Which changes fit, and in what order? A change that cannot
   be piloted on validation before 2026-10-02 goes to future work.
5. **Trust gaps no Codex turn addressed (Opus, proposals 3–5):** the Python binding of weights to
   the deployed C, the overflow argument, and a regression for omitted obligations. For each, say
   whether it must be in the paper as a stated assumption, closed before submission, or is not a
   gap. Check at least one of them in the code.

End with your revised top 3, in at most 5 lines.

## Round 3: trials, then the final plan (paste this)

> Round-table Round 3. You are `<you>`. Follow `docs/ssv2026/roundtable/KICKOFF.md` (rules and
> Round 3 section). Write `docs/ssv2026/roundtable/rounds/R3/<you>.md`.

**Round 3 runs real trials. For this round, it overrides rule 4 as follows:**

- **Where:** write code and outputs only under `scratchpad/roundtable_pilots/<you>/`. You may import
  anything from `src/`, but never edit, add or delete a file there: the protocol pins every file
  under `src/`.
- **Data:** validation image 3652 only (`output/crown_chain_v2_20260924/image3652/`). Read it; to
  change anything, work on a copy under your pilot folder.
- **Machine:** at most 1 ESBMC process at a time, with `--memlimit 2g`, and at most 45 minutes of
  run time in total. Before each run, check `free -g`; if the available column is below 8, wait.
- **Report what ran:** the exact command, the counts, the times, and every non-VERIFIED result.
  A trial that fails or cannot be finished is a result: say so.

Do your assigned trial first. Then, if time allows, try one idea from the NotebookLM speed-up
answers (the R3 NotebookLM turn in THREAD.md), on a small scale.

| Who | Assigned trial | Pass criterion |
|---|---|---|
| `gpt56-sol` | **Overflow.** Re-run ESBMC with `--overflow-check` on the 300 affine_chunk harnesses of 3652 with the most `multiply_terms`, using the command recorded in their `check_summary.json` records plus that flag | All VERIFIED; report the time ratio against the recorded `elapsed_seconds` |
| `gpt6-sol` | **Omission regression.** On a copy of one 3652 layer's check folder, delete one harness record, one chain and one citation (separately); run the aggregator (`src/scripts/aggregate_crown_chain_v2.py`) on each copy. Enumerate the expected obligations independently, without calling the renderer | The aggregator refuses every deletion; an unmodified copy passes |
| `opus` | **Binding lemma.** An ESBMC harness showing that the deployed layer-0 loop and the certificate's weight rows give the same sum for any input in range; then layer 4 | VERIFIED within 60 s per row, or an explicit TIMEOUT that is reported |

Then, with the trial results in hand, give one table of at most 5 changes, with these columns:

| Column | Content |
|---|---|
| change | what is changed |
| trust-base change | none, a new ESBMC lemma, or a new trusted component |
| expected gain | MEASURED or GUESS, with its source |
| hours | implementation cost |
| validation pilot | the pilot, with its go/no-go threshold |
| main risk | what is most likely to break it |

Then list the proposals you now reject, one line each with the reason. Then address Q3: the minimum
convincing evaluation, and the three strongest reviewer attacks, each with our answer. Under 900
words, not counting the trial commands.

Round 3 is blind again: do not open other files in `rounds/R3/` or other pilot folders.
