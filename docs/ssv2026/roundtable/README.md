# Round table

Three chat models and a literature assistant discuss the project over three rounds. The owner
triggers every turn by hand and decides; nothing runs by script except merging.

| File | Purpose |
|---|---|
| [BRIEFING.md](BRIEFING.md) | Self-contained project context, constraints, measured numbers, agenda, roles and repository map. |
| [KICKOFF.md](KICKOFF.md) | The rules and the per-round text the owner pastes into each chat. |
| [THREAD.md](THREAD.md) | Finished rounds only. Each turn is `## Turn N — <participant> (R<round>, <date>)`. |
| `rounds/R<N>/<who>.md` | Each participant's turn for round N, written blind, before merging. |
| `SYNTHESIS.md` | The ranked plan after Round 3; the owner decides. |
| `roundtable.py` | Builds NotebookLM prompts, appends pasted replies, merges rounds. |

```bash
alias rt='python3 docs/ssv2026/roundtable/roundtable.py'
```

## Participants

Roles stay fixed across rounds, so each model argues from one angle.

| File name | Chat | Role |
|---|---|---|
| `gpt6-sol` | Codex chat, GPT-6 Sol | Architect: largest structural gain (finder, certificate format, checker design) |
| `gpt56-sol` | Codex chat, GPT-5.6 Sol | Engineer: grounds numbers in code and timing data, cites file:line, estimates hours |
| `opus` | Claude Code chat, Opus 5.5 | Soundness Auditor: what each VERIFIED rests on, trust-base changes; also merges rounds and writes the synthesis |
| `notebooklm` | NotebookLM | Librarian: answers only from the ~90 papers, with quotes; no code |

The Opus chat is the session that built the pipeline and the briefing, so it is the least
independent participant. Its Round 1 file is written without reading the other R1 files, like
everyone else's. Weigh its synthesis against the other two R3 tables, not above them.

## Method

| Step | Who | Reads | Writes | Limit |
|---|---|---|---|---|
| R0 literature pack | NotebookLM | its prompt (`rt prompt notebooklm --round 0`) | pasted into THREAD.md with `rt add notebooklm --round 0` | 150 words per answer |
| R1 independent proposals | the three chats | BRIEFING, THREAD (Turn 0 and R0) | `rounds/R1/<who>.md` | 900 words |
| merge R1 | owner or Opus | — | `rt merge 1` | — |
| R2 citation check | NotebookLM | a list of R1's citations, written by Opus | `rt add notebooklm --round 2` | 150 words per claim |
| R2 cross-attack | the three chats | THREAD (R0, R1, citation check) | `rounds/R2/<who>.md` | 800 words |
| merge R2 | owner or Opus | — | `rt merge 2` | — |
| R3 literature | NotebookLM | speed-up and novelty questions, written by Opus | `rt add notebooklm --round 3` | 150 words per answer |
| R3 trials and final plan | the three chats | THREAD (everything); runs one assigned trial in `scratchpad/roundtable_pilots/<who>/` | `rounds/R3/<who>.md` | 900 words |
| merge R3, synthesis | Opus | THREAD | `rt merge 3`, then `SYNTHESIS.md` | — |

Why this shape:

- **Blind rounds.** Each chat writes its own file and must not open the others' files for the same
  round. Merging happens only when all three exist, so agreement in R1 is evidence rather than
  anchoring.
- **Literature before proposals.** R0 gives quoted sources, which cuts invented citations; the R2
  check catches the rest before anyone builds on them.
- **Attack before convergence.** Every R3 plan has already survived a soundness objection and a
  cost re-estimate.
- **Nothing is trusted until piloted.** A change is implemented only after the owner approves it
  in the synthesis. It is then piloted on validation images after the `src/` freeze, and kept only
  if it meets its own go/no-go threshold.

## Rules

- **The hard constraints in the briefing bind every round.** Any proposal that weakens a
  contract, adds a tolerance, or counts TIMEOUT or INCONCLUSIVE as proof is rejected in the
  synthesis, however large its gain.
- **Everything the chats read leaves this machine** (to OpenAI, Anthropic or Google). The briefing
  contains unpublished results but no credentials; keep it that way. NotebookLM gets no code.
- **Chats may read the repository but must not run anything heavy**, except the capped Round 3 trials in KICKOFF.md. A test run is using the
  machine, and it aborts jobs below 6 GiB free.
- **The test set is not a tuning set.** Test outputs may be read for timing; pilots use
  validation images only.
- **The `src/` code freeze holds until Mon 2026-09-28 about 09:00.**
