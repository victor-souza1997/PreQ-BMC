# Round table

Several models discuss the project in one shared, append-only thread. The owner decides.
Claude is called only for review or implementation, to save credits.

| File | Purpose |
|---|---|
| [BRIEFING.md](BRIEFING.md) | Self-contained project context, constraints, measured numbers and agenda. Every prompt starts with it. |
| [THREAD.md](THREAD.md) | The discussion. Each turn is `## Turn N — <participant> (<date>)`. |
| `roundtable.py` | Builds prompts and appends replies. |

Run from the repo root. `rt` is just shorthand:

```bash
alias rt='python3 docs/ssv2026/roundtable/roundtable.py'
```

## A typical round

```bash
# GPT (web, unlimited): the prompt goes to the Windows clipboard; paste it into chatgpt.com
rt prompt gpt
# ...copy GPT's whole answer, then:
rt add gpt

# DeepSeek (API, cheap): fully automatic
export DEEPSEEK_API_KEY=...        # from platform.deepseek.com
rt ask deepseek

# Another GPT turn that answers DeepSeek:
rt prompt gpt --task "Critique DeepSeek's Turn 2 proposals on Q1; rank them."
rt add gpt

# Codex (limited credits; read-only sandbox, can read the repo): use for code-level questions
rt ask codex --task "Inspect how affine_chunk harnesses are generated and say why they cost 2.3 s."

# Claude (uses your Claude credits): use sparingly, for final review
rt ask claude --task "Audit every proposal in the thread against the hard constraints; give the final ranked plan."
```

`--task` steers a single turn. Without it, the model continues the discussion. `--last N` sets
how many recent turns go into the prompt (default 6; Turn 0 is always included). The prompts are
saved in `.prompts/`.

## Rules

- **GPT web and DeepSeek see only the briefing and the thread, not the code.** Keep the briefing
  current when numbers change.
- **Everything sent to DeepSeek, OpenAI or Codex leaves this machine.** The briefing contains
  unpublished results but no credentials; keep it that way.
- **Nothing proposed in the thread is trusted until it has been implemented and measured** on
  validation images, and checked against the hard constraints in the briefing.
- **The `src/` code freeze holds until Mon 2026-09-28 about 09:00.** The test run pins every
  `src/**/*.py`.
