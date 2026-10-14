"""Three-round round table over BRIEFING.md and THREAD.md. Every participant is a chat the owner
triggers by hand (see KICKOFF.md); this script only builds NotebookLM prompts and merges rounds.

  prompt WHO --round R [--task T]
                          build a prompt, save it in .prompts/ and copy it to the Windows clipboard
                          (notebooklm: R0 literature pack; R2 citation check needs --task)
  add WHO --round R [--file F]
                          append a reply as the next turn (default: read the Windows clipboard)
  merge R                 append rounds/R<R>/{gpt6-sol,gpt56-sol,opus}.md to THREAD.md, in that order;
                          refuses until all three exist
"""
import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIEFING = HERE / "BRIEFING.md"
THREAD = HERE / "THREAD.md"
PROMPTS = HERE / ".prompts"
TURN = re.compile(r"^## Turn (\d+) — (.+?) \((?:R(\d+), )?", re.M)

# Roles live in BRIEFING.md; names here only label turns.
PARTICIPANTS = {
    "gpt6-sol": {"name": "GPT-6 Sol", "route": "clipboard"},
    "gpt56-sol": {"name": "GPT-5.6 Sol", "route": "clipboard"},
    "opus": {"name": "Claude Opus 5.5", "route": "clipboard"},
    "notebooklm": {"name": "NotebookLM", "route": "clipboard", "role": "the Librarian"},
    "moderator": {"name": "Moderator", "route": "clipboard", "role": "the Moderator"},
    "owner": {"name": "Owner", "route": "clipboard", "role": "the Owner"},
}
MODELS = ("gpt6-sol", "gpt56-sol", "opus")  # merge order within a round
MERGE_ORDER = {2: ("gpt56-sol", "gpt6-sol", "opus")}  # R2 is written in this order
ROUNDS = HERE / "rounds"
REFERENCE = {"NotebookLM", "Moderator", "Owner"}

ROUND_TASKS = {
    1: "**Round 1 of 3: independent proposals.** You have not seen the other participants' answers and "
       "they have not seen yours. From your role, answer the agenda (Q1 and Q2 first, then Q3 and Q4). "
       "Give at most 5 proposals, ranked, each with the fields in §7. Use the NotebookLM literature pack "
       "for citations; mark any other citation \"(verify)\". Under 900 words.",
    2: "**Round 2 of 3: cross-attack.** Read the other participants' Round 1 turns and the NotebookLM "
       "citation check. For each other participant's top 3 proposals give: (a) the strongest soundness "
       "objection, naming the hard constraint it risks; (b) whether the expected gain is credible, "
       "re-estimated from measured numbers; (c) a verdict KEEP, FIX (say how) or DROP. Check at least two "
       "of their factual claims against the code or the data and report what you found, with file:line. "
       "Then revise your own ranking in at most 5 lines. Add a new proposal only if it fixes an "
       "objection. Under 800 words.",
    3: "**Round 3 of 3: final plan.** Give one table of at most 5 changes that survived Round 2, with "
       "columns: change | trust-base change (none / new ESBMC lemma / new trusted component) | expected "
       "gain (MEASURED or GUESS, with its source) | hours | validation pilot and its go/no-go threshold | "
       "main risk. Then list the proposals you now reject, one line each with the reason. Then Q3: the "
       "minimum convincing evaluation and the three strongest reviewer attacks, each with our answer. "
       "Label every claim PROVED, MEASURED, ASSUMED or GUESS. Under 900 words.",
}

LIBRARIAN = """You are answering from the uploaded sources only. Context, without code: we certify a
5-layer quantized CNN (Q7.8 fixed point, 91% accuracy on GTSRB traffic signs) as locally robust
(L-infinity, epsilon = 1 byte) per input image. An untrusted CROWN-style finder produces linear bounds;
each bound step becomes a small C harness, and the bounded model checker ESBMC proves every harness
against the bit-precise deployed C (rounding, saturation, 128-bit accumulators). About 58,000 ESBMC
obligations per image, about 4 h per image; 72% of the time is in checking affine steps.

For every answer: cite the source (title, authors, year, section or page), quote the key sentence,
and write "not in sources" when the sources do not cover it. Do not fill gaps from general knowledge.
Keep each answer under 150 words.

"""
LIBRARIAN_R0 = """1. Certificate checking for neural-network verification: which works produce a certificate or
   proof that an independent (trusted, formally verified or simpler) checker validates? What does the
   checking cost relative to finding the bound?
2. Bit-precise verification of quantized neural networks (SMT, bounded model checking, ESBMC, integer
   arithmetic): which networks sizes and verification times are reported, and do any verify the
   deployed C code rather than a model?
3. Tightening linear relaxations (alpha-CROWN, beta-CROWN, branch and bound, input splitting): what
   gains in certified accuracy are reported on CNNs of a similar size, and how is it argued that the
   branches cover the whole input region?
4. Reusing proofs: incremental verification, proof transfer across inputs or across output classes,
   verifying a generic checker once and running it on concrete data. What speed-ups are reported?
5. Reporting: how do papers report certified robust accuracy when some instances time out or are
   inconclusive (sample sizes, confidence intervals, pre-registration)? Which GTSRB verification
   benchmarks exist?
6. Resource comparisons of quantized versus floating-point networks on embedded boards (memory,
   latency, energy): which metrics and measurement methods are used?
"""


def _turns(text: str) -> list[tuple[int, str, int, str]]:
    marks = list(TURN.finditer(text))
    return [(int(m[1]), m[2], int(m[3] or 0), text[m.start():marks[i + 1].start() if i + 1 < len(marks)
                                                   else len(text)].strip())
            for i, m in enumerate(marks)]


def build_prompt(who: str, rnd: int, task: str | None) -> str:
    if who == "notebooklm":
        if rnd == 0:
            return LIBRARIAN + (task or LIBRARIAN_R0)
        if not task:
            raise SystemExit("give --task with the claims to check (the moderator lists them from the thread)")
        return LIBRARIAN + task
    turns = _turns(THREAD.read_text(encoding="utf-8"))
    shown = [body for number, name, r, body in turns
             if number == 0 or r < rnd or (r == rnd and name in REFERENCE)]
    p = PARTICIPANTS.get(who, {"name": who, "role": ""})
    parts = [BRIEFING.read_text(encoding="utf-8"), "\n---\n\n# Discussion so far\n\n"]
    parts += [body + "\n\n" for body in shown]
    parts.append(f"---\n\n# Your task\n\nYou are **{p['name']}**, {p['role']}\n\n"
                 + (task or ROUND_TASKS.get(rnd, "Write the next turn."))
                 + "\n\nWrite only the body of your turn, in Markdown, without a `## Turn` heading.\n")
    return "".join(parts)


def append_turn(who: str, rnd: int, body: str) -> int:
    body = body.strip()
    if not body:
        raise SystemExit("empty reply; nothing appended")
    text = THREAD.read_text(encoding="utf-8")
    number = max((n for n, _, _, _ in _turns(text)), default=-1) + 1
    name = PARTICIPANTS.get(who, {"name": who})["name"]
    heading = f"## Turn {number} — {name} (R{rnd}, {datetime.date.today().isoformat()})"
    with THREAD.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{heading}\n\n{body}\n")
    return number


def to_clipboard(text: str) -> bool:
    try:
        subprocess.run(["clip.exe"], input=text.encode("utf-16le"), check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def from_clipboard() -> str:
    command = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                            capture_output=True, check=True)
    return result.stdout.decode("utf-8").replace("\r\n", "\n")


def save_prompt(who: str, rnd: int, prompt: str) -> Path:
    PROMPTS.mkdir(exist_ok=True)
    path = PROMPTS / f"prompt_R{rnd}_{who}.md"
    path.write_text(prompt, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("merge")
    p.add_argument("round", type=int, choices=sorted(ROUND_TASKS))
    for command in ("prompt", "add"):
        p = sub.add_parser(command)
        p.add_argument("who", choices=sorted(PARTICIPANTS))
        p.add_argument("--round", type=int, required=True)
        if command == "add":
            p.add_argument("--file", type=Path, help="read the reply from a file ('-' for stdin)")
        else:
            p.add_argument("--task")
    args = parser.parse_args()

    if args.command == "merge":
        files = {who: ROUNDS / f"R{args.round}" / f"{who}.md" for who in MERGE_ORDER.get(args.round, MODELS)}
        missing = [str(f) for f in files.values() if not f.exists()]
        if missing:
            raise SystemExit("round not complete; missing: " + ", ".join(missing))
        for who, f in files.items():
            print(f"{who}: appended Turn {append_turn(who, args.round, f.read_text(encoding='utf-8'))}")
        return
    if args.command == "add":
        if args.file is None:
            body = from_clipboard()
        elif str(args.file) == "-":
            body = sys.stdin.read()
        else:
            body = args.file.read_text(encoding="utf-8")
        print(f"appended Turn {append_turn(args.who, args.round, body)} ({len(body.split())} words)")
        return
    prompt = build_prompt(args.who, args.round, args.task)
    path = save_prompt(args.who, args.round, prompt)
    copied = to_clipboard(prompt)
    print(f"{path} ({len(prompt.split())} words){'; copied to clipboard' if copied else ''}")


if __name__ == "__main__":
    main()
