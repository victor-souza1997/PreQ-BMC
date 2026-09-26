"""Multi-model round table over BRIEFING.md and THREAD.md.

  prompt WHO [--task T]   build the next-turn prompt for WHO, save it and copy it to the Windows clipboard
  add WHO [--file F]      append WHO's reply as the next turn (default: read the Windows clipboard)
  ask deepseek|codex|claude [--task T]
                          send the prompt through that model's API/CLI and append the reply

WHO is free text (gpt, deepseek, codex, claude, owner). DeepSeek needs DEEPSEEK_API_KEY; the model is
DEEPSEEK_MODEL (default deepseek-reasoner). Codex runs in a read-only sandbox.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIEFING = HERE / "BRIEFING.md"
THREAD = HERE / "THREAD.md"
PROMPTS = HERE / ".prompts"
TURN = re.compile(r"^## Turn (\d+) — (.+?) \(", re.M)
NAMES = {"gpt": "GPT", "deepseek": "DeepSeek", "codex": "Codex", "claude": "Claude", "owner": "Owner"}


def _turns(text: str) -> list[tuple[int, str, str]]:
    marks = list(TURN.finditer(text))
    return [(int(m[1]), m[2], text[m.start():marks[i + 1].start() if i + 1 < len(marks) else len(text)].strip())
            for i, m in enumerate(marks)]


def build_prompt(who: str, task: str | None, last: int) -> str:
    turns = _turns(THREAD.read_text(encoding="utf-8"))
    shown = turns[:1] + turns[1:][-last:] if len(turns) > last + 1 else turns
    skipped = [f"- Turn {n} — {name} (omitted; ask the owner if you need it)" for n, name, _ in turns if
               all(n != s[0] for s in shown)]
    name = NAMES.get(who, who)
    parts = [BRIEFING.read_text(encoding="utf-8"), "\n---\n\n# Discussion so far\n"]
    if skipped:
        parts.append("\n".join(skipped) + "\n")
    parts += [body + "\n" for _, _, body in shown]
    parts.append(f"\n---\n\n# Your task\n\nYou are **{name}**. "
                 + (task or "Write the next turn: respond to the latest turns and advance the agenda.")
                 + "\n\nWrite only the body of your turn, in Markdown, without a `## Turn` heading.\n")
    return "".join(parts)


def append_turn(who: str, body: str) -> int:
    body = body.strip()
    if not body:
        raise SystemExit("empty reply; nothing appended")
    text = THREAD.read_text(encoding="utf-8")
    number = max((n for n, _, _ in _turns(text)), default=-1) + 1
    heading = f"## Turn {number} — {NAMES.get(who, who)} ({datetime.date.today().isoformat()})"
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


def save_prompt(who: str, prompt: str) -> Path:
    PROMPTS.mkdir(exist_ok=True)
    path = PROMPTS / f"prompt_{who}.md"
    path.write_text(prompt, encoding="utf-8")
    return path


def ask(who: str, prompt: str) -> str:
    if who == "deepseek":
        from openai import OpenAI
        key = os.environ.get("DEEPSEEK_API_KEY") or sys.exit("set DEEPSEEK_API_KEY")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=900)
        reply = client.chat.completions.create(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner"),
            messages=[{"role": "user", "content": prompt}])
        return reply.choices[0].message.content
    if who == "codex":
        out = PROMPTS / "codex_reply.md"
        subprocess.run(["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
                        "-o", str(out), "-"], input=prompt.encode("utf-8"), check=True, cwd=HERE)
        return out.read_text(encoding="utf-8")
    if who == "claude":
        result = subprocess.run(["claude", "-p"], input=prompt.encode("utf-8"), capture_output=True, check=True)
        return result.stdout.decode("utf-8")
    raise SystemExit(f"no automatic route for {who}; use 'prompt' and 'add'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prompt", "ask"):
        p = sub.add_parser(command)
        p.add_argument("who")
        p.add_argument("--task")
        p.add_argument("--last", type=int, default=6, help="recent turns to include besides Turn 0")
    p = sub.add_parser("add")
    p.add_argument("who")
    p.add_argument("--file", type=Path, help="read the reply from a file ('-' for stdin)")
    args = parser.parse_args()

    if args.command == "add":
        if args.file is None:
            body = from_clipboard()
        elif str(args.file) == "-":
            body = sys.stdin.read()
        else:
            body = args.file.read_text(encoding="utf-8")
        print(f"appended Turn {append_turn(args.who, body)} ({len(body.split())} words)")
        return
    prompt = build_prompt(args.who, args.task, args.last)
    path = save_prompt(args.who, prompt)
    if args.command == "prompt":
        copied = to_clipboard(prompt)
        print(f"{path} ({len(prompt.split())} words){'; copied to clipboard' if copied else ''}")
        return
    print(f"asking {args.who} ({len(prompt.split())} words)...", flush=True)
    number = append_turn(args.who, ask(args.who, prompt))
    print(f"appended Turn {number}")


if __name__ == "__main__":
    main()
