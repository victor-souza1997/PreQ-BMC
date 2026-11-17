"""Run (or watch) a crown_chain_v2 protocol and draw a live terminal dashboard.

The driver src/scripts/run_crown_chain_v2_protocol.py is launched unchanged; this
script only reads the files it writes. Percentages per image are ESTIMATES: each
stage is weighted by its measured cost on test image 06267, and inside a check
stage progress is closed chains / planned chains from export.json.

  python tools/ssv2026/run_protocol_monitor.py run   --protocol P --output DIR [--max-full N] [--screen-only]
  python tools/ssv2026/run_protocol_monitor.py watch --protocol P --output DIR
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVER = "src/scripts/run_crown_chain_v2_protocol.py"
# Measured stage seconds for test image 06267 (outcome.json, box wall, check per layer).
WEIGHTS = {"screen": 33, "box": 616, "export": 76,
           "check_l0": 64, "check_l1": 505, "check_l2": 6864, "check_l3": 3280, "check_l4": 1529,
           "complete": 30}
STAGES = list(WEIGHTS)
BOX_CALLS = 689  # box-stage ESBMC calls on both certified test images


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.split()[0]) * 1024
    return values


def gib(n):
    return f"{n / 2**30:6.1f} GiB"


def bar(fraction, width=30):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {100 * fraction:5.1f}%"


def count(directory, suffix):
    try:
        return sum(1 for entry in os.scandir(directory) if entry.name.endswith(suffix))
    except FileNotFoundError:
        return 0


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def planned_chains(work, layer):
    report = read_json(work / "export.json") or {}
    per_layer = {int(k): v for k, v in report.get("chains_per_layer", {}).items()}
    if layer in per_layer:
        return per_layer[layer]
    if layer == 4 and "chains" in report:
        return report["chains"] - sum(per_layer.values())
    return None


def image_progress(work):
    """Return (stage name, fraction of this stage, fraction of the image), all estimates."""
    if (work / "outcome.json").exists():
        return "done", 1.0, 1.0
    done = 0.0
    total = sum(WEIGHTS.values())
    if not (work / "screen.json").exists():
        return "screen", 0.0, 0.0
    done += WEIGHTS["screen"]
    if not (work / "box" / "pilot_summary.json").exists():
        calls = count(work / "box" / "proof" / "harnesses", ".stdout.log")
        frac = min(1.0, calls / BOX_CALLS)
        return "box", frac, (done + frac * WEIGHTS["box"]) / total
    done += WEIGHTS["box"]
    if not (work / "certificate" / "certificate.json").exists():
        return "export", 0.0, done / total
    done += WEIGHTS["export"]
    for layer in range(5):
        stage = f"check_l{layer}"
        shard = work / stage
        if (shard / "check_summary.json").exists():
            done += WEIGHTS[stage]
            continue
        closed = count(shard / "esbmc", "_chain_close.stdout.log")
        planned = planned_chains(work, layer)
        frac = min(1.0, closed / planned) if planned else 0.0
        return f"{stage} ({closed}/{planned or '?'} chains)", frac, (done + frac * WEIGHTS[stage]) / total
    return "aggregate", 0.0, (done) / total


def dashboard(protocol, output, child, started):
    images = protocol["images_in_draw_order"]
    n = len(images)
    outcomes = Counter()
    current = None
    for ordinal in range(n):
        work = output / f"{ordinal:03d}"
        outcome = read_json(work / "outcome.json")
        if outcome:
            outcomes[outcome["outcome"]] += 1
        elif current is None and work.exists() and (
                not (work / "screen.json").exists() or (work / "box_config.json").exists()):
            current = (ordinal, work)
    finished = sum(outcomes.values())
    mem = meminfo()
    used = mem["MemTotal"] - mem["MemAvailable"]
    swap = mem["SwapTotal"] - mem["SwapFree"]
    out_disk = shutil.disk_usage(output.resolve())
    root_disk = shutil.disk_usage("/")
    lines = [
        f"crown_chain_v2 protocol  {protocol['split']}  n={n}  eps={protocol['epsilon_raw_bytes']}"
        f"   elapsed {time.strftime('%H:%M:%S', time.gmtime(time.time() - started))}",
        f"output -> {output.resolve()}",
        "",
        f"RAM      {bar(used / mem['MemTotal'])}  {gib(used)} / {gib(mem['MemTotal'])}",
        f"Swap     {bar(swap / mem['SwapTotal'] if mem['SwapTotal'] else 0)}  {gib(swap)} / {gib(mem['SwapTotal'])}",
        f"Out disk {bar(out_disk.used / out_disk.total)}  {gib(out_disk.used)} / {gib(out_disk.total)}  free {gib(out_disk.free)}",
        f"SSD /    {bar(root_disk.used / root_disk.total)}  {gib(root_disk.used)} / {gib(root_disk.total)}  free {gib(root_disk.free)}",
        "",
    ]
    image_fraction = 0.0
    if current:
        ordinal, work = current
        stage, stage_fraction, image_fraction = image_progress(work)
        lines += [f"Image {ordinal + 1}/{n}: {images[ordinal]}",
                  f"  stage  {stage}",
                  f"  stage  {bar(stage_fraction)}",
                  f"  image  {bar(image_fraction)}  (estimate)"]
    else:
        lines.append("Image  (none in progress)")
    lines += ["",
              f"All images {bar((finished + image_fraction) / n)}  {finished}/{n} with an outcome",
              "  " + "  ".join(f"{k}={v}" for k, v in sorted(outcomes.items())),
              ""]
    if child is not None:
        lines.append("driver: running" if child.poll() is None else f"driver: exited rc={child.returncode}")
    lines.append("Ctrl-C stops the driver; interrupted stage outputs are set aside, never deleted.")
    sys.stdout.write("\x1b[H\x1b[2J" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "watch"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-full", type=int)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    child = None
    if args.mode == "run":
        argv = [sys.executable, DRIVER, "run", "--protocol", str(args.protocol), "--output", str(args.output)]
        if args.max_full is not None:
            argv += ["--max-full", str(args.max_full)]
        if args.screen_only:
            argv.append("--screen-only")
        log = (args.output / "driver.log").open("a", encoding="utf-8")
        child = subprocess.Popen(argv, cwd=REPO, env={**os.environ, "PYTHONPATH": "src"},
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    started = time.time()
    try:
        while True:
            dashboard(protocol, args.output, child, started)
            if child is not None and child.poll() is not None:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            child.wait()
    if child is not None:
        print(f"driver exited rc={child.returncode}; log: {args.output / 'driver.log'}")


if __name__ == "__main__":
    main()
