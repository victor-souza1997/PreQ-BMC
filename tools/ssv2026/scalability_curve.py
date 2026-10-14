#!/usr/bin/env python3
"""Scalability curve for monolithic ESBMC on the actual compact C (validation image 3652 only).

Each point frees k raw input bytes to the epsilon-1 box of validation image 3652 and fixes every other
byte at its centre value. k = 0 is one concrete execution. Free bytes follow a fixed permutation (seed 0),
so point k's free set contains point j's for j < k. The property is the direct harness of
quantization_candidates.py: encoder accepts, then a strict margin over all 42 competitors.

One ESBMC process at a time; never reads a test image. MEASURED: status, time, peak process-tree RSS,
and the last ESBMC phase reached. A VERIFIED covers only the stated byte box.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantization_candidates as qc  # noqa: E402

import numpy as np  # noqa: E402

from datasets.gtsrb_study import load_crop, sha256  # noqa: E402
from verification.esbmc import ESBMCConfig, ESBMCRunner  # noqa: E402

SAMPLE = "Final_Training/Images/00012/00005_00022.ppm"
PHASES = ("Parsing", "Converting", "Generating GOTO Program", "Starting Bounded Model Checking",
          "Symex completed", "Generated", "Solving with solver", "Runtime decision procedure")


def run(path, memlimit, timeout, interval, min_available, profile="paper-z3"):
    runner = ESBMCRunner(ESBMCConfig(timeout_seconds=timeout, memlimit=memlimit))
    command = list(runner.build_command(path, runner.infer_unwind(path.read_text()), profile))
    if not interval:
        command = [a for a in command if a not in ("--interval-analysis", "--interval-analysis-simplify")]
    command = [a for a in command if a != "--result-only"]
    available = qc.available_gib()
    if available < min_available:
        return {"status": "NOT_RUN_LOW_MEMORY", "available_gib": available, "command": command}
    start, peak, status = time.monotonic(), 0, None
    log = Path(str(path) + ".log")
    with log.open("x") as handle:
        proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            memory, _, _ = ESBMCRunner._process_tree_resources(proc.pid)
            peak = max(peak, memory or 0)
            if qc.available_gib() < 2 or time.monotonic() - start > timeout + 10:
                status = "KILLED_LOW_MEMORY" if qc.available_gib() < 2 else "TIMEOUT"
                os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(0.2)
        proc.wait()
    text = log.read_text()
    if status is None:
        status = qc.classify_result(text, proc.returncode, peak)
        if "Out of memory" in text:
            status = "OUT_OF_MEMORY"
    phase = next((p for p in reversed(PHASES) if p in text), "none")
    return {"status": status, "return_code": proc.returncode, "elapsed_seconds": round(time.monotonic() - start, 2),
            "peak_process_tree_rss_bytes": peak, "last_phase": phase, "available_gib_at_launch": round(available, 2),
            "source_sha256": sha256(path), "profile": profile, "command": command}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--free-bytes", type=int, nargs="+", required=True)
    p.add_argument("--memlimit", default="4g")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--no-interval", action="store_true")
    p.add_argument("--min-available-gib", type=float, default=6)
    p.add_argument("--stop-after-failure", action="store_true",
                   help="stop the series at the first point that is neither VERIFIED nor FAILED")
    args = p.parse_args()
    folder, manifest, network, encoder, study, rows = qc.load_candidate(args.artifact)
    row = next(r for r in rows if r["id"] == SAMPLE)
    image = load_crop(study["dataset_root"], row)
    centre = np.asarray(encoder.resize(image), dtype=np.int64).reshape(-1)
    box_low, box_high = (np.asarray(a, dtype=np.int64).reshape(-1) for a in encoder.byte_box(image, 1))
    if not (np.all(box_low <= centre) and np.all(centre <= box_high)):
        raise ValueError("Centre bytes are outside the epsilon-1 byte box")
    order = np.random.default_rng(0).permutation(centre.size)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "curve.jsonl"
    for k in args.free_bytes:
        low, high = centre.copy(), centre.copy()
        low[order[:k]], high[order[:k]] = box_low[order[:k]], box_high[order[:k]]
        tag = f"k{k:04d}_{'nointerval' if args.no_interval else 'interval'}"
        path = out / f"direct_{tag}.c"
        path.write_text(qc.direct_harness(folder, network, low, high, row["class_id"]))
        result = run(path, args.memlimit, args.timeout, not args.no_interval, args.min_available_gib)
        record = {"artifact": str(folder), "bits": manifest["bits"], "fractional_bits": manifest["fractional_bits"],
                  "sample_id": SAMPLE, "split": "validation", "free_bytes": k,
                  "free_byte_values": int(np.sum(high - low + 1) - (centre.size - k)) if k else 0,
                  "interval_analysis": not args.no_interval, "memlimit": args.memlimit, "timeout": args.timeout,
                  **result}
        with summary.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({x: record[x] for x in ("bits", "free_bytes", "interval_analysis", "status",
                                                "elapsed_seconds", "peak_process_tree_rss_bytes", "last_phase")}),
              flush=True)
        if args.stop_after_failure and result["status"] not in ("VERIFIED", "FAILED"):
            break


if __name__ == "__main__":
    main()
