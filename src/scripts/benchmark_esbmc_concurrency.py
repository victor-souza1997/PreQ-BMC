"""Measure bounded concurrent ESBMC throughput and aggregate process-tree RSS."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from verification.esbmc import ESBMCConfig, ESBMCRunner


def _mem_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def benchmark(
    harnesses: list[Path],
    output: Path,
    *,
    jobs: int,
    timeout: int,
    memlimit: str,
    profile: str,
    min_available_gib: float,
    poll_seconds: float = 0.05,
) -> dict[str, Any]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    output.mkdir(parents=True, exist_ok=False)
    runner = ESBMCRunner(ESBMCConfig(
        timeout_seconds=timeout,
        memlimit=memlimit,
        default_profile=profile,
        memory_poll_interval_seconds=poll_seconds,
    ))
    pending = list(enumerate(harnesses))
    active: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    peak_total_rss = 0
    minimum_available = _mem_available_bytes()
    threshold = int(min_available_gib * 1024**3)
    aborted_low_memory = False
    start = time.monotonic()

    def launch(index: int, harness: Path) -> None:
        source = harness.read_text(encoding="utf-8", errors="replace")
        command = runner.build_command(harness, runner.infer_unwind(source), profile)
        stdout_path = output / f"{index:03d}_{harness.stem}.stdout.log"
        stderr_path = output / f"{index:03d}_{harness.stem}.stderr.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8", errors="replace")
        stderr_handle = stderr_path.open("w", encoding="utf-8", errors="replace")
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=os.name == "posix",
        )
        active[index] = {
            "harness": harness,
            "command": command,
            "process": process,
            "start": time.monotonic(),
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "stdout_handle": stdout_handle,
            "stderr_handle": stderr_handle,
            "peak_rss": 0,
        }

    while pending or active:
        while pending and len(active) < jobs and not aborted_low_memory:
            index, harness = pending.pop(0)
            launch(index, harness)

        current_total = 0
        for state in active.values():
            rss, _, _ = runner._process_tree_resources(state["process"].pid)
            if rss is not None:
                state["peak_rss"] = max(state["peak_rss"], rss)
                current_total += rss
        peak_total_rss = max(peak_total_rss, current_total)

        available = _mem_available_bytes()
        if available is not None:
            minimum_available = (
                available if minimum_available is None
                else min(minimum_available, available)
            )
            if available < threshold and active:
                aborted_low_memory = True
                for state in active.values():
                    runner._terminate_process_tree(state["process"])

        now = time.monotonic()
        finished = []
        for index, state in active.items():
            process = state["process"]
            if process.poll() is None and now - state["start"] > timeout + 300:
                runner._terminate_process_tree(process)
            if process.poll() is None:
                continue
            state["stdout_handle"].close()
            state["stderr_handle"].close()
            stdout_tail = runner._tail_file(state["stdout_path"])
            stderr_tail = runner._tail_file(state["stderr_path"])
            return_code = int(process.returncode)
            status = runner._classify_status(
                f"{stdout_tail}\n{stderr_tail}",
                return_code,
                peak_memory_bytes=state["peak_rss"] or None,
                memlimit=memlimit,
            )
            records.append({
                "harness": str(state["harness"]),
                "status": status,
                "elapsed_seconds": now - state["start"],
                "return_code": return_code,
                "peak_memory_bytes": state["peak_rss"] or None,
                "peak_memory_mib": (
                    state["peak_rss"] / 1024**2 if state["peak_rss"] else None
                ),
                "command": list(state["command"]),
                "stdout_log_path": str(state["stdout_path"]),
                "stderr_log_path": str(state["stderr_path"]),
            })
            finished.append(index)
        for index in finished:
            del active[index]

        if aborted_low_memory and not active:
            break
        if pending or active:
            time.sleep(poll_seconds)

    wall = time.monotonic() - start
    summary = {
        "schema": "esbmc_concurrency_benchmark_v1",
        "claim": "resource_measurement_not_network_certificate",
        "jobs": jobs,
        "submitted": len(harnesses),
        "completed": len(records),
        "verified": sum(row["status"] == "VERIFIED" for row in records),
        "wall_seconds": wall,
        "throughput_queries_per_second": len(records) / wall if wall else None,
        "peak_total_process_tree_rss_bytes": peak_total_rss or None,
        "peak_total_process_tree_rss_mib": (
            peak_total_rss / 1024**2 if peak_total_rss else None
        ),
        "minimum_mem_available_bytes": minimum_available,
        "minimum_mem_available_gib": (
            minimum_available / 1024**3 if minimum_available is not None else None
        ),
        "min_available_stop_gib": min_available_gib,
        "aborted_low_memory": aborted_low_memory,
        "records": sorted(records, key=lambda row: row["harness"]),
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harnesses", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=int)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--memlimit", default="4g")
    parser.add_argument("--profile", default="paper-z3")
    parser.add_argument("--min-available-gib", type=float, default=6.0)
    args = parser.parse_args()
    summary = benchmark(
        args.harnesses,
        args.output,
        jobs=args.jobs,
        timeout=args.timeout,
        memlimit=args.memlimit,
        profile=args.profile,
        min_available_gib=args.min_available_gib,
    )
    print(json.dumps({
        key: summary[key] for key in (
            "jobs", "completed", "verified", "wall_seconds",
            "throughput_queries_per_second",
            "peak_total_process_tree_rss_mib",
            "minimum_mem_available_gib",
            "aborted_low_memory",
        )
    }))


if __name__ == "__main__":
    main()
