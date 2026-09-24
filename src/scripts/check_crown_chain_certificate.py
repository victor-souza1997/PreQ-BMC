"""Render and check exact-integer CROWN chains with bounded ESBMC workers."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time
from typing import Any

from scripts.benchmark_esbmc_concurrency import benchmark
from verification.crown_chain_certificate import (
    load_crown_chain_certificate,
    render_all_harnesses,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _status(counts: Counter[str], expected: int) -> str:
    if sum(counts.values()) != expected:
        return "UNKNOWN"
    if counts["FAILED"]:
        return "FAILED"
    if counts["TIMEOUT"]:
        return "TIMEOUT"
    if counts["MEMOUT"]:
        return "MEMOUT"
    if counts["UNKNOWN"]:
        return "UNKNOWN"
    return "VERIFIED" if counts["VERIFIED"] == expected else "UNKNOWN"


def check_certificate(
    certificate_path: Path,
    output: Path,
    *,
    jobs: int = 4,
    timeout: int = 60,
    memlimit: str = "4g",
    profile: str = "paper-z3",
    min_available_gib: float = 6.0,
    max_affine_terms: int = 25_000,
    max_relu_coordinates: int = 512,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    certificate = load_crown_chain_certificate(certificate_path)
    harness_root = output / "harnesses"
    manifest = render_all_harnesses(
        certificate,
        harness_root,
        max_affine_terms=max_affine_terms,
        max_relu_coordinates=max_relu_coordinates,
    )

    metadata_by_path = {
        str(Path(record["harness"]).resolve()): record
        for record in manifest["records"]
    }
    groups: dict[str, list[Path]] = {}
    for record in manifest["records"]:
        groups.setdefault(record["chain_id"], []).append(
            Path(record["harness"]).resolve()
        )

    started = time.monotonic()
    group_summaries = []
    all_records: list[dict[str, Any]] = []
    peak_aggregate_rss = 0
    minimum_available: int | None = None
    aborted = False
    run_root = output / "esbmc"
    run_root.mkdir()
    for chain_id, harnesses in groups.items():
        result = benchmark(
            harnesses,
            run_root / _safe_name(chain_id),
            jobs=jobs,
            timeout=timeout,
            memlimit=memlimit,
            profile=profile,
            min_available_gib=min_available_gib,
        )
        counts = Counter(row["status"] for row in result["records"])
        enriched = []
        for row in result["records"]:
            metadata = metadata_by_path[str(Path(row["harness"]).resolve())]
            enriched_row = {**metadata, **row}
            enriched.append(enriched_row)
            all_records.append(enriched_row)
        aggregate = int(result["peak_total_process_tree_rss_bytes"] or 0)
        peak_aggregate_rss = max(peak_aggregate_rss, aggregate)
        available = result["minimum_mem_available_bytes"]
        if available is not None:
            minimum_available = (
                int(available)
                if minimum_available is None
                else min(minimum_available, int(available))
            )
        aborted = aborted or bool(result["aborted_low_memory"])
        group_summaries.append({
            "chain_id": chain_id,
            "status": _status(counts, len(harnesses)),
            "harness_count": len(harnesses),
            "status_counts": dict(sorted(counts.items())),
            "summed_elapsed_seconds": sum(
                float(row["elapsed_seconds"]) for row in result["records"]
            ),
            "wall_seconds": float(result["wall_seconds"]),
            "peak_aggregate_rss_bytes": result[
                "peak_total_process_tree_rss_bytes"
            ],
            "peak_aggregate_rss_mib": result[
                "peak_total_process_tree_rss_mib"
            ],
            "max_query_peak_rss_mib": max(
                (
                    float(row["peak_memory_mib"])
                    for row in result["records"]
                    if row["peak_memory_mib"] is not None
                ),
                default=None,
            ),
            "records": enriched,
        })

    counts = Counter(row["status"] for row in all_records)
    local_status = _status(counts, len(manifest["records"]))
    if local_status == "VERIFIED" and manifest["conditional"]:
        result_status = "CONDITIONAL_VERIFIED"
    else:
        result_status = local_status
    summary = {
        "schema": "crown_chain_pilot_check_v1",
        "claim": "local_chain_obligations_not_end_to_end_network_certificate",
        "certificate": str(certificate_path),
        "certificate_claim": certificate["claim"],
        "status": result_status,
        "local_obligations_status": local_status,
        "conditional_on_unexported_dependencies": bool(manifest["conditional"]),
        "dependency_records": manifest["dependency_records"],
        "dependencies_exported": manifest["dependencies_exported"],
        "harness_count": len(manifest["records"]),
        "status_counts": dict(sorted(counts.items())),
        "jobs": jobs,
        "timeout_seconds": timeout,
        "memlimit": memlimit,
        "profile": profile,
        "max_affine_terms": max_affine_terms,
        "max_relu_coordinates": max_relu_coordinates,
        "summed_elapsed_seconds": sum(
            float(row["elapsed_seconds"]) for row in all_records
        ),
        "wall_seconds": time.monotonic() - started,
        "peak_aggregate_rss_bytes": peak_aggregate_rss or None,
        "peak_aggregate_rss_mib": (
            peak_aggregate_rss / 1024**2 if peak_aggregate_rss else None
        ),
        "minimum_mem_available_bytes": minimum_available,
        "aborted_low_memory": aborted,
        "chains": group_summaries,
    }
    (output / "pilot_check_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--memlimit", default="4g")
    parser.add_argument("--profile", default="paper-z3")
    parser.add_argument("--min-available-gib", type=float, default=6.0)
    parser.add_argument("--max-affine-terms", type=int, default=25_000)
    parser.add_argument("--max-relu-coordinates", type=int, default=512)
    args = parser.parse_args()
    summary = check_certificate(
        args.certificate,
        args.output,
        jobs=args.jobs,
        timeout=args.timeout,
        memlimit=args.memlimit,
        profile=args.profile,
        min_available_gib=args.min_available_gib,
        max_affine_terms=args.max_affine_terms,
        max_relu_coordinates=args.max_relu_coordinates,
    )
    print(json.dumps({
        key: summary[key]
        for key in (
            "status",
            "harness_count",
            "status_counts",
            "wall_seconds",
            "peak_aggregate_rss_mib",
        )
    }))


if __name__ == "__main__":
    main()
