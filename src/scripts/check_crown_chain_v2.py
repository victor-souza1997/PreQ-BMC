"""Check a v2 CROWN certificate or one measurement shard with ESBMC."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any

from scripts.benchmark_esbmc_concurrency import benchmark
from verification.crown_chain_v2 import (
    checker_sources, load_certificate_v2, render_v2_harnesses, split_claim,
)


def _aggregate_status(counts: Counter[str], expected: int) -> str:
    if sum(counts.values()) != expected:
        return "UNKNOWN"
    for status in ("FAILED", "TIMEOUT", "MEMOUT", "UNKNOWN"):
        if counts[status]:
            return status
    return "VERIFIED" if counts["VERIFIED"] == expected else "UNKNOWN"


def check_v2(
    certificate_dir: Path,
    deployment_c: Path,
    output: Path,
    *,
    shard_layers: list[int] | None = None,
    jobs: int = 4,
    timeout: int = 60,
    memlimit: str = "4g",
    profile: str = "paper-z3",
    min_available_gib: float = 6.0,
    max_affine_terms: int = 25_000,
    max_relu_coordinates: int = 512,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    certificate = load_certificate_v2(
        certificate_dir,
        selected_layers=shard_layers,
        deployment_c=deployment_c,
    )
    (output / "preflight.json").write_text(
        json.dumps(certificate.preflight, indent=2) + "\n", encoding="utf-8"
    )
    manifest = render_v2_harnesses(
        certificate,
        output / "harnesses",
        max_affine_terms=max_affine_terms,
        max_relu_coordinates=max_relu_coordinates,
    )
    paths = [Path(record["harness"]) for record in manifest["records"]]
    result = benchmark(
        paths,
        output / "esbmc",
        jobs=jobs,
        timeout=timeout,
        memlimit=memlimit,
        profile=profile,
        min_available_gib=min_available_gib,
    )
    metadata = {
        str(Path(record["harness"]).resolve()): record
        for record in manifest["records"]
    }
    records = []
    by_chain: dict[str, Counter[str]] = defaultdict(Counter)
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for row in result["records"]:
        record = {**metadata[str(Path(row["harness"]).resolve())], **row}
        records.append(record)
        by_chain[record["chain_id"]][record["status"]] += 1
        by_kind[record["kind"]][record["status"]] += 1
    counts = Counter(record["status"] for record in records)
    arithmetic_status = _aggregate_status(counts, manifest["harness_count"])
    complete = bool(certificate.preflight["complete_certificate_scope"])
    complete_claim, complete_guarantee, _ = split_claim(certificate.root)
    if arithmetic_status == "VERIFIED" and complete:
        final_status = "VERIFIED"
        guarantee = complete_guarantee
    elif arithmetic_status == "VERIFIED":
        final_status = "SHARD_VERIFIED"
        guarantee = "arithmetic-shard-only"
    else:
        final_status = arithmetic_status
        guarantee = "none"
    summary = {
        "schema": "crown_chain_v2_check_summary",
        "claim": complete_claim if complete else "throughput_measurement_not_network_certificate",
        "split": certificate.root["provenance"]["split"],
        "checker_sources": checker_sources(),
        "status": final_status,
        "guarantee_level": guarantee,
        "arithmetic_status": arithmetic_status,
        "complete_certificate_scope": complete,
        "selected_layers": list(certificate.selected_layers),
        "certificate_dir": str(certificate_dir),
        "deployment_c": str(deployment_c),
        "preflight": certificate.preflight,
        "chains_checked": len(certificate.chains),
        "harness_count": manifest["harness_count"],
        "status_counts": dict(sorted(counts.items())),
        "chain_status_counts": dict(sorted(Counter(
            _aggregate_status(chain_counts, sum(chain_counts.values()))
            for chain_counts in by_chain.values()
        ).items())),
        "kind_status_counts": {
            kind: dict(sorted(kind_counts.items()))
            for kind, kind_counts in sorted(by_kind.items())
        },
        "jobs": jobs,
        "timeout_seconds": timeout,
        "memlimit": memlimit,
        "profile": profile,
        "max_affine_terms": max_affine_terms,
        "max_relu_coordinates": max_relu_coordinates,
        "wall_seconds": time.monotonic() - started,
        "esbmc_wall_seconds": result["wall_seconds"],
        "summed_esbmc_seconds": sum(float(row["elapsed_seconds"]) for row in records),
        "peak_aggregate_rss_mib": result["peak_total_process_tree_rss_mib"],
        "minimum_mem_available_gib": result["minimum_mem_available_gib"],
        "aborted_low_memory": result["aborted_low_memory"],
        "records": records,
    }
    (output / "check_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate_dir", type=Path)
    parser.add_argument("--deployment-c", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-layer", action="append", type=int, dest="shard_layers")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--memlimit", default="4g")
    parser.add_argument("--profile", default="paper-z3")
    parser.add_argument("--min-available-gib", type=float, default=6.0)
    parser.add_argument("--max-affine-terms", type=int, default=25_000)
    parser.add_argument("--max-relu-coordinates", type=int, default=512)
    args = parser.parse_args()
    summary = check_v2(
        args.certificate_dir,
        args.deployment_c,
        args.output,
        shard_layers=args.shard_layers,
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
            "status", "complete_certificate_scope", "chains_checked",
            "harness_count", "status_counts", "wall_seconds",
            "peak_aggregate_rss_mib", "aborted_low_memory",
        )
    }))


if __name__ == "__main__":
    main()
