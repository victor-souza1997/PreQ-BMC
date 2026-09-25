"""Aggregate complete CROWN-chain v2 shard checks into one pilot certificate."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from verification.crown_chain_v2 import (
    Q_HIGH, Q_LOW, checker_sources, load_certificate_v2, split_claim,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_closure(certificate) -> dict[str, Any]:
    root = certificate.root
    layers = certificate.layers
    facts = certificate.facts
    bounds = certificate.chain_bounds
    target = int(root["margin_closure"]["target"])
    competitors = [int(value) for value in root["manifest"]["required_competitors"]]
    expected = [index for index in range(layers[max(layers)]["rows"]) if index != target]
    if sorted(competitors) != expected:
        raise ValueError("Competitor manifest is incomplete")
    required_roots = set(root["manifest"]["root_chains"])
    if not required_roots.issubset(bounds):
        raise ValueError("Root-chain manifest contains a missing chain")
    minimum_margin = None
    for competitor in competitors:
        margin_id = f"chain/margin/{target}-{competitor}"
        record = root["margin_closure"]["competitors"][str(competitor)]
        margin = bounds[margin_id]
        minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
        if margin != record["raw_margin_lower_bound"] or margin < 1 or not record["closed"]:
            raise ValueError(f"Competitor {competitor} does not have a positive closed margin")
        for side in ("top", "bottom"):
            edge = record[side]
            reference = edge["fact"]
            value = facts[reference]["value"] if reference.startswith("fact/") else bounds[reference]
            if value != edge["value"]:
                raise ValueError(f"Competitor {competitor} {side} reference mismatch")
            rule = edge["rule"]
            valid = {
                "raw_c <= 32766": value <= Q_HIGH - 1,
                "raw_t <= 32767": value <= Q_HIGH,
                "raw_t >= -32767": value >= Q_LOW + 1,
                "raw_c >= -32768": value >= Q_LOW,
            }.get(rule, False)
            if not valid:
                raise ValueError(f"Competitor {competitor} violates closure rule {rule}")
    return {
        "status": "VERIFIED",
        "target": target,
        "competitors_closed": len(competitors),
        "minimum_raw_margin_lower_bound": minimum_margin,
    }


def aggregate_shards(
    certificate_dir: Path,
    deployment_c: Path,
    shard_checks: list[Path],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    # Empty selection performs integrity, ledger, fact, and deployment validation
    # without retaining all 4,054 decompressed chains at once.
    certificate = load_certificate_v2(
        certificate_dir, selected_layers=[], deployment_c=deployment_c
    )
    closure = _validate_closure(certificate)
    expected_by_layer: dict[int, int] = {}
    for shard in certificate.root["shards"]:
        layer = int(shard["layer"])
        path = certificate_dir / shard["file"]
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            expected_by_layer[layer] = sum(1 for line in stream if line.strip())
    if sum(expected_by_layer.values()) != len(certificate.chain_bounds):
        raise ValueError("Authoritative shard counts do not cover every indexed chain")
    summaries = []
    layers_seen: set[int] = set()
    total_harnesses = 0
    status_counts: Counter[str] = Counter()
    harness_hashes_checked = 0
    peak_rss = 0.0
    minimum_available = None
    for directory in shard_checks:
        path = directory / "check_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        selected = [int(layer) for layer in summary["selected_layers"]]
        if len(selected) != 1 or selected[0] in layers_seen:
            raise ValueError(f"Shard {directory} does not identify one unique layer")
        layer = selected[0]
        layers_seen.add(layer)
        if (
            summary["status"] != "SHARD_VERIFIED"
            or summary["arithmetic_status"] != "VERIFIED"
            or summary["complete_certificate_scope"]
            or summary["chains_checked"] != expected_by_layer[layer]
            or sum(summary["status_counts"].values()) != summary["harness_count"]
            or summary["status_counts"] != {"VERIFIED": summary["harness_count"]}
            or summary["aborted_low_memory"]
        ):
            raise ValueError(f"Shard layer {layer} is not a complete VERIFIED shard")
        if summary.get("checker_sources") != checker_sources():
            raise ValueError(f"Shard layer {layer} was checked by different checker sources")
        if Path(summary["certificate_dir"]).resolve() != certificate_dir.resolve():
            raise ValueError(f"Shard layer {layer} names another certificate")
        if Path(summary["deployment_c"]).resolve() != deployment_c.resolve():
            raise ValueError(f"Shard layer {layer} names another deployment")
        preflight = summary["preflight"]
        if (
            preflight["chains_indexed"] != len(certificate.chain_bounds)
            or preflight["facts_validated"] != len(certificate.facts)
            or preflight["deployment_binding"]["status"] != "EXACT_ORDERED_MATCH"
            or preflight["deployment_binding"]["deployment_sha256"]
            != certificate.root["deployment"]["qnn_conv_native_c_sha256"]
            or preflight["ledger"]["box_blocks_verified"] != len(certificate.root["box_blocks"])
            or preflight["ledger"]["encoder_status"] != "VERIFIED"
        ):
            raise ValueError(f"Shard layer {layer} preflight is incomplete")
        forbidden = {"--verbosity", "--print-stack-traces", "--incremental-bmc", "--loop-invariant"}
        for record in summary["records"]:
            harness = Path(record["harness"])
            if _sha256(harness) != record["harness_sha256"]:
                raise ValueError(f"Harness changed after layer {layer} verification: {harness}")
            if record["status"] != "VERIFIED" or forbidden.intersection(record["command"]):
                raise ValueError(f"Layer {layer} has a failed or unsafe ESBMC record")
            harness_hashes_checked += 1
        total_harnesses += summary["harness_count"]
        status_counts.update(summary["status_counts"])
        peak_rss = max(peak_rss, float(summary["peak_aggregate_rss_mib"] or 0))
        available = summary["minimum_mem_available_gib"]
        if available is not None:
            minimum_available = float(available) if minimum_available is None else min(minimum_available, float(available))
        summaries.append({
            "layer": layer,
            "directory": str(directory),
            "summary_sha256": _sha256(path),
            "chains": summary["chains_checked"],
            "harnesses": summary["harness_count"],
            "wall_seconds": summary["wall_seconds"],
            "summed_esbmc_seconds": summary["summed_esbmc_seconds"],
            "peak_aggregate_rss_mib": summary["peak_aggregate_rss_mib"],
        })
    if layers_seen != set(expected_by_layer):
        raise ValueError(f"Missing shard layers: {sorted(set(expected_by_layer) - layers_seen)}")
    summaries.sort(key=lambda row: row["layer"])
    claim, guarantee_level, split_limitation = split_claim(certificate.root)
    summary = {
        "schema": "crown_chain_v2_complete_check",
        "claim": claim,
        "status": "VERIFIED",
        "guarantee_level": guarantee_level,
        "split": certificate.root["provenance"]["split"],
        "image_id": certificate.root["provenance"]["image_id"],
        "epsilon_raw_bytes": certificate.root["provenance"]["epsilon_raw_bytes"],
        "target": closure["target"],
        "certificate_dir": str(certificate_dir),
        "certificate_sha256": _sha256(certificate_dir / "certificate.json"),
        "deployment_c": str(deployment_c),
        "deployment_sha256": _sha256(deployment_c),
        "deployment_binding": certificate.preflight["deployment_binding"],
        "ledger": certificate.preflight["ledger"],
        "margin_closure": closure,
        "all_shards_verified": True,
        "chains_verified": sum(row["chains"] for row in summaries),
        "harnesses_verified": total_harnesses,
        "harness_hashes_rechecked": harness_hashes_checked,
        "status_counts": dict(sorted(status_counts.items())),
        "sequential_shard_wall_seconds": sum(float(row["wall_seconds"]) for row in summaries),
        "summed_esbmc_seconds": sum(float(row["summed_esbmc_seconds"]) for row in summaries),
        "maximum_shard_peak_aggregate_rss_mib": peak_rss,
        "minimum_mem_available_gib": minimum_available,
        "shards": summaries,
        "checker_sources": checker_sources(),
        "limitations": [
            split_limitation,
            "The certificate establishes local robustness only for the stated image and L-infinity byte radius.",
            "Certificate parsing, hashing, and proof composition remain part of the checker trusted base.",
        ],
    }
    (output / "complete_check_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate_dir", type=Path)
    parser.add_argument("--deployment-c", required=True, type=Path)
    parser.add_argument("--shard-check", action="append", required=True, type=Path, dest="shard_checks")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = aggregate_shards(
        args.certificate_dir, args.deployment_c, args.shard_checks, args.output
    )
    print(json.dumps({
        key: summary[key]
        for key in (
            "status", "claim", "chains_verified", "harnesses_verified",
            "sequential_shard_wall_seconds", "maximum_shard_peak_aggregate_rss_mib",
        )
    }))


if __name__ == "__main__":
    main()
