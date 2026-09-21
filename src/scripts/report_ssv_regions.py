"""Additional per-region/aggregate CSVs; missing device metrics remain empty."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from scripts.run_ssv_cnn_gate import write_new_json
from datasets.gtsrb_study import sha256


def summarize(reports):
    groups = defaultdict(list)
    for r in reports:
        mode = r.get("verification_mode", "region_synthesis")
        identity = r.get("fixed_artifact_source_sha256") or ""
        if mode == "fixed_qif_check" and (not identity or (
                r["byte_crop_property_verified"] and r.get("generated_source_sha256") != identity)):
            raise ValueError("Fixed-artifact certificate has missing or inconsistent C identity")
        groups[(r["epsilon"], r["block_size"], r["margin_cuts"], mode, identity)].append(r)
    rows = []
    for (eps, beta, cuts, mode, artifact), group in sorted(groups.items()):
        identities = [r["sample"]["id"] for r in group]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate region/variant: repetitions need a separate timing analysis")
        eligible = sum(r["source_region"]["status"] == "VERIFIED" for r in group)
        certified = sum(bool(r["byte_crop_property_verified"]) for r in group)
        calls = [c for r in group for c in r["calls"]]
        runtime = np.array([r["total_runtime_seconds"] for r in group])
        memory = [c["peak_memory_bytes"] for c in calls if c.get("peak_memory_bytes") is not None]
        rows.append({"epsilon_raw_bytes": eps, "beta": beta, "margin_cuts": cuts,
                     "verification_mode": mode, "fixed_artifact_source_sha256": artifact or None,
                     "certified_artifact_count": len({r.get("generated_source_sha256") for r in group
                                                       if r["byte_crop_property_verified"] and r.get("generated_source_sha256")}),
                     "n_images": len(set(identities)), "n_regions_completed": len(group),
                     "source_verified_count": eligible,
                     "source_inconclusive_count": sum(r["source_region"]["status"] in {"INCONCLUSIVE", "UNKNOWN"}
                                                      for r in group),
                     "source_refuted_count": sum(r["source_region"]["status"] == "REFUTED" for r in group),
                     "encoded_verified_count": sum(bool(r["encoded_contract_verified"]) for r in group),
                     "byte_crop_certified_count": certified,
                     "certified_fraction_completed_regions": certified / len(group),
                     "certified_fraction_given_source_verified": certified / eligible if eligible else None,
                     "android_transfer_certified_count": sum(bool(r["android_transfer_verified"]) for r in group),
                     "timeout_regions": sum(r["final_status"] == "TIMEOUT" for r in group),
                     "memout_regions": sum(r["final_status"] == "MEMOUT" for r in group),
                     "total_calls_including_diagnostics": len(calls),
                     "query_timeout_count_including_diagnostics": sum(c["status"] == "TIMEOUT" for c in calls),
                     "query_memout_count_including_diagnostics": sum(c["status"] == "MEMOUT" for c in calls),
                     "runtime_median_s": float(np.median(runtime)),
                     "runtime_q1_s": float(np.quantile(runtime, .25)), "runtime_q3_s": float(np.quantile(runtime, .75)),
                     "sampled_peak_query_rss_mib": max(memory) / 2**20 if memory else None,
                     "android_accuracy": None, "android_energy_j_per_inference": None})
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    paths = sorted(args.input.glob("*/region_summary.json"))
    reports = [json.loads(p.read_text()) for p in paths]
    expected = {r["run_id"] for r in study["runs"]}
    seen = [r["run_id"] for r in reports]
    if len(set(seen)) != len(seen) or not set(seen) <= expected:
        raise ValueError("Duplicate or unexpected run identities")
    if any(r["model_sha256"] != study["model_sha256"] for r in reports):
        raise ValueError("Cannot mix model identities")
    if study.get("fixed_artifact_sha256") and any(
            r.get("fixed_artifact_sha256") != study["fixed_artifact_sha256"] for r in reports):
        raise ValueError("Reports do not belong to this fixed-artifact campaign")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = summarize(reports)
    for row in summaries:
        configured = sum(r["epsilon"] == row["epsilon_raw_bytes"] and r["block_size"] == row["beta"]
                         and r["margin_cuts"] == row["margin_cuts"] for r in study["runs"])
        row["n_regions_configured"] = configured
        row["n_regions_without_final_report"] = configured - row["n_regions_completed"]
        row["certified_fraction_configured_regions"] = row["byte_crop_certified_count"] / configured
    write_csv(args.output / "region_certification_summary.csv", summaries)
    write_csv(args.output / "all_regions.csv", [{"run_id": r["run_id"], "sample_id": r["sample"]["id"],
              "epsilon": r["epsilon"], "beta": r["block_size"], "margin_cuts": r["margin_cuts"],
              "source_status": r["source_region"]["status"],
              "source_method": r["source_region"].get("method"),
              "deeppoly_certified_margin_lower_bound": r["source_region"].get(
                  "deeppoly_certified_margin_lower_bound", r["source_region"].get("certified_margin_lower_bound")),
              "source_certified_margin_lower_bound": r["source_region"].get("certified_margin_lower_bound"),
              "final_status": r["final_status"],
              "byte_crop_property_verified": r["byte_crop_property_verified"],
              "android_transfer_verified": r["android_transfer_verified"],
              "verification_mode": r.get("verification_mode", "region_synthesis"),
              "generated_source_sha256": r.get("generated_source_sha256"),
              "fixed_artifact_source_sha256": r.get("fixed_artifact_source_sha256"),
              "runtime_seconds": r["total_runtime_seconds"]} for r in reports])
    write_new_json(args.output / "ledger.json", {"study_sha256": sha256(args.study),
                   "configured_runs": len(expected), "completed_runs": len(seen),
                   "without_final_report": sorted(expected - set(seen)),
                   "runner_errors": [json.loads(p.read_text()) for p in sorted(args.input.glob("*/runner_error.json"))],
                   "sources": [{"path": str(p), "sha256": sha256(p)} for p in paths],
                   "interpretation": "Unrun and crashed configurations are not ESBMC timeouts or failures"})


if __name__ == "__main__":
    main()
