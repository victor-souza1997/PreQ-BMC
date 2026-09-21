"""Audit both frozen GTSRB cohorts at the float source gate, without ESBMC."""
import argparse
from collections import Counter
import json
from pathlib import Path

from datasets.gtsrb_study import sha256
from scripts.freeze_ssv_campaign import main_runs
from scripts.report_ssv_regions import write_csv
from scripts.run_ssv_cnn_gate import write_new_json
from scripts.run_ssv_gtsrb import build_region_context
from scripts.search_ssv_qif import _configured_study, _load, validate


def audit(config_path, output, *, prior_calibration=None, prior_evaluation=None):
    config_path, output = Path(config_path), Path(output)
    config = _load(config_path)
    validate(config)
    studies = {name: _configured_study(_load(config[key]), config)
               for name, key in (("calibration", "calibration_study"), ("evaluation", "evaluation_study"))}
    if studies["calibration"]["model_sha256"] != studies["evaluation"]["model_sha256"]:
        raise ValueError("Source audit cohorts use different models")
    ids = [{s["id"] for s in study["selected_images"]} for study in studies.values()]
    if ids[0] & ids[1]:
        raise ValueError("Source audit cohorts overlap")
    prior = {"calibration": Path(prior_calibration) if prior_calibration else None,
             "evaluation": Path(prior_evaluation) if prior_evaluation else None}
    output.mkdir(parents=True, exist_ok=False)
    rows, old_verified = [], []
    for cohort, study in studies.items():
        for run in main_runs(study):
            destination = output / cohort / run["run_id"]
            destination.mkdir(parents=True)
            context = build_region_context(study, run, destination)
            context.synthesizer.check_source_region(context.low, context.high)
            source = context.synthesizer.source_region_summary()
            if context.synthesizer.esbmc_call_records:
                raise ValueError("Source-only audit invoked ESBMC")
            baseline = None
            if prior[cohort] is not None:
                path = prior[cohort] / run["run_id"] / "region_summary.json"
                if path.is_file():
                    old = _load(path)
                    if old.get("model_sha256") != study["model_sha256"] or old.get("run_id") != run["run_id"]:
                        raise ValueError(f"Prior report identity mismatch: {path}")
                    baseline = old.get("final_status")
                    if baseline == "VERIFIED":
                        old_verified.append((cohort, run["run_id"], source["status"]))
            row = {"cohort": cohort, "run_id": run["run_id"], "sample_id": run["sample"]["id"],
                   "stratum": run["sample"]["stratum"], "epsilon_raw_bytes": run["epsilon"],
                   "clean_margin": run["sample"]["clean_margin"],
                   "source_status": source["status"], "source_method": source["method"],
                   "deeppoly_margin_lower_bound": source.get("deeppoly_certified_margin_lower_bound",
                                                                source.get("certified_margin_lower_bound")),
                   "certified_margin_lower_bound": source.get("certified_margin_lower_bound"),
                   "validated_counterexample": source.get("validated_counterexample") is not None,
                   "prior_final_status": baseline}
            rows.append(row)
            write_new_json(destination / "source_region.json", {"study_sha256": sha256(config[(
                "calibration_study" if cohort == "calibration" else "evaluation_study")]),
                "model_sha256": study["model_sha256"], "run_id": run["run_id"], "source_region": source})
            print(cohort, run["run_id"], source["status"], flush=True)
    write_csv(output / "source_regions.csv", rows)
    statuses = Counter(row["source_status"] for row in rows)
    summary = {"schema": "ssv_source_gate_audit_v1", "config_sha256": sha256(config_path),
               "model_sha256": studies["calibration"]["model_sha256"],
               "total_regions": len(rows), "by_status": dict(statuses),
               "baseline_verified_source_regressions": [
                   {"cohort": cohort, "run_id": run_id, "new_source_status": status}
                   for cohort, run_id, status in old_verified if status != "VERIFIED"],
               "baseline_verified_checked": len(old_verified),
               "esbmc_attempted": False,
               "interpretation": "Float source-gate outcomes only; no integer C or ESBMC certificates."}
    write_new_json(output / "source_audit.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/sign_qif_search.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-calibration", type=Path)
    parser.add_argument("--prior-evaluation", type=Path)
    args = parser.parse_args()
    result = audit(args.config, args.output,
                   prior_calibration=args.prior_calibration, prior_evaluation=args.prior_evaluation)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not result["baseline_verified_source_regressions"] else 2)


if __name__ == "__main__":
    main()
