"""Select a calibration-passing immutable artifact and materialize evaluation."""
import argparse
import json
from pathlib import Path

from datasets.gtsrb_study import sha256
from scripts.freeze_ssv_campaign import freeze
from scripts.run_ssv_cnn_gate import write_new_json


def select(sweep_path, runs_root, output):
    sweep_path, runs_root, output = Path(sweep_path), Path(runs_root), Path(output)
    sweep = json.loads(sweep_path.read_text())
    if sweep.get("schema") != "ssv_fixed_qif_sweep_materialized_v1":
        raise ValueError("Expected a materialized fixed-QIF sweep")
    if output.exists():
        raise FileExistsError(output)
    candidates = sweep["candidates"]
    if not candidates:
        raise ValueError("Sweep has no candidates")

    # Source eligibility is determined before quantization. Require a complete
    # report for the first (smallest) artifact to anchor that QIF-independent
    # classification. Source-inconclusive regions remain in the campaign report,
    # but cannot logically select a quantized format because ESBMC is never run.
    anchor = candidates[0]
    anchor_study = json.loads(Path(anchor["study"]).read_text())
    anchor_reports = {}
    for run in anchor_study["runs"]:
        path = runs_root / anchor["id"] / run["run_id"] / "region_summary.json"
        if not path.is_file():
            raise ValueError("Complete the first candidate before selecting a fixed QIF")
        report = json.loads(path.read_text())
        if (report.get("run_id") != run["run_id"]
                or report.get("model_sha256") != sweep["model_sha256"]
                or report.get("verification_mode") != "fixed_qif_check"
                or report.get("fixed_artifact_sha256") != anchor["artifact_sha256"]):
            raise ValueError(f"Invalid source-eligibility anchor report: {path}")
        anchor_reports[run["run_id"]] = report
    eligible_ids = {
        run_id for run_id, report in anchor_reports.items()
        if report.get("source_region", {}).get("status") == "VERIFIED"
    }
    if not eligible_ids:
        raise ValueError("No calibration region passed the source prerequisite")

    outcomes, winner = [], None
    for candidate in candidates:
        study_path = Path(candidate["study"])
        study = json.loads(study_path.read_text())
        reports = []
        missing = []
        for run in study["runs"]:
            path = runs_root / candidate["id"] / run["run_id"] / "region_summary.json"
            if not path.is_file():
                if run["run_id"] in eligible_ids:
                    missing.append(run["run_id"])
                continue
            report = json.loads(path.read_text())
            if (report.get("run_id") != run["run_id"]
                    or report.get("model_sha256") != sweep["model_sha256"]
                    or report.get("verification_mode") != "fixed_qif_check"
                    or report.get("fixed_artifact_sha256") != candidate["artifact_sha256"]):
                raise ValueError(f"Invalid calibration report: {path}")
            reports.append(report)
        eligible_reports = [report for report in reports if report["run_id"] in eligible_ids]
        passed = (not missing and len(eligible_reports) == len(eligible_ids) and all(
            report.get("byte_crop_property_verified") is True for report in eligible_reports
        ))
        outcome = {"id": candidate["id"], "qif": candidate["qif"], "complete": not missing,
                   "missing_run_ids": missing, "verified_regions": sum(
                       bool(report.get("byte_crop_property_verified")) for report in eligible_reports),
                   "source_eligible_regions": len(eligible_ids),
                   "source_inconclusive_regions": len(anchor_reports) - len(eligible_ids),
                   "total_regions": len(eligible_ids), "selected": False}
        outcomes.append(outcome)
        if winner is None and passed:
            winner = candidate
            outcome["selected"] = True
    output.mkdir(parents=True)
    result = {"schema": "ssv_fixed_qif_selection_v1", "sweep": str(sweep_path.resolve()),
              "sweep_sha256": sha256(sweep_path), "rule": sweep["selection"],
              "source_eligibility_anchor": {"candidate": anchor["id"],
                                             "completed_regions": len(anchor_reports),
                                             "source_eligible_regions": len(eligible_ids),
                                             "source_inconclusive_regions": len(anchor_reports) - len(eligible_ids),
                                             "source_eligible_run_ids": sorted(eligible_ids)},
              "candidates": outcomes, "selected": None, "evaluation_status": "NOT_MATERIALIZED"}
    if winner is not None:
        evaluation = freeze(sweep["evaluation_study"], output / "evaluation", winner["qif"], campaign="full")
        result["selected"] = {"id": winner["id"], "qif": winner["qif"],
                              "calibration_artifact_sha256": winner["artifact_sha256"],
                              "evaluation_study": str((output / "evaluation/study.json").resolve()),
                              "evaluation_artifact_sha256": evaluation["fixed_artifact_sha256"]}
        result["evaluation_status"] = "MATERIALIZED_NOT_RUN"
    else:
        result["evaluation_status"] = "NO_CANDIDATE_PASSED_CALIBRATION"
    write_new_json(output / "selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.sweep, args.runs_root, args.output)
    print(args.output / "selection.json")
    raise SystemExit(0 if result["selected"] else 2)


if __name__ == "__main__":
    main()
