"""Materialize a fixed-artifact calibration sweep without running ESBMC.

The calibration study determines one globally fixed Q/I/F. The evaluation study
is not materialized until selection, preventing its outcomes from choosing the
deployed artifact.
"""
import argparse
import copy
import json
from pathlib import Path

from datasets.gtsrb_study import sha256
from models.ssv_artifact import parse_qif
from scripts.freeze_ssv_campaign import freeze
from scripts.run_ssv_cnn_gate import write_new_json


def _load(path):
    return json.loads(Path(path).read_text())


def validate(config):
    if config.get("schema") != "ssv_fixed_qif_sweep_v1":
        raise ValueError("Expected ssv_fixed_qif_sweep_v1")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("At least two ordered fixed-QIF candidates are required")
    names = []
    previous = None
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
            raise ValueError("Each candidate requires a stable id")
        specs = parse_qif(candidate.get("qif"))
        widths = tuple(spec.total_bits for spec in specs)
        if previous is not None and any(new < old for new, old in zip(widths, previous, strict=True)):
            raise ValueError("Candidates must be nondecreasing in both layer widths")
        names.append(candidate["id"])
        previous = widths
    if len(set(names)) != len(names):
        raise ValueError("Candidate ids must be unique")
    selection = config.get("selection") or {}
    if selection != {"rule": "first_ordered_candidate_all_source_eligible_calibration_regions_verified"}:
        raise ValueError("The fixed selection rule is intentionally not configurable")
    return candidates


def prepare(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = _load(config_path)
    candidates = validate(config)
    calibration_path = Path(config["calibration_study"])
    evaluation_path = Path(config["evaluation_study"])
    calibration, evaluation = _load(calibration_path), _load(evaluation_path)
    if calibration["model_sha256"] != evaluation["model_sha256"]:
        raise ValueError("Calibration and evaluation must use the same immutable source model")
    if {r["id"] for r in calibration["selected_images"]} & {r["id"] for r in evaluation["selected_images"]}:
        raise ValueError("Evaluation cohort overlaps the calibration cohort")
    output.mkdir(parents=True, exist_ok=False)
    materialized = []
    for position, candidate in enumerate(candidates):
        path = output / "calibration" / candidate["id"]
        study = freeze(calibration_path, path, candidate["qif"], campaign="main")
        materialized.append({"position": position, "id": candidate["id"], "qif": candidate["qif"],
                             "study": str((path / "study.json").resolve()),
                             "study_sha256": sha256(path / "study.json"),
                             "artifact_sha256": study["fixed_artifact_sha256"]})
    sweep = {
        "schema": "ssv_fixed_qif_sweep_materialized_v1",
        "config": str(config_path.resolve()), "config_sha256": sha256(config_path),
        "calibration_study": str(calibration_path.resolve()), "calibration_study_sha256": sha256(calibration_path),
        "evaluation_study": str(evaluation_path.resolve()), "evaluation_study_sha256": sha256(evaluation_path),
        "model_sha256": calibration["model_sha256"],
        "selection": config["selection"], "calibration_runs_per_candidate": 27,
        "evaluation_runs_after_selection": 36, "candidates": materialized,
        "selection_status": "NOT_RUN", "evaluation_status": "NOT_MATERIALIZED",
    }
    write_new_json(output / "sweep.json", sweep)
    return sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.output)
    print(args.output / "sweep.json")


if __name__ == "__main__":
    main()
